"""
Option risk monitor for major ETFs and equities.

Returns eight DataFrames from run():
  spot              – last price, daily change %, relative volume,
                      5/20/60-day realized vol
  iv_term_structure – ATM IV per (symbol, expiry)
  iv_skew           – OTM put IV minus OTM call IV per (symbol, expiry)
  pc_overall        – aggregate put/call OI and volume ratios per symbol
  pc_by_expiry      – put/call ratios split by DTE bucket
  pc_by_moneyness   – put/call ratios split by strike moneyness bucket
  positioning       – max pain, call wall, put wall per symbol
                      (near-term chain only, DTE <= POSITIONING_MAX_DTE)
  derived           – per-symbol summary metrics: 30d ATM IV, term-structure
                      slope, 30d skew, IV-RV spread, relative volume and a
                      cross-sectional sentiment score
  data_quality      – per-symbol freshness flags: date of the last price bar,
                      last option trade date, and whether either is stale
                      relative to the valuation date (e.g. snapshot taken
                      before the market opened).  No values are imputed —
                      missing data stays missing and is flagged here.

Note: Yahoo only serves the *live* option chain — historical chains cannot
be fetched.  ``valuation_date`` therefore only makes sense as "today" (or
the date the snapshot was actually taken).
"""

import logging
from datetime import date, datetime, timedelta
from typing import Union

import numpy as np
import pandas as pd

from datafeed.yahoo_finance import YahooFinanceProvider

logger = logging.getLogger(__name__)


def _parse_date(d: Union[date, str, None]) -> date:
    """Coerce a date argument to a ``datetime.date``.

    Accepts a ``datetime.date`` object, an ISO-8601 string ("yyyy-mm-dd"),
    or ``None`` (returns today).
    """
    if d is None:
        return date.today()
    if isinstance(d, str):
        return datetime.strptime(d, "%Y-%m-%d").date()
    return d

# ── Universe ──────────────────────────────────────────────────────────────────

ETF_UNIVERSE: list[str] = [
    # broad market
    "SPY", "QQQ", "IWM", "DIA",
    # sectors (SPDR)
    "XLF", "XLE", "XLK", "XLV", "XLC", "XLI", "XLY", "XLP", "XLU", "XLRE",
    # fixed income & commodities
    "TLT", "GLD", "SLV", "GDX", "USO",
    # vol / leveraged
    "UVXY", "TQQQ", "SQQQ",
]

EQUITY_UNIVERSE: list[str] = [
    # Magnificent 7
    "AAPL", "MSFT", "GOOGL", "AMZN", "META", "NVDA", "TSLA",
    # financials
    "JPM", "BAC", "GS",
    # healthcare
    "JNJ", "UNH",
    # energy
    "XOM", "CVX",
]

DEFAULT_UNIVERSE: list[str] = ETF_UNIVERSE + EQUITY_UNIVERSE

# ── Bucket definitions ────────────────────────────────────────────────────────

# (lower_inclusive, upper_exclusive, label)  — None upper means open-ended
DTE_BUCKETS: list[tuple] = [
    (0,  30,  "0-30d"),
    (31, 60,  "31-60d"),
    (61, 90,  "61-90d"),
    (91, None, ">90d"),
]

# Moneyness = strike / spot; labels describe strike level relative to spot
# (lower_inclusive, upper_exclusive, label)
MONEYNESS_BUCKETS: list[tuple] = [
    (0.0,  0.85, "deep_below"),  # > 15 % below spot
    (0.85, 0.97, "below"),       # 3–15 % below spot
    (0.97, 1.03, "near_atm"),    # within 3 % of spot
    (1.03, 1.15, "above"),       # 3–15 % above spot
    (1.15, None, "deep_above"),  # > 15 % above spot
]

SKEW_WIDTH: float = 0.05   # 5 % OTM on each side for skew
HV_WINDOWS: list[int] = [5, 20, 60]

# IV sanity bounds — Yahoo fills missing IV with ~1e-5 placeholders and deep
# OTM / stale quotes can carry absurd values.  Contracts outside this range
# are dropped from the chain.
IV_MIN: float = 0.01   # 1 %
IV_MAX: float = 3.00   # 300 %

# Positioning metrics (max pain / walls) are per-expiry concepts; mixing
# LEAPS OI with weeklies dilutes the levels.  Restrict to near-term chain.
POSITIONING_MAX_DTE: int = 45

REL_VOLUME_WINDOW: int = 20  # days for the average-volume baseline

# Calendar days of price history to request — enough trading days for the
# longest realized-vol window plus the relative-volume baseline.
HISTORY_LOOKBACK_DAYS: int = 270

_CHAIN_COLS = [
    "symbol", "expiry", "dte", "side", "strike", "moneyness",
    "bid", "ask", "lastPrice", "volume", "openInterest",
    "impliedVolatility", "inTheMoney",
]


class OptionMonitor:
    """Compute option risk metrics for a universe of symbols.

    Parameters
    ----------
    symbols:
        Override the default universe.  Pass a short list for faster runs
        during development.
    valuation_date:
        Reference date used to compute DTE for every option contract.
        Accepts a ``datetime.date`` object or an ISO string ("yyyy-mm-dd").
        Defaults to today when omitted.  Pass this explicitly whenever
        ``date.today()`` may not match the date of the data you are
        fetching (e.g. running in a container with a shifted system clock,
        or analysing a historical snapshot).
    """

    def __init__(
        self,
        symbols: list[str] | None = None,
        valuation_date: date | str | None = None,
    ) -> None:
        self.symbols = symbols or DEFAULT_UNIVERSE
        self.valuation_date = _parse_date(valuation_date)
        self._provider = YahooFinanceProvider()

    # ── Public ────────────────────────────────────────────────────────────────

    def run(
        self,
        symbols: list[str] | None = None,
        valuation_date: date | str | None = None,
    ) -> dict[str, pd.DataFrame]:
        """Fetch data and compute all metrics.

        Parameters
        ----------
        symbols:
            Run-time override; falls back to ``self.symbols``.
        valuation_date:
            Run-time override for the reference date used to compute DTE.
            Falls back to ``self.valuation_date`` (set at construction time),
            which itself defaults to today.

        Returns
        -------
        dict with keys:
            ``spot``, ``iv_term_structure``, ``iv_skew``,
            ``pc_overall``, ``pc_by_expiry``, ``pc_by_moneyness``,
            ``positioning``

            Every DataFrame contains a ``valuation_date`` column (ISO string)
            so each result is self-describing.
        """
        targets = symbols or self.symbols
        vdate = _parse_date(valuation_date) if valuation_date is not None else self.valuation_date
        vdate_str = vdate.isoformat()

        spot_rows, iv_ts_rows, iv_skew_rows = [], [], []
        pc_overall_rows, pc_expiry_rows, pc_moneyness_rows = [], [], []
        positioning_rows, quality_rows = [], []

        for symbol in targets:
            try:
                spot, spot_row = self._spot_snapshot(symbol, vdate)
                spot_rows.append(spot_row)

                chain, chain_last_trade = self._build_chain(symbol, spot, vdate)
                quality_rows.append(self._quality_row(
                    symbol, vdate, spot_row, chain, chain_last_trade
                ))
                if chain.empty:
                    logger.warning("%s: empty option chain — skipped", symbol)
                    continue

                iv_ts_rows.extend(self._iv_term_structure(chain, spot, symbol))
                iv_skew_rows.extend(self._iv_skew(chain, spot, symbol))
                pc_overall_rows.append(self._pc_overall(chain, symbol))
                pc_expiry_rows.extend(self._pc_by_expiry(chain, symbol))
                pc_moneyness_rows.extend(self._pc_by_moneyness(chain, symbol))
                positioning_rows.append(self._positioning(chain, spot, symbol))

            except Exception:
                # One bad symbol must not abort the full run
                logger.exception("%s: failed — skipped", symbol)
                continue

        results = {
            "spot": pd.DataFrame(spot_rows),
            "iv_term_structure": pd.DataFrame(iv_ts_rows),
            "iv_skew": pd.DataFrame(iv_skew_rows),
            "pc_overall": pd.DataFrame(pc_overall_rows),
            "pc_by_expiry": pd.DataFrame(pc_expiry_rows),
            "pc_by_moneyness": pd.DataFrame(pc_moneyness_rows),
            "positioning": pd.DataFrame(positioning_rows),
            "data_quality": pd.DataFrame(quality_rows),
        }
        results["derived"] = self._derived(results)

        for df in results.values():
            df["valuation_date"] = vdate_str

        return results

    # ── Spot snapshot ─────────────────────────────────────────────────────────

    def _spot_snapshot(self, symbol: str, valuation_date: date) -> tuple[float, dict]:
        # yfinance treats ``end`` as *exclusive*: pass valuation_date + 1 day
        # so the close ON the valuation date is included.  ``start`` must be
        # passed explicitly — the provider ignores ``period`` when an end
        # date is supplied, and yfinance would fall back to ~1 month of data
        # (too short for rv_60d).
        end = (valuation_date + timedelta(days=1)).isoformat()
        start = (valuation_date - timedelta(days=HISTORY_LOOKBACK_DAYS)).isoformat()
        hist = self._provider.get_history(
            symbol, interval="1d", start=start, end=end
        )
        close = hist["Close"]
        log_ret = np.log(close / close.shift(1)).dropna()
        spot = float(close.iloc[-1])
        change_pct = float((close.iloc[-1] / close.iloc[-2] - 1) * 100)

        # Relative volume: last day vs trailing average (excluding last day)
        rel_volume = None
        if "Volume" in hist.columns:
            vol = hist["Volume"].dropna()
            if len(vol) > REL_VOLUME_WINDOW:
                base = float(vol.iloc[-(REL_VOLUME_WINDOW + 1):-1].mean())
                if base > 0:
                    rel_volume = round(float(vol.iloc[-1]) / base, 2)

        # Date of the last price bar.  Before the market opens (or on a
        # holiday) this lags the valuation date — flag it, don't mask it:
        # spot / change_pct / rel_volume then describe the *prior* session.
        price_date = pd.Timestamp(hist.index[-1]).date()
        price_is_stale = price_date < valuation_date
        if price_is_stale:
            logger.warning(
                "%s: last price bar is %s but valuation date is %s — market "
                "not open yet? spot/volume metrics reflect the prior session",
                symbol, price_date.isoformat(), valuation_date.isoformat(),
            )

        row: dict = {
            "symbol": symbol,
            "spot": round(spot, 4),
            "price_date": price_date.isoformat(),
            "price_is_stale": price_is_stale,
            "change_pct": round(change_pct, 2),
            "rel_volume": rel_volume,
        }
        for w in HV_WINDOWS:
            key = f"rv_{w}d"
            row[key] = (
                round(float(log_ret.rolling(w).std().iloc[-1] * np.sqrt(252) * 100), 2)
                if len(log_ret) >= w else None
            )
        return spot, row

    # ── Chain builder ─────────────────────────────────────────────────────────

    def _build_chain(
        self, symbol: str, spot: float, valuation_date: date
    ) -> tuple[pd.DataFrame, date | None]:
        """Build the filtered chain.

        Returns ``(chain, last_trade_date)`` where ``last_trade_date`` is the
        most recent option trade timestamp seen across the raw chain — used
        to flag stale (pre-open / holiday) snapshots.
        """
        expirations = self._provider.get_option_expirations(symbol)
        frames = []

        for expiry in expirations:
            exp_date = datetime.strptime(expiry, "%Y-%m-%d").date()
            dte = (exp_date - valuation_date).days
            if dte < 0:
                continue
            try:
                calls, puts = self._provider.get_option_chain(symbol, expiry)
            except Exception:
                continue

            for side, df in [("call", calls), ("put", puts)]:
                if df.empty:
                    continue
                df = df.copy()
                df["symbol"] = symbol
                df["expiry"] = expiry
                df["dte"] = dte
                df["side"] = side
                df["moneyness"] = (df["strike"] / spot).round(4)
                frames.append(df)

        if not frames:
            return pd.DataFrame(), None

        chain = pd.concat(frames, ignore_index=True)

        # Freshness: most recent option trade across the raw chain.  Pre-open
        # this is the prior session — quotes/volume are stale.
        last_trade_date: date | None = None
        if "lastTradeDate" in chain.columns:
            lt = pd.to_datetime(chain["lastTradeDate"], errors="coerce", utc=True)
            if lt.notna().any():
                last_trade_date = lt.max().date()
                if last_trade_date < valuation_date:
                    logger.warning(
                        "%s: option chain is stale — last option trade %s vs "
                        "valuation date %s (market not open yet?)",
                        symbol, last_trade_date.isoformat(),
                        valuation_date.isoformat(),
                    )

        chain = chain[[c for c in _CHAIN_COLS if c in chain.columns]]

        # Quality filters:
        #   - IV must be within sane bounds (Yahoo uses ~1e-5 as a "missing"
        #     placeholder; stale deep-OTM quotes can show absurd IV)
        #   - contract must show signs of life: a live bid or traded volume
        chain = chain[
            chain["impliedVolatility"].notna()
            & (chain["impliedVolatility"] >= IV_MIN)
            & (chain["impliedVolatility"] <= IV_MAX)
        ]
        if "bid" in chain.columns:
            alive = chain["bid"].fillna(0) > 0
            if "volume" in chain.columns:
                alive |= chain["volume"].fillna(0) > 0
            chain = chain[alive]
        chain = chain[chain["openInterest"].notna()]

        chain["dte_bucket"] = chain["dte"].apply(self._dte_bucket)
        chain["moneyness_bucket"] = chain["moneyness"].apply(self._moneyness_bucket)
        return chain.reset_index(drop=True), last_trade_date

    # ── Data quality ──────────────────────────────────────────────────────────

    @staticmethod
    def _quality_row(
        symbol: str,
        valuation_date: date,
        spot_row: dict,
        chain: pd.DataFrame,
        chain_last_trade: date | None,
    ) -> dict:
        """Per-symbol freshness/completeness flags (no values are imputed)."""
        volume_reported = bool(
            not chain.empty
            and "volume" in chain.columns
            and chain["volume"].notna().any()
        )
        chain_is_stale = (
            chain_last_trade < valuation_date
            if chain_last_trade is not None else None
        )
        return {
            "symbol": symbol,
            "price_date": spot_row.get("price_date"),
            "price_is_stale": spot_row.get("price_is_stale"),
            "chain_last_trade_date": (
                chain_last_trade.isoformat() if chain_last_trade else None
            ),
            "chain_is_stale": chain_is_stale,
            "n_contracts": int(len(chain)),
            "volume_reported": volume_reported,
        }

    # ── IV term structure ─────────────────────────────────────────────────────

    def _iv_term_structure(
        self, chain: pd.DataFrame, spot: float, symbol: str
    ) -> list[dict]:
        rows = []
        for expiry, grp in chain.groupby("expiry"):
            dte = int(grp["dte"].iloc[0])
            call_iv = self._interp_iv(grp[grp["side"] == "call"], spot)
            put_iv = self._interp_iv(grp[grp["side"] == "put"], spot)
            ivs = [v for v in [call_iv, put_iv] if v is not None]
            rows.append({
                "symbol": symbol,
                "expiry": expiry,
                "dte": dte,
                "dte_bucket": self._dte_bucket(dte),
                "atm_call_iv": round(call_iv, 4) if call_iv is not None else None,
                "atm_put_iv": round(put_iv, 4) if put_iv is not None else None,
                "atm_iv": round(float(np.mean(ivs)), 4) if ivs else None,
            })
        return rows

    # ── IV skew ───────────────────────────────────────────────────────────────

    def _iv_skew(
        self, chain: pd.DataFrame, spot: float, symbol: str
    ) -> list[dict]:
        rows = []
        put_target = spot * (1 - SKEW_WIDTH)
        call_target = spot * (1 + SKEW_WIDTH)
        for expiry, grp in chain.groupby("expiry"):
            dte = int(grp["dte"].iloc[0])
            put_iv = self._interp_iv(grp[grp["side"] == "put"], put_target)
            call_iv = self._interp_iv(grp[grp["side"] == "call"], call_target)
            skew = (
                round(put_iv - call_iv, 4)
                if (put_iv is not None and call_iv is not None) else None
            )
            rows.append({
                "symbol": symbol,
                "expiry": expiry,
                "dte": dte,
                "dte_bucket": self._dte_bucket(dte),
                "otm_put_iv": round(put_iv, 4) if put_iv is not None else None,
                "otm_call_iv": round(call_iv, 4) if call_iv is not None else None,
                "skew": skew,
            })
        return rows

    # ── Put/call ratios ───────────────────────────────────────────────────────

    def _pc_overall(self, chain: pd.DataFrame, symbol: str) -> dict:
        calls = chain[chain["side"] == "call"]
        puts = chain[chain["side"] == "put"]
        call_vol, put_vol = self._vol_sum(calls), self._vol_sum(puts)
        return {
            "symbol": symbol,
            "total_call_oi": int(calls["openInterest"].sum()),
            "total_put_oi": int(puts["openInterest"].sum()),
            "pc_oi_ratio": self._safe_ratio(
                puts["openInterest"].sum(), calls["openInterest"].sum()
            ),
            "total_call_vol": call_vol,
            "total_put_vol": put_vol,
            "pc_vol_ratio": self._safe_ratio(put_vol, call_vol),
        }

    def _pc_by_expiry(self, chain: pd.DataFrame, symbol: str) -> list[dict]:
        rows = []
        for bucket, grp in chain.groupby("dte_bucket"):
            calls = grp[grp["side"] == "call"]
            puts = grp[grp["side"] == "put"]
            call_vol, put_vol = self._vol_sum(calls), self._vol_sum(puts)
            rows.append({
                "symbol": symbol,
                "dte_bucket": bucket,
                "call_oi": int(calls["openInterest"].sum()),
                "put_oi": int(puts["openInterest"].sum()),
                "pc_oi_ratio": self._safe_ratio(
                    puts["openInterest"].sum(), calls["openInterest"].sum()
                ),
                "call_vol": call_vol,
                "put_vol": put_vol,
                "pc_vol_ratio": self._safe_ratio(put_vol, call_vol),
            })
        return rows

    def _pc_by_moneyness(self, chain: pd.DataFrame, symbol: str) -> list[dict]:
        rows = []
        for bucket, grp in chain.groupby("moneyness_bucket"):
            calls = grp[grp["side"] == "call"]
            puts = grp[grp["side"] == "put"]
            call_vol, put_vol = self._vol_sum(calls), self._vol_sum(puts)
            rows.append({
                "symbol": symbol,
                "moneyness_bucket": bucket,
                "call_oi": int(calls["openInterest"].sum()),
                "put_oi": int(puts["openInterest"].sum()),
                "pc_oi_ratio": self._safe_ratio(
                    puts["openInterest"].sum(), calls["openInterest"].sum()
                ),
                "call_vol": call_vol,
                "put_vol": put_vol,
                "pc_vol_ratio": self._safe_ratio(put_vol, call_vol),
            })
        return rows

    # ── Positioning ───────────────────────────────────────────────────────────

    def _positioning(
        self, chain: pd.DataFrame, spot: float, symbol: str
    ) -> dict:
        """Max pain / call wall / put wall on the near-term chain.

        Restricted to DTE <= POSITIONING_MAX_DTE — these are per-expiry
        concepts, and mixing LEAPS OI with weeklies dilutes the levels.
        Falls back to the full chain when no near-term contracts exist.
        """
        near = chain[chain["dte"] <= POSITIONING_MAX_DTE]
        if near.empty:
            near = chain
        call_oi = near[near["side"] == "call"].groupby("strike")["openInterest"].sum()
        put_oi = near[near["side"] == "put"].groupby("strike")["openInterest"].sum()

        call_wall = float(call_oi.idxmax()) if not call_oi.empty else None
        put_wall = float(put_oi.idxmax()) if not put_oi.empty else None
        max_pain = self._max_pain(call_oi, put_oi)

        def dist(price: float | None) -> float | None:
            return round((price / spot - 1) * 100, 2) if price is not None else None

        return {
            "symbol": symbol,
            "spot": round(spot, 4),
            "max_pain": max_pain,
            "max_pain_dist_pct": dist(max_pain),
            "call_wall": call_wall,
            "call_wall_dist_pct": dist(call_wall),
            "put_wall": put_wall,
            "put_wall_dist_pct": dist(put_wall),
        }

    # ── Derived per-symbol summary ────────────────────────────────────────────

    def _derived(self, results: dict[str, pd.DataFrame]) -> pd.DataFrame:
        """Per-symbol summary metrics assembled from the other tables.

        Columns
        -------
        atm_iv_30d / atm_iv_90d
            ATM IV interpolated to a constant 30/90-day tenor.
        term_slope
            atm_iv_30d / atm_iv_90d.  > 1 means inverted front end (event
            stress); < 1 is normal contango.
        skew_30d
            5% OTM put-call IV spread interpolated to a 30-day tenor.
        iv_rv_spread
            atm_iv_30d (in %) minus 20-day realized vol — the variance risk
            premium; elevated values = options pricing more fear than the
            tape shows.
        rel_volume
            Underlying volume vs its 20-day average.
        pc_vol_ratio
            Overall put/call volume ratio.
        sentiment
            Cross-sectional composite: negative of the mean z-score of
            (pc_vol_ratio, skew_30d, iv_rv_spread) across the universe.
            Higher = more bullish.  None when fewer than 3 symbols ran.
        """
        spot_df = results["spot"]
        if spot_df.empty:
            return pd.DataFrame()
        ts, sk, pco = (
            results["iv_term_structure"],
            results["iv_skew"],
            results["pc_overall"],
        )

        rows = []
        for _, srow in spot_df.iterrows():
            sym = srow["symbol"]
            sym_ts = ts[ts["symbol"] == sym] if not ts.empty else pd.DataFrame()
            sym_sk = sk[sk["symbol"] == sym] if not sk.empty else pd.DataFrame()
            iv30 = self._interp_tenor(sym_ts, "atm_iv", 30)
            iv90 = self._interp_tenor(sym_ts, "atm_iv", 90)
            skew30 = self._interp_tenor(sym_sk, "skew", 30)

            rv20 = srow.get("rv_20d")
            has_rv = rv20 is not None and pd.notna(rv20)
            iv_rv = (
                round(iv30 * 100 - float(rv20), 2)
                if iv30 is not None and has_rv else None
            )

            pc_vol = None
            if not pco.empty:
                m = pco[pco["symbol"] == sym]
                if len(m) and pd.notna(m["pc_vol_ratio"].iloc[0]):
                    pc_vol = float(m["pc_vol_ratio"].iloc[0])

            rows.append({
                "symbol": sym,
                "spot": srow["spot"],
                "change_pct": srow["change_pct"],
                "rel_volume": srow.get("rel_volume"),
                "rv_20d": rv20,
                "atm_iv_30d": round(iv30, 4) if iv30 is not None else None,
                "atm_iv_90d": round(iv90, 4) if iv90 is not None else None,
                "term_slope": round(iv30 / iv90, 4) if iv30 and iv90 else None,
                "skew_30d": round(skew30, 4) if skew30 is not None else None,
                "iv_rv_spread": iv_rv,
                "pc_vol_ratio": pc_vol,
            })

        df = pd.DataFrame(rows)
        df["sentiment"] = self._sentiment(df)
        return df

    @staticmethod
    def _interp_tenor(df: pd.DataFrame, col: str, target_dte: int) -> float | None:
        """Interpolate ``col`` across DTE to a constant tenor.

        Clamps to the nearest endpoint when ``target_dte`` is outside the
        available expiry range.
        """
        if df.empty or col not in df.columns:
            return None
        pts = df[["dte", col]].dropna().sort_values("dte")
        if pts.empty:
            return None
        return float(np.interp(
            target_dte,
            pts["dte"].to_numpy(dtype=float),
            pts[col].to_numpy(dtype=float),
        ))

    @staticmethod
    def _sentiment(df: pd.DataFrame) -> pd.Series:
        """Composite sentiment score; higher = more bullish.

        Negative of the mean cross-sectional z-score of pc_vol_ratio,
        skew_30d and iv_rv_spread — all three rise when downside protection
        is bid.  Requires at least 3 symbols; returns None otherwise.
        """
        components = ["pc_vol_ratio", "skew_30d", "iv_rv_spread"]
        if len(df) < 3:
            return pd.Series([None] * len(df), index=df.index, dtype=object)
        zs = []
        for col in components:
            s = pd.to_numeric(df[col], errors="coerce")
            std = s.std(ddof=0)
            zs.append((s - s.mean()) / std if std and std > 0 else s * 0.0)
        z = pd.concat(zs, axis=1).mean(axis=1)
        return (-z).round(2)

    # ── Static helpers ────────────────────────────────────────────────────────

    @staticmethod
    def _max_pain(call_oi: pd.Series, put_oi: pd.Series) -> float | None:
        strikes = sorted(set(call_oi.index) | set(put_oi.index))
        if not strikes:
            return None
        min_pain, best = float("inf"), strikes[0]
        for k in strikes:
            call_pain = sum((k - s) * oi for s, oi in call_oi.items() if s < k)
            put_pain = sum((s - k) * oi for s, oi in put_oi.items() if s > k)
            total = call_pain + put_pain
            if total < min_pain:
                min_pain, best = total, k
        return float(best)

    @staticmethod
    def _interp_iv(df: pd.DataFrame, target: float) -> float | None:
        """IV at ``target`` strike via linear interpolation across strikes.

        Less noise-sensitive than picking the single nearest strike.  When
        ``target`` lies outside the listed strike range the nearest endpoint
        is used.  Duplicate strikes are averaged first.
        """
        if df.empty:
            return None
        by_strike = df.groupby("strike")["impliedVolatility"].mean().sort_index()
        strikes = by_strike.index.to_numpy(dtype=float)
        ivs = by_strike.to_numpy(dtype=float)
        iv = float(np.interp(target, strikes, ivs))
        return iv if iv > 0 else None

    @staticmethod
    def _vol_sum(df: pd.DataFrame) -> int | None:
        """Sum traded volume, or ``None`` when no contract reports volume.

        Pre-open Yahoo serves the chain with volume unset — summing those
        as zero would fabricate a "no flow" reading.  Missing stays missing.
        """
        if "volume" not in df.columns or not df["volume"].notna().any():
            return None
        return int(df["volume"].fillna(0).sum())

    @staticmethod
    def _safe_ratio(num, denom) -> float | None:
        if num is None or denom is None:
            return None
        return round(float(num / denom), 4) if denom and denom > 0 else None

    @staticmethod
    def _dte_bucket(dte: int) -> str:
        if dte <= 30:
            return "0-30d"
        elif dte <= 60:
            return "31-60d"
        elif dte <= 90:
            return "61-90d"
        return ">90d"

    @staticmethod
    def _moneyness_bucket(moneyness: float) -> str:
        if moneyness < 0.85:
            return "deep_below"
        elif moneyness < 0.97:
            return "below"
        elif moneyness <= 1.03:
            return "near_atm"
        elif moneyness <= 1.15:
            return "above"
        return "deep_above"
