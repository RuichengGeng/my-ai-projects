"""
Option risk monitor for major ETFs and equities.

Returns seven DataFrames from run():
  spot              – last price, daily change %, 5/20/60-day realized vol
  iv_term_structure – ATM IV per (symbol, expiry)
  iv_skew           – OTM put IV minus OTM call IV per (symbol, expiry)
  pc_overall        – aggregate put/call OI and volume ratios per symbol
  pc_by_expiry      – put/call ratios split by DTE bucket
  pc_by_moneyness   – put/call ratios split by strike moneyness bucket
  positioning       – max pain, call wall, put wall per symbol
"""

from datetime import date, datetime
from typing import Union

import numpy as np
import pandas as pd

from .yahoo_finance import YahooFinanceProvider


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
        positioning_rows = []

        for symbol in targets:
            try:
                spot, spot_row = self._spot_snapshot(symbol, vdate)
                spot_rows.append(spot_row)

                chain = self._build_chain(symbol, spot, vdate)
                if chain.empty:
                    continue

                iv_ts_rows.extend(self._iv_term_structure(chain, spot, symbol))
                iv_skew_rows.extend(self._iv_skew(chain, spot, symbol))
                pc_overall_rows.append(self._pc_overall(chain, symbol))
                pc_expiry_rows.extend(self._pc_by_expiry(chain, symbol))
                pc_moneyness_rows.extend(self._pc_by_moneyness(chain, symbol))
                positioning_rows.append(self._positioning(chain, spot, symbol))

            except Exception:
                # One bad symbol must not abort the full run
                continue

        results = {
            "spot": pd.DataFrame(spot_rows),
            "iv_term_structure": pd.DataFrame(iv_ts_rows),
            "iv_skew": pd.DataFrame(iv_skew_rows),
            "pc_overall": pd.DataFrame(pc_overall_rows),
            "pc_by_expiry": pd.DataFrame(pc_expiry_rows),
            "pc_by_moneyness": pd.DataFrame(pc_moneyness_rows),
            "positioning": pd.DataFrame(positioning_rows),
        }

        for df in results.values():
            df["valuation_date"] = vdate_str

        return results

    # ── Spot snapshot ─────────────────────────────────────────────────────────

    def _spot_snapshot(self, symbol: str, valuation_date: date) -> tuple[float, dict]:
        hist = self._provider.get_history(
            symbol, period="3mo", interval="1d", end=valuation_date.isoformat()
        )
        close = hist["Close"]
        log_ret = np.log(close / close.shift(1)).dropna()
        spot = float(close.iloc[-1])
        change_pct = float((close.iloc[-1] / close.iloc[-2] - 1) * 100)

        row: dict = {
            "symbol": symbol,
            "spot": round(spot, 4),
            "change_pct": round(change_pct, 2),
        }
        for w in HV_WINDOWS:
            key = f"rv_{w}d"
            row[key] = (
                round(float(log_ret.rolling(w).std().iloc[-1] * np.sqrt(252) * 100), 2)
                if len(log_ret) >= w else None
            )
        return spot, row

    # ── Chain builder ─────────────────────────────────────────────────────────

    def _build_chain(self, symbol: str, spot: float, valuation_date: date) -> pd.DataFrame:
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
            return pd.DataFrame()

        chain = pd.concat(frames, ignore_index=True)
        chain = chain[[c for c in _CHAIN_COLS if c in chain.columns]]

        # Drop contracts with missing or non-positive IV
        chain = chain[chain["impliedVolatility"].notna() & (chain["impliedVolatility"] > 0)]
        chain = chain[chain["openInterest"].notna()]

        chain["dte_bucket"] = chain["dte"].apply(self._dte_bucket)
        chain["moneyness_bucket"] = chain["moneyness"].apply(self._moneyness_bucket)
        return chain.reset_index(drop=True)

    # ── IV term structure ─────────────────────────────────────────────────────

    def _iv_term_structure(
        self, chain: pd.DataFrame, spot: float, symbol: str
    ) -> list[dict]:
        rows = []
        for expiry, grp in chain.groupby("expiry"):
            dte = int(grp["dte"].iloc[0])
            call_iv = self._nearest_iv(grp[grp["side"] == "call"], spot)
            put_iv = self._nearest_iv(grp[grp["side"] == "put"], spot)
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
            put_iv = self._nearest_iv(grp[grp["side"] == "put"], put_target)
            call_iv = self._nearest_iv(grp[grp["side"] == "call"], call_target)
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
        return {
            "symbol": symbol,
            "total_call_oi": int(calls["openInterest"].sum()),
            "total_put_oi": int(puts["openInterest"].sum()),
            "pc_oi_ratio": self._safe_ratio(
                puts["openInterest"].sum(), calls["openInterest"].sum()
            ),
            "total_call_vol": int(calls["volume"].fillna(0).sum()),
            "total_put_vol": int(puts["volume"].fillna(0).sum()),
            "pc_vol_ratio": self._safe_ratio(
                puts["volume"].fillna(0).sum(), calls["volume"].fillna(0).sum()
            ),
        }

    def _pc_by_expiry(self, chain: pd.DataFrame, symbol: str) -> list[dict]:
        rows = []
        for bucket, grp in chain.groupby("dte_bucket"):
            calls = grp[grp["side"] == "call"]
            puts = grp[grp["side"] == "put"]
            rows.append({
                "symbol": symbol,
                "dte_bucket": bucket,
                "call_oi": int(calls["openInterest"].sum()),
                "put_oi": int(puts["openInterest"].sum()),
                "pc_oi_ratio": self._safe_ratio(
                    puts["openInterest"].sum(), calls["openInterest"].sum()
                ),
                "call_vol": int(calls["volume"].fillna(0).sum()),
                "put_vol": int(puts["volume"].fillna(0).sum()),
                "pc_vol_ratio": self._safe_ratio(
                    puts["volume"].fillna(0).sum(), calls["volume"].fillna(0).sum()
                ),
            })
        return rows

    def _pc_by_moneyness(self, chain: pd.DataFrame, symbol: str) -> list[dict]:
        rows = []
        for bucket, grp in chain.groupby("moneyness_bucket"):
            calls = grp[grp["side"] == "call"]
            puts = grp[grp["side"] == "put"]
            rows.append({
                "symbol": symbol,
                "moneyness_bucket": bucket,
                "call_oi": int(calls["openInterest"].sum()),
                "put_oi": int(puts["openInterest"].sum()),
                "pc_oi_ratio": self._safe_ratio(
                    puts["openInterest"].sum(), calls["openInterest"].sum()
                ),
                "call_vol": int(calls["volume"].fillna(0).sum()),
                "put_vol": int(puts["volume"].fillna(0).sum()),
                "pc_vol_ratio": self._safe_ratio(
                    puts["volume"].fillna(0).sum(), calls["volume"].fillna(0).sum()
                ),
            })
        return rows

    # ── Positioning ───────────────────────────────────────────────────────────

    def _positioning(
        self, chain: pd.DataFrame, spot: float, symbol: str
    ) -> dict:
        call_oi = chain[chain["side"] == "call"].groupby("strike")["openInterest"].sum()
        put_oi = chain[chain["side"] == "put"].groupby("strike")["openInterest"].sum()

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
    def _nearest_iv(df: pd.DataFrame, target: float) -> float | None:
        if df.empty:
            return None
        idx = (df["strike"] - target).abs().idxmin()
        iv = df.loc[idx, "impliedVolatility"]
        return float(iv) if pd.notna(iv) and iv > 0 else None

    @staticmethod
    def _safe_ratio(num, denom) -> float | None:
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
