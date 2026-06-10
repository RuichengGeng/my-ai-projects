"""
Yahoo Finance data provider.

Asset classes
─────────────
  Equities (US)         AAPL, MSFT          full history, financials, options
  Equities (Intl)       7203.T, BP.L        full history, financials
  ETFs                  SPY, QQQ            full history, options
  Indices               ^GSPC, ^VIX         full history, price-only
  Crypto                BTC-USD, ETH-USD    since ~2014, price-only
  Forex                 EURUSD=X            full history, price-only
  Futures               CL=F, GC=F          front-month only, price-only
  Mutual funds          VFIAX               full history, daily NAV
  Treasury rates        ^TNX, ^TYX          full history, price-only

Intraday history limits
───────────────────────
  1m               → 7 days
  2m / 5m / 15m / 30m / 90m  → 60 days
  60m / 1h         → 730 days (2 years)
  1d and coarser   → full history

Price adjustment modes (equity OHLCV)
──────────────────────────────────────
  auto_adjust=True, back_adjust=False  [DEFAULT]
      All OHLCV values are multiplied by the adjclose/close ratio for each bar.
      Adjusts for *both* cash dividends and stock splits.
      Close = adjusted close; no separate "Adj Close" column exists.
      Ex-dividend price drops are removed from the series.
      Suitable for return calculations and most backtesting.

  auto_adjust=False, back_adjust=False
      Raw, unadjusted OHLCV as actually traded on the exchange.
      Visible price gaps appear at ex-dividend dates and split events.
      Suitable for point-in-time analysis or custom corporate-action handling.

  auto_adjust=False, back_adjust=True
      Proportional back-adjustment anchored to the most recent unadjusted price.
      All historical bars are scaled so the ratio of any two prices equals the
      ratio of their unadjusted prices. Useful when you need scale-consistent
      series that are still anchored to today's price level.
      (back_adjust=True has no effect when auto_adjust=True.)
"""

from typing import Literal

import pandas as pd
import yfinance as yf

from .yahoo_finance_base import (
    INTERVAL_MAX_HISTORY,
    SUPPORTED_ASSET_CLASSES,
    YahooFinanceError,
    validate_interval,
    validate_period,
)

Period = Literal["1d", "5d", "1mo", "3mo", "6mo", "1y", "2y", "5y", "10y", "ytd", "max"]
Interval = Literal[
    "1m", "2m", "5m", "15m", "30m", "60m", "90m",
    "1h", "1d", "5d", "1wk", "1mo", "3mo",
]


class YahooFinanceProvider:
    """Unified market-data provider backed by Yahoo Finance (via yfinance).

    Supports equities (US & international), ETFs, indices, crypto, forex,
    futures, mutual funds, and Treasury rate indices — any symbol recognised
    by Yahoo Finance.

    Call :meth:`get_coverage` to inspect the full asset-class and
    interval-history table at runtime.
    """

    # ── Price history ────────────────────────────────────────────────

    def get_history(
        self,
        symbol: str,
        period: Period = "1y",
        interval: Interval = "1d",
        start: str | None = None,
        end: str | None = None,
        auto_adjust: bool = True,
        back_adjust: bool = False,
    ) -> pd.DataFrame:
        """OHLCV price history for a single symbol.

        Args:
            symbol: Yahoo Finance ticker (e.g. "AAPL", "BTC-USD", "EURUSD=X").
            period: Lookback period used when *start*/*end* are omitted.
            interval: Bar size. Intraday bars have capped history — see module
                docstring for limits.
            start: Start date as "yyyy-mm-dd"; overrides *period*.
            end: End date as "yyyy-mm-dd"; overrides *period*.
            auto_adjust: When ``True`` (default), all OHLCV values are adjusted
                for **both dividends and splits**. ``Close`` equals the adjusted
                close; no separate "Adj Close" column is returned. Set to
                ``False`` to receive raw unadjusted prices.
            back_adjust: When ``True`` (and ``auto_adjust=False``), prices are
                proportionally back-adjusted so the most recent bar matches the
                unadjusted close. Has no effect when ``auto_adjust=True``.

        Returns:
            DataFrame indexed by date with columns Open, High, Low, Close,
            Volume (plus Dividends and Stock Splits when present).

        Raises:
            ValueError: On invalid period or interval.
            YahooFinanceError: When Yahoo Finance returns no data.
        """
        validate_period(period)
        validate_interval(interval)

        ticker = yf.Ticker(symbol)
        kwargs: dict = {
            "interval": interval,
            "auto_adjust": auto_adjust,
            "back_adjust": back_adjust,
        }
        if start or end:
            if start:
                kwargs["start"] = start
            if end:
                kwargs["end"] = end
        else:
            kwargs["period"] = period

        df = ticker.history(**kwargs)
        if df.empty:
            raise YahooFinanceError(f"No price data returned for symbol '{symbol}'.")
        return df

    # ── Company info / metadata ──────────────────────────────────────

    def get_info(self, symbol: str) -> dict:
        """Metadata for a symbol: sector, industry, market cap, P/E, etc.

        Returns:
            Dict as returned by yfinance (keys vary by asset class).
        """
        return yf.Ticker(symbol).info

    # ── Corporate actions ────────────────────────────────────────────

    def get_dividends(
        self,
        symbol: str,
        start: str | None = None,
        end: str | None = None,
    ) -> pd.Series:
        """Historical cash dividends for a stock.

        Args:
            symbol: Stock ticker.
            start: Filter from this date (yyyy-mm-dd).
            end: Filter until this date (yyyy-mm-dd).

        Returns:
            Series indexed by timezone-aware datetime with dividend amounts.
        """
        divs = yf.Ticker(symbol).dividends
        if start:
            divs = divs[divs.index >= pd.to_datetime(start, utc=True)]
        if end:
            divs = divs[divs.index <= pd.to_datetime(end, utc=True)]
        return divs

    def get_splits(self, symbol: str) -> pd.Series:
        """Historical stock-split events.

        Returns:
            Series indexed by datetime with split ratios (e.g. 4.0 for 4-for-1).
        """
        return yf.Ticker(symbol).splits

    # ── Financial statements ─────────────────────────────────────────

    def get_income_statement(
        self, symbol: str, quarterly: bool = False
    ) -> pd.DataFrame:
        """Income statement (P&L) for a publicly listed company.

        Args:
            symbol: Stock ticker.
            quarterly: Return quarterly data instead of the default annual.

        Returns:
            DataFrame with financial line items as rows and reporting periods
            as columns.
        """
        ticker = yf.Ticker(symbol)
        return ticker.quarterly_income_stmt if quarterly else ticker.income_stmt

    def get_balance_sheet(
        self, symbol: str, quarterly: bool = False
    ) -> pd.DataFrame:
        """Balance sheet snapshot for a company."""
        ticker = yf.Ticker(symbol)
        return ticker.quarterly_balance_sheet if quarterly else ticker.balance_sheet

    def get_cash_flow(
        self, symbol: str, quarterly: bool = False
    ) -> pd.DataFrame:
        """Cash flow statement for a company."""
        ticker = yf.Ticker(symbol)
        return ticker.quarterly_cashflow if quarterly else ticker.cashflow

    # ── Options ──────────────────────────────────────────────────────

    def get_option_expirations(self, symbol: str) -> tuple[str, ...]:
        """Available option expiration dates for a symbol.

        Returns:
            Tuple of date strings in "yyyy-mm-dd" format.
        """
        return yf.Ticker(symbol).options

    def get_option_chain(
        self, symbol: str, expiration: str
    ) -> tuple[pd.DataFrame, pd.DataFrame]:
        """Full option chain for a specific expiration date.

        Args:
            symbol: Optionable stock ticker.
            expiration: Expiration date from :meth:`get_option_expirations`.

        Returns:
            ``(calls, puts)`` — each a DataFrame with strike, bid, ask,
            impliedVolatility, openInterest, etc.
        """
        chain = yf.Ticker(symbol).option_chain(expiration)
        return chain.calls, chain.puts

    # ── Multi-ticker bulk download ────────────────────────────────────

    def download(
        self,
        symbols: list[str],
        period: Period = "1y",
        interval: Interval = "1d",
        start: str | None = None,
        end: str | None = None,
        auto_adjust: bool = True,
    ) -> pd.DataFrame:
        """Download OHLCV data for several symbols in one request.

        Applies the same ``auto_adjust`` semantics as :meth:`get_history`:
        ``True`` (default) returns dividend-and-split adjusted prices;
        ``False`` returns raw unadjusted prices.

        Args:
            symbols: List of Yahoo Finance ticker symbols.
            period: Lookback period when *start*/*end* are omitted.
            interval: Bar size.
            start: Start date as "yyyy-mm-dd".
            end: End date as "yyyy-mm-dd".
            auto_adjust: Adjust OHLCV for dividends and splits (default ``True``).

        Returns:
            Multi-level DataFrame grouped by ticker
            (e.g. ``df["AAPL"]["Close"]``).

        Raises:
            YahooFinanceError: When Yahoo Finance returns no data.
        """
        validate_period(period)
        validate_interval(interval)

        kwargs: dict = {
            "tickers": symbols,
            "interval": interval,
            "auto_adjust": auto_adjust,
            "group_by": "ticker",
        }
        if start or end:
            if start:
                kwargs["start"] = start
            if end:
                kwargs["end"] = end
        else:
            kwargs["period"] = period

        df = yf.download(**kwargs)
        if df.empty:
            raise YahooFinanceError(
                f"No data returned for symbols: {symbols}"
            )
        return df

    # ── Analyst & news ────────────────────────────────────────────────

    def get_recommendations(self, symbol: str) -> pd.DataFrame:
        """Latest analyst recommendations for a stock.

        Returns:
            DataFrame with columns: Firm, To Grade, From Grade, Action.
        """
        return yf.Ticker(symbol).recommendations

    def get_news(self, symbol: str) -> list[dict]:
        """Recent news articles associated with a symbol.

        Returns:
            List of dicts with keys: title, link, publisher, providerPublishTime.
        """
        return yf.Ticker(symbol).news

    def get_calendar(self, symbol: str) -> dict:
        """Upcoming earnings / event calendar for a company.

        Returns:
            Dict with earningsDate, epsEstimate, revenueEstimate, etc.
        """
        return yf.Ticker(symbol).calendar

    # ── Coverage metadata ─────────────────────────────────────────────

    @staticmethod
    def get_coverage() -> dict:
        """Return a structured summary of asset-class coverage and behaviour.

        Returns a dict with three keys:

        ``asset_classes``
            Mapping of asset-class name → dict with ``description``,
            ``symbol_format``, ``examples``, ``history``, ``financials``,
            and ``options`` keys.

        ``interval_max_history``
            Mapping of interval string → max lookback string (e.g. "60 days").

        ``price_adjustment``
            Three-entry mapping describing the behaviour of ``auto_adjust``
            and ``back_adjust`` flags in :meth:`get_history`.
        """
        return {
            "asset_classes": SUPPORTED_ASSET_CLASSES,
            "interval_max_history": INTERVAL_MAX_HISTORY,
            "price_adjustment": {
                "auto_adjust_true": (
                    "DEFAULT. All OHLCV values adjusted for both dividends and "
                    "splits using the adjclose/close ratio. Close == adjusted "
                    "close; no separate Adj Close column. Ex-dividend price drops "
                    "are removed. Suitable for return calculations and backtesting."
                ),
                "auto_adjust_false": (
                    "Raw unadjusted OHLCV as traded on exchange. Visible price "
                    "gaps at ex-dividend dates and split events. Use for "
                    "point-in-time analysis or custom corporate-action handling."
                ),
                "back_adjust_true": (
                    "Proportional back-adjustment anchored to the most recent "
                    "unadjusted price. Prices are scaled so every ratio equals "
                    "the ratio of the corresponding unadjusted prices. Only "
                    "meaningful when auto_adjust=False; ignored otherwise."
                ),
            },
        }
