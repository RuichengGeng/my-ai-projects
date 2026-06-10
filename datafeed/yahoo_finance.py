"""
Yahoo Finance data provider.

Covers:
  - Equities:  OHLCV history, company info, financials, dividends, splits
  - Options:   Expiration dates, full call/put chains
  - Crypto:    Prices via BTC-USD style symbols
  - Forex:     Rates via EURUSD=X style symbols
  - Bulk:      Multi-ticker simultaneous download
  - Analyst:   Recommendations, news, earnings calendar
"""

from typing import Literal

import pandas as pd
import yfinance as yf

from .yahoo_finance_base import YahooFinanceError, validate_interval, validate_period

# Typed aliases for IDE / static-analysis support
Period = Literal["1d", "5d", "1mo", "3mo", "6mo", "1y", "2y", "5y", "10y", "ytd", "max"]
Interval = Literal[
    "1m", "2m", "5m", "15m", "30m", "60m", "90m",
    "1h", "1d", "5d", "1wk", "1mo", "3mo",
]


class YahooFinanceProvider:
    """Unified market-data provider backed by Yahoo Finance (via yfinance).

    Works with equities, ETFs, indices, crypto, and forex — any symbol
    recognised by Yahoo Finance.
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
    ) -> pd.DataFrame:
        """OHLCV price history for a single symbol.

        Args:
            symbol: Yahoo Finance ticker (e.g. "AAPL", "BTC-USD", "EURUSD=X").
            period: Lookback period used when *start*/*end* are omitted.
            interval: Bar size.
            start: Start date as "yyyy-mm-dd"; overrides *period*.
            end: End date as "yyyy-mm-dd"; overrides *period*.
            auto_adjust: Adjust prices for dividends and splits.

        Returns:
            DataFrame indexed by date with columns Open, High, Low, Close, Volume.

        Raises:
            ValueError: On invalid period or interval.
            YahooFinanceError: When Yahoo Finance returns no data.
        """
        validate_period(period)
        validate_interval(interval)

        ticker = yf.Ticker(symbol)
        kwargs: dict = {"interval": interval, "auto_adjust": auto_adjust}
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

        Args:
            symbols: List of Yahoo Finance ticker symbols.
            period: Lookback period when *start*/*end* are omitted.
            interval: Bar size.
            start: Start date as "yyyy-mm-dd".
            end: End date as "yyyy-mm-dd".
            auto_adjust: Adjust prices for corporate actions.

        Returns:
            Multi-level DataFrame with tickers as the top-level column group
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
