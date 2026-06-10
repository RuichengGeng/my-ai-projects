"""
Base utilities for the Yahoo Finance data feed.
"""

import yfinance as yf

VALID_PERIODS = {
    "1d", "5d", "1mo", "3mo", "6mo",
    "1y", "2y", "5y", "10y", "ytd", "max",
}

VALID_INTERVALS = {
    "1m", "2m", "5m", "15m", "30m", "60m", "90m",
    "1h", "1d", "5d", "1wk", "1mo", "3mo",
}


class YahooFinanceError(Exception):
    """Raised when Yahoo Finance returns no data or an unexpected response."""


def validate_period(period: str) -> None:
    if period not in VALID_PERIODS:
        raise ValueError(
            f"Invalid period '{period}'. Valid options: {sorted(VALID_PERIODS)}"
        )


def validate_interval(interval: str) -> None:
    if interval not in VALID_INTERVALS:
        raise ValueError(
            f"Invalid interval '{interval}'. Valid options: {sorted(VALID_INTERVALS)}"
        )


def get_ticker(symbol: str) -> yf.Ticker:
    return yf.Ticker(symbol)
