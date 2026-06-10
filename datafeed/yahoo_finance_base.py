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

# Maximum historical lookback Yahoo Finance returns per bar interval.
# Requesting data beyond these windows returns an empty result.
INTERVAL_MAX_HISTORY: dict[str, str] = {
    "1m":  "7 days",
    "2m":  "60 days",
    "5m":  "60 days",
    "15m": "60 days",
    "30m": "60 days",
    "90m": "60 days",
    "60m": "730 days (2 years)",
    "1h":  "730 days (2 years)",
    "1d":  "full history (varies by symbol; 20-30+ years for major US equities)",
    "5d":  "full history",
    "1wk": "full history",
    "1mo": "full history",
    "3mo": "full history",
}

# Supported asset classes, their Yahoo Finance symbol conventions, and capabilities.
SUPPORTED_ASSET_CLASSES: dict[str, dict] = {
    "equities_us": {
        "description": "US-listed stocks and ADRs",
        "symbol_format": "TICKER",
        "examples": ["AAPL", "MSFT", "TSLA", "NVDA"],
        "history": "full (20-30+ years for major names)",
        "financials": True,
        "options": True,
    },
    "equities_international": {
        "description": "Non-US listed stocks",
        "symbol_format": "TICKER.EXCHANGE",
        "examples": ["7203.T", "NESN.SW", "BP.L", "SAP.DE"],
        "history": "full (depth varies by exchange)",
        "financials": True,
        "options": False,
    },
    "etfs": {
        "description": "Exchange-traded funds",
        "symbol_format": "TICKER",
        "examples": ["SPY", "QQQ", "GLD", "TLT"],
        "history": "full (since fund inception)",
        "financials": False,
        "options": True,
    },
    "indices": {
        "description": "Market indices (informational; not directly tradeable)",
        "symbol_format": "^INDEX",
        "examples": ["^GSPC", "^IXIC", "^DJI", "^VIX", "^RUT"],
        "history": "full",
        "financials": False,
        "options": False,
    },
    "crypto": {
        "description": "Cryptocurrency spot prices denominated in a fiat currency",
        "symbol_format": "COIN-CURRENCY",
        "examples": ["BTC-USD", "ETH-USD", "SOL-USD", "BNB-USD"],
        "history": "since ~2014 for BTC; varies by coin",
        "financials": False,
        "options": False,
    },
    "forex": {
        "description": "Foreign exchange spot rates",
        "symbol_format": "BASECURRENCY=X  (or 6-char pair + =X)",
        "examples": ["EURUSD=X", "GBPUSD=X", "USDJPY=X", "AUDUSD=X"],
        "history": "full",
        "financials": False,
        "options": False,
    },
    "futures": {
        "description": "Commodity and financial futures (front-month continuous contract)",
        "symbol_format": "CONTRACT=F",
        "examples": ["CL=F", "GC=F", "ES=F", "NQ=F", "ZN=F"],
        "history": "limited (front-month only; no continuous back-adjusted series)",
        "financials": False,
        "options": False,
    },
    "mutual_funds": {
        "description": "US mutual funds (daily NAV)",
        "symbol_format": "TICKER",
        "examples": ["VFIAX", "FXAIX", "VTSAX"],
        "history": "full (since fund inception)",
        "financials": False,
        "options": False,
    },
    "bonds_rates": {
        "description": "US Treasury yield indices",
        "symbol_format": "^INDEX",
        "examples": ["^TNX", "^FVX", "^IRX", "^TYX"],
        "history": "full",
        "financials": False,
        "options": False,
    },
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
