"""
Tests for YahooFinanceProvider and related base utilities.

All yfinance network calls are mocked so the suite runs offline.
"""

from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pandas as pd
import pytest

from datafeed.yahoo_finance import YahooFinanceProvider
from datafeed.yahoo_finance_base import (
    INTERVAL_MAX_HISTORY,
    SUPPORTED_ASSET_CLASSES,
    VALID_INTERVALS,
    YahooFinanceError,
    validate_interval,
    validate_period,
)


# ── Fixtures ─────────────────────────────────────────────────────────────────

@pytest.fixture()
def provider() -> YahooFinanceProvider:
    return YahooFinanceProvider()


def _ohlcv_df(rows: int = 5) -> pd.DataFrame:
    """Return a minimal OHLCV DataFrame."""
    idx = pd.date_range("2024-01-01", periods=rows, freq="D", tz="UTC")
    return pd.DataFrame(
        {
            "Open": [100.0] * rows,
            "High": [105.0] * rows,
            "Low": [98.0] * rows,
            "Close": [102.0] * rows,
            "Volume": [1_000_000] * rows,
        },
        index=idx,
    )


def _mock_ticker(
    history_df: pd.DataFrame | None = None,
    info: dict | None = None,
    dividends: pd.Series | None = None,
    splits: pd.Series | None = None,
    income_stmt: pd.DataFrame | None = None,
    quarterly_income_stmt: pd.DataFrame | None = None,
    balance_sheet: pd.DataFrame | None = None,
    quarterly_balance_sheet: pd.DataFrame | None = None,
    cashflow: pd.DataFrame | None = None,
    quarterly_cashflow: pd.DataFrame | None = None,
    options: tuple = ("2025-01-17", "2025-02-21"),
    option_chain_result=None,
    recommendations: pd.DataFrame | None = None,
    news: list | None = None,
    calendar: dict | None = None,
) -> MagicMock:
    """Build a MagicMock that mimics a yfinance.Ticker object."""
    mock = MagicMock()
    mock.history.return_value = history_df if history_df is not None else _ohlcv_df()
    mock.info = info or {"symbol": "AAPL", "sector": "Technology", "marketCap": 3e12}
    mock.dividends = dividends if dividends is not None else pd.Series(
        [0.24, 0.25],
        index=pd.to_datetime(["2024-02-09", "2024-05-10"], utc=True),
        name="Dividends",
    )
    mock.splits = splits if splits is not None else pd.Series(
        [4.0], index=pd.to_datetime(["2020-08-31"], utc=True), name="Stock Splits"
    )
    mock.income_stmt = income_stmt if income_stmt is not None else pd.DataFrame(
        {"2023": [1e9, 2e9]}, index=["Total Revenue", "Net Income"]
    )
    mock.quarterly_income_stmt = quarterly_income_stmt if quarterly_income_stmt is not None else pd.DataFrame(
        {"Q4-2023": [2.5e8, 5e8]}, index=["Total Revenue", "Net Income"]
    )
    mock.balance_sheet = balance_sheet if balance_sheet is not None else pd.DataFrame(
        {"2023": [5e9]}, index=["Total Assets"]
    )
    mock.quarterly_balance_sheet = quarterly_balance_sheet if quarterly_balance_sheet is not None else pd.DataFrame(
        {"Q4-2023": [5.1e9]}, index=["Total Assets"]
    )
    mock.cashflow = cashflow if cashflow is not None else pd.DataFrame(
        {"2023": [1e9]}, index=["Operating Cash Flow"]
    )
    mock.quarterly_cashflow = quarterly_cashflow if quarterly_cashflow is not None else pd.DataFrame(
        {"Q4-2023": [2.5e8]}, index=["Operating Cash Flow"]
    )
    mock.options = options
    if option_chain_result is None:
        calls_df = pd.DataFrame({"strike": [150.0, 155.0], "bid": [2.0, 1.5]})
        puts_df = pd.DataFrame({"strike": [145.0, 140.0], "bid": [1.8, 1.2]})
        option_chain_result = SimpleNamespace(calls=calls_df, puts=puts_df)
    mock.option_chain.return_value = option_chain_result
    mock.recommendations = recommendations if recommendations is not None else pd.DataFrame(
        {"Firm": ["Goldman"], "To Grade": ["Buy"], "Action": ["upgrade"]}
    )
    mock.news = news if news is not None else [
        {"title": "Apple hits record high", "link": "https://example.com/1"}
    ]
    mock.calendar = calendar or {"Earnings Date": "2024-07-30"}
    return mock


# ── validate_period ───────────────────────────────────────────────────────────

class TestValidatePeriod:
    def test_valid_periods_pass(self):
        for p in ["1d", "5d", "1mo", "3mo", "6mo", "1y", "2y", "5y", "10y", "ytd", "max"]:
            validate_period(p)  # should not raise

    def test_invalid_period_raises(self):
        with pytest.raises(ValueError, match="Invalid period"):
            validate_period("3y")

    def test_empty_string_raises(self):
        with pytest.raises(ValueError):
            validate_period("")


# ── validate_interval ─────────────────────────────────────────────────────────

class TestValidateInterval:
    def test_valid_intervals_pass(self):
        for iv in ["1m", "5m", "15m", "30m", "60m", "1h", "1d", "1wk", "1mo", "3mo"]:
            validate_interval(iv)

    def test_invalid_interval_raises(self):
        with pytest.raises(ValueError, match="Invalid interval"):
            validate_interval("4h")

    def test_empty_string_raises(self):
        with pytest.raises(ValueError):
            validate_interval("")


# ── get_history ───────────────────────────────────────────────────────────────

class TestGetHistory:
    def test_returns_dataframe(self, provider):
        with patch("datafeed.yahoo_finance.yf.Ticker", return_value=_mock_ticker()):
            df = provider.get_history("AAPL")
        assert isinstance(df, pd.DataFrame)
        assert list(df.columns) == ["Open", "High", "Low", "Close", "Volume"]

    def test_period_and_interval_forwarded(self, provider):
        mock = _mock_ticker()
        with patch("datafeed.yahoo_finance.yf.Ticker", return_value=mock):
            provider.get_history("AAPL", period="6mo", interval="1wk")
        kw = mock.history.call_args.kwargs
        assert kw["interval"] == "1wk"
        assert kw["auto_adjust"] is True
        assert kw["period"] == "6mo"

    def test_start_end_override_period(self, provider):
        mock = _mock_ticker()
        with patch("datafeed.yahoo_finance.yf.Ticker", return_value=mock):
            provider.get_history("AAPL", start="2024-01-01", end="2024-06-01")
        call_kwargs = mock.history.call_args.kwargs
        assert call_kwargs["start"] == "2024-01-01"
        assert call_kwargs["end"] == "2024-06-01"
        assert "period" not in call_kwargs

    def test_only_start_no_end(self, provider):
        mock = _mock_ticker()
        with patch("datafeed.yahoo_finance.yf.Ticker", return_value=mock):
            provider.get_history("AAPL", start="2024-01-01")
        call_kwargs = mock.history.call_args.kwargs
        assert call_kwargs["start"] == "2024-01-01"
        assert "end" not in call_kwargs
        assert "period" not in call_kwargs

    def test_empty_response_raises(self, provider):
        mock = _mock_ticker(history_df=pd.DataFrame())
        with patch("datafeed.yahoo_finance.yf.Ticker", return_value=mock):
            with pytest.raises(YahooFinanceError, match="No price data"):
                provider.get_history("INVALID")

    def test_invalid_period_raises(self, provider):
        with pytest.raises(ValueError, match="Invalid period"):
            provider.get_history("AAPL", period="3y")

    def test_invalid_interval_raises(self, provider):
        with pytest.raises(ValueError, match="Invalid interval"):
            provider.get_history("AAPL", interval="4h")

    def test_auto_adjust_false(self, provider):
        mock = _mock_ticker()
        with patch("datafeed.yahoo_finance.yf.Ticker", return_value=mock):
            provider.get_history("AAPL", auto_adjust=False)
        assert mock.history.call_args.kwargs["auto_adjust"] is False

    def test_crypto_symbol(self, provider):
        with patch("datafeed.yahoo_finance.yf.Ticker", return_value=_mock_ticker()):
            df = provider.get_history("BTC-USD")
        assert not df.empty

    def test_forex_symbol(self, provider):
        with patch("datafeed.yahoo_finance.yf.Ticker", return_value=_mock_ticker()):
            df = provider.get_history("EURUSD=X")
        assert not df.empty


# ── get_info ──────────────────────────────────────────────────────────────────

class TestGetInfo:
    def test_returns_dict(self, provider):
        with patch("datafeed.yahoo_finance.yf.Ticker", return_value=_mock_ticker()):
            info = provider.get_info("AAPL")
        assert isinstance(info, dict)
        assert "symbol" in info

    def test_info_contents(self, provider):
        expected = {"symbol": "MSFT", "sector": "Technology", "marketCap": 2e12}
        mock = _mock_ticker(info=expected)
        with patch("datafeed.yahoo_finance.yf.Ticker", return_value=mock):
            info = provider.get_info("MSFT")
        assert info["sector"] == "Technology"
        assert info["marketCap"] == 2e12


# ── get_dividends ─────────────────────────────────────────────────────────────

class TestGetDividends:
    def test_returns_series(self, provider):
        with patch("datafeed.yahoo_finance.yf.Ticker", return_value=_mock_ticker()):
            divs = provider.get_dividends("AAPL")
        assert isinstance(divs, pd.Series)

    def test_start_filter(self, provider):
        divs_data = pd.Series(
            [0.20, 0.22, 0.24],
            index=pd.to_datetime(["2023-02-09", "2023-05-10", "2024-02-09"], utc=True),
        )
        mock = _mock_ticker(dividends=divs_data)
        with patch("datafeed.yahoo_finance.yf.Ticker", return_value=mock):
            result = provider.get_dividends("AAPL", start="2024-01-01")
        assert len(result) == 1
        assert result.iloc[0] == pytest.approx(0.24)

    def test_end_filter(self, provider):
        divs_data = pd.Series(
            [0.20, 0.22, 0.24],
            index=pd.to_datetime(["2023-02-09", "2023-05-10", "2024-02-09"], utc=True),
        )
        mock = _mock_ticker(dividends=divs_data)
        with patch("datafeed.yahoo_finance.yf.Ticker", return_value=mock):
            result = provider.get_dividends("AAPL", end="2023-12-31")
        assert len(result) == 2

    def test_start_and_end_filter(self, provider):
        divs_data = pd.Series(
            [0.20, 0.22, 0.24],
            index=pd.to_datetime(["2023-02-09", "2023-05-10", "2024-02-09"], utc=True),
        )
        mock = _mock_ticker(dividends=divs_data)
        with patch("datafeed.yahoo_finance.yf.Ticker", return_value=mock):
            result = provider.get_dividends("AAPL", start="2023-04-01", end="2023-12-31")
        assert len(result) == 1
        assert result.iloc[0] == pytest.approx(0.22)

    def test_no_filter_returns_all(self, provider):
        with patch("datafeed.yahoo_finance.yf.Ticker", return_value=_mock_ticker()):
            result = provider.get_dividends("AAPL")
        assert len(result) == 2


# ── get_splits ────────────────────────────────────────────────────────────────

class TestGetSplits:
    def test_returns_series(self, provider):
        with patch("datafeed.yahoo_finance.yf.Ticker", return_value=_mock_ticker()):
            splits = provider.get_splits("AAPL")
        assert isinstance(splits, pd.Series)

    def test_split_ratio(self, provider):
        with patch("datafeed.yahoo_finance.yf.Ticker", return_value=_mock_ticker()):
            splits = provider.get_splits("AAPL")
        assert splits.iloc[0] == pytest.approx(4.0)


# ── get_income_statement ──────────────────────────────────────────────────────

class TestGetIncomeStatement:
    def test_annual_returns_dataframe(self, provider):
        with patch("datafeed.yahoo_finance.yf.Ticker", return_value=_mock_ticker()):
            stmt = provider.get_income_statement("AAPL")
        assert isinstance(stmt, pd.DataFrame)

    def test_quarterly_returns_dataframe(self, provider):
        with patch("datafeed.yahoo_finance.yf.Ticker", return_value=_mock_ticker()):
            stmt = provider.get_income_statement("AAPL", quarterly=True)
        assert isinstance(stmt, pd.DataFrame)

    def test_annual_vs_quarterly_differ(self, provider):
        mock = _mock_ticker()
        with patch("datafeed.yahoo_finance.yf.Ticker", return_value=mock):
            annual = provider.get_income_statement("AAPL", quarterly=False)
            quarterly = provider.get_income_statement("AAPL", quarterly=True)
        assert not annual.equals(quarterly)


# ── get_balance_sheet ─────────────────────────────────────────────────────────

class TestGetBalanceSheet:
    def test_annual(self, provider):
        with patch("datafeed.yahoo_finance.yf.Ticker", return_value=_mock_ticker()):
            bs = provider.get_balance_sheet("AAPL")
        assert isinstance(bs, pd.DataFrame)

    def test_quarterly(self, provider):
        with patch("datafeed.yahoo_finance.yf.Ticker", return_value=_mock_ticker()):
            bs = provider.get_balance_sheet("AAPL", quarterly=True)
        assert isinstance(bs, pd.DataFrame)


# ── get_cash_flow ─────────────────────────────────────────────────────────────

class TestGetCashFlow:
    def test_annual(self, provider):
        with patch("datafeed.yahoo_finance.yf.Ticker", return_value=_mock_ticker()):
            cf = provider.get_cash_flow("AAPL")
        assert isinstance(cf, pd.DataFrame)

    def test_quarterly(self, provider):
        with patch("datafeed.yahoo_finance.yf.Ticker", return_value=_mock_ticker()):
            cf = provider.get_cash_flow("AAPL", quarterly=True)
        assert isinstance(cf, pd.DataFrame)


# ── get_option_expirations ────────────────────────────────────────────────────

class TestGetOptionExpirations:
    def test_returns_tuple(self, provider):
        with patch("datafeed.yahoo_finance.yf.Ticker", return_value=_mock_ticker()):
            exps = provider.get_option_expirations("AAPL")
        assert isinstance(exps, tuple)
        assert len(exps) == 2

    def test_expiration_format(self, provider):
        with patch("datafeed.yahoo_finance.yf.Ticker", return_value=_mock_ticker()):
            exps = provider.get_option_expirations("AAPL")
        # each should be a date string
        for exp in exps:
            pd.to_datetime(exp)  # raises if invalid


# ── get_option_chain ──────────────────────────────────────────────────────────

class TestGetOptionChain:
    def test_returns_two_dataframes(self, provider):
        with patch("datafeed.yahoo_finance.yf.Ticker", return_value=_mock_ticker()):
            calls, puts = provider.get_option_chain("AAPL", "2025-01-17")
        assert isinstance(calls, pd.DataFrame)
        assert isinstance(puts, pd.DataFrame)

    def test_calls_have_strike(self, provider):
        with patch("datafeed.yahoo_finance.yf.Ticker", return_value=_mock_ticker()):
            calls, _ = provider.get_option_chain("AAPL", "2025-01-17")
        assert "strike" in calls.columns

    def test_puts_have_strike(self, provider):
        with patch("datafeed.yahoo_finance.yf.Ticker", return_value=_mock_ticker()):
            _, puts = provider.get_option_chain("AAPL", "2025-01-17")
        assert "strike" in puts.columns

    def test_expiration_forwarded(self, provider):
        mock = _mock_ticker()
        with patch("datafeed.yahoo_finance.yf.Ticker", return_value=mock):
            provider.get_option_chain("AAPL", "2025-02-21")
        mock.option_chain.assert_called_once_with("2025-02-21")


# ── download (bulk) ───────────────────────────────────────────────────────────

class TestDownload:
    def _multi_df(self, symbols: list[str]) -> pd.DataFrame:
        idx = pd.date_range("2024-01-01", periods=5, freq="D")
        cols = pd.MultiIndex.from_product([symbols, ["Open", "High", "Low", "Close", "Volume"]])
        return pd.DataFrame(1.0, index=idx, columns=cols)

    def test_returns_dataframe(self, provider):
        df = self._multi_df(["AAPL", "MSFT"])
        with patch("datafeed.yahoo_finance.yf.download", return_value=df):
            result = provider.download(["AAPL", "MSFT"])
        assert isinstance(result, pd.DataFrame)
        assert not result.empty

    def test_period_forwarded(self, provider):
        df = self._multi_df(["AAPL"])
        with patch("datafeed.yahoo_finance.yf.download", return_value=df) as mock_dl:
            provider.download(["AAPL"], period="6mo")
        call_kwargs = mock_dl.call_args.kwargs
        assert call_kwargs["period"] == "6mo"

    def test_start_end_override_period(self, provider):
        df = self._multi_df(["AAPL"])
        with patch("datafeed.yahoo_finance.yf.download", return_value=df) as mock_dl:
            provider.download(["AAPL"], start="2024-01-01", end="2024-06-01")
        call_kwargs = mock_dl.call_args.kwargs
        assert call_kwargs["start"] == "2024-01-01"
        assert call_kwargs["end"] == "2024-06-01"
        assert "period" not in call_kwargs

    def test_empty_response_raises(self, provider):
        with patch("datafeed.yahoo_finance.yf.download", return_value=pd.DataFrame()):
            with pytest.raises(YahooFinanceError, match="No data returned"):
                provider.download(["INVALID"])

    def test_invalid_period_raises(self, provider):
        with pytest.raises(ValueError, match="Invalid period"):
            provider.download(["AAPL"], period="3y")

    def test_invalid_interval_raises(self, provider):
        with pytest.raises(ValueError, match="Invalid interval"):
            provider.download(["AAPL"], interval="4h")

    def test_group_by_ticker(self, provider):
        df = self._multi_df(["AAPL"])
        with patch("datafeed.yahoo_finance.yf.download", return_value=df) as mock_dl:
            provider.download(["AAPL"])
        assert mock_dl.call_args.kwargs.get("group_by") == "ticker"

    def test_auto_adjust_passed(self, provider):
        df = self._multi_df(["AAPL"])
        with patch("datafeed.yahoo_finance.yf.download", return_value=df) as mock_dl:
            provider.download(["AAPL"], auto_adjust=False)
        assert mock_dl.call_args.kwargs["auto_adjust"] is False


# ── get_recommendations ───────────────────────────────────────────────────────

class TestGetRecommendations:
    def test_returns_dataframe(self, provider):
        with patch("datafeed.yahoo_finance.yf.Ticker", return_value=_mock_ticker()):
            recs = provider.get_recommendations("AAPL")
        assert isinstance(recs, pd.DataFrame)

    def test_has_firm_column(self, provider):
        with patch("datafeed.yahoo_finance.yf.Ticker", return_value=_mock_ticker()):
            recs = provider.get_recommendations("AAPL")
        assert "Firm" in recs.columns


# ── get_news ──────────────────────────────────────────────────────────────────

class TestGetNews:
    def test_returns_list(self, provider):
        with patch("datafeed.yahoo_finance.yf.Ticker", return_value=_mock_ticker()):
            news = provider.get_news("AAPL")
        assert isinstance(news, list)

    def test_news_items_are_dicts(self, provider):
        with patch("datafeed.yahoo_finance.yf.Ticker", return_value=_mock_ticker()):
            news = provider.get_news("AAPL")
        assert all(isinstance(item, dict) for item in news)

    def test_news_has_title(self, provider):
        with patch("datafeed.yahoo_finance.yf.Ticker", return_value=_mock_ticker()):
            news = provider.get_news("AAPL")
        assert "title" in news[0]


# ── get_calendar ──────────────────────────────────────────────────────────────

class TestGetCalendar:
    def test_returns_dict(self, provider):
        with patch("datafeed.yahoo_finance.yf.Ticker", return_value=_mock_ticker()):
            cal = provider.get_calendar("AAPL")
        assert isinstance(cal, dict)

    def test_earnings_date_present(self, provider):
        with patch("datafeed.yahoo_finance.yf.Ticker", return_value=_mock_ticker()):
            cal = provider.get_calendar("AAPL")
        assert "Earnings Date" in cal


# ── Coverage ──────────────────────────────────────────────────────────────────

class TestCoverage:
    EXPECTED_ASSET_CLASSES = {
        "equities_us", "equities_international", "etfs", "indices",
        "crypto", "forex", "futures", "mutual_funds", "bonds_rates",
    }

    def test_all_expected_asset_classes_present(self):
        assert self.EXPECTED_ASSET_CLASSES.issubset(SUPPORTED_ASSET_CLASSES.keys())

    def test_each_asset_class_has_required_keys(self):
        required = {"description", "symbol_format", "examples", "history"}
        for name, info in SUPPORTED_ASSET_CLASSES.items():
            missing = required - info.keys()
            assert not missing, f"'{name}' is missing keys: {missing}"

    def test_examples_are_non_empty_lists(self):
        for name, info in SUPPORTED_ASSET_CLASSES.items():
            assert isinstance(info["examples"], list) and info["examples"], (
                f"'{name}' must have at least one example symbol"
            )

    def test_financials_and_options_flags_are_booleans(self):
        for name, info in SUPPORTED_ASSET_CLASSES.items():
            assert isinstance(info.get("financials"), bool), f"'{name}'.financials not bool"
            assert isinstance(info.get("options"), bool), f"'{name}'.options not bool"

    def test_equities_us_supports_financials_and_options(self):
        eq = SUPPORTED_ASSET_CLASSES["equities_us"]
        assert eq["financials"] is True
        assert eq["options"] is True

    def test_crypto_has_no_financials_or_options(self):
        cr = SUPPORTED_ASSET_CLASSES["crypto"]
        assert cr["financials"] is False
        assert cr["options"] is False

    def test_forex_has_no_financials_or_options(self):
        fx = SUPPORTED_ASSET_CLASSES["forex"]
        assert fx["financials"] is False
        assert fx["options"] is False

    def test_all_valid_intervals_have_max_history_entry(self):
        for interval in VALID_INTERVALS:
            assert interval in INTERVAL_MAX_HISTORY, (
                f"Interval '{interval}' missing from INTERVAL_MAX_HISTORY"
            )

    def test_one_minute_limited_to_7_days(self):
        assert "7" in INTERVAL_MAX_HISTORY["1m"]

    def test_sub_hour_intervals_limited_to_60_days(self):
        for iv in ["2m", "5m", "15m", "30m", "90m"]:
            assert "60" in INTERVAL_MAX_HISTORY[iv], (
                f"Interval '{iv}' should show 60-day limit"
            )

    def test_hourly_intervals_limited_to_730_days(self):
        for iv in ["60m", "1h"]:
            assert "730" in INTERVAL_MAX_HISTORY[iv], (
                f"Interval '{iv}' should show 730-day limit"
            )

    def test_daily_and_coarser_have_full_history(self):
        for iv in ["1d", "5d", "1wk", "1mo", "3mo"]:
            assert "full" in INTERVAL_MAX_HISTORY[iv].lower(), (
                f"Interval '{iv}' should indicate full history"
            )

    def test_get_coverage_has_three_sections(self, provider):
        cov = provider.get_coverage()
        assert "asset_classes" in cov
        assert "interval_max_history" in cov
        assert "price_adjustment" in cov

    def test_get_coverage_asset_classes_matches_constant(self, provider):
        assert provider.get_coverage()["asset_classes"] is SUPPORTED_ASSET_CLASSES

    def test_get_coverage_interval_history_matches_constant(self, provider):
        assert provider.get_coverage()["interval_max_history"] is INTERVAL_MAX_HISTORY

    def test_get_coverage_is_callable_as_static(self):
        # Should work without instantiating the class
        cov = YahooFinanceProvider.get_coverage()
        assert isinstance(cov, dict)


# ── Price adjustment ──────────────────────────────────────────────────────────

class TestPriceAdjustment:
    """Verify that price-adjustment flags are forwarded correctly to yfinance."""

    def test_auto_adjust_true_by_default(self, provider):
        mock = _mock_ticker()
        with patch("datafeed.yahoo_finance.yf.Ticker", return_value=mock):
            provider.get_history("AAPL")
        assert mock.history.call_args.kwargs["auto_adjust"] is True

    def test_auto_adjust_false_forwarded(self, provider):
        mock = _mock_ticker()
        with patch("datafeed.yahoo_finance.yf.Ticker", return_value=mock):
            provider.get_history("AAPL", auto_adjust=False)
        assert mock.history.call_args.kwargs["auto_adjust"] is False

    def test_back_adjust_false_by_default(self, provider):
        mock = _mock_ticker()
        with patch("datafeed.yahoo_finance.yf.Ticker", return_value=mock):
            provider.get_history("AAPL")
        assert mock.history.call_args.kwargs["back_adjust"] is False

    def test_back_adjust_true_forwarded(self, provider):
        mock = _mock_ticker()
        with patch("datafeed.yahoo_finance.yf.Ticker", return_value=mock):
            provider.get_history("AAPL", back_adjust=True)
        assert mock.history.call_args.kwargs["back_adjust"] is True

    def test_raw_prices_returned_when_auto_adjust_false(self, provider):
        raw_df = _ohlcv_df()
        mock = _mock_ticker(history_df=raw_df)
        with patch("datafeed.yahoo_finance.yf.Ticker", return_value=mock):
            df = provider.get_history("AAPL", auto_adjust=False)
        assert not df.empty

    def test_both_flags_forwarded_independently(self, provider):
        mock = _mock_ticker()
        with patch("datafeed.yahoo_finance.yf.Ticker", return_value=mock):
            provider.get_history("AAPL", auto_adjust=False, back_adjust=True)
        kw = mock.history.call_args.kwargs
        assert kw["auto_adjust"] is False
        assert kw["back_adjust"] is True

    def test_coverage_documents_three_adjustment_modes(self, provider):
        adj = provider.get_coverage()["price_adjustment"]
        assert "auto_adjust_true" in adj
        assert "auto_adjust_false" in adj
        assert "back_adjust_true" in adj

    def test_auto_adjust_true_description_mentions_dividends(self, provider):
        desc = provider.get_coverage()["price_adjustment"]["auto_adjust_true"].lower()
        assert "dividend" in desc

    def test_auto_adjust_true_description_mentions_splits(self, provider):
        desc = provider.get_coverage()["price_adjustment"]["auto_adjust_true"].lower()
        assert "split" in desc

    def test_auto_adjust_false_description_mentions_raw_or_unadjusted(self, provider):
        desc = provider.get_coverage()["price_adjustment"]["auto_adjust_false"].lower()
        assert "raw" in desc or "unadjusted" in desc

    def test_back_adjust_true_description_mentions_back_adjust(self, provider):
        desc = provider.get_coverage()["price_adjustment"]["back_adjust_true"].lower()
        assert "back" in desc or "anchor" in desc or "proportion" in desc
