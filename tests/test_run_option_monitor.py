"""
Tests for scripts/run_option_monitor.py

All network calls are mocked — no live data fetched.
"""

import sys
from datetime import date
from pathlib import Path
from unittest.mock import MagicMock, patch

import pandas as pd
import pytest
from typer.testing import CliRunner

sys.path.insert(0, str(Path(__file__).parent.parent / "scripts"))
from run_option_monitor import app, _fmt, _df_to_rich_table

# ── Shared fixture helpers ────────────────────────────────────────────────────

VDATE = "2025-06-09"

_EMPTY_RESULTS: dict[str, pd.DataFrame] = {
    "spot": pd.DataFrame(),
    "iv_term_structure": pd.DataFrame(),
    "iv_skew": pd.DataFrame(),
    "pc_overall": pd.DataFrame(),
    "pc_by_expiry": pd.DataFrame(),
    "pc_by_moneyness": pd.DataFrame(),
    "positioning": pd.DataFrame(),
}

_SPOT_DF = pd.DataFrame([{
    "symbol": "QQQ",
    "spot": 480.0,
    "change_pct": 0.5,
    "rv_5d": 12.3,
    "rv_20d": 14.1,
    "rv_60d": 16.2,
    "valuation_date": VDATE,
}])

_POSITIONING_DF = pd.DataFrame([{
    "symbol": "QQQ",
    "spot": 480.0,
    "max_pain": 475.0,
    "max_pain_dist_pct": -1.04,
    "call_wall": 500.0,
    "call_wall_dist_pct": 4.17,
    "put_wall": 460.0,
    "put_wall_dist_pct": -4.17,
    "valuation_date": VDATE,
}])

_MOCK_RESULTS = {
    **_EMPTY_RESULTS,
    "spot": _SPOT_DF,
    "positioning": _POSITIONING_DF,
}


def _make_monitor_mock(results=None):
    mock_monitor = MagicMock()
    mock_monitor.run.return_value = results or _MOCK_RESULTS
    return mock_monitor


# ── _fmt helper ───────────────────────────────────────────────────────────────

class TestFmt:
    def test_none_returns_dash(self):
        assert _fmt(None) == "-"

    def test_nan_returns_dash(self):
        import math
        assert _fmt(float("nan")) == "-"

    def test_float_four_decimals(self):
        assert _fmt(1.23456789) == "1.2346"

    def test_int_as_string(self):
        assert _fmt(42) == "42"

    def test_string_passthrough(self):
        assert _fmt("QQQ") == "QQQ"

    def test_zero_float(self):
        assert _fmt(0.0) == "0.0000"


# ── _df_to_rich_table helper ──────────────────────────────────────────────────

class TestDfToRichTable:
    def test_returns_table_with_correct_title(self):
        from rich.table import Table
        df = _SPOT_DF.copy()
        table = _df_to_rich_table(df, "Test Title")
        assert isinstance(table, Table)
        assert table.title == "Test Title"

    def test_valuation_date_column_excluded(self):
        from rich.table import Table
        df = _SPOT_DF.copy()
        table = _df_to_rich_table(df, "Spot")
        col_names = [c.header for c in table.columns]
        assert "valuation_date" not in col_names

    def test_symbol_column_present(self):
        df = _SPOT_DF.copy()
        table = _df_to_rich_table(df, "Spot")
        col_names = [c.header for c in table.columns]
        assert "symbol" in col_names

    def test_empty_df_produces_empty_table(self):
        from rich.table import Table
        table = _df_to_rich_table(pd.DataFrame(), "Empty")
        assert isinstance(table, Table)
        assert table.row_count == 0


# ── CLI: --help ───────────────────────────────────────────────────────────────

class TestHelp:
    def test_help_exits_zero(self):
        runner = CliRunner()
        result = runner.invoke(app, ["--help"])
        assert result.exit_code == 0

    def test_help_mentions_symbols(self):
        runner = CliRunner()
        result = runner.invoke(app, ["--help"])
        assert "--symbols" in result.output or "-s" in result.output

    def test_help_mentions_valuation_date(self):
        runner = CliRunner()
        result = runner.invoke(app, ["--help"])
        assert "valuation-date" in result.output or "-d" in result.output


# ── CLI: mutual-exclusion guard ───────────────────────────────────────────────

class TestMutualExclusion:
    def test_etfs_and_equities_together_exits_nonzero(self):
        runner = CliRunner()
        with patch("run_option_monitor.OptionMonitor", return_value=_make_monitor_mock()):
            result = runner.invoke(app, ["--etfs-only", "--equities-only"])
        assert result.exit_code != 0

    def test_etfs_and_equities_together_prints_error(self):
        runner = CliRunner()
        with patch("run_option_monitor.OptionMonitor", return_value=_make_monitor_mock()):
            result = runner.invoke(app, ["--etfs-only", "--equities-only"])
        assert "mutually exclusive" in result.output.lower() or "Error" in result.output


# ── CLI: symbol selection ─────────────────────────────────────────────────────

class TestSymbolSelection:
    def test_custom_symbols_uppercased_and_passed(self):
        runner = CliRunner()
        captured_symbols = []

        def fake_init(symbols=None, valuation_date=None):
            captured_symbols.extend(symbols or [])
            return _make_monitor_mock()

        with patch("run_option_monitor.OptionMonitor", side_effect=fake_init):
            runner.invoke(app, ["-s", "qqq,spy", "-d", VDATE])

        assert "QQQ" in captured_symbols
        assert "SPY" in captured_symbols

    def test_etfs_only_uses_etf_universe(self):
        from run_option_monitor import ETF_UNIVERSE
        runner = CliRunner()
        captured = []

        def fake_init(symbols=None, valuation_date=None):
            captured.extend(symbols or [])
            return _make_monitor_mock()

        with patch("run_option_monitor.OptionMonitor", side_effect=fake_init):
            runner.invoke(app, ["--etfs-only", "-d", VDATE])

        assert captured == ETF_UNIVERSE

    def test_equities_only_uses_equity_universe(self):
        from run_option_monitor import EQUITY_UNIVERSE
        runner = CliRunner()
        captured = []

        def fake_init(symbols=None, valuation_date=None):
            captured.extend(symbols or [])
            return _make_monitor_mock()

        with patch("run_option_monitor.OptionMonitor", side_effect=fake_init):
            runner.invoke(app, ["--equities-only", "-d", VDATE])

        assert captured == EQUITY_UNIVERSE

    def test_default_uses_full_universe(self):
        from run_option_monitor import DEFAULT_UNIVERSE
        runner = CliRunner()
        captured = []

        def fake_init(symbols=None, valuation_date=None):
            captured.extend(symbols or [])
            return _make_monitor_mock()

        with patch("run_option_monitor.OptionMonitor", side_effect=fake_init):
            runner.invoke(app, ["-d", VDATE])

        assert captured == DEFAULT_UNIVERSE


# ── CLI: valuation date ───────────────────────────────────────────────────────

class TestValuationDate:
    def test_explicit_date_forwarded(self):
        runner = CliRunner()
        captured = []

        def fake_init(symbols=None, valuation_date=None):
            captured.append(valuation_date)
            return _make_monitor_mock()

        with patch("run_option_monitor.OptionMonitor", side_effect=fake_init):
            runner.invoke(app, ["-d", "2025-06-09", "--equities-only"])

        assert captured == ["2025-06-09"]

    def test_default_date_is_today(self):
        runner = CliRunner()
        captured = []

        def fake_init(symbols=None, valuation_date=None):
            captured.append(valuation_date)
            return _make_monitor_mock()

        with patch("run_option_monitor.OptionMonitor", side_effect=fake_init):
            runner.invoke(app, ["--equities-only"])

        assert captured == [date.today().isoformat()]


# ── CLI: output ───────────────────────────────────────────────────────────────

class TestOutput:
    def test_exit_zero_on_success(self):
        runner = CliRunner()
        with patch("run_option_monitor.OptionMonitor", return_value=_make_monitor_mock()):
            result = runner.invoke(app, ["-d", VDATE, "--equities-only"])
        assert result.exit_code == 0

    def test_output_contains_done(self):
        runner = CliRunner()
        with patch("run_option_monitor.OptionMonitor", return_value=_make_monitor_mock()):
            result = runner.invoke(app, ["-d", VDATE, "--equities-only"])
        assert "Done" in result.output

    def test_output_contains_valuation_date(self):
        runner = CliRunner()
        with patch("run_option_monitor.OptionMonitor", return_value=_make_monitor_mock()):
            result = runner.invoke(app, ["-d", VDATE, "--equities-only"])
        assert VDATE in result.output

    def test_empty_tables_show_no_data(self):
        runner = CliRunner()
        with patch("run_option_monitor.OptionMonitor", return_value=_make_monitor_mock(_EMPTY_RESULTS)):
            result = runner.invoke(app, ["-d", VDATE, "-s", "QQQ"])
        assert "no data" in result.output.lower()


# ── CLI: CSV export ───────────────────────────────────────────────────────────

class TestCsvExport:
    def test_csv_files_created(self, tmp_path):
        runner = CliRunner()
        with patch("run_option_monitor.OptionMonitor", return_value=_make_monitor_mock()):
            result = runner.invoke(
                app, ["-d", VDATE, "--equities-only", "-o", str(tmp_path)]
            )
        assert result.exit_code == 0
        csv_files = list(tmp_path.glob("*.csv"))
        assert len(csv_files) == 7

    def test_csv_files_named_after_result_keys(self, tmp_path):
        runner = CliRunner()
        with patch("run_option_monitor.OptionMonitor", return_value=_make_monitor_mock()):
            runner.invoke(app, ["-d", VDATE, "--equities-only", "-o", str(tmp_path)])
        expected = {
            "spot.csv", "iv_term_structure.csv", "iv_skew.csv",
            "pc_overall.csv", "pc_by_expiry.csv", "pc_by_moneyness.csv",
            "positioning.csv",
        }
        actual = {f.name for f in tmp_path.glob("*.csv")}
        assert actual == expected

    def test_spot_csv_has_correct_rows(self, tmp_path):
        runner = CliRunner()
        with patch("run_option_monitor.OptionMonitor", return_value=_make_monitor_mock()):
            runner.invoke(app, ["-d", VDATE, "--equities-only", "-o", str(tmp_path)])
        df = pd.read_csv(tmp_path / "spot.csv")
        assert len(df) == 1
        assert df.iloc[0]["symbol"] == "QQQ"

    def test_no_output_dir_no_csvs(self, tmp_path):
        runner = CliRunner()
        with patch("run_option_monitor.OptionMonitor", return_value=_make_monitor_mock()):
            runner.invoke(app, ["-d", VDATE, "--equities-only"])
        assert list(tmp_path.glob("*.csv")) == []

    def test_output_dir_created_if_missing(self, tmp_path):
        new_dir = tmp_path / "subdir" / "nested"
        runner = CliRunner()
        with patch("run_option_monitor.OptionMonitor", return_value=_make_monitor_mock()):
            result = runner.invoke(
                app, ["-d", VDATE, "--equities-only", "-o", str(new_dir)]
            )
        assert result.exit_code == 0
        assert new_dir.exists()
