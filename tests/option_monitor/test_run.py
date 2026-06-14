"""
Tests for option_monitor/run.py

All network calls are mocked - no live data fetched.
"""

from unittest.mock import MagicMock, patch

import pandas as pd

from option_monitor import run as run_option_monitor
from option_monitor.run import _df_to_rich_table, _fmt, main

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


# ── Script configuration ─────────────────────────────────────────────────────

class TestScriptConfiguration:
    def test_configured_symbols_and_date_forwarded(self, monkeypatch):
        captured = {}

        def fake_init(symbols=None, valuation_date=None):
            captured["symbols"] = symbols
            captured["valuation_date"] = valuation_date
            return _make_monitor_mock()

        monkeypatch.setattr(run_option_monitor, "SYMBOLS", ["QQQ", "SPY"])
        monkeypatch.setattr(run_option_monitor, "VALUATION_DATE", VDATE)
        monkeypatch.setattr(run_option_monitor, "OUTPUT_DIR", None)

        with patch("option_monitor.run.OptionMonitor", side_effect=fake_init):
            main()

        assert captured == {"symbols": ["QQQ", "SPY"], "valuation_date": VDATE}

    def test_empty_tables_show_no_data(self, monkeypatch, capsys):
        monkeypatch.setattr(run_option_monitor, "SYMBOLS", ["QQQ"])
        monkeypatch.setattr(run_option_monitor, "VALUATION_DATE", VDATE)
        monkeypatch.setattr(run_option_monitor, "OUTPUT_DIR", None)

        with patch("option_monitor.run.OptionMonitor", return_value=_make_monitor_mock(_EMPTY_RESULTS)):
            main()

        assert "no data" in capsys.readouterr().out.lower()

    def test_output_contains_done_and_valuation_date(self, monkeypatch, capsys):
        monkeypatch.setattr(run_option_monitor, "SYMBOLS", ["QQQ"])
        monkeypatch.setattr(run_option_monitor, "VALUATION_DATE", VDATE)
        monkeypatch.setattr(run_option_monitor, "OUTPUT_DIR", None)

        with patch("option_monitor.run.OptionMonitor", return_value=_make_monitor_mock()):
            main()

        output = capsys.readouterr().out
        assert "Done" in output
        assert VDATE in output


# ── CSV export ────────────────────────────────────────────────────────────────

class TestCsvExport:
    def test_csv_files_created_in_dated_snapshot_dir(self, monkeypatch, tmp_path):
        monkeypatch.setattr(run_option_monitor, "SYMBOLS", ["QQQ"])
        monkeypatch.setattr(run_option_monitor, "VALUATION_DATE", VDATE)
        monkeypatch.setattr(run_option_monitor, "OUTPUT_DIR", str(tmp_path))

        with patch("option_monitor.run.OptionMonitor", return_value=_make_monitor_mock()):
            main()

        csv_files = list((tmp_path / VDATE).glob("*.csv"))
        assert len(csv_files) == 7

    def test_latest_dir_mirrors_snapshot(self, monkeypatch, tmp_path):
        monkeypatch.setattr(run_option_monitor, "SYMBOLS", ["QQQ"])
        monkeypatch.setattr(run_option_monitor, "VALUATION_DATE", VDATE)
        monkeypatch.setattr(run_option_monitor, "OUTPUT_DIR", str(tmp_path))

        with patch("option_monitor.run.OptionMonitor", return_value=_make_monitor_mock()):
            main()

        snap = {f.name for f in (tmp_path / VDATE).glob("*.csv")}
        latest = {f.name for f in (tmp_path / "latest").glob("*.csv")}
        assert snap == latest and len(latest) == 7

    def test_csv_files_named_after_result_keys(self, monkeypatch, tmp_path):
        monkeypatch.setattr(run_option_monitor, "SYMBOLS", ["QQQ"])
        monkeypatch.setattr(run_option_monitor, "VALUATION_DATE", VDATE)
        monkeypatch.setattr(run_option_monitor, "OUTPUT_DIR", str(tmp_path))

        with patch("option_monitor.run.OptionMonitor", return_value=_make_monitor_mock()):
            main()

        expected = {
            "spot.csv", "iv_term_structure.csv", "iv_skew.csv",
            "pc_overall.csv", "pc_by_expiry.csv", "pc_by_moneyness.csv",
            "positioning.csv",
        }
        actual = {f.name for f in (tmp_path / VDATE).glob("*.csv")}
        assert actual == expected

    def test_spot_csv_has_correct_rows(self, monkeypatch, tmp_path):
        monkeypatch.setattr(run_option_monitor, "SYMBOLS", ["QQQ"])
        monkeypatch.setattr(run_option_monitor, "VALUATION_DATE", VDATE)
        monkeypatch.setattr(run_option_monitor, "OUTPUT_DIR", str(tmp_path))

        with patch("option_monitor.run.OptionMonitor", return_value=_make_monitor_mock()):
            main()

        df = pd.read_csv(tmp_path / VDATE / "spot.csv")
        assert len(df) == 1
        assert df.iloc[0]["symbol"] == "QQQ"

    def test_no_output_dir_no_csvs(self, monkeypatch, tmp_path):
        monkeypatch.setattr(run_option_monitor, "SYMBOLS", ["QQQ"])
        monkeypatch.setattr(run_option_monitor, "VALUATION_DATE", VDATE)
        monkeypatch.setattr(run_option_monitor, "OUTPUT_DIR", None)

        with patch("option_monitor.run.OptionMonitor", return_value=_make_monitor_mock()):
            main()

        assert list(tmp_path.glob("*.csv")) == []

    def test_output_dir_created_if_missing(self, monkeypatch, tmp_path):
        new_dir = tmp_path / "subdir" / "nested"
        monkeypatch.setattr(run_option_monitor, "SYMBOLS", ["QQQ"])
        monkeypatch.setattr(run_option_monitor, "VALUATION_DATE", VDATE)
        monkeypatch.setattr(run_option_monitor, "OUTPUT_DIR", str(new_dir))

        with patch("option_monitor.run.OptionMonitor", return_value=_make_monitor_mock()):
            main()

        assert new_dir.exists()
