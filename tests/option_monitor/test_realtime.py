"""Tests for realtime quote helpers — fully offline."""

import json
import time
from datetime import datetime, timedelta, timezone

import pandas as pd

from option_monitor.realtime import (
    STALE_AFTER_SEC,
    _finalize,
    read_stream_quotes,
)


def _row(symbol="SPY", age_sec=10):
    return {
        "symbol": symbol,
        "price": 740.0,
        "prev_close": 737.76,
        "change_pct": 0.3,
        "quote_time": datetime.now(timezone.utc) - timedelta(seconds=age_sec),
    }


class TestFinalize:
    def test_fresh_quote_is_live(self):
        df = _finalize([_row(age_sec=10)], source="test")
        assert bool(df["is_live"].iloc[0]) is True

    def test_old_quote_is_not_live(self):
        df = _finalize([_row(age_sec=STALE_AFTER_SEC + 60)], source="test")
        assert bool(df["is_live"].iloc[0]) is False

    def test_source_stamped(self):
        df = _finalize([_row()], source="websocket")
        assert df["source"].iloc[0] == "websocket"

    def test_sorted_by_symbol(self):
        df = _finalize([_row("QQQ"), _row("AAPL")], source="test")
        assert df["symbol"].tolist() == ["AAPL", "QQQ"]

    def test_empty_rows_give_empty_frame_with_schema(self):
        df = _finalize([], source="test")
        assert df.empty and "is_live" in df.columns


class TestReadStreamQuotes:
    def test_missing_file_returns_empty(self, tmp_path):
        df = read_stream_quotes(tmp_path / "nope.json")
        assert df.empty

    def test_corrupt_file_returns_empty(self, tmp_path):
        p = tmp_path / "quotes.json"
        p.write_text("{not json")
        assert read_stream_quotes(p).empty

    def test_reads_quotes_and_marks_liveness(self, tmp_path):
        p = tmp_path / "quotes.json"
        p.write_text(json.dumps({
            "updated_at_utc": "2026-06-12T16:00:00+00:00",
            "quotes": {
                "SPY": {"price": 740.0, "time": time.time() - 5,
                        "change_pct": 0.3, "prev_close": 737.76},
                "QQQ": {"price": 717.0, "time": time.time() - 9999,
                        "change_pct": None, "prev_close": None},
            },
        }))
        df = read_stream_quotes(p)
        assert len(df) == 2
        spy = df[df["symbol"] == "SPY"].iloc[0]
        qqq = df[df["symbol"] == "QQQ"].iloc[0]
        assert bool(spy["is_live"]) is True
        assert bool(qqq["is_live"]) is False
        assert spy["source"] == "yahoo websocket stream"

    def test_symbol_filter(self, tmp_path):
        p = tmp_path / "quotes.json"
        p.write_text(json.dumps({"quotes": {
            "SPY": {"price": 1.0, "time": time.time()},
            "QQQ": {"price": 1.0, "time": time.time()},
        }}))
        df = read_stream_quotes(p, symbols=["SPY"])
        assert df["symbol"].tolist() == ["SPY"]
