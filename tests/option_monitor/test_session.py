"""Tests for last_completed_session — fully offline, provider mocked."""

from datetime import datetime
from unittest.mock import MagicMock
from zoneinfo import ZoneInfo

import pandas as pd
import pytest

from option_monitor.monitor import NY_TZ, last_completed_session

ET = ZoneInfo("America/New_York")


def _provider_with_bars(dates: list[str]) -> MagicMock:
    provider = MagicMock()
    idx = pd.DatetimeIndex([pd.Timestamp(d, tz=ET) for d in dates])
    provider.get_history.return_value = pd.DataFrame(
        {"Close": [100.0] * len(dates)}, index=idx
    )
    return provider


# Friday 2026-06-12 has a bar; Thursday 2026-06-11 before it.
BARS = ["2026-06-10", "2026-06-11", "2026-06-12"]


class TestLastCompletedSession:
    def test_during_market_hours_returns_previous_bar(self):
        now = datetime(2026, 6, 12, 12, 0, tzinfo=ET)  # Friday noon
        assert last_completed_session(
            _provider_with_bars(BARS), now=now
        ).isoformat() == "2026-06-11"

    def test_after_close_returns_todays_bar(self):
        now = datetime(2026, 6, 12, 16, 30, tzinfo=ET)
        assert last_completed_session(
            _provider_with_bars(BARS), now=now
        ).isoformat() == "2026-06-12"

    def test_exactly_at_cutoff_counts_as_complete(self):
        now = datetime(2026, 6, 12, 16, 5, tzinfo=ET)
        assert last_completed_session(
            _provider_with_bars(BARS), now=now
        ).isoformat() == "2026-06-12"

    def test_weekend_returns_friday(self):
        now = datetime(2026, 6, 13, 11, 0, tzinfo=ET)  # Saturday
        assert last_completed_session(
            _provider_with_bars(BARS), now=now
        ).isoformat() == "2026-06-12"

    def test_pre_open_returns_previous_session(self):
        # Friday 8am: Yahoo has no Friday bar yet
        now = datetime(2026, 6, 12, 8, 0, tzinfo=ET)
        assert last_completed_session(
            _provider_with_bars(["2026-06-10", "2026-06-11"]), now=now
        ).isoformat() == "2026-06-11"

    def test_other_timezone_normalised_to_et(self):
        # 12:00 ET expressed as 17:00 London time
        now = datetime(2026, 6, 12, 17, 0, tzinfo=ZoneInfo("Europe/London"))
        assert last_completed_session(
            _provider_with_bars(BARS), now=now
        ).isoformat() == "2026-06-11"

    def test_single_partial_bar_raises(self):
        now = datetime(2026, 6, 12, 12, 0, tzinfo=ET)
        with pytest.raises(ValueError):
            last_completed_session(_provider_with_bars(["2026-06-12"]), now=now)

    def test_ny_tz_constant(self):
        assert str(NY_TZ) == "America/New_York"
