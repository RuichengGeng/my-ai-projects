"""
Tests for OptionMonitor.

All network calls are mocked — the suite runs fully offline.

Test data (SPOT = 480.0, two expiries):
  Strikes : [400, 440, 460, 480, 500, 520, 560]
  Moneyness buckets assigned:
    400 / 480 = 0.833  → deep_below
    440 / 480 = 0.917  → below
    460 / 480 = 0.958  → below
    480 / 480 = 1.000  → near_atm
    500 / 480 = 1.042  → above
    520 / 480 = 1.083  → above
    560 / 480 = 1.167  → deep_above
"""

from datetime import date, timedelta
from unittest.mock import MagicMock, patch

import numpy as np
import pandas as pd
import pytest

from datafeed.option_monitor import (
    DEFAULT_UNIVERSE,
    EQUITY_UNIVERSE,
    ETF_UNIVERSE,
    OptionMonitor,
)

# ── Shared test constants ─────────────────────────────────────────────────────

SPOT = 480.0
STRIKES = [400.0, 440.0, 460.0, 480.0, 500.0, 520.0, 560.0]

# DTE values chosen so each falls in a different expiry bucket
_TODAY = date.today()
EXPIRY_NEAR = (_TODAY + timedelta(days=15)).strftime("%Y-%m-%d")   # bucket 0-30d
EXPIRY_MID  = (_TODAY + timedelta(days=45)).strftime("%Y-%m-%d")   # bucket 31-60d

# Per-expiry call OI per strike (index matches STRIKES)
CALL_OI = [100, 200, 300, 1000, 400, 300, 100]
PUT_OI  = [50,  150, 250,  800, 1200, 250, 80]

# Per-expiry volumes
CALL_VOL = [10, 20, 30, 100, 40, 30, 10]
PUT_VOL  = [5,  15, 25,  80, 120, 25,  8]

# IV per strike (puts have a slight upside skew to produce positive skew metric)
CALL_IV = [0.35, 0.30, 0.25, 0.20, 0.22, 0.25, 0.30]
PUT_IV  = [0.40, 0.32, 0.26, 0.20, 0.22, 0.25, 0.30]


# ── Fixtures ──────────────────────────────────────────────────────────────────

def _hist_df(n: int = 70) -> pd.DataFrame:
    idx = pd.bdate_range(end=pd.Timestamp.today(), periods=n)
    rng = np.random.default_rng(42)
    prices = 480 + np.cumsum(rng.standard_normal(n) * 2)
    prices = np.abs(prices) + 1
    prices[-1] = SPOT  # pin last close so moneyness buckets are deterministic
    return pd.DataFrame(
        {"Open": prices, "High": prices + 1, "Low": prices - 1,
         "Close": prices, "Volume": 1_000_000},
        index=idx,
    )


def _calls_df() -> pd.DataFrame:
    return pd.DataFrame({
        "strike": STRIKES,
        "bid":    [iv * 10 - 0.5 for iv in CALL_IV],
        "ask":    [iv * 10 + 0.5 for iv in CALL_IV],
        "lastPrice": [iv * 10 for iv in CALL_IV],
        "volume": CALL_VOL,
        "openInterest": CALL_OI,
        "impliedVolatility": CALL_IV,
        "inTheMoney": [s < SPOT for s in STRIKES],
    })


def _puts_df() -> pd.DataFrame:
    return pd.DataFrame({
        "strike": STRIKES,
        "bid":    [iv * 10 - 0.5 for iv in PUT_IV],
        "ask":    [iv * 10 + 0.5 for iv in PUT_IV],
        "lastPrice": [iv * 10 for iv in PUT_IV],
        "volume": PUT_VOL,
        "openInterest": PUT_OI,
        "impliedVolatility": PUT_IV,
        "inTheMoney": [s > SPOT for s in STRIKES],
    })


@pytest.fixture()
def mock_provider():
    """Patch YahooFinanceProvider inside option_monitor and yield mock instance."""
    with patch("datafeed.option_monitor.YahooFinanceProvider") as MockClass:
        inst = MockClass.return_value
        inst.get_history.return_value = _hist_df()
        inst.get_option_expirations.return_value = (EXPIRY_NEAR, EXPIRY_MID)
        inst.get_option_chain.return_value = (_calls_df(), _puts_df())
        yield inst


@pytest.fixture()
def monitor(mock_provider) -> OptionMonitor:
    # Pass valuation_date explicitly so DTE calculations are deterministic
    # regardless of the container's system clock.
    return OptionMonitor(symbols=["QQQ"], valuation_date=_TODAY)


@pytest.fixture()
def result(monitor) -> dict[str, pd.DataFrame]:
    return monitor.run()


# ── Universe constants ────────────────────────────────────────────────────────

class TestUniverse:
    def test_etf_universe_non_empty(self):
        assert len(ETF_UNIVERSE) > 0

    def test_equity_universe_non_empty(self):
        assert len(EQUITY_UNIVERSE) > 0

    def test_default_universe_is_union(self):
        assert set(DEFAULT_UNIVERSE) == set(ETF_UNIVERSE) | set(EQUITY_UNIVERSE)

    def test_mag7_in_equity_universe(self):
        for sym in ["AAPL", "MSFT", "GOOGL", "AMZN", "META", "NVDA", "TSLA"]:
            assert sym in EQUITY_UNIVERSE

    def test_broad_etfs_in_etf_universe(self):
        for sym in ["SPY", "QQQ", "IWM", "DIA"]:
            assert sym in ETF_UNIVERSE


# ── run() output shape ────────────────────────────────────────────────────────

class TestRunOutputShape:
    EXPECTED_KEYS = {
        "spot", "iv_term_structure", "iv_skew",
        "pc_overall", "pc_by_expiry", "pc_by_moneyness", "positioning",
    }

    def test_returns_all_dataframe_keys(self, result):
        assert self.EXPECTED_KEYS == set(result.keys())

    def test_all_values_are_dataframes(self, result):
        for key, val in result.items():
            assert isinstance(val, pd.DataFrame), f"'{key}' is not a DataFrame"

    def test_all_dataframes_have_valuation_date_column(self, result):
        for key, df in result.items():
            assert "valuation_date" in df.columns, f"'{key}' missing valuation_date"

    def test_spot_columns(self, result):
        expected = {"symbol", "spot", "change_pct", "rv_5d", "rv_20d", "rv_60d",
                    "valuation_date"}
        assert expected.issubset(result["spot"].columns)

    def test_iv_term_structure_columns(self, result):
        expected = {"symbol", "expiry", "dte", "dte_bucket",
                    "atm_call_iv", "atm_put_iv", "atm_iv", "valuation_date"}
        assert expected.issubset(result["iv_term_structure"].columns)

    def test_iv_skew_columns(self, result):
        expected = {"symbol", "expiry", "dte", "dte_bucket",
                    "otm_put_iv", "otm_call_iv", "skew", "valuation_date"}
        assert expected.issubset(result["iv_skew"].columns)

    def test_pc_overall_columns(self, result):
        expected = {"symbol", "total_call_oi", "total_put_oi", "pc_oi_ratio",
                    "total_call_vol", "total_put_vol", "pc_vol_ratio",
                    "valuation_date"}
        assert expected.issubset(result["pc_overall"].columns)

    def test_pc_by_expiry_columns(self, result):
        expected = {"symbol", "dte_bucket", "call_oi", "put_oi",
                    "pc_oi_ratio", "call_vol", "put_vol", "pc_vol_ratio",
                    "valuation_date"}
        assert expected.issubset(result["pc_by_expiry"].columns)

    def test_pc_by_moneyness_columns(self, result):
        expected = {"symbol", "moneyness_bucket", "call_oi", "put_oi",
                    "pc_oi_ratio", "call_vol", "put_vol", "pc_vol_ratio",
                    "valuation_date"}
        assert expected.issubset(result["pc_by_moneyness"].columns)

    def test_positioning_columns(self, result):
        expected = {"symbol", "spot", "max_pain", "max_pain_dist_pct",
                    "call_wall", "call_wall_dist_pct",
                    "put_wall", "put_wall_dist_pct", "valuation_date"}
        assert expected.issubset(result["positioning"].columns)

    def test_one_row_per_symbol_in_spot(self, result):
        assert len(result["spot"]) == 1

    def test_one_row_per_symbol_in_pc_overall(self, result):
        assert len(result["pc_overall"]) == 1

    def test_one_row_per_symbol_in_positioning(self, result):
        assert len(result["positioning"]) == 1

    def test_iv_term_structure_has_two_rows(self, result):
        # one row per (symbol, expiry) — we have 2 expiries
        assert len(result["iv_term_structure"]) == 2

    def test_pc_by_expiry_has_two_buckets(self, result):
        assert len(result["pc_by_expiry"]) == 2


# ── Spot metrics ──────────────────────────────────────────────────────────────

class TestSpotMetrics:
    def test_spot_price_is_positive(self, result):
        assert result["spot"]["spot"].iloc[0] > 0

    def test_rv_5d_is_positive(self, result):
        assert result["spot"]["rv_5d"].iloc[0] > 0

    def test_rv_20d_is_positive(self, result):
        assert result["spot"]["rv_20d"].iloc[0] > 0

    def test_rv_60d_is_positive(self, result):
        assert result["spot"]["rv_60d"].iloc[0] > 0

    def test_rv_values_are_annualised_percentages(self, result):
        # Annualised vol in % should be a single-to-double digit number
        rv = result["spot"]["rv_20d"].iloc[0]
        assert 1 < rv < 500

    def test_symbol_correct(self, result):
        assert result["spot"]["symbol"].iloc[0] == "QQQ"


# ── DTE bucketing ─────────────────────────────────────────────────────────────

class TestDteBucket:
    @pytest.mark.parametrize("dte,expected", [
        (0,  "0-30d"),
        (15, "0-30d"),
        (30, "0-30d"),
        (31, "31-60d"),
        (60, "31-60d"),
        (61, "61-90d"),
        (90, "61-90d"),
        (91, ">90d"),
        (365, ">90d"),
    ])
    def test_boundary_values(self, dte, expected):
        assert OptionMonitor._dte_bucket(dte) == expected

    def test_near_expiry_in_correct_bucket(self, result):
        df = result["pc_by_expiry"]
        assert "0-30d" in df["dte_bucket"].values

    def test_mid_expiry_in_correct_bucket(self, result):
        df = result["pc_by_expiry"]
        assert "31-60d" in df["dte_bucket"].values


# ── Moneyness bucketing ───────────────────────────────────────────────────────

class TestMoneynessBucket:
    @pytest.mark.parametrize("moneyness,expected", [
        (0.70,  "deep_below"),
        (0.84,  "deep_below"),
        (0.85,  "below"),
        (0.96,  "below"),
        (0.97,  "near_atm"),
        (1.00,  "near_atm"),
        (1.03,  "near_atm"),
        (1.031, "above"),
        (1.15,  "above"),
        (1.151, "deep_above"),
        (1.50,  "deep_above"),
    ])
    def test_boundary_values(self, moneyness, expected):
        assert OptionMonitor._moneyness_bucket(moneyness) == expected

    def test_all_five_buckets_present(self, result):
        buckets = set(result["pc_by_moneyness"]["moneyness_bucket"])
        assert buckets == {"deep_below", "below", "near_atm", "above", "deep_above"}


# ── P/C ratios — overall ──────────────────────────────────────────────────────

class TestPCOverall:
    # With 2 expiries, total OI = per-expiry × 2
    TOTAL_CALL_OI = sum(CALL_OI) * 2   # 2400 * 2 = 4800
    TOTAL_PUT_OI  = sum(PUT_OI)  * 2   # 2780 * 2  (adjust below)
    TOTAL_CALL_VOL = sum(CALL_VOL) * 2
    TOTAL_PUT_VOL  = sum(PUT_VOL)  * 2

    def test_total_call_oi(self, result):
        assert result["pc_overall"]["total_call_oi"].iloc[0] == sum(CALL_OI) * 2

    def test_total_put_oi(self, result):
        assert result["pc_overall"]["total_put_oi"].iloc[0] == sum(PUT_OI) * 2

    def test_pc_oi_ratio_matches_totals(self, result):
        row = result["pc_overall"].iloc[0]
        expected = round(row["total_put_oi"] / row["total_call_oi"], 4)
        assert row["pc_oi_ratio"] == pytest.approx(expected, rel=1e-4)

    def test_pc_vol_ratio_matches_totals(self, result):
        row = result["pc_overall"].iloc[0]
        expected = round(row["total_put_vol"] / row["total_call_vol"], 4)
        assert row["pc_vol_ratio"] == pytest.approx(expected, rel=1e-4)

    def test_total_call_vol(self, result):
        assert result["pc_overall"]["total_call_vol"].iloc[0] == sum(CALL_VOL) * 2

    def test_total_put_vol(self, result):
        assert result["pc_overall"]["total_put_vol"].iloc[0] == sum(PUT_VOL) * 2


# ── P/C ratios — by expiry bucket ────────────────────────────────────────────

class TestPCByExpiry:
    def test_two_buckets_present(self, result):
        assert len(result["pc_by_expiry"]) == 2

    def test_buckets_are_0_30d_and_31_60d(self, result):
        buckets = set(result["pc_by_expiry"]["dte_bucket"])
        assert buckets == {"0-30d", "31-60d"}

    def test_each_bucket_has_correct_call_oi(self, result):
        for _, row in result["pc_by_expiry"].iterrows():
            # Each expiry has sum(CALL_OI) call contracts
            assert row["call_oi"] == sum(CALL_OI)

    def test_each_bucket_has_correct_put_oi(self, result):
        for _, row in result["pc_by_expiry"].iterrows():
            assert row["put_oi"] == sum(PUT_OI)

    def test_pc_oi_ratio_per_bucket(self, result):
        expected = round(sum(PUT_OI) / sum(CALL_OI), 4)
        for _, row in result["pc_by_expiry"].iterrows():
            assert row["pc_oi_ratio"] == pytest.approx(expected, rel=1e-4)

    def test_pc_vol_ratio_per_bucket(self, result):
        expected = round(sum(PUT_VOL) / sum(CALL_VOL), 4)
        for _, row in result["pc_by_expiry"].iterrows():
            assert row["pc_vol_ratio"] == pytest.approx(expected, rel=1e-4)


# ── P/C ratios — by moneyness bucket ─────────────────────────────────────────

class TestPCByMoneyness:
    # Expected OI per moneyness bucket (combined across 2 expiries)
    # deep_below → strike 400
    # below      → strikes 440, 460
    # near_atm   → strike 480
    # above      → strikes 500, 520
    # deep_above → strike 560

    def _bucket_oi(self, oi_list: list, strike_idx: list) -> int:
        return sum(oi_list[i] for i in strike_idx) * 2

    def test_deep_below_call_oi(self, result):
        row = result["pc_by_moneyness"].query("moneyness_bucket == 'deep_below'").iloc[0]
        # strike 400 is index 0
        assert row["call_oi"] == self._bucket_oi(CALL_OI, [0])

    def test_below_call_oi(self, result):
        row = result["pc_by_moneyness"].query("moneyness_bucket == 'below'").iloc[0]
        # strikes 440 (idx 1) and 460 (idx 2)
        assert row["call_oi"] == self._bucket_oi(CALL_OI, [1, 2])

    def test_near_atm_put_oi(self, result):
        row = result["pc_by_moneyness"].query("moneyness_bucket == 'near_atm'").iloc[0]
        # strike 480 is index 3
        assert row["put_oi"] == self._bucket_oi(PUT_OI, [3])

    def test_above_pc_ratio(self, result):
        row = result["pc_by_moneyness"].query("moneyness_bucket == 'above'").iloc[0]
        # strikes 500 (idx 4) and 520 (idx 5)
        call_oi = self._bucket_oi(CALL_OI, [4, 5])
        put_oi  = self._bucket_oi(PUT_OI,  [4, 5])
        assert row["pc_oi_ratio"] == pytest.approx(put_oi / call_oi, rel=1e-4)

    def test_deep_above_pc_ratio(self, result):
        row = result["pc_by_moneyness"].query("moneyness_bucket == 'deep_above'").iloc[0]
        # strike 560 is index 6
        call_oi = self._bucket_oi(CALL_OI, [6])
        put_oi  = self._bucket_oi(PUT_OI,  [6])
        assert row["pc_oi_ratio"] == pytest.approx(put_oi / call_oi, rel=1e-4)


# ── IV term structure ─────────────────────────────────────────────────────────

class TestIVTermStructure:
    def test_atm_iv_is_positive(self, result):
        assert (result["iv_term_structure"]["atm_iv"] > 0).all()

    def test_atm_iv_near_20_percent(self, result):
        # ATM strike is 480; call_iv[3] = put_iv[3] = 0.20
        for atm_iv in result["iv_term_structure"]["atm_iv"]:
            assert atm_iv == pytest.approx(0.20, abs=0.01)

    def test_atm_call_iv_equals_put_iv_at_atm(self, result):
        # Both call IV and put IV at strike 480 are 0.20
        for _, row in result["iv_term_structure"].iterrows():
            assert row["atm_call_iv"] == pytest.approx(row["atm_put_iv"], abs=1e-4)

    def test_two_rows_one_per_expiry(self, result):
        assert len(result["iv_term_structure"]) == 2

    def test_dte_bucket_assigned(self, result):
        buckets = set(result["iv_term_structure"]["dte_bucket"])
        assert {"0-30d", "31-60d"} == buckets


# ── IV skew ───────────────────────────────────────────────────────────────────

class TestIVSkew:
    def test_skew_is_positive(self, result):
        # OTM put target = 480 * 0.95 = 456 → nearest strike = 460, put_iv[2] = 0.26
        # OTM call target = 480 * 1.05 = 504 → nearest strike = 500, call_iv[4] = 0.22
        # skew = 0.26 - 0.22 = 0.04
        for skew in result["iv_skew"]["skew"]:
            assert skew > 0, "Positive skew expected (puts more expensive)"

    def test_skew_value(self, result):
        expected_put_iv  = PUT_IV[2]   # strike 460, nearest to 456
        expected_call_iv = CALL_IV[4]  # strike 500, nearest to 504
        expected_skew = round(expected_put_iv - expected_call_iv, 4)
        for skew in result["iv_skew"]["skew"]:
            assert skew == pytest.approx(expected_skew, abs=1e-4)

    def test_otm_put_iv_column_present(self, result):
        assert result["iv_skew"]["otm_put_iv"].notna().all()

    def test_otm_call_iv_column_present(self, result):
        assert result["iv_skew"]["otm_call_iv"].notna().all()


# ── Max pain ──────────────────────────────────────────────────────────────────

class TestMaxPain:
    def test_simple_case(self):
        # Three strikes: 100, 110, 120
        # Call OI: 100 at 100, 50 at 110, 10 at 120
        # Put  OI: 10  at 100, 50 at 110, 100 at 120
        # Pain at K=110:
        #   call pain = (110-100)*100 = 1000
        #   put  pain = (120-110)*100 = 1000   → total 2000
        # Pain at K=100:
        #   call pain = 0
        #   put  pain = (110-100)*50 + (120-100)*100 = 2500 → total 2500
        # Pain at K=120:
        #   call pain = (120-100)*100 + (120-110)*50 = 2500 → total 2500
        # Min pain → max pain strike = 110
        call_oi = pd.Series({100: 100, 110: 50, 120: 10})
        put_oi  = pd.Series({100: 10,  110: 50, 120: 100})
        assert OptionMonitor._max_pain(call_oi, put_oi) == 110.0

    def test_single_strike_returns_that_strike(self):
        call_oi = pd.Series({200: 100})
        put_oi  = pd.Series({200: 50})
        assert OptionMonitor._max_pain(call_oi, put_oi) == 200.0

    def test_empty_series_returns_none(self):
        assert OptionMonitor._max_pain(pd.Series(dtype=float), pd.Series(dtype=float)) is None

    def test_max_pain_in_result(self, result):
        assert result["positioning"]["max_pain"].iloc[0] is not None


# ── Positioning ───────────────────────────────────────────────────────────────

class TestPositioning:
    def test_call_wall_is_strike_with_max_call_oi(self, result):
        # Max call OI per strike across both expiries → strike 480 (1000 × 2 = 2000)
        assert result["positioning"]["call_wall"].iloc[0] == pytest.approx(480.0)

    def test_put_wall_is_strike_with_max_put_oi(self, result):
        # Max put OI per strike → strike 500 (1200 × 2 = 2400)
        assert result["positioning"]["put_wall"].iloc[0] == pytest.approx(500.0)

    def test_call_wall_dist_pct_sign(self, result):
        # Call wall at 480 == spot, so distance = 0 %
        assert result["positioning"]["call_wall_dist_pct"].iloc[0] == pytest.approx(0.0)

    def test_put_wall_above_spot_has_positive_dist(self, result):
        # Put wall at 500 > spot 480, so dist > 0
        assert result["positioning"]["put_wall_dist_pct"].iloc[0] > 0

    def test_spot_stored_correctly(self, result):
        assert result["positioning"]["spot"].iloc[0] == pytest.approx(SPOT, rel=1e-3)


# ── Safe ratio helper ─────────────────────────────────────────────────────────

class TestSafeRatio:
    def test_normal_ratio(self):
        assert OptionMonitor._safe_ratio(3, 4) == pytest.approx(0.75)

    def test_zero_denominator_returns_none(self):
        assert OptionMonitor._safe_ratio(5, 0) is None

    def test_none_denominator_returns_none(self):
        assert OptionMonitor._safe_ratio(5, None) is None

    def test_zero_numerator(self):
        assert OptionMonitor._safe_ratio(0, 10) == pytest.approx(0.0)


# ── Edge cases ────────────────────────────────────────────────────────────────

class TestEdgeCases:
    def test_failed_symbol_skipped(self, mock_provider):
        mock_provider.get_option_expirations.side_effect = Exception("network error")
        monitor = OptionMonitor(symbols=["BAD"])
        result = monitor.run()
        # spot may be populated before options fail, but chain-derived DFs are empty
        assert result["pc_overall"].empty

    def test_no_expiries_produces_empty_chain_dfs(self, mock_provider):
        mock_provider.get_option_expirations.return_value = ()
        monitor = OptionMonitor(symbols=["QQQ"])
        result = monitor.run()
        assert result["iv_term_structure"].empty
        assert result["pc_overall"].empty

    def test_expired_expirations_skipped(self, mock_provider):
        past_expiry = (_TODAY - timedelta(days=1)).strftime("%Y-%m-%d")
        mock_provider.get_option_expirations.return_value = (past_expiry,)
        monitor = OptionMonitor(symbols=["QQQ"])
        result = monitor.run()
        assert result["pc_overall"].empty

    def test_zero_call_oi_gives_none_ratio(self, mock_provider):
        calls = _calls_df().copy()
        calls["openInterest"] = 0
        mock_provider.get_option_chain.return_value = (calls, _puts_df())
        monitor = OptionMonitor(symbols=["QQQ"])
        result = monitor.run()
        assert result["pc_overall"]["pc_oi_ratio"].iloc[0] is None

    def test_run_with_symbol_override(self, mock_provider):
        monitor = OptionMonitor(symbols=["SPY", "QQQ"])
        result = monitor.run(symbols=["QQQ"])
        assert len(result["spot"]) == 1
        assert result["spot"]["symbol"].iloc[0] == "QQQ"


# ── Valuation date ────────────────────────────────────────────────────────────

class TestValuationDate:
    FIXED_DATE = date(2025, 6, 9)
    FIXED_DATE_STR = "2025-06-09"

    def _expiries_for(self, vdate: date):
        """Build expiry strings relative to a given valuation date."""
        near = (vdate + timedelta(days=15)).strftime("%Y-%m-%d")
        mid  = (vdate + timedelta(days=45)).strftime("%Y-%m-%d")
        return near, mid

    def test_valuation_date_stamped_on_all_dataframes(self, mock_provider):
        vdate = self.FIXED_DATE
        near, mid = self._expiries_for(vdate)
        mock_provider.get_option_expirations.return_value = (near, mid)
        monitor = OptionMonitor(symbols=["QQQ"], valuation_date=vdate)
        result = monitor.run()
        for key, df in result.items():
            if not df.empty:
                assert df["valuation_date"].iloc[0] == self.FIXED_DATE_STR, (
                    f"'{key}' has wrong valuation_date"
                )

    def test_string_valuation_date_accepted(self, mock_provider):
        near, mid = self._expiries_for(self.FIXED_DATE)
        mock_provider.get_option_expirations.return_value = (near, mid)
        monitor = OptionMonitor(symbols=["QQQ"], valuation_date=self.FIXED_DATE_STR)
        result = monitor.run()
        assert result["spot"]["valuation_date"].iloc[0] == self.FIXED_DATE_STR

    def test_run_time_valuation_date_overrides_init(self, mock_provider):
        # Monitor created with one date, run() called with another
        init_date = date(2025, 1, 1)
        run_date  = self.FIXED_DATE
        near, mid = self._expiries_for(run_date)
        mock_provider.get_option_expirations.return_value = (near, mid)
        monitor = OptionMonitor(symbols=["QQQ"], valuation_date=init_date)
        result = monitor.run(valuation_date=run_date)
        assert result["spot"]["valuation_date"].iloc[0] == self.FIXED_DATE_STR

    def test_past_expiries_skipped_relative_to_valuation_date(self, mock_provider):
        # Expiry one day before valuation date → DTE = -1 → skipped
        vdate = self.FIXED_DATE
        past_expiry = (vdate - timedelta(days=1)).strftime("%Y-%m-%d")
        mock_provider.get_option_expirations.return_value = (past_expiry,)
        monitor = OptionMonitor(symbols=["QQQ"], valuation_date=vdate)
        result = monitor.run()
        assert result["pc_overall"].empty

    def test_future_expiries_accepted_relative_to_valuation_date(self, mock_provider):
        # Expiry 30 days after valuation date → DTE = 30 → kept
        vdate = self.FIXED_DATE
        future_expiry = (vdate + timedelta(days=30)).strftime("%Y-%m-%d")
        mock_provider.get_option_expirations.return_value = (future_expiry,)
        monitor = OptionMonitor(symbols=["QQQ"], valuation_date=vdate)
        result = monitor.run()
        assert not result["pc_overall"].empty

    def test_dte_computed_relative_to_valuation_date(self, mock_provider):
        vdate = self.FIXED_DATE
        expiry = (vdate + timedelta(days=20)).strftime("%Y-%m-%d")
        mock_provider.get_option_expirations.return_value = (expiry,)
        monitor = OptionMonitor(symbols=["QQQ"], valuation_date=vdate)
        result = monitor.run()
        assert result["iv_term_structure"]["dte"].iloc[0] == 20

    def test_default_valuation_date_is_today(self, mock_provider):
        monitor = OptionMonitor(symbols=["QQQ"])
        assert monitor.valuation_date == date.today()

    def test_empty_dataframes_also_carry_valuation_date_column(self, mock_provider):
        mock_provider.get_option_expirations.return_value = ()
        monitor = OptionMonitor(symbols=["QQQ"], valuation_date=self.FIXED_DATE)
        result = monitor.run()
        # pc_overall is empty (no chain data) but must still have the column
        assert "valuation_date" in result["pc_overall"].columns

    def test_spot_history_fetched_up_to_valuation_date(self, mock_provider):
        # _spot_snapshot must pass end=valuation_date to get_history so that
        # spot price and realized vol reflect the same date as DTE calculation.
        vdate = self.FIXED_DATE
        near, mid = self._expiries_for(vdate)
        mock_provider.get_option_expirations.return_value = (near, mid)
        monitor = OptionMonitor(symbols=["QQQ"], valuation_date=vdate)
        monitor.run()
        call_kwargs = mock_provider.get_history.call_args.kwargs
        assert call_kwargs.get("end") == self.FIXED_DATE_STR
