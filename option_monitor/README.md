# Option Market Monitor

A self-contained pipeline that snapshots the US listed-options market from
Yahoo Finance and visualizes it in a Streamlit dashboard. The goal is to read
market dynamics from **three angles at once**: price action, underlying
volume, and option-implied sentiment (IV, skew, put/call flow, positioning).

```
option_monitor/
├── monitor.py     # data pipeline: OptionMonitor (fetch + compute all metrics)
├── run.py         # CLI runner: prints tables, saves daily CSV snapshots
├── dashboard.py   # Streamlit dashboard reading those snapshots
└── README.md      # this file
```

The only external dependency inside this repo is
`datafeed.yahoo_finance.YahooFinanceProvider` (shared Yahoo Finance client).

---

## Quick start

```bash
# 1. Take a snapshot (prints tables, writes CSVs to output/option_monitor/)
uv run python -m option_monitor.run

# 2. Launch the dashboard
uv run streamlit run option_monitor/dashboard.py
```

Run step 1 once per day (ideally **after the US close, ~4:15pm ET**, when
open interest and volume are settled). Each run saves into
`output/option_monitor/<date>/` and mirrors to `output/option_monitor/latest/`;
history accumulates so the dashboard's *Trends* tab works after 2+ days.

Configuration lives at the top of `run.py`:

| Constant         | Meaning                                                       |
|------------------|---------------------------------------------------------------|
| `SYMBOLS`        | universe to scan (`DEFAULT_UNIVERSE` = 21 ETFs + 14 equities) |
| `VALUATION_DATE` | `None` = today; only changes DTE reference + snapshot label   |
| `OUTPUT_DIR`     | snapshot root, `None` to skip CSV export                      |

Programmatic use:

```python
from option_monitor import OptionMonitor

results = OptionMonitor(symbols=["SPY", "QQQ"]).run()
results["derived"]        # per-symbol summary DataFrame
results["data_quality"]   # freshness flags
```

Tests (fully offline, all network mocked):

```bash
uv run pytest tests/option_monitor/
```

---

## Data-quality philosophy: missing stays missing

Yahoo only serves the **live** option chain — historical chains cannot be
fetched. If you run the pipeline **before the market opens** (or on a
holiday), Yahoo returns the *prior session's* prices and quotes. The pipeline
**never imputes or back-fills**:

- Values that are unavailable are written as empty/`None`, not zero.
  (E.g. pre-open option volume is *missing*, not "no flow".)
- Every snapshot includes a `data_quality.csv` flagging staleness per symbol.
- `run.py` prints a ⚠ warning summary; the dashboard shows a warning banner
  whenever the selected snapshot is stale or incomplete.

`data_quality.csv` columns:

| Column                  | Meaning                                                        |
|-------------------------|----------------------------------------------------------------|
| `price_date`            | date of the last daily price bar actually received             |
| `price_is_stale`        | `True` when `price_date` < valuation date (market not open yet)|
| `chain_last_trade_date` | most recent option trade seen in the raw chain                 |
| `chain_is_stale`        | `True` when the chain reflects the prior session               |
| `n_contracts`           | contracts surviving the quality filters                        |
| `volume_reported`       | `False` when no contract reports volume (volume metrics = missing) |

Chain quality filters (contracts dropped, never modified):
IV outside [1%, 300%] (Yahoo uses ~1e-5 as a "missing" placeholder), no live
bid **and** no traded volume (dead quotes), missing open interest.

---

## Output tables & metrics

Each run produces 9 CSVs. Every table carries a `valuation_date` column.

### `spot.csv` — price, volume, realized vol

| Column | Definition |
|---|---|
| `spot` | last daily close |
| `price_date` / `price_is_stale` | date of that close + staleness flag |
| `change_pct` | 1-day % change |
| `rel_volume` | last day's underlying volume ÷ its 20-day average (>1.5 = unusually active) |
| `rv_5d` / `rv_20d` / `rv_60d` | annualized realized volatility (%) from log returns, √252 scaling |

### `iv_term_structure.csv` — ATM implied vol per expiry

| Column | Definition |
|---|---|
| `atm_call_iv` / `atm_put_iv` | IV linearly interpolated to the spot strike, per side |
| `atm_iv` | mean of the two — the headline ATM vol for that expiry |
| `dte`, `dte_bucket` | days to expiry and bucket (`0-30d`, `31-60d`, `61-90d`, `>90d`) |

**Read:** front-end IV spiking above the back = event premium / near-term stress.

### `iv_skew.csv` — downside vs upside pricing per expiry

| Column | Definition |
|---|---|
| `otm_put_iv` | IV at strike = 95% of spot (5% OTM put) |
| `otm_call_iv` | IV at strike = 105% of spot (5% OTM call) |
| `skew` | `otm_put_iv − otm_call_iv` |

**Read:** positive skew is normal in equities (crash insurance costs more);
a front-end skew spike = hedging rush.

### `pc_overall.csv` / `pc_by_expiry.csv` / `pc_by_moneyness.csv` — put/call flow

| Column | Definition |
|---|---|
| `*_oi` | summed open interest (standing positions — slow-moving) |
| `*_vol` | summed traded volume (today's flow — fast sentiment); **missing when not reported** |
| `pc_oi_ratio` / `pc_vol_ratio` | put ÷ call; > 1 = put-heavy |

Splits: by DTE bucket, and by moneyness bucket (strike/spot):
`deep_below` <0.85, `below` 0.85–0.97, `near_atm` 0.97–1.03, `above`
1.03–1.15, `deep_above` >1.15.

**Read:** put OI piled below spot = hedges / support; call OI above spot =
upside bets / likely resistance.

### `positioning.csv` — dealer-positioning levels (near-term chain, DTE ≤ 45)

| Column | Definition |
|---|---|
| `max_pain` | strike minimizing total intrinsic payout to option holders |
| `call_wall` / `put_wall` | strike with the largest call / put OI |
| `*_dist_pct` | distance of each level from spot (%) |

**Read:** walls often act as magnets/barriers near expiry. Restricted to
DTE ≤ 45 because mixing LEAPS OI with weeklies dilutes the levels.

### `derived.csv` — one row per symbol, the dashboard's main table

| Column | Definition |
|---|---|
| `atm_iv_30d` / `atm_iv_90d` | ATM IV interpolated to a constant 30/90-day tenor |
| `term_slope` | `atm_iv_30d ÷ atm_iv_90d`; > 1 = inverted front end (stress), < 1 = normal contango |
| `skew_30d` | 5% OTM put−call IV spread at the 30-day tenor |
| `iv_rv_spread` | `atm_iv_30d×100 − rv_20d` — the variance risk premium; high = options price in more fear than the tape shows |
| `pc_vol_ratio` | overall put/call volume ratio |
| `sentiment` | composite score (below) |

**Sentiment** = negative of the mean cross-sectional z-score of
(`pc_vol_ratio`, `skew_30d`, `iv_rv_spread`) across the universe — all three
rise when downside protection is bid. Higher = more bullish *relative to
peers on that day*; requires ≥ 3 symbols.

### `data_quality.csv`

See "Data-quality philosophy" above.

---

## Dashboard tabs

| Tab | Contents |
|---|---|
| **📊 Market Overview** | headline KPIs (SPY, median IV−RV, inverted term structures, heavy-volume count), sentiment bar chart, full sortable metric table |
| **🔍 Symbol Detail** | per symbol: IV term structure vs realized vol, skew by expiry, OI by strike level, P/C ratios by tenor, key levels (max pain / walls) |
| **📈 Trends** | any metric's history across snapshot dates (needs ≥ 2 run dates) |

A ⚠ data-quality banner appears at the top whenever the selected snapshot
contains stale prices, stale chains, or unreported volume.

**How to read it together:** rising skew + rising put/call volume + falling
sentiment is a classic risk-off rotation, often *before* price confirms it.
The *change* in these series matters more than the level.

---

## Known limitations

- Yahoo serves only the live chain — no historical option data; history is
  built by running the snapshot daily.
- Pre-open runs are flagged stale and reflect the prior session (by design,
  no imputation). Prefer post-close runs.
- IV is Yahoo's own computation; deep-OTM and 0-DTE quotes are noisy even
  after the sanity filters.
- `valuation_date` should be today (or the actual capture date) — it cannot
  fetch the past.
