# Option Market Monitor

A self-contained system that monitors the US equity & listed-options market
from Yahoo Finance: a **batch EOD pipeline**, an optional **real-time quote
service**, and a Streamlit dashboard that displays both — always labelled
with what is real-time, what is end-of-day, and when the data is from.

## Architecture

```
                    ┌─────────────────────────┐   output/option_monitor/<session>/
  Yahoo daily bars ─┤ BATCH  run.py + monitor.py  ├─►   *.csv + meta.json
  Yahoo option API ─┤ (once per day)              │   one folder per completed session
                    └─────────────────────────┘
                    ┌─────────────────────────┐   output/option_monitor/realtime/
  Yahoo websocket ──┤ REAL-TIME  stream.py        ├─►   quotes.json (ticks)
                    │ (optional long-running)     │
                    └─────────────────────────┘
                    ┌─────────────────────────┐
  Yahoo 1m bars ────┤ REAL-TIME  realtime.py      │  (polling fallback, no extra
                    │ (used by dashboard)         │   process needed)
                    └─────────────────────────┘
                              ▼
                    dashboard.py (Streamlit): 🔴 Live tab + 📁 EOD tabs
```

```
option_monitor/
├── __init__.py    # package exports (OptionMonitor, universes)
├── monitor.py     # batch pipeline: metrics + last_completed_session()
├── run.py         # batch runner: session-labelled CSV snapshots + meta.json
├── realtime.py    # real-time quotes: stream-file reader + 1m polling
├── stream.py      # optional websocket streamer (writes quotes.json)
├── dashboard.py   # Streamlit app (live + EOD views)
└── README.md      # this file
```

Within this repo it depends only on
`datafeed.yahoo_finance.YahooFinanceProvider` (shared Yahoo Finance client);
third-party packages (yfinance, pandas, streamlit, plotly, rich) come from
the project's `pyproject.toml`.

---

## Data sources — what each one really is

| Data | Source | Mode | Vintage you get | Used by |
|---|---|---|---|---|
| Daily OHLCV bars | Yahoo daily history | **batch** | official exchange closes — final EOD for completed sessions | spot, change %, rel. volume, realized vol |
| Option chain (quotes, IV) | Yahoo options API | **batch** (live-only API) | snapshot of *now*, ~15-min delayed; **no history exists** | term structure, skew, P/C volume |
| Open interest | Yahoo options API | **batch** | always the **previous session's** settled figure (OCC settles overnight) | P/C OI, max pain, walls |
| Price ticks | Yahoo websocket (`wss://streamer.finance.yahoo.com`) | **real-time** | live ticks while `stream.py` runs | dashboard 🔴 Live tab |
| 1-minute bars | Yahoo intraday history | **real-time (polled)** | ~1-min delayed, fetched on demand | dashboard 🔴 Live tab (fallback) |

Key asymmetry to understand: **prices have history, options do not.** The
option-implied metrics can only ever be captured "as of now", which is why
the batch job should run daily — that *builds* the option history that Yahoo
doesn't provide.

---

## How to run

### 1. Batch EOD snapshot (once per day)

```bash
uv run python -m option_monitor.run
```

The runner **auto-detects the last completed US session** — you can run it
any time, the label is always correct:

| When you run | Snapshot folder (session) |
|---|---|
| during market hours | yesterday's session |
| after ~4:05pm ET | today's session |
| weekend / holiday | last trading day |

Each run writes `output/option_monitor/<session-date>/` (CSVs + `meta.json`
recording when it ran and what regime it captured) and mirrors to
`latest/`. Best practice: **run shortly after 4:05pm ET** — then EOD prices,
the closing IV surface and the day's option flow are all coherent. If you run
intraday instead, prices describe the completed session but the option chain
is captured live (Yahoo cannot serve yesterday's chain); `meta.json` flags
this (`chain_captured_live_during_newer_session`).

Open interest always lags one session regardless of run time — unavoidable.

### 2. Real-time quotes (optional, for the dashboard's Live tab)

```bash
uv run python -m option_monitor.stream     # long-running; Ctrl-C to stop
```

Subscribes to Yahoo's websocket and maintains
`output/option_monitor/realtime/quotes.json`. **Optional**: without it the
dashboard polls Yahoo's 1-minute bars instead (~1-min delay, cached 60 s).
On some networks the websocket connects but delivers no ticks — the
dashboard detects the stale file and falls back to polling automatically.

### 3. Dashboard

```bash
uv run streamlit run option_monitor/dashboard.py
```

Configuration lives at the top of `run.py`:

| Constant         | Meaning                                                       |
|------------------|---------------------------------------------------------------|
| `SYMBOLS`        | universe to scan (`DEFAULT_UNIVERSE` = 22 ETFs + 14 equities) |
| `VALUATION_DATE` | `None` = auto-detect last completed session (recommended)     |
| `OUTPUT_DIR`     | snapshot root, `None` to skip CSV export                      |

Programmatic use:

```python
from option_monitor import OptionMonitor
from option_monitor.monitor import last_completed_session
from option_monitor.realtime import RealtimeQuoteService

session = last_completed_session()                       # e.g. 2026-06-11
results = OptionMonitor(symbols=["SPY", "QQQ"],
                        valuation_date=session).run()    # batch EOD
results["derived"]        # per-symbol summary DataFrame
results["data_quality"]   # freshness flags

live = RealtimeQuoteService().fetch(["SPY", "QQQ"])      # real-time quotes
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
does not impute or back-fill:

- Values that are unavailable are written as empty/`None`, not zero.
  (E.g. pre-open option volume is *missing*, not "no flow".)
  One pragmatic exception: when *some* contracts of a chain report volume,
  the remaining unreported contracts are summed as zero — intraday, Yahoo
  genuinely omits volume for contracts that have not traded. Volume metrics
  are `None` only when the **whole** chain reports nothing.
- Every snapshot includes a `data_quality.csv` flagging staleness per symbol.
- `run.py` prints a ⚠ warning summary; the dashboard shows a warning banner
  whenever the selected snapshot is stale or incomplete.

`data_quality.csv` columns:

| Column                  | Meaning                                                        |
|-------------------------|----------------------------------------------------------------|
| `price_date`            | date of the last daily price bar actually received             |
| `price_is_stale`        | `True` when `price_date` < valuation date (pre-open, weekend or holiday) |
| `chain_last_trade_date` | most recent option trade seen in the raw chain                 |
| `chain_is_stale`        | `True` when the chain reflects the prior session               |
| `n_contracts`           | contracts surviving the quality filters                        |
| `volume_reported`       | `False` when no contract reports volume (volume metrics = missing) |
| `iv_reliable`           | `False` when Yahoo served placeholder IVs instead of market data — all IV-based metrics are reported missing |

Chain quality rules:

- **Junk IV surface** — outside market hours Yahoo sometimes wipes real IVs
  and serves a synthetic default surface: nearly every contract gets a
  power-of-two value (50%, 25%, …, 3.125%, 1.5625%). When > 50% of a
  chain's IVs match that pattern, every IV is masked to missing
  (`iv_reliable = False`): ATM IV, skew, IV−RV and term slope come out
  empty rather than absurd (e.g. "SPY 30d ATM IV = 2%").
- **Per-contract IV sanity** — IV outside [1%, 300%] (Yahoo uses ~1e-5 as a
  "missing" placeholder; stale deep-OTM quotes can show absurd IV) is masked
  to missing; the contract's OI/volume still count.
- **Dead quotes dropped** — contracts with no live bid **and** no traded
  volume, or missing open interest, are excluded entirely.

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
`deep_below` < 0.85 ≤ `below` < 0.97 ≤ `near_atm` ≤ 1.03 < `above` ≤ 1.15
< `deep_above`.

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

| Tab | Data vintage | Contents |
|---|---|---|
| **🔴 Live** | real-time (tick or ~1-min delayed; quotes > 3 min old are marked ⚪ stale) | per-symbol live price, change vs previous close, quote timestamp, source (websocket vs polling), change bar chart |
| **📊 Market Overview (EOD)** | completed session shown in the caption | headline KPIs (SPY, median IV−RV, inverted term structures, heavy-volume count), sentiment bar chart, full sortable metric table |
| **🔍 Symbol Detail (EOD)** | completed session | per symbol: IV term structure vs realized vol, skew by expiry, OI by strike level, P/C ratios by tenor, key levels (max pain / walls) |
| **📈 Trends (EOD)** | one point per completed session | any metric's history across snapshot dates (needs ≥ 2 run dates) |

Every tab carries an explicit vintage caption (🔴 REAL-TIME vs 📁 END-OF-DAY
with the session date and capture time from `meta.json`). A ⚠ data-quality
banner appears whenever the selected snapshot contains stale prices, stale
chains, unreliable IVs or unreported volume.

**How to read it together:** rising skew + rising put/call volume + falling
sentiment is a classic risk-off rotation, often *before* price confirms it.
The *change* in these series matters more than the level.

---

## Known limitations

- Yahoo serves only the live chain — no historical option data; history is
  built by running the snapshot daily.
- Real-time data covers **prices only**; there is no real-time option feed
  on Yahoo (the chain endpoint is ~15-min delayed and rate-limited).
- The websocket streamer may connect but receive no ticks on some networks /
  outside market hours; the dashboard falls back to 1-minute polling.
- Pre-open runs are flagged stale and reflect the prior session (by design,
  no imputation). Prefer post-close runs — but note open interest always
  lags one session regardless of run time (see "When to run" above).
- IV is Yahoo's own computation; deep-OTM and 0-DTE quotes are noisy even
  after the sanity filters.
- `valuation_date` should be today (or the actual capture date) — it cannot
  fetch the past.
