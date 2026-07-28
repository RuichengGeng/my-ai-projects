# Oil Portfolio Stress Testing with TimesFM 2.5

Demo: zero-shot probabilistic forecasting of WTI / Brent / Gasoil with Google's
[TimesFM](https://github.com/google-research/timesfm) foundation model, plus
exogenous stress-scenario overlays for an oil book.

## Effort evaluation (measured on this machine: M3 Pro, 18GB, Python 3.13)

| Step | Effort | Measured |
|---|---|---|
| Install | `uv pip install "timesfm[torch]"` (timesfm 2.0.2, torch 2.13) | ~2 min, no issues |
| Model download | `google/timesfm-2.5-200m-pytorch` from HF, public, no token needed | ~800MB, ~23s once (then cached) |
| Inference | CPU-only (MPS available but unnecessary) | **0.26s per series** (90d horizon, 512d context) |
| RAM | 200M-param model | trivial vs 18GB |
| GPU / TPU | not required | — |

**Verdict: fully viable locally.** Total setup-to-first-forecast < 5 minutes.
The bottleneck is zero; the real work is scenario design (below), not the model.

## What the demo does

1. **Simulates** 3y of correlated daily WTI/Brent/Gasoil prices (GBM + vol regimes
   + OU Brent-WTI spread and gasoil crack, one embedded supply-scare event).
   → Replace `simulate_prices()` with your real history.
2. **TimesFM 2.5 baseline**: 90d point forecast + q10–q90 cone per asset, zero-shot.
3. **Scenarios** (exogenous multiplicative overlays on the baseline, cones scaled
   by a vol multiplier):
   - `war_escalation`: +35% crude impulse in 10d, fading to +28%; gasoil +45%; 1.8× vol
   - `opec_cut`: +12% over 30d, sustained; 1.25× vol
   - `demand_recession`: −22% crude / −25% gasoil over 75d; 1.5× vol
4. **Portfolio P&L**: book of +100 WTI, +60 Brent, −30 Gasoil contracts
   (1,000 bbl-eq each), marked to last actual price.

## Key methodological finding

**TimesFM does not understand "war" or "OPEC".** It is a univariate pattern
forecaster. The demo proves this with the *context-injection* experiment
(dashed red line in `03_scenarios_and_portfolio.png`): when 10 days of +3.5%/day
"war breaks out" prices are appended to the context, TimesFM treats the spike as
a blip and forecasts a **fade back down** — the opposite of an escalation regime.

Therefore the auditable stress architecture is:

```
scenario price path  =  TimesFM baseline point  ×  (1 + exogenous shock path)
scenario cone        =  baseline q10/q90 spread ×  scenario vol multiplier
```

where shock paths are calibrated from historical analogs (2022 invasion: Brent
+~40%/3m; 2023 OPEC cuts: +~10-15%; 2008/2020 demand shocks: −30-60%).
TimesFM supplies the probabilistic baseline (trend/vol/mean-reversion learned
from your data); your risk team supplies the scenarios. 

## Results (simulated data, seed 42)

```
scenario                terminal P&L, $M   worst path P&L, $M
baseline (TimesFM)               -0.38               -0.38
baseline q10 (P10 band)          -2.43               -2.43
war_escalation                   +2.21               +0.08
opec_cut                         +0.91               +0.08
demand_recession                 -2.50               -2.50
```

## Run

```bash
uv venv ../.venv-timesfm --python 3.13          # once
uv pip install --python ../.venv-timesfm/bin/python "timesfm[torch]" matplotlib pandas
cd timesfm_stress_test
../.venv-timesfm/bin/python stress_test_demo.py
```

Outputs in `output/`: `simulated_prices.csv`, `01_simulated_history.png`,
`02_baseline_forecast.png`, `03_scenarios_and_portfolio.png`.

## Caveats / next steps

- Quantile output is fixed to mean + q10…q90 (80% band). For 95/99 tails, fit a
  parametric tail or scale the band; true tail stress comes from scenarios anyway.
- Baseline is per-asset univariate; cross-asset dependence in the cone is ignored
  (portfolio band uses the crude "all assets same-side" bound). For joint paths,
  sample correlated residuals around the point paths, or consider multivariate
  models (e.g. Chronos + copula, or TimesFM XReg with covariates).
- TimesFM 2.5 supports covariates via XReg (`pip install "timesfm[xreg]"`) —
  e.g. feed inventory data, rig counts, term structure as known-future covariates.
- LoRA fine-tuning on your own price history is possible (HF PEFT example in repo).
- Swap in real data: replace `simulate_prices()` with your WTI/Brent/Gasoil feed
  (business-day, $/bbl; gasoil $/mt ÷ 7.45 for bbl-equivalent).
