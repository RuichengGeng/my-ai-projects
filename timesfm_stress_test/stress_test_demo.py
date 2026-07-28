"""
Oil Portfolio Stress-Testing Demo with TimesFM 2.5 (Google Research)
====================================================================

Pipeline
--------
1. Simulate 3y of daily WTI / Brent / Gasoil prices (correlated, vol regimes,
   one embedded supply-shock event for realism). Replace with real data later.
2. TimesFM 2.5 zero-shot baseline forecast: 90d point + q10..q90 cone per asset.
3. Stress scenarios (exogenous overlays calibrated to historical analogs):
     - war_escalation    : supply disruption impulse (2022-style)
     - opec_cut          : gradual, sustained support (2023-style cuts)
     - demand_recession  : 2008/2020-style demand collapse
   Plus a pure-TimesFM "context injection" experiment: prepend hypothetical
   shock days to the context and let the model continue the shocked regime.
4. Portfolio P&L under each scenario + baseline P10/P90 band.

Key honesty note: TimesFM is a univariate *pattern* forecaster. It has no
concept of geopolitics or OPEC. Scenarios are therefore exogenous shocks
applied on top of the model's probabilistic baseline (the auditable approach
a risk manager would use); context injection is shown as an experiment.

Run:  ../.venv-timesfm/bin/python stress_test_demo.py
"""

from dataclasses import dataclass

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

OUT = "output"
HORIZON = 90          # forecast horizon, days
CONTEXT = 756         # days of history fed to TimesFM
SEED = 42

# Portfolio: futures contracts of 1,000 bbl(-equivalent) each.
POSITIONS = {"WTI": +100, "Brent": +60, "Gasoil": -30}  # contracts
CONTRACT_SIZE = 1000                                    # bbl / contract


# --------------------------------------------------------------------------- #
# 1. Data simulation (replace with real WTI/Brent/Gasoil history later)        #
# --------------------------------------------------------------------------- #
def simulate_prices(n: int = 756, seed: int = SEED) -> pd.DataFrame:
    """Correlated daily prices via WTI log-returns + OU spread/crack ratios."""
    rng = np.random.default_rng(seed)
    dates = pd.bdate_range("2023-01-02", periods=n)

    # WTI: regime-switching vol GBM (~28% ann. base vol, 10% of days 2.2x).
    sigma = np.where(rng.random(n) < 0.10, 0.018 * 2.2, 0.018)
    wti_ret = 0.0002 + sigma * rng.standard_normal(n)
    # Embedded historical-like supply scare at ~day 450: +25% decaying over 60d.
    shock = np.zeros(n)
    t0 = 450
    shock[t0 : t0 + 60] = 0.25 * np.exp(-np.arange(60) / 25.0)
    log_wti = np.log(75.0) + np.cumsum(wti_ret) + shock

    def ou_ratio(mean_log: float, theta: float, vol: float) -> np.ndarray:
        x = np.empty(n)
        x[0] = mean_log
        for t in range(1, n):
            x[t] = x[t - 1] + theta * (mean_log - x[t - 1]) + vol * rng.standard_normal()
        return np.exp(x)

    brent = np.exp(log_wti) * ou_ratio(np.log(1.045), 0.05, 0.004)   # ~$78-79
    gasoil = np.exp(log_wti) * ou_ratio(np.log(1.42), 0.04, 0.008)   # ~$106 ($/bbl-eq)

    df = pd.DataFrame({"WTI": np.exp(log_wti), "Brent": brent, "Gasoil": gasoil}, index=dates)
    return df.round(2)


# --------------------------------------------------------------------------- #
# 2. TimesFM baseline forecast                                                 #
# --------------------------------------------------------------------------- #
def load_model():
    import timesfm

    model = timesfm.TimesFM_2p5_200M_torch.from_pretrained(
        "google/timesfm-2.5-200m-pytorch"
    )
    model.compile(
        timesfm.ForecastConfig(
            max_context=1024,
            max_horizon=256,
            normalize_inputs=True,
            use_continuous_quantile_head=True,
            force_flip_invariance=True,
            infer_is_positive=True,      # prices are positive
            fix_quantile_crossing=True,
        )
    )
    return model


def forecast_baseline(model, prices: pd.DataFrame):
    """Return point (A,H), q10 (A,H), q90 (A,H) arrays for all assets."""
    inputs = [prices[c].values[-CONTEXT:] for c in prices.columns]
    point, quant = model.forecast(horizon=HORIZON, inputs=inputs)
    q10, q90 = quant[:, :, 1], quant[:, :, 9]  # slot0=mean, slots1..9=q10..q90
    return point, q10, q90


def forecast_with_context_injection(model, prices: pd.DataFrame, shock_days: int = 10):
    """Experiment: append hypothetical 'war breaks out' days to the context,
    then let TimesFM continue the shocked regime on its own."""
    rng = np.random.default_rng(7)
    injected = []
    for c in prices.columns:
        hist = prices[c].values[-CONTEXT:]
        daily_ret = {"WTI": 0.035, "Brent": 0.035, "Gasoil": 0.045}[c]
        ext = hist[-1] * np.cumprod(1 + daily_ret + 0.025 * rng.standard_normal(shock_days))
        injected.append(np.concatenate([hist, ext]))
    point, quant = model.forecast(horizon=HORIZON, inputs=injected)
    return point, quant[:, :, 1], quant[:, :, 9]


# --------------------------------------------------------------------------- #
# 3. Scenario overlays (multiplicative shock path + vol multiplier on cone)    #
# --------------------------------------------------------------------------- #
@dataclass
class Scenario:
    name: str
    # shock multipliers per asset as functions of day t (0..HORIZON-1)
    shock: dict
    vol_mult: float = 1.0
    desc: str = ""


def ramp(target: float, days: int, fade_to: float | None = None):
    """Linear ramp to `target` over `days`, then hold (optionally fade to `fade_to`)."""
    def f(t):
        r = np.minimum(t / max(days, 1), 1.0) * target
        if fade_to is not None:
            after = np.maximum(t - days, 0) / max(HORIZON - days, 1)
            r = np.where(t >= days, target + (fade_to - target) * after, r)
        return r
    return f


SCENARIOS = [
    Scenario(
        "war_escalation",
        {"WTI": ramp(0.35, 10, fade_to=0.28), "Brent": ramp(0.35, 10, fade_to=0.28),
         "Gasoil": ramp(0.45, 10, fade_to=0.35)},   # products spike harder
        vol_mult=1.8,
        desc="Supply disruption: +35% crude impulse in 10d, partial fade; 1.8x vol",
    ),
    Scenario(
        "opec_cut",
        {"WTI": ramp(0.12, 30), "Brent": ramp(0.12, 30), "Gasoil": ramp(0.10, 30)},
        vol_mult=1.25,
        desc="OPEC cut: +12% over 30d, sustained; 1.25x vol",
    ),
    Scenario(
        "demand_recession",
        {"WTI": ramp(-0.22, 75), "Brent": ramp(-0.22, 75), "Gasoil": ramp(-0.25, 75)},
        vol_mult=1.5,
        desc="Demand collapse: -22% crude over 75d, cracks compress; 1.5x vol",
    ),
]


def apply_scenario(base_point, base_q10, base_q90, sc: Scenario, assets):
    t = np.arange(HORIZON)
    point = np.stack([base_point[i] * (1 + sc.shock[a](t)) for i, a in enumerate(assets)])
    # shift cone centre by shock, scale spread by vol multiplier
    q10 = point - sc.vol_mult * (base_point - base_q10)
    q90 = point + sc.vol_mult * (base_q90 - base_point)
    return point, q10, q90


# --------------------------------------------------------------------------- #
# 4. Portfolio P&L                                                             #
# --------------------------------------------------------------------------- #
def portfolio_pnl(price_paths: np.ndarray, assets, entry: np.ndarray) -> np.ndarray:
    """price_paths: (A, H); entry: last actual price per asset (A,). P&L path in $."""
    pnl = np.zeros(price_paths.shape[1])
    for i, a in enumerate(assets):
        pnl += POSITIONS[a] * CONTRACT_SIZE * (price_paths[i] - entry[i])
    return pnl


# --------------------------------------------------------------------------- #
# Main                                                                         #
# --------------------------------------------------------------------------- #
def main():
    import os
    import time

    os.makedirs(OUT, exist_ok=True)
    prices = simulate_prices()
    prices.to_csv(f"{OUT}/simulated_prices.csv")
    assets = list(prices.columns)
    fdates = pd.bdate_range(prices.index[-1] + pd.Timedelta(days=1), periods=HORIZON)
    print(f"Simulated {len(prices)} days through {prices.index[-1].date()}; "
          f"last prices: {prices.iloc[-1].round(1).to_dict()}")

    # --- history plot ---
    fig, axes = plt.subplots(3, 1, figsize=(11, 8), sharex=True)
    for ax, c in zip(axes, assets):
        ax.plot(prices.index, prices[c], lw=0.9)
        ax.set_ylabel(f"{c} $/bbl-eq")
        ax.grid(alpha=0.3)
    axes[0].set_title("Simulated price history (demo data — replace with real feeds)")
    fig.tight_layout()
    fig.savefig(f"{OUT}/01_simulated_history.png", dpi=110)
    plt.close(fig)

    # --- TimesFM baseline ---
    t0 = time.time()
    model = load_model()
    base_point, base_q10, base_q90 = forecast_baseline(model, prices)
    print(f"TimesFM baseline forecast: {time.time() - t0:.1f}s (incl. model load)")

    fig, axes = plt.subplots(3, 1, figsize=(11, 9), sharex=True)
    for i, (ax, c) in enumerate(zip(axes, assets)):
        ax.plot(prices.index[-250:], prices[c].values[-250:], lw=1.0, label="history")
        ax.plot(fdates, base_point[i], color="tab:red", lw=1.4, label="TimesFM point")
        ax.fill_between(fdates, base_q10[i], base_q90[i], color="tab:red", alpha=0.2,
                        label="q10–q90 cone")
        ax.set_ylabel(f"{c} $")
        ax.grid(alpha=0.3)
        ax.legend(loc="upper left", fontsize=8)
    axes[0].set_title(f"TimesFM 2.5 zero-shot baseline, {HORIZON}d horizon")
    fig.tight_layout()
    fig.savefig(f"{OUT}/02_baseline_forecast.png", dpi=110)
    plt.close(fig)

    # --- context-injection experiment (war) ---
    ci_point, ci_q10, ci_q90 = forecast_with_context_injection(model, prices)

    # --- scenarios ---
    results = {}
    for sc in SCENARIOS:
        p, lo, hi = apply_scenario(base_point, base_q10, base_q90, sc, assets)
        results[sc.name] = (sc, p, lo, hi)

    # --- scenario + portfolio plot ---
    fig = plt.figure(figsize=(12, 9))
    ax1 = fig.add_subplot(2, 1, 1)
    i_wti = assets.index("WTI")
    ax1.plot(fdates, base_point[i_wti], color="black", lw=1.6, label="TimesFM baseline")
    ax1.fill_between(fdates, base_q10[i_wti], base_q90[i_wti], color="gray", alpha=0.25)
    colors = {"war_escalation": "tab:red", "opec_cut": "tab:orange",
              "demand_recession": "tab:blue"}
    for name, (sc, p, lo, hi) in results.items():
        ax1.plot(fdates, p[i_wti], color=colors[name], lw=1.5, label=name)
    ax1.plot(fdates, ci_point[i_wti], color="tab:red", lw=1.2, ls="--",
             label="war via context-injection (pure TimesFM)")
    ax1.set_title("WTI stress scenarios vs TimesFM baseline")
    ax1.set_ylabel("$/bbl")
    ax1.grid(alpha=0.3)
    ax1.legend(fontsize=8)

    ax2 = fig.add_subplot(2, 1, 2)
    entry = prices.iloc[-1].values.astype(float)  # mark P&L to last actual price
    base_pnl = portfolio_pnl(base_point, assets, entry)
    pnl_q10 = portfolio_pnl(base_q10, assets, entry)
    pnl_q90 = portfolio_pnl(base_q90, assets, entry)
    ax2.fill_between(fdates, pnl_q10 / 1e6, pnl_q90 / 1e6, color="gray", alpha=0.3,
                     label="baseline q10–q90 (all assets same-side)")
    ax2.plot(fdates, base_pnl / 1e6, color="black", lw=1.6, label="baseline point")
    rows = []
    for name, (sc, p, lo, hi) in results.items():
        pnl = portfolio_pnl(p, assets, entry)
        ax2.plot(fdates, pnl / 1e6, color=colors[name], lw=1.6, label=name)
        rows.append((name, sc.desc, pnl[-1] / 1e6, pnl.min() / 1e6))
    ax2.axhline(0, color="k", lw=0.6)
    ax2.set_title("Portfolio cumulative P&L under scenarios  "
                  "(+100 WTI, +60 Brent, -30 Gasoil contracts)")
    ax2.set_ylabel("P&L, $M")
    ax2.grid(alpha=0.3)
    ax2.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(f"{OUT}/03_scenarios_and_portfolio.png", dpi=110)
    plt.close(fig)

    # --- stress table ---
    print("\n=== Portfolio stress table (90d horizon) ===")
    print(f"{'scenario':<20} {'terminal P&L, $M':>17} {'worst path P&L, $M':>20}")
    print(f"{'baseline (TimesFM point)':<20} {base_pnl[-1]/1e6:>17.2f} {base_pnl.min()/1e6:>20.2f}")
    print(f"{'baseline q10 (P10 band)':<20} {pnl_q10[-1]/1e6:>17.2f} {pnl_q10.min()/1e6:>20.2f}")
    for name, desc, term, worst in rows:
        print(f"{name:<20} {term:>17.2f} {worst:>20.2f}   {desc}")

    print(f"\nOutputs written to ./{OUT}/: simulated_prices.csv, 01-03 png")


if __name__ == "__main__":
    main()
