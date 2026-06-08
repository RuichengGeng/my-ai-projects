"""
Risk management strategies from Chapter 8 of "Algorithmic Trading".

- Kelly formula for optimal leverage
- Optimal capital allocation under leverage constraint (Example 8.2)
- Monte Carlo simulation for tail risk
"""

import numpy as np
from scipy.stats import norm


# ---------------------------------------------------------------------------
# Kelly formula
# ---------------------------------------------------------------------------

def kelly_leverage(
    returns: np.ndarray,
    risk_free_rate: float = 0.0,
) -> float:
    """Kelly optimal leverage fraction.

    f* = μ / σ² for Gaussian returns.
    A more conservative approach uses half-Kelly: f = f* / 2.

    Args:
        returns: array of strategy returns (not log returns).
        risk_free_rate: risk-free rate (annualized).

    Returns:
        Optimal Kelly leverage fraction.
    """
    excess = returns - risk_free_rate / 252
    mu = np.mean(excess)
    var = np.var(excess, ddof=1)
    if var == 0:
        return 0.0
    return mu / var


def kelly_with_constraints(
    returns: np.ndarray,
    max_leverage: float = 2.0,
    risk_free_rate: float = 0.0,
) -> float:
    """Kelly optimal leverage capped at max_leverage (Example 8.2).

    Args:
        returns: array of strategy returns.
        max_leverage: maximum allowed leverage.
        risk_free_rate: risk-free rate.

    Returns:
        Optimal leverage fraction, capped.
    """
    f_star = kelly_leverage(returns, risk_free_rate)
    f_star = max(0, f_star)
    return min(f_star, max_leverage)


# ---------------------------------------------------------------------------
# Monte Carlo tail risk assessment
# ---------------------------------------------------------------------------

def monte_carlo_var(
    returns: np.ndarray,
    num_simulations: int = 10000,
    horizon_days: int = 252,
    confidence: float = 0.95,
) -> dict:
    """Monte Carlo simulation for Value-at-Risk and tail risk.

    Args:
        returns: daily strategy returns.
        num_simulations: number of simulation paths.
        horizon_days: forecast horizon in days.
        confidence: VaR confidence level.

    Returns:
        dict with VaR, CVaR, max_drawdown distribution.
    """
    mu = np.mean(returns)
    sigma = np.std(returns, ddof=1)

    # Simulate paths
    simulated_ret = np.random.normal(mu, sigma, (num_simulations, horizon_days))
    cum_ret = np.cumsum(simulated_ret, axis=1)

    final_returns = cum_ret[:, -1]

    var = -np.percentile(final_returns, 100 * (1 - confidence))
    cvar = -np.mean(final_returns[final_returns <= -var])

    # Max drawdown distribution
    max_dd = np.zeros(num_simulations)
    for i in range(num_simulations):
        cum = cum_ret[i]
        peak = np.maximum.accumulate(cum)
        dd = (peak - cum)
        max_dd[i] = np.max(dd)

    return {
        "var": var,
        "cvar": cvar,
        "max_drawdown_median": np.median(max_dd),
        "max_drawdown_95": np.percentile(max_dd, 95),
        "max_drawdown_99": np.percentile(max_dd, 99),
    }


# ---------------------------------------------------------------------------
# Constant Proportion Portfolio Insurance (CPPI)
# ---------------------------------------------------------------------------

def cppi(
    returns: np.ndarray,
    floor: float = 0.9,
    multiplier: float = 3.0,
    initial_capital: float = 1.0,
) -> dict:
    """Constant Proportion Portfolio Insurance.

    Exposure = multiplier * (Capital - Floor).
    Floor grows at risk-free rate.

    Args:
        returns: daily strategy returns.
        floor: minimum portfolio value as fraction of initial.
        multiplier: CPPI multiplier.
        initial_capital: starting capital.

    Returns:
        dict with capital evolution, exposure.
    """
    T = len(returns)
    capital = np.zeros(T + 1)
    exposure = np.zeros(T)
    floor_series = np.zeros(T + 1)

    capital[0] = initial_capital
    floor_series[0] = floor * initial_capital

    for t in range(T):
        cushion = capital[t] - floor_series[t]
        exposure[t] = max(0, multiplier * cushion)
        capital[t + 1] = capital[t] + exposure[t] * returns[t]
        floor_series[t + 1] = floor_series[t]  # simplified: no rf growth

    return {
        "capital": capital,
        "exposure": exposure,
        "final_capital": capital[-1],
        "total_return": capital[-1] / initial_capital - 1,
        "max_drawdown": np.max(np.maximum.accumulate(capital[:-1]) - capital[:-1]) / initial_capital,
    }


# ---------------------------------------------------------------------------
# Stop-loss analysis
# ---------------------------------------------------------------------------

def stop_loss_backtest(
    returns: np.ndarray,
    stop_loss_pct: float = 0.05,
) -> dict:
    """Simple stop-loss: if cumulative loss from peak exceeds stop_loss_pct,
    exit position and wait for re-entry signal (e.g. positive return).

    Args:
        returns: daily strategy returns.
        stop_loss_pct: stop-loss threshold as fraction of peak.

    Returns:
        dict with stop-loss adjusted returns.
    """
    T = len(returns)
    active = np.ones(T, dtype=bool)
    peak = returns[0]
    cum = returns[0]

    for t in range(1, T):
        if active[t - 1]:
            cum += returns[t]
            if cum > peak:
                peak = cum
            dd = (peak - cum) / peak if peak > 0 else 0
            if dd > stop_loss_pct:
                active[t] = False
                cum = 0  # exit position
        else:
            # Re-entry: wait for positive return day
            if returns[t] > 0:
                active[t] = True
                peak = returns[t]
                cum = returns[t]

    adjusted_returns = returns * active
    adjusted_returns = np.roll(adjusted_returns, -1)  # align

    return {
        "adjusted_returns": adjusted_returns,
        "active": active,
        "annual_return_original": np.sum(returns) / T * 252,
        "annual_return_with_stop": np.sum(adjusted_returns[:-1]) / T * 252,
    }
