"""
FX & Futures mean-reversion strategies from Chapter 5 of "Algorithmic Trading".

- Example 5.1: Pair trading AUD.USD vs CAD.USD (Johansen eigenvector)
- Example 5.2: Pair trading AUD.CAD with rollover interests
- Example 5.3: Estimating spot and roll returns (Constant Returns Model)
- Example 5.4: Mean reversion of calendar spreads
"""

import numpy as np

from .statistical_tests import johansen_test, adf_test, half_life, moving_average, moving_std
from .mean_reversion import linear_mean_reversion


# ---------------------------------------------------------------------------
# Example 5.1: Pair trading AUD.USD vs CAD.USD
# ---------------------------------------------------------------------------

def currency_pair_trading_johansen(
    price1: np.ndarray,  # e.g. AUD.USD
    price2: np.ndarray,  # e.g. CAD.USD
    train_size: int = 250,
    lookback: int = 20,
) -> dict:
    """Pair trading via Johansen eigenvector (Example 5.1).

    Both price series must have the same quote currency (USD).
    Uses Johansen test to find hedge ratio, then linear mean reversion.

    Args:
        price1, price2: price series in same quote currency.
        train_size: training period for Johansen test.
        lookback: lookback for z-score computation.

    Returns:
        dict with backtest results.
    """
    # Training period
    Y_train = np.column_stack([price1[:train_size], price2[:train_size]])
    jr = johansen_test(Y_train)
    evec = jr["eigenvectors"][:, 0]  # best eigenvector

    # Full portfolio price
    portfolio = evec[0] * price1 + evec[1] * price2

    # Linear mean reversion on portfolio
    results = linear_mean_reversion(portfolio, lookback=lookback)
    results["hedge_weights"] = evec
    results["portfolio_price"] = portfolio
    return results


# ---------------------------------------------------------------------------
# Example 5.2: AUD.CAD with rollover interests
# ---------------------------------------------------------------------------

def currency_pair_with_rollover(
    prices: np.ndarray,
    daily_rates_long: np.ndarray,
    daily_rates_short: np.ndarray,
    lookback: int = None,
) -> dict:
    """Pair trading with rollover interest (Example 5.2).

    Linear mean reversion on AUD.CAD, including daily rollover
    (overnight interest rate differential).

    Args:
        prices: daily closing prices of the cross-rate (e.g. AUD.CAD).
        daily_rates_long: annualized interest rate for long (e.g. AUD).
        daily_rates_short: annualized interest rate for short (e.g. CAD).
        lookback: lookback window.

    Returns:
        dict with results including rollover-adjusted P&L.
    """
    if lookback is None:
        hl = half_life(prices)
        lookback = max(2, int(round(hl)))

    # Linear mean reversion positions
    ma = moving_average(prices, lookback)
    ms = moving_std(prices, lookback)
    ms = np.where(ms == 0, 1e-10, ms)

    zscore = (prices - ma) / ms
    mkt_val = -zscore

    # Price return component
    ret = np.diff(prices) / prices[:-1]
    price_pnl = mkt_val[:-1] * ret

    # Rollover component (daily rate differential applied to position)
    # Long currency earns interest, short pays
    rate_diff = (daily_rates_long - daily_rates_short) / 365
    rollover_pnl = mkt_val[1:] * rate_diff[1:]

    total_pnl = price_pnl + rollover_pnl

    return {
        "positions": mkt_val,
        "price_pnl": price_pnl,
        "rollover_pnl": rollover_pnl,
        "total_pnl": total_pnl,
        "cum_pnl": np.cumsum(total_pnl),
        "annual_return": np.sum(total_pnl) / len(total_pnl) * 252,
        "sharpe": np.mean(total_pnl) / np.std(total_pnl) * np.sqrt(252) if np.std(total_pnl) > 0 else 0,
    }


# ---------------------------------------------------------------------------
# Example 5.3: Spot and roll returns estimation (Constant Returns Model)
# ---------------------------------------------------------------------------

def constant_returns_model(
    spot_prices: np.ndarray,
    futures_prices: np.ndarray,
    time_to_expiry: np.ndarray,
) -> dict:
    """Estimate spot and roll returns from futures prices (Example 5.3).

    Model: F(t, T) = S(t) * exp((r + gamma) * (T - t))
    where r = risk-free + storage cost, gamma = convenience yield - r.

    log(F/S) = constant * time_to_expiry

    Roll return = change in futures price due to passage of time,
    holding spot price constant.

    Args:
        spot_prices: spot price series.
        futures_prices: futures price series.
        time_to_expiry: years to expiry for each observation.

    Returns:
        dict with estimated spot and roll returns.
    """
    valid = (spot_prices > 0) & (futures_prices > 0) & (time_to_expiry >= 0)
    spot = np.log(spot_prices[valid])
    fut = np.log(futures_prices[valid])
    tte = time_to_expiry[valid]

    # Linear regression: log(F) - log(S) = const * TTE
    X = np.column_stack([tte, np.ones(len(tte))])
    y = fut - spot
    beta = np.linalg.lstsq(X, y, rcond=None)[0]
    slope = beta[0]  # (r + gamma)

    # Roll return = -slope * Δt (negative because time to expiry decreases)
    roll_return = -slope

    # Spot return = total return - roll return
    total_ret = np.diff(fut)  # log returns approximation

    return {
        "convenience_yield_plus_storage": slope,
        "roll_return_daily": roll_return / 252,
        "roll_return_annual": roll_return,
    }


# ---------------------------------------------------------------------------
# Example 5.4: Calendar spread mean reversion
# ---------------------------------------------------------------------------

def calendar_spread_mean_reversion(
    near_prices: np.ndarray,
    far_prices: np.ndarray,
    lookback: int = None,
    log_spread: bool = True,
) -> dict:
    """Mean reversion of calendar spreads (Example 5.4).

    Calendar spread = near_month - far_month futures price.
    Or log calendar spread = log(near) - log(far).

    The roll return of the spread is mean-reverting.

    Args:
        near_prices: front-month futures prices.
        far_prices: next-month futures prices.
        lookback: lookback window.
        log_spread: if True, use log spread.

    Returns:
        dict with backtest results.
    """
    if log_spread:
        spread = np.log(near_prices) - np.log(far_prices)
    else:
        spread = near_prices - far_prices

    # Test stationarity
    adf = adf_test(spread[~np.isnan(spread)])
    hl = half_life(spread[~np.isnan(spread)])

    if lookback is None:
        lookback = max(2, int(round(hl))) if hl < np.inf else 20

    results = linear_mean_reversion(spread, lookback=lookback)
    results["adf_statistic"] = adf["statistic"]
    results["adf_pvalue"] = adf["pvalue"]
    results["half_life"] = hl
    results["spread"] = spread
    return results
