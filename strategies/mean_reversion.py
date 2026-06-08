"""
Mean-reversion trading strategies from Chapters 2-3 of "Algorithmic Trading".

- Linear mean reversion (Examples 2.5, 2.8)
- Bollinger Band (Example 3.2)
- Kalman Filter dynamic hedge ratio (Example 3.3)
"""

import numpy as np

from .statistical_tests import half_life, moving_average, moving_std


# ---------------------------------------------------------------------------
# Example 2.5 / 2.8: Linear mean-reverting strategy
# ---------------------------------------------------------------------------

def linear_mean_reversion(
    prices: np.ndarray,
    lookback: int = None,
) -> dict:
    """Simple linear mean-reversion strategy (Examples 2.5, 2.8).

    Position = -(price - MA(price)) / std(price), scaled negatively to
    deviation from mean. For single-series, or for a portfolio price series.

    Args:
        prices: 1-D array of price series (or portfolio price).
        lookback: Lookback window. If None, auto-set to half-life.

    Returns:
        dict with keys: positions (market value), pnl (daily), cum_pnl.
    """
    if lookback is None:
        hl = half_life(prices)
        lookback = max(2, int(round(hl)))

    ma = moving_average(prices, lookback)
    ms = moving_std(prices, lookback)
    ms = np.where(ms == 0, 1e-10, ms)

    # Normalized deviation (z-score)
    zscore = (prices - ma) / ms

    # Market value: negative of z-score
    mkt_val = -zscore

    # Daily P&L (in percent)
    ret = np.diff(prices) / prices[:-1]
    pnl = mkt_val[:-1] * ret

    cum_pnl = np.cumsum(pnl)

    return {
        "positions": mkt_val,
        "pnl": pnl,
        "cum_pnl": cum_pnl,
        "annual_return": np.sum(pnl) / len(pnl) * 252,
        "sharpe": np.mean(pnl) / np.std(pnl) * np.sqrt(252) if np.std(pnl) > 0 else 0,
    }


# ---------------------------------------------------------------------------
# Example 3.1: Price spread, log spread, and ratio trading
# ---------------------------------------------------------------------------

def spread_trading(
    y: np.ndarray,
    x: np.ndarray,
    mode: str = "price",
    lookback: int = 20,
    hedge_lookback: int = 20,
) -> dict:
    """Trading on spread/ratio between two price series (Example 3.1).

    Args:
        y, x: price series of two instruments.
        mode: 'price', 'log', or 'ratio'.
        lookback: lookback for z-score computation.
        hedge_lookback: lookback for rolling hedge ratio.

    Returns:
        dict with results.
    """
    T = len(y)

    if mode == "price":
        # Rolling hedge ratio via OLS
        spread = np.full(T, np.nan)
        for t in range(hedge_lookback, T):
            y_win = y[t - hedge_lookback : t]
            x_win = x[t - hedge_lookback : t]
            X = np.column_stack([x_win, np.ones(hedge_lookback)])
            beta = np.linalg.lstsq(X, y_win, rcond=None)[0]
            spread[t] = y[t] - beta[0] * x[t]
    elif mode == "log":
        log_y = np.log(y)
        log_x = np.log(x)
        spread = np.full(T, np.nan)
        for t in range(hedge_lookback, T):
            y_win = log_y[t - hedge_lookback : t]
            x_win = log_x[t - hedge_lookback : t]
            X = np.column_stack([x_win, np.ones(hedge_lookback)])
            beta = np.linalg.lstsq(X, y_win, rcond=None)[0]
            spread[t] = log_y[t] - beta[0] * log_x[t]
    elif mode == "ratio":
        spread = np.full(T, np.nan)
        for t in range(hedge_lookback, T):
            spread[t] = y[t] / x[t]

    valid = ~np.isnan(spread)
    if valid.sum() < lookback:
        return {"error": "Not enough data"}

    results = linear_mean_reversion(spread[valid], lookback=lookback)
    results["mode"] = mode
    results["spread"] = spread
    return results


# ---------------------------------------------------------------------------
# Example 3.2: Bollinger Band mean reversion
# ---------------------------------------------------------------------------

def bollinger_band(
    prices: np.ndarray,
    lookback: int = 20,
    entry_z: float = 1.0,
    exit_z: float = 0.0,
) -> dict:
    """Bollinger Band mean reversion strategy (Example 3.2).

    Enter short when price > MA + entry_z*std,
    enter long  when price < MA - entry_z*std.
    Exit when price crosses MA + exit_z*std (for shorts) or
    MA - exit_z*std (for longs).

    Args:
        prices: 1-D price series (or spread).
        lookback: lookback for MA and std.
        entry_z: entry threshold in std units.
        exit_z: exit threshold in std units.

    Returns:
        dict with trades, pnl, cum_pnl.
    """
    T = len(prices)
    ma = moving_average(prices, lookback)
    ms = moving_std(prices, lookback)
    ms = np.where(ms == 0, 1e-10, ms)

    upper_entry = ma + entry_z * ms
    lower_entry = ma - entry_z * ms
    upper_exit = ma + exit_z * ms
    lower_exit = ma - exit_z * ms

    positions = np.zeros(T)
    pos = 0  # current position: 1=long, -1=short, 0=flat

    for t in range(lookback, T):
        if np.isnan(ma[t]):
            continue

        if pos == 0:
            if prices[t] > upper_entry[t]:
                pos = -1  # short
            elif prices[t] < lower_entry[t]:
                pos = 1   # long
        elif pos == 1:
            if prices[t] >= lower_exit[t]:
                pos = 0
        elif pos == -1:
            if prices[t] <= upper_exit[t]:
                pos = 0

        positions[t] = pos

    ret = np.diff(prices) / prices[:-1]
    pnl = positions[:-1] * ret
    cum_pnl = np.cumsum(pnl)

    return {
        "positions": positions,
        "pnl": pnl,
        "cum_pnl": cum_pnl,
        "num_trades": np.sum(np.diff(positions) != 0),
        "annual_return": np.sum(pnl) / len(pnl) * 252,
        "sharpe": np.mean(pnl) / np.std(pnl) * np.sqrt(252) if np.std(pnl) > 0 else 0,
    }


# ---------------------------------------------------------------------------
# Example 3.3: Kalman Filter dynamic hedge ratio
# ---------------------------------------------------------------------------

class KalmanFilterPair:
    """Kalman Filter for dynamic hedge ratio estimation (Example 3.3).

    State: [beta (hedge ratio), alpha (intercept)]
    Observation: y_t = beta_t * x_t + alpha_t + noise
    """

    def __init__(self, delta: float = 1e-4, vw: float = 0.001, ve: float = 0.001):
        self.delta = delta
        self.Vw = vw  # state noise covariance scaling
        self.Ve = ve  # measurement noise variance
        self._initialized = False

    def update(self, y: float, x: float) -> tuple[float, float, float, float]:
        """Single-step update. Returns (beta, alpha, spread, predicted_y)."""
        if not self._initialized:
            self.beta = np.array([0.0, 1.0])  # [alpha, beta]
            self.P = np.eye(2) * 1.0
            self._initialized = True

        H = np.array([1.0, x])

        # Predict
        beta_pred = self.beta
        P_pred = self.P + np.eye(2) * self.Vw

        # Update
        S = H @ P_pred @ H + self.Ve
        K = P_pred @ H / S
        innovation = y - H @ beta_pred
        self.beta = beta_pred + K * innovation
        self.P = P_pred - np.outer(K, H) * P_pred

        alpha, beta = self.beta[0], self.beta[1]
        spread = y - (beta * x + alpha) if beta != 0 else np.nan
        predicted_y = beta * x + alpha

        return beta, alpha, spread, predicted_y


def kalman_filter_strategy(
    y: np.ndarray,
    x: np.ndarray,
    lookback: int = 20,
    delta: float = 1e-4,
) -> dict:
    """Kalman filter mean-reversion strategy (Example 3.3).

    Uses Kalman filter to estimate dynamic hedge ratio between y and x,
    then applies linear mean reversion on the resulting spread.

    Args:
        y, x: price series of two instruments.
        lookback: lookback for computing z-scores on spread.
        delta: Kalman filter delta parameter.

    Returns:
        dict with results.
    """
    kf = KalmanFilterPair(delta=delta)
    T = len(y)

    spreads = np.full(T, np.nan)
    betas = np.full(T, np.nan)
    alphas = np.full(T, np.nan)

    for t in range(T):
        beta, alpha, spread, _ = kf.update(y[t], x[t])
        betas[t] = beta
        alphas[t] = alpha
        spreads[t] = spread

    valid = ~np.isnan(spreads)
    results = linear_mean_reversion(spreads[valid], lookback=lookback)
    results["hedge_ratios"] = betas
    results["alphas"] = alphas
    results["spread"] = spreads
    return results
