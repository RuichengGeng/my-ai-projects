"""
Momentum strategies from Chapters 6-7 of "Algorithmic Trading".

- Example 6.1: TU time-series momentum
- Example 6.2: Cross-sectional momentum
- Example 7.1: Opening gap strategy
- Example 7.2: Post-earnings announcement drift (PEAD)
"""

import numpy as np


# ---------------------------------------------------------------------------
# Example 6.1: Time-series momentum (TU futures)
# ---------------------------------------------------------------------------

def time_series_momentum(
    prices: np.ndarray,
    lookback: int = 252,  # 12 months
    holding_period: int = 21,  # 1 month
) -> dict:
    """Time-series momentum strategy (Example 6.1).

    Buy if 12-month return > 0, short if < 0. Hold for 1 month.
    Originally applied to TU (2-year Treasury note) futures.

    Args:
        prices: daily price series.
        lookback: lookback in days.
        holding_period: holding period in days.

    Returns:
        dict with positions, pnl, cum_pnl.
    """
    T = len(prices)
    positions = np.zeros(T)
    returns = np.zeros(T)

    t = lookback
    while t < T - holding_period:
        past_return = prices[t] / prices[t - lookback] - 1
        if past_return > 0:
            positions[t : t + holding_period] = 1
        else:
            positions[t : t + holding_period] = -1
        t += holding_period

    ret = np.diff(prices) / prices[:-1]
    pnl = positions[:-1] * ret
    cum_pnl = np.cumsum(pnl)

    return {
        "positions": positions,
        "pnl": pnl,
        "cum_pnl": cum_pnl,
        "annual_return": np.sum(pnl) / len(pnl) * 252,
        "sharpe": np.mean(pnl) / np.std(pnl) * np.sqrt(252) if np.std(pnl) > 0 else 0,
    }


# ---------------------------------------------------------------------------
# Example 6.2: Cross-sectional momentum
# ---------------------------------------------------------------------------

def cross_sectional_momentum(
    prices: np.ndarray,
    lookback: int = 252,
    holding_period: int = 21,
    top_frac: float = 0.2,
) -> dict:
    """Cross-sectional momentum strategy (Example 6.2).

    Each period: rank stocks by past return, go long top decile,
    short bottom decile. Hold for holding_period.

    Args:
        prices: T×N array of stock prices.
        lookback: lookback for momentum ranking.
        holding_period: holding period in days.
        top_frac: fraction in each leg (e.g. 0.2 = top/bottom quintile).

    Returns:
        dict with daily returns, cumulative P&L.
    """
    T, N = prices.shape
    portfolio_ret = np.zeros(T)

    t = lookback
    while t < T - holding_period:
        # Compute returns over lookback
        past_ret = prices[t] / prices[t - lookback] - 1
        valid = ~np.isnan(past_ret)

        if valid.sum() < 10:
            t += holding_period
            continue

        # Sort by past returns
        sorted_idx = np.argsort(past_ret[valid])
        n_valid = valid.sum()
        n_each = max(1, int(n_valid * top_frac))

        long_idx = sorted_idx[-n_each:]   # top performers
        short_idx = sorted_idx[:n_each]    # bottom performers

        # Equal weight in each leg
        long_ret = np.nanmean(prices[t + holding_period, valid][long_idx] / prices[t, valid][long_idx] - 1)
        short_ret = np.nanmean(prices[t + holding_period, valid][short_idx] / prices[t, valid][short_idx] - 1)

        portfolio_ret[t + holding_period] = long_ret - short_ret
        t += holding_period

    cum_ret = np.cumsum(portfolio_ret)

    return {
        "daily_returns": portfolio_ret,
        "cum_returns": cum_ret,
        "annual_return": np.sum(portfolio_ret) / T * 252,
        "sharpe": np.mean(portfolio_ret[portfolio_ret != 0]) / np.std(portfolio_ret[portfolio_ret != 0]) * np.sqrt(252)
        if np.std(portfolio_ret[portfolio_ret != 0]) > 0 else 0,
    }


# ---------------------------------------------------------------------------
# Example 7.1: Opening gap strategy (FSTX)
# ---------------------------------------------------------------------------

def opening_gap_strategy(
    open_prices: np.ndarray,
    prev_closes: np.ndarray,
    threshold: float = 0.01,
) -> dict:
    """Opening gap momentum strategy (Example 7.1).

    If stock opens significantly above (below) previous close,
    go long (short) expecting continuation of the move.

    Args:
        open_prices: T×N array of opens.
        prev_closes: T×N array of prior closes.
        threshold: gap threshold (e.g. 0.01 = 1%).

    Returns:
        dict with results.
    """
    T, N = open_prices.shape
    gap = np.zeros((T, N))
    valid_mask = ~np.isnan(prev_closes) & (prev_closes > 0)
    gap[valid_mask] = open_prices[valid_mask] / prev_closes[valid_mask] - 1

    # Entry signals
    long_signal = gap > threshold
    short_signal = gap < -threshold

    # Next-day close returns (using same array for simplicity)
    intraday_ret = np.zeros((T, N))
    valid_mask2 = ~np.isnan(open_prices) & (open_prices > 0)
    intraday_ret[valid_mask2] = prev_closes[valid_mask2] / open_prices[valid_mask2] - 1  # placeholder

    # Simpler: if gap up, expect continuation
    positions = np.zeros((T, N))
    positions[long_signal] = 1
    positions[short_signal] = -1

    ret = np.diff(open_prices, axis=0) / open_prices[:-1]
    pnl = np.nansum(positions[:-1] * ret, axis=1) / np.maximum(np.sum(~np.isnan(ret), axis=1), 1)
    cum_pnl = np.cumsum(pnl)

    return {
        "pnl": pnl,
        "cum_pnl": cum_pnl,
        "num_long_signals": np.sum(long_signal),
        "num_short_signals": np.sum(short_signal),
        "annual_return": np.nansum(pnl) / T * 252,
        "sharpe": np.nanmean(pnl) / np.nanstd(pnl) * np.sqrt(252) if np.nanstd(pnl) > 0 else 0,
    }


# ---------------------------------------------------------------------------
# Example 7.2: Post-earnings announcement drift (PEAD)
# ---------------------------------------------------------------------------

def post_earnings_drift(
    prices: np.ndarray,
    earnings_surprise: np.ndarray,  # per stock per event
    holding_period: int = 60,
    top_frac: float = 0.2,
) -> dict:
    """Post-earnings announcement drift strategy (Example 7.2).

    Stocks with positive earnings surprises tend to drift up for weeks;
    stocks with negative surprises drift down.

    Args:
        prices: T×N array of stock prices.
        earnings_surprise: T×N array of earnings surprises (0 = no event).
            Positive = beat, negative = miss.
        holding_period: days to hold after announcement.
        top_frac: fraction of stocks to trade.

    Returns:
        dict with results.
    """
    T, N = prices.shape
    portfolio_ret = np.zeros(T)

    for t in range(T):
        surprises = earnings_surprise[t, :]
        has_event = ~np.isnan(surprises) & (surprises != 0)

        if has_event.sum() < 5:
            continue

        valid_surprises = surprises[has_event]
        n_each = max(1, int(has_event.sum() * top_frac))

        sorted_idx = np.argsort(valid_surprises)
        short_idx = sorted_idx[:n_each]
        long_idx = sorted_idx[-n_each:]

        # Map back to original indices
        event_indices = np.where(has_event)[0]

        end_t = min(t + holding_period, T - 1)
        long_stocks = event_indices[long_idx]
        short_stocks = event_indices[short_idx]

        long_ret = np.nanmean(prices[end_t, long_stocks] / prices[t, long_stocks] - 1)
        short_ret = np.nanmean(prices[end_t, short_stocks] / prices[t, short_stocks] - 1)

        portfolio_ret[t] = long_ret - short_ret

    cum_ret = np.cumsum(portfolio_ret)

    return {
        "daily_returns": portfolio_ret,
        "cum_returns": cum_ret,
        "annual_return": np.sum(portfolio_ret) / T * 252,
        "sharpe": np.mean(portfolio_ret[portfolio_ret != 0]) / np.std(portfolio_ret[portfolio_ret != 0]) * np.sqrt(252)
        if np.std(portfolio_ret[portfolio_ret != 0]) > 0 else 0,
    }
