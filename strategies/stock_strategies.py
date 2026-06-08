"""
Stock & ETF mean-reversion strategies from Chapter 4 of "Algorithmic Trading".

- Example 4.1: Buy-on-Gap model
- Example 4.2: SPY vs component stocks arbitrage
- Example 4.3: Cross-sectional mean reversion (daily)
- Example 4.4: Intraday cross-sectional mean reversion
"""

import numpy as np

from .statistical_tests import johansen_test, half_life
from .mean_reversion import linear_mean_reversion


# ---------------------------------------------------------------------------
# Example 4.1: Buy-on-Gap model
# ---------------------------------------------------------------------------

def buy_on_gap(
    open_prices: np.ndarray,
    prev_closes: np.ndarray,
    holding_period: int = 1,
) -> dict:
    """Buy-on-Gap model on SPX stocks (Example 4.1).

    If a stock gaps down significantly from previous close, buy at open
    and sell after holding_period days.

    Args:
        open_prices: T×N array of daily opens.
        prev_closes: T×N array of prior-day closes.
        holding_period: days to hold (default 1).

    Returns:
        dict with daily returns and cumulative P&L.
    """
    T, N = open_prices.shape
    gap = open_prices / prev_closes - 1  # overnight returns

    # Threshold: buy bottom decile gappers
    threshold = np.percentile(gap[~np.isnan(gap)], 10)

    positions = np.zeros((T, N))
    ret = np.zeros((T, N))

    for t in range(1, T - holding_period):
        for i in range(N):
            if np.isnan(gap[t, i]):
                continue
            if gap[t, i] < threshold:
                # Enter long at open, exit after holding_period
                ret[t, i] = open_prices[t + holding_period, i] / open_prices[t, i] - 1
                positions[t, i] = 1

    daily_ret = np.nanmean(ret, axis=1)
    cum_ret = np.cumsum(daily_ret)

    return {
        "daily_returns": daily_ret,
        "cum_returns": cum_ret,
        "annual_return": np.nansum(daily_ret) / T * 252,
        "sharpe": np.nanmean(daily_ret) / np.nanstd(daily_ret) * np.sqrt(252) if np.nanstd(daily_ret) > 0 else 0,
    }


# ---------------------------------------------------------------------------
# Example 4.2: SPY vs component stocks arbitrage
# ---------------------------------------------------------------------------

def index_arbitrage(
    stock_prices: np.ndarray,
    etf_price: np.ndarray,
    train_size: int = 252,
    coint_pvalue: float = 0.10,
) -> dict:
    """SPY vs component stocks arbitrage (Example 4.2).

    1. Train period: find stocks cointegrating with ETF via Johansen test.
    2. Form equal-weight long portfolio of cointegrating stocks.
    3. Confirm portfolio cointegrates with ETF.
    4. Trade linear mean reversion on the spread.

    Args:
        stock_prices: T×N array of stock closes.
        etf_price: T×1 array of ETF closes.
        train_size: number of days for training.
        coint_pvalue: p-value threshold for cointegration.

    Returns:
        dict with backtest results.
    """
    from statsmodels.tsa.stattools import coint

    T, N = stock_prices.shape
    train_end = min(train_size, T // 2)

    # Find cointegrating stocks with ETF
    coint_stocks = []
    for i in range(N):
        stock = stock_prices[:train_end, i]
        valid = ~np.isnan(stock)
        if valid.sum() < train_end * 0.8:
            continue
        score, pvalue, _ = coint(etf_price[:train_end][valid], stock[valid])
        if pvalue < coint_pvalue:
            coint_stocks.append(i)

    if len(coint_stocks) == 0:
        return {"error": "No cointegrating stocks found"}

    # Form equally-weighted long portfolio
    portfolio = np.nanmean(stock_prices[:, coint_stocks], axis=1)

    # johansen on [portfolio, etf] to confirm and get hedge ratio
    valid = ~np.isnan(portfolio) & ~np.isnan(etf_price)
    Y = np.column_stack([portfolio[valid], etf_price[valid]])
    jr = johansen_test(Y)
    evec = jr["eigenvectors"][:, 0]  # best cointegrating vector

    # Construct spread
    spread = evec[0] * portfolio + evec[1] * etf_price

    # Trade linear mean reversion on spread
    results = linear_mean_reversion(spread)
    results["num_coint_stocks"] = len(coint_stocks)
    results["coint_stocks"] = coint_stocks
    results["portfolio_weights"] = evec
    return results


# ---------------------------------------------------------------------------
# Example 4.3: Cross-sectional mean reversion (daily)
# ---------------------------------------------------------------------------

def cross_sectional_mr(
    returns: np.ndarray,
    lag: int = 1,
) -> dict:
    """Cross-sectional linear long-short on stocks (Example 4.3, Khandani & Lo).

    W_i = -(r_i - mean(r)) / sum(|r_j - mean(r)|)
    where r_i is the daily return of stock i.

    Args:
        returns: T×N array of daily returns.
        lag: lookback for signal (1 = previous day).

    Returns:
        dict with daily returns and cumulative P&L.
    """
    T, N = returns.shape
    portfolio_ret = np.zeros(T)

    for t in range(lag + 1, T):
        r = returns[t - lag, :]
        valid = ~np.isnan(r)
        if valid.sum() < 2:
            continue

        r_mean = np.mean(r[valid])
        deviation = r[valid] - r_mean
        denom = np.sum(np.abs(deviation))
        if denom == 0:
            continue

        weights = -deviation / denom
        portfolio_ret[t] = np.dot(weights, returns[t, valid])

    cum_ret = np.cumsum(portfolio_ret)

    return {
        "daily_returns": portfolio_ret,
        "cum_returns": cum_ret,
        "annual_return": np.sum(portfolio_ret) / T * 252,
        "sharpe": np.mean(portfolio_ret[lag + 1 :]) / np.std(portfolio_ret[lag + 1 :]) * np.sqrt(252)
        if np.std(portfolio_ret[lag + 1 :]) > 0 else 0,
    }


# ---------------------------------------------------------------------------
# Example 4.4: Intraday cross-sectional mean reversion
# ---------------------------------------------------------------------------

def intraday_cross_sectional_mr(
    open_prices: np.ndarray,
    prev_closes: np.ndarray,
    close_prices: np.ndarray,
) -> dict:
    """Intraday cross-sectional mean reversion (Example 4.4).

    Uses overnight return (open/prev_close) to determine weights at open.
    All positions liquidated at close.

    Args:
        open_prices: T×N daily opens.
        prev_closes: T×N prior-day closes.
        close_prices: T×N daily closes.

    Returns:
        dict with daily returns.
    """
    T, N = open_prices.shape
    overnight_ret = np.zeros((T, N))
    valid_mask = ~np.isnan(prev_closes) & (prev_closes > 0)
    overnight_ret[valid_mask] = open_prices[valid_mask] / prev_closes[valid_mask] - 1

    intraday_ret = np.zeros((T, N))
    valid_mask2 = ~np.isnan(open_prices) & (open_prices > 0)
    intraday_ret[valid_mask2] = close_prices[valid_mask2] / open_prices[valid_mask2] - 1

    portfolio_ret = np.zeros(T)

    for t in range(1, T):
        r = overnight_ret[t, :]
        valid = ~np.isnan(r)
        if valid.sum() < 2:
            continue

        r_mean = np.mean(r[valid])
        deviation = r[valid] - r_mean
        denom = np.sum(np.abs(deviation))
        if denom == 0:
            continue

        weights = -deviation / denom
        portfolio_ret[t] = np.dot(weights, intraday_ret[t, valid])

    cum_ret = np.cumsum(portfolio_ret)

    return {
        "daily_returns": portfolio_ret,
        "cum_returns": cum_ret,
        "annual_return": np.sum(portfolio_ret) / T * 252,
        "sharpe": np.mean(portfolio_ret[1:]) / np.std(portfolio_ret[1:]) * np.sqrt(252)
        if np.std(portfolio_ret[1:]) > 0 else 0,
    }
