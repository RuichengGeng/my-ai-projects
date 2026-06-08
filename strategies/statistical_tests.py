"""
Statistical tests from Chapter 2 of "Algorithmic Trading" (E.P. Chan).

Tests for stationarity, cointegration, half-life, Hurst exponent, etc.
"""

import numpy as np
from numpy import linalg as la
from statsmodels.tsa.stattools import adfuller
from statsmodels.regression.linear_model import OLS


def adf_test(y: np.ndarray, maxlag: int = 1) -> dict:
    """Augmented Dickey-Fuller test for mean reversion (Example 2.1).

    Returns dict with keys: statistic, pvalue, critical_values, is_stationary.
    H0: unit root exists (non-stationary). Reject at p < 0.10.
    """
    result = adfuller(y, maxlag=maxlag, regression="c", autolag=None)
    return {
        "statistic": result[0],
        "pvalue": result[1],
        "critical_values": result[4],
        "is_stationary": result[1] < 0.10,
    }


def half_life(y: np.ndarray) -> float:
    """Compute half-life of mean reversion via OLS (Example 2.4).

    Regresses Δy(t) on y(t-1). λ = slope. Half-life = -log(2) / λ.
    A negative λ indicates mean reversion.
    """
    ylag = np.roll(y, 1)
    ylag[0] = np.nan
    delta_y = y - ylag
    delta_y = delta_y[1:]
    ylag = ylag[1:]

    X = np.column_stack([ylag, np.ones(len(ylag))])
    beta = la.lstsq(X, delta_y, rcond=None)[0]
    lam = beta[0]
    if lam >= 0:
        return np.inf  # not mean-reverting
    return -np.log(2) / lam


def hurst_exponent(ts: np.ndarray, max_lag: int = 20) -> float:
    """Generalized Hurst exponent (q=2) (Example 2.2).

    Uses log prices. H < 0.5 = mean-reverting, H = 0.5 = random walk,
    H > 0.5 = trending.
    """
    z = np.log(ts)
    lags = range(2, max_lag + 1)
    tau = [np.sqrt(np.std(np.subtract(z[lag:], z[:-lag]))) for lag in lags]
    poly = np.polyfit(np.log(lags), np.log(tau), 1)
    return poly[0]


def variance_ratio_test(y: np.ndarray, lag: int = 2) -> dict:
    """Variance Ratio test for random walk hypothesis (Example 2.3).

    H0: price series is a random walk.
    Returns dict with statistic and pvalue.
    """
    z = np.log(y)
    n = len(z)
    dz = np.diff(z, lag)
    dz1 = np.diff(z, 1)

    var_lag = np.var(dz, ddof=1)
    var_1 = np.var(dz1, ddof=1)

    vr = (var_lag / lag) / var_1

    # Asymptotic standard error (Lo-MacKinlay)
    m = lag
    se_num = 2 * (2 * m - 1) * (m - 1)
    se_den = 3 * m * n
    se = np.sqrt(se_num / se_den)
    z_stat = (vr - 1) / se

    from scipy.stats import norm

    pvalue = 2 * norm.sf(abs(z_stat))

    return {
        "variance_ratio": vr,
        "z_statistic": z_stat,
        "pvalue": pvalue,
        "is_random_walk": pvalue >= 0.10,
    }


def cadf_test(y: np.ndarray, x: np.ndarray, maxlag: int = 1) -> dict:
    """Cointegrated ADF test (Engle-Granger) — Example 2.6.

    Regresses y on x to find hedge ratio, then ADF-test the residuals.
    """
    X = np.column_stack([x, np.ones(len(x))])
    beta = la.lstsq(X, y, rcond=None)[0]
    hedge_ratio = beta[0]
    residuals = y - hedge_ratio * x
    adf = adfuller(residuals[~np.isnan(residuals)], maxlag=maxlag, regression="c", autolag=None)
    return {
        "hedge_ratio": hedge_ratio,
        "adf_statistic": adf[0],
        "pvalue": adf[1],
        "critical_values": adf[4],
        "is_cointegrated": adf[1] < 0.10,
    }


def johansen_test(Y: np.ndarray, p: int = 0, k: int = 1) -> dict:
    """Johansen cointegration test (Example 2.7).

    Args:
        Y: T×N matrix of price series.
        p: 0 = constant offset, no trend; 1 = with trend.
        k: number of lags.

    Returns:
        dict with eigenvectors, eigenvalues, trace/eigen statistics.
    """
    from statsmodels.tsa.vector_ar.vecm import coint_johansen

    result = coint_johansen(Y, det_order=p, k_ar_diff=k)

    # Compute half-life for the strongest cointegrating vector (eigenvector with max eigenvalue)
    evec = result.evec[:, 0]  # strongest portfolio
    yport = Y @ evec
    hl = half_life(yport)

    return {
        "eigenvalues": result.eig,
        "eigenvectors": result.evec,
        "trace_stat": result.lr1,
        "trace_crit": result.cvt,
        "eigen_stat": result.lr2,
        "eigen_crit": result.cvm,
        "half_life": hl,
        "best_portfolio_weights": evec,
        "portfolio_price": yport,
    }


def moving_average(x: np.ndarray, window: int) -> np.ndarray:
    """Simple moving average."""
    weights = np.ones(window) / window
    return np.convolve(x, weights, mode="same")


def moving_std(x: np.ndarray, window: int) -> np.ndarray:
    """Moving standard deviation."""
    ma = moving_average(x, window)
    ma2 = moving_average(x * x, window)
    return np.sqrt(np.maximum(ma2 - ma * ma, 0))
