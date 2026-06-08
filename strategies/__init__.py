"""
Algorithmic Trading Strategies — based on "Algorithmic Trading" by Ernest P. Chan.

Modules:
    statistical_tests  — ADF, Hurst, Variance Ratio, Half-Life, CADF, Johansen
    mean_reversion     — Linear MR, Spread/Ratio, Bollinger Band, Kalman Filter
    stock_strategies   — Buy-on-Gap, Index Arb, Cross-Sectional MR, Intraday MR
    fx_futures         — Currency Pair (Johansen), Rollover, Calendar Spreads
    momentum           — Time-Series Momentum, Cross-Sectional Momentum,
                         Opening Gap, Post-Earnings Drift
    risk_mgmt          — Kelly, CPPI, Monte Carlo VaR, Stop-Loss
    backtest           — Backtesting engine with portfolio tracking & metrics
"""

from . import statistical_tests
from . import mean_reversion
from . import stock_strategies
from . import fx_futures
from . import momentum
from . import risk_mgmt
from . import backtest
