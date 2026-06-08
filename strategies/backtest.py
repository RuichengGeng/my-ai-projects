"""
Unified backtesting engine for algorithmic trading strategies.

Features:
- Portfolio management (cash, positions, margin)
- Trade logging with entry/exit tracking
- Performance metrics (Sharpe, max drawdown, win rate, etc.)
- Transaction cost modeling
"""

from dataclasses import dataclass, field
from typing import Optional

import numpy as np
import pandas as pd


@dataclass
class Trade:
    """A completed trade."""
    symbol: str
    entry_date: int
    exit_date: int
    entry_price: float
    exit_price: float
    side: str  # 'long' or 'short'
    size: float
    pnl: float = 0.0
    pnl_pct: float = 0.0


@dataclass
class BacktestResult:
    """Container for backtest results."""
    total_return: float = 0.0
    annual_return: float = 0.0
    annual_volatility: float = 0.0
    sharpe_ratio: float = 0.0
    sortino_ratio: float = 0.0
    max_drawdown: float = 0.0
    max_drawdown_duration: int = 0
    calmar_ratio: float = 0.0
    win_rate: float = 0.0
    profit_factor: float = 0.0
    avg_trade_pnl: float = 0.0
    total_trades: int = 0
    equity_curve: np.ndarray = field(default_factory=lambda: np.array([]))
    daily_returns: np.ndarray = field(default_factory=lambda: np.array([]))
    trades: list = field(default_factory=list)

    def summary(self) -> str:
        lines = [
            "=" * 55,
            "  BACKTEST RESULTS",
            "=" * 55,
            f"  Total Return:      {self.total_return:>12.4%}",
            f"  Annual Return:     {self.annual_return:>12.4%}",
            f"  Annual Volatility: {self.annual_volatility:>12.4%}",
            f"  Sharpe Ratio:      {self.sharpe_ratio:>12.2f}",
            f"  Sortino Ratio:     {self.sortino_ratio:>12.2f}",
            f"  Max Drawdown:      {self.max_drawdown:>12.4%}",
            f"  Max DD Duration:   {self.max_drawdown_duration:>9} days",
            f"  Calmar Ratio:      {self.calmar_ratio:>12.2f}",
            f"  Win Rate:          {self.win_rate:>12.2%}",
            f"  Profit Factor:     {self.profit_factor:>12.2f}",
            f"  Total Trades:      {self.total_trades:>12}",
            f"  Avg Trade P&L ($): {self.avg_trade_pnl:>12,.2f}",
            "=" * 55,
        ]
        return "\n".join(lines)


class BacktestEngine:
    """Event-driven backtesting engine.

    Tracks capital, positions, and trades. Supports:
    - Long/short positions
    - Commission and slippage modeling
    - Portfolio-level risk management
    """

    def __init__(
        self,
        initial_capital: float = 1_000_000.0,
        commission_per_share: float = 0.005,
        slippage_bps: float = 1.0,  # basis points
    ):
        self.initial_capital = initial_capital
        self.commission_per_share = commission_per_share
        self.slippage_bps = slippage_bps

        self.reset()

    def reset(self):
        """Reset engine state."""
        self.capital = self.initial_capital
        self.cash = self.initial_capital
        self.positions: dict[str, float] = {}  # symbol -> shares
        self.position_cost: dict[str, float] = {}  # symbol -> avg cost
        self.trades: list[Trade] = []
        self.open_trades: dict[str, Trade] = {}  # symbol -> active trade
        self.equity: list[float] = [self.initial_capital]
        self._bar = 0

    # ------------------------------------------------------------------
    # Order execution
    # ------------------------------------------------------------------

    def order(
        self,
        symbol: str,
        shares: float,
        price: float,
        date_index: Optional[int] = None,
    ) -> Optional[Trade]:
        """Place a market order. Positive shares = buy, negative = sell.

        Returns the completed Trade if this order closes a position.
        """
        if shares == 0:
            return None

        # Apply slippage
        sign = 1 if shares > 0 else -1
        executed_price = price * (1 + sign * self.slippage_bps / 10000)

        # Commission
        commission = abs(shares) * self.commission_per_share
        total_cost = shares * executed_price + commission

        if abs(total_cost) > self.cash:
            # Scale down if insufficient cash
            if total_cost > 0:
                max_shares = int(self.cash / (executed_price + self.commission_per_share))
                if max_shares <= 0:
                    return None
                shares = max_shares * sign
                total_cost = shares * executed_price + abs(shares) * self.commission_per_share

        # Update position
        old_shares = self.positions.get(symbol, 0)
        new_shares = old_shares + shares
        self.positions[symbol] = new_shares
        self.cash -= total_cost

        # Track trade
        closed_trade = None
        if symbol in self.open_trades:
            t = self.open_trades[symbol]
            # Check if position reversed or closed
            if (t.side == "long" and shares < 0) or (t.side == "short" and shares > 0):
                close_size = min(abs(shares), abs(old_shares)) * (1 if old_shares > 0 else -1)
                if close_size != 0:
                    t.exit_date = date_index or self._bar
                    t.exit_price = executed_price
                    if t.side == "long":
                        t.pnl = (executed_price - t.entry_price) * abs(close_size) - commission
                        t.pnl_pct = (executed_price / t.entry_price - 1) * 100
                    else:
                        t.pnl = (t.entry_price - executed_price) * abs(close_size) - commission
                        t.pnl_pct = (t.entry_price / executed_price - 1) * 100
                    t.size = abs(close_size)
                    self.trades.append(t)
                    closed_trade = t

                    if new_shares == 0:
                        del self.open_trades[symbol]
                    else:
                        # Partial close - update entry
                        remaining = abs(new_shares)
                        new_side = "long" if new_shares > 0 else "short"
                        self.open_trades[symbol] = Trade(
                            symbol=symbol,
                            entry_date=date_index or self._bar,
                            exit_date=0,
                            entry_price=executed_price,
                            exit_price=0,
                            side=new_side,
                            size=remaining,
                        )

        # If new position opened
        if new_shares != 0 and symbol not in self.open_trades:
            self.open_trades[symbol] = Trade(
                symbol=symbol,
                entry_date=date_index or self._bar,
                exit_date=0,
                entry_price=executed_price,
                exit_price=0,
                side="long" if new_shares > 0 else "short",
                size=abs(new_shares),
            )

        # Update cost basis
        if new_shares != 0:
            self.position_cost[symbol] = executed_price

        return closed_trade

    def order_target_percent(
        self,
        symbol: str,
        target_pct: float,
        price: float,
        date_index: Optional[int] = None,
    ) -> Optional[Trade]:
        """Adjust position to target percentage of current equity."""
        current_equity = self.equity[-1]
        target_value = current_equity * target_pct
        current_shares = self.positions.get(symbol, 0)
        current_value = current_shares * price
        delta_value = target_value - current_value
        delta_shares = delta_value / price if price > 0 else 0
        return self.order(symbol, delta_shares, price, date_index)

    def close_all(self, prices: dict[str, float], date_index: Optional[int] = None):
        """Close all open positions at given prices."""
        for symbol, shares in list(self.positions.items()):
            if shares != 0 and symbol in prices:
                self.order(symbol, -shares, prices[symbol], date_index)

    # ------------------------------------------------------------------
    # Daily update
    # ------------------------------------------------------------------

    def mark_to_market(self, prices: dict[str, float]):
        """Update equity based on current prices."""
        position_value = sum(
            self.positions.get(sym, 0) * prices[sym]
            for sym in self.positions
            if sym in prices
        )
        total_equity = self.cash + position_value
        self.equity.append(total_equity)
        self._bar += 1

    def next(self, prices: dict[str, float], date_index: Optional[int] = None):
        """Advance to next bar: mark-to-market with current prices."""
        self.mark_to_market(prices)

    # ------------------------------------------------------------------
    # Results
    # ------------------------------------------------------------------

    def results(self) -> BacktestResult:
        """Compute and return backtest performance metrics."""
        equity = np.array(self.equity)
        daily_returns = np.diff(equity) / equity[:-1]
        daily_returns = np.where(np.isnan(daily_returns), 0, daily_returns)

        n_days = len(daily_returns)
        total_return = equity[-1] / equity[0] - 1 if equity[0] > 0 else 0
        annual_return = (1 + total_return) ** (252 / max(n_days, 1)) - 1
        annual_vol = np.std(daily_returns, ddof=1) * np.sqrt(252) if n_days > 1 else 0
        sharpe = (np.mean(daily_returns) / np.std(daily_returns, ddof=1) * np.sqrt(252)) if np.std(daily_returns, ddof=1) > 0 else 0

        # Sortino ratio
        downside = daily_returns[daily_returns < 0]
        downside_std = np.std(downside, ddof=1) if len(downside) > 1 else 0
        sortino = (np.mean(daily_returns) / downside_std * np.sqrt(252)) if downside_std > 0 else 0

        # Max drawdown
        peak = np.maximum.accumulate(equity)
        drawdown = (equity - peak) / peak
        max_dd = np.min(drawdown)

        # Drawdown duration
        dd_duration = 0
        current_duration = 0
        for dd in drawdown:
            if dd < 0:
                current_duration += 1
                dd_duration = max(dd_duration, current_duration)
            else:
                current_duration = 0

        calmar = annual_return / abs(max_dd) if max_dd != 0 else 0

        # Trade statistics
        winning_trades = [t for t in self.trades if t.pnl > 0]
        losing_trades = [t for t in self.trades if t.pnl <= 0]
        win_rate = len(winning_trades) / len(self.trades) if self.trades else 0
        gross_profit = sum(t.pnl for t in winning_trades) if winning_trades else 0
        gross_loss = abs(sum(t.pnl for t in losing_trades)) if losing_trades else 1
        profit_factor = gross_profit / gross_loss if gross_loss > 0 else float("inf")
        avg_trade_pnl = np.mean([t.pnl for t in self.trades]) if self.trades else 0

        return BacktestResult(
            total_return=total_return,
            annual_return=annual_return,
            annual_volatility=annual_vol,
            sharpe_ratio=sharpe,
            sortino_ratio=sortino,
            max_drawdown=max_dd,
            max_drawdown_duration=dd_duration,
            calmar_ratio=calmar,
            win_rate=win_rate,
            profit_factor=profit_factor,
            avg_trade_pnl=avg_trade_pnl,
            total_trades=len(self.trades),
            equity_curve=equity,
            daily_returns=daily_returns,
            trades=self.trades,
        )


# ---------------------------------------------------------------------------
# Convenience: Run a signal-based strategy through the engine
# ---------------------------------------------------------------------------

def run_backtest(
    prices: pd.DataFrame,
    signals: pd.DataFrame,
    initial_capital: float = 1_000_000.0,
    commission: float = 0.005,
) -> BacktestResult:
    """Run a complete backtest from price data and trading signals.

    Args:
        prices: T×N DataFrame of prices (columns = symbols, index = dates).
        signals: T×N DataFrame of target positions in shares (positive=long, negative=short).
        initial_capital: starting capital.
        commission: per-share commission.

    Returns:
        BacktestResult with performance metrics.
    """
    engine = BacktestEngine(
        initial_capital=initial_capital,
        commission_per_share=commission,
    )

    symbols = list(prices.columns)
    for i in range(len(prices)):
        current_prices = {sym: prices.iloc[i][sym] for sym in symbols if not pd.isna(prices.iloc[i][sym])}

        # Execute orders
        for sym in symbols:
            if sym in signals.columns and i < len(signals):
                target = signals.iloc[i][sym]
                if not pd.isna(target) and sym in current_prices:
                    current_shares = engine.positions.get(sym, 0)
                    delta = target - current_shares
                    if delta != 0:
                        engine.order(sym, delta, current_prices[sym], i)

        # Mark to market
        engine.mark_to_market(current_prices)

    # Close all positions at final prices
    final_prices = {sym: prices.iloc[-1][sym] for sym in symbols if not pd.isna(prices.iloc[-1][sym])}
    engine.close_all(final_prices, len(prices) - 1)

    return engine.results()
