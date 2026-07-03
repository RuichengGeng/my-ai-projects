"""Script runner for OptionMonitor.

Edit the configuration constants below, then run:

    uv run python scripts/run_option_monitor.py
"""

import logging
import shutil
import sys
from datetime import date
from pathlib import Path

import pandas as pd
from rich.console import Console
from rich.table import Table
from rich import box

sys.path.insert(0, str(Path(__file__).parent.parent))

from datafeed.option_monitor import (
    DEFAULT_UNIVERSE,
    ETF_UNIVERSE,
    EQUITY_UNIVERSE,
    OptionMonitor,
)

logging.basicConfig(level=logging.WARNING, format="%(levelname)s %(name)s: %(message)s")

console = Console()

# ── Script configuration ─────────────────────────────────────────────────────

# Choose the symbols to monitor. Other useful options:
#   ETF_UNIVERSE
#   EQUITY_UNIVERSE
#   ["QQQ", "SPY", "AAPL"]
SYMBOLS: list[str] = DEFAULT_UNIVERSE

# None → today.  A "yyyy-mm-dd" string only changes the DTE reference and the
# snapshot label — Yahoo always serves the *live* option chain, so this should
# stay None (or be the actual capture date) in normal use.
VALUATION_DATE: str | None = None

# Snapshots are written to OUTPUT_DIR/<valuation_date>/ (one folder per day,
# preserving history for trend analysis) and mirrored to OUTPUT_DIR/latest/.
# Set to None to skip export.
OUTPUT_DIR: str | None = "output/option_monitor"

# ── Table display helpers ─────────────────────────────────────────────────────

_TABLE_TITLES: dict[str, str] = {
    "derived":          "Market Summary (IV-RV, term slope, skew, sentiment)",
    "spot":             "Spot Prices, Volume & Realized Volatility",
    "iv_term_structure": "IV Term Structure (ATM implied volatility by expiry)",
    "iv_skew":          "IV Skew (OTM put IV − OTM call IV @ ±5%)",
    "pc_overall":       "Put/Call Ratios — Overall",
    "pc_by_expiry":     "Put/Call Ratios — By DTE Bucket",
    "pc_by_moneyness":  "Put/Call Ratios — By Moneyness Bucket",
    "positioning":      "Positioning (Max Pain / Call Wall / Put Wall, DTE ≤ 45)",
}

_DISPLAY_ORDER = [
    "derived",
    "spot",
    "positioning",
    "iv_term_structure",
    "iv_skew",
    "pc_overall",
    "pc_by_expiry",
    "pc_by_moneyness",
]


def _fmt(value) -> str:
    """Format a cell value for display."""
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return "-"
    if isinstance(value, float):
        return f"{value:.4f}"
    return str(value)


def _df_to_rich_table(df: pd.DataFrame, title: str) -> Table:
    display_df = df.drop(columns=["valuation_date"], errors="ignore")
    table = Table(
        title=title,
        box=box.SIMPLE_HEAVY,
        show_lines=False,
        header_style="bold cyan",
        title_style="bold white",
        min_width=60,
    )
    for col in display_df.columns:
        table.add_column(col, justify="right" if col not in ("symbol", "expiry", "dte_bucket", "moneyness_bucket", "side") else "left")

    for _, row in display_df.iterrows():
        table.add_row(*[_fmt(v) for v in row])

    return table


def _save_csvs(
    results: dict[str, pd.DataFrame], output_dir: Path, vdate: str
) -> None:
    """Write a dated snapshot folder and mirror it to ``latest/``."""
    snap_dir = output_dir / vdate
    snap_dir.mkdir(parents=True, exist_ok=True)
    for name, df in results.items():
        path = snap_dir / f"{name}.csv"
        df.to_csv(path, index=False)
        console.print(f"  [green]saved[/green] {path}")

    latest = output_dir / "latest"
    if latest.exists():
        shutil.rmtree(latest)
    shutil.copytree(snap_dir, latest)
    console.print(f"  [green]mirrored[/green] {latest}")


# ── Script entry point ───────────────────────────────────────────────────────

def main() -> None:
    target_symbols = SYMBOLS
    vdate_str = VALUATION_DATE or date.today().isoformat()

    console.rule(f"[bold]OptionMonitor — {vdate_str}[/bold]")
    console.print(f"Symbols ({len(target_symbols)}): {', '.join(target_symbols)}\n")

    monitor = OptionMonitor(symbols=target_symbols, valuation_date=vdate_str)

    with console.status("[bold green]Fetching data…[/bold green]"):
        results = monitor.run()

    # ── Display tables ────────────────────────────────────────────────────────
    for key in _DISPLAY_ORDER:
        df = results.get(key, pd.DataFrame())
        title = _TABLE_TITLES.get(key, key)
        if df.empty:
            console.print(f"\n[yellow]{title}[/yellow]: no data")
            continue
        table = _df_to_rich_table(df, title)
        console.print()
        console.print(table)

    # ── Optional CSV export ───────────────────────────────────────────────────
    if OUTPUT_DIR:
        console.print("\n[bold]Saving CSVs…[/bold]")
        _save_csvs(results, Path(OUTPUT_DIR), vdate_str)

    console.print(f"\n[bold green]Done.[/bold green]  Valuation date: {vdate_str}")


if __name__ == "__main__":
    main()




