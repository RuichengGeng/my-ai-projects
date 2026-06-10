"""
CLI runner for OptionMonitor.

Usage examples
--------------
  # Full default universe (all ETFs + equities)
  uv run python scripts/run_option_monitor.py

  # Custom symbols with explicit valuation date
  uv run python scripts/run_option_monitor.py -s QQQ SPY AAPL -d 2025-06-09

  # ETFs only, save CSVs
  uv run python scripts/run_option_monitor.py --etfs-only -o /tmp/option_data

  # Equities only
  uv run python scripts/run_option_monitor.py --equities-only
"""

import sys
from datetime import date
from pathlib import Path
from typing import Optional

import pandas as pd
import typer
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

app = typer.Typer(add_completion=False, help="Run the OptionMonitor and display results.")
console = Console()

# ── Table display helpers ─────────────────────────────────────────────────────

_TABLE_TITLES: dict[str, str] = {
    "spot":             "Spot Prices & Realized Volatility",
    "iv_term_structure": "IV Term Structure (ATM implied volatility by expiry)",
    "iv_skew":          "IV Skew (OTM put IV − OTM call IV @ ±5%)",
    "pc_overall":       "Put/Call Ratios — Overall",
    "pc_by_expiry":     "Put/Call Ratios — By DTE Bucket",
    "pc_by_moneyness":  "Put/Call Ratios — By Moneyness Bucket",
    "positioning":      "Positioning (Max Pain / Call Wall / Put Wall)",
}

_DISPLAY_ORDER = [
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


def _save_csvs(results: dict[str, pd.DataFrame], output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    for name, df in results.items():
        path = output_dir / f"{name}.csv"
        df.to_csv(path, index=False)
        console.print(f"  [green]saved[/green] {path}")


# ── CLI entry point ───────────────────────────────────────────────────────────

@app.command()
def main(
    symbols: Optional[str] = typer.Option(
        None, "--symbols", "-s", help="Comma-separated symbols to analyse, e.g. QQQ,SPY,AAPL."
    ),
    valuation_date: Optional[str] = typer.Option(
        None, "--valuation-date", "-d",
        help="Reference date for DTE and spot history (yyyy-mm-dd). Defaults to today.",
    ),
    etfs_only: bool = typer.Option(
        False, "--etfs-only", help="Restrict to the ETF universe."
    ),
    equities_only: bool = typer.Option(
        False, "--equities-only", help="Restrict to the equity universe."
    ),
    output_dir: Optional[str] = typer.Option(
        None, "--output-dir", "-o",
        help="If set, save every result table as a CSV in this directory.",
    ),
) -> None:
    if etfs_only and equities_only:
        console.print("[red]Error:[/red] --etfs-only and --equities-only are mutually exclusive.")
        raise typer.Exit(code=1)

    if symbols:
        target_symbols = [s.strip().upper() for s in symbols.split(",") if s.strip()]
    elif etfs_only:
        target_symbols = ETF_UNIVERSE
    elif equities_only:
        target_symbols = EQUITY_UNIVERSE
    else:
        target_symbols = DEFAULT_UNIVERSE

    vdate_str = valuation_date or date.today().isoformat()

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
    if output_dir:
        console.print("\n[bold]Saving CSVs…[/bold]")
        _save_csvs(results, Path(output_dir))

    console.print(f"\n[bold green]Done.[/bold green]  Valuation date: {vdate_str}")


if __name__ == "__main__":
    app()
