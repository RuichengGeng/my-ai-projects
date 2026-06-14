"""Script runner for OptionMonitor.

Edit the configuration constants below, then run:

    uv run python -m option_monitor.run

See option_monitor/README.md for the full documentation.
"""

import json
import logging
import shutil
import sys
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
from rich.console import Console
from rich.table import Table
from rich import box

sys.path.insert(0, str(Path(__file__).parent.parent))

from option_monitor.monitor import (
    DEFAULT_UNIVERSE,
    ETF_UNIVERSE,
    EQUITY_UNIVERSE,
    NY_TZ,
    OptionMonitor,
    last_completed_session,
)

logging.basicConfig(level=logging.WARNING, format="%(levelname)s %(name)s: %(message)s")

console = Console()

# ── Script configuration ─────────────────────────────────────────────────────

# Choose the symbols to monitor. Other useful options:
#   ETF_UNIVERSE
#   EQUITY_UNIVERSE
#   ["QQQ", "SPY", "AAPL"]
SYMBOLS: list[str] = DEFAULT_UNIVERSE

# None → auto-detect the last *completed* US session (run during market
# hours → yesterday; after ~4:05pm ET → today; weekend → Friday).  A
# "yyyy-mm-dd" string overrides the label/DTE reference — but Yahoo always
# serves the *live* option chain, so overriding cannot fetch the past.
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
    "data_quality":     "Data Quality (freshness & completeness flags)",
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
    "data_quality",
]


def _print_quality_warnings(results: dict[str, pd.DataFrame]) -> None:
    """Surface stale/missing-data flags prominently after the run."""
    dq = results.get("data_quality", pd.DataFrame())
    if dq.empty:
        return
    stale_price = dq[dq["price_is_stale"] == True]["symbol"].tolist()  # noqa: E712
    stale_chain = dq[dq["chain_is_stale"] == True]["symbol"].tolist()  # noqa: E712
    no_volume = dq[dq["volume_reported"] == False]["symbol"].tolist()  # noqa: E712
    junk_iv = (
        dq[dq["iv_reliable"] == False]["symbol"].tolist()  # noqa: E712
        if "iv_reliable" in dq.columns else []
    )
    if not (stale_price or stale_chain or no_volume or junk_iv):
        console.print("\n[green]Data quality: all symbols fresh.[/green]")
        return
    console.print("\n[bold yellow]⚠ Data quality warnings[/bold yellow]")
    if stale_price:
        console.print(
            f"  [yellow]stale prices[/yellow] (last bar before valuation date — "
            f"pre-open, weekend or holiday?): {', '.join(stale_price)}"
        )
    if stale_chain:
        console.print(
            f"  [yellow]stale option chains[/yellow] (last option trade before "
            f"valuation date): {', '.join(stale_chain)}"
        )
    if no_volume:
        console.print(
            f"  [yellow]no option volume reported[/yellow] (volume metrics are "
            f"missing, not zero): {', '.join(no_volume)}"
        )
    if junk_iv:
        console.print(
            f"  [red]unreliable IVs[/red] (Yahoo served placeholder values — "
            f"all IV metrics reported missing): {', '.join(junk_iv)}"
        )


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
    results: dict[str, pd.DataFrame], output_dir: Path, vdate: str,
    meta: dict,
) -> None:
    """Write a dated snapshot folder and mirror it to ``latest/``."""
    snap_dir = output_dir / vdate
    snap_dir.mkdir(parents=True, exist_ok=True)
    for name, df in results.items():
        path = snap_dir / f"{name}.csv"
        df.to_csv(path, index=False)
        console.print(f"  [green]saved[/green] {path}")

    meta_path = snap_dir / "meta.json"
    meta_path.write_text(json.dumps(meta, indent=2))
    console.print(f"  [green]saved[/green] {meta_path}")

    latest = output_dir / "latest"
    if latest.exists():
        shutil.rmtree(latest)
    shutil.copytree(snap_dir, latest)
    console.print(f"  [green]mirrored[/green] {latest}")


# ── Script entry point ───────────────────────────────────────────────────────

def main() -> None:
    target_symbols = SYMBOLS
    run_at = datetime.now(timezone.utc)
    now_et = run_at.astimezone(NY_TZ)

    if VALUATION_DATE:
        vdate_str = VALUATION_DATE
        session_note = "manual override"
    else:
        vdate_str = last_completed_session(now=now_et).isoformat()
        session_note = "auto-detected last completed session"

    market_open_now = vdate_str != now_et.date().isoformat() and now_et.weekday() < 5

    console.rule(f"[bold]OptionMonitor — session {vdate_str}[/bold]")
    console.print(
        f"Session: {vdate_str} ({session_note})  ·  "
        f"run at {now_et:%Y-%m-%d %H:%M %Z}"
    )
    if market_open_now:
        console.print(
            "[yellow]Note:[/yellow] a newer session may be in progress — EOD "
            "price metrics describe the completed session, but the option "
            "chain is captured [bold]live now[/bold] (Yahoo has no historical "
            "chains). For a coherent EOD snapshot, re-run after ~4:05pm ET."
        )
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

    _print_quality_warnings(results)

    # ── Optional CSV export ───────────────────────────────────────────────────
    if OUTPUT_DIR:
        meta = {
            "session_date": vdate_str,
            "session_note": session_note,
            "run_at_utc": run_at.isoformat(timespec="seconds"),
            "run_at_et": now_et.isoformat(timespec="seconds"),
            "chain_captured_live_during_newer_session": market_open_now,
            "n_symbols": len(target_symbols),
        }
        console.print("\n[bold]Saving CSVs…[/bold]")
        _save_csvs(results, Path(OUTPUT_DIR), vdate_str, meta)

    console.print(f"\n[bold green]Done.[/bold green]  Session: {vdate_str}")


if __name__ == "__main__":
    main()
