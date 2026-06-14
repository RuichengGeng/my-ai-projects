"""Real-time quote streamer (Yahoo Finance websocket).

Subscribes to live ticks for the monitor universe and maintains a small
JSON file the dashboard reads:

    output/option_monitor/realtime/quotes.json

Run it in its own terminal alongside the dashboard:

    uv run python -m option_monitor.stream

Stop with Ctrl-C.  This process is optional — without it the dashboard
falls back to polling Yahoo's 1-minute bars (see realtime.py).

Notes
-----
- Yahoo streams price ticks only (no option chains over websocket).
- Outside market hours Yahoo sends few/no ticks; the dashboard marks any
  quote older than ~3 minutes as not-live instead of pretending.
"""

import json
import logging
import os
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import certifi

# The websocket library uses the system SSL store, which on macOS Python
# installs is often empty — point it at certifi's bundle before connecting.
os.environ.setdefault("SSL_CERT_FILE", certifi.where())

import yfinance as yf  # noqa: E402  (must come after the SSL env fix)

sys.path.insert(0, str(Path(__file__).parent.parent))

from option_monitor.monitor import DEFAULT_UNIVERSE

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
logger = logging.getLogger("option_monitor.stream")

# ── Configuration ─────────────────────────────────────────────────────────────

SYMBOLS: list[str] = DEFAULT_UNIVERSE
OUT_FILE = Path("output/option_monitor/realtime/quotes.json")
WRITE_EVERY_SEC: float = 1.0     # throttle disk writes

# ── State ─────────────────────────────────────────────────────────────────────

_quotes: dict[str, dict] = {}
_last_write: float = 0.0


def _write(force: bool = False) -> None:
    """Atomically persist the current quote map (throttled)."""
    global _last_write
    now = time.monotonic()
    if not force and now - _last_write < WRITE_EVERY_SEC:
        return
    _last_write = now
    OUT_FILE.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "updated_at_utc": datetime.now(timezone.utc).isoformat(),
        "quotes": _quotes,
    }
    tmp = OUT_FILE.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(payload))
    tmp.replace(OUT_FILE)  # atomic on POSIX


def _on_message(msg: dict) -> None:
    """Handle one websocket tick.

    Yahoo tick fields used: id (symbol), price, time (epoch ms),
    change_percent.  Anything missing is stored as None — not fabricated.
    """
    sym = msg.get("id")
    price = msg.get("price")
    ts_ms = msg.get("time")
    if not sym or price is None or ts_ms is None:
        return

    change_pct = msg.get("change_percent")
    prev_close = (
        round(price / (1 + change_pct / 100), 4)
        if change_pct is not None and change_pct > -100 else None
    )
    _quotes[sym] = {
        "price": round(float(price), 4),
        "time": float(ts_ms) / 1000.0,  # epoch seconds
        "change_pct": round(float(change_pct), 2) if change_pct is not None else None,
        "prev_close": prev_close,
    }
    _write()


def main() -> None:
    logger.info("streaming %d symbols -> %s", len(SYMBOLS), OUT_FILE)
    ws = yf.WebSocket(verbose=False)
    ws.subscribe(SYMBOLS)
    try:
        ws.listen(_on_message)
    except KeyboardInterrupt:
        logger.info("stopping…")
    finally:
        _write(force=True)
        ws.close()


if __name__ == "__main__":
    main()
