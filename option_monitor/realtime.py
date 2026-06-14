"""Real-time (delayed ~1 min) quote access for the option monitor.

Two sources, same output schema:

1. ``read_stream_quotes()`` — reads the JSON file maintained by the
   websocket streamer (``python -m option_monitor.stream``).  Lowest
   latency, but only available while the streamer process is running.
2. ``RealtimeQuoteService.fetch()`` — polls Yahoo's 1-minute bars in one
   batched HTTP request.  No extra process needed; this is what the
   dashboard falls back to.

Output schema (one row per symbol):
    symbol, price, prev_close, change_pct, quote_time (UTC), age_sec,
    is_live, source

``is_live`` is True when the quote is younger than ``STALE_AFTER_SEC`` —
otherwise you are looking at the last print of a closed session.
No values are imputed: symbols Yahoo returns nothing for are omitted.
"""

from __future__ import annotations

import json
import logging
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
import yfinance as yf

logger = logging.getLogger(__name__)

STALE_AFTER_SEC: int = 180          # quote older than this => not live
STREAM_FILE = Path("output/option_monitor/realtime/quotes.json")

_COLUMNS = [
    "symbol", "price", "prev_close", "change_pct",
    "quote_time", "age_sec", "is_live", "source",
]


def _finalize(rows: list[dict], source: str) -> pd.DataFrame:
    """Stamp age/liveness/source on raw quote rows."""
    now = datetime.now(timezone.utc)
    for r in rows:
        age = (now - r["quote_time"]).total_seconds()
        r["age_sec"] = round(age)
        r["is_live"] = age < STALE_AFTER_SEC
        r["source"] = source
    df = pd.DataFrame(rows, columns=_COLUMNS)
    return df.sort_values("symbol").reset_index(drop=True)


class RealtimeQuoteService:
    """Batched polling of Yahoo 1-minute bars (delayed ~1 minute)."""

    def fetch(self, symbols: list[str]) -> pd.DataFrame:
        """Latest 1-minute close + previous session close per symbol.

        Two batched requests for the whole universe.  Symbols with no
        intraday data (e.g. delisted) are omitted, not faked.
        """
        intraday = yf.download(
            symbols, period="1d", interval="1m",
            group_by="ticker", auto_adjust=False,
            progress=False, threads=True,
        )
        daily = yf.download(
            symbols, period="5d", interval="1d",
            group_by="ticker", auto_adjust=False,
            progress=False, threads=True,
        )

        rows = []
        for sym in symbols:
            try:
                intra = intraday[sym].dropna(subset=["Close"])
                day = daily[sym].dropna(subset=["Close"])
            except KeyError:
                logger.warning("%s: no realtime data returned — omitted", sym)
                continue
            if intra.empty or day.empty:
                logger.warning("%s: no realtime data returned — omitted", sym)
                continue

            quote_time = pd.Timestamp(intra.index[-1]).tz_convert("UTC")
            price = float(intra["Close"].iloc[-1])

            # Previous *session* close: last daily bar strictly before the
            # quote's session date.
            day_dates = [pd.Timestamp(ts).date() for ts in day.index]
            quote_date = quote_time.tz_convert("America/New_York").date()
            prev_closes = [
                float(c) for d, c in zip(day_dates, day["Close"]) if d < quote_date
            ]
            prev_close = prev_closes[-1] if prev_closes else None

            rows.append({
                "symbol": sym,
                "price": round(price, 4),
                "prev_close": round(prev_close, 4) if prev_close else None,
                "change_pct": (
                    round((price / prev_close - 1) * 100, 2) if prev_close else None
                ),
                "quote_time": quote_time.to_pydatetime(),
            })

        return _finalize(rows, source="yahoo 1m poll (~1 min delay)")


def read_stream_quotes(
    path: Path = STREAM_FILE, symbols: list[str] | None = None
) -> pd.DataFrame:
    """Read the websocket streamer's quote file.

    Returns an empty DataFrame when the file does not exist or is
    unparsable — callers should fall back to :class:`RealtimeQuoteService`.
    """
    if not path.exists():
        return pd.DataFrame(columns=_COLUMNS)
    try:
        payload = json.loads(path.read_text())
    except (json.JSONDecodeError, OSError):
        logger.warning("stream file %s unreadable — ignoring", path)
        return pd.DataFrame(columns=_COLUMNS)

    rows = []
    for sym, q in payload.get("quotes", {}).items():
        if symbols and sym not in symbols:
            continue
        rows.append({
            "symbol": sym,
            "price": q.get("price"),
            "prev_close": q.get("prev_close"),
            "change_pct": q.get("change_pct"),
            "quote_time": datetime.fromtimestamp(q["time"], tz=timezone.utc),
        })
    return _finalize(rows, source="yahoo websocket stream")
