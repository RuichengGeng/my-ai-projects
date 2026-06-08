"""Demo: fetch Brent and WTI crude oil prices across all intervals."""

import sys
import os
import time

# Ensure project root is on path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from dotenv import load_dotenv

load_dotenv()

from . import CommoditiesProvider

cp = CommoditiesProvider()


def fetch(label, fn, interval):
    """Fetch with rate-limit delay for free-tier keys (1 req/sec, 25/day)."""
    time.sleep(1.1)  # respect free-tier per-second limit
    df = fn(interval=interval)
    print(f"\n--- {label} {interval} (latest 5 rows) ---")
    print(df.tail(5).to_string(index=False))


# ── WTI ─────────────────────────────────────────────────────────────
print("=" * 60)
print("WTI Crude Oil (West Texas Intermediate)")
print("=" * 60)

fetch("WTI", cp.get_wti, "daily")
fetch("WTI", cp.get_wti, "weekly")
fetch("WTI", cp.get_wti, "monthly")

# ── Brent ───────────────────────────────────────────────────────────
print("\n" + "=" * 60)
print("Brent Crude Oil")
print("=" * 60)

fetch("Brent", cp.get_brent, "daily")
fetch("Brent", cp.get_brent, "weekly")
fetch("Brent", cp.get_brent, "monthly")
