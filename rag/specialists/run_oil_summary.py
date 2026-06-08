#!/usr/bin/env python
"""Generate a structured oil market summary from indexed documents."""

import sys
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from rag.specialists.oil_market import OilMarketSummaryAgent

# ── Configure here ────────────────────────────────────────────────────────────
QUESTION = "Summarize last week's APAC oil market update"
N_RESULTS = 5   # chunks retrieved per sub-query (6 sub-queries run in total)
# ─────────────────────────────────────────────────────────────────────────────


def main() -> None:
    agent = OilMarketSummaryAgent(n_results=N_RESULTS)
    answer = agent.run(QUESTION)
    print("\nOil Market Summary")
    print("=" * 60)
    print(answer)


if __name__ == "__main__":
    main()
