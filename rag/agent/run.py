#!/usr/bin/env python
"""Run the RAG agent for a single question."""

import sys
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from rag.agent.service import RAGAgent

# ── Configure here ────────────────────────────────────────────────────────────
QUESTION = "What is the latest market update for Asia-Pacific Arab Gulf?"
N_RESULTS = 5      # chunks retrieved per iteration
MAX_ITERATIONS = 3  # max retrieval-generate-evaluate loops
# ─────────────────────────────────────────────────────────────────────────────


def main() -> None:
    agent = RAGAgent(n_results=N_RESULTS, max_iterations=MAX_ITERATIONS)
    answer = agent.run(QUESTION)
    print("\nFinal Answer:")
    print("=" * 60)
    print(answer)


if __name__ == "__main__":
    main()
