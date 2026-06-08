#!/usr/bin/env python
"""Run a single RAG query against the indexed documents."""

import sys
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from rag.query.service import RAGRetriever

# ── Configure here ────────────────────────────────────────────────────────────
QUESTION = "How should a portfolio manager hedge against inflation using commodities?"
N_RESULTS = 5
# ─────────────────────────────────────────────────────────────────────────────


def main() -> None:
    rag = RAGRetriever()

    print(f"Question: {QUESTION}\n")

    chunks = rag.retrieve(QUESTION, n_results=N_RESULTS)
    print(f"Top {len(chunks)} retrieved chunks:")
    for i, c in enumerate(chunks, 1):
        print(f"  [{i}] score={c['score']:.3f}  {c['metadata']['doc_name'][:50]} / {c['metadata']['section'][:50]}")

    print("\nAnswer:")
    print("-" * 60)
    print(rag.ask(QUESTION, n_results=N_RESULTS))


if __name__ == "__main__":
    main()
