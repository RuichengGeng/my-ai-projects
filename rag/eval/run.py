#!/usr/bin/env python
"""Run RAG performance evaluation against the test set."""

import sys
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from rag.eval.service import EvalService

# ── Configure here ────────────────────────────────────────────────────────────
TEST_SET = Path(__file__).parent / "test_set.json"
N_RESULTS = 5
# ─────────────────────────────────────────────────────────────────────────────


def main() -> None:
    svc = EvalService(n_results=N_RESULTS)
    svc.run_and_save(TEST_SET)


if __name__ == "__main__":
    main()
