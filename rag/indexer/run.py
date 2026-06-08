#!/usr/bin/env python
"""Index all parsed PDFs into the vector store."""

import sys
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from rag.indexer.service import build_index, list_docs

# ── Configure here ────────────────────────────────────────────────────────────
RESET = True   # set True to drop and rebuild the index from scratch
LIST_ONLY = False  # set True to just print indexed docs and exit
# ─────────────────────────────────────────────────────────────────────────────


def main() -> None:
    if LIST_ONLY:
        list_docs()
    else:
        build_index(reset=RESET)


if __name__ == "__main__":
    main()
