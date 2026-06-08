#!/usr/bin/env python
"""Index, update, or inspect the vector store."""

import sys
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from rag.indexer.service import build_index, delete_doc, list_docs, reindex_doc

# ── Configure here ────────────────────────────────────────────────────────────
RESET      = False   # True  → drop everything and rebuild from scratch
LIST_ONLY  = False   # True  → just print indexed docs and exit

# To delete a single document, set its folder name here (leave "" to skip):
DELETE_DOC = ""      # e.g. "APAC_Oil_Week_05_Jun_2026"

# To re-index an updated document, set its folder name here (leave "" to skip):
REINDEX_DOC = ""     # e.g. "APAC_Oil_Week_05_Jun_2026"
# ─────────────────────────────────────────────────────────────────────────────


def main() -> None:
    if LIST_ONLY:
        list_docs()
    elif DELETE_DOC:
        delete_doc(DELETE_DOC)
    elif REINDEX_DOC:
        reindex_doc(REINDEX_DOC)
    else:
        build_index(reset=RESET)


if __name__ == "__main__":
    main()
