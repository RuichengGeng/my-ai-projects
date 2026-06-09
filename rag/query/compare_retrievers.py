#!/usr/bin/env python
"""Compare Chroma-backed and markdown-backed retrieval performance."""

from __future__ import annotations

import argparse
import statistics
import sys
import time
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from rag.query.md_service import MarkdownRAGRetriever
from rag.query.service import RAGRetriever


DEFAULT_QUERIES = [
    "crude daily market analysis commentary benchmark assessments price changes Dubai Oman Murban Brent WTI ESPO",
    "fuel oil daily market analysis bunker HSFO VLSFO 380cst 180cst crack premiums discounts demand",
    "gasoil jet kerosene daily market analysis middle distillates diesel crack refinery margin demand",
    "naphtha gasoline LPG daily market analysis light ends crack spreads premiums demand",
    "arbitrage spread East West EFS Asia Europe Atlantic Basin flows premiums discounts freight",
    "supply demand inventory stocks draw build production consumption trade flows refinery runs OPEC",
]


def _timer() -> float:
    return time.perf_counter()


def _key(chunk: dict) -> str:
    meta = chunk.get("metadata", {})
    return f"{meta.get('doc_name', '')}::{meta.get('section', '')}"


def _short_hit(chunk: dict) -> str:
    meta = chunk.get("metadata", {})
    score = chunk.get("ce_score", chunk.get("rrf_score", chunk.get("score", 0.0)))
    return (
        f"{score:.4f} | "
        f"{meta.get('doc_name', '')[:54]} / {meta.get('section', '')[:54]}"
    )


def _measure_init(label: str, factory):
    start = _timer()
    retriever = factory()
    elapsed = _timer() - start
    print(f"{label} init: {elapsed:.3f}s")
    return retriever, elapsed


def _measure_retrieve(
    retriever,
    query: str,
    n_results: int,
    repeats: int,
    where: dict | None = None,
) -> tuple[list[dict], list[float]]:
    timings = []
    chunks: list[dict] = []
    for _ in range(repeats):
        start = _timer()
        chunks = retriever.retrieve(query, n_results=n_results, where=where)
        timings.append(_timer() - start)
    return chunks, timings


def _fmt_ms(seconds: float) -> str:
    return f"{seconds * 1000:.1f}ms"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compare Chroma RAGRetriever with MarkdownRAGRetriever.",
    )
    parser.add_argument(
        "--query",
        action="append",
        help="Query to benchmark. Can be passed multiple times.",
    )
    parser.add_argument(
        "--n-results",
        type=int,
        default=5,
        help="Chunks returned per query.",
    )
    parser.add_argument(
        "--repeats",
        type=int,
        default=3,
        help="Retrieval repeats per query after retriever construction.",
    )
    parser.add_argument(
        "--rerank",
        action="store_true",
        help="Enable cross-encoder reranking in both retrievers.",
    )
    parser.add_argument(
        "--no-bm25-hybrid",
        action="store_true",
        help="Disable BM25 hybrid mode for the Chroma retriever.",
    )
    parser.add_argument(
        "--doc-name",
        action="append",
        help="Restrict both retrievers to a document name. Can be passed multiple times.",
    )
    parser.add_argument(
        "--raw-md-bm25",
        action="store_true",
        help="Disable markdown section-aware rescoring.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    queries = args.query or DEFAULT_QUERIES

    chroma, chroma_init = _measure_init(
        "Chroma",
        lambda: RAGRetriever(
            use_bm25=not args.no_bm25_hybrid,
            use_reranker=args.rerank,
        ),
    )
    markdown, markdown_init = _measure_init(
        "Markdown",
        lambda: MarkdownRAGRetriever(
            use_reranker=args.rerank,
            use_section_rescore=not args.raw_md_bm25,
        ),
    )
    where = None
    if args.doc_name:
        where = (
            {"doc_name": {"$eq": args.doc_name[0]}}
            if len(args.doc_name) == 1
            else {"doc_name": {"$in": args.doc_name}}
        )

    chroma_times: list[float] = []
    markdown_times: list[float] = []
    overlaps: list[float] = []

    print("\nPer-query results")
    print("=" * 80)
    for i, query in enumerate(queries, 1):
        chroma_chunks, ct = _measure_retrieve(
            chroma,
            query,
            n_results=args.n_results,
            repeats=args.repeats,
            where=where,
        )
        markdown_chunks, mt = _measure_retrieve(
            markdown,
            query,
            n_results=args.n_results,
            repeats=args.repeats,
            where=where,
        )
        chroma_times.extend(ct)
        markdown_times.extend(mt)

        chroma_keys = {_key(c) for c in chroma_chunks}
        markdown_keys = {_key(c) for c in markdown_chunks}
        overlap = len(chroma_keys & markdown_keys) / max(len(chroma_keys | markdown_keys), 1)
        overlaps.append(overlap)

        print(f"\n[{i}] {query}")
        print(
            "  timing: "
            f"Chroma median={_fmt_ms(statistics.median(ct))} "
            f"Markdown median={_fmt_ms(statistics.median(mt))} "
            f"overlap={overlap:.0%}"
        )
        print("  Chroma top hit:   " + (_short_hit(chroma_chunks[0]) if chroma_chunks else "none"))
        print("  Markdown top hit: " + (_short_hit(markdown_chunks[0]) if markdown_chunks else "none"))

    print("\nSummary")
    print("=" * 80)
    print(f"Chroma init:        {chroma_init:.3f}s")
    print(f"Markdown init:      {markdown_init:.3f}s")
    print(f"Chroma retrieve:    median={_fmt_ms(statistics.median(chroma_times))}")
    print(f"Markdown retrieve:  median={_fmt_ms(statistics.median(markdown_times))}")
    print(f"Top-{args.n_results} overlap:  mean={statistics.mean(overlaps):.0%}")


if __name__ == "__main__":
    main()
