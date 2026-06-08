"""P2 retrieval helpers: BM25 index, cross-encoder reranker, RRF merge."""

from __future__ import annotations


def _matches_where(metadata: dict, where: dict) -> bool:
    """Evaluate a ChromaDB-style where filter against a metadata dict."""
    if not where:
        return True
    for key, condition in where.items():
        if key == "$and":
            return all(_matches_where(metadata, clause) for clause in condition)
        if not isinstance(condition, dict):
            continue
        op, val = next(iter(condition.items()))
        actual = metadata.get(key)
        if op == "$eq" and actual != val:
            return False
        if op == "$in" and actual not in val:
            return False
    return True


class BM25Index:
    """In-memory BM25 keyword index over a chunk corpus."""

    def __init__(self, chunks: list[dict]):
        from rank_bm25 import BM25Okapi
        self._chunks = chunks
        self._bm25 = BM25Okapi([c["text"].lower().split() for c in chunks])

    def search(self, query: str, top_n: int, where: dict | None = None) -> list[dict]:
        import numpy as np
        scores = self._bm25.get_scores(query.lower().split())
        order = np.argsort(scores)[::-1]
        results: list[dict] = []
        for idx in order:
            if len(results) >= top_n:
                break
            if scores[idx] <= 0:
                break
            chunk = self._chunks[int(idx)]
            if where and not _matches_where(chunk.get("metadata", {}), where):
                continue
            results.append({**chunk, "bm25_score": float(scores[idx])})
        return results


class CrossEncoderReranker:
    """Rerank (query, chunk) pairs with a cross-encoder model."""

    def __init__(self, model_name: str):
        from sentence_transformers.cross_encoder import CrossEncoder
        self._model = CrossEncoder(model_name)

    def rerank(self, query: str, chunks: list[dict], top_n: int) -> list[dict]:
        if not chunks:
            return []
        scores = self._model.predict([(query, c["text"]) for c in chunks])
        ranked = sorted(zip(chunks, scores), key=lambda x: x[1], reverse=True)
        return [{**chunk, "ce_score": float(score)} for chunk, score in ranked[:top_n]]


def reciprocal_rank_fusion(
    semantic_chunks: list[dict],
    bm25_chunks: list[dict],
    k: int = 60,
) -> list[dict]:
    """Merge two ranked lists via RRF. Returns merged list sorted by combined RRF score."""
    def _key(c: dict) -> str:
        m = c.get("metadata", {})
        return f"{m.get('doc_name', '')}::{m.get('section', c.get('text', '')[:40])}"

    rrf: dict[str, float] = {}
    by_key: dict[str, dict] = {}

    for rank, chunk in enumerate(semantic_chunks, 1):
        key = _key(chunk)
        rrf[key] = rrf.get(key, 0.0) + 1.0 / (k + rank)
        by_key[key] = chunk

    for rank, chunk in enumerate(bm25_chunks, 1):
        key = _key(chunk)
        rrf[key] = rrf.get(key, 0.0) + 1.0 / (k + rank)
        if key not in by_key:
            by_key[key] = chunk

    merged = []
    for key, score in sorted(rrf.items(), key=lambda x: x[1], reverse=True):
        merged.append({**by_key[key], "rrf_score": score})
    return merged
