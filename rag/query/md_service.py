"""Markdown-file backed retriever.

This retriever keeps the same public result shape as ``RAGRetriever`` but uses
the parsed markdown files directly instead of ChromaDB embeddings. It is useful
for source-scoped, exact-term heavy workflows where the markdown corpus is
already available and metadata filtering matters more than semantic search.
"""

from __future__ import annotations

import re
from datetime import datetime, timezone
from pathlib import Path

from rag.config import PDF_FILES_DIR, RERANKER_MODEL
from rag.indexer.service import _extract_topic, _parse_series
from rag.query.reranker import BM25Index, CrossEncoderReranker, tokenize


DEFAULT_CANDIDATE_MULTIPLIER = 8
SECTION_TITLE_WEIGHT = 1.5
DOC_TITLE_WEIGHT = 0.4
ANALYSIS_SECTION_BOOST = 2.0
NOTICE_SECTION_PENALTY = 0.65

ANALYSIS_SECTION_TERMS = (
    "market analysis",
    "daily market analysis",
    "weekly oil recap",
    "commentary",
    "benchmarks",
    "assessments",
    "price assessments",
    "demand",
    "fuel oil",
    "gasoil",
    "kerosene",
    "jet fuel",
    "naphtha",
    "gasoline",
    "crude",
)

NOTICE_SECTION_TERMS = (
    "subscriber note",
    "subscriber notes",
    "launch",
    "to launch",
    "invites feedback",
    "proposes",
    "corrects",
    "withdrawals",
    "rationale",
    "rationales",
    "exclusions",
    "bids, offers, trades",
    "bids offers trades",
    "reported deals",
)


def _find_md_files(root: Path) -> list[Path]:
    return sorted(root.rglob("*.md"))


def _split_by_h2(text: str, doc_name: str) -> list[dict]:
    """Split markdown into the same H2 chunks used by the Chroma indexer."""
    h2 = re.compile(r"^(##\s+.+)$", re.MULTILINE)
    positions = [(m.start(), m.group(1)) for m in h2.finditer(text)]

    def _make_chunk(section: str, raw: str) -> dict:
        images = re.findall(r"!\[.*?\]\((images/[^)]+)\)", raw)
        return {"section": section, "text": raw.strip(), "images": images}

    if not positions:
        return [_make_chunk(doc_name, text)]

    chunks = []
    preamble = text[: positions[0][0]].strip()
    if preamble:
        chunks.append(_make_chunk(doc_name, preamble))

    for i, (pos, header) in enumerate(positions):
        end = positions[i + 1][0] if i + 1 < len(positions) else len(text)
        chunks.append(_make_chunk(header.lstrip("#").strip(), text[pos:end]))

    return chunks


class MarkdownRAGRetriever:
    """Retrieve chunks directly from parsed markdown files using BM25.

    Args:
        pdf_files_dir: Root containing parser output folders with ``*.md`` files.
        use_reranker: Apply the same cross-encoder reranker as ``RAGRetriever``.
        min_score: Optional BM25 threshold. Scores are raw BM25 values.
    """

    def __init__(
        self,
        pdf_files_dir: Path = PDF_FILES_DIR,
        use_reranker: bool = False,
        min_score: float = 0.0,
        candidate_multiplier: int = DEFAULT_CANDIDATE_MULTIPLIER,
        use_section_rescore: bool = True,
    ):
        self.pdf_files_dir = pdf_files_dir
        self.use_reranker = use_reranker
        self.min_score = min_score
        self.candidate_multiplier = candidate_multiplier
        self.use_section_rescore = use_section_rescore
        self._chunks = self._load_chunks()
        self._bm25_index = BM25Index(self._chunks)
        self._reranker = None

        if use_reranker:
            print(f"  [md-retriever] loading cross-encoder: {RERANKER_MODEL}")
            self._reranker = CrossEncoderReranker(RERANKER_MODEL)

    def _load_chunks(self) -> list[dict]:
        chunks: list[dict] = []
        for md_path in _find_md_files(self.pdf_files_dir):
            doc_name = md_path.parent.name
            series_name, series_date = _parse_series(doc_name)
            topic = _extract_topic(doc_name, series_name)
            indexed_at = datetime.fromtimestamp(
                md_path.stat().st_mtime,
                timezone.utc,
            ).strftime("%Y-%m-%dT%H:%M:%SZ")

            text = md_path.read_text(encoding="utf-8")
            for chunk in _split_by_h2(text, doc_name):
                chunks.append(
                    {
                        "text": chunk["text"],
                        "search_text": "\n".join(
                            [
                                chunk["section"],
                                chunk["section"],
                                doc_name,
                                topic,
                                chunk["text"],
                            ]
                        ),
                        "metadata": {
                            "doc_name": doc_name,
                            "source": str(md_path),
                            "section": chunk["section"],
                            "images": ",".join(chunk["images"]),
                            "topic": topic,
                            "series_name": series_name or "",
                            "series_date": series_date.isoformat() if series_date else "",
                            "indexed_at": indexed_at,
                        },
                    }
                )
        return chunks

    @staticmethod
    def _phrase_hits(text: str, phrases: tuple[str, ...]) -> int:
        lowered = text.lower()
        return sum(1 for phrase in phrases if phrase in lowered)

    def _rescore(self, query: str, candidates: list[dict]) -> list[dict]:
        query_terms = set(tokenize(query))
        rescored = []

        for candidate in candidates:
            meta = candidate.get("metadata", {})
            section = meta.get("section", "")
            doc_name = meta.get("doc_name", "")
            section_terms = set(tokenize(section))
            doc_terms = set(tokenize(doc_name))

            section_hits = len(query_terms & section_terms)
            doc_hits = len(query_terms & doc_terms)
            analysis_hits = self._phrase_hits(section, ANALYSIS_SECTION_TERMS)
            notice_hits = self._phrase_hits(section, NOTICE_SECTION_TERMS)

            score = float(candidate.get("bm25_score", 0.0))
            score += section_hits * SECTION_TITLE_WEIGHT
            score += doc_hits * DOC_TITLE_WEIGHT
            score += analysis_hits * ANALYSIS_SECTION_BOOST
            if notice_hits:
                score *= NOTICE_SECTION_PENALTY ** notice_hits

            rescored.append(
                {
                    **candidate,
                    "score": score,
                    "md_score": score,
                    "section_hits": section_hits,
                    "analysis_hits": analysis_hits,
                    "notice_hits": notice_hits,
                }
            )

        return sorted(rescored, key=lambda c: c["md_score"], reverse=True)

    def retrieve(
        self,
        question: str,
        n_results: int = 15,
        topic: str = "",
        where: dict | None = None,
    ) -> list[dict]:
        """Return top matching markdown chunks.

        The method mirrors ``RAGRetriever.retrieve`` enough for agents to swap
        retrievers without changing prompt/context code.
        """
        if where is None and topic:
            where = {"topic": {"$eq": topic}}

        candidates = self._bm25_index.search(
            question,
            top_n=max(n_results * self.candidate_multiplier, n_results),
            where=where,
        )
        if self.min_score > 0:
            candidates = [
                c for c in candidates if c.get("bm25_score", 0.0) >= self.min_score
            ]

        if self.use_section_rescore:
            candidates = self._rescore(question, candidates)

        if self.use_reranker and self._reranker is not None and candidates:
            results = self._reranker.rerank(question, candidates, top_n=n_results)
        else:
            results = candidates[:n_results]

        for result in results:
            if "score" not in result:
                result["score"] = float(result.get("bm25_score", 0.0))

        return results

    def ask(self, question: str, n_results: int = 5, topic: str = "") -> str:
        """Generate an answer from markdown-backed retrieval context."""
        import os

        from openai import OpenAI

        from rag.config import DEEPSEEK_BASE_URL, DEEPSEEK_MODEL
        from rag.utils.token_tracker import get_active_tracker
        from utils.retry import call_with_retry

        chunks = self.retrieve(question, n_results=n_results, topic=topic)
        context = "\n\n---\n\n".join(
            f"[{c['metadata']['doc_name']} / {c['metadata']['section']}]\n{c['text']}"
            for c in chunks
        )

        client = OpenAI(
            api_key=os.environ.get("DEEPSEEK_API_KEY"),
            base_url=DEEPSEEK_BASE_URL,
        )
        response = call_with_retry(
            lambda: client.chat.completions.create(
                model=DEEPSEEK_MODEL,
                max_tokens=1024,
                messages=[
                    {
                        "role": "user",
                        "content": (
                            "Answer the question using only the provided context. "
                            "Cite the document and section for each key claim.\n\n"
                            f"Context:\n{context}\n\n"
                            f"Question: {question}"
                        ),
                    }
                ],
            )
        )
        tracker = get_active_tracker()
        if tracker is not None:
            tracker.record(response.usage, label="md_retriever_ask")
        return response.choices[0].message.content
