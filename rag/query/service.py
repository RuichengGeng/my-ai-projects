"""Query the ChromaDB index and generate answers with DeepSeek."""

import os

import chromadb
from chromadb.utils.embedding_functions import SentenceTransformerEmbeddingFunction

from rag.config import (
    CHROMA_PATH, COLLECTION_NAME, CONFIDENCE_THRESHOLD,
    DEEPSEEK_BASE_URL, DEEPSEEK_MODEL, EMBED_MODEL,
    RERANKER_MODEL, RERANK_CANDIDATES,
)


class RAGRetriever:
    """ChromaDB retriever with optional P2 enhancements.

    P2 features (disabled by default for backward-compat):
      min_score     – drop chunks below this cosine similarity (0.0 = off)
      use_reranker  – cross-encoder second-stage reranking
      use_bm25      – BM25 keyword search merged with semantic via RRF
    """

    def __init__(
        self,
        min_score: float = 0.0,
        use_reranker: bool = False,
        use_bm25: bool = False,
    ):
        ef = SentenceTransformerEmbeddingFunction(model_name=EMBED_MODEL)
        client = chromadb.PersistentClient(path=str(CHROMA_PATH))
        self.collection = client.get_collection(name=COLLECTION_NAME, embedding_function=ef)
        self.min_score = min_score
        self.use_reranker = use_reranker
        self.use_bm25 = use_bm25
        self._reranker = None
        self._bm25_index = None

        if use_reranker:
            from rag.query.reranker import CrossEncoderReranker
            print(f"  [retriever] loading cross-encoder: {RERANKER_MODEL}")
            self._reranker = CrossEncoderReranker(RERANKER_MODEL)

        if use_bm25:
            self._bm25_index = self._build_bm25_index()

    def _build_bm25_index(self):
        from rag.query.reranker import BM25Index
        print("  [retriever] building BM25 index …", end="", flush=True)
        all_data = self.collection.get(include=["documents", "metadatas"])
        chunks = [
            {"text": doc, "metadata": meta}
            for doc, meta in zip(all_data["documents"], all_data["metadatas"])
        ]
        idx = BM25Index(chunks)
        print(f" {len(chunks)} chunks indexed")
        return idx

    def retrieve(
        self,
        question: str,
        n_results: int = 5,
        topic: str = "",
        where: dict | None = None,
    ) -> list[dict]:
        """Return the top-n most relevant chunks for a question.

        Each result dict has:
          text      – chunk content
          metadata  – doc_name, section, source, images, topic
          score     – cosine similarity (1 = identical, 0 = orthogonal)
          ce_score  – cross-encoder score (present when use_reranker=True)
          rrf_score – RRF combined score (present when use_bm25=True)

        Pass topic or where to restrict results by metadata.
        When use_reranker or use_bm25 are enabled, RERANK_CANDIDATES are
        retrieved for recall, then narrowed to n_results for precision.
        """
        if where is None and topic:
            where = {"topic": {"$eq": topic}}

        recall_n = RERANK_CANDIDATES if (self.use_reranker or self.use_bm25) else n_results

        # Stage 1a: semantic search
        try:
            raw = self.collection.query(
                query_texts=[question],
                n_results=recall_n,
                where=where,
                include=["documents", "metadatas", "distances"],
            )
        except Exception:
            print("  [retriever] where-filter matched nothing, retrying unfiltered")
            raw = self.collection.query(
                query_texts=[question],
                n_results=recall_n,
                include=["documents", "metadatas", "distances"],
            )

        semantic = [
            {"text": doc, "metadata": meta, "score": round(1 - dist, 4)}
            for doc, meta, dist in zip(
                raw["documents"][0],
                raw["metadatas"][0],
                raw["distances"][0],
            )
        ]

        # Confidence threshold: drop semantic chunks below min_score
        if self.min_score > 0:
            before = len(semantic)
            semantic = [c for c in semantic if c["score"] >= self.min_score]
            dropped = before - len(semantic)
            if dropped:
                print(f"  [retriever] threshold {self.min_score}: dropped {dropped}/{before} chunk(s)")

        # Stage 1b: BM25 keyword search
        if self.use_bm25 and self._bm25_index is not None:
            bm25 = self._bm25_index.search(question, top_n=recall_n, where=where)
        else:
            bm25 = []

        # Merge with RRF when both lists are available
        if bm25:
            from rag.query.reranker import reciprocal_rank_fusion
            candidates = reciprocal_rank_fusion(semantic, bm25)[:recall_n]
        else:
            candidates = semantic

        # Stage 2: cross-encoder reranking
        if self.use_reranker and self._reranker is not None and candidates:
            candidates = self._reranker.rerank(question, candidates, top_n=n_results)
        else:
            candidates = candidates[:n_results]

        return candidates

    def ask(self, question: str, n_results: int = 5, topic: str = "") -> str:
        """Retrieve relevant chunks and generate an answer with DeepSeek.

        Pass topic to restrict retrieval to a single topic category.
        """
        from openai import OpenAI
        from utils.retry import call_with_retry

        chunks = self.retrieve(question, n_results, topic=topic)
        context = "\n\n---\n\n".join(
            f"[{c['metadata']['doc_name']} / {c['metadata']['section']}]\n{c['text']}"
            for c in chunks
        )

        client = OpenAI(
            api_key=os.environ.get("DEEPSEEK_API_KEY"),
            base_url=DEEPSEEK_BASE_URL,
        )
        response = call_with_retry(lambda: client.chat.completions.create(
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
        ))
        from rag.utils.token_tracker import get_active_tracker
        tracker = get_active_tracker()
        if tracker is not None:
            tracker.record(response.usage, label="retriever_ask")
        return response.choices[0].message.content
