"""Query the ChromaDB index and generate answers with DeepSeek."""

import os

import chromadb
from chromadb.utils.embedding_functions import SentenceTransformerEmbeddingFunction

from rag.config import CHROMA_PATH, COLLECTION_NAME, DEEPSEEK_BASE_URL, DEEPSEEK_MODEL, EMBED_MODEL


class RAGRetriever:
    def __init__(self):
        ef = SentenceTransformerEmbeddingFunction(model_name=EMBED_MODEL)
        client = chromadb.PersistentClient(path=str(CHROMA_PATH))
        self.collection = client.get_collection(
            name=COLLECTION_NAME,
            embedding_function=ef,
        )

    def retrieve(self, question: str, n_results: int = 5) -> list[dict]:
        """Return the top-n most relevant chunks for a question.

        Each result dict has:
          text      – chunk content
          metadata  – doc_name, section, source, images
          score     – cosine similarity (1 = identical, 0 = orthogonal)
        """
        results = self.collection.query(
            query_texts=[question],
            n_results=n_results,
            include=["documents", "metadatas", "distances"],
        )
        return [
            {
                "text": doc,
                "metadata": meta,
                "score": round(1 - dist, 4),
            }
            for doc, meta, dist in zip(
                results["documents"][0],
                results["metadatas"][0],
                results["distances"][0],
            )
        ]

    def ask(self, question: str, n_results: int = 5) -> str:
        """Retrieve relevant chunks and generate an answer with DeepSeek."""
        from openai import OpenAI

        chunks = self.retrieve(question, n_results)
        context = "\n\n---\n\n".join(
            f"[{c['metadata']['doc_name']} / {c['metadata']['section']}]\n{c['text']}"
            for c in chunks
        )

        client = OpenAI(
            api_key=os.environ.get("DEEPSEEK_API_KEY"),
            base_url=DEEPSEEK_BASE_URL,
        )
        response = client.chat.completions.create(
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
        return response.choices[0].message.content
