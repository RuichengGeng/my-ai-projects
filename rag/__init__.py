"""
RAG — Retrieval-Augmented Generation module.

This package is designed to be used independently by any agent in this project.

Quick start
-----------
    from rag import RAGAgent, RAGRetriever, MarkdownRAGRetriever, OilMarketSummaryAgent

    # General Q&A agent with topic/date-aware planning and self-evaluation loop
    agent = RAGAgent()
    answer = agent.run("What is the latest APAC oil update?")

    # Structured oil market summary (crude / products / arb / fundamentals)
    oil_agent = OilMarketSummaryAgent()
    report = oil_agent.run("Summarise last week's APAC oil market")

    # Direct retrieval without the agent loop
    retriever = RAGRetriever()
    chunks = retriever.retrieve("roll yield futures", n_results=5)
    answer  = retriever.ask("What is roll yield?")

    md_retriever = MarkdownRAGRetriever()
    chunks = md_retriever.retrieve("Murban Dubai spread", n_results=5)

    # Index documents from pdf_parser output
    from rag import build_index, list_docs
    build_index()
    list_docs()

Token tracking
--------------
Every agent exposes a .tracker attribute (UsageTracker) after each run().
For use outside this package:

    from utils.token_tracker import UsageTracker, set_active_tracker
"""

from rag.agent.service import RAGAgent
from rag.eval.service import EvalService
from rag.indexer.service import build_index, delete_doc, list_docs, reindex_doc
from rag.query.md_service import MarkdownRAGRetriever
from rag.query.service import RAGRetriever
from rag.specialists.oil_market import OilMarketSummaryAgent

__all__ = [
    "RAGAgent",
    "OilMarketSummaryAgent",
    "RAGRetriever",
    "MarkdownRAGRetriever",
    "EvalService",
    "build_index",
    "list_docs",
    "delete_doc",
    "reindex_doc",
]
