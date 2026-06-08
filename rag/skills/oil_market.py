"""
Oil market domain skill: structured RAG summary with product categorisation.

Instead of a single free-form query, this agent runs six targeted sub-queries
(one per report category), deduplicates the retrieved chunks, then generates a
structured markdown report with fixed sections:

  Crude Oil
  Oil Products – Heavy (Fuel Oil / Bunker)
  Oil Products – Middle Distillates (Diesel / Gasoil / Jet)
  Oil Products – Light Ends (Naphtha / Gasoline / LPG)
  Arbitrage & Spreads
  Supply & Demand Fundamentals

Topic / date scoping is handled by the same _plan() node used in the general
RAG agent, so "last week's APAC update" / "latest report" / "all reports" all
work identically.
"""

import os

from rag.agent.service import AgentState, _deepseek, _json, _plan
from rag.query.service import RAGRetriever

# ---------------------------------------------------------------------------
# Sub-queries — one per report category
# ---------------------------------------------------------------------------

# Each query is tuned to surface chunks relevant to that section even when the
# document uses different terminology (e.g. "380cst" instead of "HSFO").
SUB_QUERIES: dict[str, str] = {
    "crude":        "crude oil benchmark price Brent WTI Dubai Oman ESPO Murban change direction",
    "heavy":        "fuel oil HSFO VLSFO LSFO bunker 380cst 180cst heavy products price crack spread",
    "middle":       "diesel gasoil jet fuel kerosene middle distillates GO crack spread refinery margin",
    "light":        "naphtha gasoline mogas unleaded LPG propane butane light ends crack spread",
    "arb":          "arbitrage arb spread East West EW Asia Europe Atlantic Basin premium discount pairs",
    "fundamentals": "supply demand inventory stock draw build production consumption trade flows OPEC",
}

# ---------------------------------------------------------------------------
# Structured generation prompt
# ---------------------------------------------------------------------------

_STRUCTURED_PROMPT = """\
You are a senior oil market analyst. Using ONLY the context provided, write a \
structured weekly market summary in markdown.

Rules:
- Include a section ONLY when the context contains relevant information for it.
- Omit sections entirely when no relevant context exists — do not fabricate.
- For each material claim, cite [document / section].
- Be concise: bullet points preferred over long paragraphs.

Use EXACTLY these section headers (in this order):

## Crude Oil
Benchmark crudes (Brent, WTI, Dubai, Oman, ESPO, etc.): price level, change \
vs prior period, direction, and key drivers. Note spreads between benchmarks.

## Oil Products — Heavy (Fuel Oil, Bunker, VLSFO, HSFO)
Price changes, crack spreads, regional premium/discount, demand trends.

## Oil Products — Middle Distillates (Diesel, Gasoil, Jet Fuel, Kerosene)
Price changes, crack spreads, demand trends.

## Oil Products — Light Ends (Naphtha, Gasoline, LPG)
Price changes, crack spreads, demand trends.

## Arbitrage & Spreads
Open or closed arb windows between regions; key pair prices and inter-regional \
or inter-product spreads (e.g. East-West, Asia-Europe, Atlantic Basin).

## Supply & Demand Fundamentals
Production, consumption, inventory builds/draws, trade flows, OPEC+ decisions, \
refinery run-rates.

---
Context:
{context}

---
Question / scope: {question}"""


# ---------------------------------------------------------------------------
# Agent
# ---------------------------------------------------------------------------

class OilMarketSummaryAgent:
    """Structured oil market summary using multi-category retrieval."""

    def __init__(self, n_results: int = 5):
        self.n_results = n_results
        self._retriever = RAGRetriever()
        from rag.utils.token_tracker import UsageTracker
        self.tracker = UsageTracker()

    # ── ChromaDB filter helpers ──────────────────────────────────────────────

    def _build_where(self, plan: dict) -> dict | None:
        """Convert plan output into a ChromaDB where-filter."""
        if plan.get("doc_filter"):
            return {"doc_name": {"$eq": plan["doc_filter"]}}
        if plan.get("doc_filters"):
            df = plan["doc_filters"]
            return {"doc_name": {"$eq": df[0]}} if len(df) == 1 else {"doc_name": {"$in": df}}
        if plan.get("series_filter"):
            return {"series_name": {"$eq": plan["series_filter"]}}
        if plan.get("topic_filter"):
            return {"topic": {"$eq": plan["topic_filter"]}}
        return None

    def _query_category(self, subquery: str, where: dict | None) -> list[dict]:
        try:
            results = self._retriever.collection.query(
                query_texts=[subquery],
                n_results=self.n_results,
                where=where,
                include=["documents", "metadatas", "distances"],
            )
        except Exception:
            print(f"    [warn] filter matched nothing, retrying unfiltered")
            results = self._retriever.collection.query(
                query_texts=[subquery],
                n_results=self.n_results,
                include=["documents", "metadatas", "distances"],
            )
        return [
            {"text": doc, "metadata": meta, "score": round(1 - dist, 4)}
            for doc, meta, dist in zip(
                results["documents"][0],
                results["metadatas"][0],
                results["distances"][0],
            )
        ]

    # ── Public entry point ───────────────────────────────────────────────────

    def run(self, question: str) -> str:
        """Run the structured oil market summary pipeline and return markdown."""
        from rag.config import USAGE_LOG_PATH
        from rag.utils.token_tracker import set_active_tracker
        print(f"\nOil market summary: {question}\n")
        self.tracker.reset()

        # Step 1: detect topic / date scope using the shared plan node
        init_state: AgentState = {
            "question": question,
            "query": question,
            "topic_filter": "",
            "series_filter": "",
            "doc_filter": "",
            "doc_filters": [],
            "date_from": "",
            "date_to": "",
            "chunks": [],
            "context": "",
            "draft_answer": "",
            "complete": False,
            "missing": "",
            "final_answer": "",
            "iteration": 0,
        }
        with set_active_tracker(self.tracker):
            plan = _plan(init_state)
            where = self._build_where(plan)

            # Step 2: multi-category retrieval — deduplicate by (doc, section),
            # keeping the highest-scoring copy of any repeated chunk
            seen: dict[str, dict] = {}
            for category, subquery in SUB_QUERIES.items():
                print(f"  [retrieve:{category}]  '{subquery[:60]}'")
                for chunk in self._query_category(subquery, where):
                    key = f"{chunk['metadata']['doc_name']}::{chunk['metadata']['section']}"
                    if key not in seen or chunk["score"] > seen[key]["score"]:
                        seen[key] = chunk

            chunks = sorted(seen.values(), key=lambda c: c["score"], reverse=True)
            print(f"  [retrieve] {len(chunks)} unique chunks from {len(SUB_QUERIES)} sub-queries")

            if not chunks:
                return "No relevant documents found for this query."

            context = "\n\n---\n\n".join(
                f"[{c['metadata']['doc_name']} / {c['metadata']['section']}]\n{c['text']}"
                for c in chunks
            )

            # Step 3: structured generation
            answer = _deepseek(
                messages=[{
                    "role": "user",
                    "content": _STRUCTURED_PROMPT.format(context=context, question=question),
                }],
                max_tokens=2048,
                temperature=0.2,
                label="generate",
            )

        print(f"  [generate] answer length={len(answer)}")
        self.tracker.print_summary()
        self.tracker.save_jsonl(
            USAGE_LOG_PATH,
            {"agent": "OilMarketSummaryAgent", "question": question},
        )
        return answer
