"""Oil market specialist built on the general RAG agent.

RAGAgent owns reusable RAG capabilities: planning, retrieval, LLM generation,
and token tracking. This specialist owns the oil-market workflow: product
section definitions, category-scoped retrieval, and report assembly.
"""

from rag.agent.service import RAGAgent
from rag.config import CONFIDENCE_THRESHOLD


NO_RELEVANT_CONTENT = "NO_RELEVANT_CONTENT"


SECTIONS: list[dict[str, str]] = [
    {
        "key": "crude",
        "title": "Crude Oil",
        "query": "crude oil benchmark price Brent WTI Dubai Oman ESPO Murban change direction",
        "instructions": (
            "Benchmark crudes including Brent, WTI, Dubai, Oman, ESPO, Murban, "
            "price levels, changes, direction, drivers, and benchmark spreads."
        ),
    },
    {
        "key": "heavy",
        "title": "Oil Products - Heavy (Fuel Oil, Bunker, VLSFO, HSFO)",
        "query": "fuel oil HSFO VLSFO LSFO bunker 380cst 180cst heavy products price crack spread",
        "instructions": (
            "Fuel oil and bunker markets, HSFO, VLSFO, LSFO, 380cst, 180cst, "
            "crack spreads, premiums or discounts, and demand trends."
        ),
    },
    {
        "key": "middle",
        "title": "Oil Products - Middle Distillates (Diesel, Gasoil, Jet Fuel, Kerosene)",
        "query": "diesel gasoil jet fuel kerosene middle distillates GO crack spread refinery margin",
        "instructions": (
            "Diesel, gasoil, jet fuel, kerosene, middle distillate price moves, "
            "crack spreads, margins, and demand trends."
        ),
    },
    {
        "key": "light",
        "title": "Oil Products - Light Ends (Naphtha, Gasoline, LPG)",
        "query": "naphtha gasoline mogas unleaded LPG propane butane light ends crack spread",
        "instructions": (
            "Naphtha, gasoline, mogas, LPG, propane, butane, light-end price "
            "moves, crack spreads, and demand trends."
        ),
    },
    {
        "key": "arb",
        "title": "Arbitrage & Spreads",
        "query": "arbitrage arb spread East West EW Asia Europe Atlantic Basin premium discount pairs",
        "instructions": (
            "Open or closed arbitrage windows, East-West spreads, Asia-Europe "
            "or Atlantic Basin flows, premiums, discounts, and inter-product spreads."
        ),
    },
    {
        "key": "fundamentals",
        "title": "Supply & Demand Fundamentals",
        "query": "supply demand inventory stock draw build production consumption trade flows OPEC",
        "instructions": (
            "Supply, demand, inventories, stock builds or draws, production, "
            "consumption, trade flows, OPEC decisions, and refinery run-rates."
        ),
    },
]


_SECTION_PROMPT = """\
You are a senior oil market analyst. Write ONLY the "{title}" section of an
oil market report using ONLY the context below.

Focus:
{instructions}

Rules:
- If the context has no material information for this section, return exactly:
  {no_relevant_content}
- Otherwise, start with this exact markdown heading:
  ## {title}
- Use concise bullets.
- Cite [document / section] for each material claim.
- Do not use information that is not in the context.

Context:
{context}

Question / scope: {question}"""


class OilMarketSummaryAgent:
    """Structured oil market summary using RAGAgent as its engine."""

    def __init__(
        self,
        n_results: int = 5,
        min_score: float = CONFIDENCE_THRESHOLD,
        use_reranker: bool = True,
        use_bm25: bool = True,
    ):
        self.n_results = n_results
        self.rag = RAGAgent(
            n_results=n_results,
            min_score=min_score,
            use_reranker=use_reranker,
            use_bm25=use_bm25,
        )
        self.tracker = self.rag.tracker
        self.last_plan: dict | None = None
        self.last_chunks_by_category: dict[str, list[dict]] = {}
        self.last_sections: dict[str, str] = {}

    def _query_category(self, query: str, plan: dict) -> list[dict]:
        return self.rag.retrieve(query, plan=plan, n_results=self.n_results)

    @staticmethod
    def _dedupe_chunks(chunks: list[dict]) -> list[dict]:
        seen: dict[str, dict] = {}
        for chunk in chunks:
            key = f"{chunk['metadata']['doc_name']}::{chunk['metadata']['section']}"
            if key not in seen or chunk["score"] > seen[key]["score"]:
                seen[key] = chunk
        return sorted(seen.values(), key=lambda c: c["score"], reverse=True)

    @staticmethod
    def _format_context(chunks: list[dict]) -> str:
        return "\n\n---\n\n".join(
            f"[{c['metadata']['doc_name']} / {c['metadata']['section']}]\n{c['text']}"
            for c in chunks
        )

    def _generate_section(self, section: dict[str, str], chunks: list[dict], question: str) -> str:
        if not chunks:
            return ""

        prompt = _SECTION_PROMPT.format(
            title=section["title"],
            instructions=section["instructions"],
            no_relevant_content=NO_RELEVANT_CONTENT,
            context=self._format_context(chunks),
            question=question,
        )
        text = self.rag.generate(
            prompt,
            max_tokens=768,
            temperature=0.2,
            label=f"generate_{section['key']}",
        ).strip()
        if text == NO_RELEVANT_CONTENT:
            return ""
        return text

    def run(self, question: str) -> str:
        """Run the structured oil market summary pipeline and return markdown."""
        from rag.config import USAGE_LOG_PATH

        print(f"\nOil market summary: {question}\n")
        self.tracker.reset()
        self.last_chunks_by_category = {}
        self.last_sections = {}

        self.last_plan = self.rag.plan(question)

        for section in SECTIONS:
            print(f"  [retrieve:{section['key']}]  '{section['query'][:60]}'")
            chunks = self._dedupe_chunks(self._query_category(section["query"], self.last_plan))
            self.last_chunks_by_category[section["key"]] = chunks
            print(f"    -> {len(chunks)} chunk(s)")

        for section in SECTIONS:
            chunks = self.last_chunks_by_category.get(section["key"], [])
            generated = self._generate_section(section, chunks, question)
            if generated:
                self.last_sections[section["key"]] = generated
                print(f"  [generate:{section['key']}] length={len(generated)}")
            else:
                print(f"  [generate:{section['key']}] skipped")

        if not self.last_sections:
            return "No relevant documents found for this query."

        answer = "\n\n".join(
            self.last_sections[section["key"]]
            for section in SECTIONS
            if section["key"] in self.last_sections
        )

        print(f"  [generate] final answer length={len(answer)}")
        self.tracker.print_summary()
        self.tracker.save_jsonl(
            USAGE_LOG_PATH,
            {"agent": "OilMarketSummaryAgent", "question": question},
        )
        return answer
