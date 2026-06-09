"""Oil market specialist built on the general RAG agent.

RAGAgent owns reusable RAG capabilities: planning, retrieval, LLM generation,
and token tracking. This specialist owns the oil-market workflow: product
section definitions, category-scoped retrieval, and report assembly.

Per-section retrieval loop
--------------------------
For each section: retrieve → try to generate → if the LLM finds no relevant
content, ask it for a better query and retry. This mirrors the RAGAgent
evaluate loop but is scoped to a single report section, so a weak first pass
on (say) arbitrage can recover without affecting the other sections.
"""

import json
import re
from datetime import date

from rag.agent.service import RAGAgent
from rag.config import CONFIDENCE_THRESHOLD
from rag.config import MANIFEST_PATH
from rag.indexer.service import _parse_series


NO_RELEVANT_CONTENT = "NO_RELEVANT_CONTENT"
MAX_SECTION_RETRIES = 2
OIL_SOURCE_SERIES = (
    "Crude Oil Marketwire",
    "Asia-Pacific - Arab Gulf Marketscan",
    "Asia-Pacific Weekly Oil Recap",
)


SECTIONS: list[dict[str, str]] = [
    {
        "key": "crude",
        "title": "Crude Oil",
        "query": (
            "crude daily market analysis commentary benchmark assessments "
            "price changes Dubai Oman Murban Brent WTI ESPO"
        ),
        "instructions": (
            "Benchmark crudes including Brent, WTI, Dubai, Oman, ESPO, Murban, "
            "price levels, changes, direction, drivers, and benchmark spreads."
        ),
    },
    {
        "key": "heavy",
        "title": "Oil Products - Heavy (Fuel Oil, Bunker, VLSFO, HSFO)",
        "query": (
            "fuel oil daily market analysis bunker HSFO VLSFO 380cst 180cst "
            "crack premiums discounts demand"
        ),
        "instructions": (
            "Fuel oil and bunker markets, HSFO, VLSFO, LSFO, 380cst, 180cst, "
            "crack spreads, premiums or discounts, and demand trends."
        ),
    },
    {
        "key": "middle",
        "title": "Oil Products - Middle Distillates (Diesel, Gasoil, Jet Fuel, Kerosene)",
        "query": (
            "gasoil jet kerosene daily market analysis middle distillates "
            "diesel crack refinery margin demand"
        ),
        "instructions": (
            "Diesel, gasoil, jet fuel, kerosene, middle distillate price moves, "
            "crack spreads, margins, and demand trends."
        ),
    },
    {
        "key": "light",
        "title": "Oil Products - Light Ends (Naphtha, Gasoline, LPG)",
        "query": (
            "naphtha gasoline LPG daily market analysis light ends crack "
            "spreads premiums demand"
        ),
        "instructions": (
            "Naphtha, gasoline, mogas, LPG, propane, butane, light-end price "
            "moves, crack spreads, and demand trends."
        ),
    },
    {
        "key": "arb",
        "title": "Arbitrage & Spreads",
        "query": (
            "arbitrage spread East West EFS Asia Europe Atlantic Basin "
            "flows premiums discounts freight"
        ),
        "instructions": (
            "Open or closed arbitrage windows, East-West spreads, Asia-Europe "
            "or Atlantic Basin flows, premiums, discounts, and inter-product spreads."
        ),
    },
    {
        "key": "fundamentals",
        "title": "Supply & Demand Fundamentals",
        "query": (
            "supply demand inventory stocks draw build production consumption "
            "trade flows refinery runs OPEC"
        ),
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
        max_section_retries: int = MAX_SECTION_RETRIES,
    ):
        self.n_results = n_results
        self.max_section_retries = max_section_retries
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

    # ── Retrieval helpers ────────────────────────────────────────────────────

    def _query_category(self, query: str, plan: dict) -> list[dict]:
        return self.rag.retrieve(query, plan=plan, n_results=self.n_results)

    @staticmethod
    def _best_score(chunk: dict) -> float:
        """Best available relevance score across ce_score / rrf_score / score."""
        return chunk.get("ce_score", chunk.get("rrf_score", chunk.get("score", 0.0)))

    def _dedupe_chunks(self, chunks: list[dict]) -> list[dict]:
        seen: dict[str, dict] = {}
        for chunk in chunks:
            key = f"{chunk['metadata']['doc_name']}::{chunk['metadata']['section']}"
            if key not in seen or self._best_score(chunk) > self._best_score(seen[key]):
                seen[key] = chunk
        return sorted(seen.values(), key=self._best_score, reverse=True)

    @staticmethod
    def _format_context(chunks: list[dict]) -> str:
        return "\n\n---\n\n".join(
            f"[{c['metadata']['doc_name']} / {c['metadata']['section']}]\n{c['text']}"
            for c in chunks
        )

    @staticmethod
    def _as_of_date(question: str) -> str:
        match = re.search(r"\b(20\d{2}-\d{2}-\d{2})\b", question)
        if match:
            return match.group(1)
        return date.today().isoformat()

    @staticmethod
    def _source_for_doc(doc: dict) -> str:
        series_name = doc.get("series_name") or ""
        if series_name in OIL_SOURCE_SERIES:
            return series_name

        doc_name = doc.get("doc_name", "")
        parsed_series, _ = _parse_series(doc_name)
        if parsed_series in OIL_SOURCE_SERIES:
            return parsed_series

        topic = doc.get("topic", "")
        for source in OIL_SOURCE_SERIES:
            normalized_source = re.sub(r"[\s_-]+", " ", source).strip()
            if topic.startswith(normalized_source):
                return source

        return ""

    @staticmethod
    def _date_for_doc(doc: dict) -> str:
        if doc.get("series_date"):
            return doc["series_date"]
        _, parsed_date = _parse_series(doc.get("doc_name", ""))
        return parsed_date.isoformat() if parsed_date else ""

    def _oil_source_plan(self, question: str, base_plan: dict) -> dict:
        """Scope broad oil summaries to the latest issue from each oil source."""
        if not MANIFEST_PATH.exists():
            return base_plan

        manifest = json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))
        as_of = self._as_of_date(question)
        latest_by_source: dict[str, dict] = {}

        for doc in manifest.get("docs", []):
            source = self._source_for_doc(doc)
            if not source:
                continue

            series_date = self._date_for_doc(doc)
            if not series_date or series_date > as_of:
                continue

            current = latest_by_source.get(source)
            if current is None or series_date > self._date_for_doc(current):
                latest_by_source[source] = doc

        if not latest_by_source:
            return base_plan

        doc_filters = [
            latest_by_source[source]["doc_name"]
            for source in OIL_SOURCE_SERIES
            if source in latest_by_source
        ]
        print(
            f"  [oil-scope] sources={len(doc_filters)}  as_of={as_of}  "
            + ", ".join(doc_filters)
        )

        return {
            **base_plan,
            "topic_filter": "",
            "series_filter": "",
            "doc_filter": "",
            "doc_filters": doc_filters,
            "date_from": "",
            "date_to": as_of,
            "query": question,
        }

    # ── Generation helpers ───────────────────────────────────────────────────

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

    def _refine_section_query(
        self, section: dict[str, str], failed_query: str, question: str
    ) -> str:
        """Ask the LLM for an alternative search query for this section."""
        prompt = (
            f"You are helping search an oil market document database.\n"
            f"Report scope: {question}\n"
            f"Section needed: {section['title']}\n"
            f"Previous search query that returned no relevant data: \"{failed_query}\"\n\n"
            f"Suggest ONE alternative search query using different terminology that may "
            f"find relevant {section['title'].lower()} data. "
            "Return only the query string, no explanation."
        )
        return self.rag.generate(
            prompt,
            max_tokens=64,
            temperature=0.4,
            label=f"refine_{section['key']}",
        ).strip().strip("\"'")

    # ── Per-section retrieve→generate→refine loop ────────────────────────────

    def _run_section(
        self, section: dict[str, str], plan: dict, question: str
    ) -> str:
        """Retrieve and generate one section with query refinement on miss."""
        query = section["query"]

        for attempt in range(1, self.max_section_retries + 1):
            chunks = self._dedupe_chunks(self._query_category(query, plan))
            self.last_chunks_by_category[section["key"]] = chunks
            print(
                f"  [retrieve:{section['key']}] attempt={attempt}  "
                f"query='{query[:55]}'  chunks={len(chunks)}"
            )

            generated = self._generate_section(section, chunks, question)
            if generated:
                return generated

            if attempt < self.max_section_retries:
                refined = self._refine_section_query(section, query, question)
                if refined and refined != query:
                    print(f"  [refine:{section['key']}]   -> '{refined[:60]}'")
                    query = refined
                else:
                    break  # LLM couldn't improve on the query

        return ""

    # ── Public entry point ───────────────────────────────────────────────────

    def run(self, question: str) -> str:
        """Run the structured oil market summary pipeline and return markdown."""
        from rag.config import USAGE_LOG_PATH

        print(f"\nOil market summary: {question}\n")
        self.tracker.reset()
        self.last_chunks_by_category = {}
        self.last_sections = {}

        self.last_plan = self._oil_source_plan(question, self.rag.plan(question))

        for section in SECTIONS:
            generated = self._run_section(section, self.last_plan, question)
            if generated:
                self.last_sections[section["key"]] = generated
                print(f"  [generate:{section['key']}] length={len(generated)}")
            else:
                print(f"  [generate:{section['key']}] skipped — no relevant data")

        if not self.last_sections:
            return "No relevant documents found for this query."

        answer = "\n\n".join(
            self.last_sections[s["key"]]
            for s in SECTIONS
            if s["key"] in self.last_sections
        )

        print(f"\n  Total sections: {len(self.last_sections)}/{len(SECTIONS)}  "
              f"answer length={len(answer)}")
        self.tracker.print_summary()
        self.tracker.save_jsonl(
            USAGE_LOG_PATH,
            {"agent": "OilMarketSummaryAgent", "question": question},
        )
        return answer
