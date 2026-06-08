"""
Agentic RAG using LangGraph.

Graph
-----
START → plan → retrieve → generate → evaluate ──► END
                  ▲                      │
                  └──── (if incomplete) ──┘

Nodes
-----
plan      Reads the manifest and asks DeepSeek whether the question targets a
          specific document series. Sets series_filter and the initial query.

retrieve  Queries ChromaDB, optionally filtered by series_name or doc_name
          (for "latest in series" queries).

generate  Calls DeepSeek to produce a draft answer from the retrieved chunks.

evaluate  Asks DeepSeek to score completeness. If incomplete, provides a
          refined query for the next retrieval round.
"""

import json
import os
from typing import TypedDict

from langgraph.graph import END, START, StateGraph

from rag.config import CHROMA_PATH, DEEPSEEK_BASE_URL, DEEPSEEK_MODEL, MANIFEST_PATH
from rag.query.service import RAGRetriever

# ---------------------------------------------------------------------------
# Agent state
# ---------------------------------------------------------------------------

class AgentState(TypedDict):
    question: str
    query: str               # current retrieval query (refined each loop)
    topic_filter: str        # topic label to filter on, or ""
    series_filter: str       # series_name to filter on (within a topic), or ""
    doc_filter: str          # specific doc_name to filter on, or ""
    chunks: list[dict]
    context: str             # formatted context string
    draft_answer: str
    complete: bool
    missing: str             # what is still missing, if not complete
    final_answer: str
    iteration: int


# ---------------------------------------------------------------------------
# LLM helper
# ---------------------------------------------------------------------------

def _deepseek(messages: list[dict], max_tokens: int = 512, temperature: float = 0) -> str:
    from openai import OpenAI
    client = OpenAI(api_key=os.environ.get("DEEPSEEK_API_KEY"), base_url=DEEPSEEK_BASE_URL)
    resp = client.chat.completions.create(
        model=DEEPSEEK_MODEL,
        max_tokens=max_tokens,
        temperature=temperature,
        messages=messages,
    )
    return resp.choices[0].message.content.strip()


def _json(text: str) -> dict:
    """Parse JSON from LLM output, stripping markdown fences if present."""
    text = text.strip()
    if text.startswith("```"):
        text = "\n".join(text.split("\n")[1:])
        text = text.rstrip("`").strip()
    return json.loads(text)


# ---------------------------------------------------------------------------
# Node implementations
# ---------------------------------------------------------------------------

def _plan(state: AgentState) -> dict:
    """Identify whether the question targets a specific topic and/or series."""
    _empty = {"topic_filter": "", "series_filter": "", "doc_filter": "", "query": state["question"]}
    if not MANIFEST_PATH.exists():
        return _empty

    manifest = json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))
    docs = manifest.get("docs", [])
    if not docs:
        return _empty

    # Build topic → docs mapping and topic → series_name mapping
    topic_docs: dict[str, list[dict]] = {}
    topic_series: dict[str, str] = {}
    for d in docs:
        topic = d.get("topic", "")
        if not topic:
            continue
        topic_docs.setdefault(topic, []).append(d)
        sn = d.get("series_name", "")
        if sn:
            topic_series[topic] = sn

    if not topic_docs:
        return _empty

    # Build a concise description of each topic for the LLM
    topic_lines = []
    for t in sorted(topic_docs.keys()):
        tdocs = topic_docs[t]
        dated = [d["series_date"] for d in tdocs if d.get("series_date")]
        if dated:
            topic_lines.append(f"  - {t}  ({len(tdocs)} reports, latest: {max(dated)})")
        else:
            topic_lines.append(f"  - {t}  (standalone)")

    prompt = (
        "Available document topics:\n"
        + "\n".join(topic_lines)
        + f"\n\nQuestion: {state['question']}\n\n"
        "Does this question target one of the topics above? "
        "If yes, also say whether the user wants the LATEST report or all reports in that topic. "
        'Respond with JSON only: {"topic": "<exact topic name or empty string>", "latest_only": true/false}'
    )

    try:
        result = _json(_deepseek([{"role": "user", "content": prompt}]))
        topic = result.get("topic", "").strip()
        latest_only = result.get("latest_only", False)

        series_filter = topic_series.get(topic, "") if topic else ""
        doc_filter = ""

        if topic and latest_only and topic in topic_docs:
            candidates = [d for d in topic_docs[topic] if d.get("series_date")]
            if candidates:
                newest = max(candidates, key=lambda d: d.get("series_date") or "")
                doc_filter = newest["doc_name"]
                series_filter = ""  # doc_filter is more specific

        print(f"  [plan] topic={topic or 'all'}  latest_only={latest_only}"
              + (f"  → doc_filter={doc_filter[:50]}" if doc_filter else
                 f"  → series_filter={series_filter}" if series_filter else ""))
        return {"topic_filter": topic, "series_filter": series_filter, "doc_filter": doc_filter, "query": state["question"]}
    except Exception:
        return _empty


def _retrieve(state: AgentState, retriever: RAGRetriever, n_results: int) -> dict:
    where: dict | None = None
    if state["doc_filter"]:
        where = {"doc_name": {"$eq": state["doc_filter"]}}
    elif state["series_filter"]:
        where = {"series_name": {"$eq": state["series_filter"]}}
    elif state["topic_filter"]:
        where = {"topic": {"$eq": state["topic_filter"]}}

    try:
        results = retriever.collection.query(
            query_texts=[state["query"]],
            n_results=n_results,
            where=where,
            include=["documents", "metadatas", "distances"],
        )
    except Exception:
        # Filter matched nothing — fall back to unfiltered search
        print(f"  [retrieve] filter matched nothing, falling back to unfiltered search")
        results = retriever.collection.query(
            query_texts=[state["query"]],
            n_results=n_results,
            include=["documents", "metadatas", "distances"],
        )
    chunks = [
        {"text": doc, "metadata": meta, "score": round(1 - dist, 4)}
        for doc, meta, dist in zip(
            results["documents"][0],
            results["metadatas"][0],
            results["distances"][0],
        )
    ]
    context = "\n\n---\n\n".join(
        f"[{c['metadata']['doc_name']} / {c['metadata']['section']}]\n{c['text']}"
        for c in chunks
    )
    print(f"  [retrieve] iteration={state['iteration']+1}  query='{state['query'][:60]}'  chunks={len(chunks)}")
    return {"chunks": chunks, "context": context, "iteration": state["iteration"] + 1}


def _generate(state: AgentState) -> dict:
    answer = _deepseek(
        messages=[{
            "role": "user",
            "content": (
                "Answer the question using only the provided context. "
                "Cite the document and section for each key claim.\n\n"
                f"Context:\n{state['context']}\n\n"
                f"Question: {state['question']}"
            ),
        }],
        max_tokens=1024,
        temperature=0.2,
    )
    print(f"  [generate] answer length={len(answer)}")
    return {"draft_answer": answer}


def _evaluate(state: AgentState) -> dict:
    result = _deepseek(
        messages=[{
            "role": "user",
            "content": (
                "You are a RAG self-evaluation judge.\n\n"
                f"Question: {state['question']}\n\n"
                f"Answer: {state['draft_answer']}\n\n"
                "Is the answer complete and fully grounded in the context? "
                "If not, what specific information is missing, and what refined search query "
                "would retrieve it?\n\n"
                "Respond with JSON only:\n"
                '{"complete": true/false, "missing": "<what is missing or empty>", '
                '"refined_query": "<better query or empty>"}'
            ),
        }],
        max_tokens=256,
    )
    try:
        parsed = _json(result)
        complete = bool(parsed.get("complete", False))
        missing = parsed.get("missing", "")
        refined = parsed.get("refined_query", "")
    except Exception:
        complete = True
        missing = ""
        refined = ""

    print(f"  [evaluate] complete={complete}" + (f"  missing='{missing[:60]}'" if not complete else ""))

    next_query = refined if (refined and not complete) else state["query"]
    final = state["draft_answer"] if complete else state["final_answer"]
    return {
        "complete": complete,
        "missing": missing,
        "query": next_query,
        "final_answer": final,
    }


def _should_continue(state: AgentState, max_iterations: int) -> str:
    if state["complete"] or state["iteration"] >= max_iterations:
        return "end"
    return "continue"


# ---------------------------------------------------------------------------
# Public agent class
# ---------------------------------------------------------------------------

class RAGAgent:
    def __init__(self, n_results: int = 5, max_iterations: int = 3):
        self.n_results = n_results
        self.max_iterations = max_iterations
        self._retriever = RAGRetriever()
        self._graph = self._build_graph()

    def _build_graph(self):
        n = self.n_results
        mi = self.max_iterations
        retriever = self._retriever

        builder = StateGraph(AgentState)
        builder.add_node("plan", _plan)
        builder.add_node("retrieve", lambda s: _retrieve(s, retriever, n))
        builder.add_node("generate", _generate)
        builder.add_node("evaluate", _evaluate)

        builder.add_edge(START, "plan")
        builder.add_edge("plan", "retrieve")
        builder.add_edge("retrieve", "generate")
        builder.add_edge("generate", "evaluate")
        builder.add_conditional_edges(
            "evaluate",
            lambda s: _should_continue(s, mi),
            {"continue": "retrieve", "end": END},
        )
        return builder.compile()

    def run(self, question: str) -> str:
        """Run the full agent loop and return the final answer."""
        print(f"\nAgent question: {question}\n")
        result = self._graph.invoke({
            "question": question,
            "query": question,
            "topic_filter": "",
            "series_filter": "",
            "doc_filter": "",
            "chunks": [],
            "context": "",
            "draft_answer": "",
            "complete": False,
            "missing": "",
            "final_answer": "",
            "iteration": 0,
        })
        # If loop exited on max_iterations without completing, use the last draft
        return result["final_answer"] or result["draft_answer"]
