"""General-purpose agentic RAG engine using LangGraph.

RAGAgent owns reusable RAG capabilities:
- manifest-aware planning via rag.planning
- scoped retrieval via RAGRetriever
- LLM generation with token tracking
- optional self-evaluation loop for general Q&A
"""

import json
import os
from typing import TypedDict

from langgraph.graph import END, START, StateGraph

from rag.config import CONFIDENCE_THRESHOLD, DEEPSEEK_BASE_URL, DEEPSEEK_MODEL, USAGE_LOG_PATH
from rag.planning import build_where, plan_retrieval_scope
from rag.query.service import RAGRetriever


class AgentState(TypedDict):
    question: str
    query: str
    topic_filter: str
    series_filter: str
    doc_filter: str
    doc_filters: list[str]
    date_from: str
    date_to: str
    chunks: list[dict]
    context: str
    draft_answer: str
    complete: bool
    missing: str
    final_answer: str
    iteration: int


def _deepseek(
    messages: list[dict],
    max_tokens: int = 512,
    temperature: float = 0,
    label: str = "",
) -> str:
    from openai import OpenAI
    from rag.utils.token_tracker import get_active_tracker
    from utils.retry import call_with_retry

    client = OpenAI(api_key=os.environ.get("DEEPSEEK_API_KEY"), base_url=DEEPSEEK_BASE_URL)
    resp = call_with_retry(lambda: client.chat.completions.create(
        model=DEEPSEEK_MODEL,
        max_tokens=max_tokens,
        temperature=temperature,
        messages=messages,
    ))
    tracker = get_active_tracker()
    if tracker is not None:
        tracker.record(resp.usage, label=label)
    return resp.choices[0].message.content.strip()


def _json(text: str) -> dict:
    """Parse JSON from LLM output, stripping markdown fences if present."""
    text = text.strip()
    if text.startswith("```"):
        text = "\n".join(text.split("\n")[1:])
        text = text.rstrip("`").strip()
    return json.loads(text)


def _plan(state: AgentState) -> dict:
    """LangGraph adapter for shared retrieval planning."""
    return plan_retrieval_scope(state["question"], _deepseek)


def _retrieve(state: AgentState, retriever: RAGRetriever, n_results: int) -> dict:
    where = build_where(state)
    chunks = retriever.retrieve(state["query"], n_results=n_results, where=where)
    context = "\n\n---\n\n".join(
        f"[{c['metadata']['doc_name']} / {c['metadata']['section']}]\n{c['text']}"
        for c in chunks
    )
    print(
        f"  [retrieve] iteration={state['iteration'] + 1}  "
        f"query='{state['query'][:60]}'  chunks={len(chunks)}"
    )
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
        label="generate",
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
        label="evaluate",
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

    print(
        f"  [evaluate] complete={complete}"
        + (f"  missing='{missing[:60]}'" if not complete else "")
    )

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


class RAGAgent:
    """Reusable RAG engine plus a general Q&A graph."""

    def __init__(
        self,
        n_results: int = 5,
        max_iterations: int = 3,
        min_score: float = CONFIDENCE_THRESHOLD,
        use_reranker: bool = True,
        use_bm25: bool = True,
    ):
        self.n_results = n_results
        self.max_iterations = max_iterations
        self._retriever = RAGRetriever(
            min_score=min_score,
            use_reranker=use_reranker,
            use_bm25=use_bm25,
        )
        self._graph = self._build_graph()
        from rag.utils.token_tracker import UsageTracker
        self.tracker = UsageTracker()

    def _initial_state(self, question: str) -> AgentState:
        """Create the default graph/planning state for a question."""
        return {
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

    def plan(self, question: str) -> dict:
        """Return a manifest-aware retrieval plan for a question."""
        from rag.utils.token_tracker import set_active_tracker
        with set_active_tracker(self.tracker):
            return plan_retrieval_scope(question, _deepseek)

    def retrieve(
        self,
        query: str,
        plan: dict | None = None,
        n_results: int | None = None,
    ) -> list[dict]:
        """Retrieve chunks for a query, optionally scoped by a retrieval plan."""
        where = build_where(plan) if plan else None
        return self._retriever.retrieve(
            query,
            n_results=n_results or self.n_results,
            where=where,
        )

    def generate(
        self,
        prompt: str,
        max_tokens: int = 1024,
        temperature: float = 0.2,
        label: str = "generate",
    ) -> str:
        """Generate text from a fully formed prompt using the shared LLM client."""
        from rag.utils.token_tracker import set_active_tracker
        with set_active_tracker(self.tracker):
            return _deepseek(
                messages=[{"role": "user", "content": prompt}],
                max_tokens=max_tokens,
                temperature=temperature,
                label=label,
            )

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
        """Run the full general Q&A loop and return the final answer."""
        from rag.utils.token_tracker import set_active_tracker

        print(f"\nAgent question: {question}\n")
        self.tracker.reset()
        with set_active_tracker(self.tracker):
            result = self._graph.invoke(self._initial_state(question))
        answer = result["final_answer"] or result["draft_answer"]
        self.tracker.print_summary()
        self.tracker.save_jsonl(USAGE_LOG_PATH, {"agent": "RAGAgent", "question": question})
        return answer

    def answer(self, question: str) -> str:
        """Alias for run(), for callers that treat RAGAgent as a QA engine."""
        return self.run(question)
