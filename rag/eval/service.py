"""RAG evaluation service using LLM-as-judge (DeepSeek).

Metrics per question
--------------------
faithfulness   (1-5): answer contains only information inferable from retrieved context.
relevance      (1-5): answer directly addresses the question.
completeness   (1-5): answer covers all key aspects the question asks about.
context_precision (0-1): fraction of retrieved chunks that are actually useful for the question.
latency_ms     (int): end-to-end time from question to answer.

A JSON report is written to rag_store/eval_report_<timestamp>.json.
The summary breaks scores down by question_type and topic.
"""

import json
import os
import time
from datetime import datetime, timezone
from pathlib import Path

from rag.config import CHROMA_PATH, DEEPSEEK_BASE_URL, DEEPSEEK_MODEL, USAGE_LOG_PATH
from rag.query.service import RAGRetriever

# ---------------------------------------------------------------------------
# Judge prompts
# ---------------------------------------------------------------------------

_ANSWER_JUDGE_PROMPT = """\
You are a RAG evaluation judge. Score the answer on three dimensions.

Faithfulness (1-5): Does the answer contain ONLY information inferable from the context?
  5 = fully grounded, no hallucinations.  1 = clear fabrications.

Relevance (1-5): Does the answer directly address the question?
  5 = complete and on-point.  1 = off-topic or evasive.

Completeness (1-5): Does the answer cover all key aspects the question asks about?
  5 = nothing important is left out.  1 = only a small part of the question is addressed.

Retrieved context:
{context}

Question: {question}

Answer: {answer}

Respond with JSON only, no markdown fences:
{{"faithfulness": <1-5>, "relevance": <1-5>, "completeness": <1-5>, "reasoning": "<one sentence>"}}"""

_CONTEXT_PRECISION_PROMPT = """\
Given the question below, decide for each numbered chunk whether it contains \
information that is useful for answering the question.

Question: {question}

{chunks}

Respond with JSON only, no markdown fences:
{{"chunk_relevance": [<true/false per chunk in order>]}}"""


# ---------------------------------------------------------------------------
# LLM helper
# ---------------------------------------------------------------------------

def _deepseek(content: str, max_tokens: int = 256, label: str = "") -> str:
    from openai import OpenAI
    from rag.utils.token_tracker import get_active_tracker
    client = OpenAI(api_key=os.environ.get("DEEPSEEK_API_KEY"), base_url=DEEPSEEK_BASE_URL)
    resp = client.chat.completions.create(
        model=DEEPSEEK_MODEL,
        max_tokens=max_tokens,
        temperature=0,
        messages=[{"role": "user", "content": content}],
    )
    tracker = get_active_tracker()
    if tracker is not None:
        tracker.record(resp.usage, label=label)
    return resp.choices[0].message.content.strip()


def _parse_json(raw: str) -> dict:
    raw = raw.strip()
    if raw.startswith("```"):
        raw = "\n".join(raw.split("\n")[1:]).rstrip("`").strip()
    return json.loads(raw)


# ---------------------------------------------------------------------------
# EvalService
# ---------------------------------------------------------------------------

class EvalService:
    def __init__(self, n_results: int = 5):
        self.retriever = RAGRetriever()
        self.n_results = n_results
        self._report_dir = CHROMA_PATH
        from rag.utils.token_tracker import UsageTracker
        self.tracker = UsageTracker()

    def _judge_answer(self, question: str, context: str, answer: str) -> dict:
        raw = _deepseek(
            _ANSWER_JUDGE_PROMPT.format(context=context, question=question, answer=answer),
            max_tokens=256,
            label="judge_answer",
        )
        try:
            return _parse_json(raw)
        except Exception:
            return {"faithfulness": None, "relevance": None, "completeness": None, "reasoning": raw}

    def _judge_context_precision(self, question: str, chunks: list[dict]) -> float | None:
        """Ask the LLM which retrieved chunks are actually useful. Returns fraction 0-1."""
        numbered = "\n\n".join(
            f"Chunk {i+1}:\n{c['text'][:400]}"
            for i, c in enumerate(chunks)
        )
        raw = _deepseek(
            _CONTEXT_PRECISION_PROMPT.format(question=question, chunks=numbered),
            max_tokens=128,
            label="judge_context_precision",
        )
        try:
            relevances: list[bool] = _parse_json(raw)["chunk_relevance"]
            return round(sum(relevances) / len(relevances), 3) if relevances else None
        except Exception:
            return None

    def _eval_one(self, item: dict) -> dict:
        question = item["question"]
        topic = item.get("topic", "")
        expected_keywords = item.get("expected_keywords", [])

        t0 = time.monotonic()
        chunks = self.retriever.retrieve(question, n_results=self.n_results, topic=topic)
        context = "\n\n---\n\n".join(
            f"[{c['metadata']['doc_name']} / {c['metadata']['section']}]\n{c['text']}"
            for c in chunks
        )
        answer = self.retriever.ask(question, n_results=self.n_results, topic=topic)
        latency_ms = int((time.monotonic() - t0) * 1000)

        scores = self._judge_answer(question, context, answer)
        context_precision = self._judge_context_precision(question, chunks)

        # Lightweight keyword check — fraction of expected_keywords present in the answer
        keyword_hit_rate = None
        if expected_keywords:
            answer_lower = answer.lower()
            hits = sum(1 for kw in expected_keywords if kw.lower() in answer_lower)
            keyword_hit_rate = round(hits / len(expected_keywords), 3)

        return {
            "id": item["id"],
            "question": question,
            "topic": topic,
            "question_type": item.get("question_type", ""),
            "expected_keywords": expected_keywords,
            "answer": answer,
            "retrieved_chunks": [
                {
                    "doc_name": c["metadata"]["doc_name"],
                    "section": c["metadata"]["section"],
                    "score": c["score"],
                    "topic": c["metadata"].get("topic", ""),
                }
                for c in chunks
            ],
            "scores": {
                **scores,
                "context_precision": context_precision,
                "keyword_hit_rate": keyword_hit_rate,
            },
            "latency_ms": latency_ms,
        }

    def run(self, test_set_path: Path) -> dict:
        """Evaluate all questions and return the full report dict."""
        from rag.utils.token_tracker import set_active_tracker
        questions = json.loads(test_set_path.read_text(encoding="utf-8"))["questions"]
        results = []

        self.tracker.reset()
        with set_active_tracker(self.tracker):
            for item in questions:
                print(f"  [{item['id']}] ({item.get('question_type','?')}) {item['question'][:70]}")
                r = self._eval_one(item)
                s = r["scores"]
                print(
                    f"         F={s.get('faithfulness')}  R={s.get('relevance')}  "
                    f"C={s.get('completeness')}  CP={s.get('context_precision')}  "
                    f"{r['latency_ms']}ms  — {s.get('reasoning','')[:70]}"
                )
                results.append(r)

        return {
            "run_at": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
            "model": DEEPSEEK_MODEL,
            "n_results": self.n_results,
            "summary": _compute_summary(results),
            "results": results,
        }

    def run_and_save(self, test_set_path: Path) -> Path:
        """Run evaluation, write a timestamped JSON report, and return its path."""
        n = len(json.loads(test_set_path.read_text())["questions"])
        print(f"Running eval on {n} questions...\n")
        report = self.run(test_set_path)

        ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        out_path = self._report_dir / f"eval_report_{ts}.json"
        out_path.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")

        s = report["summary"]["overall"]
        print(
            f"\nOverall — F={s['avg_faithfulness']}  R={s['avg_relevance']}  "
            f"C={s['avg_completeness']}  CP={s['avg_context_precision']}  "
            f"latency={s['avg_latency_ms']}ms"
        )
        print(f"Report → {out_path}")
        self.tracker.print_summary()
        self.tracker.save_jsonl(
            USAGE_LOG_PATH,
            {"agent": "EvalService", "test_set": str(test_set_path), "n_questions": n},
        )
        return out_path


# ---------------------------------------------------------------------------
# Summary computation
# ---------------------------------------------------------------------------

def _avg(values: list) -> float | None:
    vals = [v for v in values if v is not None]
    return round(sum(vals) / len(vals), 3) if vals else None


def _compute_summary(results: list[dict]) -> dict:
    def _slice_stats(subset: list[dict]) -> dict:
        return {
            "n": len(subset),
            "avg_faithfulness":     _avg([r["scores"].get("faithfulness")     for r in subset]),
            "avg_relevance":        _avg([r["scores"].get("relevance")        for r in subset]),
            "avg_completeness":     _avg([r["scores"].get("completeness")     for r in subset]),
            "avg_context_precision":_avg([r["scores"].get("context_precision") for r in subset]),
            "avg_keyword_hit_rate": _avg([r["scores"].get("keyword_hit_rate")  for r in subset]),
            "avg_latency_ms":       _avg([r.get("latency_ms")                  for r in subset]),
        }

    # By question_type
    by_type: dict[str, list] = {}
    for r in results:
        by_type.setdefault(r.get("question_type") or "unknown", []).append(r)

    # By topic
    by_topic: dict[str, list] = {}
    for r in results:
        by_topic.setdefault(r.get("topic") or "generic", []).append(r)

    return {
        "overall": _slice_stats(results),
        "by_question_type": {k: _slice_stats(v) for k, v in sorted(by_type.items())},
        "by_topic": {k: _slice_stats(v) for k, v in sorted(by_topic.items())},
    }
