"""RAG evaluation service using LLM-as-judge (DeepSeek).

Metrics per question:
  faithfulness (1-5): does the answer contain only information inferable from
                      the retrieved context? penalises hallucination.
  relevance    (1-5): does the answer directly address the question asked?

A JSON report is written to rag_store/eval_report_<timestamp>.json.
"""

import json
import os
from datetime import datetime, timezone
from pathlib import Path

from rag.config import CHROMA_PATH, DEEPSEEK_BASE_URL, DEEPSEEK_MODEL
from rag.query.service import RAGRetriever

_JUDGE_PROMPT = """\
You are a RAG evaluation judge. Given the retrieved context, a question, and an answer, \
score the answer on two dimensions.

Faithfulness (1-5): Does the answer contain ONLY information that can be inferred from \
the context below? 5 = fully grounded in context, 1 = contains clear hallucinations.

Relevance (1-5): Does the answer directly address the question? \
5 = complete and on-point, 1 = off-topic or evasive.

Retrieved context:
{context}

Question: {question}

Answer: {answer}

Respond with JSON only, no markdown fences:
{{"faithfulness": <1-5>, "relevance": <1-5>, "reasoning": "<one sentence>"}}"""


class EvalService:
    def __init__(self, n_results: int = 5):
        self.retriever = RAGRetriever()
        self.n_results = n_results
        self._report_dir = CHROMA_PATH

    def _judge(self, question: str, context: str, answer: str) -> dict:
        from openai import OpenAI

        client = OpenAI(
            api_key=os.environ.get("DEEPSEEK_API_KEY"),
            base_url=DEEPSEEK_BASE_URL,
        )
        response = client.chat.completions.create(
            model=DEEPSEEK_MODEL,
            max_tokens=256,
            temperature=0,
            messages=[
                {
                    "role": "user",
                    "content": _JUDGE_PROMPT.format(
                        context=context, question=question, answer=answer
                    ),
                }
            ],
        )
        raw = response.choices[0].message.content.strip()
        try:
            return json.loads(raw)
        except json.JSONDecodeError:
            return {"faithfulness": None, "relevance": None, "reasoning": raw}

    def run(self, test_set_path: Path) -> dict:
        """Evaluate all questions in the test set and return a full report dict."""
        questions = json.loads(test_set_path.read_text(encoding="utf-8"))["questions"]
        results = []

        for item in questions:
            qid = item["id"]
            question = item["question"]
            print(f"  [{qid}] {question[:70]}")

            chunks = self.retriever.retrieve(question, n_results=self.n_results)
            context = "\n\n---\n\n".join(
                f"[{c['metadata']['doc_name']} / {c['metadata']['section']}]\n{c['text']}"
                for c in chunks
            )
            answer = self.retriever.ask(question, n_results=self.n_results)
            scores = self._judge(question, context, answer)

            f = scores.get("faithfulness")
            r = scores.get("relevance")
            print(f"         faithfulness={f}  relevance={r}  — {scores.get('reasoning', '')[:80]}")

            results.append({
                "id": qid,
                "question": question,
                "answer": answer,
                "retrieved_chunks": [
                    {"doc_name": c["metadata"]["doc_name"], "section": c["metadata"]["section"], "score": c["score"]}
                    for c in chunks
                ],
                "scores": scores,
            })

        valid = [r for r in results if r["scores"].get("faithfulness") is not None]
        summary = {
            "n_questions": len(results),
            "avg_faithfulness": round(sum(r["scores"]["faithfulness"] for r in valid) / len(valid), 2) if valid else None,
            "avg_relevance": round(sum(r["scores"]["relevance"] for r in valid) / len(valid), 2) if valid else None,
        }

        return {
            "run_at": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
            "model": DEEPSEEK_MODEL,
            "n_results": self.n_results,
            "summary": summary,
            "results": results,
        }

    def run_and_save(self, test_set_path: Path) -> Path:
        """Run evaluation and write a timestamped JSON report. Returns the report path."""
        print(f"Running eval on {len(json.loads(test_set_path.read_text())['questions'])} questions...\n")
        report = self.run(test_set_path)

        ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        out_path = self._report_dir / f"eval_report_{ts}.json"
        out_path.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")

        print(f"\nSummary: faithfulness={report['summary']['avg_faithfulness']}  relevance={report['summary']['avg_relevance']}")
        print(f"Report saved → {out_path}")
        return out_path
