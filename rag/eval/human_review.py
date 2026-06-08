#!/usr/bin/env python
"""
Generate a human-readable Markdown review sheet from an eval report JSON.

Usage:
    uv run python rag/eval/human_review.py                  # uses latest report
    uv run python rag/eval/human_review.py path/to/report.json
"""

import json
import sys
from pathlib import Path

from rag.config import CHROMA_PATH

SCORE_BAR = {1: "▓░░░░", 2: "▓▓░░░", 3: "▓▓▓░░", 4: "▓▓▓▓░", 5: "▓▓▓▓▓"}


def _score_bar(val) -> str:
    if val is None:
        return "N/A"
    return f"{SCORE_BAR.get(round(val), '?')} {val:.1f}/5"


def _cp_bar(val) -> str:
    if val is None:
        return "N/A"
    filled = round(val * 5)
    return f"{'▓' * filled}{'░' * (5 - filled)} {val:.0%}"


def _load_report(path: Path | None) -> dict:
    if path:
        return json.loads(path.read_text(encoding="utf-8"))
    reports = sorted(CHROMA_PATH.glob("eval_report_*.json"))
    if not reports:
        raise FileNotFoundError(f"No eval reports found under {CHROMA_PATH}")
    return json.loads(reports[-1].read_text(encoding="utf-8"))


def generate(report: dict) -> str:
    lines: list[str] = []

    def h(level: int, text: str) -> None:
        lines.append(f"\n{'#' * level} {text}\n")

    def row(*cells) -> str:
        return "| " + " | ".join(str(c) for c in cells) + " |"

    h(1, f"RAG Human Review — {report['run_at'][:10]}")
    lines.append(
        f"**Model**: {report['model']}  |  "
        f"**n_results**: {report['n_results']}  |  "
        f"**Questions**: {report['summary']['overall']['n']}\n"
    )

    # ── Overall summary table ────────────────────────────────────────────────
    h(2, "Overall Scores")
    s = report["summary"]["overall"]
    lines.append(row("Metric", "Score"))
    lines.append(row("---", "---"))
    lines.append(row("Faithfulness",      _score_bar(s["avg_faithfulness"])))
    lines.append(row("Relevance",         _score_bar(s["avg_relevance"])))
    lines.append(row("Completeness",      _score_bar(s["avg_completeness"])))
    lines.append(row("Context Precision", _cp_bar(s["avg_context_precision"])))
    lines.append(row("Keyword Hit Rate",  f"{s['avg_keyword_hit_rate']:.0%}" if s['avg_keyword_hit_rate'] is not None else "N/A"))
    lines.append(row("Avg Latency",       f"{s['avg_latency_ms']} ms"))
    lines.append("")

    # ── By question type ────────────────────────────────────────────────────
    h(2, "Scores by Question Type")
    lines.append(row("Type", "n", "Faithfulness", "Relevance", "Completeness", "Context Precision"))
    lines.append(row("---", "---", "---", "---", "---", "---"))
    for qtype, ts in report["summary"]["by_question_type"].items():
        lines.append(row(
            f"`{qtype}`", ts["n"],
            _score_bar(ts["avg_faithfulness"]),
            _score_bar(ts["avg_relevance"]),
            _score_bar(ts["avg_completeness"]),
            _cp_bar(ts["avg_context_precision"]),
        ))
    lines.append("")

    # ── By topic ────────────────────────────────────────────────────────────
    h(2, "Scores by Topic")
    lines.append(row("Topic", "n", "Faithfulness", "Relevance", "Completeness", "Context Precision"))
    lines.append(row("---", "---", "---", "---", "---", "---"))
    for topic, ts in report["summary"]["by_topic"].items():
        lines.append(row(
            topic or "*(generic)*", ts["n"],
            _score_bar(ts["avg_faithfulness"]),
            _score_bar(ts["avg_relevance"]),
            _score_bar(ts["avg_completeness"]),
            _cp_bar(ts["avg_context_precision"]),
        ))
    lines.append("")

    # ── Per-question review ──────────────────────────────────────────────────
    h(2, "Per-Question Review")
    lines.append(
        "> For each question: read the answer, check the retrieved chunks, "
        "then fill in the human review checkboxes.\n"
    )

    for r in report["results"]:
        s = r["scores"]
        qtype = r.get("question_type", "?")
        topic = r.get("topic") or "generic"

        h(3, f"{r['id']}: {r['question']}")
        lines.append(
            f"**Type**: `{qtype}` | **Topic**: `{topic}` | "
            f"**Latency**: {r.get('latency_ms', '?')} ms\n"
        )

        # Scores summary
        lines.append(
            f"| F={_score_bar(s.get('faithfulness'))} "
            f"| R={_score_bar(s.get('relevance'))} "
            f"| C={_score_bar(s.get('completeness'))} "
            f"| CP={_cp_bar(s.get('context_precision'))} |\n"
        )

        # LLM judge reasoning
        if s.get("reasoning"):
            lines.append(f"*Judge reasoning*: {s['reasoning']}\n")

        # Keywords
        kws = r.get("expected_keywords", [])
        if kws:
            answer_lower = r["answer"].lower()
            kw_results = ", ".join(
                f"~~{kw}~~" if kw.lower() not in answer_lower else f"**{kw}**"
                for kw in kws
            )
            lines.append(f"*Expected keywords* (bold = found): {kw_results}\n")

        # Answer
        lines.append("**Generated Answer:**\n")
        lines.append(f"> {r['answer'].replace(chr(10), chr(10) + '> ')}\n")

        # Retrieved chunks
        lines.append("**Retrieved Chunks:**\n")
        for i, c in enumerate(r["retrieved_chunks"], 1):
            lines.append(
                f"{i}. `{c['doc_name'][:50]}` / *{c['section'][:50]}* — score {c['score']:.3f}"
                + (f" | topic: {c['topic']}" if c.get("topic") else "")
            )
        lines.append("")

        # Human review checkboxes
        lines.append("**Human Review:**\n")
        lines.append("- [ ] The answer is factually accurate based on your domain knowledge")
        lines.append("- [ ] The answer fully addresses the question (nothing important is omitted)")
        lines.append("- [ ] The answer does not contain hallucinated information")
        lines.append("- [ ] The retrieved chunks were the right sources for this question")
        lines.append("- [ ] You would trust this answer without further verification")
        lines.append("")
        lines.append("*Human notes / corrections:*")
        lines.append("")
        lines.append("---")

    return "\n".join(lines)


def main() -> None:
    report_path = Path(sys.argv[1]) if len(sys.argv) > 1 else None
    report = _load_report(report_path)
    md = generate(report)

    out_path = CHROMA_PATH / f"human_review_{report['run_at'][:10]}.md"
    out_path.write_text(md, encoding="utf-8")
    print(f"Human review sheet → {out_path}")


if __name__ == "__main__":
    main()
