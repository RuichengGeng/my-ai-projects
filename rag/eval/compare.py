#!/usr/bin/env python
"""
Compare two eval report JSONs and print a delta table.

Usage:
    uv run python rag/eval/compare.py report_baseline.json report_new.json

Positive delta means the new run is better.
"""

import json
import sys
from pathlib import Path


def _load(path: str) -> dict:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def _delta(a, b) -> str:
    if a is None or b is None:
        return "N/A"
    diff = b - a
    sign = "+" if diff >= 0 else ""
    return f"{sign}{diff:.3f}"


def _print_table(title: str, baseline: dict, new: dict) -> None:
    metrics = [
        ("avg_faithfulness",      "Faithfulness   (/5)"),
        ("avg_relevance",         "Relevance      (/5)"),
        ("avg_completeness",      "Completeness   (/5)"),
        ("avg_context_precision", "Context Prec.  (0-1)"),
        ("avg_keyword_hit_rate",  "Keyword Hits   (0-1)"),
        ("avg_latency_ms",        "Latency        (ms) "),
    ]

    col_w = 12
    print(f"\n{'═' * 70}")
    print(f"  {title}")
    print(f"{'═' * 70}")
    print(f"  {'Metric':<28} {'Baseline':>{col_w}} {'New':>{col_w}} {'Delta':>{col_w}}")
    print(f"  {'-' * 28} {'-' * col_w} {'-' * col_w} {'-' * col_w}")

    for key, label in metrics:
        a = baseline.get(key)
        b = new.get(key)
        a_str = f"{a:.3f}" if a is not None else "N/A"
        b_str = f"{b:.3f}" if b is not None else "N/A"
        delta = _delta(a, b)
        # Highlight improvement / regression
        flag = ""
        if a is not None and b is not None:
            diff = b - a
            if key == "avg_latency_ms":
                flag = " ✓" if diff < 0 else (" ✗" if diff > 50 else "")
            else:
                flag = " ✓" if diff > 0.05 else (" ✗" if diff < -0.05 else "")
        print(f"  {label:<28} {a_str:>{col_w}} {b_str:>{col_w}} {delta:>{col_w}}{flag}")


def main() -> None:
    if len(sys.argv) != 3:
        print("Usage: compare.py <baseline_report.json> <new_report.json>")
        sys.exit(1)

    baseline = _load(sys.argv[1])
    new = _load(sys.argv[2])

    print(f"\nBaseline : {baseline['run_at']}  model={baseline['model']}  n_results={baseline['n_results']}")
    print(f"New      : {new['run_at']}  model={new['model']}  n_results={new['n_results']}")

    _print_table("OVERALL", baseline["summary"]["overall"], new["summary"]["overall"])

    # By question type
    all_types = sorted(
        set(baseline["summary"]["by_question_type"]) | set(new["summary"]["by_question_type"])
    )
    for qtype in all_types:
        b = baseline["summary"]["by_question_type"].get(qtype, {})
        n = new["summary"]["by_question_type"].get(qtype, {})
        _print_table(f"Question type: {qtype}", b, n)

    # By topic
    all_topics = sorted(
        set(baseline["summary"]["by_topic"]) | set(new["summary"]["by_topic"])
    )
    for topic in all_topics:
        b = baseline["summary"]["by_topic"].get(topic, {})
        n = new["summary"]["by_topic"].get(topic, {})
        _print_table(f"Topic: {topic or 'generic'}", b, n)

    print()


if __name__ == "__main__":
    main()
