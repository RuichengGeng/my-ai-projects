"""Shared manifest-aware retrieval planning for RAG agents."""

import json
from datetime import date
from typing import Callable, TypedDict

from rag.config import MANIFEST_PATH


class RetrievalPlan(TypedDict):
    topic_filter: str
    series_filter: str
    doc_filter: str
    doc_filters: list[str]
    date_from: str
    date_to: str
    query: str


PlannerLLM = Callable[..., str]


def _json(text: str) -> dict:
    text = text.strip()
    if text.startswith("```"):
        text = "\n".join(text.split("\n")[1:])
        text = text.rstrip("`").strip()
    return json.loads(text)


def empty_plan(question: str) -> RetrievalPlan:
    return {
        "topic_filter": "",
        "series_filter": "",
        "doc_filter": "",
        "doc_filters": [],
        "date_from": "",
        "date_to": "",
        "query": question,
    }


def plan_retrieval_scope(question: str, llm: PlannerLLM) -> RetrievalPlan:
    """Identify whether a question targets a manifest topic and date scope."""
    fallback = empty_plan(question)
    if not MANIFEST_PATH.exists():
        return fallback

    manifest = json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))
    docs = manifest.get("docs", [])
    if not docs:
        return fallback

    topic_docs: dict[str, list[dict]] = {}
    topic_series: dict[str, str] = {}
    for d in docs:
        topic = d.get("topic", "")
        if not topic:
            continue
        topic_docs.setdefault(topic, []).append(d)
        series_name = d.get("series_name", "")
        if series_name:
            topic_series[topic] = series_name

    if not topic_docs:
        return fallback

    topic_lines = []
    for topic in sorted(topic_docs.keys()):
        tdocs = topic_docs[topic]
        dated = sorted(d["series_date"] for d in tdocs if d.get("series_date"))
        if dated:
            topic_lines.append(
                f"  - {topic}  ({len(tdocs)} reports, dates: {dated[0]} to {dated[-1]})"
            )
        else:
            topic_lines.append(f"  - {topic}  (standalone)")

    prompt = (
        f"Today's date: {date.today().isoformat()}\n\n"
        "Available document topics:\n"
        + "\n".join(topic_lines)
        + f"\n\nQuestion: {question}\n\n"
        "Does this question target one of the topics above?\n"
        "If yes, determine the reporting scope:\n"
        '  "latest"     - only the single most recent report\n'
        '  "date_range" - reports within a specific date window '
        '(e.g. "last week", "this month", "week of June 2"). '
        "Compute date_from and date_to as YYYY-MM-DD strings relative to today.\n"
        '  "all"        - all reports in the topic\n\n'
        "Respond with JSON only:\n"
        '{"topic": "<exact topic name or empty>", '
        '"scope": "latest|date_range|all", '
        '"date_from": "<YYYY-MM-DD or empty>", '
        '"date_to": "<YYYY-MM-DD or empty>"}'
    )

    try:
        result = _json(llm([{"role": "user", "content": prompt}], label="plan"))
        topic = result.get("topic", "").strip()
        scope = result.get("scope", "all")
        date_from = result.get("date_from", "").strip()
        date_to = result.get("date_to", "").strip()

        series_filter = topic_series.get(topic, "") if topic else ""
        doc_filter = ""
        doc_filters: list[str] = []

        if topic and scope == "latest" and topic in topic_docs:
            candidates = [d for d in topic_docs[topic] if d.get("series_date")]
            if candidates:
                newest = max(candidates, key=lambda d: d.get("series_date") or "")
                doc_filter = newest["doc_name"]
                series_filter = ""
            date_from = date_to = ""

        elif topic and scope == "date_range" and topic in topic_docs and date_from:
            candidates = []
            for d in topic_docs[topic]:
                series_date = d.get("series_date", "")
                if not series_date:
                    continue
                if series_date < date_from:
                    continue
                if date_to and series_date > date_to:
                    continue
                candidates.append(d)

            if candidates:
                doc_filters = [
                    d["doc_name"]
                    for d in sorted(candidates, key=lambda x: x.get("series_date") or "")
                ]
                series_filter = ""
            else:
                date_from = date_to = ""

        elif scope != "date_range":
            date_from = date_to = ""

        log = f"  [plan] topic={topic or 'all'}  scope={scope}"
        if doc_filter:
            log += f"  -> doc={doc_filter[:50]}"
        elif doc_filters:
            log += f"  -> docs={len(doc_filters)}  {date_from} to {date_to or 'latest'}"
        elif date_from:
            log += f"  -> {date_from} to {date_to}"
        elif series_filter:
            log += f"  -> series={series_filter}"
        print(log)

        return {
            "topic_filter": topic,
            "series_filter": series_filter,
            "doc_filter": doc_filter,
            "doc_filters": doc_filters,
            "date_from": date_from,
            "date_to": date_to,
            "query": question,
        }
    except Exception:
        return fallback


def build_where(plan: RetrievalPlan) -> dict | None:
    """Translate a retrieval plan into a ChromaDB/BM25 metadata filter."""
    if plan["doc_filter"]:
        return {"doc_name": {"$eq": plan["doc_filter"]}}
    if plan["doc_filters"]:
        doc_filters = plan["doc_filters"]
        if len(doc_filters) == 1:
            return {"doc_name": {"$eq": doc_filters[0]}}
        return {"doc_name": {"$in": doc_filters}}
    if plan["series_filter"]:
        return {"series_name": {"$eq": plan["series_filter"]}}
    if plan["topic_filter"]:
        return {"topic": {"$eq": plan["topic_filter"]}}
    return None
