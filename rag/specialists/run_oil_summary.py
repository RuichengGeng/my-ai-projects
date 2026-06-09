#!/usr/bin/env python
"""Generate a structured oil market summary from indexed documents."""

import argparse
import re
import sys
from datetime import datetime
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from rag.specialists.oil_market import OilMarketSummaryAgent

# Configure here
QUESTION = "Summarize APAC oil market update as of 2026-06-08"
N_RESULTS = 5   # chunks retrieved per sub-query (6 sub-queries run in total)
EXPORT_MD = False
EXPORT_DIR = Path("rag_outputs") / "oil_market"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate a structured oil market summary from indexed documents.",
    )
    parser.add_argument(
        "--question",
        default=QUESTION,
        help="Question/scope for the oil market summary.",
    )
    parser.add_argument(
        "--n-results",
        type=int,
        default=N_RESULTS,
        help="Chunks retrieved per section query.",
    )
    parser.add_argument(
        "--export-md",
        action="store_true",
        default=EXPORT_MD,
        help="Write the generated summary to a markdown file.",
    )
    parser.add_argument(
        "--output-file",
        type=Path,
        help="Exact markdown output path. Implies --export-md.",
    )
    return parser.parse_args()


def _slug(text: str) -> str:
    slug = re.sub(r"[^A-Za-z0-9]+", "-", text.lower()).strip("-")
    return slug[:80] or "oil-market-summary"


def _default_output_file(question: str) -> Path:
    date_match = re.search(r"\b(20\d{2}-\d{2}-\d{2})\b", question)
    suffix = date_match.group(1) if date_match else datetime.now().strftime("%Y%m%d-%H%M%S")
    return EXPORT_DIR / f"{_slug(question)}-{suffix}.md"


def export_markdown(answer: str, question: str, output_file: Path | None = None) -> Path:
    path = output_file or _default_output_file(question)
    path.parent.mkdir(parents=True, exist_ok=True)
    content = f"# Oil Market Summary\n\n**Question:** {question}\n\n{answer.strip()}\n"
    path.write_text(content, encoding="utf-8")
    return path


def main() -> None:
    args = parse_args()
    export_md = True

    agent = OilMarketSummaryAgent(n_results=args.n_results)
    answer = agent.run(args.question)
    print("\nOil Market Summary")
    print("=" * 60)
    print(answer)

    if export_md:
        output_path = export_markdown(answer, args.question, args.output_file)
        print(f"\nMarkdown exported: {output_path}")


if __name__ == "__main__":
    main()
