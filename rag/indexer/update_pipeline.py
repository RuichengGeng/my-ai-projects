#!/usr/bin/env python
"""Parse PDFs and update the RAG vector index in one command."""

import argparse
import os
import sys
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from pdf_parser import MinerUClient, MinerUError, MinerUTimeoutError
from rag.config import PDF_FILES_DIR
from rag.indexer.service import build_index, delete_doc, list_docs, reindex_doc


DEFAULT_COLLECTION_DIR = PDF_FILES_DIR / "APAC Oil week"


def _path(value: str) -> Path:
    return Path(value).expanduser().resolve()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Parse PDFs with MinerU and update the Chroma RAG index.",
    )
    parser.add_argument(
        "--pdf-folder",
        type=_path,
        default=DEFAULT_COLLECTION_DIR,
        help=f"Folder containing PDFs to parse. Default: {DEFAULT_COLLECTION_DIR}",
    )
    parser.add_argument(
        "--pdf-file",
        type=_path,
        help="Parse one PDF instead of a folder.",
    )
    parser.add_argument(
        "--parsed-root",
        type=_path,
        help=(
            "Root containing parsed markdown doc folders to index. "
            "Defaults to --pdf-file parent or --pdf-folder."
        ),
    )
    parser.add_argument(
        "--skip-parse",
        action="store_true",
        help="Do not call MinerU; only update the index from existing markdown.",
    )
    parser.add_argument(
        "--skip-index",
        action="store_true",
        help="Only parse PDFs; do not update Chroma.",
    )
    parser.add_argument(
        "--force-parse",
        action="store_true",
        help="Re-parse PDFs even when an output markdown folder already exists.",
    )
    parser.add_argument(
        "--reset-index",
        action="store_true",
        help="Drop the Chroma collection and rebuild from parsed markdown.",
    )
    parser.add_argument(
        "--list",
        action="store_true",
        help="Print indexed documents after the requested operation.",
    )
    parser.add_argument(
        "--delete-doc",
        help="Delete one indexed document by document/folder name, then exit.",
    )
    parser.add_argument(
        "--reindex-doc",
        help="Re-index one existing parsed document by document/folder name, then exit.",
    )
    return parser.parse_args()


def _parse(args: argparse.Namespace, output_root: Path) -> None:
    token = os.getenv("MINERU_API_TOKEN")
    if not token:
        raise MinerUError("MINERU_API_TOKEN is not set. Add it to .env or the environment.")

    client = MinerUClient(api_token=token)
    if args.pdf_file:
        print(f"Parsing PDF: {args.pdf_file}")
        client.parse_pdf(args.pdf_file, output_path=output_root)
    else:
        print(f"Parsing PDF folder: {args.pdf_folder}")
        client.parse_folder(
            args.pdf_folder,
            output_path=output_root,
            skip_existing=not args.force_parse,
        )


def main() -> None:
    args = parse_args()
    parsed_root = args.parsed_root or (args.pdf_file.parent if args.pdf_file else args.pdf_folder)

    if args.delete_doc:
        delete_doc(args.delete_doc)
        if args.list:
            list_docs()
        return

    if args.reindex_doc:
        reindex_doc(args.reindex_doc, pdf_files_dir=parsed_root)
        if args.list:
            list_docs()
        return

    if not args.skip_parse:
        _parse(args, parsed_root)

    if not args.skip_index:
        build_index(pdf_files_dir=parsed_root, reset=args.reset_index)

    if args.list:
        list_docs()


if __name__ == "__main__":
    try:
        main()
    except MinerUTimeoutError as exc:
        sys.exit(f"TIMEOUT: {exc}")
    except MinerUError as exc:
        sys.exit(f"ERROR: {exc}")
    except (FileNotFoundError, NotADirectoryError) as exc:
        sys.exit(f"ERROR: {exc}")
