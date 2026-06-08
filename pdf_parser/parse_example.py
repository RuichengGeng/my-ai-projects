#!/usr/bin/env python
"""Parse PDFs to Markdown using MinerU's v4 Batch File Extract API."""

import os
import sys
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from pdf_parser import MinerUClient, MinerUError, MinerUTimeoutError

# ── Configure here ────────────────────────────────────────────────────────────
PDF_DIR = Path(__file__).resolve().parent / "pdf_files"

# Single-file mode: set PDF_PATH to a .pdf file
# PDF_PATH = PDF_DIR / "Machine Learning for Algorithmic Trading.pdf"
PDF_PATH = None

# Folder mode: set PDF_FOLDER to a directory of PDFs (overrides PDF_PATH)
PDF_FOLDER = PDF_DIR / "APAC Oil week"

# URL mode: set to a public URL string (overrides both above)
PDF_URL = None
# ─────────────────────────────────────────────────────────────────────────────


def main():
    token = os.getenv("MINERU_API_TOKEN")
    if not token:
        sys.exit("ERROR: MINERU_API_TOKEN not set. Add it to .env or environment.")

    client = MinerUClient(api_token=token)

    if PDF_URL:
        print(f"Parsing URL: {PDF_URL}")
        markdown = client.parse_url(PDF_URL)
        print(f"Done ({len(markdown):,} chars)")
        print("\n--- Preview (first 500 chars) ---")
        print(markdown[:500])

    elif PDF_FOLDER:
        results = client.parse_folder(PDF_FOLDER, output_path=PDF_FOLDER)
        print(f"\n--- Summary ---")
        for stem, md in results.items():
            print(f"  {stem}: {len(md):,} chars")

    else:
        print(f"Parsing: {PDF_PATH} ({PDF_PATH.stat().st_size / 1024:.1f} KB)")
        markdown = client.parse_pdf(PDF_PATH, output_path=PDF_DIR)
        print("\n--- Preview (first 500 chars) ---")
        print(markdown[:500])


if __name__ == "__main__":
    try:
        main()
    except MinerUTimeoutError as e:
        sys.exit(f"TIMEOUT: {e}")
    except MinerUError as e:
        sys.exit(f"ERROR: {e}")
    except (FileNotFoundError, NotADirectoryError) as e:
        sys.exit(f"ERROR: {e}")
