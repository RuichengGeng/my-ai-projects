"""
Scan all parsed-PDF output folders under PDF_FILES_DIR, chunk each markdown file
by H2 section headers, and store in a local ChromaDB collection.

Expected folder layout (produced by pdf_parser):
  pdf_files/
    <doc-name>/
      <doc-name>.md
      images/
        *.jpg

Series detection
----------------
Doc names containing a trailing date (e.g. "Report_05_Jun_2026", "Report-260511")
are grouped into a series. The manifest tracks series_name, series_date, and
which doc is the latest in each series.
"""

import json
import re
from datetime import date, datetime, timezone
from pathlib import Path

import chromadb
from chromadb.utils.embedding_functions import SentenceTransformerEmbeddingFunction

from rag.config import CHROMA_PATH, COLLECTION_NAME, EMBED_MODEL, MANIFEST_PATH, PDF_FILES_DIR

# ---------------------------------------------------------------------------
# Manifest helpers
# ---------------------------------------------------------------------------

def _load_manifest() -> dict:
    if MANIFEST_PATH.exists():
        return json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))
    return {"docs": []}


def _save_manifest(manifest: dict) -> None:
    MANIFEST_PATH.write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")


# ---------------------------------------------------------------------------
# Series detection
# ---------------------------------------------------------------------------

_MONTH = {"jan":1,"feb":2,"mar":3,"apr":4,"may":5,"jun":6,
          "jul":7,"aug":8,"sep":9,"oct":10,"nov":11,"dec":12}

# Patterns tried in order; each yields (series_name, date) or skips on failure
_DATE_PATTERNS = [
    # _DD_Mon_YYYY  e.g. _05_Jun_2026
    (re.compile(r'[_-](\d{1,2})[_-]([A-Za-z]{3})[_-](\d{4})$'), 'dmy'),
    # _YYYY-MM-DD  e.g. _2026-06-05
    (re.compile(r'[_-](\d{4})-(\d{2})-(\d{2})$'), 'ymd_dash'),
    # _YYYYMMDD  e.g. _20260511  (8 digits)
    (re.compile(r'[_-](\d{4})(\d{2})(\d{2})$'), 'ymd'),
    # _YYMMDD  e.g. -260511  (6 digits → 20YY)
    (re.compile(r'[_-](\d{2})(\d{2})(\d{2})$'), 'yymmdd'),
]


def _parse_series(doc_name: str) -> tuple[str | None, date | None]:
    """Return (series_name, series_date) if the doc name contains a trailing date, else (None, None)."""
    for pattern, fmt in _DATE_PATTERNS:
        m = pattern.search(doc_name)
        if not m:
            continue
        try:
            g = m.groups()
            if fmt == 'dmy':
                d = date(int(g[2]), _MONTH[g[1].lower()], int(g[0]))
            elif fmt == 'ymd_dash':
                d = date(int(g[0]), int(g[1]), int(g[2]))
            elif fmt == 'ymd':
                d = date(int(g[0]), int(g[1]), int(g[2]))
            elif fmt == 'yymmdd':
                d = date(2000 + int(g[0]), int(g[1]), int(g[2]))
            else:
                continue

            series_name = doc_name[:m.start()].strip()
            return series_name, d
        except (ValueError, KeyError):
            continue

    return None, None


def _compute_series_info(manifest: dict) -> dict[str, str]:
    """Return {doc_name: latest_doc_name} mapping per series from the manifest."""
    by_series: dict[str, list[dict]] = {}
    for doc in manifest["docs"]:
        sn = doc.get("series_name")
        if sn:
            by_series.setdefault(sn, []).append(doc)

    latest: dict[str, str] = {}
    for sn, docs in by_series.items():
        newest = max(docs, key=lambda d: d.get("series_date") or "")
        latest[sn] = newest["doc_name"]
    return latest


# ---------------------------------------------------------------------------
# Markdown chunking
# ---------------------------------------------------------------------------

def _find_md_files(root: Path) -> list[Path]:
    return sorted(root.glob("*/*.md"))


def _split_by_h2(text: str, doc_name: str) -> list[dict]:
    h2 = re.compile(r'^(##\s+.+)$', re.MULTILINE)
    positions = [(m.start(), m.group(1)) for m in h2.finditer(text)]

    def _make_chunk(section: str, raw: str) -> dict:
        images = re.findall(r'!\[.*?\]\((images/[^)]+)\)', raw)
        return {"section": section, "text": raw.strip(), "images": images}

    if not positions:
        return [_make_chunk(doc_name, text)]

    chunks = []
    preamble = text[:positions[0][0]].strip()
    if preamble:
        chunks.append(_make_chunk(doc_name, preamble))

    for i, (pos, header) in enumerate(positions):
        end = positions[i + 1][0] if i + 1 < len(positions) else len(text)
        chunks.append(_make_chunk(header.lstrip("#").strip(), text[pos:end]))

    return chunks


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def build_index(pdf_files_dir: Path = PDF_FILES_DIR, reset: bool = False) -> None:
    """Index all parsed .md files into ChromaDB.

    Args:
        pdf_files_dir: Root directory containing parsed-PDF subfolders.
        reset:         If True, drop and rebuild the collection from scratch.
    """
    ef = SentenceTransformerEmbeddingFunction(model_name=EMBED_MODEL)
    client = chromadb.PersistentClient(path=str(CHROMA_PATH))

    if reset:
        try:
            client.delete_collection(COLLECTION_NAME)
            print(f"Dropped existing collection '{COLLECTION_NAME}'")
        except Exception:
            pass
        _save_manifest({"docs": []})

    collection = client.get_or_create_collection(
        name=COLLECTION_NAME,
        embedding_function=ef,
        metadata={"hnsw:space": "cosine"},
    )

    md_files = _find_md_files(pdf_files_dir)
    if not md_files:
        print(f"No parsed markdown files found under {pdf_files_dir}")
        return

    print(f"Found {len(md_files)} document(s) under {pdf_files_dir}")

    manifest = _load_manifest()
    indexed_names = {d["doc_name"] for d in manifest["docs"]}

    for md_path in md_files:
        doc_name = md_path.parent.name

        if doc_name in indexed_names:
            print(f"  Skipping (already indexed): {doc_name[:60]}")
            continue

        series_name, series_date = _parse_series(doc_name)
        print(f"  Indexing: {doc_name[:60]}"
              + (f"  [series: {series_name}  date: {series_date}]" if series_name else ""))

        text = md_path.read_text(encoding="utf-8")
        chunks = _split_by_h2(text, doc_name)

        collection.add(
            ids=[f"{doc_name}::{i}" for i in range(len(chunks))],
            documents=[c["text"] for c in chunks],
            metadatas=[
                {
                    "doc_name": doc_name,
                    "source": str(md_path),
                    "section": c["section"],
                    "images": ",".join(c["images"]),
                    "series_name": series_name or "",
                    "series_date": series_date.isoformat() if series_date else "",
                }
                for c in chunks
            ],
        )

        manifest["docs"].append({
            "doc_name": doc_name,
            "source": str(md_path),
            "chunks": len(chunks),
            "indexed_at": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
            "series_name": series_name or "",
            "series_date": series_date.isoformat() if series_date else "",
        })
        _save_manifest(manifest)
        print(f"    {len(chunks)} chunks added")

    print(f"\nIndex ready — {collection.count()} total chunks at {CHROMA_PATH}")


def list_docs() -> None:
    """Print every document recorded in the index manifest, grouped by series."""
    manifest = _load_manifest()
    docs = manifest.get("docs", [])
    if not docs:
        print(f"No documents indexed yet. Manifest: {MANIFEST_PATH}")
        return

    # Compute latest per series
    latest_map = _compute_series_info(manifest)

    series_docs: dict[str, list[dict]] = {}
    standalone: list[dict] = []
    for d in docs:
        sn = d.get("series_name")
        if sn:
            series_docs.setdefault(sn, []).append(d)
        else:
            standalone.append(d)

    total_chunks = sum(d["chunks"] for d in docs)
    print(f"{len(docs)} document(s) indexed — {total_chunks} total chunks:\n")

    for sn, members in sorted(series_docs.items()):
        print(f"  Series: {sn}")
        for d in sorted(members, key=lambda x: x.get("series_date", ""), reverse=True):
            tag = " ← latest" if latest_map.get(sn) == d["doc_name"] else ""
            print(f"    {d['chunks']:>4} chunks  [{d['series_date']}]{tag}  {d['doc_name']}")

    for d in sorted(standalone, key=lambda x: x["doc_name"]):
        print(f"  {d['chunks']:>4} chunks  [{d['indexed_at']}]  {d['doc_name']}")
