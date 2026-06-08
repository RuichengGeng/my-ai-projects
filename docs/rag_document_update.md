# RAG Document Update Workflow

This document explains how to add new PDFs to the RAG store and update the
Chroma index.

## Folder Layout

Raw PDFs and parsed markdown live under:

```text
pdf_parser/pdf_files/
```

For the APAC oil reports, use:

```text
pdf_parser/pdf_files/APAC Oil week/
```

After parsing, each PDF should have its own output folder:

```text
pdf_parser/pdf_files/APAC Oil week/
  Crude Oil Marketwire_08_Jun_2026.pdf
  Crude Oil Marketwire_08_Jun_2026/
    Crude Oil Marketwire_08_Jun_2026.md
    images/
```

The indexer reads the markdown folders, chunks them, embeds them, writes chunks
to ChromaDB, and records document metadata in:

```text
rag_store/manifest.json
```

## Daily Update

1. Put new PDFs in:

```text
pdf_parser/pdf_files/APAC Oil week/
```

2. Run the all-in-one pipeline:

```powershell
.\.venv\Scripts\python.exe rag\indexer\update_pipeline.py --list
```

This will:

- parse any new PDFs with MinerU
- skip PDFs that already have parsed markdown
- index parsed markdown that is not already in `manifest.json`
- print the indexed document list

## Index Existing Parsed Markdown Only

Use this when markdown folders already exist and you do not want to call MinerU:

```powershell
.\.venv\Scripts\python.exe rag\indexer\update_pipeline.py --skip-parse --list
```

## Parse Only

Use this when you want markdown output but do not want to update Chroma yet:

```powershell
.\.venv\Scripts\python.exe rag\indexer\update_pipeline.py --skip-index
```

## Single PDF

```powershell
.\.venv\Scripts\python.exe rag\indexer\update_pipeline.py --pdf-file "pdf_parser\pdf_files\APAC Oil week\Crude Oil Marketwire_08_Jun_2026.pdf" --list
```

## Re-Index One Updated Document

Use this if you edited or regenerated a markdown file for an existing document:

```powershell
.\.venv\Scripts\python.exe rag\indexer\update_pipeline.py --reindex-doc "Crude Oil Marketwire_08_Jun_2026" --list
```

## Delete One Document From The Index

```powershell
.\.venv\Scripts\python.exe rag\indexer\update_pipeline.py --delete-doc "Crude Oil Marketwire_08_Jun_2026" --list
```

This removes the document from ChromaDB and `rag_store/manifest.json`. It does
not delete the source PDF or parsed markdown folder.

## Full Rebuild

Use only when you want to rebuild the entire Chroma collection from parsed
markdown:

```powershell
.\.venv\Scripts\python.exe rag\indexer\update_pipeline.py --skip-parse --reset-index --list
```

## Custom Folder

If you add another report collection under `pdf_parser/pdf_files`, pass it as
the PDF folder:

```powershell
.\.venv\Scripts\python.exe rag\indexer\update_pipeline.py --pdf-folder "pdf_parser\pdf_files\My New Reports" --list
```

If you only want to index an existing parsed folder:

```powershell
.\.venv\Scripts\python.exe rag\indexer\update_pipeline.py --skip-parse --parsed-root "pdf_parser\pdf_files\My New Reports" --list
```

## Requirements

Parsing requires:

```text
MINERU_API_TOKEN
```

Set it in `.env` or in the environment. Index-only commands do not require the
MinerU token.

## Notes

- Normal updates should keep `--reset-index` off.
- Document names come from PDF stems / parsed folder names.
- Dated names such as `Crude Oil Marketwire_08_Jun_2026` are automatically
  grouped into a series with `series_name` and `series_date`.
- The planner uses `manifest.json`, so new documents become available to
  `RAGAgent` and `OilMarketSummaryAgent` after indexing.
