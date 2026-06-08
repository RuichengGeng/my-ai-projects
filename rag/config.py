import os
from pathlib import Path

# Suppress HuggingFace tokenizers fork warning (triggered by ChromaDB's process pool)
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

# Where ChromaDB persists its files
CHROMA_PATH = Path(__file__).resolve().parent.parent / "rag_store"

# Root directory that contains parsed-PDF output folders
PDF_FILES_DIR = Path(__file__).resolve().parent.parent / "pdf_parser" / "pdf_files"

# Local sentence-transformer model for embeddings (downloads once, cached by HuggingFace)
EMBED_MODEL = "BAAI/bge-base-en-v1.5"

COLLECTION_NAME = "finance_docs"
MANIFEST_PATH = CHROMA_PATH / "manifest.json"

# DeepSeek model used for answer generation (OpenAI-compatible API)
DEEPSEEK_BASE_URL = "https://api.deepseek.com"
DEEPSEEK_MODEL = "deepseek-chat"

# Token usage log — one JSON line appended per agent run
USAGE_LOG_PATH = CHROMA_PATH / "usage_log.jsonl"
