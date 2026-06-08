"""
Configuration for the MinerU PDF parser.

Set your API token via the MINERU_API_TOKEN environment variable,
or pass it directly to the MinerUClient constructor.
"""

import os

# MinerU cloud API base URL
MINERU_BASE_URL = os.getenv("MINERU_BASE_URL", "https://mineru.net")

# API token from https://mineru.net/apiManage/token
MINERU_API_TOKEN = os.getenv("MINERU_API_TOKEN", "")

# v4 Batch page limit per item (API max is 200)
BATCH_PAGE_LIMIT = 200

# Polling settings
POLL_INTERVAL_SECONDS = 5   # seconds between status checks
POLL_TIMEOUT_SECONDS = 600  # max total time to wait (large PDFs can take >5 min)


