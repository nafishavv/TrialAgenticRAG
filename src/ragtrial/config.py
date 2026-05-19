"""Project-wide paths and env loading.

All paths are absolute, resolved from this file's location — works regardless of cwd.
"""

from __future__ import annotations

import os
from pathlib import Path

from dotenv import load_dotenv

PROJECT_ROOT: Path = Path(__file__).resolve().parent.parent.parent

DATA_DIR: Path = PROJECT_ROOT / "data"
VECTOR_STORE_DIR: Path = DATA_DIR / "vector_stores"
PROCESSED_DIR: Path = DATA_DIR
RAW_DIR: Path = DATA_DIR

DUKCAPIL_VECTOR_STORE: Path = DATA_DIR / "dukcapil_vector_store"
OPD_VECTOR_STORE: Path = DATA_DIR / "opd_vector_store"

DUKCAPIL_PROCESSED_PKL: Path = DATA_DIR / "cleaned_docs.pkl"
OPD_PROCESSED_PKL: Path = DATA_DIR / "cleaned_opd_docs.pkl"


def load_env() -> str:
    """Load .env from project root, return resolved Gemini API key."""
    load_dotenv(PROJECT_ROOT / ".env")
    api_key = os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")
    assert api_key, f"GEMINI_API_KEY tidak ditemukan di {PROJECT_ROOT / '.env'}"
    os.environ["GOOGLE_API_KEY"] = api_key
    return api_key
