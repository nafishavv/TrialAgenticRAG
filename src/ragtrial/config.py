"""Project-wide paths and env loading.

All paths are absolute, resolved from this file's location — works regardless of cwd.
"""

from __future__ import annotations

import os
from pathlib import Path

from dotenv import load_dotenv

PROJECT_ROOT: Path = Path(__file__).resolve().parent.parent.parent

DATA_DIR: Path = PROJECT_ROOT / "data"
RAW_DIR: Path = DATA_DIR / "raw"
PROCESSED_DIR: Path = DATA_DIR / "processed"
VECTOR_STORE_DIR: Path = DATA_DIR / "vector_stores"

# Per-domain raw directories — a domain may hold many files / nested folders.
DUKCAPIL_RAW_DIR: Path = RAW_DIR / "dukcapil"
OPD_RAW_DIR: Path = RAW_DIR / "opd"

# Legacy single-file pointers (kept for reference; ingestion now walks the dir).
DUKCAPIL_RAW_PDF: Path = DUKCAPIL_RAW_DIR / "Buku-Saku-Dafduk-Capil-2023.pdf"
OPD_RAW_PDF: Path = OPD_RAW_DIR / "Nama dan Alamat OPD Kab Batang.pdf"

DUKCAPIL_PROCESSED_PKL: Path = PROCESSED_DIR / "dukcapil.pkl"
OPD_PROCESSED_PKL: Path = PROCESSED_DIR / "opd.pkl"

DUKCAPIL_VECTOR_STORE: Path = VECTOR_STORE_DIR / "dukcapil"
OPD_VECTOR_STORE: Path = VECTOR_STORE_DIR / "opd"

# Unified collection for naive RAG — all domains in one store (built by copying
# vectors from the per-domain stores, no re-embed).
UNIFIED_VECTOR_STORE: Path = VECTOR_STORE_DIR / "_unified"
UNIFIED_COLLECTION: str = "unified"


def load_env() -> str:
    """Load .env from project root, return resolved Gemini API key."""
    load_dotenv(PROJECT_ROOT / ".env")
    api_key = os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")
    assert api_key, f"GEMINI_API_KEY tidak ditemukan di {PROJECT_ROOT / '.env'}"
    os.environ["GOOGLE_API_KEY"] = api_key
    return api_key
