"""Perizinan domain — izin/perizinan layanan SIPUAS Kab. Batang.

Sumber: data/raw/perizinan/perizinan_data.json (34 jenis izin). Tiap perizinan
jadi satu Document atomik (lihat preprocess.py); chunking = identity.
"""

from __future__ import annotations

from typing import List

from langchain_core.documents import Document

from ragtrial.config import PERIZINAN_PROCESSED_PKL, PERIZINAN_RAW_DIR
from ragtrial.sources.base import Source
from ragtrial.sources.perizinan.capability import perizinan_capability
from ragtrial.sources.perizinan.chunk import chunk_for_vectorstore
from ragtrial.sources.perizinan.preprocess import preprocess

_DATA_FILE = "perizinan_data.json"


def build_documents() -> List[Document]:
    """Parse JSON perizinan di raw dir jadi Documents atomik."""
    return preprocess(PERIZINAN_RAW_DIR / _DATA_FILE)


source = Source(
    name="perizinan",
    raw_dir=PERIZINAN_RAW_DIR,
    processed_pkl=PERIZINAN_PROCESSED_PKL,
    capability=perizinan_capability,
    build_documents=build_documents,
    chunk=chunk_for_vectorstore,
)
