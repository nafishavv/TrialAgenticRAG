"""Sosial domain — produk hukum sektor sosial JDIH Kab. Batang.

Korpus heterogen (Perda/Perbup/Kepbup/Abstrak/Artikel/...). `preprocess()`
metadata-driven: baca `metadata.json`, filter usability (skip scan/partial),
de-spacing PDF text-layer rusak. `chunk_for_vectorstore()` hybrid per-dokumen:
Pasal-split bila ber-Pasal, narrative bila tidak.
"""

from __future__ import annotations

from typing import List

from langchain_core.documents import Document

from ragtrial.config import SOSIAL_PROCESSED_PKL, SOSIAL_RAW_DIR
from ragtrial.sources.base import Source
from ragtrial.sources.sosial.capability import sosial_capability
from ragtrial.sources.sosial.chunk import chunk_for_vectorstore
from ragtrial.sources.sosial.preprocess import preprocess


def build_documents() -> List[Document]:
    """Proses semua dokumen sosial usable jadi Documents per-halaman."""
    return preprocess(SOSIAL_RAW_DIR)


source = Source(
    name="sosial",
    raw_dir=SOSIAL_RAW_DIR,
    processed_pkl=SOSIAL_PROCESSED_PKL,
    capability=sosial_capability,
    build_documents=build_documents,
    chunk=chunk_for_vectorstore,
)
