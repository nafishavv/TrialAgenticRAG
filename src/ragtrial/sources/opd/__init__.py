"""OPD domain — direktori Organisasi Perangkat Daerah (alamat/kontak/struktur)."""

from __future__ import annotations

from typing import List

from langchain_core.documents import Document

from ragtrial.config import OPD_PROCESSED_PKL, OPD_RAW_DIR
from ragtrial.sources.base import Source
from ragtrial.sources.opd.capability import opd_capability
from ragtrial.sources.opd.chunk import chunk_for_vectorstore
from ragtrial.sources.opd.preprocess import preprocess


def build_documents() -> List[Document]:
    """Parse every PDF under the domain's raw dir into atomic OPD Documents."""
    docs: List[Document] = []
    for pdf in sorted(OPD_RAW_DIR.glob("*.pdf")):
        docs.extend(preprocess(pdf))
    return docs


source = Source(
    name="opd",
    raw_dir=OPD_RAW_DIR,
    processed_pkl=OPD_PROCESSED_PKL,
    capability=opd_capability,
    build_documents=build_documents,
    chunk=chunk_for_vectorstore,
)
