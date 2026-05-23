"""Dukcapil domain — administrasi kependudukan (Buku Saku Dafduk Capil).

NOTE: `preprocess()` page ranges are calibrated to the 2023 Buku Saku PDF. When
this domain grows to multiple/heterogeneous PDFs, give each its own handler here
(dispatch by filename/type) rather than applying one page map to every file.
"""

from __future__ import annotations

from typing import List

from langchain_core.documents import Document

from ragtrial.config import DUKCAPIL_PROCESSED_PKL, DUKCAPIL_RAW_DIR
from ragtrial.sources.base import Source
from ragtrial.sources.dukcapil.capability import dukcapil_capability
from ragtrial.sources.dukcapil.chunk import chunk_for_vectorstore
from ragtrial.sources.dukcapil.preprocess import preprocess


def build_documents() -> List[Document]:
    """Parse every PDF under the domain's raw dir into cleaned Documents."""
    docs: List[Document] = []
    for pdf in sorted(DUKCAPIL_RAW_DIR.glob("*.pdf")):
        docs.extend(preprocess(pdf))
    return docs


source = Source(
    name="dukcapil",
    raw_dir=DUKCAPIL_RAW_DIR,
    processed_pkl=DUKCAPIL_PROCESSED_PKL,
    capability=dukcapil_capability,
    build_documents=build_documents,
    chunk=chunk_for_vectorstore,
)
