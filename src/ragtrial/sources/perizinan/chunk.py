"""Perizinan chunking — identity. Tiap perizinan sudah atomic, tidak perlu dipecah."""

from __future__ import annotations

from typing import List

from langchain_core.documents import Document


def chunk_for_vectorstore(docs: List[Document]) -> List[Document]:
    return list(docs)
