"""Naive RAG — the honest baseline.

ONE unified collection (all domains), dense top-k similarity search, stuff the
chunks into a generic prompt, single LLM call. No hybrid, no router, no rewrite,
no rerank — that is the point: a floor to measure enhanced/agentic against.

Build the unified collection first:
    uv run python scripts/build_vectorstore.py --source unified
"""

from __future__ import annotations

import time
from typing import List, Optional

from langchain_chroma import Chroma
from langchain_core.documents import Document

from ragtrial.config import UNIFIED_COLLECTION, UNIFIED_VECTOR_STORE
from ragtrial.llm import embeddings, llm
from ragtrial.rag.prompts import PROMPT_NAIVE
from ragtrial.result import RagResult

_store: Optional[Chroma] = None


def _get_store() -> Chroma:
    global _store
    if _store is None:
        _store = Chroma(
            collection_name=UNIFIED_COLLECTION,
            embedding_function=embeddings,
            persist_directory=str(UNIFIED_VECTOR_STORE),
        )
    return _store


def _stuff(docs: List[Document]) -> str:
    """Plain numbered concat — no source-aware headers (stays truly naive)."""
    return "\n\n---\n\n".join(f"[{i}] {d.page_content}" for i, d in enumerate(docs, 1))


def ask_naive(question: str, k: int = 8, verbose: bool = True) -> RagResult:
    t0 = time.perf_counter()

    t_r0 = time.perf_counter()
    docs = _get_store().similarity_search(question, k=k)
    t_retrieve = time.perf_counter() - t_r0

    t_g0 = time.perf_counter()
    answer = llm.invoke(PROMPT_NAIVE.format(context=_stuff(docs), question=question)).content
    t_generate = time.perf_counter() - t_g0

    result = RagResult(
        question=question,
        answer=answer,
        documents=docs,
        query=question,
        route="",
        source_used="unified",
        mode="naive",
        timings={
            "route": 0.0,
            "retrieve": t_retrieve,
            "generate": t_generate,
            "total": time.perf_counter() - t0,
        },
    )

    if verbose:
        t = result.timings
        print(f"Q: {question}")
        print(f"   Docs: {len(docs)} (unified, dense top-{k})")
        print(
            f"   Timing — retrieve: {t['retrieve']:.2f}s | "
            f"generate: {t['generate']:.2f}s | TOTAL: {t['total']:.2f}s"
        )
        print(f"   Answer: {answer[:400]}{'...' if len(answer) > 400 else ''}\n")
    return result
