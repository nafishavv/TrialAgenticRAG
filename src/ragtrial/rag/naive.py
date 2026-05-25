"""Naive RAG — the honest baseline.

Fans out a dense top-k search to the SAME per-domain collections that enhanced
and agentic use, then merges by distance into a global top-k — no router, no
rewrite, no rerank, no hybrid. The database is identical across all three modes
(controlled variable); naive differs only by searching every domain blindly and
stuffing the merged chunks into a generic prompt.
"""

from __future__ import annotations

import time
from typing import List, Tuple

from langchain_core.documents import Document

from ragtrial.capabilities.registry import SEARCHABLE_CAPABILITIES
from ragtrial.llm import llm
from ragtrial.rag.prompts import PROMPT_NAIVE
from ragtrial.result import RagResult


def _stuff(docs: List[Document]) -> str:
    """Plain numbered concat — no source-aware headers (stays truly naive)."""
    return "\n\n---\n\n".join(f"[{i}] {d.page_content}" for i, d in enumerate(docs, 1))


def _sources_in(docs: List[Document]) -> List[str]:
    out: List[str] = []
    for d in docs:
        s = (d.metadata or {}).get("_source")
        if s and s not in out:
            out.append(s)
    return out


def ask_naive(question: str, k: int = 8, verbose: bool = True) -> RagResult:
    t0 = time.perf_counter()

    t_r0 = time.perf_counter()
    scored: List[Tuple[Document, float]] = []
    for cap in SEARCHABLE_CAPABILITIES.values():
        scored.extend(cap.search_with_scores(question, k=k))
    scored.sort(key=lambda ds: ds[1])  # lower distance = more similar
    docs = [d for d, _ in scored[:k]]
    t_retrieve = time.perf_counter() - t_r0

    t_g0 = time.perf_counter()
    answer = llm.invoke(PROMPT_NAIVE.format(context=_stuff(docs), question=question)).content
    t_generate = time.perf_counter() - t_g0

    srcs = _sources_in(docs)
    source_used = "none" if not srcs else (srcs[0] if len(srcs) == 1 else "both")

    result = RagResult(
        question=question,
        answer=answer,
        documents=docs,
        query=question,
        route="",
        source_used=source_used,
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
        print(f"   Docs: {len(docs)} (fan-out all collections, dense top-{k} merged)")
        print(
            f"   Timing — retrieve: {t['retrieve']:.2f}s | "
            f"generate: {t['generate']:.2f}s | TOTAL: {t['total']:.2f}s"
        )
        print(f"   Answer: {answer[:400]}{'...' if len(answer) > 400 else ''}\n")
    return result
