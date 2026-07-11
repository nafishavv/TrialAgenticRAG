"""Naive RAG — the honest baseline.

Canonical minimal RAG: a single dense top-k similarity search over the ONE unified
collection (all domains), stuffed into a generic prompt. No router, no rewrite, no
rerank, no hybrid, no intent gate. The unified index is the SAME store enhanced and
agentic read (controlled variable); naive differs only by being the plainest path —
dense retrieve → stuff → generate.
"""

from __future__ import annotations

import time
from typing import List

from langchain_core.documents import Document

from ragtrial.llm import llm, invoke_with_retry
from ragtrial.rag.prompts import PROMPT_NAIVE
from ragtrial.result import RagResult
from ragtrial.vectorstore.store import unified_store


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


def ask_naive(question: str, k: int = 5, verbose: bool = True) -> RagResult:
    t0 = time.perf_counter()

    t_r0 = time.perf_counter()
    docs = unified_store.search(question, k=k, strategy="dense")  # global, no filter
    t_retrieve = time.perf_counter() - t_r0

    t_g0 = time.perf_counter()
    answer = invoke_with_retry(llm, PROMPT_NAIVE.format(context=_stuff(docs), question=question)).content
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
        # No intent handling — naive always retrieves (baseline/control group).
        meta={"intent": "valid"},
        decisions={
            "intent": "retrieve",   # naive never refuses
            "rewrite": False,
            "routing": "global",    # unified, no domain filter
            "retrieval": "dense",
            "rerank": False,
            "iterations": 1,
        },
    )

    if verbose:
        t = result.timings
        print(f"Q: {question}")
        print(f"   Docs: {len(docs)} (unified dense top-{k})")
        print(
            f"   Timing — retrieve: {t['retrieve']:.2f}s | "
            f"generate: {t['generate']:.2f}s | TOTAL: {t['total']:.2f}s"
        )
        print(f"   Answer: {answer[:400]}{'...' if len(answer) > 400 else ''}\n")
    return result
