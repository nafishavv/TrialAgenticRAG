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
from ragtrial.result import RagResult, collapse_sources, make_decisions, sources_in
from ragtrial.vectorstore.store import unified_store


def _stuff(docs: List[Document]) -> str:
    """Plain numbered concat — no source-aware headers (stays truly naive)."""
    return "\n\n---\n\n".join(f"[{i}] {d.page_content}" for i, d in enumerate(docs, 1))


def ask_naive(question: str, k: int = 5, verbose: bool = True) -> RagResult:
    t0 = time.perf_counter()

    t_r0 = time.perf_counter()
    sub: dict = {}
    docs = unified_store.search(question, k=k, strategy="dense", timings=sub)  # global, no filter
    t_retrieve = time.perf_counter() - t_r0

    t_g0 = time.perf_counter()
    answer = invoke_with_retry(llm, PROMPT_NAIVE.format(context=_stuff(docs), question=question)).content
    t_generate = time.perf_counter() - t_g0

    source_used = collapse_sources(sources_in(docs))

    result = RagResult(
        question=question,
        answer=answer,
        documents=docs,
        query=question,
        route="",
        source_used=source_used,
        mode="naive",
        timings={
            # Sub-timings from the retrieve stage (dense-only -> bm25/fuse = 0.0).
            "t_embed_query": sub.get("t_embed_query", 0.0),
            "t_search": sub.get("t_search", 0.0),
            "t_bm25": sub.get("t_bm25", 0.0),
            "t_fuse": sub.get("t_fuse", 0.0),
            "retrieve": t_retrieve,          # kept as the whole-stage sum (back-compat)
            "generate": t_generate,
            "total": time.perf_counter() - t0,
            "n_llm_calls": 1,                # single generate call, no control-flow LLM
            "n_iterations": 1,
        },
        # No intent handling — naive always retrieves (baseline/control group).
        meta={"intent": "valid"},
        # naive never refuses; unified global search, plain dense.
        decisions=make_decisions(),
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
