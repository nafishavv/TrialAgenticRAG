"""Naive Combined RAG — fan out query to every SEARCHABLE capability, concat docs.

Replaces the hardcoded 2-store implementation in notebook/rag_chat_main.py.
Adding a new searchable capability automatically extends this pipeline.
"""

from __future__ import annotations

import time
from typing import Any, Dict, List

from langchain_core.documents import Document

from ragtrial.capabilities import format_context
from ragtrial.capabilities.registry import CAPABILITIES, SEARCHABLE_CAPABILITIES
from ragtrial.llm import llm
from ragtrial.rag.prompts import (
    PROMPT_COMBINED,
    build_citation_rules,
    build_sources_brief,
)


def retrieve_combined(query: str, k_per_source: int = 4) -> List[Document]:
    """Run every searchable capability, concat results in registry order."""
    docs: List[Document] = []
    for cap in SEARCHABLE_CAPABILITIES.values():
        docs.extend(cap.invoke(query, k=k_per_source))
    return docs


def ask_main(
    question: str,
    k_per_source: int = 4,
    verbose: bool = True,
) -> Dict[str, Any]:
    """Naive combined RAG entry point.

    Returns dict with `question, source_used, documents, answer, timings`.
    """
    t0 = time.perf_counter()

    t_r0 = time.perf_counter()
    docs = retrieve_combined(question, k_per_source=k_per_source)
    t_retrieve = time.perf_counter() - t_r0

    t_g0 = time.perf_counter()
    ctx = format_context(docs, CAPABILITIES)
    prompt = PROMPT_COMBINED.format(
        sources_brief=build_sources_brief(SEARCHABLE_CAPABILITIES),
        citation_rules=build_citation_rules(SEARCHABLE_CAPABILITIES),
        context=ctx,
        question=question,
    )
    answer = llm.invoke(prompt).content
    t_generate = time.perf_counter() - t_g0

    total = time.perf_counter() - t0
    result = {
        "question": question,
        "source_used": "combined",
        "documents": docs,
        "answer": answer,
        "timings": {
            "route": 0.0,
            "retrieve": t_retrieve,
            "generate": t_generate,
            "total": total,
        },
    }

    if verbose:
        t = result["timings"]
        counts = {
            name: sum(1 for d in docs if d.metadata.get("_source") == name)
            for name in SEARCHABLE_CAPABILITIES
        }
        print(f"Q: {question}")
        print(
            f"   Docs: {' + '.join(f'{v} {k}' for k, v in counts.items())} = {len(docs)} total"
        )
        print(
            f"   Timing — retrieve: {t['retrieve']:.2f}s | "
            f"generate: {t['generate']:.2f}s | TOTAL: {t['total']:.2f}s"
        )
        print(f"   Answer: {answer[:400]}{'...' if len(answer) > 400 else ''}\n")
    return result
