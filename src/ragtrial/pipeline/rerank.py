"""Rerank stages — reorder/trim retrieved docs before generation.

Implemented:
  - none (NoopReranker) — passthrough, optional top-N trim.
  - cross_encoder (CrossEncoderReranker) — BAAI/bge-reranker-v2-m3, a native
    XLM-RoBERTa cross-encoder (loads via sentence-transformers, NO trust_remote_code).

The scoring itself lives in `rerank_documents()` — a framework-agnostic helper so
BOTH the enhanced pipeline Stage AND the agentic tools node share ONE reranker
(no drift, fair comparison). The Stage is a thin RagState wrapper around it.
"""

from __future__ import annotations

import os
from typing import List, Optional, Tuple

from langchain_core.documents import Document

from ragtrial.pipeline.base import RagState, Stage

DEFAULT_RERANK_MODEL = os.getenv("RERANK_MODEL", "BAAI/bge-reranker-v2-m3")
"""Override per run with $RERANK_MODEL (e.g. BAAI/bge-reranker-base — 278M, ~2x
faster on CPU than the 560M v2-m3, standard arch so it loads cleanly)."""
DEFAULT_MAX_LENGTH = 256
"""Truncation length for (query, doc) pairs. bge-v2-m3 is a 560M model on CPU;
512 is ~2x slower than 256 with little rerank gain (legal signal is early in the
chunk). 128 ~2x faster again but risks cutting pasal bodies. 256 = the balance."""

# Reranker backend: 'local' (bge cross-encoder on CPU, free/reproducible, ~4-8s) or
# 'jina' (Jina Reranker API on their GPU, ~0.7s but paid/networked). Default local;
# opt into jina by setting RERANK_BACKEND=jina + JINA_API_KEY in .env.
JINA_RERANK_URL = "https://api.jina.ai/v1/rerank"
DEFAULT_JINA_MODEL = "jina-reranker-v2-base-multilingual"

# Lazy cache keyed by (model_name, max_length) — the cross-encoder is ~2GB.
_MODELS: dict = {}


def _rerank_jina(query: str, docs, top_n, model: str = None):
    """Rerank via the Jina Reranker API (remote GPU). Needs JINA_API_KEY."""
    import requests

    key = os.getenv("JINA_API_KEY")
    if not key:
        raise RuntimeError("RERANK_BACKEND=jina but JINA_API_KEY is not set (.env).")
    model = model or os.getenv("JINA_RERANK_MODEL", DEFAULT_JINA_MODEL)
    resp = requests.post(
        JINA_RERANK_URL,
        headers={"Authorization": f"Bearer {key}", "Content-Type": "application/json"},
        json={
            "model": model,
            "query": query,
            "documents": [d.page_content for d in docs],
            "top_n": top_n or len(docs),
        },
        timeout=60,
    )
    resp.raise_for_status()
    results = resp.json()["results"]
    return [docs[r["index"]] for r in results], [float(r["relevance_score"]) for r in results]


def _get_cross_encoder(model_name: str, max_length: int = DEFAULT_MAX_LENGTH):
    """Lazily load + cache a sentence-transformers CrossEncoder (heavy import)."""
    key = (model_name, max_length)
    if key not in _MODELS:
        import torch
        from sentence_transformers import CrossEncoder  # local import: torch is heavy

        # Use all CPU cores (default torch threads is often half) — ~2x on CPU.
        torch.set_num_threads(os.cpu_count() or 4)
        _MODELS[key] = CrossEncoder(model_name, max_length=max_length)
    return _MODELS[key]


def rerank_documents(
    query: str,
    docs: List[Document],
    top_n: Optional[int] = None,
    model_name: str = DEFAULT_RERANK_MODEL,
    max_length: int = DEFAULT_MAX_LENGTH,
    backend: Optional[str] = None,
) -> Tuple[List[Document], List[float]]:
    """Score (query, doc) pairs with a cross-encoder, sort desc, keep top_n.

    Shared by enhanced (Stage) and agentic (tools node). Returns the reranked
    documents AND their scores (aligned), so callers can stash scores in meta.
    A no-op on empty input. Backend ('local'|'jina') defaults to $RERANK_BACKEND.
    """
    if not docs:
        return docs, []
    backend = (backend or os.getenv("RERANK_BACKEND", "local")).lower()
    if backend == "jina":
        return _rerank_jina(query, docs, top_n)
    model = _get_cross_encoder(model_name, max_length)
    scores = model.predict([(query, d.page_content) for d in docs])
    ranked = sorted(zip(docs, scores), key=lambda ds: float(ds[1]), reverse=True)
    if top_n is not None:
        ranked = ranked[:top_n]
    return [d for d, _ in ranked], [float(s) for _, s in ranked]


class NoopReranker(Stage):
    """No reranking. Optionally trims to `top_n` to cap generator context."""

    name = "rerank"

    def __init__(self, top_n: Optional[int] = None):
        self.top_n = top_n

    def run(self, state: RagState) -> RagState:
        if self.top_n is not None:
            state.documents = state.documents[: self.top_n]
        return state


class CrossEncoderReranker(Stage):
    """Cross-encoder rerank (BAAI/bge-reranker-v2-m3). An encoder model, not an LLM.

    Scores each retrieved doc against the REAL user question (meta['original_query']
    when a rewriter stashed it — e.g. HyDE — else the raw question), so reranking
    always targets the user's intent even if retrieval used a rewritten query.
    Sorts desc, keeps top_n, records scores in meta['rerank_scores'].
    """

    name = "rerank"

    def __init__(
        self,
        top_n: int = 5,
        model_name: str = DEFAULT_RERANK_MODEL,
        max_length: int = DEFAULT_MAX_LENGTH,
    ):
        self.top_n = top_n
        self.model_name = model_name
        self.max_length = max_length

    def run(self, state: RagState) -> RagState:
        if not state.documents:
            return state
        query = state.meta.get("original_query") or state.question
        docs, scores = rerank_documents(
            query, state.documents, self.top_n, self.model_name, self.max_length
        )
        state.documents = docs
        state.meta["rerank_scores"] = scores
        return state


RERANKERS: dict[str, type[Stage]] = {
    "none": NoopReranker,
    "cross_encoder": CrossEncoderReranker,
}
