"""Rerank stages — reorder/trim retrieved docs before generation.

Implemented:  none (NoopReranker) — passthrough, optional top-N trim.
Stubbed (TODO): cross_encoder — BGE/cross-encoder model (sentence-transformers
is already a dependency); slot exists + registered.
"""

from __future__ import annotations

from typing import Optional

from ragtrial.pipeline.base import RagState, Stage


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
    """Cross-encoder rerank (e.g. BAAI/bge-reranker). Encoder model, not an LLM.

    TODO(enhanced): load a sentence-transformers CrossEncoder, score
    (query, doc) pairs, sort desc, keep top_n. Set state.meta['rerank_scores'].
    """

    name = "rerank"

    def __init__(self, top_n: int = 5, model_name: str = "BAAI/bge-reranker-v2-m3"):
        self.top_n = top_n
        self.model_name = model_name

    def run(self, state: RagState) -> RagState:
        raise NotImplementedError(
            "CrossEncoderReranker belum diimplementasi (Tahap lanjutan)."
        )


RERANKERS: dict[str, type[Stage]] = {
    "none": NoopReranker,
    "cross_encoder": CrossEncoderReranker,
}
