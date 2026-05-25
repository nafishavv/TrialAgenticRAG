"""Rerank stages — reorder/trim retrieved docs before generation.

Implemented:  none (NoopReranker) — passthrough, optional top-N trim.
Stubbed (TODO): cross_encoder — slot exists + registered. Implementation drafted but
DEFERRED: jina-reranker-v2 needs transformers ~4.4x (removed fn in transformers 5.x);
pick a native multilingual model (e.g. BAAI/bge-reranker-v2-m3) when wiring it in.
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
    (query, doc) pairs against meta['original_query'], sort desc, keep top_n,
    set state.meta['rerank_scores']. Use a native-transformers model (bge-reranker-*)
    — jina-reranker-v2 relies on remote code incompatible with transformers 5.x.
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
