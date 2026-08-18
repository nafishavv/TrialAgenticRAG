"""Retrieval stage — fetch a candidate pool from the unified store.

Enhanced always retrieves GLOBALLY: the whole unified index, no domain filter.
Same search scope as naive; the difference is `strategy` (hybrid vs naive's
dense) plus the cross-encoder rerank that follows. Domain filtering exists only
in the agentic tier, where the LLM picks a `search_<domain>` tool.

The only way this stage retrieves nothing is an 'invalid' intent — the gate
already decided no retrieval is needed, and the generator answers directly.

`k` is the CANDIDATE POOL size (into rerank); the reranker trims to the final
top_n. `strategy` (dense|hybrid) is passed through to the unified store.
"""

from __future__ import annotations

from ragtrial.pipeline.base import RagState, Stage
from ragtrial.vectorstore.store import unified_store


class RetrieveStage(Stage):
    name = "retrieve"

    def __init__(self, strategy: str = "hybrid", k: int = 20):
        self.strategy = strategy
        """'dense' | 'hybrid'."""
        self.k = k
        """Candidate pool size retrieved (before rerank trims to top_n)."""

    def run(self, state: RagState) -> RagState:
        if state.intent == "invalid":
            state.documents = []
            return state
        # Capture per-stage sub-timings (embed/search/bm25/fuse) alongside the
        # coarse `retrieve` stage timing the Pipeline records around this call.
        sub: dict = {}
        state.documents = unified_store.search(
            state.query, k=self.k, strategy=self.strategy, timings=sub,
        )
        for key, val in sub.items():
            state.timings[key] = state.timings.get(key, 0.0) + val
        return state
