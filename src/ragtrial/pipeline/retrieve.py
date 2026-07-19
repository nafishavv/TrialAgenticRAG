"""Retrieval stage — fetch a candidate pool from the unified store.

Enhanced retrieves GLOBALLY (no domain routing) by default: the router is 'none',
so `state.route` is None and we search the whole unified index. If a router IS
plugged in (ablation), `state.route` maps to a domain filter:

  - None / 'all' / 'both'  -> global (no filter)
  - 'none'                 -> retrieve nothing (out-of-scope; generator refuses)
  - '<domain>'             -> filter to that domain

`k` is the CANDIDATE POOL size (into rerank); the reranker trims to the final
top_n. `strategy` (dense|hybrid) is passed through to the unified store.
"""

from __future__ import annotations

from typing import Optional

from ragtrial.pipeline.base import RagState, Stage
from ragtrial.vectorstore.store import unified_store

_GLOBAL_ROUTES = {None, "all", "both"}


class RetrieveStage(Stage):
    name = "retrieve"

    def __init__(self, strategy: str = "hybrid", k: int = 20):
        self.strategy = strategy
        """'dense' | 'hybrid'."""
        self.k = k
        """Candidate pool size retrieved (before rerank trims to top_n)."""

    def _domain(self, route: Optional[str]) -> Optional[str]:
        return None if route in _GLOBAL_ROUTES else route

    def run(self, state: RagState) -> RagState:
        if state.intent == "invalid":
            state.documents = []
            return state
        route = state.route
        if route == "none":
            state.documents = []
            return state
        # Capture per-stage sub-timings (embed/search/bm25/fuse) alongside the
        # coarse `retrieve` stage timing the Pipeline records around this call.
        sub: dict = {}
        state.documents = unified_store.search(
            state.query, k=self.k, strategy=self.strategy,
            domain=self._domain(route), timings=sub,
        )
        for key, val in sub.items():
            state.timings[key] = state.timings.get(key, 0.0) + val
        return state
