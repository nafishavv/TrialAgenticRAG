"""Retrieval stage — fetch context per the router's decision.

route semantics:
  - None / 'all' / 'both'  -> fan out to every searchable capability
  - 'none'                 -> retrieve nothing (out-of-scope; generator refuses)
  - '<capability name>'    -> that single capability

`strategy` (dense|hybrid) is forwarded to each capability's invoke(), so Enhanced
can run dense retrieval even when a capability defaults to hybrid.
"""

from __future__ import annotations

from typing import Dict, List, Optional

from langchain_core.documents import Document

from ragtrial.capabilities.base import Capability
from ragtrial.capabilities.registry import SEARCHABLE_CAPABILITIES
from ragtrial.pipeline.base import RagState, Stage

_FANOUT_ROUTES = {None, "all", "both"}


class RetrieveStage(Stage):
    name = "retrieve"

    def __init__(
        self,
        capabilities: Optional[Dict[str, Capability]] = None,
        strategy: Optional[str] = None,
        k: int = 5,
        k_per_source: int = 4,
    ):
        self.caps = capabilities or SEARCHABLE_CAPABILITIES
        self.strategy = strategy
        """None -> each capability uses its own default; else 'dense'|'hybrid'."""
        self.k = k
        """top-k for single-domain retrieval."""
        self.k_per_source = k_per_source
        """top-k per capability when fanning out."""

    def run(self, state: RagState) -> RagState:
        if state.intent == "invalid":
            # Intent gate decided no retrieval — answer directly downstream.
            state.documents = []
            return state
        route = state.route
        if route == "none":
            state.documents = []
        elif route in self.caps:
            state.documents = self.caps[route].invoke(
                state.query, k=self.k, strategy=self.strategy
            )
        elif route in _FANOUT_ROUTES:
            docs: List[Document] = []
            for cap in self.caps.values():
                docs.extend(
                    cap.invoke(state.query, k=self.k_per_source, strategy=self.strategy)
                )
            state.documents = docs
        else:
            # Unknown route value -> safest is no retrieval.
            state.documents = []
        return state
