"""Routing stages — decide which domain(s) a query goes to. Fixed step (no agent).

Implemented:
  - none      NoRouter      -> no decision; retrieve fans out to ALL searchable caps.
  - semantic  SemanticRouter-> embedding similarity vs per-domain centroid (NOT an
                               LLM call). The canonical Enhanced default.
  - llm       LLMRouter     -> one-shot LLM classification (ported from the old
                               agentic router; kept as an opt-in alternative).

`state.route` ends as a capability name, 'both', 'none' (out-of-scope), or None
(NoRouter -> downstream retrieves from all).
"""

from __future__ import annotations

import json
from typing import Dict, Optional

import numpy as np

from ragtrial.capabilities.base import Capability
from ragtrial.capabilities.registry import SEARCHABLE_CAPABILITIES
from ragtrial.llm import embeddings as default_embeddings
from ragtrial.llm import llm as default_llm
from ragtrial.pipeline.base import RagState, Stage
from ragtrial.rag.prompts import build_router_prompt


class NoRouter(Stage):
    """No routing decision — retrieve fans out to every searchable capability."""

    name = "route"

    def run(self, state: RagState) -> RagState:
        state.route = None
        state.meta["route_reason"] = "no-router (fan-out all)"
        return state


class SemanticRouter(Stage):
    """Embedding classifier: cosine(query, per-domain centroid). No LLM call.

    Centroid per domain = mean embedding of its description + router_examples,
    computed lazily on first query and cached.
    """

    name = "route"

    def __init__(
        self,
        capabilities: Optional[Dict[str, Capability]] = None,
        embedder=None,
        threshold: float = 0.55,
        both_margin: float = 0.04,
    ):
        self.caps = capabilities or SEARCHABLE_CAPABILITIES
        self.embedder = embedder or default_embeddings
        self.threshold = threshold
        """Below this top similarity -> route 'none' (out of scope)."""
        self.both_margin = both_margin
        """If top-2 domains are within this gap (both above threshold) -> 'both'."""
        self._centroids: Optional[Dict[str, np.ndarray]] = None

    def _ensure_centroids(self) -> None:
        if self._centroids is not None:
            return
        self._centroids = {}
        for name, cap in self.caps.items():
            texts = [cap.description] + list(getattr(cap, "router_examples", []) or [])
            vecs = self.embedder.embed_documents(texts)
            self._centroids[name] = np.mean(np.asarray(vecs, dtype=float), axis=0)

    @staticmethod
    def _cosine(a: np.ndarray, b: np.ndarray) -> float:
        denom = (np.linalg.norm(a) * np.linalg.norm(b)) or 1.0
        return float(np.dot(a, b) / denom)

    def run(self, state: RagState) -> RagState:
        self._ensure_centroids()
        qv = np.asarray(self.embedder.embed_query(state.query), dtype=float)
        sims = {name: self._cosine(qv, c) for name, c in self._centroids.items()}
        ranked = sorted(sims.items(), key=lambda kv: -kv[1])
        best, best_s = ranked[0]

        if best_s < self.threshold:
            state.route = "none"
        elif (
            len(ranked) > 1
            and ranked[1][1] >= self.threshold
            and (best_s - ranked[1][1]) <= self.both_margin
        ):
            state.route = "both"
        else:
            state.route = best

        state.meta["route_reason"] = f"semantic top={best}:{best_s:.3f}"
        state.meta["route_scores"] = {k: round(v, 4) for k, v in sims.items()}
        return state


class LLMRouter(Stage):
    """One-shot LLM classification into a capability name / 'both' / 'none'.

    Ported from the retired static-agentic router. NOT the default — semantic
    routing is preferred for Enhanced; this stays for ablation/comparison.
    """

    name = "route"

    def __init__(self, capabilities: Optional[Dict[str, Capability]] = None, llm=None):
        self.caps = capabilities or SEARCHABLE_CAPABILITIES
        self.llm = llm or default_llm
        self._route_values = set(self.caps.keys()) | {"both", "none"}

    def run(self, state: RagState) -> RagState:
        prompt = build_router_prompt(state.query, self.caps)
        raw = self.llm.invoke(prompt).content.strip()
        if raw.startswith("```"):
            raw = raw.strip("`")
            if raw.startswith("json"):
                raw = raw[4:]
            raw = raw.strip()
        try:
            parsed = json.loads(raw)
            route = parsed.get("route", "none")
            if route not in self._route_values:
                route = "none"
            state.route = route
            state.meta["route_reason"] = parsed.get("reason", "")
        except Exception as e:
            state.route = "none"
            state.meta["route_reason"] = f"parse_error: {e}; raw={raw[:80]}"
        return state


ROUTERS: dict[str, type[Stage]] = {
    "none": NoRouter,
    "semantic": SemanticRouter,
    "llm": LLMRouter,
}
