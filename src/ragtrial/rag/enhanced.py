"""Enhanced RAG — a FIXED pipeline assembled from a config dataclass.

Canonical Enhanced (on the unified index): intent gate -> retrieve GLOBAL top-N
(hybrid) -> cross-encoder rerank to top_n -> generate. No domain routing, no query
rewriting, NO LLM in the control flow (that is what makes it 'enhanced', not
agentic). Every valid query follows the identical path.

    cfg = EnhancedRAGConfig(retrieval="hybrid", reranker="cross_encoder")
    rag = build_enhanced(cfg)
    result = rag.ask("...")          # -> RagResult

The retrieval stack is the only lever, exposed as a 2x2 ablation (hybrid {on,off} x
rerank {on,off}) via PRESETS — that is the clean "does the retrieval stack pay off?"
experiment. Domain routers (route.py) and HyDE (rewrite.py) remain in the tree as
optional ablation plugs, but are OFF in the default.
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import List, Optional

from langchain_core.documents import Document

from ragtrial.pipeline import GenerateStage, Pipeline, RERANKERS, ROUTERS, REWRITERS, RetrieveStage
from ragtrial.pipeline.intent import INTENT_GATES
from ragtrial.result import RagResult

_GLOBAL_ROUTES = {None, "all", "both"}


@dataclass
class EnhancedRAGConfig:
    intent: str = "semantic"        # none | semantic (VALID/INVALID gate)
    rewriter: str = "passthrough"   # passthrough (default) | hyde | multiquery*
    router: str = "none"            # none (global fan-out) | semantic | llm  (ablation)
    retrieval: str = "hybrid"       # dense | hybrid
    reranker: str = "cross_encoder" # none | cross_encoder
    k_candidates: int = 10          # global candidate pool retrieved (into rerank)
    top_n: int = 5                  # final docs after rerank -> generator


# Presets. Default = canonical (intent + hybrid + cross-encoder rerank, global).
# The 4 retrieval-stack ablation cells (hybrid x rerank) share an identical
# candidate pool — only the two switches differ, so the deltas are attributable.
PRESETS: dict[str, EnhancedRAGConfig] = {
    "default": EnhancedRAGConfig(),                                             # hybrid + rerank
    "dense_only": EnhancedRAGConfig(retrieval="dense", reranker="none"),        # neither (~naive stack)
    "hybrid_no_rerank": EnhancedRAGConfig(retrieval="hybrid", reranker="none"), # hybrid only
    "rerank_no_hybrid": EnhancedRAGConfig(retrieval="dense", reranker="cross_encoder"),  # rerank only
    "no_intent": EnhancedRAGConfig(intent="none"),                              # ablation: intent gate off
}


def _sources_in(docs: List[Document]) -> List[str]:
    out: List[str] = []
    for d in docs:
        s = (d.metadata or {}).get("_source")
        if s and s not in out:
            out.append(s)
    return out


class EnhancedRAG:
    """A built pipeline + the config that produced it."""

    def __init__(self, config: EnhancedRAGConfig, pipeline: Pipeline):
        self.config = config
        self.pipeline = pipeline

    def _decisions(self, state) -> dict:
        cfg = self.config
        route = state.route
        routing = "global" if route in _GLOBAL_ROUTES else route
        return {
            "intent": "direct" if state.intent == "invalid" else "retrieve",
            "rewrite": cfg.rewriter != "passthrough",
            "routing": routing,
            "retrieval": cfg.retrieval,
            "rerank": cfg.reranker != "none",
            "iterations": 1,
        }

    def ask(self, question: str, verbose: bool = True) -> RagResult:
        t0 = time.perf_counter()
        state = self.pipeline.run(question)
        total = time.perf_counter() - t0

        srcs = _sources_in(state.documents)
        source_used = "none" if not srcs else (srcs[0] if len(srcs) == 1 else "both")

        result = RagResult(
            question=question,
            answer=state.answer,
            documents=state.documents,
            query=state.query,
            route=state.route if state.route is not None else "global",
            source_used=source_used,
            mode="enhanced",
            timings={**state.timings, "total": total},
            meta={**state.meta, "config": self.config.__dict__},
            decisions=self._decisions(state),
        )

        if verbose:
            t = result.timings
            print(f"Q: {question}")
            if state.query != question:
                print(f"   Rewritten: {state.query}")
            print(f"   Route: {result.route}  ({state.meta.get('route_reason', '')})")
            print(f"   Source used: {source_used}  | docs: {len(state.documents)}")
            print("   Timing — " + " | ".join(f"{k}: {v:.2f}s" for k, v in t.items()))
            print(f"   Answer: {state.answer[:400]}{'...' if len(state.answer) > 400 else ''}\n")
        return result


def build_enhanced(config: Optional[EnhancedRAGConfig] = None) -> EnhancedRAG:
    """Assemble an EnhancedRAG from a config (defaults to the canonical pipeline)."""
    cfg = config or EnhancedRAGConfig()

    # intent gate FIRST (retrieve-or-not on the original question), then route
    # (default none=global), then rewrite (default passthrough), retrieve a global
    # candidate pool, rerank to top_n, generate.
    stages = []
    intent_cls = INTENT_GATES[cfg.intent]
    if intent_cls is not None:
        stages.append(intent_cls())
    stages += [
        ROUTERS[cfg.router](),
        REWRITERS[cfg.rewriter](),
        RetrieveStage(strategy=cfg.retrieval, k=cfg.k_candidates),
        RERANKERS[cfg.reranker](top_n=cfg.top_n),
        GenerateStage(),
    ]
    return EnhancedRAG(cfg, Pipeline(stages))


_default_rag: Optional[EnhancedRAG] = None


def ask_enhanced(
    question: str,
    verbose: bool = True,
    config: Optional[EnhancedRAGConfig] = None,
) -> RagResult:
    """Convenience entry point. Reuses a lazily-built default unless `config` given."""
    global _default_rag
    if config is not None:
        return build_enhanced(config).ask(question, verbose=verbose)
    if _default_rag is None:
        _default_rag = build_enhanced()
    return _default_rag.ask(question, verbose=verbose)
