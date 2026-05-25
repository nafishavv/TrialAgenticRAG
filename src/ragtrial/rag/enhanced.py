"""Enhanced RAG — a FIXED pipeline assembled from a config dataclass.

The developer designs the path; every query flows through the same stages, with
NO LLM deciding the control flow (that is what makes it 'enhanced', not agentic).
Swap a component by changing one field on EnhancedRAGConfig — build_enhanced()
looks each up in the stage factory dicts.

    cfg = EnhancedRAGConfig(rewriter="hyde", router="semantic",
                            retrieval="dense", reranker="none")
    rag = build_enhanced(cfg)
    result = rag.ask("...")          # -> RagResult

Default = canonical enhanced: semantic(embedding) route -> HyDE rewrite ->
dense retrieve -> no rerank -> generate. Cross-encoder rerank is deferred (stub).
Presets reproduce the retired modules' behavior so nothing is lost (see PRESETS).
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import List, Optional

from langchain_core.documents import Document

from ragtrial.pipeline import GenerateStage, Pipeline, RERANKERS, ROUTERS, REWRITERS, RetrieveStage
from ragtrial.result import RagResult


@dataclass
class EnhancedRAGConfig:
    rewriter: str = "hyde"          # passthrough | hyde | multiquery*
    router: str = "semantic"        # none | semantic | llm
    retrieval: str = "dense"        # dense | hybrid
    reranker: str = "none"          # none | cross_encoder* (rerank deferred)
    k: int = 5                      # top-k for single-domain retrieval
    k_per_source: int = 4           # top-k per capability when fanning out
    rerank_top_n: Optional[int] = None


# Presets. Default = canonical enhanced (HyDE + semantic + dense). Reranker deferred.
PRESETS: dict[str, EnhancedRAGConfig] = {
    "default": EnhancedRAGConfig(),
    "no_hyde": EnhancedRAGConfig(rewriter="passthrough"),                         # ablation: HyDE off
    "fanout_hybrid": EnhancedRAGConfig(router="none", retrieval="hybrid"),       # ~ old naive_combined
    "llm_router_hybrid": EnhancedRAGConfig(router="llm", retrieval="hybrid"),    # ~ old static agentic
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
            route=state.route if state.route is not None else "all",
            source_used=source_used,
            mode="enhanced",
            timings={**state.timings, "total": total},
            meta={**state.meta, "config": self.config.__dict__},
        )

        if verbose:
            t = result.timings
            print(f"Q: {question}")
            if state.query != question:
                print(f"   Rewritten: {state.query}")
            print(f"   Route: {result.route}  ({state.meta.get('route_reason', '')})")
            print(f"   Source used: {source_used}  | docs: {len(state.documents)}")
            print(
                "   Timing — "
                + " | ".join(f"{k}: {v:.2f}s" for k, v in t.items())
            )
            print(f"   Answer: {state.answer[:400]}{'...' if len(state.answer) > 400 else ''}\n")
        return result


def build_enhanced(config: Optional[EnhancedRAGConfig] = None) -> EnhancedRAG:
    """Assemble an EnhancedRAG from a config (defaults to the canonical pipeline)."""
    cfg = config or EnhancedRAGConfig()

    rerank_cls = RERANKERS[cfg.reranker]
    reranker = (
        rerank_cls(top_n=cfg.rerank_top_n) if cfg.rerank_top_n is not None else rerank_cls()
    )

    # route BEFORE rewrite: the router must classify the original question, not a
    # fabricated HyDE passage. Rewrite then reshapes the query for retrieval only.
    stages = [
        ROUTERS[cfg.router](),
        REWRITERS[cfg.rewriter](),
        RetrieveStage(strategy=cfg.retrieval, k=cfg.k, k_per_source=cfg.k_per_source),
        reranker,
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
