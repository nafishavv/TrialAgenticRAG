"""Pipeline primitives for Enhanced RAG — Stage ABC, RagState, Pipeline runner.

Enhanced RAG is a FIXED sequence of stages (no LLM flow-control). Each Stage
reads + mutates a RagState threaded through the pipeline. The Pipeline runs them
in order and times each.

Stage order for the canonical enhanced pipeline:
    intent -> rewrite -> retrieve -> rerank -> generate
(intent runs first so it classifies the original question, not a HyDE passage)

Retrieval is always GLOBAL — there is no domain routing. Domain selection exists
only in the agentic tier, where the LLM picks a `search_<domain>` tool.

Adding a capability to a stage (e.g. a new reranker) = implement a Stage
subclass + register it in that stage module's factory dict. `build_enhanced`
(rag/enhanced.py) assembles the list from an EnhancedRAGConfig.
"""

from __future__ import annotations

import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from langchain_core.documents import Document


@dataclass
class RagState:
    """Mutable working state passed through enhanced-pipeline stages."""

    question: str
    """Original user question."""
    query: str = ""
    """Effective query (rewriter sets this; defaults to question)."""
    intent: Optional[str] = None
    """Intent gate decision: 'valid' (retrieve) | 'invalid' (answer directly) | None (no gate)."""
    documents: List[Document] = field(default_factory=list)
    answer: str = ""
    timings: Dict[str, float] = field(default_factory=dict)
    meta: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.query:
            self.query = self.question


class Stage(ABC):
    """One step in an enhanced pipeline. Reads + returns a RagState."""

    name: str = "stage"

    @abstractmethod
    def run(self, state: RagState) -> RagState:
        ...


class Pipeline:
    """Run an ordered list of stages, timing each by its `name`."""

    def __init__(self, stages: List[Stage]):
        self.stages = stages

    def run(self, question: str) -> RagState:
        state = RagState(question=question, query=question)
        for stage in self.stages:
            t0 = time.perf_counter()
            state = stage.run(state)
            dt = time.perf_counter() - t0
            state.timings[stage.name] = state.timings.get(stage.name, 0.0) + dt
        return state

    def __repr__(self) -> str:
        return f"Pipeline({' -> '.join(s.name for s in self.stages)})"
