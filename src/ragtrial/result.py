"""RagResult — the single output contract returned by ALL three RAG modes.

naive / enhanced / agentic each return a RagResult. Eval, chat, and UI consume
ONLY this contract — they never reach into mode internals. That decoupling is
what lets the eval framework (or the UI) change without touching the pipelines,
and lets a new mode slot in as long as it returns a RagResult.

Mode-specific extras (router reason, rewritten query, agent step trace, ...)
live in `meta` so the top-level shape stays stable across modes.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Union

from langchain_core.documents import Document

CANONICAL_TIMING_KEYS = (
    "intent", "route", "rewrite", "retrieve", "rerank",
    "t_orchestrate", "t_generate", "generate", "total",
)
"""Shared timing vocabulary across the 3 modes (eval/analyze.py reads these with
.get(key, 0.0)). A mode only emits the keys for stages it actually ran — absent
stages are OMITTED, never zero-filled. `rewrite_followup` (conversational rewrite)
is a ChatSession-level timing, not a pipeline one."""


def sources_in(docs: List[Document]) -> List[str]:
    """Ordered unique `_source` tags across docs (single shared implementation)."""
    out: List[str] = []
    for d in docs:
        s = (d.metadata or {}).get("_source")
        if s and s not in out:
            out.append(s)
    return out


def collapse_sources(srcs: List[str]) -> str:
    """Collapse a source list to the `source_used` label: 'none' | <single> | 'both'."""
    if not srcs:
        return "none"
    return srcs[0] if len(srcs) == 1 else "both"


def make_decisions(
    *,
    intent: str = "retrieve",
    rewrite: bool = False,
    routing: Union[str, List[str]] = "global",
    retrieval: str = "dense",
    rerank: bool = False,
    iterations: int = 1,
) -> Dict[str, Any]:
    """Keyword-only builder for the normalized `decisions` log.

    Every mode builds its decisions through this, so the key set can never drift
    between pipelines (a typo'd key becomes a TypeError instead of silent skew).
    """
    return {
        "intent": intent,
        "rewrite": rewrite,
        "routing": routing,
        "retrieval": retrieval,
        "rerank": rerank,
        "iterations": iterations,
    }


@dataclass
class RagResult:
    question: str
    """Original user question, as typed."""
    answer: str = ""
    documents: List[Document] = field(default_factory=list)
    query: str = ""
    """Effective query used for retrieval (== question unless rewritten)."""
    route: str = ""
    """Domain(s) chosen: a capability name, 'both', 'none', or '' (n/a)."""
    source_used: str = ""
    """Which capabilities actually contributed context."""
    mode: str = ""
    """'naive' | 'enhanced' | 'agentic'."""
    timings: Dict[str, float] = field(default_factory=dict)
    meta: Dict[str, Any] = field(default_factory=dict)
    """Mode-specific extras: route_reason, rewritten_query, steps (agent trace), ..."""
    decisions: Dict[str, Any] = field(default_factory=dict)
    """Normalized execution log — SAME keys across all 3 modes for uniform
    visualization/debugging. Schema: intent ('retrieve'|'direct'), rewrite (bool),
    routing ('global'|domain|[domains]|'none'), retrieval ('dense'|'hybrid'),
    rerank (bool), iterations (int). Built from existing state; `meta` keeps the
    rich per-mode detail."""

    def __post_init__(self) -> None:
        if not self.query:
            self.query = self.question

    def to_dict(self) -> Dict[str, Any]:
        """Flat dict with legacy top-level keys + meta merged in.

        Keeps existing dict-based consumers (eval/run_eval, app.py) working while
        callers migrate to attribute access.
        """
        d: Dict[str, Any] = {
            "question": self.question,
            "answer": self.answer,
            "documents": self.documents,
            "query": self.query,
            "route": self.route,
            "source_used": self.source_used,
            "mode": self.mode,
            "timings": self.timings,
            "meta": self.meta,
            "decisions": self.decisions,
        }
        # Surface common meta keys at top level for back-compat.
        for k in ("route_reason", "rewritten_query", "original_query", "steps"):
            if k in self.meta:
                d[k] = self.meta[k]
        return d
