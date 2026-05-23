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
from typing import Any, Dict, List

from langchain_core.documents import Document


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
        }
        # Surface common meta keys at top level for back-compat.
        for k in ("route_reason", "rewritten_query", "original_query", "steps"):
            if k in self.meta:
                d[k] = self.meta[k]
        return d
