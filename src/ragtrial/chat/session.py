"""ChatSession — conversation state + orchestration over ANY of the 3 RAG modes.

Pure Python class, zero UI coupling. Wraps a stateless mode entry point
(ask_naive / ask_enhanced / ask_agentic — all return RagResult) with:
  - chat history (in-memory; swap-able for Redis/DB later without API changes)
  - optional query rewriting for follow-up questions
  - history trimming to prevent unbounded growth

UI layers (Streamlit, CLI, future FastAPI) all consume the same ChatSession API.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Literal

from ragtrial.chat.rewriter import rewrite_query
from ragtrial.llm import llm as default_llm
from ragtrial.rag.agentic import ask_agentic
from ragtrial.rag.enhanced import ask_enhanced
from ragtrial.rag.naive import ask_naive
from ragtrial.result import RagResult

Role = Literal["user", "assistant"]
Mode = Literal["naive", "enhanced", "agentic"]

_MODE_FN: Dict[str, Callable[..., RagResult]] = {
    "naive": ask_naive,
    "enhanced": ask_enhanced,
    "agentic": ask_agentic,
}


@dataclass
class Turn:
    role: Role
    content: str
    timestamp: float = field(default_factory=time.time)


class ChatSession:
    """Stateful multi-turn conversation over a chosen RAG mode.

    Parameters
    ----------
    mode
        Which RAG pipeline to use: 'naive' | 'enhanced' | 'agentic'. Default 'enhanced'.
    max_history_turns
        Round-trip turns (user+assistant = 1) to keep; older dropped FIFO.
    rewrite_followups
        If True, follow-ups are rewritten into standalone queries before the
        pipeline runs (resolves pronouns/references via recent history).
    """

    def __init__(
        self,
        mode: Mode = "enhanced",
        max_history_turns: int = 5,
        rewrite_followups: bool = True,
    ):
        if mode not in _MODE_FN:
            raise ValueError(f"mode harus salah satu dari {list(_MODE_FN)}, dapat {mode!r}")
        self.mode = mode
        self._ask = _MODE_FN[mode]
        self.max_history_turns = max_history_turns
        self.rewrite_followups = rewrite_followups
        self.history: List[Turn] = []

    def ask(self, user_message: str, verbose: bool = False) -> Dict[str, Any]:
        """Process one user turn. Returns a UI-friendly dict (not RagResult)."""
        t_total0 = time.perf_counter()

        t_r0 = time.perf_counter()
        if self.rewrite_followups and self.history:
            rewritten = rewrite_query(self.history, user_message, default_llm)
        else:
            rewritten = user_message
        t_rewrite = time.perf_counter() - t_r0

        result: RagResult = self._ask(rewritten, verbose=verbose)

        self.history.append(Turn(role="user", content=user_message))
        self.history.append(Turn(role="assistant", content=result.answer))
        self._trim_history()

        return {
            "original_query": user_message,
            "rewritten_query": rewritten,
            "answer": result.answer,
            "documents": result.documents,
            "source_used": result.source_used,
            "mode": result.mode,
            "timings": {
                "rewrite": t_rewrite,
                "retrieve": result.timings.get("retrieve", 0.0),
                "generate": result.timings.get("generate", 0.0),
                "total": time.perf_counter() - t_total0,
            },
            "meta": result.meta,
        }

    def reset(self) -> None:
        self.history = []

    def get_history(self) -> List[Turn]:
        return list(self.history)

    def _trim_history(self) -> None:
        max_entries = self.max_history_turns * 2
        if len(self.history) > max_entries:
            self.history = self.history[-max_entries:]
