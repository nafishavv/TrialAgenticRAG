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
from ragtrial.result import RagResult

Role = Literal["user", "assistant"]
Mode = Literal["naive", "enhanced", "agentic"]

_VALID_MODES = {"naive", "enhanced", "agentic"}


def _load_mode_fn(mode: str) -> Callable[..., RagResult]:
    if mode == "naive":
        from ragtrial.rag.naive import ask_naive
        return ask_naive
    elif mode == "enhanced":
        from ragtrial.rag.enhanced import ask_enhanced
        return ask_enhanced
    else:
        from ragtrial.rag.agentic import ask_agentic
        return ask_agentic


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
        if mode not in _VALID_MODES:
            raise ValueError(f"mode harus salah satu dari {sorted(_VALID_MODES)}, dapat {mode!r}")
        self.mode = mode
        self._ask: Callable[..., RagResult] | None = None
        self.max_history_turns = max_history_turns
        self.rewrite_followups = rewrite_followups
        self.history: List[Turn] = []

    def _get_ask(self) -> Callable[..., RagResult]:
        if self._ask is None:
            self._ask = _load_mode_fn(self.mode)
        return self._ask

    def ask(self, user_message: str, verbose: bool = False) -> Dict[str, Any]:
        """Process one user turn. Returns a UI-friendly dict (not RagResult)."""
        t_total0 = time.perf_counter()

        t_r0 = time.perf_counter()
        if self.rewrite_followups and self.history:
            from ragtrial.llm import llm as _llm
            rewritten = rewrite_query(self.history, user_message, _llm)
        else:
            rewritten = user_message
        t_rewrite = time.perf_counter() - t_r0

        result: RagResult = self._get_ask()(rewritten, verbose=verbose)

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
