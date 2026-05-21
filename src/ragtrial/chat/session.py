"""ChatSession — conversation state + orchestration on top of naive RAG.

Pure Python class, zero UI coupling. Wraps the stateless `ask_main()` core with:
  - chat history (in-memory; swap-able for Redis/DB later without API changes)
  - optional query rewriting for follow-up questions
  - history trimming to prevent unbounded growth

UI layers (Streamlit, CLI, future FastAPI) all consume the same ChatSession API.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Literal

from ragtrial.chat.rewriter import rewrite_query
from ragtrial.llm import llm as default_llm
from ragtrial.rag.naive_combined import ask_main


Role = Literal["user", "assistant"]


@dataclass
class Turn:
    role: Role
    content: str
    timestamp: float = field(default_factory=time.time)


class ChatSession:
    """Stateful conversation wrapper around naive-combined RAG.

    Parameters
    ----------
    max_history_turns
        Maximum number of round-trip turns (user+assistant pair = 1 turn) to keep.
        Older turns are dropped FIFO. Default 5.
    rewrite_followups
        If True, follow-up questions get rewritten into standalone queries via LLM
        before retrieval. Disable for cheaper single-LLM-call per turn.
    k_per_source
        Forwarded to ask_main(): docs retrieved per searchable capability.
    """

    def __init__(
        self,
        max_history_turns: int = 5,
        rewrite_followups: bool = True,
        k_per_source: int = 4,
    ):
        self.max_history_turns = max_history_turns
        self.rewrite_followups = rewrite_followups
        self.k_per_source = k_per_source
        self.history: List[Turn] = []

    def ask(self, user_message: str, verbose: bool = False) -> Dict[str, Any]:
        """Process one user turn, return result dict.

        Returns
        -------
        dict with keys:
            original_query    str   — user message as typed
            rewritten_query   str   — standalone query sent to retriever
            answer            str
            documents         list[Document]
            source_used       str   — "combined" for naive
            timings           dict  — {rewrite, retrieve, generate, total}
        """
        t_total0 = time.perf_counter()

        t_r0 = time.perf_counter()
        if self.rewrite_followups and self.history:
            rewritten = rewrite_query(self.history, user_message, default_llm)
        else:
            rewritten = user_message
        t_rewrite = time.perf_counter() - t_r0

        rag_result = ask_main(
            rewritten,
            k_per_source=self.k_per_source,
            verbose=verbose,
        )

        self.history.append(Turn(role="user", content=user_message))
        self.history.append(Turn(role="assistant", content=rag_result["answer"]))
        self._trim_history()

        timings = {
            "rewrite": t_rewrite,
            "retrieve": rag_result["timings"]["retrieve"],
            "generate": rag_result["timings"]["generate"],
            "total": time.perf_counter() - t_total0,
        }

        return {
            "original_query": user_message,
            "rewritten_query": rewritten,
            "answer": rag_result["answer"],
            "documents": rag_result["documents"],
            "source_used": rag_result["source_used"],
            "timings": timings,
        }

    def reset(self) -> None:
        """Clear conversation history."""
        self.history = []

    def get_history(self) -> List[Turn]:
        """Return a copy of current history."""
        return list(self.history)

    def _trim_history(self) -> None:
        max_entries = self.max_history_turns * 2
        if len(self.history) > max_entries:
            self.history = self.history[-max_entries:]
