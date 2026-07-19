"""ChatSession — conversation state + orchestration over ANY of the 3 RAG modes.

Pure Python class, zero UI coupling. Wraps a stateless mode entry point
(ask_naive / ask_enhanced / ask_agentic — all return RagResult) with:
  - chat history (in-memory; swap-able for Redis/DB later without API changes)
  - optional query rewriting for follow-up questions
  - history trimming to prevent unbounded growth
  - automatic trace persistence per turn (see ragtrial.tracing)

ALL interactive consumers (web server, CLI) go through this class, so every
query is traced and every consumer sees the same ChatTurnResult envelope —
the FULL RagResult travels inside it, nothing is dropped in translation.

Two rewriters exist BY DESIGN — do not merge them:
  - chat/rewriter.py (here): CONVERSATIONAL rewrite — resolves pronouns and
    references against recent history so a follow-up becomes standalone.
  - pipeline/rewrite.py: RETRIEVAL rewrite (e.g. HyDE) — an enhanced-pipeline
    stage that reformulates a standalone query for better retrieval.
"""

from __future__ import annotations

import time
import uuid
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Literal

from ragtrial.chat.rewriter import rewrite_query
from ragtrial.modes import DEFAULT_MODE, MODES, Mode, get_ask_fn
from ragtrial.result import RagResult

Role = Literal["user", "assistant"]

_DEFAULT_WRITER = object()
"""Sentinel: resolve the shared trace writer lazily (ragtrial.tracing). Pass
trace_writer=None to disable tracing entirely (tests, --no-trace)."""


@dataclass
class Turn:
    role: Role
    content: str
    timestamp: float = field(default_factory=time.time)


@dataclass
class ChatTurnResult:
    """Typed envelope for one processed turn. Carries the FULL RagResult."""

    session_id: str
    query_id: str
    original_query: str
    """User message as typed."""
    effective_query: str
    """After conversational follow-up rewrite (== original if none applied)."""
    rewrite_applied: bool
    result: RagResult
    timings: Dict[str, float]
    """Session-level timings: rewrite_followup, pipeline_total, total.
    Per-stage pipeline timings live in result.timings."""

    # Convenience passthroughs so callers don't have to dig into .result.
    @property
    def answer(self) -> str:
        return self.result.answer

    @property
    def documents(self) -> list:
        return self.result.documents

    @property
    def source_used(self) -> str:
        return self.result.source_used

    @property
    def mode(self) -> str:
        return self.result.mode


class ChatSession:
    """Stateful multi-turn conversation over a chosen RAG mode.

    Parameters
    ----------
    mode
        Which RAG pipeline to use: 'naive' | 'enhanced' | 'agentic'.
    max_history_turns
        Round-trip turns (user+assistant = 1) to keep; older dropped FIFO.
    rewrite_followups
        If True, follow-ups are rewritten into standalone queries before the
        pipeline runs (resolves pronouns/references via recent history).
    client
        Interface tag recorded in traces: 'cli' | 'web' | ...
    trace_writer
        Trace sink. Default = the shared writer from ragtrial.tracing (resolved
        lazily); None disables tracing for this session.
    """

    def __init__(
        self,
        mode: Mode = DEFAULT_MODE,
        max_history_turns: int = 5,
        rewrite_followups: bool = True,
        client: str = "cli",
        trace_writer: Any = _DEFAULT_WRITER,
    ):
        if mode not in MODES:
            raise ValueError(f"mode harus salah satu dari {list(MODES)}, dapat {mode!r}")
        self.mode = mode
        self._ask: Callable[..., RagResult] | None = None
        self.max_history_turns = max_history_turns
        self.rewrite_followups = rewrite_followups
        self.client = client
        self.session_id = uuid.uuid4().hex
        self._trace_writer = trace_writer
        self.history: List[Turn] = []

    def _get_ask(self) -> Callable[..., RagResult]:
        if self._ask is None:
            self._ask = get_ask_fn(self.mode)
        return self._ask

    def set_mode(self, mode: str) -> None:
        """Switch pipeline mid-conversation; history is kept."""
        if mode not in MODES:
            raise ValueError(f"mode harus salah satu dari {list(MODES)}, dapat {mode!r}")
        if mode != self.mode:
            self.mode = mode
            self._ask = None

    def ask(self, user_message: str, verbose: bool = False) -> ChatTurnResult:
        """Process one user turn. Traces are written even when the pipeline fails."""
        query_id = uuid.uuid4().hex
        t_total0 = time.perf_counter()

        t_r0 = time.perf_counter()
        if self.rewrite_followups and self.history:
            from ragtrial.llm import llm as _llm
            effective = rewrite_query(self.history, user_message, _llm)
        else:
            effective = user_message
        t_rewrite = time.perf_counter() - t_r0

        t_p0 = time.perf_counter()
        try:
            result: RagResult = self._get_ask()(effective, verbose=verbose)
        except Exception as e:
            failed = ChatTurnResult(
                session_id=self.session_id,
                query_id=query_id,
                original_query=user_message,
                effective_query=effective,
                rewrite_applied=effective != user_message,
                result=RagResult(question=user_message, query=effective, mode=self.mode),
                timings={
                    "rewrite_followup": t_rewrite,
                    "pipeline_total": time.perf_counter() - t_p0,
                    "total": time.perf_counter() - t_total0,
                },
            )
            self._trace(failed, error=f"{type(e).__name__}: {e}")
            raise
        t_pipeline = time.perf_counter() - t_p0

        self.history.append(Turn(role="user", content=user_message))
        self.history.append(Turn(role="assistant", content=result.answer))
        self._trim_history()

        turn = ChatTurnResult(
            session_id=self.session_id,
            query_id=query_id,
            original_query=user_message,
            effective_query=effective,
            rewrite_applied=effective != user_message,
            result=result,
            timings={
                "rewrite_followup": t_rewrite,
                "pipeline_total": t_pipeline,
                "total": time.perf_counter() - t_total0,
            },
        )
        self._trace(turn)
        return turn

    def _trace(self, turn: ChatTurnResult, error: str | None = None) -> None:
        """Persist the turn. MUST never raise — a lost trace beats a lost answer."""
        try:
            writer = self._trace_writer
            if writer is _DEFAULT_WRITER:
                from ragtrial.tracing import default_writer
                writer = default_writer
            if writer is None:
                return
            from ragtrial.tracing import build_trace_records
            basic, full = build_trace_records(turn, client=self.client, error=error)
            writer.write(basic, full)
        except Exception:
            import logging
            logging.getLogger(__name__).warning("trace write failed", exc_info=True)

    def reset(self) -> None:
        self.history = []

    def get_history(self) -> List[Turn]:
        return list(self.history)

    def _trim_history(self) -> None:
        max_entries = self.max_history_turns * 2
        if len(self.history) > max_entries:
            self.history = self.history[-max_entries:]
