"""Single source of truth for RAG mode dispatch.

Every interactive consumer (ChatSession, CLI, server) resolves mode names HERE —
one registry, lazy imports (the pipelines pull in torch/chroma/langgraph, so
nothing heavy loads until a mode is actually used).

Eval keeps its own registry in eval/run_eval.py because it needs the eval-only
`enhanced@<preset>` addressing; it still calls the same ask_* entry points.

The returned callables have a NORMALIZED signature — (question, *, verbose=False)
-> RagResult — so callers never see the per-mode extras (naive's `k`, enhanced's
`config`); those stay internal defaults of the underlying functions.
"""

from __future__ import annotations

from typing import Callable, Literal

from ragtrial.result import RagResult

Mode = Literal["naive", "enhanced", "agentic"]
MODES: tuple[str, ...] = ("naive", "enhanced", "agentic")
DEFAULT_MODE: str = "enhanced"

AskFn = Callable[..., RagResult]


def get_ask_fn(mode: str) -> AskFn:
    """Resolve a mode name to its ask function (lazy import; ValueError if unknown)."""
    if mode == "naive":
        from ragtrial.rag.naive import ask_naive

        return lambda question, verbose=False: ask_naive(question, verbose=verbose)
    if mode == "enhanced":
        from ragtrial.rag.enhanced import ask_enhanced

        return lambda question, verbose=False: ask_enhanced(question, verbose=verbose)
    if mode == "agentic":
        from ragtrial.rag.agentic import ask_agentic

        return lambda question, verbose=False: ask_agentic(question, verbose=verbose)
    raise ValueError(f"mode harus salah satu dari {list(MODES)}, dapat {mode!r}")
