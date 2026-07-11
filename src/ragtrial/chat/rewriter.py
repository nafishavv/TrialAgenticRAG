"""Query rewriting — resolve pronouns/references using chat history.

Preprocessing step (deterministic, always-run when history exists). Conceptually
still naive RAG: rewriter is a utility, not a decision-maker. The agentic/naive
distinction lives in the retrieval+generation layer, not here.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, List

from ragtrial.llm import invoke_with_retry
from ragtrial.rag.prompts import REWRITE_PROMPT

if TYPE_CHECKING:
    from ragtrial.chat.session import Turn


REWRITE_CONTEXT_TURNS: int = 3
"""How many recent turn-pairs to include in the rewrite prompt."""


def _format_history(history: List["Turn"], max_pairs: int) -> str:
    """Render recent history as a transcript for the rewrite prompt."""
    recent = history[-(max_pairs * 2):]
    lines = []
    for t in recent:
        prefix = "User" if t.role == "user" else "Assistant"
        lines.append(f"{prefix}: {t.content}")
    return "\n".join(lines)


def rewrite_query(
    history: List["Turn"],
    current_question: str,
    llm,
    max_context_pairs: int = REWRITE_CONTEXT_TURNS,
) -> str:
    """Rewrite a follow-up question into a standalone query using recent history.

    Returns current_question as-is if history is empty.
    """
    if not history:
        return current_question

    prompt = REWRITE_PROMPT.format(
        history=_format_history(history, max_context_pairs),
        question=current_question,
    )
    rewritten = invoke_with_retry(llm, prompt).content.strip()

    if rewritten.startswith('"') and rewritten.endswith('"'):
        rewritten = rewritten[1:-1].strip()

    return rewritten or current_question
