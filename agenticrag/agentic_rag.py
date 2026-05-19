"""DEPRECATED — this module has moved to `ragtrial.rag.agentic`.

Kept as a thin shim so existing notebooks keep working. New code should import
directly from the package:

    from ragtrial.rag.agentic import ask_agentic, agentic_app, route_query
    from ragtrial.capabilities import CAPABILITIES, format_context
    from ragtrial.rag.prompts import PROMPT_COMBINED, PROMPT_SINGLE, PROMPT_NONE
"""

from ragtrial.capabilities import CAPABILITIES, format_context
from ragtrial.llm import embeddings, llm
from ragtrial.rag.agentic import (
    AgentState,
    agentic_app,
    ask_agentic,
    route_query,
)
from ragtrial.rag.prompts import (
    PROMPT_COMBINED,
    PROMPT_NONE,
    PROMPT_SINGLE,
)

# Legacy prompt aliases (the old module had per-source prompts; the new one is generic).
PROMPT_DUKCAPIL = PROMPT_SINGLE
PROMPT_OPD = PROMPT_SINGLE
PROMPT_BOTH = PROMPT_COMBINED

# Back-compat vector store handles
CAPABILITIES["dukcapil"]._ensure_initialized()  # type: ignore[attr-defined]
CAPABILITIES["opd"]._ensure_initialized()  # type: ignore[attr-defined]
vs_dukcapil = CAPABILITIES["dukcapil"]._vectorstore  # type: ignore[attr-defined]
vs_opd = CAPABILITIES["opd"]._vectorstore  # type: ignore[attr-defined]


def retrieve_dukcapil(query, k=5):
    return CAPABILITIES["dukcapil"].invoke(query, k=k)


def retrieve_opd(query, k=5):
    return CAPABILITIES["opd"].invoke(query, k=k)


__all__ = [
    "ask_agentic",
    "agentic_app",
    "route_query",
    "AgentState",
    "format_context",
    "retrieve_dukcapil",
    "retrieve_opd",
    "PROMPT_DUKCAPIL",
    "PROMPT_OPD",
    "PROMPT_BOTH",
    "PROMPT_NONE",
    "PROMPT_COMBINED",
    "PROMPT_SINGLE",
    "llm",
    "embeddings",
    "vs_dukcapil",
    "vs_opd",
]
