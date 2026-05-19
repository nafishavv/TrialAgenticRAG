"""DEPRECATED — this module has moved to `ragtrial.rag.naive_combined`.

Kept as a thin shim so existing notebooks keep working. New code should import
directly from the package:

    from ragtrial.rag.naive_combined import ask_main
    from ragtrial.capabilities import CAPABILITIES, SEARCHABLE_CAPABILITIES
"""

from ragtrial.capabilities import CAPABILITIES, SEARCHABLE_CAPABILITIES, format_context
from ragtrial.llm import embeddings, llm
from ragtrial.rag.naive_combined import ask_main, retrieve_combined
from ragtrial.rag.prompts import PROMPT_COMBINED

# Back-compat aliases — force lazy init so legacy callers see the Chroma handle
CAPABILITIES["dukcapil"]._ensure_initialized()  # type: ignore[attr-defined]
CAPABILITIES["opd"]._ensure_initialized()  # type: ignore[attr-defined]
vs_dukcapil = CAPABILITIES["dukcapil"]._vectorstore  # type: ignore[attr-defined]
vs_opd = CAPABILITIES["opd"]._vectorstore  # type: ignore[attr-defined]


def retrieve_dukcapil_hybrid(query, k=4):
    return CAPABILITIES["dukcapil"].invoke(query, k=k)


def retrieve_opd_hybrid(query, k=4):
    return CAPABILITIES["opd"].invoke(query, k=k)


def format_context_combined(docs):
    return format_context(docs, CAPABILITIES)


__all__ = [
    "ask_main",
    "retrieve_combined",
    "retrieve_dukcapil_hybrid",
    "retrieve_opd_hybrid",
    "format_context_combined",
    "PROMPT_COMBINED",
    "llm",
    "embeddings",
    "vs_dukcapil",
    "vs_opd",
]
