"""Generation stage — the single LLM call at the end of the enhanced pipeline.

Prompt is chosen from the intent decision + retrieved docs:
  - intent 'invalid' -> PROMPT_INVALID (chit-chat/identity vs out-of-scope)
  - no docs          -> PROMPT_NONE (polite "not found")
  - docs found       -> PROMPT_COMBINED (with registry-derived citation rules)

Retrieval is global, so the context is always a fan-out across domains — there
is no single-domain prompt variant.
"""

from __future__ import annotations

from typing import Dict, Optional

from ragtrial.capabilities.base import Capability, format_context
from ragtrial.capabilities.registry import CAPABILITIES, SEARCHABLE_CAPABILITIES
from ragtrial.llm import invoke_with_retry
from ragtrial.llm import llm as default_llm
from ragtrial.pipeline.base import RagState, Stage
from ragtrial.rag.prompts import (
    PROMPT_COMBINED,
    PROMPT_INVALID,
    PROMPT_NONE,
    build_citation_rules,
    build_sources_brief,
    service_categories_block,
)


class GenerateStage(Stage):
    name = "generate"

    def __init__(
        self,
        capabilities: Optional[Dict[str, Capability]] = None,
        llm=None,
    ):
        self.caps = capabilities or SEARCHABLE_CAPABILITIES
        self.llm = llm or default_llm

    def run(self, state: RagState) -> RagState:
        q = state.question
        docs = state.documents or []

        if state.intent == "invalid":
            # Intent gate skipped retrieval — one smart prompt handles chit-chat
            # (friendly identity + categories) vs out-of-scope (polite refusal).
            prompt = PROMPT_INVALID.format(
                categories=service_categories_block(), question=q
            )
        elif not docs:
            prompt = PROMPT_NONE.format(question=q)
        else:  # global fan-out across domains
            prompt = PROMPT_COMBINED.format(
                sources_brief=build_sources_brief(self.caps),
                citation_rules=build_citation_rules(self.caps),
                context=format_context(docs, CAPABILITIES),
                question=q,
            )

        state.answer = invoke_with_retry(self.llm, prompt).content
        return state
