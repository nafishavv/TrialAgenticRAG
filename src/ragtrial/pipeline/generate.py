"""Generation stage — the single LLM call at the end of the enhanced pipeline.

Prompt is chosen from the route + retrieved docs (mirrors the old agentic
generate node, now de-hardcoded via the registry):
  - no docs / route 'none' -> PROMPT_NONE (polite out-of-scope)
  - single capability route -> PROMPT_SINGLE
  - fan-out / 'both'        -> PROMPT_COMBINED (with registry-derived citation rules)
"""

from __future__ import annotations

from typing import Dict, Optional

from ragtrial.capabilities.base import Capability, format_context
from ragtrial.capabilities.registry import CAPABILITIES, SEARCHABLE_CAPABILITIES
from ragtrial.llm import llm as default_llm
from ragtrial.pipeline.base import RagState, Stage
from ragtrial.rag.prompts import (
    PROMPT_COMBINED,
    PROMPT_INVALID,
    PROMPT_NONE,
    PROMPT_SINGLE,
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
        route = state.route

        if state.intent == "invalid":
            # Intent gate skipped retrieval — one smart prompt handles chit-chat
            # (friendly identity + categories) vs out-of-scope (polite refusal).
            prompt = PROMPT_INVALID.format(
                categories=service_categories_block(), question=q
            )
        elif not docs or route == "none":
            prompt = PROMPT_NONE.format(question=q)
        elif route in self.caps:
            prompt = PROMPT_SINGLE.format(
                source_description=self.caps[route].description,
                context=format_context(docs, CAPABILITIES),
                question=q,
            )
        else:  # fan-out / both
            prompt = PROMPT_COMBINED.format(
                sources_brief=build_sources_brief(self.caps),
                citation_rules=build_citation_rules(self.caps),
                context=format_context(docs, CAPABILITIES),
                question=q,
            )

        state.answer = self.llm.invoke(prompt).content
        return state
