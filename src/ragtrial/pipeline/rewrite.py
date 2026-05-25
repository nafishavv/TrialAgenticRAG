"""Query-rewrite stages. Enhanced rewrites EVERY query (no LLM 'should I?' choice).

Implemented:  passthrough (no-op), hyde.
Stubbed (TODO): multiquery — slot exists + registered; fill in `run()`.

Add one: implement Stage subclass, add to REWRITERS.
"""

from __future__ import annotations

from ragtrial.llm import llm as default_llm
from ragtrial.pipeline.base import RagState, Stage
from ragtrial.rag.prompts import PROMPT_HYDE


class PassthroughRewriter(Stage):
    """No rewrite — effective query stays the original question."""

    name = "rewrite"

    def run(self, state: RagState) -> RagState:
        state.query = state.question
        return state


class HyDERewriter(Stage):
    """HyDE: rewrite query into a hypothetical answer passage before retrieval.

    Prompts the LLM for a document-shaped hypothetical passage and uses THAT as
    the retrieval query — its embedding matches formal-document chunks better than
    a short question. The original question is stashed in meta['original_query']
    so downstream stages (rerank) can still score against the real intent.
    """

    name = "rewrite"

    def __init__(self, llm=None):
        self.llm = llm or default_llm

    def run(self, state: RagState) -> RagState:
        if state.intent == "invalid":
            # No retrieval will happen — skip the HyDE LLM call entirely.
            return state
        passage = self.llm.invoke(PROMPT_HYDE.format(question=state.question)).content
        state.meta["original_query"] = state.question
        state.meta["hyde_passage"] = passage
        state.query = passage
        return state


class MultiQueryRewriter(Stage):
    """Expand query into N paraphrases, retrieve for each, union results.

    TODO(enhanced): generate paraphrases; downstream retrieve must fan over them.
    """

    name = "rewrite"

    def run(self, state: RagState) -> RagState:
        raise NotImplementedError("MultiQueryRewriter belum diimplementasi (Tahap lanjutan).")


REWRITERS: dict[str, type[Stage]] = {
    "passthrough": PassthroughRewriter,
    "hyde": HyDERewriter,
    "multiquery": MultiQueryRewriter,
}
