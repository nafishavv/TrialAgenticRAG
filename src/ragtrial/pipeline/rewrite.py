"""Query-rewrite stages. Enhanced rewrites EVERY query (no LLM 'should I?' choice).

Implemented:  passthrough (no-op).
Stubbed (TODO): hyde, multiquery — slot exists + registered; fill in `run()`.

Add one: implement Stage subclass, add to REWRITERS.
"""

from __future__ import annotations

from ragtrial.pipeline.base import RagState, Stage


class PassthroughRewriter(Stage):
    """No rewrite — effective query stays the original question."""

    name = "rewrite"

    def run(self, state: RagState) -> RagState:
        state.query = state.question
        return state


class HyDERewriter(Stage):
    """HyDE: rewrite query into a hypothetical answer passage before retrieval.

    TODO(enhanced): prompt the LLM for a hypothetical passage, set state.query to
    it (and stash the original in state.meta['original_query']). Embedding that
    passage matches document-shaped chunks better than a short question.
    """

    name = "rewrite"

    def run(self, state: RagState) -> RagState:
        raise NotImplementedError("HyDERewriter belum diimplementasi (Tahap lanjutan).")


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
