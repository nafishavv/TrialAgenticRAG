"""De-leakage + near-duplicate question rejection.

De-leakage stops the LLM from smuggling a long verbatim span of the source chunk
into a question that is supposed to be a paraphrase/semantic/analytical probe —
otherwise the "hard" types collapse into lexical matches and the benchmark loses
discriminative power.
"""

from __future__ import annotations

import re
from typing import List

_WORD = re.compile(r"\w+", re.UNICODE)

# Types where verbatim overlap with the chunk is a problem (lexical_exact is exempt).
_DELEAK_TYPES = {"paraphrase", "semantic", "analytical"}

MAX_VERBATIM_RUN = 6      # longest shared consecutive token run allowed
MAX_JACCARD = 0.60        # token-set overlap allowed
DUP_JACCARD = 0.85        # question-vs-question near-duplicate threshold


def _tokens(text: str) -> List[str]:
    return _WORD.findall((text or "").lower())


def max_verbatim_run(question: str, chunk_text: str) -> int:
    """Longest run of consecutive tokens shared between question and chunk."""
    q = _tokens(question)
    c = _tokens(chunk_text)
    if not q or not c:
        return 0
    c_set_positions = {}
    for i, t in enumerate(c):
        c_set_positions.setdefault(t, []).append(i)
    best = 0
    for qi in range(len(q)):
        for ci in c_set_positions.get(q[qi], []):
            run = 0
            while (qi + run) < len(q) and (ci + run) < len(c) and q[qi + run] == c[ci + run]:
                run += 1
            best = max(best, run)
    return best


def jaccard(a: str, b: str) -> float:
    sa, sb = set(_tokens(a)), set(_tokens(b))
    if not sa or not sb:
        return 0.0
    return len(sa & sb) / len(sa | sb)


def check_leakage(question: str, chunk_text: str, query_type: str) -> str:
    """Return 'ok' | 'regenerate' | 'downgrade' for a generated question.

    'downgrade' means: keep the question but relabel it lexical_exact (it leaked
    too much to count as a paraphrase/semantic probe).
    """
    if query_type not in _DELEAK_TYPES:
        return "ok"
    if max_verbatim_run(question, chunk_text) >= MAX_VERBATIM_RUN or jaccard(question, chunk_text) >= MAX_JACCARD:
        return "regenerate"
    return "ok"


class QuestionDeduper:
    """Rejects questions that are near-duplicates of already-accepted ones."""

    def __init__(self) -> None:
        self._seen: List[set] = []

    def add_existing(self, question: str) -> None:
        self._seen.append(set(_tokens(question)))

    def is_duplicate(self, question: str) -> bool:
        toks = set(_tokens(question))
        if not toks:
            return False
        for prev in self._seen:
            inter = len(toks & prev)
            union = len(toks | prev)
            if union and inter / union >= DUP_JACCARD:
                return True
        return False

    def accept(self, question: str) -> None:
        self._seen.append(set(_tokens(question)))
