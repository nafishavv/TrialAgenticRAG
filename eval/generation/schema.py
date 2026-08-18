"""Test-set schema: enums, the per-question record, and gold-id handling.

The on-disk shape matches what `eval/run_eval.py` consumes. The single subtle
contract: `gold_chunks` values OMIT the domain prefix, because
`eval/eval_core.py:gold_chunks_to_set` rebuilds `f"{source}:{gid}"`. So a chunk
whose capability returns `"sosial:id:foo#pasal:2"` is stored as
`gold_chunks["sosial"] = ["id:foo#pasal:2"]`.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Tuple

# ---- enums (kept identical to the existing eval/candidate_testset.json _meta) ----
DOMAINS: List[str] = ["sosial", "dukcapil", "opd", "perizinan"]  # gold_chunks keys
ROUTES = {"sosial", "dukcapil", "opd", "perizinan", "both", "none"}
DIFFICULTIES = {"easy", "medium", "hard"}
QUERY_TYPES = {
    "lexical_exact", "paraphrase", "semantic", "multi_chunk",
    "cross_store", "analytical", "negation_edge", "out_of_scope",
}
CHUNK_SCOPES = {"single", "multi", "cross_store", "none"}
ID_PREFIX = {
    "sosial": "SO", "dukcapil": "DK", "opd": "OP",
    "perizinan": "PZ", "both": "CR", "none": "NO",
}


@dataclass
class QRecord:
    """One test question. Field order mirrors the existing candidate_testset.json."""

    id: str
    question: str
    expected_route: str
    difficulty: str
    query_type: str
    chunk_scope: str
    gold_chunks: Dict[str, List[str]]  # registry-keyed, STRIPPED ids
    expected_answer: str
    expected_facts: List[str]
    notes: str


def strip_gold_prefix(domain: str, prefixed: str) -> str:
    """`('sosial', 'sosial:id:foo#pasal:2') -> 'id:foo#pasal:2'`."""
    pfx = f"{domain}:"
    assert prefixed.startswith(pfx), f"gold-id {prefixed!r} missing prefix {pfx!r}"
    return prefixed[len(pfx):]


def empty_gold() -> Dict[str, List[str]]:
    return {d: [] for d in DOMAINS}


def build_gold_chunks(entries: List[Tuple[str, str]]) -> Dict[str, List[str]]:
    """Build a gold_chunks dict from `(domain, prefixed_gold_id)` pairs.

    Strips the domain prefix and dedups while preserving order.
    """
    gold = empty_gold()
    for domain, prefixed in entries:
        stripped = strip_gold_prefix(domain, prefixed)
        if stripped not in gold[domain]:
            gold[domain].append(stripped)
    return gold


def to_dict(rec: QRecord) -> dict:
    """Serialize in the exact key order run_eval expects (all 4 gold keys present)."""
    return {
        "id": rec.id,
        "question": rec.question,
        "expected_route": rec.expected_route,
        "difficulty": rec.difficulty,
        "query_type": rec.query_type,
        "chunk_scope": rec.chunk_scope,
        "gold_chunks": rec.gold_chunks,
        "expected_answer": rec.expected_answer,
        "expected_facts": rec.expected_facts,
        "notes": rec.notes,
    }
