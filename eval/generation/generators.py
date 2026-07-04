"""Question builders: single / multi / cross / none.

Every builder sets `expected_route`, `gold_chunks`, `query_type` and `chunk_scope`
deterministically from the chosen chunk(s). The LLM only supplies the
natural-language fields. This keeps ground-truth correct regardless of model
behavior.
"""

from __future__ import annotations

from typing import List, Optional

from langchain_core.documents import Document

from eval.generation.chunks import SampleUnit
from eval.generation.dedup import check_leakage
from eval.generation.llm_client import LLMClient
from eval.generation.prompts import (
    cross_prompt,
    multi_prompt,
    none_prompt,
    parse_record,
    single_prompt,
)
from eval.generation.schema import (
    DIFFICULTIES,
    QRecord,
    build_gold_chunks,
    empty_gold,
)

_MAX_LEAK_RETRY = 2


def _difficulty(raw: dict, fallback: str = "medium") -> str:
    d = str(raw.get("difficulty", "")).strip().lower()
    return d if d in DIFFICULTIES else fallback


def _facts(raw: dict) -> List[str]:
    facts = raw.get("expected_facts") or []
    if isinstance(facts, str):
        facts = [facts]
    return [str(f).strip() for f in facts if str(f).strip()]


def _natural(raw: dict) -> Optional[dict]:
    """Validate the LLM's natural-language payload; None if unusable."""
    q = str(raw.get("question", "")).strip()
    ans = str(raw.get("expected_answer", "")).strip()
    facts = _facts(raw)
    if not q or not ans or not facts:
        return None
    return {
        "question": q,
        "expected_answer": ans,
        "expected_facts": facts,
        "difficulty": _difficulty(raw),
        "notes": str(raw.get("notes", "")).strip(),
    }


def gen_single(qid: str, unit: SampleUnit, llm: LLMClient) -> Optional[QRecord]:
    query_type = unit.target_query_type
    chunk_text = unit.chunk.page_content or ""
    nat = None
    for attempt in range(_MAX_LEAK_RETRY + 1):
        raw = parse_record(llm.generate(single_prompt(unit.domain, unit.chunk, query_type)))
        nat = _natural(raw)
        if nat is None:
            continue
        verdict = check_leakage(nat["question"], chunk_text, query_type)
        if verdict == "ok":
            break
        if attempt == _MAX_LEAK_RETRY:
            query_type = "lexical_exact"  # downgrade rather than discard
    if nat is None:
        return None
    return QRecord(
        id=qid,
        question=nat["question"],
        expected_route=unit.domain,
        difficulty=nat["difficulty"],
        query_type=query_type,
        chunk_scope="single",
        gold_chunks=build_gold_chunks([(unit.domain, unit.gold_id)]),
        expected_answer=nat["expected_answer"],
        expected_facts=nat["expected_facts"],
        notes=nat["notes"] or f"single/{query_type} grounded on {unit.gold_id}",
    )


def gen_multi(qid: str, units: List[SampleUnit], llm: LLMClient) -> Optional[QRecord]:
    domain = units[0].domain
    docs = [u.chunk for u in units]
    raw = parse_record(llm.generate(multi_prompt(domain, docs)))
    nat = _natural(raw)
    if nat is None:
        return None
    return QRecord(
        id=qid,
        question=nat["question"],
        expected_route=domain,
        difficulty=nat["difficulty"],
        query_type="multi_chunk",
        chunk_scope="multi",
        gold_chunks=build_gold_chunks([(domain, u.gold_id) for u in units]),
        expected_answer=nat["expected_answer"],
        expected_facts=nat["expected_facts"],
        notes=nat["notes"] or f"multi_chunk over {[u.gold_id for u in units]}",
    )


def gen_cross(qid: str, unit_a: SampleUnit, unit_b: SampleUnit, llm: LLMClient) -> Optional[QRecord]:
    raw = parse_record(
        llm.generate(cross_prompt(unit_a.chunk, unit_a.domain, unit_b.chunk, unit_b.domain))
    )
    nat = _natural(raw)
    if nat is None:
        return None
    return QRecord(
        id=qid,
        question=nat["question"],
        expected_route="both",
        difficulty=nat["difficulty"],
        query_type="cross_store",
        chunk_scope="cross_store",
        gold_chunks=build_gold_chunks([(unit_a.domain, unit_a.gold_id), (unit_b.domain, unit_b.gold_id)]),
        expected_answer=nat["expected_answer"],
        expected_facts=nat["expected_facts"],
        notes=nat["notes"] or f"cross_store {unit_a.domain}+{unit_b.domain}",
    )


def gen_none(qid: str, seed_examples: List[str], llm: LLMClient) -> Optional[QRecord]:
    raw = parse_record(llm.generate(none_prompt(seed_examples)))
    q = str(raw.get("question", "")).strip()
    if not q:
        return None
    return QRecord(
        id=qid,
        question=q,
        expected_route="none",
        difficulty="easy",
        query_type="out_of_scope",
        chunk_scope="none",
        gold_chunks=empty_gold(),
        expected_answer="",
        expected_facts=[],
        notes=str(raw.get("notes", "")).strip() or "out-of-scope (synthetic)",
    )
