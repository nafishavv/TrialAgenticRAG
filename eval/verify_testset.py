"""Validate eval/testset.json — schema, gold-id existence, consistency, distribution.

Default run is pure-Python (no API). The gold-id existence check is the load-
bearing gate: every gold_chunks entry must resolve to a real chunk in its domain,
otherwise retrieval metrics are meaningless.

    python -m eval.verify_testset
    python -m eval.verify_testset --sample-retrieval 15 --k 5   # opt-in, spends embed quota

Exit code is non-zero on any HARD failure (schema, gold-id existence, consistency).
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path
from typing import Any, Dict, List

from eval.generation.chunks import valid_gold_ids
from eval.generation.dedup import jaccard
from eval.generation.schema import (
    CHUNK_SCOPES,
    DIFFICULTIES,
    DOMAINS,
    QUERY_TYPES,
    ROUTES,
)

ROOT = Path(__file__).resolve().parent.parent
DEFAULT = ROOT / "eval" / "testset.json"

_REQUIRED = ["id", "question", "expected_route", "difficulty", "query_type",
             "chunk_scope", "gold_chunks", "expected_answer", "expected_facts", "notes"]


def _load(path: Path) -> List[dict]:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)["questions"]


def _check_schema(qs: List[dict], errs: List[str], warns: List[str]) -> None:
    seen_ids = set()
    for q in qs:
        qid = q.get("id", "?")
        for f in _REQUIRED:
            if f not in q:
                errs.append(f"{qid}: missing field '{f}'")
        if qid in seen_ids:
            errs.append(f"{qid}: duplicate id")
        seen_ids.add(qid)
        if q.get("expected_route") not in ROUTES:
            errs.append(f"{qid}: bad expected_route {q.get('expected_route')!r}")
        if q.get("difficulty") not in DIFFICULTIES:
            errs.append(f"{qid}: bad difficulty {q.get('difficulty')!r}")
        if q.get("query_type") not in QUERY_TYPES:
            errs.append(f"{qid}: bad query_type {q.get('query_type')!r}")
        if q.get("chunk_scope") not in CHUNK_SCOPES:
            errs.append(f"{qid}: bad chunk_scope {q.get('chunk_scope')!r}")
        if "TODO_FILL" in json.dumps(q, ensure_ascii=False):
            warns.append(f"{qid}: contains TODO_FILL placeholder")


def _check_consistency(qs: List[dict], errs: List[str]) -> None:
    for q in qs:
        qid = q.get("id", "?")
        route = q.get("expected_route")
        gold = q.get("gold_chunks") or {}
        n_ids = sum(len(v) for v in gold.values())
        if route == "none":
            if n_ids or q.get("expected_facts") or q.get("expected_answer"):
                errs.append(f"{qid}: none-route must have empty gold/facts/answer")
            if q.get("chunk_scope") != "none":
                errs.append(f"{qid}: none-route must have chunk_scope=none")
        elif route == "both":
            non_empty = [d for d, v in gold.items() if v]
            if len(non_empty) < 2:
                errs.append(f"{qid}: both-route needs >=2 domains in gold (got {non_empty})")
        else:
            if q.get("chunk_scope") == "single" and n_ids != 1:
                errs.append(f"{qid}: single scope must have exactly 1 gold id (got {n_ids})")


def _check_gold_exists(qs: List[dict], errs: List[str]) -> None:
    cache: Dict[str, set] = {}
    for q in qs:
        qid = q.get("id", "?")
        for domain, ids in (q.get("gold_chunks") or {}).items():
            if not ids:
                continue
            if domain not in DOMAINS:
                errs.append(f"{qid}: unknown gold domain {domain!r}")
                continue
            if domain not in cache:
                cache[domain] = valid_gold_ids(domain)
            for gid in ids:
                if f"{domain}:{gid}" not in cache[domain]:
                    errs.append(f"{qid}: gold-id '{domain}:{gid}' not found in {domain} chunks")


def _check_dups(qs: List[dict], warns: List[str]) -> None:
    for i in range(len(qs)):
        for j in range(i + 1, len(qs)):
            if jaccard(qs[i].get("question", ""), qs[j].get("question", "")) >= 0.85:
                warns.append(f"near-dup: {qs[i].get('id')} ~ {qs[j].get('id')}")


def _distribution(qs: List[dict]) -> None:
    print("\n-- distribution --")
    for dim in ["expected_route", "query_type", "difficulty", "chunk_scope"]:
        c = Counter(q.get(dim) for q in qs)
        print(f"  {dim}: {dict(c)}")


def _sample_retrieval(qs: List[dict], n: int, k: int) -> None:
    """Opt-in retrieval sanity (spends embedding quota)."""
    import random

    from eval.eval_core import doc_to_gold_id, gold_chunks_to_set, hit_at_k
    from ragtrial.rag.enhanced import ask_enhanced

    pool = [q for q in qs if q.get("expected_route") != "none"]
    sample = random.Random(0).sample(pool, min(n, len(pool)))
    print(f"\n-- retrieval sanity (n={len(sample)}, k={k}) --")
    hits = 0
    for q in sample:
        try:
            res = ask_enhanced(q["question"], verbose=False)
            docs = getattr(res, "documents", []) or []
            rids = [doc_to_gold_id(d) for d in docs]
            rids = [r for r in rids if r]
            h = hit_at_k(rids, gold_chunks_to_set(q["gold_chunks"]), k=k)
            hits += 1 if h == 1.0 else 0
            print(f"  {q['id']}: hit@{k}={h:.0f}")
        except Exception as e:  # noqa: BLE001
            print(f"  {q['id']}: ERROR {e}")
    print(f"  hit-rate: {hits}/{len(sample)}")


def verify(path: Path, sample_retrieval: int = 0, k: int = 5) -> bool:
    qs = _load(path)
    errs: List[str] = []
    warns: List[str] = []

    _check_schema(qs, errs, warns)
    _check_consistency(qs, errs)
    _check_gold_exists(qs, errs)
    _check_dups(qs, warns)

    print(f"=== verify {path.name}: {len(qs)} questions ===")
    _distribution(qs)

    if warns:
        print(f"\n-- warnings ({len(warns)}) --")
        for w in warns[:40]:
            print(f"  [warn] {w}")
    if errs:
        print(f"\n-- ERRORS ({len(errs)}) --")
        for e in errs[:60]:
            print(f"  [ERR] {e}")
    else:
        print("\nAll hard checks passed.")

    report = {"n": len(qs), "errors": errs, "warnings": warns}
    (ROOT / "eval" / "results").mkdir(parents=True, exist_ok=True)
    with open(ROOT / "eval" / "results" / "verify_report.json", "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)

    if sample_retrieval and not errs:
        _sample_retrieval(qs, sample_retrieval, k)

    return not errs


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--testset", default=str(DEFAULT))
    ap.add_argument("--sample-retrieval", type=int, default=0)
    ap.add_argument("--k", type=int, default=5)
    args = ap.parse_args()
    ok = verify(Path(args.testset), sample_retrieval=args.sample_retrieval, k=args.k)
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
