"""Generate eval/candidate_testset.json synthetically, grounded on real chunks.

Usage:
    python -m eval.gen_testset --domain all --dry-run        # plumbing, no API
    python -m eval.gen_testset --domain opd --limit 5        # small live run
    python -m eval.gen_testset --domain all --inter-call-delay 4
    python -m eval.gen_testset --domain sosial --resume      # continue after a crash

The LLM only writes natural-language fields; route/gold/query_type/chunk_scope are
assigned deterministically from the chosen chunks (see generators.py). After
writing, a pure-Python verification runs and the script refuses to leave an
invalid file.
"""

from __future__ import annotations

import argparse
import json
import os
import random
import sys
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Optional

from langchain_core.documents import Document

from eval.generation.chunks import (
    SampleUnit,
    gold_id_of,
    load_domain_chunks,
    stratified_sample,
)
from eval.generation.dedup import QuestionDeduper
from eval.generation.generators import gen_cross, gen_multi, gen_none, gen_single
from eval.generation.llm_client import DryRunLLM, GenLLM, LLMClient
from eval.generation.schema import ID_PREFIX, QRecord, to_dict

ROOT = Path(__file__).resolve().parent.parent
DEFAULT_OUT = ROOT / "eval" / "candidate_testset.json"

# Out-of-scope seed phrases (style only) — borrowed from the intent gate.
_NONE_SEEDS = [
    "berapa harga beras hari ini?", "siapa presiden Indonesia?",
    "resep nasi goreng spesial?", "bagaimana cara belajar coding?",
    "jadwal pertandingan bola malam ini?", "cuaca besok hujan tidak?",
    "rekomendasi film bagus dong", "halo", "siapa kamu?", "terima kasih ya",
]

PLAN = {
    "sosial": {"single": 60, "multi": 10},
    "dukcapil": {"single": 50, "multi": 5},
    "perizinan": {"single": 28, "multi": 2},
    "opd": {"single": 25, "multi": 0},
    "both": 10,
    "none": 10,
}

_META = {
    "version": "1.0",
    "description": "Gold test set RAG Kab Batang (4 domain) — synthetic, grounded ke chunk asli.",
    "gold_id_format": {
        "dukcapil": "page:<page_start>",
        "opd": "nomor:<nomor>",
        "perizinan": "id:<perizinan_id>",
        "sosial": "id:<id>#pasal:<n> | #penjelasan:<n> | #preamble[.part] | #penjelasan-preamble | #narr:<k>",
    },
    "fields": {
        "id": "string, unik",
        "question": "pertanyaan user (Bahasa Indonesia)",
        "expected_route": "sosial | dukcapil | opd | perizinan | both | none",
        "difficulty": "easy | medium | hard",
        "query_type": "lexical_exact | paraphrase | semantic | multi_chunk | cross_store | analytical | negation_edge | out_of_scope",
        "chunk_scope": "single | multi | cross_store | none",
        "gold_chunks": "dict per source, list gold ID TANPA prefix domain",
        "expected_answer": "canonical jawaban ringkas (kosong utk none)",
        "expected_facts": "list fakta atomik untuk LLM-judge fact_recall (kosong utk none)",
        "notes": "kenapa soal ini ada / apa yang diuji",
    },
}


def _id(route: str, n: int) -> str:
    return f"{ID_PREFIX[route]}{n:03d}"


# ---------------- multi / cross pairing ----------------
def _multi_key(domain: str, chunk: Document) -> Optional[str]:
    m = chunk.metadata or {}
    if domain == "sosial":
        if m.get("chunk_type") in ("pasal", "penjelasan_pasal"):
            return str(m.get("id"))
        return None
    if domain == "dukcapil":
        if m.get("chunk_type") == "qa":
            return str(m.get("subsection") or m.get("section"))
        return None
    if domain == "perizinan":
        return str(m.get("kategori"))
    return None


def multi_groups(domain: str, n_groups: int, seed: int, exclude: set) -> List[List[SampleUnit]]:
    """Yield up to n_groups lists of 2-3 chunks that share a coherent group key."""
    rng = random.Random(f"{seed}:multi:{domain}")
    groups: Dict[str, List[Document]] = defaultdict(list)
    for c in load_domain_chunks(domain):
        gid = gold_id_of(domain, c)
        if not gid or gid in exclude:
            continue
        key = _multi_key(domain, c)
        if key:
            groups[key].append(c)
    candidates = [g for g in groups.values() if len(g) >= 2]
    rng.shuffle(candidates)
    out: List[List[SampleUnit]] = []
    for g in candidates[:n_groups]:
        rng.shuffle(g)
        chosen = g[:rng.choice([2, 3]) if len(g) >= 3 else 2]
        units = [SampleUnit(domain, c, gold_id_of(domain, c), "multi_chunk", "") for c in chosen]  # type: ignore[arg-type]
        out.append(units)
    return out


_CROSS_COMBOS = [
    ("perizinan", "opd"), ("dukcapil", "opd"), ("sosial", "opd"),
    ("dukcapil", "perizinan"), ("perizinan", "sosial"),
]


def cross_pairs(n: int, seed: int, exclude: set) -> List[tuple]:
    rng = random.Random(f"{seed}:cross")
    pairs = []
    for i in range(n):
        da, db = _CROSS_COMBOS[i % len(_CROSS_COMBOS)]
        ua = _pick_one(da, rng, exclude)
        ub = _pick_one(db, rng, exclude)
        if ua and ub:
            pairs.append((ua, ub))
    return pairs


def _pick_one(domain: str, rng: random.Random, exclude: set) -> Optional[SampleUnit]:
    chunks = [c for c in load_domain_chunks(domain) if gold_id_of(domain, c) and gold_id_of(domain, c) not in exclude]
    if not chunks:
        return None
    c = rng.choice(chunks)
    return SampleUnit(domain, c, gold_id_of(domain, c), "cross_store", "")  # type: ignore[arg-type]


# ---------------- persistence ----------------
def _load_existing(path: Path) -> List[dict]:
    if not path.exists():
        return []
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f).get("questions", [])


def _has_todo(q: dict) -> bool:
    blob = json.dumps(q, ensure_ascii=False)
    return "TODO_FILL" in blob or "TODO_" in blob


def _flush(path: Path, records: List[dict]) -> None:
    tmp = path.with_suffix(".json.tmp")
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump({"_meta": _META, "questions": records}, f, ensure_ascii=False, indent=2)
    os.replace(tmp, path)


# ---------------- main orchestration ----------------
def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--domain", default="all",
                    choices=["sosial", "dukcapil", "opd", "perizinan", "both", "none", "all"])
    ap.add_argument("--limit", type=int, default=None, help="cap single-chunk questions per domain")
    ap.add_argument("--dry-run", action="store_true", help="templated output, no API")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--resume", action="store_true", help="keep existing rows, skip routes already at target")
    ap.add_argument("--fresh", action="store_true", help="ignore all existing rows")
    ap.add_argument("--inter-call-delay", type=float, default=4.0)
    ap.add_argument("--temperature", type=float, default=0.5)
    ap.add_argument("--out", default=str(DEFAULT_OUT))
    args = ap.parse_args()

    out = Path(args.out)
    llm: LLMClient = DryRunLLM() if args.dry_run else GenLLM(
        temperature=args.temperature, inter_call_delay=args.inter_call_delay)

    # --- existing rows ---
    existing = [] if args.fresh else _load_existing(out)
    deduper = QuestionDeduper()
    records: List[dict] = []
    preserved_routes: Dict[str, int] = defaultdict(int)
    if args.resume or (not args.fresh and existing):
        for q in existing:
            if _has_todo(q):
                continue  # drop placeholder rows; they get regenerated
            records.append(q)
            deduper.add_existing(q.get("question", ""))
            preserved_routes[q.get("expected_route", "?")] += 1
        if records:
            print(f"Preserved {len(records)} existing rows (non-TODO).")

    used_gids: set = set()
    for q in records:
        for dom, ids in (q.get("gold_chunks") or {}).items():
            for gid in ids:
                used_gids.add(f"{dom}:{gid}")

    counters: Dict[str, int] = defaultdict(int)
    domains = ["sosial", "dukcapil", "opd", "perizinan"] if args.domain in ("all",) else \
        ([args.domain] if args.domain in ("sosial", "dukcapil", "opd", "perizinan") else [])
    do_both = args.domain in ("all", "both")
    do_none = args.domain in ("all", "none")

    def emit(rec: Optional[QRecord]) -> None:
        if rec is None:
            return
        if deduper.is_duplicate(rec.question):
            print(f"  [skip dup] {rec.question[:60]}")
            return
        deduper.accept(rec.question)
        records.append(to_dict(rec))
        for dom, ids in rec.gold_chunks.items():
            for gid in ids:
                used_gids.add(f"{dom}:{gid}")
        _flush(out, records)

    try:
        # ---- single + multi per domain ----
        for domain in domains:
            if args.resume and preserved_routes.get(domain, 0) >= PLAN[domain]["single"]:
                print(f"[resume] {domain} already at target, skip.")
                continue
            n_single = PLAN[domain]["single"] if args.limit is None else min(args.limit, PLAN[domain]["single"])
            print(f"\n=== {domain}: {n_single} single ===")
            units = stratified_sample(domain, n_single, seed=args.seed)
            for u in units:
                counters[domain] += 1
                emit(gen_single(_id(domain, counters[domain]), u, llm))
                used_gids.add(u.gold_id)

            n_multi = PLAN[domain]["multi"] if args.limit is None else 0
            if n_multi:
                print(f"=== {domain}: {n_multi} multi ===")
                for grp in multi_groups(domain, n_multi, args.seed, used_gids):
                    counters[domain] += 1
                    emit(gen_multi(_id(domain, counters[domain]), grp, llm))
                    for u in grp:
                        used_gids.add(u.gold_id)

        # ---- cross (both) ----
        if do_both:
            n = PLAN["both"] if args.limit is None else min(args.limit, PLAN["both"])
            print(f"\n=== both: {n} cross ===")
            for ua, ub in cross_pairs(n, args.seed, used_gids):
                counters["both"] += 1
                emit(gen_cross(_id("both", counters["both"]), ua, ub, llm))

        # ---- none ----
        if do_none:
            n = PLAN["none"] if args.limit is None else min(args.limit, PLAN["none"])
            print(f"\n=== none: {n} out-of-scope ===")
            rng = random.Random(f"{args.seed}:none")
            for i in range(n):
                counters["none"] += 1
                seeds = rng.sample(_NONE_SEEDS, k=3)
                emit(gen_none(_id("none", counters["none"]), seeds, llm))

    except KeyboardInterrupt:
        print("\n[interrupted] flushing progress...", file=sys.stderr)
        _flush(out, records)
        raise

    _flush(out, records)
    print(f"\nWrote {len(records)} questions -> {out}")

    # ---- verify (pure-Python, no API) ----
    from eval.verify_testset import verify
    ok = verify(out, sample_retrieval=0)
    if not ok:
        print("\nVERIFY FAILED — see report above.", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
