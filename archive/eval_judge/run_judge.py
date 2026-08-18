"""ARSIP — JUDGE phase custom, sudah TIDAK DIPAKAI (pensiun 18 Agustus 2026).

Digantikan oleh eval/run_ragas.py (RAGAS Faithfulness + Semantic Similarity).
Lihat custom_judge.py di folder ini untuk alasan lengkapnya. Disimpan supaya
angka lama di docs/EVAL_REPORT.md §6 masih bisa ditelusuri/direproduksi.

CATATAN MENJALANKAN ULANG: implementasi judge-nya sekarang ada di
`archive/eval_judge/custom_judge.py` (dulu di eval/eval_core.py §4), dan skrip ini
mengimpornya dari sana. Jalankan dari root project dengan .venv utama, mis.:
    python archive/eval_judge/run_judge.py --results-dir <dir>

--- dokumentasi asli di bawah ini ---

The generate phase (`eval.run_eval --no-judge`) persists answer + retrieved_context
+ ids + decisions + timings per query. This script reads those per_query files and
computes the three reference-free judges — faithfulness, answer_relevance, refusal —
writing the scores back into each record's `answer_eval` block.

Why decouple: judging costs LLM calls, but generation is the expensive/rate-bound
step. Splitting them means the judge phase can be re-run cheaply, and a persistent
cache means UNCHANGED answers are never re-judged (zero LLM call on a second pass).

Cache design:
    key   = "<id>|<system>|<judge_name>|<sha1(answer)>"
    value = the judged scalar (refused: bool | faithfulness/answer_relevance: float)
    store = eval/results/judge_cache.json  (persisted across runs)
Re-running over the same answers is a pure cache hit. Change one answer -> only that
id's judges (whose answer hash changed) are recomputed. NOTE: the key hashes the
ANSWER only (per spec). In the decoupled flow the retrieved_context is produced
together with the answer, so an unchanged answer implies unchanged context; if you
ever mutate context without changing the answer, clear judge_cache.json.

Usage:
    python -m eval.run_judge --systems naive enhanced agentic
    python -m eval.run_judge --systems agentic --ids DK001 OP005   # judge a subset
    python -m eval.run_judge --systems naive --resume              # skip already-judged
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from pathlib import Path
from typing import Any, Callable, Dict, List, Tuple

ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(Path(__file__).resolve().parent))

# Dulu: `from eval.eval_core import ...`. Judge-nya sudah dipindah ke arsip.
from custom_judge import (  # noqa: E402
    judge_answer_relevance,
    judge_faithfulness,
    judge_refusal,
)

JUDGE_NAMES = ("refusal", "faithfulness", "answer_relevance")


# ---------- cache key ----------
def _sha1(s: str) -> str:
    return hashlib.sha1((s or "").encode("utf-8")).hexdigest()


def cache_key(rec_id: str, system: str, judge_name: str, answer: str) -> str:
    return f"{rec_id}|{system}|{judge_name}|{_sha1(answer)}"


# ---------- context helper ----------
def context_of(rec: Dict[str, Any]) -> str:
    """Reconstruct the judge context from persisted retrieved_context (P1.3)."""
    rc = rec.get("retrieved_context")
    if isinstance(rc, list):
        return "\n\n---\n\n".join(rc)
    return rc or ""


# ---------- default judges (injectable for tests) ----------
# Each maps a record -> the judged scalar. Kept as a dict so a caller/test can pass
# monkeypatched constants without hitting the API.
def _default_judges() -> Dict[str, Callable[[Dict[str, Any]], Any]]:
    return {
        "refusal": lambda rec: judge_refusal(rec.get("answer", ""))["refused"],
        "faithfulness": lambda rec: judge_faithfulness(
            rec.get("answer", ""), context_of(rec)
        )["faithfulness"],
        "answer_relevance": lambda rec: judge_answer_relevance(
            rec.get("question", ""), rec.get("answer", "")
        )["answer_relevance"],
    }


def judge_record(
    rec: Dict[str, Any],
    system: str,
    cache: Dict[str, Any],
    judges: Dict[str, Callable[[Dict[str, Any]], Any]],
) -> Tuple[Dict[str, Any], int]:
    """Score ONE record. Returns (answer_eval, n_llm_calls_made).

    Mirrors run_eval's inline judge policy (minus fact_recall):
      - refused: always
      - expected_route == 'none' -> refusal_correct = refused (no LLM beyond refusal)
      - else -> faithfulness (needs context) + answer_relevance
    A cache hit reuses the stored scalar and makes NO LLM call.
    """
    if "error" in rec:
        return rec.get("answer_eval", {}) or {}, 0

    rid = rec["id"]
    answer = rec.get("answer", "")
    ae = dict(rec.get("answer_eval") or {})
    n_calls = 0

    def _judged(judge_name: str) -> Any:
        nonlocal n_calls
        key = cache_key(rid, system, judge_name, answer)
        if key in cache:
            return cache[key]
        val = judges[judge_name](rec)
        cache[key] = val
        n_calls += 1
        return val

    ae["refused"] = _judged("refusal")
    if rec.get("expected_route") == "none":
        ae["refusal_correct"] = ae["refused"]
    else:
        ae["faithfulness"] = _judged("faithfulness")
        ae["answer_relevance"] = _judged("answer_relevance")
    return ae, n_calls


def seed_cache_from_records(
    records: List[Dict[str, Any]], system: str, cache: Dict[str, Any]
) -> int:
    """--resume: treat already-judged records as cache hits (seed by answer hash),
    so a rerun after judge_cache.json was lost does not re-pay the LLM. Returns the
    number of entries seeded."""
    seeded = 0
    for rec in records:
        if "error" in rec:
            continue
        ae = rec.get("answer_eval") or {}
        answer = rec.get("answer", "")
        rid = rec["id"]
        for judge_name, field in (
            ("refusal", "refused"),
            ("faithfulness", "faithfulness"),
            ("answer_relevance", "answer_relevance"),
        ):
            if field in ae:
                key = cache_key(rid, system, judge_name, answer)
                if key not in cache:
                    cache[key] = ae[field]
                    seeded += 1
    return seeded


def judge_records(
    records: List[Dict[str, Any]],
    system: str,
    cache: Dict[str, Any],
    judges: Dict[str, Callable[[Dict[str, Any]], Any]] | None = None,
    ids: set | None = None,
    sleep: float = 0.0,
    checkpoint: Callable[[], None] | None = None,
    checkpoint_every: int = 15,
) -> int:
    """Judge a list of records in place. Returns total LLM calls made.

    If `checkpoint` is given, it is invoked every `checkpoint_every` records that
    actually made an LLM call, so an interrupted judge phase (teardown/429) keeps
    all progress so far — a re-run with --resume then skips the already-judged.
    """
    judges = judges or _default_judges()
    total_calls = 0
    since_ckpt = 0
    for rec in records:
        if ids is not None and rec.get("id") not in ids:
            continue
        ae, n = judge_record(rec, system, cache, judges)
        rec["answer_eval"] = ae
        total_calls += n
        if n:
            since_ckpt += 1
            if checkpoint and since_ckpt >= checkpoint_every:
                checkpoint()
                since_ckpt = 0
            if sleep:
                time.sleep(sleep)
    if checkpoint:
        checkpoint()  # final flush of the tail
    return total_calls


# ---------- persistence ----------
def load_json(path: Path, default):
    if not path.exists():
        return default
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def save_json(path: Path, obj) -> None:
    """Atomic write (temp+rename) so a kill mid-checkpoint never corrupts the file."""
    tmp = path.with_suffix(path.suffix + ".tmp")
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)
    tmp.replace(path)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--systems", nargs="+", default=["naive", "enhanced", "agentic"])
    ap.add_argument("--results-dir", default=str(ROOT / "eval" / "results"))
    ap.add_argument("--ids", nargs="+", default=None,
                    help="judge only these question IDs (others left untouched)")
    ap.add_argument("--resume", action="store_true",
                    help="seed cache from existing answer_eval -> skip already-judged answers")
    ap.add_argument("--sleep", type=float, default=0.0,
                    help="detik jeda setelah setiap LLM judge call (throttle 429)")
    args = ap.parse_args()

    results = Path(args.results_dir)
    cache_path = results / "judge_cache.json"
    cache: Dict[str, Any] = load_json(cache_path, {})
    ids = set(args.ids) if args.ids else None

    for system in args.systems:
        path = results / f"per_query_{system}.json"
        records = load_json(path, None)
        if records is None:
            print(f"[skip] no per_query file for {system}: {path}")
            continue

        if args.resume:
            seeded = seed_cache_from_records(records, system, cache)
            print(f"  [resume] seeded {seeded} cache entries from {path.name}")

        before = len(cache)

        def _ckpt() -> None:
            save_json(path, records)
            save_json(cache_path, cache)

        calls = judge_records(records, system, cache, ids=ids, sleep=args.sleep,
                              checkpoint=_ckpt)
        _ckpt()
        print(f"=== {system}: {len(records)} records | LLM calls: {calls} | "
              f"cache {before}->{len(cache)} -> saved {path.name}")

    save_json(cache_path, cache)


if __name__ == "__main__":
    main()
