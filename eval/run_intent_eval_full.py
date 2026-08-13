"""Runner: evaluasi intent gate (VALID/INVALID) atas SELURUH testset.json (198 soal).

Beda dari run_intent_eval.py (yang pakai eval/intent_testset.json — 40 soal
dedicated, balanced 50/50, hanya domain kependudukan+opd+chitchat/oos):
ini reuse testset.json (198 soal, retrieval eval) dan menurunkan expected_intent
dari expected_route ("none" -> invalid, selainnya -> valid). Trade-off: coverage
domain penuh (sosial/perizinan ikut teruji) tapi imbalanced (189 valid / 9
invalid) dan tanpa contoh chitchat (testset.json cuma punya out_of_scope).

Cara pakai (dari root project):
    python -m eval.run_intent_eval_full
    python -m eval.run_intent_eval_full --systems enhanced agentic --sleep 2

Output:
    eval/results/intent_full_<system>.json
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any, Dict, List

ROOT = Path(__file__).resolve().parent.parent

from eval.eval_core import intent_eval


def load_testset(path: Path) -> List[Dict[str, Any]]:
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)["questions"]
    out = []
    for q in data:
        expected_intent = "invalid" if q["expected_route"] == "none" else "valid"
        out.append({**q, "expected_intent": expected_intent})
    return out


def run_one(q: Dict[str, Any], ask_fn) -> Dict[str, Any]:
    t0 = time.perf_counter()
    res = ask_fn(q["question"], verbose=False)
    wall = time.perf_counter() - t0

    meta = getattr(res, "meta", {}) or {}
    predicted = meta.get("intent", "valid")
    timings = getattr(res, "timings", {}) or {}

    return {
        "id": q["id"],
        "question": q["question"],
        "expected_intent": q["expected_intent"],
        "expected_route": q["expected_route"],
        "query_type": q.get("query_type"),
        "predicted_intent": predicted,
        "correct": predicted == q["expected_intent"],
        "retrieved_docs": len(getattr(res, "documents", []) or []),
        "latency_total": timings.get("total", wall),
        "latency_wall": wall,
    }


def summarize(records: List[Dict[str, Any]]) -> Dict[str, Any]:
    valid_recs = [r for r in records if "error" not in r]
    preds = [r["predicted_intent"] for r in valid_recs]
    gold = [r["expected_intent"] for r in valid_recs]
    metrics = intent_eval(preds, gold)

    lat = sorted(r["latency_total"] for r in valid_recs)
    mean_lat = sum(lat) / len(lat) if lat else 0.0
    p95_lat = lat[int(0.95 * (len(lat) - 1))] if lat else 0.0

    return {
        "accuracy": metrics["accuracy"],
        "macro_f1": metrics["macro_f1"],
        "recall_invalid": metrics["recall_invalid"],
        "recall_valid": metrics["recall_valid"],
        "per_class": metrics["per_class"],
        "confusion_matrix": metrics["confusion_matrix"],
        "latency_mean": mean_lat,
        "latency_p95": p95_lat,
        "n": metrics["n"],
    }


def print_summary(system: str, s: Dict[str, Any]) -> None:
    print(f"\n=== INTENT (full 198) — {system} (n={s['n']}) ===")
    print(f"  accuracy={s['accuracy']:.3f}  macro-F1={s['macro_f1']:.3f}  "
          f"recall(invalid)={s['recall_invalid']:.3f}  recall(valid)={s['recall_valid']:.3f}")
    for cls, m in s["per_class"].items():
        print(f"    {cls:8} P={m['precision']:.3f} R={m['recall']:.3f} "
              f"F1={m['f1']:.3f} (support={m['support']})")
    print(f"  latency: mean={s['latency_mean']:.2f}s  p95={s['latency_p95']:.2f}s")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--testset", default=str(ROOT / "eval" / "testset.json"))
    ap.add_argument("--outdir", default=str(ROOT / "eval" / "results"))
    ap.add_argument("--systems", nargs="+", default=["naive", "enhanced", "agentic"],
                    choices=["naive", "enhanced", "agentic"])
    ap.add_argument("--limit", type=int, default=None, help="run only first N (smoke)")
    ap.add_argument("--sleep", type=float, default=0.0,
                    help="detik jeda antar query — throttle utk hindari 429 quota embedding")
    args = ap.parse_args()

    out = Path(args.outdir)
    out.mkdir(parents=True, exist_ok=True)

    testset = load_testset(Path(args.testset))
    if args.limit:
        testset = testset[: args.limit]
    n_invalid = sum(1 for q in testset if q["expected_intent"] == "invalid")
    print(f"Loaded {len(testset)} queries ({n_invalid} invalid / {len(testset) - n_invalid} valid). "
          f"systems={args.systems}")

    ask_fns = {}
    if "naive" in args.systems:
        from ragtrial.rag.naive import ask_naive
        ask_fns["naive"] = ask_naive
    if "enhanced" in args.systems:
        from ragtrial.rag.enhanced import ask_enhanced
        ask_fns["enhanced"] = ask_enhanced
    if "agentic" in args.systems:
        from ragtrial.rag.agentic import ask_agentic
        ask_fns["agentic"] = ask_agentic

    for system in args.systems:
        ask = ask_fns[system]
        print(f"\n=== Running system: {system} ===")
        records: List[Dict[str, Any]] = []
        for i, q in enumerate(testset, 1):
            print(f"  [{i}/{len(testset)}] {q['id']}: {q['question'][:55]}...")
            try:
                records.append(run_one(q, ask))
            except Exception as e:
                print(f"    ERROR on {q['id']}: {e}")
                records.append({"id": q["id"], "question": q["question"], "error": str(e)})
            if args.sleep:
                time.sleep(args.sleep)

        summary = summarize(records)
        print_summary(system, summary)

        payload = {"system": system, "summary": summary, "records": records}
        outpath = out / f"intent_full_{system}.json"
        with open(outpath, "w", encoding="utf-8") as f:
            json.dump(payload, f, ensure_ascii=False, indent=2)
        print(f"  -> saved {outpath}")


if __name__ == "__main__":
    main()
