"""Runner: jalankan agentic & naive-combined RAG pada testset, simpan hasil per-query.

Cara pakai (dari root project):
    python -m eval.run_eval --systems agentic naive --k 5
    python -m eval.run_eval --systems agentic --k 5 --limit 5   # cepet, test 5 soal
    python -m eval.run_eval --no-judge                          # skip LLM judge (cepet, retrieval-only)
    python -m eval.run_eval --systems agentic --resume          # lanjutkan run yg terputus

Checkpointing: hasil ditulis ulang (atomik) tiap SATU soal selesai, jadi run yg
mati di tengah (Ctrl-C / crash kuota / laptop sleep) tetap simpan progresnya.
Pakai --resume untuk lanjut: soal yg sudah sukses dilewati, yg error/belum diulang.

Output:
    eval/results/per_query_<system>.json   — full per-query record
    eval/results/timing_<system>.json      — flat timing snapshot
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any, Dict, List

ROOT = Path(__file__).resolve().parent.parent

from eval.eval_core import (
    doc_to_gold_id,
    gold_chunks_to_set,
    hit_at_k,
    recall_at_k,
    precision_at_k,
    mrr,
    judge_faithfulness,
    judge_refusal,
    judge_answer_relevance,
    store_correct,
)


def load_testset(path: Path) -> List[Dict[str, Any]]:
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    return data["questions"]


def _load_existing(outpath: Path) -> List[Dict[str, Any]]:
    """Read prior per-query records (for --resume). Tolerant of a missing/corrupt file."""
    if not outpath.exists():
        return []
    try:
        with open(outpath, "r", encoding="utf-8") as f:
            return json.load(f)
    except (json.JSONDecodeError, OSError):
        return []


def _save_atomic(outpath: Path, records: List[Dict[str, Any]]) -> None:
    """Write via temp+rename so a kill mid-write never corrupts the results file.

    Called after EVERY question, so an interrupted run (Ctrl-C, quota crash, sleep)
    keeps all progress so far — re-run with --resume to finish only what's left.
    """
    tmp = outpath.with_suffix(outpath.suffix + ".tmp")
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(records, f, ensure_ascii=False, indent=2)
    tmp.replace(outpath)


def run_one_query(
    q: Dict[str, Any],
    system: str,
    k: int,
    use_judge: bool,
    ask_fn,
) -> Dict[str, Any]:
    """Run one question through one system, compute all metrics."""
    question = q["question"]

    # 1) Execute pipeline (all modes return RagResult -> normalize to dict)
    t0 = time.perf_counter()
    res = ask_fn(question, verbose=False)
    wall = time.perf_counter() - t0
    result = res.to_dict() if hasattr(res, "to_dict") else res

    answer = result.get("answer", "")
    docs = result.get("documents", []) or []
    retrieved_ids = [doc_to_gold_id(d) for d in docs]
    retrieved_ids = [r for r in retrieved_ids if r is not None]

    gold_set = gold_chunks_to_set(q["gold_chunks"])

    # 2) Routing — AGENTIC ONLY. Enhanced retrieves globally (router='none'), so a
    #    routing metric is moot there; naive has no route. Only agentic makes a
    #    domain decision worth scoring.
    routing = None
    if system == "agentic":
        routing = {
            "predicted_route": result.get("route", ""),
            "expected_route": q["expected_route"],
            "correct": result.get("route") == q["expected_route"],
            "route_reason": result.get("route_reason", ""),
            "source_used": result.get("source_used", ""),
            "store_correct": store_correct(q["expected_route"], result.get("source_used", "")),
        }

    # 3) Retrieval metrics (skip for none-route queries since no gold chunks)
    if q["expected_route"] == "none":
        retrieval = {"skipped_reason": "out_of_scope_no_gold"}
    else:
        retrieval = {
            f"hit@{k}": hit_at_k(retrieved_ids, gold_set, k=k),
            f"recall@{k}": recall_at_k(retrieved_ids, gold_set, k=k),
            f"precision@{k}": precision_at_k(retrieved_ids, gold_set, k=k),
            "mrr": mrr(retrieved_ids, gold_set, k=10),
        }

    # 4) Persist retrieval context (page_content of retrieved docs) so the judge
    #    phase (run_judge.py) can score faithfulness OFFLINE without re-running the
    #    pipeline. Stored as a list of strings; joined at judge time.
    retrieved_context = [d.page_content for d in docs]

    # 5) Answer quality (LLM judges) — OPTIONAL inline convenience path.
    #    The recommended flow is the GENERATE phase (--no-judge) + a separate,
    #    cacheable judge phase (eval.run_judge). fact_recall is DROPPED (circular:
    #    LLM-generated key judged by an LLM). Only faithfulness/answer_relevance/
    #    refusal remain — the same three run_judge computes. Correctness -> experts.
    answer_eval: Dict[str, Any] = {}
    if use_judge:
        ref = judge_refusal(answer)
        answer_eval["refused"] = ref["refused"]
        # For 'none' queries, refusal=YES is the correct behavior.
        if q["expected_route"] == "none":
            answer_eval["refusal_correct"] = ref["refused"]
        else:
            # Faithfulness vs retrieved context + answer relevance (non-none only).
            ctx = "\n\n---\n\n".join(retrieved_context)
            answer_eval["faithfulness"] = judge_faithfulness(answer, ctx)["faithfulness"]
            answer_eval["answer_relevance"] = judge_answer_relevance(question, answer)["answer_relevance"]

    # 6) Timing
    timings = result.get("timings", {}) or {}

    return {
        "id": q["id"],
        "question": question,
        "expected_route": q["expected_route"],
        "difficulty": q["difficulty"],
        "query_type": q["query_type"],
        "chunk_scope": q["chunk_scope"],
        "system": system,
        "answer": answer,
        "retrieved_ids": retrieved_ids,
        "retrieved_context": retrieved_context,
        "gold_ids": sorted(gold_set),
        "routing": routing,
        "retrieval": retrieval,
        "answer_eval": answer_eval,
        "timings": {**timings, "wall": wall},
        # Normalized execution log + agent trace (observability for self-correction).
        "decisions": result.get("decisions", {}),
        "steps": (result.get("meta") or {}).get("steps", []),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--testset", default=str(ROOT / "eval" / "testset.json"))
    ap.add_argument("--outdir", default=str(ROOT / "eval" / "results"))
    ap.add_argument("--systems", nargs="+", default=["naive", "enhanced", "agentic"],
                    help="naive | agentic | enhanced | enhanced@<preset> "
                         "(presets: default, dense_only, hybrid_no_rerank, rerank_no_hybrid, no_intent)")
    ap.add_argument("--k", type=int, default=5, help="top-k for retrieval metrics")
    ap.add_argument("--limit", type=int, default=None,
                    help="run only first N questions (smoke test)")
    ap.add_argument("--no-judge", action="store_true",
                    help="skip LLM-as-judge (retrieval & routing only)")
    ap.add_argument("--ids", nargs="+", default=None,
                    help="run only specific question IDs (e.g. DK001 OP005)")
    ap.add_argument("--sleep", type=float, default=0.0,
                    help="detik jeda antar query — throttle utk hindari 429 quota embedding")
    ap.add_argument("--resume", action="store_true",
                    help="lanjutkan run yg terputus: skip soal yg SUDAH sukses di "
                         "per_query_<system>.json, ulang yg error/belum ada")
    args = ap.parse_args()

    out = Path(args.outdir)
    out.mkdir(parents=True, exist_ok=True)
    use_judge = not args.no_judge

    testset = load_testset(Path(args.testset))
    if args.ids:
        testset = [q for q in testset if q["id"] in set(args.ids)]
    if args.limit:
        testset = testset[: args.limit]

    print(f"Loaded {len(testset)} questions. systems={args.systems}  k={args.k}  judge={use_judge}")

    # System registry — add/swap a system here; metrics stay system-agnostic
    # because every system returns a RagResult. Enhanced ablation cells are
    # addressed as "enhanced@<preset>" (e.g. enhanced@dense_only); bare "enhanced"
    # == enhanced@default. The pipeline is built ONCE per system (reuses the
    # reranker model + intent router across queries).
    def _mk_enhanced(preset: str):
        from ragtrial.rag.enhanced import PRESETS, build_enhanced
        rag = build_enhanced(PRESETS[preset])
        return lambda q, verbose=False: rag.ask(q, verbose=verbose)

    ask_fns = {}
    for system in args.systems:
        if system == "naive":
            from ragtrial.rag.naive import ask_naive
            ask_fns[system] = ask_naive
        elif system == "agentic":
            from ragtrial.rag.agentic import ask_agentic
            ask_fns[system] = ask_agentic
        elif system == "enhanced" or system.startswith("enhanced@"):
            preset = system.split("@", 1)[1] if "@" in system else "default"
            ask_fns[system] = _mk_enhanced(preset)
        else:
            raise SystemExit(f"unknown system: {system!r}")

    for system in args.systems:
        ask = ask_fns[system]
        outpath = out / f"per_query_{system}.json"
        print(f"\n=== Running system: {system} ===")

        # Resume: keep prior SUCCESSFUL records, skip those ids; re-run errored/missing.
        by_id: Dict[str, Dict[str, Any]] = {}
        if args.resume:
            done_ok = {r["id"]: r for r in _load_existing(outpath) if "error" not in r}
            by_id.update(done_ok)
            if done_ok:
                print(f"  resume: {len(done_ok)} soal sudah sukses, dilewati.")

        def _flush() -> None:
            # Persist after every question, in testset order (extras appended).
            ordered = [by_id[q["id"]] for q in testset if q["id"] in by_id]
            seen = {q["id"] for q in testset}
            ordered += [r for i, r in by_id.items() if i not in seen]
            _save_atomic(outpath, ordered)

        for i, q in enumerate(testset, 1):
            if args.resume and q["id"] in by_id:
                print(f"  [{i}/{len(testset)}] {q['id']}: SKIP (sudah sukses)")
                continue
            print(f"  [{i}/{len(testset)}] {q['id']}: {q['question'][:60]}...")
            try:
                by_id[q["id"]] = run_one_query(q, system, args.k, use_judge, ask)
            except Exception as e:
                print(f"    ERROR on {q['id']}: {e}")
                by_id[q["id"]] = {
                    "id": q["id"], "system": system, "error": str(e),
                    "question": q["question"],
                }
            _flush()  # checkpoint: progress survives an interrupted run
            if args.sleep:
                time.sleep(args.sleep)

        n_err = sum(1 for r in by_id.values() if "error" in r)
        print(f"  -> saved {outpath}  ({len(by_id)} records, {n_err} error)")


if __name__ == "__main__":
    main()
