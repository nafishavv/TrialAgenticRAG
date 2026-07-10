"""Runner: jalankan agentic & naive-combined RAG pada testset, simpan hasil per-query.

Cara pakai (dari root project):
    python -m eval.run_eval --systems agentic naive --k 5
    python -m eval.run_eval --systems agentic --k 5 --limit 5   # cepet, test 5 soal
    python -m eval.run_eval --no-judge                          # skip LLM judge (cepet, retrieval-only)

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
    judge_fact_recall,
    judge_faithfulness,
    judge_refusal,
    judge_answer_relevance,
    store_correct,
)


def load_testset(path: Path) -> List[Dict[str, Any]]:
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    return data["questions"]


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

    # 2) Routing (modes that produce a route decision)
    routing = None
    if system in ("agentic", "enhanced"):
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

    # 4) Answer quality (LLM judges)
    answer_eval: Dict[str, Any] = {}
    if use_judge:
        # Refusal check
        ref = judge_refusal(answer)
        should_refuse = q["expected_route"] == "none" or all(
            f.startswith("TODO_") for f in q.get("expected_facts", [])
        ) and not gold_set
        # For 'none' queries, refusal=YES is correct
        if q["expected_route"] == "none":
            answer_eval["refusal_correct"] = ref["refused"]
        answer_eval["refused"] = ref["refused"]

        # Fact recall (only for non-none queries with real facts)
        if q["expected_route"] != "none":
            facts = judge_fact_recall(answer, q.get("expected_facts", []))
            answer_eval["fact_recall"] = facts["fact_recall"]
            answer_eval["fact_recall_skipped"] = facts.get("skipped", False)
            answer_eval["fact_per"] = facts["per_fact"]

            # Faithfulness vs retrieved context
            ctx = "\n\n---\n\n".join(d.page_content for d in docs)
            faith = judge_faithfulness(answer, ctx)
            answer_eval["faithfulness"] = faith["faithfulness"]

            # Answer relevance
            rel = judge_answer_relevance(question, answer)
            answer_eval["answer_relevance"] = rel["answer_relevance"]

    # 5) Timing
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
        "gold_ids": sorted(gold_set),
        "routing": routing,
        "retrieval": retrieval,
        "answer_eval": answer_eval,
        "timings": {**timings, "wall": wall},
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
        print(f"\n=== Running system: {system} ===")
        records = []
        for i, q in enumerate(testset, 1):
            print(f"  [{i}/{len(testset)}] {q['id']}: {q['question'][:60]}...")
            try:
                rec = run_one_query(q, system, args.k, use_judge, ask)
                records.append(rec)
            except Exception as e:
                print(f"    ERROR on {q['id']}: {e}")
                records.append({
                    "id": q["id"], "system": system, "error": str(e),
                    "question": q["question"],
                })
            if args.sleep:
                time.sleep(args.sleep)

        # save
        outpath = out / f"per_query_{system}.json"
        with open(outpath, "w", encoding="utf-8") as f:
            json.dump(records, f, ensure_ascii=False, indent=2)
        print(f"  -> saved {outpath}")


if __name__ == "__main__":
    main()
