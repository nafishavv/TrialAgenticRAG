"""DEPRECATED — do not run. Pending deletion once the retrieval ablation is settled.

Two problems, both fatal:
1. Sampled from eval/main_testset.json via build_subset(seed=42, n_per_domain=2) —
   a DIFFERENT draw than the main grid's n_per_domain=5 (same seed, different
   sample size => different RNG consumption => not a subset of the grid's 25
   questions). Its results were never comparable to the grid it was meant to
   extend. See PHASE_1_..._INSPECTION.md §20.3 (A-6).
2. It imports `build_subset` and `load_testset` from run_ablation_retrieval.py,
   both of which no longer exist there — that script now loads the fixed
   eval/retrieval_validation_set.json directly, no sampling. This file will
   raise ImportError if run as-is.

Its output (rewrite_trial_LEAKED_deprecated.json) is kept only as a historical
record of what NOT to do; do not treat its numbers as valid.

If/when a query-rewriting ablation axis is designed properly, rebuild this
against eval/retrieval_validation_set.json (all 25 questions, no subsetting)
so the "rewrite on" and "rewrite off" cells share the exact same questions.

----- Original docstring, for reference -----
Quick trial: agentic WITH query rewriting on, dense_rerank + hybrid_rerank only.

The main ablation (run_ablation_retrieval.py) fixed allow_rewrite=False for all
agentic cells. This is a cheap add-on to see whether letting the LLM rewrite the
tool-call query (vs forcing the literal question) changes retrieval quality for
the two rerank-on cells specifically. Small subset (10 questions: 2 per domain +
2 cross-domain) to keep it a fast/cheap trial run.

Usage (DO NOT RUN — see deprecation note above):
    python -m eval.run_ablation_rewrite_trial
"""
from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any, Dict

ROOT = Path(__file__).resolve().parent.parent

from eval.eval_core import nanmean, percentile
from eval.run_ablation_retrieval import (
    K,
    _agentic_runner,
    build_subset,
    load_testset,
    retrieval_metrics,
)

SEED = 42
N_PER_DOMAIN = 2
OUT_PATH = ROOT / "eval" / "results" / "ablation" / "rewrite_trial.json"


def main():
    from ragtrial.rag.agentic import AgenticConfig

    testset = load_testset(ROOT / "eval" / "main_testset.json")
    subset = build_subset(testset, seed=SEED, n_per_domain=N_PER_DOMAIN)
    print(f"Subset: {len(subset)} questions -> {[q['id'] for q in subset]}\n")

    configs = {
        "agentic_dense_rerank_rewrite": AgenticConfig(
            retrieval="dense", reranker="cross_encoder",
            allow_rewrite=True, sufficiency_check=False, allow_reretrieve=False,
        ),
        "agentic_hybrid_rerank_rewrite": AgenticConfig(
            retrieval="hybrid", reranker="cross_encoder",
            allow_rewrite=True, sufficiency_check=False, allow_reretrieve=False,
        ),
    }

    all_results: Dict[str, Any] = {}
    for name, cfg in configs.items():
        t0 = time.perf_counter()
        runner = _agentic_runner(cfg)

        per_query = []
        for q in subset:
            docs, timings = runner(q["question"])
            m = retrieval_metrics(docs, q["gold_chunks"])
            per_query.append({
                "id": q["id"], "expected_route": q["expected_route"],
                "n_docs": len(docs), **m, "timings": timings,
            })

        metric_summary = {
            key: nanmean([r[key] for r in per_query])
            for key in (f"hit@{K}", f"recall@{K}", f"precision@{K}", f"mrr@{K}")
        }
        wall_vals = [r["timings"]["wall"] for r in per_query]
        elapsed = time.perf_counter() - t0
        all_results[name] = {
            "config": cfg.__dict__,
            "summary": metric_summary,
            "wall_mean": round(nanmean(wall_vals), 3),
            "per_query": per_query,
            "elapsed_s": round(elapsed, 1),
        }
        print(
            f"{name:34s} hit@{K}={metric_summary[f'hit@{K}']:.3f}  recall@{K}={metric_summary[f'recall@{K}']:.3f}  "
            f"precision@{K}={metric_summary[f'precision@{K}']:.3f}  mrr@{K}={metric_summary[f'mrr@{K}']:.3f}  "
            f"wall_mean={nanmean(wall_vals):.3f}s  ({elapsed:.1f}s)"
        )

    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(OUT_PATH, "w", encoding="utf-8") as f:
        json.dump(all_results, f, ensure_ascii=False, indent=2)
    print(f"\nSaved -> {OUT_PATH}")


if __name__ == "__main__":
    main()
