"""Retrieval-quality + latency ablation on eval/retrieval_validation_set.json.

Retrieval-only: no generation, no LLM judge — we score recall@5 / precision@5 /
mrr@5 (hit@5 intentionally dropped — recall/precision/mrr already cover what it
would show, see the removal note below) AND per-stage wall-clock latency, in ONE
pass per config (avoids running the grid twice). The cross-encoder reranker is
warmed once before any config is timed, so no single cell's latency is inflated
by the one-time ~10-20s model load.

Sample: ALL 25 questions in eval/retrieval_validation_set.json — no random
subsetting. That file was purpose-built (gold chunks 25/25 manually verified
against the corpus) to be disjoint from eval/main_testset.json, so configuration
selection here never touches the final test set. Do NOT point this script at
main_testset.json — an earlier version of this script did, and that run
(eval/results/ablation/results.json) is leaked: 25 of its 115 questions were
used for both config selection and (later) final testing. That file is left on
disk as a historical artifact; this script no longer produces it.

Grid: 7 cells, not a full (dense|hybrid) x (none|cross_encoder) x (naive|
enhanced|agentic) cross product. Reranker is only evaluated on hybrid, not
dense, by explicit request:
  - naive_dense              dense retrieval, k=5, no reranker (the honest baseline)
  - enhanced_dense           dense, no reranker   (EnhancedRAGConfig dense_only)
  - agentic_dense            dense, no reranker
  - enhanced_hybrid          hybrid, no reranker  (EnhancedRAGConfig hybrid_no_rerank)
  - agentic_hybrid           hybrid, no reranker
  - enhanced_hybrid_rerank   hybrid + cross-encoder rerank (EnhancedRAGConfig default)
  - agentic_hybrid_rerank    hybrid + cross-encoder rerank

Enhanced: semantic intent gate stays ON (not bypassed) — a question that gets
gated 'invalid' scores 0 retrieval, which is itself a signal this ablation
surfaces, not noise to hide. (Checked against this exact 25-question set on
2026-08-17: 0/25 incorrectly gated — see PHASE_1_..._INSPECTION.md §20.5-adjacent
note. If the intent gate's example sets change, re-check this.) No query
rewriting axis: enhanced's rewriter is passthrough, always.

Agentic: query rewriting is OFF for every cell (allow_rewrite=False -> the
tool-call query is forced to the literal question). Self-correction is also OFF
(sufficiency_check=False, allow_reretrieve=False) — the LLM is called EXACTLY
ONCE per question (temperature=0, see ragtrial.llm), purely to pick the domain
tool(s) (routing), via ragtrial.rag.agentic.route_and_retrieve. We never reach
generation, so there's nothing to self-correct. Rerank happens PER TOOL CALL
(per domain), not once globally like enhanced — a cross-domain ('both')
question can end up with up to 2x RERANK_TOP_N docs.

Usage:
    python -m eval.run_ablation_retrieval
"""
from __future__ import annotations

import argparse
import json
import time
from dataclasses import asdict
from pathlib import Path
from typing import Any, Dict, List, Tuple

ROOT = Path(__file__).resolve().parent.parent

from eval.eval_core import (
    doc_to_gold_id,
    gold_chunks_to_set,
    recall_at_k,
    precision_at_k,
    mrr,
    nanmean,
    percentile,
)

K = 5
VALSET_PATH = ROOT / "eval" / "retrieval_validation_set.json"


def load_validation_set(path: Path) -> List[Dict[str, Any]]:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)["questions"]


def retrieval_metrics(docs, gold_chunks: Dict[str, List[str]]) -> Dict[str, float]:
    """recall/precision/mrr @K only. hit@K dropped by request: for this dataset
    (mostly one gold chunk per question) hit@K and recall@K carry the same
    information — hit is just recall's binary/all-or-nothing shadow when there's
    a single gold id, so it added a column without adding a question answered."""
    retrieved_ids = [r for r in (doc_to_gold_id(d) for d in docs) if r is not None]
    gold_set = gold_chunks_to_set(gold_chunks)
    return {
        f"recall@{K}": recall_at_k(retrieved_ids, gold_set, k=K),
        f"precision@{K}": precision_at_k(retrieved_ids, gold_set, k=K),
        f"mrr@{K}": mrr(retrieved_ids, gold_set, k=K),
    }


def _warm_reranker():
    """Force the cross-encoder's one-time ~10-20s model load BEFORE any config is timed."""
    from langchain_core.documents import Document
    from ragtrial.pipeline.rerank import rerank_documents
    rerank_documents("warmup", [Document(page_content="warmup text")], top_n=1)


# ============ Runners: (question) -> (docs, timings). Built ONCE per config. ============
def _naive_runner():
    from ragtrial.vectorstore.store import unified_store

    def run(question: str) -> Tuple[List, Dict[str, float]]:
        sub: dict = {}
        t0 = time.perf_counter()
        docs = unified_store.search(question, k=K, strategy="dense", timings=sub)
        wall = time.perf_counter() - t0
        return docs, {**sub, "wall": wall}

    return run


_shared_intent_stage = None  # one IntentStage instance shared across all enhanced cells


def _enhanced_runner(cfg):
    from ragtrial.pipeline import RERANKERS, REWRITERS, Pipeline, RetrieveStage
    from ragtrial.pipeline.intent import INTENT_GATES

    global _shared_intent_stage
    stages = []
    intent_cls = INTENT_GATES[cfg.intent]
    if intent_cls is not None:
        if _shared_intent_stage is None:
            _shared_intent_stage = intent_cls()
        stages.append(_shared_intent_stage)
    stages += [
        REWRITERS[cfg.rewriter](),
        RetrieveStage(strategy=cfg.retrieval, k=cfg.k_candidates),
        RERANKERS[cfg.reranker](top_n=cfg.top_n),
    ]
    pipeline = Pipeline(stages)

    def run(question: str) -> Tuple[List, Dict[str, float]]:
        t0 = time.perf_counter()
        state = pipeline.run(question)
        wall = time.perf_counter() - t0
        return state.documents, {**state.timings, "wall": wall}

    return run


def _agentic_runner(cfg):
    from ragtrial.rag.agentic import route_and_retrieve

    def run(question: str) -> Tuple[List, Dict[str, float]]:
        t0 = time.perf_counter()
        out = route_and_retrieve(question, cfg)
        wall = time.perf_counter() - t0
        return out["documents"], {**out["timings"], "wall": wall}

    return run


def build_configs() -> Dict[str, Any]:
    """The 7 cells requested: naive_dense; {enhanced,agentic} x {dense, hybrid,
    hybrid+rerank}. Reranker is deliberately only crossed with hybrid, not dense."""
    from ragtrial.rag.agentic import AgenticConfig
    from ragtrial.rag.enhanced import EnhancedRAGConfig

    def agentic_cfg(retrieval: str, reranker: str) -> AgenticConfig:
        return AgenticConfig(
            retrieval=retrieval,
            reranker=reranker,
            allow_rewrite=False,       # query rewriting OFF, fixed
            sufficiency_check=False,   # self-correction OFF, fixed
            allow_reretrieve=False,    # self-correction OFF, fixed
        )

    return {
        "naive_dense": None,
        "enhanced_dense": EnhancedRAGConfig(retrieval="dense", reranker="none"),
        "agentic_dense": agentic_cfg(retrieval="dense", reranker="none"),
        "enhanced_hybrid": EnhancedRAGConfig(retrieval="hybrid", reranker="none"),
        "agentic_hybrid": agentic_cfg(retrieval="hybrid", reranker="none"),
        "enhanced_hybrid_rerank": EnhancedRAGConfig(retrieval="hybrid", reranker="cross_encoder"),
        "agentic_hybrid_rerank": agentic_cfg(retrieval="hybrid", reranker="cross_encoder"),
    }


def _cfg_to_dict(cfg) -> Dict[str, Any]:
    if cfg is None:
        return {"system": "naive", "retrieval": "dense", "reranker": "none"}
    return asdict(cfg)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--valset", default=str(VALSET_PATH))
    ap.add_argument("--outdir", default=str(ROOT / "eval" / "results" / "ablation"))
    ap.add_argument("--outfile", default="results_valset.json",
                     help="separate filename from the old (leaked, main_testset-sampled) "
                          "results.json so the two are never confused or merged")
    args = ap.parse_args()

    questions = load_validation_set(Path(args.valset))

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    print(f"Validation set: {len(questions)} questions -> {args.valset}")

    print("Warming cross-encoder reranker (one-time model load, excluded from timings)...")
    _warm_reranker()

    configs = build_configs()
    print(f"Configs: {len(configs)} cells\n")

    results_path = outdir / args.outfile
    all_results: Dict[str, Any] = {}
    if results_path.exists():
        try:
            with open(results_path, encoding="utf-8") as f:
                all_results = json.load(f)
            done = [n for n in all_results if n in configs]
            if done:
                print(f"Resuming: {len(done)} cell(s) already done, skipping -> {done}")
        except (json.JSONDecodeError, OSError):
            all_results = {}

    for name, cfg in configs.items():
        if name in all_results:
            continue
        t0 = time.perf_counter()
        if name == "naive_dense":
            runner = _naive_runner()
        elif name.startswith("enhanced"):
            runner = _enhanced_runner(cfg)
        else:
            runner = _agentic_runner(cfg)

        per_query = []
        for q in questions:
            docs, timings = runner(q["question"])
            m = retrieval_metrics(docs, q["gold_chunks"])
            per_query.append({
                "id": q["id"],
                "expected_route": q["expected_route"],
                "n_docs": len(docs),
                **m,
                "timings": timings,
            })

        metric_summary = {
            key: nanmean([r[key] for r in per_query])
            for key in (f"recall@{K}", f"precision@{K}", f"mrr@{K}")
        }
        timing_keys = sorted({k for r in per_query for k in r["timings"]})
        latency_summary = {}
        for key in timing_keys:
            vals = [r["timings"][key] for r in per_query if key in r["timings"]]
            if not vals or nanmean(vals) == 0.0:
                continue
            latency_summary[key] = {
                "mean": round(nanmean(vals), 4),
                "p50": round(percentile(vals, 0.5), 4),
                "p95": round(percentile(vals, 0.95), 4),
            }

        elapsed = time.perf_counter() - t0
        all_results[name] = {
            "config": _cfg_to_dict(cfg),
            "summary": metric_summary,
            "latency_summary": latency_summary,
            "per_query": per_query,
            "elapsed_s": round(elapsed, 1),
        }
        wall_mean = latency_summary.get("wall", {}).get("mean", float("nan"))
        print(
            f"{name:24s} recall@{K}={metric_summary[f'recall@{K}']:.3f}  "
            f"precision@{K}={metric_summary[f'precision@{K}']:.3f}  mrr@{K}={metric_summary[f'mrr@{K}']:.3f}  "
            f"wall_mean={wall_mean:.3f}s  ({elapsed:.1f}s total)"
        )

        # Checkpoint after EVERY config cell — a crash on cell 6 must not lose 1-5.
        with open(results_path, "w", encoding="utf-8") as f:
            json.dump(all_results, f, ensure_ascii=False, indent=2)

    print(f"\nSaved -> {results_path}")


if __name__ == "__main__":
    main()
