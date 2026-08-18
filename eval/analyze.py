"""Aggregate per-query results → summary metrics + breakdown per dimensi.

Cara pakai:
    python -m eval.analyze
    python -m eval.analyze --system agentic
    python -m eval.analyze --breakdown query_type difficulty
"""
from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List

from eval.eval_core import routing_eval, nanmean, percentile

ROOT = Path(__file__).resolve().parent.parent
RESULTS = ROOT / "eval" / "results"


def load(system: str) -> List[Dict[str, Any]]:
    path = RESULTS / f"per_query_{system}.json"
    if not path.exists():
        return []
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def aggregate(records: List[Dict[str, Any]], k: int = 5) -> Dict[str, Any]:
    """Compute global metric averages across all records."""
    valid = [r for r in records if "error" not in r]
    non_none = [r for r in valid if r["expected_route"] != "none"]
    none_only = [r for r in valid if r["expected_route"] == "none"]

    summary: Dict[str, Any] = {"n_total": len(records), "n_valid": len(valid)}

    # ---- Routing (agentic only) ----
    routing_records = [r for r in valid if r.get("routing")]
    if routing_records:
        preds = [r["routing"]["predicted_route"] for r in routing_records]
        golds = [r["routing"]["expected_route"] for r in routing_records]
        summary["routing"] = routing_eval(preds, golds)
        # store correctness
        store_oks = [r["routing"]["store_correct"] for r in routing_records
                     if r["routing"]["store_correct"] is not None]
        summary["routing"]["store_correct_rate"] = nanmean([1.0 if x else 0.0 for x in store_oks])

    # ---- Retrieval ----
    retr_keys = [f"hit@{k}", f"recall@{k}", f"precision@{k}", f"mrr@{k}"]
    retr = {}
    for key in retr_keys:
        vals = [r["retrieval"].get(key) for r in non_none if r.get("retrieval") and key in r["retrieval"]]
        retr[key] = nanmean(vals)
    summary["retrieval"] = retr

    # ---- Answer quality ----
    # DIPINDAH ke fase RAGAS: lihat eval/run_ragas.py -> summary_ragas_<system>.json
    # (faithfulness + semantic_similarity). Custom judge lama diarsipkan di
    # archive/eval_judge/. Skrip ini sekarang murni retrieval/routing/latency.

    # ---- Latency (per-stage) ----
    # Union across tiers; keys absent for a tier are simply skipped. Zero-valued
    # entries (e.g. route=0.0, or t_bm25=0.0 for naive dense) are dropped so a
    # stage only appears where it actually ran.
    lat_keys = [
        "t_embed_query", "t_search", "t_bm25", "t_fuse",  # retrieve sub-stages (P1.1)
        "retrieve",                                       # whole retrieve (back-compat)
        "rerank", "t_rerank",                             # cross-encoder
        "intent", "route", "rewrite", "t_decision",       # decision overhead
        "t_orchestrate",                                  # agentic tool-selection turns
        "generate", "t_generate",                         # final answer synthesis
        "total", "wall",                                  # end-to-end
    ]
    lat = {}
    for key in lat_keys:
        vals = [r["timings"].get(key, 0.0) for r in valid if r.get("timings")]
        vals = [v for v in vals if v]
        if vals:
            lat[key] = {
                "mean": nanmean(vals),
                "p50": percentile(vals, 0.5),
                "p95": percentile(vals, 0.95),
            }
    summary["latency"] = lat

    # ---- Counts (LLM calls / agentic loop iterations) ----
    counts = {}
    for key in ["n_llm_calls", "n_iterations"]:
        vals = [r["timings"].get(key) for r in valid
                if r.get("timings") and isinstance(r["timings"].get(key), (int, float))]
        if vals:
            counts[key] = {"mean": nanmean(vals), "p50": percentile(vals, 0.5),
                           "p95": percentile(vals, 0.95)}
    summary["counts"] = counts

    # ---- Decisions breakdown (rewrite / re-retrieve tagging, free from the log) ----
    dec_records = [r for r in valid if r.get("decisions")]
    if dec_records:
        rewrite_ids = [r["id"] for r in dec_records if r["decisions"].get("rewrite")]
        iters = [(r["id"], int(r["decisions"].get("iterations", 1))) for r in dec_records]
        re_retrieve_ids = [rid for rid, it in iters if it > 1]
        summary["decisions"] = {
            "rewrite_rate": len(rewrite_ids) / len(dec_records),
            "avg_iterations": nanmean([it for _, it in iters]),
            "re_retrieve_rate": len(re_retrieve_ids) / len(dec_records),
            "rewrite_ids": rewrite_ids,
            "re_retrieve_ids": re_retrieve_ids,
        }

    return summary


def breakdown(records: List[Dict[str, Any]], dim: str, k: int = 5) -> Dict[str, Dict[str, Any]]:
    """Group records by `dim` (e.g. 'query_type'), return per-group aggregates."""
    groups: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    for r in records:
        groups[r.get(dim, "?")].append(r)
    return {g: aggregate(rs, k=k) for g, rs in groups.items()}


def print_summary(name: str, s: Dict[str, Any]) -> None:
    print(f"\n{'='*60}\n  {name.upper()}  (n={s['n_valid']}/{s['n_total']})\n{'='*60}")

    if "routing" in s:
        r = s["routing"]
        print(f"Routing:    accuracy={r['accuracy']:.3f}  store_correct={r.get('store_correct_rate', 0):.3f}")
        for c, m in r["per_class"].items():
            print(f"  {c:9s}  P={m['precision']:.2f}  R={m['recall']:.2f}  F1={m['f1']:.2f}  (n={m['support']})")
        print("  Confusion matrix (rows=gold, cols=pred):")
        labels = list(r["confusion_matrix"].keys())
        print("           " + "  ".join(f"{l:>9}" for l in labels))
        for g in labels:
            row = "  ".join(f"{r['confusion_matrix'][g][p]:>9d}" for p in labels)
            print(f"  {g:9s} {row}")

    if "retrieval" in s:
        rt = s["retrieval"]
        print(f"Retrieval:  " + "  ".join(f"{k}={v:.3f}" for k, v in rt.items() if v == v))

    if "latency" in s:
        lat = s["latency"]
        for k, v in lat.items():
            print(f"Latency {k:14s}  mean={v['mean']:.2f}s  p50={v['p50']:.2f}s  p95={v['p95']:.2f}s")

    if s.get("counts"):
        for k, v in s["counts"].items():
            print(f"Count   {k:14s}  mean={v['mean']:.2f}   p50={v['p50']:.2f}   p95={v['p95']:.2f}")

    if s.get("decisions"):
        d = s["decisions"]
        print(f"Decisions:  rewrite_rate={d['rewrite_rate']:.3f}  "
              f"avg_iterations={d['avg_iterations']:.2f}  re_retrieve_rate={d['re_retrieve_rate']:.3f}")
        if d["rewrite_ids"]:
            print(f"  rewrite ids ({len(d['rewrite_ids'])}): {', '.join(d['rewrite_ids'])}")
        if d["re_retrieve_ids"]:
            print(f"  re-retrieve ids ({len(d['re_retrieve_ids'])}): {', '.join(d['re_retrieve_ids'])}")


def print_comparison(summaries: Dict[str, Dict[str, Any]], k: int = 5) -> None:
    """Print a side-by-side comparison table across all systems."""
    systems = list(summaries.keys())
    w = 12

    def _fmt(v) -> str:
        if isinstance(v, float) and v != v:
            return "-".center(w)
        if isinstance(v, float):
            return f"{v:.3f}".center(w)
        return str(v).center(w)

    print(f"\n{'='*60}")
    print("  COMPARISON TABLE")
    print(f"{'='*60}")
    header = "Metric".ljust(28) + "".join(s.center(w) for s in systems)
    print(header)
    print("-" * len(header))

    rows = []

    # Retrieval
    for key in [f"hit@{k}", f"recall@{k}", f"precision@{k}", f"mrr@{k}"]:
        vals = [summaries[s].get("retrieval", {}).get(key, float("nan")) for s in systems]
        rows.append((f"retrieval.{key}", vals))

    # Answer quality -> lihat eval/results/ragas/SUMMARY_ragas.txt (fase RAGAS).

    # Routing (agentic/enhanced only)
    for key in ["accuracy", "store_correct_rate"]:
        vals = [summaries[s].get("routing", {}).get(key, float("nan")) for s in systems]
        rows.append((f"routing.{key}", vals))

    # Latency — per-stage means (union across tiers) + key p95s
    for stage in ["t_embed_query", "t_search", "t_bm25", "t_fuse", "retrieve",
                  "rerank", "t_orchestrate", "t_generate", "generate", "total", "wall"]:
        vals = [summaries[s].get("latency", {}).get(stage, {}).get("mean", float("nan"))
                for s in systems]
        rows.append((f"latency.{stage}(mean)", vals))
    for stage in ["retrieve", "generate", "total"]:
        vals = [summaries[s].get("latency", {}).get(stage, {}).get("p95", float("nan"))
                for s in systems]
        rows.append((f"latency.{stage}(p95)", vals))

    # Counts
    for key in ["n_llm_calls", "n_iterations"]:
        vals = [summaries[s].get("counts", {}).get(key, {}).get("mean", float("nan"))
                for s in systems]
        rows.append((f"counts.{key}(mean)", vals))

    for label, vals in rows:
        print(label.ljust(28) + "".join(_fmt(v) for v in vals))

    print()


def load_testset_ids(path: Path) -> set:
    """Set of question ids in the testset (for stale-id detection)."""
    if not path.exists():
        return set()
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    return {q["id"] for q in data.get("questions", [])}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--systems", nargs="+", default=["naive", "enhanced", "agentic"])
    ap.add_argument("--breakdown", nargs="+", default=[],
                    help="dimensi breakdown: query_type difficulty expected_route chunk_scope")
    ap.add_argument("--k", type=int, default=5)
    ap.add_argument("--ids", nargs="+", default=None,
                    help="aggregate over ONLY these question ids (subset agregasi)")
    ap.add_argument("--testset", default=str(ROOT / "eval" / "candidate_testset.json"),
                    help="testset for stale-id detection (ids in results not in testset)")
    ap.add_argument("--save", action="store_true",
                    help="simpan summary ke eval/results/summary_<system>.json")
    ap.add_argument("--no-compare", action="store_true",
                    help="skip comparison table")
    args = ap.parse_args()

    id_filter = set(args.ids) if args.ids else None
    testset_ids = load_testset_ids(Path(args.testset))

    summaries: Dict[str, Dict[str, Any]] = {}

    for system in args.systems:
        recs = load(system)
        if not recs:
            print(f"[skip] no records for {system}")
            continue

        # (d) warn about stale ids present in results but absent from the testset.
        if testset_ids:
            stale = sorted({r["id"] for r in recs if r.get("id") not in testset_ids})
            if stale:
                print(f"[warn] {system}: {len(stale)} stale id(s) not in testset: "
                      f"{', '.join(stale)}")

        # (b) subset filter — aggregate over only the requested ids.
        if id_filter is not None:
            recs = [r for r in recs if r.get("id") in id_filter]
            if not recs:
                print(f"[skip] {system}: no records match --ids filter")
                continue

        s = aggregate(recs, k=args.k)
        summaries[system] = s
        print_summary(system, s)

        for dim in args.breakdown:
            print(f"\n  -- breakdown by {dim} ({system}) --")
            bd = breakdown(recs, dim, k=args.k)
            for g, sub in sorted(bd.items()):
                line_parts = [f"{dim}={g} (n={sub['n_valid']})"]
                if "retrieval" in sub and sub["retrieval"]:
                    line_parts.append(f"hit@{args.k}={sub['retrieval'].get(f'hit@{args.k}', float('nan')):.2f}")
                    line_parts.append(f"recall={sub['retrieval'].get(f'recall@{args.k}', float('nan')):.2f}")
                print("   " + "  ".join(line_parts))

        if args.save:
            outpath = RESULTS / f"summary_{system}.json"
            with open(outpath, "w", encoding="utf-8") as f:
                json.dump({"global": s, "breakdowns": {d: breakdown(recs, d, k=args.k)
                                                       for d in args.breakdown}},
                          f, ensure_ascii=False, indent=2, default=str)
            print(f"\n  -> saved {outpath}")

    if len(summaries) > 1 and not args.no_compare:
        print_comparison(summaries, k=args.k)


if __name__ == "__main__":
    main()
