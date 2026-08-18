"""Eval utilities — retrieval/routing metrics, gold ID normalization.

Designed to be system-agnostic: works on dict output of `ask_agentic(...)` (with
route + documents + answer) and `ask_main(...)` (without route).

Gold ID format:
    dukcapil  → "page:<page_start>"
    opd       → "nomor:<nomor>"
    sosial    → "id:<id>"

Kualitas jawaban TIDAK dinilai di sini. Custom LLM-as-a-Judge yang dulu menghuni
file ini sudah dipensiunkan (18 Agustus 2026) dan dipindah ke
archive/eval_judge/custom_judge.py. Penggantinya: evaluasi berbasis RAGAS di
eval/run_ragas.py (Faithfulness + Semantic Similarity).
"""
from __future__ import annotations

from typing import Any, Dict, List, Optional

from langchain_core.documents import Document

from ragtrial.capabilities.registry import CAPABILITIES, SEARCHABLE_CAPABILITIES


# ============================================================
# 1. GOLD ID NORMALIZATION
# ============================================================
def doc_to_gold_id(doc: Document) -> Optional[str]:
    """Convert a retrieved Document into a normalized gold-ID string.

    Delegates to the owning capability's `gold_id()` (resolved via the `_source`
    tag), so eval carries zero source-specific metadata knowledge. Falls back to
    probing every capability when `_source` is absent.
    """
    src = (doc.metadata or {}).get("_source")
    if src and src in CAPABILITIES:
        return CAPABILITIES[src].gold_id(doc)
    for cap in CAPABILITIES.values():
        gid = cap.gold_id(doc)
        if gid is not None:
            return gid
    return None


def gold_chunks_to_set(gold: Dict[str, List[str]]) -> set[str]:
    """Convert testset.gold_chunks dict into a flat set of normalized IDs.

    Input: {"dukcapil": ["page:40"], "opd": ["nomor:1.a"]}
    Output: {"dukcapil:page:40", "opd:nomor:1.a"}
    """
    out: set[str] = set()
    for source, ids in (gold or {}).items():
        for gid in ids:
            out.add(f"{source}:{gid}")
    return out


# ============================================================
# 2. RETRIEVAL METRICS
# ============================================================
def hit_at_k(retrieved_ids: List[str], gold_ids: set[str], k: int = 5) -> float:
    """1.0 if any gold ID appears in top-k retrieved, else 0.0. (binary)

    If gold_ids is empty (out-of-scope query), returns NaN-equivalent: we skip
    the question in the aggregator. Here we return 1.0 if also retrieved is
    empty or irrelevant; caller should typically not call this for none-route.
    """
    if not gold_ids:
        return float("nan")
    topk = retrieved_ids[:k]
    return 1.0 if any(gid in gold_ids for gid in topk) else 0.0


def recall_at_k(retrieved_ids: List[str], gold_ids: set[str], k: int = 5) -> float:
    """Fraction of gold IDs that appear in top-k retrieved."""
    if not gold_ids:
        return float("nan")
    topk = set(retrieved_ids[:k])
    return len(topk & gold_ids) / len(gold_ids)


def precision_at_k(retrieved_ids: List[str], gold_ids: set[str], k: int = 5) -> float:
    """Fraction of top-k retrieved that are in gold (penalizes over-retrieval)."""
    if not gold_ids:
        return float("nan")
    topk = retrieved_ids[:k]
    if not topk:
        return 0.0
    return sum(1 for gid in topk if gid in gold_ids) / len(topk)


def mrr(retrieved_ids: List[str], gold_ids: set[str], k: int = 5) -> float:
    """Mean Reciprocal Rank: 1/rank of first gold hit, else 0."""
    if not gold_ids:
        return float("nan")
    for i, gid in enumerate(retrieved_ids[:k], start=1):
        if gid in gold_ids:
            return 1.0 / i
    return 0.0


# ============================================================
# 3. ROUTING METRICS
# ============================================================
def routing_labels() -> List[str]:
    """Routing label space, derived from the registry: searchable caps + both/none."""
    return list(SEARCHABLE_CAPABILITIES.keys()) + ["both", "none"]


def routing_eval(
    predictions: List[str],
    gold: List[str],
    labels: Optional[List[str]] = None,
) -> Dict[str, Any]:
    """Compute confusion matrix + per-class P/R/F1 + overall accuracy.

    `labels` defaults to the registry-derived routing label space.
    """
    if labels is None:
        labels = routing_labels()
    cm = {g: {p: 0 for p in labels} for g in labels}
    for p, g in zip(predictions, gold):
        if g in labels and p in labels:
            cm[g][p] += 1

    per_class: Dict[str, Dict[str, float]] = {}
    for c in labels:
        tp = cm[c][c]
        fp = sum(cm[other][c] for other in labels if other != c)
        fn = sum(cm[c][other] for other in labels if other != c)
        prec = tp / (tp + fp) if (tp + fp) else 0.0
        rec = tp / (tp + fn) if (tp + fn) else 0.0
        f1 = 2 * prec * rec / (prec + rec) if (prec + rec) else 0.0
        per_class[c] = {"precision": prec, "recall": rec, "f1": f1, "support": tp + fn}

    n = len(predictions)
    correct = sum(1 for p, g in zip(predictions, gold) if p == g)
    accuracy = correct / n if n else 0.0

    return {
        "accuracy": accuracy,
        "confusion_matrix": cm,
        "per_class": per_class,
        "n": n,
    }


def intent_eval(predictions: List[str], gold: List[str]) -> Dict[str, Any]:
    """Binary intent-handling metric (paper: VALID vs INVALID retrieval decision).

    Same confusion-matrix + per-class P/R/F1 shape as routing_eval, fixed to the
    2-class label space. The headline number per the reference paper is recall on
    INVALID (did the system correctly avoid retrieval?) and macro-F1.
    """
    res = routing_eval(predictions, gold, labels=["valid", "invalid"])
    pc = res["per_class"]
    res["macro_f1"] = (pc["valid"]["f1"] + pc["invalid"]["f1"]) / 2
    res["recall_invalid"] = pc["invalid"]["recall"]
    res["recall_valid"] = pc["valid"]["recall"]
    return res


# ============================================================
# 4. STORE-CORRECTNESS (agentic only)
# ============================================================
def store_correct(expected_route: str, source_used: str) -> Optional[bool]:
    """Did agentic query the correct store(s)?
    Returns None if expected_route='none' (no store should be queried).
    """
    if expected_route == "none":
        return source_used == "none" or source_used == ""
    return source_used == expected_route


# ============================================================
# 5. AGGREGATION HELPERS
# ============================================================
def nanmean(xs: List[float]) -> float:
    vals = [x for x in xs if isinstance(x, (int, float)) and x == x]  # filter NaN
    return sum(vals) / len(vals) if vals else float("nan")


def percentile(xs: List[float], p: float) -> float:
    if not xs:
        return float("nan")
    s = sorted(xs)
    k = (len(s) - 1) * p
    f = int(k)
    c = min(f + 1, len(s) - 1)
    if f == c:
        return s[f]
    return s[f] + (s[c] - s[f]) * (k - f)
