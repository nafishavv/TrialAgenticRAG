"""Agregasi per arsitektur.

Sengaja ringkas: mean/median/std + hitungan sample. Per-question tetap menjadi
sumber kebenaran; agregat hanya ringkasan di atasnya.
"""
from __future__ import annotations

import math
from typing import Any, Dict, List, Optional, Sequence, Tuple

from .config import ALL_METRICS, METRIC_FAITHFULNESS, METRIC_SEMANTIC_SIMILARITY


def _median(xs: Sequence[float]) -> float:
    s = sorted(xs)
    n = len(s)
    mid = n // 2
    return s[mid] if n % 2 else (s[mid - 1] + s[mid]) / 2


def _stdev(xs: Sequence[float]) -> float:
    """Standard deviation sampel (n-1); 0.0 kalau cuma satu titik."""
    n = len(xs)
    if n < 2:
        return 0.0
    mean = sum(xs) / n
    return math.sqrt(sum((x - mean) ** 2 for x in xs) / (n - 1))


def metric_stats(values: List[Optional[float]]) -> Dict[str, Any]:
    ok = [v for v in values if isinstance(v, (int, float)) and not math.isnan(v)]
    if not ok:
        return {
            "mean": None, "median": None, "std": None, "min": None, "max": None,
            "n_evaluated": 0, "n_failed": len(values),
        }
    return {
        "mean": sum(ok) / len(ok),
        "median": _median(ok),
        "std": _stdev(ok),
        "min": min(ok),
        "max": max(ok),
        "n_evaluated": len(ok),
        "n_failed": len(values) - len(ok),
    }


def summarize(
    records: List[Dict[str, Any]],
    system: str,
    *,
    n_testset: int,
    missing_ids: List[str],
    metrics: Tuple[str, ...] = ALL_METRICS,
) -> Dict[str, Any]:
    summary: Dict[str, Any] = {
        "architecture": system,
        "n_testset": n_testset,
        "n_records": len(records),
        "n_missing_generation": len(missing_ids),
        "missing_ids": missing_ids,
        "status_counts": {
            status: sum(1 for r in records if r.get("status") == status)
            for status in ("ok", "partial", "failed")
        },
    }
    for metric in metrics:
        summary[metric] = metric_stats([r.get(metric) for r in records])

    reasons: Dict[str, int] = {}
    for r in records:
        for metric, reason in (r.get("errors") or {}).items():
            reasons[f"{metric}: {reason.split(':')[0]}"] = (
                reasons.get(f"{metric}: {reason.split(':')[0]}", 0) + 1
            )
    summary["error_reasons"] = dict(sorted(reasons.items(), key=lambda kv: -kv[1]))
    return summary


def _fmt(value: Optional[float], width: int = 6) -> str:
    return "  n/a " if value is None else f"{value:.{max(width - 3, 1)}f}"


def render_table(summaries: List[Dict[str, Any]]) -> str:
    """Tabel perbandingan antar arsitektur, gaya SUMMARY_main.txt."""
    lines: List[str] = []
    lines.append("=" * 78)
    lines.append("RAGAS GENERATION EVALUATION — perbandingan arsitektur")
    lines.append("=" * 78)
    lines.append("")
    header = f"{'architecture':<14}{'faithfulness':>26}{'semantic_similarity':>26}"
    lines.append(header)
    lines.append(f"{'':<14}{'mean   median   std   n':>26}{'mean   median   std   n':>26}")
    lines.append("-" * 78)
    for s in summaries:
        row = f"{s['architecture']:<14}"
        for metric in (METRIC_FAITHFULNESS, METRIC_SEMANTIC_SIMILARITY):
            st = s.get(metric) or {}
            row += (
                f"{_fmt(st.get('mean')):>8}{_fmt(st.get('median')):>8}"
                f"{_fmt(st.get('std')):>7}{st.get('n_evaluated', 0):>3}"
            )
        lines.append(row)
    lines.append("-" * 78)
    lines.append("")
    for s in summaries:
        lines.append(
            f"{s['architecture']:<14} records={s['n_records']:<4} "
            f"ok={s['status_counts']['ok']:<4} partial={s['status_counts']['partial']:<4} "
            f"failed={s['status_counts']['failed']:<4} "
            f"missing_generation={s['n_missing_generation']}"
        )
        if s.get("error_reasons"):
            for reason, count in s["error_reasons"].items():
                lines.append(f"{'':<14}  ! {reason} x{count}")
    lines.append("")
    lines.append("Faithfulness       : response vs retrieved_contexts (TANPA ground truth)")
    lines.append("Semantic Similarity: response vs expected_answer (ground truth)")
    lines.append("Rentang kedua metric 0-1, makin tinggi makin baik.")
    return "\n".join(lines)
