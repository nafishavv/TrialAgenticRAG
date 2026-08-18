"""Pemetaan artefak generation -> input RAGAS.

Mapping yang dipakai (dan HANYA ini):

    Faithfulness        user_input         <- per_query[].question
                        response           <- per_query[].answer
                        retrieved_contexts <- per_query[].retrieved_context

    SemanticSimilarity  reference          <- main_testset.questions[].expected_answer
                        response           <- per_query[].answer

`expected_facts` dan `gold_chunks` TIDAK dipakai sama sekali. `gold_chunks`
khususnya tidak boleh menyentuh `retrieved_contexts` — itu ground-truth retrieval,
bukan konteks yang benar-benar diberikan ke generator.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple


class DatasetError(RuntimeError):
    """Input tidak layak dievaluasi — run dibatalkan, bukan diam-diam dilanjut."""


@dataclass
class RagasSample:
    """Satu (pertanyaan x arsitektur) siap dinilai."""

    question_id: str
    architecture: str
    question: str
    expected_answer: str
    generated_answer: str
    retrieved_contexts: List[str] = field(default_factory=list)
    #: alasan record tidak bisa dinilai sama sekali (mis. generation error)
    fatal_reason: Optional[str] = None

    @property
    def n_contexts(self) -> int:
        return len(self.retrieved_contexts)

    def blockers(self) -> Dict[str, str]:
        """Alasan per-metric kenapa metric itu tidak bisa dihitung.

        Dikembalikan sebagai dict {metric: reason} supaya kegagalan satu metric
        tidak ikut menjatuhkan metric lain.
        """
        out: Dict[str, str] = {}
        if self.fatal_reason:
            return {"faithfulness": self.fatal_reason, "semantic_similarity": self.fatal_reason}
        if not (self.generated_answer or "").strip():
            return {
                "faithfulness": "empty_generated_answer",
                "semantic_similarity": "empty_generated_answer",
            }
        if not self.retrieved_contexts:
            out["faithfulness"] = "empty_retrieved_context"
        if not (self.question or "").strip():
            out["faithfulness"] = "empty_question"
        if not (self.expected_answer or "").strip():
            out["semantic_similarity"] = "missing_expected_answer"
        return out


@dataclass
class SystemDataset:
    """Sample satu arsitektur + laporan coverage-nya."""

    system: str
    samples: List[RagasSample]
    n_testset: int
    missing_ids: List[str]

    @property
    def n_matched(self) -> int:
        return len(self.samples)


def load_testset(path: Path) -> Dict[str, Dict[str, Any]]:
    """Baca test set -> {id: case}. Sumber `expected_answer` (reference)."""
    if not path.exists():
        raise DatasetError(f"testset tidak ditemukan: {path}")
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    questions = data.get("questions") if isinstance(data, dict) else data
    if not questions:
        raise DatasetError(f"testset kosong / format tak dikenal: {path}")
    return {q["id"]: q for q in questions}


def load_per_query(path: Path) -> List[Dict[str, Any]]:
    if not path.exists():
        raise DatasetError(
            f"artefak generation tidak ditemukan: {path}\n"
            "Jalankan fase generation dulu:\n"
            "  python -m eval.run_eval --testset eval/main_testset.json "
            "--outdir eval/results/main"
        )
    with open(path, "r", encoding="utf-8") as f:
        records = json.load(f)
    if not isinstance(records, list):
        raise DatasetError(f"format per_query tak dikenal (harus array): {path}")
    return records


def build_dataset(
    testset: Dict[str, Dict[str, Any]],
    records: List[Dict[str, Any]],
    system: str,
    ids: Optional[set] = None,
    limit: Optional[int] = None,
) -> SystemDataset:
    """Join per_query x testset by `id`.

    Guard penting: kalau ada id di per_query yang tidak dikenal test set, run
    DIBATALKAN. Ini yang mencegah artefak lama (mis. 202-record candidate set)
    tanpa sengaja ikut dievaluasi.
    """
    unknown = sorted({r.get("id") for r in records if r.get("id") not in testset})
    if unknown:
        preview = ", ".join(str(u) for u in unknown[:10])
        raise DatasetError(
            f"[{system}] {len(unknown)} id di per_query tidak ada di testset: {preview}"
            f"{' ...' if len(unknown) > 10 else ''}\n"
            "Kemungkinan --results-dir menunjuk ke artefak dari testset yang berbeda."
        )

    samples: List[RagasSample] = []
    seen: set = set()
    for rec in records:
        rid = rec["id"]
        if ids is not None and rid not in ids:
            continue
        seen.add(rid)
        case = testset[rid]
        contexts = rec.get("retrieved_context") or []
        if isinstance(contexts, str):  # toleransi format lama
            contexts = [contexts]
        samples.append(
            RagasSample(
                question_id=rid,
                architecture=system,
                question=rec.get("question") or case.get("question", ""),
                expected_answer=case.get("expected_answer", ""),
                generated_answer=rec.get("answer") or "",
                retrieved_contexts=[c for c in contexts if isinstance(c, str) and c.strip()],
                fatal_reason=f"generation_error: {rec['error']}" if "error" in rec else None,
            )
        )

    expected_ids = set(testset) if ids is None else (set(testset) & ids)
    missing = sorted(expected_ids - seen)

    order = {qid: i for i, qid in enumerate(testset)}  # urutan testset, bukan urutan file
    samples.sort(key=lambda s: order[s.question_id])
    if limit:
        samples = samples[:limit]
    return SystemDataset(
        system=system,
        samples=samples,
        n_testset=len(expected_ids),
        missing_ids=missing,
    )


def load_system_dataset(
    testset: Dict[str, Dict[str, Any]],
    results_dir: Path,
    system: str,
    ids: Optional[set] = None,
    limit: Optional[int] = None,
) -> SystemDataset:
    records = load_per_query(results_dir / f"per_query_{system}.json")
    return build_dataset(testset, records, system, ids=ids, limit=limit)


def estimate_calls(
    datasets: List[SystemDataset], metrics: Tuple[str, ...]
) -> Dict[str, int]:
    """Estimasi biaya SEBELUM satu pun API call terjadi."""
    from .config import (
        EMBED_CALLS_PER_SEMANTIC_SIMILARITY,
        LLM_CALLS_PER_FAITHFULNESS,
        METRIC_FAITHFULNESS,
        METRIC_SEMANTIC_SIMILARITY,
    )

    n_faith = 0
    n_sim = 0
    for ds in datasets:
        for s in ds.samples:
            blocked = s.blockers()
            if METRIC_FAITHFULNESS in metrics and METRIC_FAITHFULNESS not in blocked:
                n_faith += 1
            if (
                METRIC_SEMANTIC_SIMILARITY in metrics
                and METRIC_SEMANTIC_SIMILARITY not in blocked
            ):
                n_sim += 1
    return {
        "samples_total": sum(len(ds.samples) for ds in datasets),
        "faithfulness_samples": n_faith,
        "semantic_similarity_samples": n_sim,
        "llm_calls": n_faith * LLM_CALLS_PER_FAITHFULNESS,
        "embedding_calls": n_sim * EMBED_CALLS_PER_SEMANTIC_SIMILARITY,
    }
