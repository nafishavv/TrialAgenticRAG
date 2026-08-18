"""Pembungkus tipis di atas metric bawaan RAGAS.

Yang dilakukan file ini HANYA: membangun scorer, memanggilnya, menangani error,
dan menormalkan hasilnya. Tidak ada prompt, rubrik, atau rumus skor buatan sendiri
— seluruh logika penilaian berasal dari `ragas.metrics.collections`.
"""
from __future__ import annotations

import asyncio
import hashlib
import math
from typing import Any, Dict, List, Optional, Tuple

from .config import (
    ALL_METRICS,
    METRIC_FAITHFULNESS,
    METRIC_SEMANTIC_SIMILARITY,
    RETRY_BACKOFF,
    RETRY_INITIAL_WAIT,
    RETRY_MAX,
    RETRYABLE_MARKERS,
)
from .dataset import RagasSample


def _is_retryable(exc: Exception) -> bool:
    msg = f"{type(exc).__name__}: {exc}".lower()
    return any(marker in msg for marker in RETRYABLE_MARKERS)


async def with_retry(coro_factory, *, label: str, log=print):
    """Jalankan coroutine dengan exponential backoff pada error transient.

    Pola & konstanta menyalin `_with_retry` di src/ragtrial/llm.py (max 5, wait
    awal 30s, backoff x1.5); ditulis ulang di sini karena paket ini hidup di venv
    terpisah dan sengaja tidak meng-import `ragtrial`.
    """
    wait = RETRY_INITIAL_WAIT
    last: Optional[Exception] = None
    for attempt in range(1, RETRY_MAX + 1):
        try:
            return await coro_factory()
        except Exception as exc:  # noqa: BLE001 - sengaja luas, diklasifikasi di bawah
            last = exc
            if not _is_retryable(exc) or attempt == RETRY_MAX:
                raise
            log(f"      [retry {attempt}/{RETRY_MAX - 1}] {label}: {exc} -> tunggu {wait:.0f}s")
            await asyncio.sleep(wait)
            wait *= RETRY_BACKOFF
    raise last  # pragma: no cover - tidak tercapai


def sha1(s: str) -> str:
    return hashlib.sha1((s or "").encode("utf-8")).hexdigest()


def cache_key(sample: RagasSample, metric: str, model_id: str) -> str:
    """Kunci cache skor.

    Berbeda dari judge_cache lama yang hanya mem-hash jawaban: di sini konteks
    (atau reference) DAN identitas model ikut masuk kunci, sehingga mengganti
    evaluator/embedding otomatis meng-invalidate cache — tidak perlu hapus file
    manual.
    """
    if metric == METRIC_FAITHFULNESS:
        second = sha1("\n".join(sample.retrieved_contexts))
    else:
        second = sha1(sample.expected_answer)
    return (
        f"{sample.question_id}|{sample.architecture}|{metric}|"
        f"{sha1(sample.generated_answer)}|{second}|{model_id}"
    )


class RagasScorer:
    """Menilai satu sample dengan metric RAGAS yang diminta."""

    def __init__(
        self,
        metrics: Tuple[str, ...] = ALL_METRICS,
        judge_model: Optional[str] = None,
        judge_provider: str = "openai",
        embedding_model: Optional[str] = None,
        embedding_provider: str = "openai",
        log=print,
    ) -> None:
        from . import config

        self.metrics = tuple(metrics)
        self.judge_model = judge_model or config.JUDGE_MODEL
        self.judge_provider = judge_provider
        self.embedding_model = embedding_model or config.EMBEDDING_MODEL
        self.embedding_provider = embedding_provider
        self.log = log
        self.n_llm_calls = 0
        self.n_embed_calls = 0

        from ragas.metrics.collections import Faithfulness, SemanticSimilarity

        self._faithfulness = None
        self._similarity = None
        if METRIC_FAITHFULNESS in self.metrics:
            self._faithfulness = Faithfulness(
                llm=config.build_judge_llm(
                    self.judge_model, config.JUDGE_TEMPERATURE, provider=self.judge_provider
                )
            )
        if METRIC_SEMANTIC_SIMILARITY in self.metrics:
            self._similarity = SemanticSimilarity(
                embeddings=config.build_embeddings(
                    self.embedding_model, provider=self.embedding_provider
                )
            )

    def model_id(self, metric: str) -> str:
        """Identitas model yang menilai metric ini — ikut masuk cache key."""
        if metric == METRIC_FAITHFULNESS:
            return f"{self.judge_provider}:{self.judge_model}@t0"
        return f"{self.embedding_provider}:{self.embedding_model}"

    @staticmethod
    def _clean(result: Any) -> Optional[float]:
        """MetricResult -> float, atau None kalau NaN (mis. tak ada statement)."""
        value = float(result.value)
        if math.isnan(value):
            return None
        return value

    async def score_metric(self, sample: RagasSample, metric: str) -> Tuple[Optional[float], Optional[str]]:
        """Hitung SATU metric. Return (score, error_reason)."""
        try:
            if metric == METRIC_FAITHFULNESS:
                result = await with_retry(
                    lambda: self._faithfulness.ascore(
                        user_input=sample.question,
                        response=sample.generated_answer,
                        retrieved_contexts=sample.retrieved_contexts,
                    ),
                    label=f"{sample.question_id}/{sample.architecture}/faithfulness",
                    log=self.log,
                )
                self.n_llm_calls += 2  # statement extraction + NLI verification
            else:
                result = await with_retry(
                    lambda: self._similarity.ascore(
                        reference=sample.expected_answer,
                        response=sample.generated_answer,
                    ),
                    label=f"{sample.question_id}/{sample.architecture}/semantic_similarity",
                    log=self.log,
                )
                self.n_embed_calls += 2  # reference + response
        except Exception as exc:  # noqa: BLE001 - dicatat, tidak menjatuhkan run
            return None, f"{type(exc).__name__}: {exc}"[:500]

        score = self._clean(result)
        if score is None:
            return None, "nan_no_statements_extracted"
        return score, None

    async def score_sample(
        self,
        sample: RagasSample,
        cache: Dict[str, Any],
        metrics: Optional[Tuple[str, ...]] = None,
    ) -> Dict[str, Any]:
        """Nilai satu sample untuk semua metric. Selalu mengembalikan record penuh."""
        metrics = metrics or self.metrics
        blocked = sample.blockers()
        scores: Dict[str, Optional[float]] = {}
        errors: Dict[str, str] = {}

        for metric in metrics:
            if metric in blocked:
                scores[metric] = None
                errors[metric] = blocked[metric]
                continue

            key = cache_key(sample, metric, self.model_id(metric))
            if key in cache:
                entry = cache[key]
                scores[metric] = entry.get("score")
                if entry.get("error"):
                    errors[metric] = entry["error"]
                continue

            score, error = await self.score_metric(sample, metric)
            scores[metric] = score
            if error:
                errors[metric] = error
            # Error transien TIDAK di-cache: biar --resume bisa mencobanya lagi.
            # Hanya hasil final (skor, atau NaN yang memang deterministik) disimpan.
            if error is None or error == "nan_no_statements_extracted":
                cache[key] = {"score": score, "error": error}

        n_ok = sum(1 for m in metrics if scores.get(m) is not None)
        if n_ok == len(metrics):
            status = "ok"
        elif n_ok == 0:
            status = "failed"
        else:
            status = "partial"

        return {
            "question_id": sample.question_id,
            "architecture": sample.architecture,
            "question": sample.question,
            "expected_answer": sample.expected_answer,
            "generated_answer": sample.generated_answer,
            "retrieved_contexts": sample.retrieved_contexts,
            "n_contexts": sample.n_contexts,
            METRIC_FAITHFULNESS: scores.get(METRIC_FAITHFULNESS),
            METRIC_SEMANTIC_SIMILARITY: scores.get(METRIC_SEMANTIC_SIMILARITY),
            "status": status,
            "errors": errors,
        }
