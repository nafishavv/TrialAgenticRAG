"""Konfigurasi evaluator RAGAS — satu-satunya tempat model & versi didefinisikan.

Semua keputusan metodologis yang perlu dikutip di Bab III skripsi ada di file ini,
dan ikut ditulis ke setiap file summary lewat `provenance()` supaya hasil evaluasi
selalu membawa jejak konfigurasinya sendiri.
"""
from __future__ import annotations

import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict

# eval/ragas_eval/config.py -> eval/ragas_eval -> eval -> <root>
ROOT = Path(__file__).resolve().parent.parent.parent

# ---------------------------------------------------------------- versi & model
RAGAS_VERSION = "0.4.3"

#: Evaluator LLM untuk Faithfulness. SENGAJA vendor lain (OpenAI) dari generator
#: produksi (Gemini, src/ragtrial/llm.py: LLM_MODEL) — supaya evaluator dan yang
#: dievaluasi tidak berasal dari model/vendor yang sama, menekan potensi
#: self-preference bias. Butuh OPENAI_API_KEY sendiri (lihat load_openai_api_key).
JUDGE_MODEL = "gpt-4o-mini"
JUDGE_TEMPERATURE = 0.0

#: Embedding untuk Semantic Similarity. SENGAJA berbeda dari embedding retrieval
#: (`models/gemini-embedding-2`) supaya model yang sama tidak sekaligus memilih
#: bukti dan menilai jawaban. Default OpenAI `text-embedding-3-small` — pilihan
#: paling umum di ekosistem RAGAS (dipakai di hampir semua contoh dokumentasi
#: resminya), dan me-reuse OPENAI_API_KEY yang sudah dibutuhkan untuk judge, jadi
#: tidak menambah key baru. Alternatif tanpa key OpenAI: `gemini-embedding-001`
#: via --embedding-provider google (masih GEMINI_API_KEY).
EMBEDDING_MODEL = "text-embedding-3-small"
GEMINI_EMBEDDING_MODEL = "gemini-embedding-001"

# ---------------------------------------------------------------- metric & path
METRIC_FAITHFULNESS = "faithfulness"
METRIC_SEMANTIC_SIMILARITY = "semantic_similarity"
ALL_METRICS = (METRIC_FAITHFULNESS, METRIC_SEMANTIC_SIMILARITY)

#: Faithfulness = 2 panggilan LLM per sample: statement extraction lalu NLI
#: verification (lihat Faithfulness.ascore di ragas 0.4.3). Semantic Similarity
#: nol panggilan LLM, hanya embedding.
LLM_CALLS_PER_FAITHFULNESS = 2
EMBED_CALLS_PER_SEMANTIC_SIMILARITY = 2

DEFAULT_TESTSET = ROOT / "eval" / "main_testset.json"
DEFAULT_RESULTS_DIR = ROOT / "eval" / "results" / "main"
DEFAULT_OUTDIR = ROOT / "eval" / "results" / "ragas"
DEFAULT_SYSTEMS = ("naive", "enhanced", "agentic")

CACHE_FILENAME = "ragas_cache.json"

# ---------------------------------------------------------------- retry (mengikuti pola src/ragtrial/llm.py)
RETRY_MAX = 5
RETRY_INITIAL_WAIT = 30.0
RETRY_BACKOFF = 1.5
RETRYABLE_MARKERS = (
    "429",
    "resource_exhausted",
    "rate limit",
    "quota",
    "503",
    "unavailable",
    "500",
    "internal error",
    "remoteprotocolerror",
    "server disconnected",
    "connectionerror",
    "connection reset",
    "connection aborted",
    "timed out",
    "timeout",
)


def _dotenv_loaded_once() -> None:
    try:
        from dotenv import load_dotenv

        load_dotenv(ROOT / ".env")
    except ImportError:
        pass


def load_gemini_api_key() -> str:
    """Ambil Gemini API key — dipakai HANYA untuk embedding (Semantic Similarity)."""
    key = os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY")
    if not key:
        _dotenv_loaded_once()
        key = os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY")
    if not key:
        raise SystemExit(
            "GEMINI_API_KEY tidak ditemukan. Set environment variable atau isi "
            f"{ROOT / '.env'}"
        )
    return key


def load_openai_api_key() -> str:
    """Ambil OpenAI API key — dipakai HANYA untuk evaluator Faithfulness.

    Ini API key BARU yang harus disiapkan sendiri (billing OpenAI terpisah dari
    Gemini). Project ini tidak punya key OpenAI apa pun sebelumnya — evaluator
    sengaja dipilih beda vendor dari generator (Gemini) untuk menekan potensi
    self-preference bias.
    """
    key = os.environ.get("OPENAI_API_KEY")
    if not key:
        _dotenv_loaded_once()
        key = os.environ.get("OPENAI_API_KEY")
    if not key:
        raise SystemExit(
            "OPENAI_API_KEY tidak ditemukan. Evaluator Faithfulness sengaja memakai "
            "OpenAI (vendor beda dari generator Gemini) supaya tidak self-judging. "
            f"Isi OPENAI_API_KEY di {ROOT / '.env'} atau environment variable.\n"
            "(Kalau tidak ingin bikin akun/billing OpenAI baru, jalankan dengan "
            "--judge-provider google sebagai alternatif yang tidak butuh key baru — "
            "cek model Gemini yang masih aktif di https://ai.google.dev/gemini-api/docs/models "
            "sebelum memilih --judge-model, karena nama model lama bisa deprecated tanpa "
            "peringatan. Lihat build_judge_llm().)"
        )
    return key


def build_judge_llm(
    model: str = JUDGE_MODEL,
    temperature: float = JUDGE_TEMPERATURE,
    provider: str = "openai",
):
    """Evaluator LLM untuk Faithfulness — prompt sepenuhnya bawaan RAGAS.

    Default: OpenAI asli (`provider="openai"`), vendor beda dari generator
    produksi (Gemini) — pilihan sengaja untuk menekan self-preference bias.

    Alternatif tanpa key baru: `provider="google"` memakai Gemini lewat endpoint
    OpenAI-compatible-nya sendiri (masih GEMINI_API_KEY). Rute ini WAJIB dipakai
    untuk Gemini, bukan `llm_factory(provider="google", client=genai.Client(...))`
    langsung — sudah diverifikasi empiris pada ragas 0.4.3:
      * adapter `google`+`genai.Client` -> instructor, dan ragas sendiri
        memperingatkan bug upstream instructor+google-genai
        (HARM_CATEGORY_JAILBREAK, instructor#1658), menyarankan endpoint
        OpenAI-compatible sebagai gantinya.
      * `genai.Client(...).aio` ditolak: adapter mewajibkan `google.genai.Client`.
      * `Faithfulness.score()` (sync) hanya membungkus `ascore()`, tetap butuh
        client ASYNC.
    """
    from openai import AsyncOpenAI
    from ragas.llms import llm_factory

    if provider == "google":
        client = AsyncOpenAI(
            api_key=load_gemini_api_key(),
            base_url="https://generativelanguage.googleapis.com/v1beta/openai/",
        )
    else:
        client = AsyncOpenAI(api_key=load_openai_api_key())
    return llm_factory(model, provider="openai", client=client, temperature=temperature)


def build_embeddings(model: str = EMBEDDING_MODEL, provider: str = "openai"):
    """Embedding untuk Semantic Similarity.

    Default: OpenAI `text-embedding-3-small` — pilihan paling umum di ekosistem
    RAGAS. Alternatif `provider="google"` (`gemini-embedding-001`) tersedia untuk
    yang tidak ingin memakai key OpenAI sama sekali (lihat build_judge_llm()).
    """
    if provider == "google":
        from google import genai
        from ragas.embeddings import GoogleEmbeddings

        client = genai.Client(api_key=load_gemini_api_key())
        return GoogleEmbeddings(client=client, model=model)

    from openai import AsyncOpenAI
    from ragas.embeddings import OpenAIEmbeddings

    client = AsyncOpenAI(api_key=load_openai_api_key())
    return OpenAIEmbeddings(client=client, model=model)


def provenance(
    *,
    judge_model: str = JUDGE_MODEL,
    judge_provider: str = "openai",
    embedding_model: str = EMBEDDING_MODEL,
    embedding_provider: str = "openai",
    temperature: float = JUDGE_TEMPERATURE,
    testset: str | Path | None = None,
    results_dir: str | Path | None = None,
    metrics: tuple[str, ...] = ALL_METRICS,
) -> Dict[str, Any]:
    """Blok konfigurasi yang ditempelkan ke setiap file summary."""
    try:
        import ragas

        installed = ragas.__version__
    except Exception:  # pragma: no cover - hanya untuk pelaporan
        installed = "unknown"
    return {
        "ragas_version": installed,
        "ragas_version_pinned": RAGAS_VERSION,
        "api": "ragas.metrics.collections (Collections API)",
        "faithfulness": {
            "evaluator_llm": judge_model,
            "evaluator_provider": judge_provider,
            "temperature": temperature,
            "generator_llm": "gemini-2.5-flash (berbeda vendor dari evaluator, by design)",
            "prompts": "bawaan RAGAS (StatementGeneratorPrompt + NLIStatementPrompt)",
            "uses_ground_truth": False,
        },
        "semantic_similarity": {
            "embedding_model": embedding_model,
            "embedding_provider": embedding_provider,
            "retrieval_embedding": "models/gemini-embedding-2 (berbeda dari embedding evaluator, by design)",
            "uses_ground_truth": True,
            "reference_field": "expected_answer",
        },
        "metrics_run": list(metrics),
        "testset": str(testset) if testset else None,
        "results_dir": str(results_dir) if results_dir else None,
        "generated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
    }
