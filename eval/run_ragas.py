"""Fase EVALUASI GENERATION berbasis RAGAS — Faithfulness + Semantic Similarity.

Skrip ini membaca artefak yang sudah dihasilkan fase generation
(`eval/results/main/per_query_<system>.json`) lalu menilainya. Ia TIDAK pernah
menjalankan pipeline RAG: tidak ada import `ragtrial`, tidak menyentuh Chroma,
tidak melakukan retrieval maupun generation ulang.

Jalankan memakai virtualenv evaluator yang terpisah (lihat
eval/ragas_eval/requirements.txt):

    # estimasi biaya dulu, NOL panggilan API
    .venv-ragas\\Scripts\\python.exe eval\\run_ragas.py --dry-run

    # smoke test 5 soal
    .venv-ragas\\Scripts\\python.exe eval\\run_ragas.py \\
        --results-dir eval/results/main_smoke --outdir eval/results/ragas_smoke

    # evaluasi penuh
    .venv-ragas\\Scripts\\python.exe eval\\run_ragas.py --sleep 1 --resume

Output (di --outdir):
    per_query_ragas_<system>.json  — per-question, bisa ditelusuri penuh
    summary_ragas_<system>.json    — agregat + blok konfigurasi
    SUMMARY_ragas.txt              — tabel perbandingan 3 arsitektur
    ragas_cache.json               — cache skor (kunci memuat konteks + model)

Checkpoint atomik ditulis tiap 15 sample yang benar-benar memanggil API, jadi run
yang mati di tengah (Ctrl-C / 429 / laptop sleep) tetap menyimpan progresnya.
"""
from __future__ import annotations

import argparse
import asyncio
import json
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

# Dijalankan lewat path file, jadi `eval/` perlu ditaruh di sys.path agar paket
# `ragas_eval` bisa diimport tanpa ikut mengimpor package `eval` (yang tidak ada
# di venv evaluator).
sys.path.insert(0, str(Path(__file__).resolve().parent))

from ragas_eval import config as cfg  # noqa: E402
from ragas_eval.aggregate import render_table, summarize  # noqa: E402
from ragas_eval.dataset import (  # noqa: E402
    DatasetError,
    SystemDataset,
    estimate_calls,
    load_system_dataset,
    load_testset,
)
from ragas_eval.metrics import RagasScorer  # noqa: E402

CHECKPOINT_EVERY = 15


# ------------------------------------------------------------------ persistence
def load_json(path: Path, default):
    if not path.exists():
        return default
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except (json.JSONDecodeError, OSError):
        return default


def save_json(path: Path, obj) -> None:
    """Tulis atomik (temp+rename) supaya kill di tengah tidak merusak file."""
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)
    tmp.replace(path)


def save_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with open(tmp, "w", encoding="utf-8") as f:
        f.write(text)
    tmp.replace(path)


# ------------------------------------------------------------------ reporting
def print_estimate(
    datasets: List[SystemDataset],
    metrics: tuple,
    judge_model: str = cfg.JUDGE_MODEL,
    judge_provider: str = "openai",
    embedding_model: str = cfg.EMBEDDING_MODEL,
    embedding_provider: str = "openai",
) -> Dict[str, int]:
    est = estimate_calls(datasets, metrics)
    print("\n" + "=" * 70)
    print("ESTIMASI BIAYA (sebelum satu pun panggilan API)")
    print("=" * 70)
    for ds in datasets:
        missing = f"  [!] {len(ds.missing_ids)} soal belum ada hasil generation" if ds.missing_ids else ""
        print(f"  {ds.system:<12} coverage {ds.n_matched}/{ds.n_testset}{missing}")
    print("-" * 70)
    print(f"  total sample                : {est['samples_total']}")
    print(f"  sample faithfulness         : {est['faithfulness_samples']}")
    print(f"  sample semantic_similarity  : {est['semantic_similarity_samples']}")
    print(f"  ESTIMASI LLM CALL           : {est['llm_calls']}"
          f"  ({cfg.LLM_CALLS_PER_FAITHFULNESS} per sample faithfulness:"
          f" claim extraction + NLI verification)")
    print(f"  ESTIMASI EMBEDDING CALL     : {est['embedding_calls']}"
          f"  ({cfg.EMBED_CALLS_PER_SEMANTIC_SIMILARITY} per sample: reference + response)")
    print(f"  evaluator LLM               : {judge_model} via {judge_provider}"
          f" (temperature {cfg.JUDGE_TEMPERATURE})")
    print(f"  embedding                   : {embedding_model} via {embedding_provider}")
    print("=" * 70 + "\n")
    return est


# ------------------------------------------------------------------ core loop
async def run_system(
    ds: SystemDataset,
    scorer: RagasScorer,
    cache: Dict[str, Any],
    outdir: Path,
    cache_path: Path,
    metrics: tuple,
    sleep: float,
    resume: bool,
) -> List[Dict[str, Any]]:
    outpath = outdir / f"per_query_ragas_{ds.system}.json"

    by_id: Dict[str, Dict[str, Any]] = {}
    if resume:
        prior = load_json(outpath, [])
        for rec in prior:
            if rec.get("status") in ("ok", "partial"):
                by_id[rec["question_id"]] = rec
        if by_id:
            print(f"  [resume] {len(by_id)} record sudah dinilai, dilewati.")

    def flush() -> None:
        ordered = [by_id[s.question_id] for s in ds.samples if s.question_id in by_id]
        save_json(outpath, ordered)
        save_json(cache_path, cache)

    since_ckpt = 0
    total = len(ds.samples)
    for i, sample in enumerate(ds.samples, 1):
        if resume and sample.question_id in by_id:
            continue
        calls_before = scorer.n_llm_calls + scorer.n_embed_calls
        record = await scorer.score_sample(sample, cache, metrics)
        by_id[sample.question_id] = record
        made_call = (scorer.n_llm_calls + scorer.n_embed_calls) > calls_before

        flag = "" if record["status"] == "ok" else f"  <{record['status']}>"
        print(
            f"  [{i}/{total}] {sample.question_id:<8} "
            f"faith={_p(record.get('faithfulness'))} "
            f"sim={_p(record.get('semantic_similarity'))}{flag}"
        )
        if record["status"] != "ok":
            for metric, reason in (record.get("errors") or {}).items():
                print(f"           ! {metric}: {reason}")

        if made_call:
            since_ckpt += 1
            if since_ckpt >= CHECKPOINT_EVERY:
                flush()
                since_ckpt = 0
            if sleep:
                await asyncio.sleep(sleep)

    flush()
    return [by_id[s.question_id] for s in ds.samples if s.question_id in by_id]


def _p(v: Optional[float]) -> str:
    return " n/a " if v is None else f"{v:.3f}"


def _resolve_models(args) -> Optional[str]:
    """Isi default model per-provider; tolak kombinasi ambigu. Return pesan error, atau None kalau OK."""
    if args.judge_model is None:
        if args.judge_provider == "google":
            return (
                "--judge-model wajib diisi manual kalau --judge-provider google "
                "(nama model Gemini sering deprecated tanpa peringatan, tidak ada "
                "default yang aman — cek model aktif di "
                "https://ai.google.dev/gemini-api/docs/models)."
            )
        args.judge_model = cfg.JUDGE_MODEL
    if args.embedding_model is None:
        args.embedding_model = (
            cfg.GEMINI_EMBEDDING_MODEL if args.embedding_provider == "google" else cfg.EMBEDDING_MODEL
        )
    return None


async def amain(args) -> int:
    if (err := _resolve_models(args)) is not None:
        print(f"\nERROR: {err}\n", file=sys.stderr)
        return 2

    metrics = tuple(args.metrics)
    testset_path = Path(args.testset)
    results_dir = Path(args.results_dir)
    outdir = Path(args.outdir)
    ids = set(args.ids) if args.ids else None

    try:
        testset = load_testset(testset_path)
        datasets = [
            load_system_dataset(testset, results_dir, system, ids=ids, limit=args.limit)
            for system in args.systems
        ]
    except DatasetError as e:
        print(f"\nERROR: {e}\n", file=sys.stderr)
        return 2

    print(f"testset     : {testset_path}  ({len(testset)} soal)")
    print(f"results-dir : {results_dir}")
    print(f"outdir      : {outdir}")
    print(f"systems     : {', '.join(args.systems)}")
    print(f"metrics     : {', '.join(metrics)}")

    print_estimate(datasets, metrics, args.judge_model, args.judge_provider,
                   args.embedding_model, args.embedding_provider)
    if args.dry_run:
        print("--dry-run: berhenti di sini, tidak ada panggilan API.")
        return 0

    cache_path = outdir / cfg.CACHE_FILENAME
    cache: Dict[str, Any] = load_json(cache_path, {})
    print(f"cache       : {len(cache)} entri dimuat dari {cache_path.name}\n")

    scorer = RagasScorer(metrics=metrics, judge_model=args.judge_model,
                         judge_provider=args.judge_provider,
                         embedding_model=args.embedding_model,
                         embedding_provider=args.embedding_provider)

    summaries: List[Dict[str, Any]] = []
    prov = cfg.provenance(
        judge_model=scorer.judge_model,
        judge_provider=scorer.judge_provider,
        embedding_model=scorer.embedding_model,
        embedding_provider=scorer.embedding_provider,
        testset=testset_path,
        results_dir=results_dir,
        metrics=metrics,
    )

    t0 = time.perf_counter()
    for ds in datasets:
        print(f"=== {ds.system} ({len(ds.samples)} sample) ===")
        records = await run_system(
            ds, scorer, cache, outdir, cache_path, metrics, args.sleep, args.resume
        )
        summary = summarize(
            records, ds.system, n_testset=ds.n_testset,
            missing_ids=ds.missing_ids, metrics=metrics,
        )
        summary["config"] = prov
        save_json(outdir / f"summary_ragas_{ds.system}.json", summary)
        summaries.append(summary)
        print(f"  -> {outdir / f'per_query_ragas_{ds.system}.json'}\n")

    save_json(cache_path, cache)
    table = render_table(summaries)
    save_text(outdir / "SUMMARY_ragas.txt", table)
    print(table)
    print(f"\nLLM call terpakai      : {scorer.n_llm_calls}")
    print(f"Embedding call terpakai: {scorer.n_embed_calls}")
    print(f"Durasi                 : {time.perf_counter() - t0:.1f}s")
    print(f"Tersimpan di           : {outdir}")

    n_failed = sum(s["status_counts"]["failed"] for s in summaries)
    return 1 if n_failed else 0


def main() -> int:
    ap = argparse.ArgumentParser(
        description="Evaluasi generation dengan RAGAS (Faithfulness + Semantic Similarity)."
    )
    ap.add_argument("--testset", default=str(cfg.DEFAULT_TESTSET),
                    help="sumber expected_answer (reference)")
    ap.add_argument("--results-dir", default=str(cfg.DEFAULT_RESULTS_DIR),
                    help="folder berisi per_query_<system>.json hasil fase generation")
    ap.add_argument("--outdir", default=str(cfg.DEFAULT_OUTDIR))
    ap.add_argument("--systems", nargs="+", default=list(cfg.DEFAULT_SYSTEMS))
    ap.add_argument("--metrics", nargs="+", default=list(cfg.ALL_METRICS),
                    choices=list(cfg.ALL_METRICS))
    ap.add_argument("--ids", nargs="+", default=None, help="nilai hanya id tertentu")
    ap.add_argument("--limit", type=int, default=None, help="ambil N sample pertama")
    ap.add_argument("--sleep", type=float, default=0.0,
                    help="jeda detik setelah tiap sample yang memanggil API (throttle 429)")
    ap.add_argument("--resume", action="store_true",
                    help="lewati record yang sudah dinilai di per_query_ragas_<system>.json")
    ap.add_argument("--dry-run", action="store_true",
                    help="cetak coverage + estimasi biaya lalu berhenti; nol panggilan API")
    ap.add_argument("--judge-model", default=None,
                    help=f"default {cfg.JUDGE_MODEL} kalau --judge-provider openai; "
                         "WAJIB diisi manual kalau --judge-provider google (nama model "
                         "Gemini sering berubah/deprecated, tidak ada default aman)")
    ap.add_argument("--judge-provider", default="openai", choices=["openai", "google"],
                    help="'openai' (default, butuh OPENAI_API_KEY, vendor beda dari "
                         "generator Gemini) atau 'google' (Gemini lewat endpoint "
                         "OpenAI-compatible, pakai GEMINI_API_KEY yang sudah ada)")
    ap.add_argument("--embedding-model", default=None,
                    help=f"default {cfg.EMBEDDING_MODEL} kalau --embedding-provider openai, "
                         f"{cfg.GEMINI_EMBEDDING_MODEL} kalau --embedding-provider google")
    ap.add_argument("--embedding-provider", default="openai", choices=["openai", "google"],
                    help="'openai' (default, butuh OPENAI_API_KEY, reuse key judge) atau "
                         "'google' (gemini-embedding-001, pakai GEMINI_API_KEY yang sudah ada)")
    args = ap.parse_args()
    return asyncio.run(amain(args))


if __name__ == "__main__":
    raise SystemExit(main())
