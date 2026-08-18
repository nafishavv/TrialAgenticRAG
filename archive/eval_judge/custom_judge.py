"""ARSIP — custom LLM-as-a-Judge yang dipakai RAGTrial sebelum migrasi ke RAGAS.

STATUS: TIDAK DIPAKAI LAGI. Dipensiunkan pada 18 Agustus 2026, digantikan
evaluasi generation berbasis RAGAS (lihat eval/run_ragas.py dan
docs/EVAL_GENERATION_RAGAS.md). File ini tidak diimport dari mana pun; ia
disimpan sebagai catatan sejarah metodologi, bukan kode hidup.

KENAPA DIPENSIUNKAN
    Instrumen ini adalah judge holistik buatan sendiri: satu panggilan LLM per
    metric, prompt Bahasa Indonesia yang ditulis manual, skala kasar 0/1/2 yang
    lalu dibagi 2. Dua kelemahannya sudah tercatat sendiri di docs/EVAL_REPORT.md
    §11 item 5:

      1. Tidak pernah divalidasi siapa pun — validitasnya sepenuhnya bersandar
         pada penilaian penulis sendiri, sehingga sulit dipertahankan sebagai
         alat ukur dalam skripsi.
      2. Holistik, bukan statement-level. Jawaban dengan 5 klaim yang 4 di
         antaranya didukung konteks hanya bisa mendarat di 0, 0.5, atau 1.0.

    RAGAS Faithfulness menggantikannya dengan pendekatan statement-level
    (pecah jawaban jadi klaim atomik -> verifikasi entailment tiap klaim ->
    skor = proporsi klaim yang didukung), memakai prompt bawaan yang sudah
    published.

APA YANG HILANG DAN TIDAK DIGANTIKAN
    `judge_answer_relevance` dan `judge_refusal` TIDAK punya padanan di dua
    metric RAGAS yang dipilih (Faithfulness + Semantic Similarity). Angkanya di
    docs/EVAL_REPORT.md §6 karena itu bersifat historis — dihasilkan instrumen
    lama pada candidate testset (n=202), dan tidak dilanjutkan.

    Catatan yang meringankan: eval/main_testset.json (test set final, n=115)
    sama sekali tidak memuat soal dengan expected_route == "none", sehingga
    cabang `refusal_correct` memang tidak akan pernah aktif di sana.

    `judge_fact_recall` sudah lebih dulu dibuang pada commit 8d5e126 karena
    sirkular: expected_facts dihasilkan LLM, lalu dinilai lagi oleh LLM.

CARA MENJALANKAN ULANG (kalau suatu saat perlu reproduksi angka lama)
    Kode di bawah bergantung pada `ragtrial.llm.make_judge_llm` dan
    `invoke_with_retry`, dan dirancang untuk .venv utama — BUKAN .venv-ragas.
    Pasangannya adalah run_judge.py di folder yang sama, plus judge_cache.json
    (cache skor: "<id>|<system>|<judge_name>|<sha1(answer)>" -> skor).
"""
from __future__ import annotations

import re
from typing import Any, Dict, List

from ragtrial.llm import invoke_with_retry, make_judge_llm

# temperature=0 dipakai untuk konsistensi antar-run.
_judge_llm = make_judge_llm(temperature=0.0, max_tokens=256)


JUDGE_FACT_PROMPT = """Kamu evaluator. Cek apakah FAKTA berikut benar-benar terkandung dalam JAWABAN (boleh berbeda kata, asal makna sama dan tidak bertentangan).

FAKTA: {fact}

JAWABAN:
{answer}

ATURAN OUTPUT:
- Jawab HANYA "YES" atau "NO" (tanpa penjelasan).
- "YES" kalau fakta tercakup di jawaban (eksplisit atau paraphrased).
- "NO" kalau fakta tidak ada / kontradiktif / tidak dapat diverifikasi dari jawaban.
"""

JUDGE_FAITHFULNESS_PROMPT = """Kamu evaluator faithfulness. Periksa apakah JAWABAN hanya mengandung klaim yang didukung oleh KONTEKS. Jawaban tidak boleh menambah info di luar konteks.

KONTEKS:
{context}

JAWABAN:
{answer}

ATURAN OUTPUT:
- Skor 0-2 saja:
  - "2" = semua klaim didukung konteks (faithful)
  - "1" = sebagian klaim didukung, sebagian tidak ada di konteks (mixed)
  - "0" = ada klaim yang bertentangan dengan konteks atau jelas halusinasi
- Jawab HANYA angka 0/1/2 (tanpa penjelasan).
"""

JUDGE_REFUSAL_PROMPT = """Kamu evaluator. Periksa apakah JAWABAN berisi refusal/pernyataan bahwa info tidak tersedia atau pertanyaan di luar cakupan.

JAWABAN:
{answer}

ATURAN OUTPUT:
- Jawab "YES" kalau jawaban benar-benar nolak / bilang "tidak ditemukan" / "di luar cakupan" / sejenis.
- Jawab "NO" kalau jawaban tetap mencoba memberi info substantif.
"""

JUDGE_ANSWER_RELEVANCE_PROMPT = """Kamu evaluator relevansi. Periksa apakah JAWABAN menjawab PERTANYAAN secara langsung dan substansial.

PERTANYAAN: {question}

JAWABAN: {answer}

ATURAN OUTPUT:
- Skor 0-2:
  - "2" = jawaban langsung menjawab pertanyaan dengan informasi substansial.
  - "1" = jawaban menyentuh topik tapi kurang fokus / kurang lengkap.
  - "0" = jawaban tidak nyambung / off-topic / hanya menolak padahal info ada.
- Jawab HANYA angka 0/1/2.
"""


def _judge_invoke(prompt: str) -> str:
    """Single LLM call, returns stripped text."""
    return invoke_with_retry(_judge_llm, prompt).content.strip()


def judge_fact_recall(answer: str, expected_facts: List[str]) -> Dict[str, Any]:
    """DIBUANG lebih dulu (commit 8d5e126) — sirkular: kunci LLM dinilai LLM.

    Returns:
      {"fact_recall": 0.83, "per_fact": [{"fact": ..., "present": True}, ...]}
    """
    real_facts = [f for f in expected_facts if not f.startswith("TODO_")]
    if not real_facts:
        return {"fact_recall": float("nan"), "per_fact": [], "skipped": True}

    per_fact = []
    hits = 0
    for fact in real_facts:
        prompt = JUDGE_FACT_PROMPT.format(fact=fact, answer=answer)
        raw = _judge_invoke(prompt).upper()
        present = "YES" in raw
        per_fact.append({"fact": fact, "present": present, "raw": raw})
        if present:
            hits += 1
    return {
        "fact_recall": hits / len(real_facts),
        "per_fact": per_fact,
        "skipped": False,
    }


def judge_faithfulness(answer: str, context: str) -> Dict[str, Any]:
    """DIGANTI oleh ragas.metrics.collections.Faithfulness.

    LLM-judged 0/1/2 score, dinormalkan ke 0.0/0.5/1.0.
    """
    if not context.strip():
        return {"faithfulness": float("nan"), "raw": "no_context"}
    prompt = JUDGE_FAITHFULNESS_PROMPT.format(context=context, answer=answer)
    raw = _judge_invoke(prompt)
    m = re.search(r"\b([012])\b", raw)
    score = int(m.group(1)) / 2.0 if m else float("nan")
    return {"faithfulness": score, "raw": raw}


def judge_refusal(answer: str) -> Dict[str, Any]:
    """TIDAK ADA PENGGANTI di suite RAGAS yang dipilih.

    Boolean: did the answer refuse / say info not found?
    """
    prompt = JUDGE_REFUSAL_PROMPT.format(answer=answer)
    raw = _judge_invoke(prompt).upper()
    refused = "YES" in raw
    return {"refused": refused, "raw": raw}


def judge_answer_relevance(question: str, answer: str) -> Dict[str, Any]:
    """TIDAK ADA PENGGANTI di suite RAGAS yang dipilih."""
    prompt = JUDGE_ANSWER_RELEVANCE_PROMPT.format(question=question, answer=answer)
    raw = _judge_invoke(prompt)
    m = re.search(r"\b([012])\b", raw)
    score = int(m.group(1)) / 2.0 if m else float("nan")
    return {"answer_relevance": score, "raw": raw}


def ragas_eval_single(
    question: str,
    answer: str,
    contexts: List[str],
    ground_truth: str | None = None,
) -> Dict[str, float]:
    """ARSIP GANDA — stub RAGAS lama yang tidak pernah jalan.

    Memakai legacy API (`from ragas.metrics import faithfulness`) yang sudah
    di-deprecate di ragas 0.4 dan akan dihapus di 1.0, dan mem-pass
    ChatGoogleGenerativeAI mentah sebagai `llm=` — yang tidak diterima RAGAS
    modern. `ragas` juga tidak pernah terpasang di venv utama, jadi fungsi ini
    selalu mengembalikan {"_ragas_status": "not_installed"}.

    Penggantinya: eval/run_ragas.py + eval/ragas_eval/, memakai Collections API.
    """
    try:
        from ragas import evaluate
        from ragas.metrics import faithfulness, answer_relevancy
        from datasets import Dataset
    except ImportError:
        return {"_ragas_status": "not_installed"}

    try:
        ds = Dataset.from_dict({
            "question": [question],
            "answer": [answer],
            "contexts": [contexts],
            "ground_truth": [ground_truth or ""],
        })
        result = evaluate(
            ds,
            metrics=[faithfulness, answer_relevancy],
            llm=_judge_llm,
        )
        return {k: float(v) for k, v in result.to_pandas().iloc[0].to_dict().items()
                if isinstance(v, (int, float))}
    except Exception as e:
        return {"_ragas_status": f"error: {e}"}
