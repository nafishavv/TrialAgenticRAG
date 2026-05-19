"""Eval utilities — metrics, judges, gold ID normalization.

Designed to be system-agnostic: works on dict output of `ask_agentic(...)` (with
route + documents + answer) and `ask_main(...)` (without route).

Gold ID format:
    dukcapil  → "page:<page_start>"
    opd       → "nomor:<nomor>"

Judges (LLM-as-judge) menggunakan Gemini dengan temperature=0 untuk konsistensi.
"""
from __future__ import annotations

import re
from typing import Any, Dict, List, Optional

from langchain_core.documents import Document

from ragtrial.llm import make_judge_llm

# ---------- LLM judge setup ----------
_judge_llm = make_judge_llm(temperature=0.0, max_tokens=256)


# ============================================================
# 1. GOLD ID NORMALIZATION
# ============================================================
def doc_to_gold_id(doc: Document) -> Optional[str]:
    """Convert a retrieved Document into a normalized gold-ID string.

    Returns e.g. "dukcapil:page:40" or "opd:nomor:1.a".
    Returns None if doc cannot be normalized (e.g. narrative dukcapil chunk
    without page metadata, or unknown source).
    """
    m = doc.metadata or {}
    # OPD: ada metadata 'nomor'
    if "nomor" in m and m.get("doc_type") == "opd_directory":
        return f"opd:nomor:{m['nomor']}"
    # Dukcapil Q&A: ada page_start
    if "page_start" in m:
        return f"dukcapil:page:{m['page_start']}"
    # Dukcapil narrative chunk fallback: pakai 'page'
    if m.get("section") and "page" in m:
        return f"dukcapil:page:{m['page']}"
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


def mrr(retrieved_ids: List[str], gold_ids: set[str], k: int = 10) -> float:
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
def routing_eval(predictions: List[str], gold: List[str]) -> Dict[str, Any]:
    """Compute confusion matrix + per-class P/R/F1 + overall accuracy.

    predictions[i], gold[i] in {"dukcapil","opd","both","none"}.
    """
    labels = ["dukcapil", "opd", "both", "none"]
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


# ============================================================
# 4. LLM-AS-JUDGE
# ============================================================
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
    return _judge_llm.invoke(prompt).content.strip()


def judge_fact_recall(answer: str, expected_facts: List[str]) -> Dict[str, Any]:
    """For each expected fact, ask judge if it's present in answer.

    Returns:
      {"fact_recall": 0.83, "per_fact": [{"fact": ..., "present": True}, ...]}
    """
    # Skip TODO placeholders
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
    """LLM-judged 0/1/2 score. Returns dict with score & raw."""
    if not context.strip():
        return {"faithfulness": float("nan"), "raw": "no_context"}
    prompt = JUDGE_FAITHFULNESS_PROMPT.format(context=context, answer=answer)
    raw = _judge_invoke(prompt)
    m = re.search(r"\b([012])\b", raw)
    score = int(m.group(1)) / 2.0 if m else float("nan")
    return {"faithfulness": score, "raw": raw}


def judge_refusal(answer: str) -> Dict[str, Any]:
    """Boolean: did the answer refuse / say info not found?"""
    prompt = JUDGE_REFUSAL_PROMPT.format(answer=answer)
    raw = _judge_invoke(prompt).upper()
    refused = "YES" in raw
    return {"refused": refused, "raw": raw}


def judge_answer_relevance(question: str, answer: str) -> Dict[str, Any]:
    prompt = JUDGE_ANSWER_RELEVANCE_PROMPT.format(question=question, answer=answer)
    raw = _judge_invoke(prompt)
    m = re.search(r"\b([012])\b", raw)
    score = int(m.group(1)) / 2.0 if m else float("nan")
    return {"answer_relevance": score, "raw": raw}


# ============================================================
# 5. OPTIONAL RAGAS WRAPPER
# ============================================================
def ragas_eval_single(
    question: str,
    answer: str,
    contexts: List[str],
    ground_truth: str | None = None,
) -> Dict[str, float]:
    """Optional: run RAGAS metrics (faithfulness, answer_relevancy) for ONE sample.

    Returns empty dict on import error or RAGAS failure (so eval pipeline tetap jalan).
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


# ============================================================
# 6. STORE-CORRECTNESS (agentic only)
# ============================================================
def store_correct(expected_route: str, source_used: str) -> Optional[bool]:
    """Did agentic query the correct store(s)?
    Returns None if expected_route='none' (no store should be queried).
    """
    if expected_route == "none":
        return source_used == "none" or source_used == ""
    return source_used == expected_route


# ============================================================
# 7. AGGREGATION HELPERS
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
