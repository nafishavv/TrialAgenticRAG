"""LLM prompt skeletons + strict-JSON parsing.

The LLM only writes the natural-language fields (question / answer / facts /
difficulty / notes). Route, gold_chunks, query_type and chunk_scope are assigned
deterministically by the generators — never trusted from the model output.
"""

from __future__ import annotations

import json
from typing import Any, Dict

from langchain_core.documents import Document

_QTYPE_GUIDE = {
    "lexical_exact": "pakai istilah yang persis ada di KONTEKS.",
    "paraphrase": "tanyakan hal yang sama TANPA menyalin frasa >=6 kata dari KONTEKS.",
    "semantic": "gunakan sinonim / maksud sama dengan kata yang berbeda dari KONTEKS.",
    "analytical": "minta penalaran/perbandingan/ringkasan dari isi KONTEKS (bukan kutip mentah).",
    "negation_edge": "soal batas/larangan/pengecualian/status (mis. masih berlaku atau dicabut).",
}

_JSON_RULES = (
    'Keluarkan HANYA satu objek JSON valid (tanpa teks lain, tanpa code fence):\n'
    '{"question":"...","difficulty":"easy|medium|hard",'
    '"expected_answer":"1-3 kalimat ringkas, hanya dari KONTEKS",'
    '"expected_facts":["fakta atomik 1","fakta atomik 2"],'
    '"notes":"1 kalimat: apa yang diuji soal ini"}\n'
    "ATURAN: 2-3 fakta atomik yang BISA diverifikasi dari KONTEKS; "
    "pertanyaan harus terjawab HANYA dari KONTEKS; Bahasa Indonesia."
)


def _meta_line(doc: Document) -> str:
    m = doc.metadata or {}
    bits = []
    for key in ("tipe_dokumen", "nomor", "tahun", "status", "section", "pasal_number",
                "judul", "nama_opd", "tipe", "nama_perizinan", "kategori"):
        v = m.get(key)
        if v not in (None, "", "?"):
            bits.append(f"{key}={v}")
    return ", ".join(bits) if bits else "(tidak ada metadata penting)"


def single_prompt(domain: str, doc: Document, query_type: str) -> str:
    guide = _QTYPE_GUIDE.get(query_type, _QTYPE_GUIDE["paraphrase"])
    return (
        "Kamu membuat 1 soal evaluasi sistem tanya-jawab (RAG) dari potongan dokumen "
        f"resmi Pemkab Batang. JANGAN mengarang fakta di luar KONTEKS.\n"
        f"DOMAIN: {domain}\n"
        f"METADATA: {_meta_line(doc)}\n"
        "KONTEKS:\n"
        f"{(doc.page_content or '').strip()}\n\n"
        f"TIPE SOAL yang diminta: {query_type} — {guide}\n"
        f"{_JSON_RULES}"
    )


def multi_prompt(domain: str, docs: list[Document]) -> str:
    blocks = []
    for i, d in enumerate(docs, 1):
        blocks.append(f"KONTEKS {i} [{_meta_line(d)}]:\n{(d.page_content or '').strip()}")
    body = "\n\n".join(blocks)
    return (
        "Kamu membuat 1 soal evaluasi RAG yang HANYA bisa dijawab bila SEMUA potongan "
        f"berikut digunakan (tiap potongan menyumbang minimal 1 fakta). DOMAIN: {domain}. "
        "JANGAN mengarang fakta di luar KONTEKS.\n"
        f"{body}\n\n"
        "TIPE SOAL: multi_chunk — pertanyaan menggabungkan info dari semua potongan.\n"
        f"{_JSON_RULES}"
    )


def cross_prompt(doc_a: Document, domain_a: str, doc_b: Document, domain_b: str) -> str:
    return (
        "Kamu membuat 1 soal evaluasi RAG lintas-topik yang butuh DUA sumber berbeda "
        "(satu pertanyaan natural yang menggabungkan keduanya). JANGAN mengarang.\n"
        f"SUMBER A (domain {domain_a}) [{_meta_line(doc_a)}]:\n{(doc_a.page_content or '').strip()}\n\n"
        f"SUMBER B (domain {domain_b}) [{_meta_line(doc_b)}]:\n{(doc_b.page_content or '').strip()}\n\n"
        "TIPE SOAL: cross_store — pertanyaan yang masuk akal ditanyakan warga, butuh A dan B.\n"
        f"{_JSON_RULES}"
    )


def none_prompt(seed_examples: list[str]) -> str:
    seeds = "; ".join(seed_examples)
    return (
        "Buat 1 pertanyaan yang BERADA DI LUAR CAKUPAN layanan Pemkab Batang "
        "(kependudukan, OPD/direktori dinas, perizinan, produk hukum sosial). "
        "Pertanyaan harus terdengar wajar tapi TIDAK bisa dijawab dari data pemkab "
        "(mis. topik nasional, hiburan, cuaca, resep, olahraga, atau sekadar sapaan).\n"
        f"Contoh gaya: {seeds}\n"
        'Keluarkan HANYA JSON valid: {"question":"...","notes":"alasan singkat kenapa out-of-scope"}'
    )


class LLMOutputError(ValueError):
    pass


def parse_record(raw: str) -> Dict[str, Any]:
    """Extract the first balanced {...} object and json.loads it (tolerates fences)."""
    if not raw:
        raise LLMOutputError("empty LLM output")
    # strip code fences (```json ... ```) the model sometimes wraps output in
    raw = raw.replace("```json", "").replace("```JSON", "").replace("```", "")
    start = raw.find("{")
    if start == -1:
        raise LLMOutputError(f"no JSON object in output: {raw[:120]!r}")
    depth = 0
    in_str = False
    esc = False
    for i in range(start, len(raw)):
        ch = raw[i]
        if in_str:
            if esc:
                esc = False
            elif ch == "\\":
                esc = True
            elif ch == '"':
                in_str = False
            continue
        if ch == '"':
            in_str = True
        elif ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0:
                blob = raw[start:i + 1]
                try:
                    return json.loads(blob)
                except json.JSONDecodeError as e:
                    raise LLMOutputError(f"bad JSON: {e}: {blob[:120]!r}") from e
    raise LLMOutputError(f"unbalanced JSON braces: {raw[:120]!r}")
