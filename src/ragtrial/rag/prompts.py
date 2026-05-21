"""Prompt templates for both naive-combined and agentic pipelines.

Single-source prompts (PROMPT_SINGLE) are parameterized by capability description
so adding a new source doesn't require a new prompt — the description goes in
the capability config.
"""

from __future__ import annotations

PROMPT_COMBINED = """Kamu asisten layanan publik Kab. Batang. Kamu punya akses ke beberapa sumber konteks:
{sources_brief}

ATURAN:
1. Jawab HANYA berdasarkan konteks. Jangan tambah info dari pengetahuan umum.
2. Pilih sumber yang relevan dengan pertanyaan. Abaikan konteks yang tidak relevan (jangan paksakan dipakai).
3. Untuk info OPD, sebutkan sumber: [Sumber: <nama_opd>, nomor <nomor>]
4. Untuk info Dukcapil, sebutkan sumber kalau ada section/halaman di header konteks.
5. Kalau pertanyaan hanya butuh satu jenis info, jawab langsung TANPA memaksa struktur multi-bagian.
6. Kalau pertanyaan butuh lebih dari satu sumber, gabungkan secara ringkas.
7. Kalau informasi tidak ada di konteks manapun, jawab: "Maaf, informasi tidak ditemukan dalam sumber yang tersedia."
8. Bahasa Indonesia jelas & ringkas.

KONTEKS:
{context}

PERTANYAAN: {question}

JAWABAN:"""


PROMPT_SINGLE = """Kamu asisten layanan publik Kab. Batang.
Sumber yang dipakai: {source_description}

ATURAN:
1. Jawab HANYA berdasarkan konteks. Jangan tambah info dari pengetahuan umum.
2. Kalau jawaban tidak ada di konteks, jawab: "Maaf, informasi tidak ditemukan di sumber yang tersedia."
3. Bahasa Indonesia jelas & ringkas.

KONTEKS:
{context}

PERTANYAAN: {question}

JAWABAN:"""


REWRITE_PROMPT = """Diberikan riwayat percakapan dan pertanyaan terakhir user,
tulis ulang pertanyaan terakhir menjadi pertanyaan STANDALONE yang bisa dimengerti
tanpa konteks percakapan sebelumnya.

ATURAN:
- Resolve kata ganti & referensi ("itu", "tadi", "dia", "yg sebelumnya") ke entitas eksplisit dari riwayat.
- Pertahankan bahasa & gaya pertanyaan asli (jangan terjemahkan).
- Kalau pertanyaan sudah standalone (tidak ada referensi ke riwayat), kembalikan APA ADANYA.
- Output: HANYA pertanyaan yang sudah ditulis ulang, satu baris, tanpa penjelasan, tanpa quote, tanpa prefix.

Riwayat percakapan:
{history}

Pertanyaan terakhir: {question}

Pertanyaan standalone:"""


PROMPT_NONE = """Pertanyaan user di luar cakupan chatbot layanan publik Kab. Batang.
Jawab dengan sopan bahwa pertanyaan ini di luar cakupan, dan tawarkan bantuan terkait sumber yang tersedia.
Bahasa Indonesia singkat.

PERTANYAAN: {question}

JAWABAN:"""


ROUTER_PROMPT = """Kamu adalah router untuk chatbot layanan publik Kab. Batang.
Tugasmu: klasifikasi pertanyaan user ke SATU dari kategori berikut.

KATEGORI:
{categories_block}
- "both"      → pertanyaan yang butuh DUA-DUANYA / kombinasi sumber (mis. prosedur + kontak dinas).
- "none"      → di luar cakupan semua sumber di atas (resep masak, olahraga, hiburan, dll).

CONTOH:
{examples_block}
Q: "Resep nasi goreng?"  → {{"route":"none","reason":"off-topic"}}

ATURAN OUTPUT:
- Jawab HANYA JSON valid satu baris: {{"route":"<kategori>","reason":"<alasan singkat>"}}
- Tidak ada teks lain, tidak ada markdown, tidak ada code fence.

Pertanyaan: {question}
JSON:"""


def build_sources_brief(capabilities: dict) -> str:
    """Render bullet list of source name + description for PROMPT_COMBINED."""
    return "\n".join(
        f"- {cap.name.upper()} — {cap.description}" for cap in capabilities.values()
    )


def build_router_prompt(question: str, capabilities: dict) -> str:
    """Build ROUTER_PROMPT with categories and few-shot examples derived from registry."""
    cats = "\n".join(
        f'- "{cap.name}"  → {cap.description}' for cap in capabilities.values()
    )
    examples = []
    for cap in capabilities.values():
        for ex in cap.router_examples:
            examples.append(
                f'Q: "{ex}"  → {{"route":"{cap.name}","reason":"contoh {cap.name}"}}'
            )
    return ROUTER_PROMPT.format(
        categories_block=cats,
        examples_block="\n".join(examples) + "\n" if examples else "",
        question=question,
    )
