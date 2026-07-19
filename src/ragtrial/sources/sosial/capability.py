"""Sosial capability — produk hukum sektor sosial JDIH Kab. Batang."""

from __future__ import annotations

from langchain_core.documents import Document

from ragtrial.capabilities.vector_source import VectorSourceCapability
from ragtrial.config import SOSIAL_VECTOR_STORE
from ragtrial.sources.sosial.chunk import gold_id_suffix


def _header(doc: Document, idx: int) -> str:
    m = doc.metadata or {}
    tipe = m.get("tipe_dokumen", "?")
    nomor = m.get("nomor", "")
    tahun = m.get("tahun", "")
    no_thn = f" No.{nomor}/{tahun}" if nomor else f"/{tahun}"
    judul = (m.get("judul") or "")[:60]
    pasal = f", Pasal {m['pasal_number']}" if m.get("pasal_number") else ""
    page = m.get("page_start", "?")
    status = m.get("status") or "?"
    return f"[Sosial {idx}: {tipe}{no_thn} — {judul}{pasal}, hal {page}, status {status}]"


def _gold_id(doc: Document) -> str | None:
    m = doc.metadata or {}
    if m.get("id") and m.get("doc_type") == "sosial":
        return f"sosial:id:{m['id']}{gold_id_suffix(m)}"
    return None


def _label(doc: Document) -> str:
    m = doc.metadata or {}
    tipe = m.get("tipe_dokumen") or "Produk Hukum"
    nomor = m.get("nomor")
    tahun = m.get("tahun")
    no_thn = f" No. {nomor} Tahun {tahun}" if nomor and tahun else ""
    return f"JDIH Kab. Batang — {tipe}{no_thn}"


sosial_capability = VectorSourceCapability(
    name="sosial",
    description=(
        "Produk hukum sektor sosial Kab. Batang (JDIH): Peraturan Daerah/Bupati, "
        "Keputusan & Instruksi Bupati seputar perlindungan anak, kesejahteraan & "
        "bantuan sosial, perlindungan perempuan, jaminan sosial, cadangan pangan, "
        "serta abstrak/artikel hukum terkait. Berisi pasal, dasar hukum, dan status "
        "berlaku/dicabutnya peraturan."
    ),
    collection_name="sosial",
    persist_directory=SOSIAL_VECTOR_STORE,
    router_examples=[
        "Bagaimana aturan perlindungan anak di Kabupaten Batang?",
        "Apa dasar hukum bantuan sosial bagi warga miskin?",
        "Peraturan tentang jaminan sosial ketenagakerjaan pekerja rentan?",
        "Aturan penyelenggaraan cadangan pangan daerah?",
        "Perda tentang kesejahteraan sosial nomor berapa?",
    ],
    strategy="hybrid",
    header_formatter=_header,
    gold_id_fn=_gold_id,
    citation="sebutkan jenis/nomor/tahun peraturan dan Pasal kalau ada di header konteks; "
             "kalau status 'Tidak Berlaku', ingatkan bahwa peraturan sudah dicabut.",
    label_formatter=_label,
)
