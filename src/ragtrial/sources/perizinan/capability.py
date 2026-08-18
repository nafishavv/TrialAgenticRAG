"""Perizinan capability — izin/perizinan layanan SIPUAS Kab. Batang."""

from __future__ import annotations

from langchain_core.documents import Document

from ragtrial.capabilities.vector_source import VectorSourceCapability
from ragtrial.config import PERIZINAN_VECTOR_STORE


def _header(doc: Document, idx: int) -> str:
    m = doc.metadata or {}
    kategori = f", kategori {m['kategori']}" if m.get("kategori") else ""
    return f"[Perizinan {idx}: {m.get('nama_perizinan', '?')}{kategori}]"


def _gold_id(doc: Document) -> str | None:
    m = doc.metadata or {}
    if m.get("perizinan_id") and m.get("doc_type") == "perizinan":
        return f"perizinan:id:{m['perizinan_id']}"
    return None


def _label(doc: Document) -> str:
    m = doc.metadata or {}
    nama = m.get("nama_perizinan")
    return f"SIPUAS Kab. Batang — {nama}" if nama else "SIPUAS Kab. Batang"


perizinan_capability = VectorSourceCapability(
    name="perizinan",
    description=(
        "Syarat, prosedur/mekanisme, dasar hukum, biaya & estimasi waktu pengurusan "
        "izin/perizinan di Kab. Batang (SIPUAS): izin reklame, izin kerja tenaga "
        "kesehatan (fisioterapis, radiografer, dll.), dan izin usaha lainnya."
    ),
    collection_name="perizinan",
    persist_directory=PERIZINAN_VECTOR_STORE,
    strategy="hybrid",
    header_formatter=_header,
    gold_id_fn=_gold_id,
    citation="sebutkan nama perizinan.",
    label_formatter=_label,
)
