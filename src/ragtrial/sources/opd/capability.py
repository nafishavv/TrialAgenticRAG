"""OPD capability — Direktori Organisasi Perangkat Daerah Kab. Batang."""

from __future__ import annotations

from langchain_core.documents import Document

from ragtrial.capabilities.vector_source import VectorSourceCapability
from ragtrial.config import OPD_VECTOR_STORE


def _header(doc: Document, idx: int) -> str:
    m = doc.metadata or {}
    parent = f" (bagian dari {m['parent_opd']})" if m.get("parent_opd") else ""
    return (
        f"[OPD {idx}: {m.get('nama_opd', '?')}{parent}, "
        f"nomor {m.get('nomor', '?')}, tipe {m.get('tipe', '?')}]"
    )


def _gold_id(doc: Document) -> str | None:
    m = doc.metadata or {}
    if "nomor" in m and m.get("doc_type") == "opd_directory":
        return f"opd:nomor:{m['nomor']}"
    return None


def _label(doc: Document) -> str:
    m = doc.metadata or {}
    nama = m.get("nama_opd")
    return f"Direktori OPD Kab. Batang — {nama}" if nama else "Direktori OPD Kab. Batang"


opd_capability = VectorSourceCapability(
    name="opd",
    description=(
        "Direktori OPD Kab. Batang: alamat kantor, nomor telepon, email, "
        "struktur dinas/bagian/kecamatan/kelurahan."
    ),
    collection_name="opd_directory",
    persist_directory=OPD_VECTOR_STORE,
    strategy="hybrid",
    header_formatter=_header,
    gold_id_fn=_gold_id,
    citation="sebutkan [Sumber: <nama_opd>, nomor <nomor>].",
    label_formatter=_label,
)
