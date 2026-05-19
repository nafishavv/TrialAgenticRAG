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


opd_capability = VectorSourceCapability(
    name="opd",
    description=(
        "Direktori OPD Kab. Batang: alamat kantor, nomor telepon, email, "
        "struktur dinas/bagian/kecamatan/kelurahan."
    ),
    collection_name="opd_directory",
    persist_directory=OPD_VECTOR_STORE,
    router_examples=[
        "Alamat Dinas Pariwisata Batang?",
        "Nomor telepon Sekretariat Daerah?",
    ],
    strategy="hybrid",
    header_formatter=_header,
)
