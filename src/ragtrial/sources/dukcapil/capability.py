"""Dukcapil capability — Buku Saku administrasi kependudukan (Q&A + narasi)."""

from __future__ import annotations

from langchain_core.documents import Document

from ragtrial.capabilities.vector_source import VectorSourceCapability
from ragtrial.config import DUKCAPIL_VECTOR_STORE


def _header(doc: Document, idx: int) -> str:
    m = doc.metadata or {}
    section = m.get("section", "?")
    page = m.get("page", m.get("page_start", "?"))
    return f"[Dukcapil {idx}: section={section}, hal={page}]"


def _gold_id(doc: Document) -> str | None:
    m = doc.metadata or {}
    if "page_start" in m:
        return f"dukcapil:page:{m['page_start']}"
    if m.get("section") and "page" in m:
        return f"dukcapil:page:{m['page']}"
    return None


def _label(doc: Document) -> str:
    m = doc.metadata or {}
    page = m.get("page", m.get("page_start"))
    hal = f", hal. {page}" if page is not None else ""
    return f"Buku Saku Adminduk Kab. Batang 2023{hal}"


dukcapil_capability = VectorSourceCapability(
    name="dukcapil",
    description=(
        "Prosedur / syarat / aturan administrasi kependudukan: KTP, KK, akta "
        "kelahiran, akta kematian, pindah domisili, NIK, KITAP, dll. Sumber: "
        "Buku Saku Dafduk Capil Kab. Batang 2023."
    ),
    collection_name="dukcapil_qa",
    persist_directory=DUKCAPIL_VECTOR_STORE,
    router_examples=[
        "Apa syarat penerbitan KTP elektronik?",
        "Bagaimana cara pindah domisili?",
    ],
    strategy="hybrid",
    header_formatter=_header,
    gold_id_fn=_gold_id,
    citation="sebutkan section/halaman kalau ada di header konteks.",
    label_formatter=_label,
)
