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
)
