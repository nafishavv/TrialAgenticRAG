"""Perizinan SIPUAS preprocessing — JSON load + atomic Document per perizinan.

Source: data/raw/perizinan/perizinan_data.json (34 perizinan, hasil scraping
website SIPUAS Batang). Tiap perizinan punya 4 section: persyaratan, mekanisme
pelayanan, dasar hukum, keterangan (estimasi/masa berlaku/biaya).

Output: list[Document], satu per perizinan (34 total), ATOMIC (tanpa chunking —
chunk.py = identity). Semua section digabung dalam satu page_content terstruktur
agar LLM punya konteks lengkap begitu perizinan yang benar keretrieve.
"""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import List

from langchain_core.documents import Document


def _clean(text: str) -> str:
    """Normalisasi whitespace artefak scraping (\\r\\n di tengah kalimat, spasi ganda).

    Typo sumber (mis. 'BBerkas') TIDAK diubah — biarkan apa adanya.
    """
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def _render(item: dict) -> str:
    """Susun page_content terstruktur dari semua section satu perizinan."""
    lines = [f"Perizinan: {item['nama_perizinan']}"]
    if item.get("kategori"):
        lines.append(f"Kategori: {item['kategori']}")

    lines.append("\n=== PERSYARATAN ===")
    for syarat in item.get("persyaratan", []):
        lines.append(f"• {_clean(syarat)}")

    lines.append("\n=== MEKANISME PELAYANAN ===")
    for step in item.get("mekanisme_pelayanan", []):
        lines.append(f"{step['nomor']}. {_clean(step['deskripsi'])}")

    lines.append("\n=== DASAR HUKUM ===")
    for hukum in item.get("dasar_hukum", []):
        lines.append(f"{hukum['nomor']}. {_clean(hukum['referensi'])}")

    lines.append("\n=== KETERANGAN ===")
    ket = item.get("keterangan", {})
    if ket.get("estimasi_selesai"):
        lines.append(f"Estimasi Selesai: {_clean(ket['estimasi_selesai'])}")
    if ket.get("masa_berlaku"):
        lines.append(f"Masa Berlaku: {_clean(ket['masa_berlaku'])}")
    if ket.get("biaya"):
        lines.append(f"Biaya: {_clean(ket['biaya'])}")

    return "\n".join(lines)


def preprocess(json_path: Path) -> List[Document]:
    """Load perizinan JSON dan ubah jadi satu Document atomik per perizinan."""
    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    docs: List[Document] = []
    for item in data:
        metadata = {
            "source": "perizinan_data.json",
            "doc_type": "perizinan",
            "perizinan_id": item["id"],
            "nama_perizinan": item["nama_perizinan"],
            "kategori": item.get("kategori", ""),
            "url": item.get("url_sumber", ""),
            "crawl_date": item.get("crawl_timestamp", ""),
        }
        docs.append(Document(page_content=_render(item), metadata=metadata))
    return docs
