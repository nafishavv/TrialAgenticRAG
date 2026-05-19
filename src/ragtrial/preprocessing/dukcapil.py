"""Dukcapil Buku Saku preprocessing — page filter, text cleanup, section tagging.

Source: Buku Saku Dafduk Capil Kab. Batang 2023 (282 pages PDF).
Output: list[Document], one per page (258 after filtering), each tagged with
`section` metadata.

The exact page ranges below are calibrated to this specific PDF; if the source
PDF is replaced the constants must be re-tuned (see notebooks/exploration/
preprocessing_dukcapil.ipynb for the manual inspection that produced them).
"""

from __future__ import annotations

import copy
import re
from pathlib import Path
from typing import List

from langchain_community.document_loaders import PyMuPDFLoader
from langchain_core.documents import Document

# Pages dibuang: cover scan (rasio karakter "noise" tinggi) + daftar isi.
COVER_PAGES: set[int] = {0, 281}
DAFTAR_ISI_PAGES: set[int] = set(range(3, 25))
SKIP_PAGES: set[int] = COVER_PAGES | DAFTAR_ISI_PAGES


def load_pdf(pdf_path: Path) -> List[Document]:
    return PyMuPDFLoader(str(pdf_path)).load()


def filter_pages(docs: List[Document]) -> List[Document]:
    return [d for d in docs if d.metadata["page"] not in SKIP_PAGES]


def clean_page(text: str) -> str:
    """Strip table-of-contents dots, lonely page numbers, hard wraps, double whitespace."""
    # 1. Hapus deretan titik (artefak daftar isi)
    text = re.sub(r"\.{3,}", "", text)
    # 2. Hapus nomor halaman standalone (baris hanya berisi angka)
    text = re.sub(r"^\s*\d{1,3}\s*$", "", text, flags=re.MULTILINE)
    # 3. Sambung baris yang terpotong di tengah kalimat (bukan setelah tanda baca)
    text = re.sub(r"(?<![.!?:\-])\n(?=[a-zA-Z])", " ", text)
    # 4. Normalisasi whitespace
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def tag_section(page: int) -> str:
    if page in (1, 2):
        return "Kata Pengantar"
    if 25 <= page <= 31:
        return "BAB I - Pendahuluan"
    if 32 <= page <= 277:
        return "BAB II - Pertanyaan dan Jawaban"
    if 278 <= page <= 280:
        return "BAB III - Penutup"
    return "Unknown"


def preprocess(pdf_path: Path) -> List[Document]:
    """Run the full Dukcapil pipeline. Returns cleaned + tagged Documents."""
    pdf_docs = load_pdf(pdf_path)
    filtered = filter_pages(pdf_docs)

    cleaned: List[Document] = []
    for doc in filtered:
        new = copy.copy(doc)
        new.page_content = clean_page(doc.page_content)
        if new.page_content:
            new.metadata["section"] = tag_section(new.metadata["page"])
            cleaned.append(new)
    return cleaned
