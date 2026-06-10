"""Sosial chunking — hybrid per-dokumen: Pasal-split atau narrative.

Input: cleaned Documents per-HALAMAN dari `sosial.preprocess` (dikelompokkan
lewat metadata `id`). Output: chunks siap embed.

Gate per-dokumen (data-driven, bukan per-tipe):
- Dokumen dengan >= MIN_PASAL penanda "Pasal N" → Pasal-split: 1 chunk = 1 Pasal
  utuh (pola identik `chunk_qa_bab2` dukcapil), plus 1 chunk "preamble" untuk
  teks sebelum Pasal pertama (judul + Menimbang + Mengingat — penting di dok hukum).
- Selain itu (Abstrak, Artikel, Terjemahan, dll) → narrative split via
  RecursiveCharacterTextSplitter.

Setiap chunk membawa metadata dokumen (id, nomor, tipe, status, dll) dari halaman.
"""

from __future__ import annotations

import re
from collections import OrderedDict
from typing import Dict, List, Tuple

from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

PASAL_PATTERN = re.compile(r"(?:^|\n)\s*Pasal\s+(\d+)", re.IGNORECASE)
MIN_PASAL = 2            # ambang dokumen dianggap "ber-Pasal"
MIN_CHUNK_CHARS = 50     # chunk lebih pendek dari ini dibuang
MAX_CHUNK_CHARS = 6000   # chunk lebih besar (Pasal w/ lampiran, preamble panjang) di-sub-split

NARRATIVE_CHUNK_SIZE = 1200
NARRATIVE_OVERLAP = 200

_narrative_splitter = RecursiveCharacterTextSplitter(
    chunk_size=NARRATIVE_CHUNK_SIZE,
    chunk_overlap=NARRATIVE_OVERLAP,
    separators=["\n\n", "\n", ". ", " ", ""],
)


# ── Helpers ────────────────────────────────────────────────────────────────────
def _group_by_id(docs: List[Document]) -> "OrderedDict[str, List[Document]]":
    groups: "OrderedDict[str, List[Document]]" = OrderedDict()
    for d in docs:
        groups.setdefault(d.metadata.get("id", "?"), []).append(d)
    return groups


def _doc_meta(pages: List[Document]) -> dict:
    """Metadata dokumen (sama lintas halaman) tanpa field `page`."""
    m = dict(pages[0].metadata)
    m.pop("page", None)
    return m


def _concat(pages: List[Document]) -> Tuple[str, List[Tuple[int, int]]]:
    """Gabung page_content urut halaman; return teks + map (char_offset, page)."""
    ordered = sorted(pages, key=lambda d: d.metadata.get("page", 0))
    text = ""
    offset_map: List[Tuple[int, int]] = []
    for p in ordered:
        offset_map.append((len(text), p.metadata.get("page", 0)))
        text += p.page_content + "\n\n"
    return text, offset_map


def _page_at(offset_map: List[Tuple[int, int]], char_idx: int) -> int:
    page = offset_map[0][1]
    for off, p in offset_map:
        if off > char_idx:
            break
        page = p
    return page


# ── Pasal-split ─────────────────────────────────────────────────────────────────
def _chunk_pasal(text: str, offsets: List[Tuple[int, int]], base: dict) -> List[Document]:
    matches = list(PASAL_PATTERN.finditer(text))
    chunks: List[Document] = []

    # Preamble: teks sebelum Pasal pertama (judul, Menimbang, Mengingat).
    preamble = text[: matches[0].start()].strip()
    if len(preamble) >= MIN_CHUNK_CHARS:
        chunks.append(Document(
            page_content=preamble,
            metadata={**base, "chunk_type": "preamble",
                      "page_start": _page_at(offsets, 0),
                      "page_end": _page_at(offsets, matches[0].start())},
        ))

    for i, m in enumerate(matches):
        start = m.start()
        end = matches[i + 1].start() if i + 1 < len(matches) else len(text)
        chunk_text = text[start:end].strip()
        if len(chunk_text) < MIN_CHUNK_CHARS:
            continue
        chunks.append(Document(
            page_content=chunk_text,
            metadata={**base, "chunk_type": "pasal", "pasal_number": int(m.group(1)),
                      "page_start": _page_at(offsets, start),
                      "page_end": _page_at(offsets, end - 1)},
        ))
    return chunks


# ── Narrative-split ──────────────────────────────────────────────────────────────
def _chunk_narrative(text: str, offsets: List[Tuple[int, int]], base: dict) -> List[Document]:
    raw_chunks = _narrative_splitter.split_text(text)
    out: List[Document] = []
    cursor = 0
    for ch in raw_chunks:
        if len(ch.strip()) < MIN_CHUNK_CHARS:
            continue
        start = text.find(ch, cursor)
        if start == -1:
            start = cursor
        end = start + len(ch)
        out.append(Document(
            page_content=ch,
            metadata={**base, "chunk_type": "narrative",
                      "page_start": _page_at(offsets, start),
                      "page_end": _page_at(offsets, max(start, end - 1))},
        ))
        cursor = end - NARRATIVE_OVERLAP
    return out


# ── Safeguard: pecah chunk yang kelewat besar ────────────────────────────────────
def _enforce_max(chunks: List[Document]) -> List[Document]:
    """Sub-split chunk > MAX_CHUNK_CHARS (Pasal yang menyerap lampiran, preamble
    panjang) agar muat di batas token embedding. Metadata dipertahankan, ditambah
    `part` index. Halaman tetap pakai rentang chunk induk (perkiraan)."""
    out: List[Document] = []
    for c in chunks:
        if len(c.page_content) <= MAX_CHUNK_CHARS:
            out.append(c)
            continue
        for j, part in enumerate(_narrative_splitter.split_text(c.page_content)):
            if len(part.strip()) < MIN_CHUNK_CHARS:
                continue
            out.append(Document(page_content=part, metadata={**c.metadata, "part": j}))
    return out


# ── Entry point ──────────────────────────────────────────────────────────────────
def chunk_for_vectorstore(docs: List[Document]) -> List[Document]:
    """Hybrid per-dokumen: Pasal-split bila ber-Pasal, narrative bila tidak."""
    out: List[Document] = []
    for _id, pages in _group_by_id(docs).items():
        text, offsets = _concat(pages)
        if not text.strip():
            continue
        base = _doc_meta(pages)
        if len(PASAL_PATTERN.findall(text)) >= MIN_PASAL:
            out.extend(_chunk_pasal(text, offsets, base))
        else:
            out.extend(_chunk_narrative(text, offsets, base))
    return _enforce_max(out)
