"""Dukcapil chunking — Q&A-aware split for BAB II, narrative split for others.

Input: cleaned Documents from `preprocessing.dukcapil.preprocess()`.
Output: chunks ready for embedding into the dukcapil_qa Chroma collection.

The BAB II strategy detects per-question boundaries via regex (Q&A book format)
so each chunk = one full question + answer, even when crossing page breaks.
Narrative sections (Kata Pengantar, BAB I, BAB III) use RecursiveCharacterTextSplitter.
"""

from __future__ import annotations

import re
from typing import List, Tuple

from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

QUESTION_PATTERN = re.compile(
    r"(?:^|\n)\s*(\d{1,3})[\.\s]+(Apakah|Apa|Bagaimana|Berapa|Mengapa|Kapan|Di mana|Dimana|Siapa)\b",
    re.IGNORECASE,
)
SUBSECTION_PATTERN = re.compile(r"\n\s*([A-Z])\.\s+([A-Z][A-Z\s/&,-]{3,})\s*\n")

NARRATIVE_CHUNK_SIZE = 1200
NARRATIVE_OVERLAP = 200


# ---------- Helpers ----------
def _concat_section(docs: List[Document], section_name: str) -> Tuple[str, List[Tuple[int, int]]]:
    """Concat page contents of a section, return full text + (char_offset, page) map."""
    section_docs = sorted(
        [d for d in docs if d.metadata["section"] == section_name],
        key=lambda d: d.metadata["page"],
    )
    full_text = ""
    offset_map: List[Tuple[int, int]] = []
    for doc in section_docs:
        offset_map.append((len(full_text), doc.metadata["page"]))
        full_text += doc.page_content + "\n\n"
    return full_text, offset_map


def _page_at(offset_map: List[Tuple[int, int]], char_idx: int) -> int:
    page = offset_map[0][1]
    for off, p in offset_map:
        if off > char_idx:
            break
        page = p
    return page


def _find_subsections(text: str) -> List[Tuple[int, str]]:
    return [
        (m.start(), f"{m.group(1)}. {m.group(2).strip()}")
        for m in SUBSECTION_PATTERN.finditer(text)
    ]


def _subsection_at(subsections: List[Tuple[int, str]], char_idx: int) -> str:
    label = "Unknown"
    for off, lbl in subsections:
        if off > char_idx:
            break
        label = lbl
    return label


# ---------- BAB II Q&A chunking ----------
def chunk_qa_bab2(docs: List[Document]) -> List[Document]:
    text, offsets = _concat_section(docs, "BAB II - Pertanyaan dan Jawaban")
    if not text.strip():
        return []
    matches = list(QUESTION_PATTERN.finditer(text))
    subsections = _find_subsections(text)
    chunks = []
    for i, m in enumerate(matches):
        start = m.start()
        end = matches[i + 1].start() if i + 1 < len(matches) else len(text)
        chunk_text = text[start:end].strip()
        if len(chunk_text) < 50:
            continue
        q_match = re.match(r"\s*\d{1,3}[\.\s]+([^?]+\??)", chunk_text)
        question_text = q_match.group(1).strip() if q_match else ""
        chunks.append(Document(
            page_content=chunk_text,
            metadata={
                "section": "BAB II - Pertanyaan dan Jawaban",
                "subsection": _subsection_at(subsections, start),
                "question_number": int(m.group(1)),
                "question_text": question_text[:200],
                "page_start": _page_at(offsets, start),
                "page_end": _page_at(offsets, end - 1),
                "chunk_type": "qa",
            },
        ))
    return chunks


# ---------- Narrative chunking ----------
_narrative_splitter = RecursiveCharacterTextSplitter(
    chunk_size=NARRATIVE_CHUNK_SIZE,
    chunk_overlap=NARRATIVE_OVERLAP,
    separators=["\n\n", "\n", ". ", " ", ""],
)


def chunk_narrative(docs: List[Document], section_name: str) -> List[Document]:
    text, offsets = _concat_section(docs, section_name)
    if not text.strip():
        return []
    raw_chunks = _narrative_splitter.split_text(text)
    out = []
    cursor = 0
    for ch in raw_chunks:
        start = text.find(ch, cursor)
        if start == -1:
            start = cursor
        end = start + len(ch)
        out.append(Document(
            page_content=ch,
            metadata={
                "section": section_name,
                "page_start": _page_at(offsets, start),
                "page_end": _page_at(offsets, max(start, end - 1)),
                "chunk_type": "narrative",
            },
        ))
        cursor = end - NARRATIVE_OVERLAP
    return out


def chunk_for_vectorstore(docs: List[Document]) -> List[Document]:
    """Two-track chunking: Q&A for BAB II + narrative for everything else."""
    chunks: List[Document] = []
    chunks.extend(chunk_qa_bab2(docs))
    for section in ("Kata Pengantar", "BAB I - Pendahuluan", "BAB III - Penutup"):
        chunks.extend(chunk_narrative(docs, section))
    return chunks
