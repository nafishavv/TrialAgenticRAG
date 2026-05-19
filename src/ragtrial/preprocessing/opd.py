"""OPD directory preprocessing — pdfplumber table extraction + structured parsing.

Source: "Nama dan Alamat OPD Kab Batang.pdf" (3 pages, tabular).
Output: list[Document], one per OPD record (~61), atomic (no chunking).

Pipeline:
  1. Extract all table rows via pdfplumber.
  2. Merge cross-page continuation rows (entry whose body bleeds onto next page).
  3. Parse each row -> structured record (nomor, nama_opd, parent_opd, tipe,
     alamat, email, no_telp, page).
  4. Render record as a Document with formatted page_content + rich metadata.
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import List, Optional, Tuple

import pdfplumber
from langchain_core.documents import Document

BULLET = ""
SUB_ENTRY_RE = re.compile(r"^[a-z]\.\s")
EMAIL_RE = re.compile(r"[\w.-]+@[\w.-]+")

TIPE_KEYWORDS = [
    "Sekretariat", "Bagian", "Inspektorat", "Badan", "Dinas",
    "Satpol", "Kantor", "Kecamatan", "Kelurahan", "RSUD",
]


# ---------- Step 1: raw table extraction ----------
def extract_raw_rows(pdf_path: Path) -> List[dict]:
    """Return raw rows with original cell strings, skipping header row."""
    raw_rows = []
    with pdfplumber.open(pdf_path) as pdf:
        for page_idx, page in enumerate(pdf.pages):
            for table in page.extract_tables():
                for row in table:
                    if row and row[0] and str(row[0]).strip().upper() == "NO":
                        continue
                    raw_rows.append({"page": page_idx + 1, "cells": row})
    return raw_rows


# ---------- Step 2: continuation-row merging ----------
def _is_empty(v) -> bool:
    return v is None or str(v).strip() == ""


def _is_sub_entry(nama_cell: str) -> bool:
    return bool(nama_cell) and bool(SUB_ENTRY_RE.match(nama_cell.strip()))


def _merge_cell(a: Optional[str], b: Optional[str], sep: str = " ") -> str:
    a = (a or "").strip()
    b = (b or "").strip()
    if not a:
        return b
    if not b:
        return a
    return f"{a}{sep}{b}"


def merge_continuation_rows(raw_rows: List[dict]) -> List[dict]:
    """An entry whose first column is empty AND nama is not a sub-entry pattern
    is treated as a continuation of the previous row (PDF cross-page split)."""
    merged = []
    for r in raw_rows:
        cells = r["cells"]
        nomor, nama, alamat_email, telp = cells[0], cells[1], cells[2], cells[3]
        if _is_empty(nomor) and not _is_sub_entry(nama or ""):
            if merged:
                prev = merged[-1]
                prev["nama"] = _merge_cell(prev["nama"], nama)
                prev["alamat_email"] = _merge_cell(prev["alamat_email"], alamat_email, sep="\n")
                prev["telp"] = _merge_cell(prev["telp"], telp, sep="\n")
            continue
        merged.append({
            "page": r["page"],
            "nomor": nomor,
            "nama": nama,
            "alamat_email": alamat_email,
            "telp": telp,
        })
    return merged


# ---------- Step 3: per-field parsers ----------
def _clean_text(s: Optional[str]) -> str:
    if s is None:
        return ""
    s = s.replace(BULLET, "")
    s = re.sub(r"[ \t]+", " ", s)
    return s.strip()


def parse_alamat_email(cell: Optional[str]) -> Tuple[str, Optional[str]]:
    """Split combined alamat+email cell: lines with '@' = email, others = alamat."""
    if not cell:
        return "", None
    lines = [_clean_text(ln) for ln in cell.split("\n")]
    lines = [ln for ln in lines if ln]
    alamat_parts, email = [], None
    for ln in lines:
        m = EMAIL_RE.search(ln)
        if m:
            email = m.group(0).strip().rstrip(".")
        else:
            alamat_parts.append(ln)
    return " ".join(alamat_parts).strip(), email


def parse_telp(cell: Optional[str]) -> List[str]:
    if not cell:
        return []
    text = _clean_text(cell.replace("\n", " "))
    return [p.strip() for p in text.split("/") if p.strip()]


def infer_tipe(nama: str) -> str:
    nama = nama.strip()
    for kw in TIPE_KEYWORDS:
        if nama.startswith(kw) or nama.startswith(f"{kw} "):
            return kw
    return "Lainnya"


def parse_nama_and_subid(nama_raw: str) -> Tuple[Optional[str], str]:
    """For 'a. Bagian Pemerintahan' -> ('a', 'Bagian Pemerintahan').
    For 'Sekretariat Daerah' -> (None, 'Sekretariat Daerah')."""
    nama_raw = _clean_text(nama_raw)
    m = re.match(r"^([a-z])\.\s+(.+)$", nama_raw)
    if m:
        return m.group(1), m.group(2).strip()
    return None, nama_raw


def parse_main_nomor(nomor_raw: Optional[str]) -> str:
    return _clean_text(nomor_raw).rstrip(".")


# ---------- Step 4: rows -> records -> Documents ----------
def rows_to_records(merged_rows: List[dict]) -> List[dict]:
    records = []
    current_main_nomor: Optional[str] = None
    current_main_nama: Optional[str] = None

    for r in merged_rows:
        sub_letter, nama_clean = parse_nama_and_subid(r["nama"])
        alamat, email = parse_alamat_email(r["alamat_email"])
        no_telp = parse_telp(r["telp"])

        if sub_letter is None:
            nomor = parse_main_nomor(r["nomor"])
            current_main_nomor = nomor
            current_main_nama = nama_clean
            parent = None
        else:
            nomor = f"{current_main_nomor}.{sub_letter}"
            parent = current_main_nama

        records.append({
            "nomor": nomor,
            "nama_opd": nama_clean,
            "parent_opd": parent,
            "tipe": infer_tipe(nama_clean),
            "alamat": alamat,
            "email": email,
            "no_telp": no_telp,
            "page": r["page"],
        })
    return records


def record_to_document(rec: dict, source_name: str) -> Document:
    lines = [f"Nama OPD: {rec['nama_opd']}"]
    if rec["parent_opd"]:
        lines.append(f"Bagian dari: {rec['parent_opd']}")
    lines.append(f"Tipe: {rec['tipe']}")
    lines.append(f"Alamat: {rec['alamat']}")
    lines.append(f"Email: {rec['email'] if rec['email'] else '-'}")
    telp_str = ", ".join(rec["no_telp"]) if rec["no_telp"] else "-"
    lines.append(f"No. Telp: {telp_str}")

    metadata = {
        "source": source_name,
        "doc_type": "opd_directory",
        "nomor": rec["nomor"],
        "nama_opd": rec["nama_opd"],
        "parent_opd": rec["parent_opd"] or "",
        "tipe": rec["tipe"],
        "page": rec["page"],
        "has_email": bool(rec["email"]),
        "has_telp": bool(rec["no_telp"]),
    }
    return Document(page_content="\n".join(lines), metadata=metadata)


def preprocess(pdf_path: Path) -> List[Document]:
    """Run full OPD pipeline: extract → merge → parse → Documents."""
    raw = extract_raw_rows(pdf_path)
    merged = merge_continuation_rows(raw)
    records = rows_to_records(merged)
    return [record_to_document(r, pdf_path.name) for r in records]
