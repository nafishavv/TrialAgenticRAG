"""Sosial (JDIH Batang) preprocessing — usability filter, de-spacing, cleanup.

Source: 79 PDF hukum sektor sosial (Perda/Perbup/Kepbup/Abstrak/Artikel/dll) +
`metadata.json` dari scraper. Output: list[Document], satu per HALAMAN dokumen
usable, masing-masing membawa metadata dokumen (id, nomor, tipe, status, dll)
sehingga `chunk.py` bisa mengelompokkan per-dokumen dan deteksi struktur Pasal.

Tiga keputusan korpus (lihat plan):
1. Usability filter: PDF dengan < MIN_CHAR_PER_PAGE char/halaman = scan/partial-scan
   (badan dokumen berupa gambar) → di-skip, dicatat di skip log.
2. De-spacing: sebagian PDF punya text layer rusak (tiap huruf ter-spasi,
   "P a s a l") → dinormalisasi agar struktur & konten kembali terbaca.
3. Status "Tidak Berlaku" tetap dimasukkan, ditandai di metadata.
"""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import List, Tuple

from langchain_community.document_loaders import PyMuPDFLoader
from langchain_core.documents import Document

from ragtrial.config import SOSIAL_PROCESSED_PKL, SOSIAL_RAW_DIR

# PDF dengan rata-rata char/halaman di bawah ini dianggap scan/partial-scan.
MIN_CHAR_PER_PAGE = 300

# Field dokumen dari metadata.json yang dibawa ke metadata tiap Document.
_DOC_FIELDS = (
    "id", "nomor", "judul", "tahun", "tipe_dokumen",
    "status", "bidang", "opd_pemrakarsa", "source_url",
)


# ── De-spacing (perbaikan text layer rusak) ───────────────────────────────────
def fix_charspacing(text: str) -> str:
    """Gabungkan teks yang ter-spasi per-huruf ("P a s a l" → "Pasal").

    Bekerja per baris. Baris dianggap char-spaced jika ≥4 token non-kosong dan
    >60% token sepanjang 1 karakter. Pada baris itu: token kosong (spasi-ganda)
    = batas kata, spasi-tunggal = antar-huruf → huruf digabung per kata.
    Baris normal dibiarkan apa adanya.
    """
    out: List[str] = []
    for line in text.split("\n"):
        toks = line.split(" ")
        nonempty = [t for t in toks if t != ""]
        if len(nonempty) >= 4 and sum(1 for t in nonempty if len(t) == 1) / len(nonempty) > 0.6:
            words: List[str] = []
            cur: List[str] = []
            for t in toks:
                if t == "":
                    if cur:
                        words.append("".join(cur))
                        cur = []
                else:
                    cur.append(t)
            if cur:
                words.append("".join(cur))
            out.append(" ".join(words))
        else:
            out.append(line)
    return "\n".join(out)


# ── Cleaning (pola dari dukcapil/preprocess.py) ────────────────────────────────
def clean_text(text: str) -> str:
    """Strip deretan titik, nomor halaman lonely, hard-wrap, double whitespace."""
    text = re.sub(r"\.{3,}", "", text)
    text = re.sub(r"^\s*\d{1,3}\s*$", "", text, flags=re.MULTILINE)
    text = re.sub(r"(?<![.!?:\-])\n(?=[a-zA-Z])", " ", text)
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


# ── Per-dokumen ────────────────────────────────────────────────────────────────
def _load_metadata(raw_dir: Path) -> List[dict]:
    with open(raw_dir / "metadata.json", encoding="utf-8") as f:
        return json.load(f)


def _process_one(meta: dict, raw_dir: Path) -> Tuple[List[Document], dict | None]:
    """Return (page_documents, skip_record). Salah satunya kosong/None."""
    filename = meta.get("filename", "")
    pdf_path = raw_dir / filename
    if not pdf_path.exists():
        return [], {"id": meta.get("id"), "filename": filename,
                    "reason": "file_missing", "char_per_page": 0,
                    "detail": "PDF tidak ditemukan di raw dir"}

    pages = PyMuPDFLoader(str(pdf_path)).load()
    raw_chars = sum(len(p.page_content) for p in pages)
    cpp = raw_chars // max(len(pages), 1)

    if cpp < MIN_CHAR_PER_PAGE:
        return [], {
            "id": meta.get("id"), "filename": filename,
            "reason": "scan_or_partial", "char_per_page": cpp,
            "detail": f"{len(pages)} halaman, hanya {raw_chars} char (kemungkinan "
                      f"badan dokumen berupa gambar, hanya footer/ttd punya text layer)",
        }

    doc_meta = {f: meta.get(f, "") for f in _DOC_FIELDS}
    doc_meta["source"] = filename
    doc_meta["doc_type"] = "sosial"

    out: List[Document] = []
    for p in pages:
        cleaned = clean_text(fix_charspacing(p.page_content))
        if not cleaned:
            continue
        out.append(Document(
            page_content=cleaned,
            metadata={**doc_meta, "page": p.metadata.get("page", 0)},
        ))
    return out, None


def preprocess(raw_dir: Path = SOSIAL_RAW_DIR, write_skip_log: bool = True) -> List[Document]:
    """Proses semua dokumen sosial usable jadi Documents per-halaman.

    Menulis audit trail dokumen yang di-skip ke
    `data/processed/sosial_skipped.json` (kecuali write_skip_log=False).
    """
    metas = _load_metadata(raw_dir)
    docs: List[Document] = []
    skipped: List[dict] = []

    for meta in metas:
        page_docs, skip = _process_one(meta, raw_dir)
        if skip is not None:
            skipped.append(skip)
        docs.extend(page_docs)

    n_usable = len(metas) - len(skipped)
    print(f"  Usable: {n_usable}/{len(metas)} dokumen -> {len(docs)} halaman Documents")
    print(f"  Skipped: {len(skipped)} dokumen (lihat sosial_skipped.json)")

    if write_skip_log:
        skip_path = SOSIAL_PROCESSED_PKL.parent / "sosial_skipped.json"
        skip_path.parent.mkdir(parents=True, exist_ok=True)
        with open(skip_path, "w", encoding="utf-8") as f:
            json.dump({"total_skipped": len(skipped), "skipped_docs": skipped},
                      f, ensure_ascii=False, indent=2)

    return docs
