"""Sosial chunking — hybrid per-dokumen: Pasal-split atau narrative.

Input: cleaned Documents per-HALAMAN dari `sosial.preprocess` (dikelompokkan
lewat metadata `id`). Output: chunks siap embed.

Gate per-dokumen (data-driven, bukan per-tipe):
- Dokumen dengan >= MIN_PASAL heading "Pasal N" valid → Pasal-split: 1 chunk =
  1 Pasal utuh, plus 1 chunk "preamble" untuk teks sebelum Pasal pertama
  (judul + Menimbang + Mengingat — penting di dok hukum).
- Selain itu (Abstrak, Artikel, Terjemahan, dll) → narrative split via
  RecursiveCharacterTextSplitter.

Deteksi heading "Pasal N" tahan-banting (penting untuk dok ber-text-layer rusak
yang sudah di-de-spacing jadi satu aliran teks tanpa newline heading):
- **Section-aware**: pisahkan "Batang Tubuh" vs "Penjelasan" (penomoran Pasal
  reset 1..N di bagian Penjelasan) lewat penanda "PASAL DEMI PASAL"/"PENJELASAN
  ATAS". chunk_type → `pasal` (batang) vs `penjelasan_pasal`.
- **Anti cross-reference**: "Pasal N" yang didahului kata rujukan ("dalam",
  "dimaksud", "sebagaimana", dll) atau yang nomornya tak naik monoton dalam
  section-nya (rujukan ke pasal sebelumnya) ditolak sebagai heading.

Setiap chunk membawa metadata dokumen (id, nomor, tipe, status, dll) + identitas
chunk (chunk_type, section, pasal_number / narr_index, part) untuk gold-id
chunk-precise di `capability.gold_id`.
"""

from __future__ import annotations

import re
from collections import OrderedDict
from typing import Dict, List, Optional, Tuple

from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

# Dok bersih: heading "Pasal N" ada di awal baris (newline utuh) — andal & presisi.
PASAL_LINESTART = re.compile(r"(?:^|\n)\s*Pasal\s+(\d+)", re.IGNORECASE)
# Dok charspaced (newline hancur pasca-rekonstruksi): kandidat "Pasal N" di mana
# saja, disaring lewat anchor MEMUTUSKAN + anti cross-reference + langkah ter-batas.
PASAL_CANDIDATE = re.compile(r"Pasal\s+(\d{1,3})\b", re.IGNORECASE)

# Penanda awal bagian Penjelasan (penomoran Pasal reset di sini).
PENJELASAN_MARKER = re.compile(r"PASAL\s+DEMI\s+PASAL|PENJELASAN\s+ATAS", re.IGNORECASE)

# Penanda awal Batang Tubuh (dispositif). "Pasal N" sebelum ini hanyalah sitasi
# dasar hukum di bagian "Mengingat" (mis. "Pasal 18 ayat (6) UUD 1945") — bukan
# heading. Anchor ini mencegah sitasi meracuni urutan monoton nomor pasal.
MEMUTUSKAN_MARKER = re.compile(r"MEMUTUSKAN", re.IGNORECASE)

# Kata rujukan tepat sebelum "Pasal N" → ini cross-reference, bukan heading.
CROSSREF_CUES = re.compile(
    r"(dalam|pada|dimaksud|melanggar|sebagaimana|huruf|ayat|dan|atau|sampai|"
    r"berdasarkan|menurut|mengubah|ketentuan|bunyi|yaitu|s\.?\s*d\.?)\s*$",
    re.IGNORECASE,
)

MIN_PASAL = 2            # ambang dokumen dianggap "ber-Pasal"
# Heading "Pasal N" diterima hanya bila nomornya naik 1..GAP dari pasal terakhir.
# Pasal hukum bernomor urut 1,2,3,…; lompatan besar = angka tabel/lampiran atau
# cross-reference ke pasal jauh → ditolak (mencegah satu angka liar meracuni
# seluruh urutan). Toleransi >1 menampung pasal yang sesekali terlewat karena OCR.
PASAL_STEP_MAX = 3
MIN_CHUNK_CHARS = 50     # chunk lebih pendek dari ini dibuang
MAX_CHUNK_CHARS = 6000   # chunk lebih besar (Pasal w/ lampiran, preamble panjang) di-sub-split

NARRATIVE_CHUNK_SIZE = 1200
NARRATIVE_OVERLAP = 200

_narrative_splitter = RecursiveCharacterTextSplitter(
    chunk_size=NARRATIVE_CHUNK_SIZE,
    chunk_overlap=NARRATIVE_OVERLAP,
    separators=["\n\n", "\n", ". ", " ", ""],
)


# ── Gold-id suffix (sumber kebenaran tunggal; dipakai capability.gold_id) ────────
def gold_id_suffix(m: dict) -> str:
    """Suffix gold-id chunk-precise dari metadata chunk (lihat capability.gold_id).

    Unik per chunk dalam satu dokumen (di-assert di `chunk_for_vectorstore`):
      pasal            → #pasal:<n>           penjelasan_pasal → #penjelasan:<n>
      preamble         → #preamble            penjelasan_preamble → #penjelasan-preamble
      narrative        → #narr:<k>            (fallback) → #p<page_start>
    Chunk hasil sub-split `_enforce_max` menambah ".<part>".
    """
    ct = m.get("chunk_type")
    part = m.get("part")
    suffix = f".{part}" if part is not None else ""
    if ct == "pasal":
        key = f"#pasal:{m.get('pasal_number')}"
    elif ct == "penjelasan_pasal":
        key = f"#penjelasan:{m.get('pasal_number')}"
    elif ct == "preamble":
        key = "#preamble"
    elif ct == "penjelasan_preamble":
        key = "#penjelasan-preamble"
    elif ct == "narrative":
        key = f"#narr:{m.get('narr_index')}"
    else:
        key = f"#p{m.get('page_start')}"
    return key + suffix


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


# ── Deteksi heading Pasal (section-aware, anti cross-reference, monotonic) ───────
def _penjelasan_offset(text: str) -> Optional[int]:
    m = PENJELASAN_MARKER.search(text)
    return m.start() if m else None


def _body_start(text: str) -> int:
    """Offset awal Batang Tubuh (setelah 'MEMUTUSKAN'); 0 bila penanda tak ada."""
    m = MEMUTUSKAN_MARKER.search(text)
    return m.end() if m else 0


def _section_of(start: int, pj: Optional[int]) -> str:
    return "penjelasan" if (pj is not None and start >= pj) else "batang"


def _headings_linestart(text: str, pj: Optional[int]) -> List[Tuple[int, int, str]]:
    """Dok bersih: heading = "Pasal N" di awal baris (newline utuh). Klasifikasi
    section via penanda Penjelasan; dedup naik-monoton per-section (membuang
    cross-reference awal-baris sesekali agar gold-id tetap unik).

    Guard MEMUTUSKAN: "Pasal N" di bagian Mengingat/Menimbang (sebelum kata
    'MEMUTUSKAN') hanya sitasi dasar hukum ("Pasal 18 ayat (6) UUD 1945") —
    bukan heading. Bila 'MEMUTUSKAN' tidak ada (pj_body=0), guard nonaktif.
    """
    body = _body_start(text)
    headings: List[Tuple[int, int, str]] = []
    last = {"batang": 0, "penjelasan": 0}
    for m in PASAL_LINESTART.finditer(text):
        section = _section_of(m.start(), pj)
        if body and section == "batang" and m.start() < body:
            continue
        n = int(m.group(1))
        if n <= last[section]:
            continue
        headings.append((m.start(), n, section))
        last[section] = n
    return headings


def _headings_inline(text: str, pj: Optional[int]) -> List[Tuple[int, int, str]]:
    """Dok charspaced: newline heading hancur → deteksi "Pasal N" inline. Saring:
    hanya di Batang Tubuh (setelah 'MEMUTUSKAN', agar sitasi "Mengingat" tak
    terhitung), bukan didahului kata rujukan, dan nomornya naik 1..PASAL_STEP_MAX
    dari pasal terakhir (menolak angka tabel/lampiran liar & cross-reference)."""
    body = _body_start(text)
    headings: List[Tuple[int, int, str]] = []
    last = {"batang": 0, "penjelasan": 0}
    for m in PASAL_CANDIDATE.finditer(text):
        section = _section_of(m.start(), pj)
        if section == "batang" and m.start() < body:
            continue
        n = int(m.group(1))
        pre = text[max(0, m.start() - 25):m.start()]
        if CROSSREF_CUES.search(pre):
            continue
        if not (last[section] < n <= last[section] + PASAL_STEP_MAX):
            continue
        headings.append((m.start(), n, section))
        last[section] = n
    return headings


def _find_headings(text: str, charspaced: bool) -> Tuple[List[Tuple[int, int, str]], Optional[int]]:
    """Return ([(start, pasal_number, section)], penjelasan_offset).

    Strategi deteksi heading dipilih per kualitas text-layer dokumen (lihat
    `_headings_linestart` vs `_headings_inline`).
    """
    pj = _penjelasan_offset(text)
    finder = _headings_inline if charspaced else _headings_linestart
    return finder(text, pj), pj


def _is_pasal_doc(headings: List[Tuple[int, int, str]]) -> bool:
    """Dokumen layak Pasal-split bila punya >= MIN_PASAL heading batang DAN
    Batang Tubuh dimulai di Pasal ≤5.

    Ambang 5 (bukan 2) menampung doc dimana Pasal 1 & 2 ada di tengah baris
    ("BAB I KETENTUAN UMUM Pasal 1") sehingga regex awal-baris melewatkannya;
    pasal pertama yang ter-deteksi bisa Pasal 3+. Pasal 1-2 yang terlewat masuk
    ke preamble chunk (ketentuan umum/definisi — wajar).

    Gate ini tetap menolak: (a) dok naratif/akademis yang cuma menyebut "Pasal X"
    tinggi secara tersebar, (b) dok Perubahan/Amandemen yang hanya punya "Pasal I"
    roman + pasal yang diamandemen (mis. Pasal 26A, Pasal 39A) — keduanya
    menghasilkan chunk Pasal raksasa (mis-segmentasi). Narrative lebih aman.
    """
    batang = [n for _, n, s in headings if s == "batang"]
    return len(batang) >= MIN_PASAL and batang[0] <= 5


# ── Pasal-split (section-aware) ─────────────────────────────────────────────────
def _make_pasal_chunk(text: str, start: int, end: int, offsets, base: dict,
                      section: str, n: int) -> Optional[Document]:
    chunk_text = text[start:end].strip()
    if len(chunk_text) < MIN_CHUNK_CHARS:
        return None
    ct = "pasal" if section == "batang" else "penjelasan_pasal"
    return Document(page_content=chunk_text, metadata={
        **base, "chunk_type": ct, "section": section, "pasal_number": n,
        "page_start": _page_at(offsets, start), "page_end": _page_at(offsets, end - 1),
    })


def _make_preamble(text: str, start: int, end: int, offsets, base: dict,
                   section: str) -> Optional[Document]:
    pre = text[start:end].strip()
    if len(pre) < MIN_CHUNK_CHARS:
        return None
    ct = "preamble" if section == "batang" else "penjelasan_preamble"
    return Document(page_content=pre, metadata={
        **base, "chunk_type": ct, "section": section,
        "page_start": _page_at(offsets, start), "page_end": _page_at(offsets, max(start, end - 1)),
    })


def _chunk_pasal(text: str, offsets, base: dict,
                 headings: List[Tuple[int, int, str]], pj: Optional[int]) -> List[Document]:
    chunks: List[Document] = []
    first_batang = next((h for h in headings if h[2] == "batang"), None)
    first_penj = next((h for h in headings if h[2] == "penjelasan"), None)

    # Preamble batang: teks sebelum Pasal batang pertama (judul, Menimbang, Mengingat).
    if first_batang:
        pre = _make_preamble(text, 0, first_batang[0], offsets, base, "batang")
        if pre:
            chunks.append(pre)
    # Preamble penjelasan: teks antara penanda PENJELASAN dan Pasal penjelasan pertama.
    if pj is not None and first_penj is not None:
        pre = _make_preamble(text, pj, first_penj[0], offsets, base, "penjelasan")
        if pre:
            chunks.append(pre)

    for i, (start, n, section) in enumerate(headings):
        end = headings[i + 1][0] if i + 1 < len(headings) else len(text)
        # Batang terakhir: jangan telan teks bagian Penjelasan.
        if section == "batang" and pj is not None and end > pj:
            end = pj
        c = _make_pasal_chunk(text, start, end, offsets, base, section, n)
        if c:
            chunks.append(c)
    return chunks


# ── Narrative-split ──────────────────────────────────────────────────────────────
def _chunk_narrative(text: str, offsets, base: dict) -> List[Document]:
    raw_chunks = _narrative_splitter.split_text(text)
    out: List[Document] = []
    cursor = 0
    k = 0
    for ch in raw_chunks:
        if len(ch.strip()) < MIN_CHUNK_CHARS:
            continue
        start = text.find(ch, cursor)
        if start == -1:
            start = cursor
        end = start + len(ch)
        out.append(Document(page_content=ch, metadata={
            **base, "chunk_type": "narrative", "narr_index": k,
            "page_start": _page_at(offsets, start),
            "page_end": _page_at(offsets, max(start, end - 1)),
        }))
        k += 1
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
    """Hybrid per-dokumen: Pasal-split bila ber-Pasal, narrative bila tidak.

    Meng-assert gold-id chunk-precise unik per dokumen (gagal-cepat bila tabrakan).
    """
    out: List[Document] = []
    for _id, pages in _group_by_id(docs).items():
        text, offsets = _concat(pages)
        if not text.strip():
            continue
        base = _doc_meta(pages)
        headings, pj = _find_headings(text, bool(base.get("charspaced")))
        if _is_pasal_doc(headings):
            doc_chunks = _enforce_max(_chunk_pasal(text, offsets, base, headings, pj))
        else:
            doc_chunks = _enforce_max(_chunk_narrative(text, offsets, base))

        # Assertion: gold-id chunk-precise wajib unik dalam satu dokumen.
        keys = [gold_id_suffix(c.metadata) for c in doc_chunks]
        if len(set(keys)) != len(keys):
            from collections import Counter
            dups = [k for k, c in Counter(keys).items() if c > 1]
            raise AssertionError(f"Gold-id tabrakan di dokumen {_id}: {dups}")
        out.extend(doc_chunks)
    return out
