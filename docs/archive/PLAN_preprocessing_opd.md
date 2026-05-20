# Plan: Preprocessing Data OPD Kab Batang

Dokumen ini berisi rencana preprocessing untuk file `data/pdf/Nama dan Alamat OPD Kab Batang.pdf` agar siap di-chunk, di-embed, dan dimasukkan ke vector store.

---

## 1. Hasil Investigasi Struktur Data

PDF ini berisi **tabel directory OPD Kabupaten Batang** dengan karakteristik:

| Aspek | Detail |
|---|---|
| Total halaman | 3 |
| Total entries | 43 OPD utama + sub-entries → **±62 total** |
| Kolom tabel | NO, NAMA OPD, ALAMAT DAN EMAIL, NO TELP |
| Sub-entries | No.1 (Sekretariat Daerah) punya 9 bagian (a–i); No.25 (Kec. Batang) punya 9 kelurahan (a–i) |
| Tipe entitas | Sekretariat, Bagian, Inspektorat, Badan, Dinas, Satpol, Kantor, Kecamatan, Kelurahan, RSUD |

**Edge cases yang harus ditangani:**
1. **Multi-line address**: alamat + email ada di baris berbeda dalam satu cell
2. **Multi telp**: beberapa entry punya dua nomor dipisahkan `/`
3. **Empty fields**: beberapa entry tanpa email atau tanpa no telp (mis. Kel. Proyonanggan Utara, Kec. Pecalungan)
4. **Sub-entry parent**: nomor 1 utama dan nomor 25 utama tidak punya email, tapi punya telp utama yang berlaku untuk semua sub-entry-nya
5. **Footer halaman 3**: ada tanda tangan Bupati ("an. BUPATI BATANG...") yang **harus di-skip total**

**Catatan teknis ekstraksi:**
- PyMuPDF (yang dipakai di preprocessing buku saku) **kurang bagus** untuk tabel — output text-nya bisa kacau urutan kolomnya
- **`pdfplumber.extract_tables()`** jauh lebih reliable karena pakai grid detection — cocok untuk tabel dengan border yang jelas seperti file ini
- Decision: **pakai pdfplumber** (sudah install ke venv)

---

## 2. Keputusan Struktur File

**Pisah dari preprocessing buku saku, jangan gabung.** Alasannya:

| Aspek | Buku Saku (existing) | OPD (baru) |
|---|---|---|
| Nature data | Narrative Q&A | Tabular directory |
| Chunking strategy | Q&A-aware regex split | 1 row = 1 chunk (atomic) |
| Cleaning needs | Heavy (dots, page nums, line breaks) | Minimal (parse table cells) |
| Metadata schema | section, subsection, question_number | nomor, tipe, parent_opd |

Kalau digabung di satu notebook, jadi "god notebook" yang susah di-maintain. Pisah lebih sesuai prinsip **single-responsibility**.

### Struktur File yang Diusulkan

```
notebook/
  preprocessing.ipynb            # existing → buku saku
  preprocessing_opd.ipynb        # BARU → OPD parsing
  build_vectorstore.ipynb        # existing → collection "dukcapil_qa"
  build_vectorstore_opd.ipynb    # BARU → collection "dukcapil_opd"
  rag_chat.ipynb                 # existing (nanti di-update untuk pakai 2 collections)

data/
  pdf/Nama dan Alamat OPD Kab Batang.pdf  # sudah ada
  cleaned_docs.pkl                # existing (buku saku)
  cleaned_opd_docs.pkl            # BARU (output preprocessing OPD)
  dukcapil_vector_store/          # existing dir, akan punya 2 collections di dalam
```

### Kenapa Share Folder `dukcapil_vector_store/` Tapi Collection Terpisah

Chroma mendukung multiple collections di satu `persist_directory`. Ini lebih efisien (satu Chroma client, satu folder), tapi tetap isolate collection-nya. Saat retrieval nanti di `rag_chat.ipynb`, tinggal load dua Chroma instance dengan `collection_name` berbeda.

---

## 3. Step-by-Step Plan untuk `preprocessing_opd.ipynb`

### Step 1 — Load PDF & Extract Tables
- Library: `pdfplumber`
- Loop tiap halaman → `page.extract_tables()` → return list of rows
- Concat semua tables dari 3 halaman jadi 1 list `raw_rows`
- Skip footer halaman 3 (baris yang berisi "an. BUPATI BATANG", "Sekretaris Daerah", nama, NIP)
- **Output**: list of `[nomor, nama, alamat_email, no_telp]` strings

### Step 2 — Parse Rows → Structured Records

Schema yang akan digunakan:

```python
{
  "nomor": "1" | "1.a" | "25.b",       # full hierarchical id
  "nama_opd": "Sekretariat Daerah",
  "parent_opd": None | "Sekretariat Daerah",   # for sub-entries
  "tipe": "Sekretariat" | "Bagian" | "Dinas" | "Kecamatan" | "Kelurahan" | "Badan" | "Kantor" | "RSUD" | "Inspektorat" | "Satpol",
  "alamat": "Jl. RA Kartini No. 1 Batang",
  "email": "bag_pemerintahan@batangkab.go.id" | None,
  "no_telp": ["(0285) 392729", "(0285) 391571"]   # list, bisa multi
}
```

**Sub-tasks:**
- Detect parent vs sub-entry: row `nomor` matches `^\d+\.$` → main entry; matches `^[a-z]\.$` → sub-entry (inherit `parent_opd` dari baris main sebelumnya)
- Parse `alamat_email` cell: split by newline → first non-empty line = alamat, line yang match pattern `@batangkab.go.id` = email
- Parse `no_telp`: split by `/` jika ada → list of normalized phone strings
- Infer `tipe` dari kata pertama `nama_opd` (keyword match: "Sekretariat", "Bagian", "Dinas", "Kecamatan", "Kelurahan", "Badan", "Kantor", "RSUD", "Inspektorat", "Satpol")

### Step 3 — Quality Validation

Run automated checks:
- Assert total record count masuk akal (60-65)
- Assert nomor 1–43 semua ada (no missing)
- Print distribusi by `tipe`
- Count entry dengan email kosong dan no_telp kosong (sebagai sanity, bukan error — memang ada yang kosong di sumbernya)
- Flag warning untuk row dengan karakter aneh atau field yang terlalu pendek/panjang

### Step 4 — Convert ke LangChain Documents

Format `page_content` setiap Document agar embedding-friendly:

```
Nama OPD: Bagian Pemerintahan
Bagian dari: Sekretariat Daerah
Tipe: Bagian
Alamat: Jl. RA Kartini No. 1 Batang
Email: bag_pemerintahan@batangkab.go.id
No. Telp: (0285) 392729, (0285) 391571
```

**Kenapa format ini:**
- **Self-contained**: tiap chunk berdiri sendiri tanpa butuh konteks chunk lain
- **Keyword-rich**: nama OPD, jenis, alamat — semua searchable oleh BM25 di V3
- **Semantic-friendly**: prose-like (bukan CSV) → embedding model paham
- **Parent context**: sub-entry tetap bisa di-retrieve walau user query nama parent OPD

**Metadata setiap Document:**

```python
{
  "source": "Nama dan Alamat OPD Kab Batang.pdf",
  "doc_type": "opd_directory",          # ← pembeda dari buku saku
  "nomor": "1.a",
  "nama_opd": "Bagian Pemerintahan",
  "parent_opd": "Sekretariat Daerah",
  "tipe": "Bagian",
  "has_email": True,
  "has_telp": True
}
```

> **Catatan format `no_telp`**: di `page_content` dirender sebagai string `"telp1, telp2"` (embedding-friendly). Di metadata structured asli (list) tidak disimpan karena Chroma tidak support list values di metadata — kalau perlu, simpan sebagai joined string.

### Step 5 — Export ke Pickle
- Save `cleaned_opd_docs` ke `data/cleaned_opd_docs.pkl`
- Mengikuti pola yang sama dengan buku saku
- Ready untuk dikonsumsi `build_vectorstore_opd.ipynb`

### Step 6 — Manual Quality Verification (untuk user)

Cell terakhir menampilkan **semua Documents** dalam format yang gampang dibandingkan dengan PDF aslinya:

```
[1] Sekretariat Daerah (Sekretariat)
    Alamat : Jl. RA Kartini No. 1 Batang
    Email  : -
    Telp   : (0285) 391571
─────────────
[1.a] Bagian Pemerintahan (Bagian)
    Parent : Sekretariat Daerah
    Alamat : Jl. RA Kartini No. 1 Batang
    Email  : bag_pemerintahan@batangkab.go.id
    Telp   : (0285) 392729, (0285) 391571
─────────────
...dst...
```

Plus **statistics summary cell**:
- Total Documents
- Distribusi by `tipe`
- Count entries missing email / telp
- Sample 3 Documents lengkap dengan metadata

Dengan ini bisa scroll cepat dan cross-check vs PDF mana yang salah parse.

---

## 4. Prinsip Coding yang Diterapkan

| Prinsip | Implementasi |
|---|---|
| **Modular** | Preprocessing OPD pisah dari buku saku karena beda nature data |
| **Reusable** | Schema metadata konsisten dengan buku saku (`source`, `doc_type`) supaya `rag_chat.ipynb` bisa pakai retriever yang sama tanpa rewrite besar |
| **Scalable** | Kalau nanti ada PDF directory dari kabupaten lain, tinggal duplicate notebook ini, ganti input path. Parser pakai pdfplumber yang generik untuk tabel apa saja |
| **Verifiable** | Step 6 explicit untuk human-eye QA sebelum embedding |
| **Idempotent** | Pickle output bisa re-generate kapan aja tanpa side effect |

---

## 5. Dependency

Sebelum jalanin notebook, pastikan:

```powershell
.venv\Scripts\pip install pdfplumber
```

(Saat ini `pdfplumber 0.11.9` ter-install di system Python, tapi belum di venv `.venv` yang dipakai notebook kernel.)

---

## 6. Out of Scope (Untuk Tahap Berikutnya)

Yang **TIDAK** dilakukan di notebook ini, akan dikerjakan di notebook lain:

- **Embedding & vector store build** → `build_vectorstore_opd.ipynb`
- **Update `rag_chat.ipynb`** untuk pakai 2 collections (search dukcapil_qa + dukcapil_opd lalu merge)
- **Text-to-SQL** untuk OPD → ditunda ke fase agentic RAG (nanti akan jadi tool terpisah, bukan menggantikan vector approach ini)

---

## 7. Hal yang Sudah Dikonfirmasi User

- pdfplumber dipilih (bukan PyMuPDF + regex)
- Footer halaman 3 (tanda tangan Bupati) di-skip total
- pdfplumber sudah di-install (perlu verifikasi ulang ke venv)
