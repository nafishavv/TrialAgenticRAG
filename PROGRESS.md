# Progress Report — RAG Dukcapil & OPD Kab Batang

**Update terakhir**: 2026-05-19 (sore: naive combined RAG + 4-pipeline comparison)

---

## 1. Data Sources

| File | Status | Notes |
|---|---|---|
| `data/dukcapil_pdf/Buku-Saku-Dafduk-Capil-2023.pdf` (282 hal) | ✅ Done preprocessing & vector store | Buku saku Q&A administrasi kependudukan |
| `data/pdf/Nama dan Alamat OPD Kab Batang.pdf` (3 hal) | ✅ Done preprocessing, ⏳ belum vectorize & chat | Direktori OPD (61 records) |
| `data/pdf/PERDA NOMOR 1 TAHUN 2019.pdf` | ❌ Belum disentuh | Belum diputuskan apakah masuk pipeline RAG ini |
| `data/pdf/Analisis-dan-evaluasi-hukum-no-3-tahun-2025.pdf` | ❌ Belum disentuh | Belum diputuskan apakah masuk pipeline RAG ini |

---

## 2. Yang Sudah Dikerjakan

### A. Pipeline Dukcapil (Buku Saku) — END-TO-END JALAN

**Preprocessing** ([notebook/preprocessing.ipynb](notebook/preprocessing.ipynb))
- Load 282 halaman PDF via PyMuPDFLoader
- Filter halaman: buang cover scan (hal 0, 281) & daftar isi (hal 3–24) → 258 halaman
- Cleaning: hapus deretan titik daftar isi, nomor halaman standalone, normalisasi whitespace, gabung baris yang terpotong word-wrap PDF
- Tag section metadata: `Kata Pengantar` / `BAB I - Pendahuluan` / `BAB II - Pertanyaan dan Jawaban` / `BAB III - Penutup`
- Output: `data/cleaned_docs.pkl` (258 docs) + `data/cleaned_docs.json` (untuk inspeksi manual, baru ditambah)

**Build Vector Store** ([notebook/build_vectorstore.ipynb](notebook/build_vectorstore.ipynb))
- Chunking strategy 2-track:
  - **BAB II (Q&A)**: regex-based split per pertanyaan → tiap chunk = 1 Q+A utuh (139 chunks)
  - **BAB I, III, Kata Pengantar**: `RecursiveCharacterTextSplitter` (chunk_size=1200, overlap=200) → 11 chunks
- **Total: 150 chunks**
- Embedding: Gemini `models/gemini-embedding-2`, `output_dimensionality=768`, `task_type="retrieval_document"`
- Vector store: ChromaDB di `data/dukcapil_vector_store/`, collection `dukcapil_qa`
- Built dengan batch=40 + retry exponential backoff untuk handle rate limit Gemini

**RAG Chat & Eksperimen** ([notebook/rag_chat.ipynb](notebook/rag_chat.ipynb))
- Implementasi 4 retrieval variants:
  - **V1 — Naive Dense**: similarity search top-k
  - **V2 — Dense + Reranker (bge-reranker-v2-m3)**: fetch top-20 → rerank → top-k
  - **V3 — Hybrid (BM25 + Dense, RRF)**: `EnsembleRetriever`
  - **V4 — Hybrid + Reranker**: ensemble fetch top-20 → rerank → top-k
- Prompt template grounded ke konteks (refuse answer kalau info tidak ada di buku, format `[Sumber: <section>, hal <page>]`)
- LLM: Gemini `gemini-2.5-flash`, temperature=0.1
- Test harness: 6 query (literal/paraphrase/narrative/out-of-scope) × 4 variant — sudah ada output retrieval inspection + latency benchmark

**Hasil benchmark latency (6 query × 4 variant):**

| Variant | Retrieval avg | Total avg | Catatan |
|---|---|---|---|
| V1 (Naive) | ~0.7s | ~3.8s | Cepat, baseline reliable |
| V2 (+Rerank) | ~70s | ~73s | Reranker bottleneck — 25min total |
| V3 (Hybrid) | ~0.7s | ~4.4s | Hampir secepat V1, cakupan lebih luas |
| V4 (Hyb+Rerank) | ~228s | ~232s | Slowest, semua sample konsisten |

### B. Pipeline OPD — Preprocessing Done, Sisanya Baru Saja Disiapkan

**Preprocessing** ([notebook/preprocessing_opd.ipynb](notebook/preprocessing_opd.ipynb))
- Extract tabel pakai `pdfplumber.extract_tables()` (table-aware, beda dengan PyMuPDFLoader yang text-only)
- Handle edge cases yang ditemukan:
  - Bullet character `` di kolom alamat/email
  - Sub-entry pattern `a. Bagian Pemerintahan` (parent: Sekretariat Daerah)
  - Continuation row cross-page (entry #12 "Dinas Pariwisata, Kepemudaan dan…" terpotong dari hal 1 → hal 2 "…Olahraga")
- Parse jadi structured records dengan schema: `{nomor, nama_opd, parent_opd, tipe, alamat, email, no_telp, page}`
- Quality validation: assert main entries 1–43 lengkap, distribusi tipe (Dinas: 17, Kecamatan: 15, Bagian: 9, Kelurahan: 9, dst)
- Output: `data/cleaned_opd_docs.pkl` (61 docs) + `data/cleaned_opd_docs.json` (baru ditambah)

**Build Vector Store** ([notebook/build_vectorstore_opd.ipynb](notebook/build_vectorstore_opd.ipynb)) — siap dijalankan
- Tidak perlu chunking (1 record = 1 Document atomic, ~157 chars rata-rata)
- 61 docs cukup 1 batch (jauh di bawah rate limit 100/menit)
- Output: `data/opd_vector_store/`, collection `opd_directory`

**RAG Chat OPD** ([notebook/rag_chat_opd.ipynb](notebook/rag_chat_opd.ipynb)) — siap dijalankan
- 3 retrieval variants (reranker diskip karena 61 docs terlalu kecil):
  - V1 Dense, V2 BM25, V3 Hybrid
- Prompt template adaptasi domain direktori
- Test harness 6 query termasuk paraphrase test (`"Nomor telp Disdukcapil"` vs nama lengkap `"Dinas Kependudukan dan Pencatatan Sipil"`)

### C. Pipeline Combined (Naive Unified RAG) — END-TO-END JALAN

**Module** ([notebook/rag_chat_main.py](notebook/rag_chat_main.py))
- Query DUA vector store sekaligus dalam 1 pass, tanpa router LLM call
- Hybrid retrieval per store (BM25 + dense, RRF, weights 0.5/0.5)
- `k_per_store=4` → total 8 docs (match agentic-both 4+4=8 supaya apple-to-apple)
- Tag tiap doc dgn `metadata._source` (`dukcapil` / `opd`)
- `PROMPT_COMBINED` generic — tidak force struktur 2-bagian (bisa handle query single-domain juga)
- Entry point: `ask_main(question)` → return dict {documents, answer, timings}

**Demo notebook** ([notebook/rag_chat_main.ipynb](notebook/rag_chat_main.ipynb))
- Step 1: import module
- Step 2: inspect retrieval (cek mix dukcapil/opd dalam top-4 per store)
- Step 3: smoke test 6 query (dukcapil-only / opd-only / both / off-topic)
- Step 4-5: latency summary + observasi kualitatif

**Comparison head-to-head** ([agenticrag/3-compare_agentic_vs_naive.ipynb](agenticrag/3-compare_agentic_vs_naive.ipynb))
- Sudah dirombak: dari 3 pipeline (agentic, naive_dukcapil, naive_opd) → **4 pipeline** (+ naive_combined)
- Fokus geser ke head-to-head Agentic vs Naive Combined (keduanya bisa handle semua route)
- Per-segment latency breakdown — kapan masing-masing menang
- Hipotesis utama: di dataset kecil (150+61 docs), router overhead (~2.4s) > ekstra retrieve cost combined → naive combined kemungkinan menang. Agentic baru menang besar di query `none` (skip retrieve+generate)

---

## 3. Decisions & Insights yang Sudah Diambil

| Topik | Decision | Reasoning |
|---|---|---|
| Chunking BAB II | Q&A-aware (regex per nomor pertanyaan) | Karena nature buku saku: jawaban melintasi batas halaman, tiap Q+A self-contained |
| Chunking OPD | No chunking (1 row = 1 Document) | Data directory atomic, ga ada narrative |
| Embedding | Gemini `text-embedding-2` 768d | Multilingual support (Bahasa Indonesia) |
| OPD vector store | Pisah dari Dukcapil (`opd_vector_store/`) | Nature data beda (Q&A vs directory lookup) — embedding ga ke-noise |
| Reranker | Skip untuk OPD | 61 docs terlalu kecil, latency tidak worth |
| Format export | Pickle (primary) + JSON (inspeksi) | PKL preserve Document; JSON readable buat debugging |

---

## 4. To-Do Selanjutnya

### Immediate (next session)
- [ ] **Run end-to-end OPD pipeline**: jalankan `preprocessing_opd.ipynb` (untuk generate JSON) → `build_vectorstore_opd.ipynb` → `rag_chat_opd.ipynb`
- [ ] **Evaluasi kualitatif OPD chat**: catat behavior tiap variant terutama untuk query paraphrase (`Disdukcapil` vs nama lengkap)
- [ ] **Re-run dukcapil preprocessing** untuk generate `cleaned_docs.json` (cell baru sudah ada di notebook)

### Short-term
- [x] **Unified RAG chat (naive combined)** ✅ — `notebook/rag_chat_main.ipynb` + module `notebook/rag_chat_main.py`. Versi tanpa router (query semua store sekaligus). Comparison vs agentic ada di `agenticrag/3-compare_agentic_vs_naive.ipynb`.
- [ ] **Run end-to-end comparison** — eksekusi `agenticrag/3-compare_agentic_vs_naive.ipynb` versi 4-pipeline yang sudah dirombak, catat verdict head-to-head agentic vs naive combined
- [ ] **Evaluasi kuantitatif Dukcapil 4 variants**: bikin gold-set ~20-30 Q&A dengan expected source (page/Q#), hitung Hit@k, MRR, atau Recall@k. Sekarang baru evaluasi kualitatif via sample queries.
- [ ] **Decision: 2 PDF lain** (`PERDA NOMOR 1 TAHUN 2019.pdf`, `Analisis-dan-evaluasi-hukum-no-3-tahun-2025.pdf`) — masuk pipeline atau tidak? Kalau ya, perlu preprocessing strategy baru (legal/regulation document beda dengan Q&A book & directory).

### Medium-term
- [ ] **Cleanup**: hapus `data/vector_store/` lama (sisa dari experiment awal, tidak dipakai)
- [ ] **Reranker latency investigation di Dukcapil V2/V4**: 70s–228s di V2/V4 terlalu lama untuk produksi. Cek apakah karena cold start bge-reranker, GPU/CPU, atau batch size — apa bisa di-optimize.
- [ ] **Conversation memory / multi-turn chat**: sekarang masih single-turn (query → answer). Kalau perlu follow-up question, butuh chat history handling.
- [ ] **Out-of-scope handling**: query `"Siapa presiden Indonesia 2024?"` sudah handled by prompt (return "Informasi tidak ditemukan..."), tapi perlu test lebih banyak edge case.

### Open questions untuk diskusi bimbingan
1. **Scope final RAG**: cuma 2 dokumen (Dukcapil + OPD) atau ditambah PERDA + analisis hukum?
2. **Metrik evaluasi**: cukup kualitatif (manual review) atau perlu kuantitatif (gold-set + retrieval metrics)?
3. **Output akhir**: notebook eksperimen, atau perlu jadi aplikasi (Streamlit/FastAPI)?
4. **Variant final yang dipilih untuk Dukcapil**: V3 (Hybrid) tampak paling balanced — latency mirip V1 tapi cakupan retrieval lebih luas. V2/V4 reranker overhead-nya berat. Setuju kah pakai V3 sebagai default?

---

## 5. File Map (untuk navigasi cepat)

```
RAGTrial/
├── data/
│   ├── dukcapil_pdf/Buku-Saku-Dafduk-Capil-2023.pdf
│   ├── pdf/Nama dan Alamat OPD Kab Batang.pdf
│   ├── pdf/PERDA NOMOR 1 TAHUN 2019.pdf                ⚠ belum diproses
│   ├── pdf/Analisis-dan-evaluasi-hukum-no-3-tahun-2025.pdf  ⚠ belum diproses
│   ├── cleaned_docs.pkl              ← output preprocessing.ipynb
│   ├── cleaned_docs.json             ← (akan di-generate next run)
│   ├── cleaned_opd_docs.pkl          ← output preprocessing_opd.ipynb
│   ├── cleaned_opd_docs.json         ← (akan di-generate next run)
│   ├── dukcapil_vector_store/        ← Chroma, collection: dukcapil_qa
│   └── opd_vector_store/             ← (akan di-generate next run)
└── notebook/
    ├── preprocessing.ipynb           ✅ dukcapil cleaning
    ├── preprocessing_opd.ipynb       ✅ OPD parsing
    ├── build_vectorstore.ipynb       ✅ dukcapil → Chroma
    ├── build_vectorstore_opd.ipynb   🆕 OPD → Chroma (siap run)
    ├── rag_chat.ipynb                ✅ dukcapil RAG, 4 variants
    ├── rag_chat_opd.ipynb            🆕 OPD RAG, 3 variants (siap run)
    ├── rag_chat_main.py              🆕 module: naive combined RAG (ask_main)
    └── rag_chat_main.ipynb           🆕 demo + smoke test naive combined
agenticrag/
    ├── agentic_rag.py                ✅ module: router + retrievers + LangGraph workflow
    ├── 2-agentic_router.ipynb        ✅ build & test agentic pipeline
    └── 3-compare_agentic_vs_naive.ipynb  🆕 4-pipeline comparison (agentic vs naive×3)
```
