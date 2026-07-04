# Laporan Data, Chunking & Rekomendasi Evaluasi

**Proyek:** RAG Layanan Publik Kabupaten Batang
**Tanggal:** 2026-06-30
**Status dokumen:** sumber data ter-update (menggantikan angka `PROGRESS.md` yang stale)

> Dokumen ini punya dua tujuan: **(1)** laporan progres data — apa yang dipakai, berapa banyak, karakteristik, jumlah chunk, dan strategi chunking; **(2)** bahan **meninjau ulang rencana evaluasi** — framework & metrik yang sebaiknya dipakai dan cara menjalankannya. Semua angka chunk diverifikasi langsung dari `data/vector_stores/<domain>/chroma.sqlite3` (`count(*) from embeddings`), bukan estimasi.

---

## 1. Ringkasan Eksekutif

Sistem RAG untuk pelayanan publik Kab. Batang, berbahasa **Indonesia**, mencakup **4 domain**. Pipeline: `raw → preprocess → chunk → embed → Chroma (per-domain)`, lalu dilayani oleh **3 mode** RAG (naive / enhanced / agentic) dengan kontrak output seragam `RagResult`.

- **Embedding:** `models/gemini-embedding-2` (768-dim, multilingual), `task_type` `retrieval_document` saat indexing & `retrieval_query` saat query.
- **Generation LLM:** `gemini-2.5-flash` (temperature 0.1).
- **Vector store:** Chroma, satu collection per domain.

| Domain | Sumber raw | Cleaned docs | **Vector chunks** | Strategi chunking |
|---|---|---|---|---|
| dukcapil | 1 PDF (282 hal) | 258 | **150** | Q&A-aware + narrative recursive |
| opd | 1 PDF (3 hal) | 61 | **61** | identity (1 OPD = 1 chunk) |
| perizinan | 1 JSON (34 izin) | 34 | **34** | identity (1 izin = 1 chunk) |
| sosial | 79 PDF (51 terpakai, 28 di-skip) | 1.132 | **2.433** | hybrid pasal-split / narrative / penjelasan |
| **Total** | **~82 file** | **1.485** | **2.678** | |

**Catatan penting untuk progres:** domain **perizinan & sosial sudah end-to-end** (data → chunk → vector store), tetapi **belum tercakup di testset evaluasi** (lihat §7). `PROGRESS.md` masih mencatat 211 vektor (dukcapil+opd saja) — angka aktual sekarang **2.678 vektor**.

> **Update 2026-06-30:** domain sosial di-remediasi total — lihat §4 untuk detail.

---

## 2. Sumber Data per Domain

### 2.1 dukcapil — Administrasi Kependudukan
- **Raw:** `data/raw/dukcapil/Buku-Saku-Dafduk-Capil-2023.pdf` (1,5 MB, 282 halaman, PDF native).
- **Jenis konten:** buku saku FAQ pendaftaran penduduk & pencatatan sipil (KTP, KK, akta lahir/mati/nikah, prosedur administrasi).
- **Record:** 258 cleaned docs (1 halaman = 1 doc, setelah filter cover & daftar isi).
- **Metadata:** `source`, `page`, `section` (Kata Pengantar / BAB I–V), `total_pages`, `format`, `creator`, `creationdate`.
- **Contoh isi:** pasangan tanya–jawab bernomor, mis. *"40. Apakah pengurusan KTP-el dipungut biaya? ..."*.

### 2.2 opd — Organisasi Perangkat Daerah (Direktori)
- **Raw:** `data/raw/opd/Nama dan Alamat OPD Kab Batang.pdf` (111 KB, 3 halaman, tabular).
- **Jenis konten:** direktori kontak instansi (nama OPD, induk, tipe, alamat, email, telepon).
- **Record:** 61 (1 OPD = 1 record atomic).
- **Metadata:** `nomor`, `nama_opd`, `parent_opd`, `tipe`, `alamat`, `email`, `no_telp`, `has_email`, `has_telp`, `page`, `doc_type=opd_directory`.
- **Contoh record:**
  ```
  Nama OPD: Bagian Pemerintahan
  Bagian dari: Sekretariat Daerah | Tipe: Bagian
  Alamat: Jl. RA Kartini No. 1 Batang
  Email: bag_pemerintahan@batangkab.go.id | Telp: (0285) 392729
  ```

### 2.3 perizinan — Layanan Perizinan (SIPUAS)
- **Raw:** `data/raw/perizinan/perizinan_data.json` (68 KB, hasil crawl SIPUAS).
- **Jenis konten:** prosedur & syarat izin (34 jenis izin: kesehatan, reklame, dll).
- **Record:** 34 (1 izin = 1 record self-contained: persyaratan + mekanisme + dasar hukum + keterangan).
- **Metadata:** `perizinan_id`, `nama_perizinan`, `kategori`, `url`, `crawl_date`, `doc_type`.
- **Struktur tiap izin:** `=== PERSYARATAN ===`, `=== MEKANISME PELAYANAN ===`, `=== DASAR HUKUM ===`, `=== KETERANGAN ===` (estimasi waktu, masa berlaku, biaya).

### 2.4 sosial — Produk Hukum (JDIH)
- **Raw:** 79 PDF di `data/raw/sosial/` (~53 MB) + `metadata.json`. Tipe: Peraturan Daerah, Peraturan/Keputusan/Instruksi Bupati, Surat Edaran, Naskah Akademis, dll.
- **Jenis konten:** produk hukum bidang sosial (perlindungan anak, KDRT, ketenagakerjaan, program sosial), rentang tahun lebar (~1989–2025).
- **Record:** 1.132 cleaned docs dari **51 PDF** yang lolos filter; **28 PDF di-skip** (`total_skipped: 28` di `data/processed/sosial_skipped.json`) karena hasil scan / OCR minim (`avg_chars_per_page < 300`).
- **Metadata:** `id`, `nomor`, `judul`, `tahun`, `tipe_dokumen`, `status`, `bidang`, `opd_pemrakarsa`, `source_url`, `charspaced` (flag text-layer rusak).
- **Remediasi 2026-06-30:** 4 dokumen vintage 2014–2016 memiliki text-layer rusak (tiap karakter ter-spasi, "P a s a l"). `fix_charspacing()` diperbaiki ke rekonstruksi level-halaman sehingga angka pasal multi-digit terbaca benar. Deteksi pasal diperkuat dengan guard MEMUTUSKAN (menolak sitasi di Mengingat), section-aware split (Batang Tubuh vs Penjelasan), gate kredibilitas `_is_pasal_doc()`, dan gold-id chunk-precise (`sosial:id:<id>#pasal:<n>`).

### 2.5 Belum diintegrasi
- `data/raw/unprocessed/PERDA NOMOR 1 TAHUN 2019.pdf` (275 KB)
- `data/raw/unprocessed/Analisis-dan-evaluasi-hukum-no-3-tahun-2025.pdf` (449 KB)

---

## 3. Pipeline Preprocessing

Alur: `data/raw/<domain>/ → src/ragtrial/sources/<domain>/preprocess.py → data/processed/<domain>.pkl (+ .json)`. Entry point: `scripts/preprocess.py --source <domain|all>`.

| Domain | Langkah utama | File |
|---|---|---|
| dukcapil | Filter halaman (buang cover & TOC, sisa 258); normalisasi teks (hapus titik-titik TOC, nomor halaman lepas, sambung hard-wrap, rapikan whitespace); tag `section` per halaman | `sources/dukcapil/preprocess.py` |
| opd | Ekstraksi tabel (pdfplumber); merge baris kelanjutan antar halaman; parse email/telp via regex; infer `tipe`; render record terstruktur | `sources/opd/preprocess.py` |
| perizinan | Load JSON; normalisasi whitespace & `\r\n`; render 4 seksi jadi satu `page_content` terstruktur | `sources/perizinan/preprocess.py` |
| sosial | Ingest dari `metadata.json`; **usability filter** (skip jika <300 char/hal → log ke `sosial_skipped.json`); **de-spacing** (`P a s a l` → `Pasal`); cleaning sama seperti dukcapil | `sources/sosial/preprocess.py` |

---

## 4. Strategi Chunking

Chunking terjadi saat build vector store: `scripts/build_vectorstore.py` → `sources/<domain>/chunk.py` → `vectorstore/builder.py` (embed batch 40, exponential backoff untuk rate limit).

**Parameter umum (untuk mode recursive):** `chunk_size=1200`, `overlap=200`, `separators=["\n\n","\n",". "," ",""]`, `min_chunk=50` char.

| Domain | Strategi | Alasan desain | Hasil |
|---|---|---|---|
| **dukcapil** | **Q&A-aware** untuk BAB II (regex `nomor + kata-tanya` → 1 Q&A = 1 chunk) + **narrative recursive** untuk bagian naratif | Menjaga keutuhan pasangan tanya–jawab; bagian naratif dipotong semantik | 258 docs → **150 chunks**; metadata `chunk_type` (`qa`/`narrative`), `question_number`, `page_start/end` |
| **opd** | **Identity** (tanpa split) | Tiap OPD sudah atomic & pendek; split malah memecah entitas | 61 → **61** |
| **perizinan** | **Identity** | Tiap izin self-contained (semua seksi dalam 1 doc); retrieval butuh konteks utuh | 34 → **34** |
| **sosial** | **Hybrid per-dokumen:** *pasal-split* bila memenuhi gate kredibilitas (`_is_pasal_doc`), selain itu *narrative recursive*; **section-aware**: Batang Tubuh & Penjelasan dipisah (`chunk_type` `pasal`/`penjelasan_pasal`); safeguard chunk >6000 char di-split ulang | Dokumen hukum terstruktur per pasal → unit retrieval & sitasi alami; penomoran pasal reset di bagian Penjelasan ditangani lewat section split; dokumen non-pasal (artikel, naskah, perubahan) pakai recursive | 1.132 docs → **2.433** chunks; metadata `chunk_type` (`pasal`/`penjelasan_pasal`/`preamble`/`penjelasan_preamble`/`narrative`), `pasal_number`, + seluruh metadata dokumen diwariskan |

Pewarisan metadata per chunk (page/section/pasal/nomor/judul) penting untuk **sitasi** di jawaban.

---

## 5. Karakteristik Data (Lintas Domain)

- **Bahasa:** Indonesia seluruhnya; istilah administratif/hukum formal.
- **Cakupan:** geografis Kab. Batang, Jawa Tengah; temporal terlebar di sosial (~1989–2025).
- **Heterogenitas format:** PDF native (dukcapil, opd, sebagian sosial), JSON terstruktur (perizinan), PDF scan/parsial (28 dokumen sosial — sengaja di-skip agar tidak mengotori index).
- **Distribusi ukuran chunk sangat timpang antar domain:**
  - **Atomic & pendek** (opd, perizinan): 1 record = 1 chunk, kaya metadata → cocok untuk *lookup* eksak.
  - **Sedang** (dukcapil): chunk per Q&A.
  - **Panjang & banyak** (sosial, 2.283 chunk = ~90% total): chunk per pasal/naratif → dominan di index, berpotensi mendominasi hasil retrieval lintas domain bila tidak ada routing.
- **Implikasi retrieval:** karena volume sosial jauh lebih besar, **routing/intent gating** (enhanced & agentic) lebih krusial daripada di mode naive yang fan-out ke semua collection.

---

## 6. Arsitektur 3 Mode yang Dibandingkan

Inti eksperimen: membandingkan tiga arsitektur RAG di atas **data & embedding yang identik**, dengan kontrak output seragam `RagResult` (`answer`, `documents`, `route`, `timings`, `meta`). Karena variabel data dikontrol, perbedaan hasil murni dari **arsitektur**. **Sumbu pembeda utamanya = siapa yang mengontrol alur** (developer vs LLM vs tidak ada).

| Aspek | **naive** | **enhanced** | **agentic** |
|---|---|---|---|
| Kontrol alur | Tetap, minimal | **Developer** (config) | **LLM** (otonom) |
| Intent gating | ❌ selalu retrieve | ✅ semantic gate VALID/INVALID | ✅ LLM putuskan perlu/tidak |
| Routing domain | ❌ fan-out ke semua | ✅ SemanticRouter (atau LLMRouter) | ✅ implisit lewat pilihan tool |
| Query rewrite | ❌ | ✅ HyDE (default) | ✅ LLM tulis ulang query di tool call |
| Iterasi/retry retrieval | ❌ sekali | ❌ sekali (pipeline lurus) | ✅ loop, bisa re-search (`MAX_ITERATIONS=5`) |
| Reranker | ❌ | opsional (stub CrossEncoder) | ❌ |
| Jumlah LLM call | 1 (generate) | 1–2 (intent/HyDE + generate) | 2–N (tiap putaran agent) |
| Implementasi | `rag/naive.py` | `rag/enhanced.py` + `pipeline/` | `rag/agentic.py` (LangGraph) |

### 6.1 naive — baseline jujur
Fan-out dense similarity ke **semua** collection per-domain → merge global top-k → `PROMPT_NAIVE` (stuff polos tanpa header per-source) → **1 LLM call**. Tanpa routing, intent, rewrite, atau rerank. Selalu meng-retrieve (bahkan untuk chit-chat). Gunanya: **batas bawah** yang jujur untuk mengukur nilai tambah dua mode lain.

### 6.2 enhanced — pipeline terkontrol developer
Rangkaian stage komposabel: **`intent → router → rewriter → retrieve → reranker → generate`**, tiap stage swappable lewat `EnhancedRAGConfig`. Default kanonik:
```python
EnhancedRAGConfig(intent="semantic", router="semantic", rewriter="hyde",
                  retrieval="dense", reranker="none", k=5, k_per_source=4)
```
- **Intent stage:** `semantic_router` (encoder Gemini) klasifikasi VALID/INVALID; INVALID → lewati retrieval, jawab via `PROMPT_INVALID`. Fail-open (no-match → VALID).
- **Router:** SemanticRouter (cosine query vs centroid per-capability; threshold 0.65; margin 0.04 → "both"; di bawah threshold → "none"). Alternatif `LLMRouter` untuk ablation.
- **Rewriter:** HyDE (default) / passthrough; MultiQuery & CrossEncoderReranker masih stub.
- Cocok untuk **ablation sistematis** (nyalakan/matikan tiap komponen, sweep config).

### 6.3 agentic — orkestrasi oleh LLM
State machine LangGraph **`agent ⇄ tools`**. LLM dibekali satu tool `search_<domain>` per capability dan **memutuskan sendiri**: retrieve atau tidak (intent implisit), domain mana (routing implisit), tulis ulang query, dan **iterasi** (re-search bila hasil lemah) sampai `MAX_ITERATIONS=5`. Tiap langkah dicatat di `meta.steps` (`{tool, query, n_docs}`). Paling fleksibel untuk pertanyaan multi-domain/kompleks, tapi paling mahal (banyak LLM call) dan paling sulit diprediksi.

---

## 7. Status Evaluasi Saat Ini

Framework eval sudah ada dan cukup matang secara kode, tapi **datasetnya tertinggal di belakang data**.

**Komponen yang sudah ada:**
- `eval/run_eval.py` — eval RAG penuh (retrieval + routing + answer quality) untuk ketiga sistem.
- `eval/run_intent_eval.py` — eval klasifikasi intent VALID/INVALID.
- `eval/eval_core.py` — metrik + LLM-as-judge.
- `eval/analyze.py` — agregasi & breakdown (by query_type / difficulty / route).

**Metrik yang sudah terpasang:**
- **Retrieval:** `hit@k`, `recall@k`, `precision@k`, `MRR` (gold-id dinormalisasi via `Capability.gold_id()`).
- **Routing/Intent:** accuracy, confusion matrix, macro-F1, recall_valid/recall_invalid.
- **Answer quality (LLM judge, `gemini` temp 0):** fact recall, faithfulness (0/1/2), answer relevance (0/1/2), refusal.
- **Sistem:** timing per-stage (route/retrieve/generate/total), p50/p95.

**Testset saat ini:**
- `eval/testset.json` — **60 pertanyaan**. Distribusi route: `dukcapil 22`, `opd 18`, `both 14`, `none 6`. Query type: lexical_exact 18, cross_store 14, paraphrase 10, semantic 7, out_of_scope 6, negation_edge 2, multi_chunk 2, analytical 1. Difficulty: easy 21 / medium 25 / hard 14.
- `eval/intent_testset.json` — **40 pertanyaan** (≈20 valid / 20 invalid; subtype chitchat & out-of-scope).

**Gap yang harus disorot (penting untuk peninjauan):**
1. **Testset hanya mencakup dukcapil & opd.** Tidak ada satu pun pertanyaan ber-route `perizinan` atau `sosial`, padahal keduanya sudah live dan sosial = ~90% index. Evaluasi sekarang **tidak mengukur 2 dari 4 domain**.
2. **37 dari 60** pertanyaan masih `TODO_FILL` pada `expected_answer`/`expected_facts` → metrik answer-quality (fact recall) belum bisa dihitung untuk mayoritas soal.
3. **Format gold_id** — dukcapil (`page:<page_start>`), opd (`nomor:<nomor>`), perizinan (`perizinan:id:<perizinan_id>`), dan sosial (`sosial:id:<id>#pasal:<n>` / `#penjelasan:<n>` / `#preamble` / `#narr:<k>`) sudah chunk-precise. Retrieval metric bisa dihitung untuk semua 4 domain.
4. **Testset kecil & timpang** (60 soal, condong dukcapil/opd, hanya 1 soal analytical).

---

## 8. Rekomendasi Evaluasi (Inti Peninjauan)

### 8.1 Framework: custom harness **+** RAGAS (komplementer, bukan saling ganti)

| Lapisan | Pakai | Alasan |
|---|---|---|
| Retrieval, routing, intent | **Custom harness (sudah ada)** | Sudah ground-truth–aware (gold_id, expected_route, expected_intent). RAGAS lemah di sini karena butuh label eksplisit yang sudah kita punya. |
| Answer quality berskala | **RAGAS** | Metrik `faithfulness`, `answer_relevancy`, `context_precision`, `context_recall` sudah baku & reference-free sebagian → mengurangi beban mengisi `expected_facts` manual. RAGAS bisa pakai LLM + embeddings **Gemini yang sudah terpasang** (tinggal set sebagai evaluator LLM/embeddings). |

Rekomendasi: pertahankan judge custom yang sudah ada sebagai pembanding, tapi tambahkan RAGAS untuk skala & kredibilitas akademis (metrik yang dikenal luas).

### 8.2 Metrik yang direkomendasikan, per lapisan

- **Retrieval:** `recall@k` (utama, k=5), `MRR`, `context_precision` (RAGAS). Target rujukan awal: recall@5 ≥ 0.80.
- **Routing & Intent:** accuracy + **macro-F1** (kelas tak seimbang), **refusal rate** untuk soal out-of-scope (route `none`). Target: macro-F1 ≥ 0.85; refusal benar pada ≥ 0.9 soal OOS.
- **Generation:** **faithfulness/groundedness** (anti-halusinasi — paling kritis untuk layanan publik), **answer relevance**, **fact recall**, dan **citation correctness** (apakah pasal/halaman yang disebut benar). Target faithfulness ≥ 1.5/2.
- **Sistem:** latency p50/p95 per mode, dan (bila memungkinkan) token/biaya per query — relevan membandingkan agentic (multi-call) vs naive.

> Target di atas adalah **rujukan awal**, bukan patokan keras — kalibrasi ulang setelah baseline pertama.

### 8.3 Prioritas perbaikan testset (urut kepentingan)

1. **Tambah soal perizinan & sosial** ke `testset.json` (mis. +15 per domain, beragam query_type) — ini blocker utama; tanpa ini 2 domain tak terukur.
2. ~~**Finalisasi gold_id** perizinan & sosial~~ — **SELESAI** (2026-06-30): gold-id chunk-precise sudah aktif untuk semua 4 domain. Sosial: `sosial:id:<id>#pasal:<n>` / `#penjelasan:<n>` / `#preamble` / `#narr:<k>`.
3. **Isi 37 `TODO_FILL`** `expected_answer`/`expected_facts` — atau alihkan sebagian beban ke RAGAS reference-free.
4. **Seimbangkan distribusi** per domain & query_type; tambah kasus negatif/out-of-scope lintas domain (mis. tanya sosial saat hanya ada di perizinan).

### 8.4 Cara menjalankan (alur yang disarankan)

```bash
# 1. Baseline kuantitatif 3 mode
python -m eval.run_eval --systems naive enhanced agentic --k 5
# 2. Eval intent terpisah
python -m eval.run_intent_eval --systems naive enhanced agentic
# 3. Analisis + breakdown
python -m eval.analyze --systems naive enhanced agentic --breakdown query_type difficulty --save
```

Baca hasil sebagai perbandingan **naive (baseline jujur) vs enhanced (pipeline terkontrol) vs agentic (otonom)**, dipecah per `query_type`, `difficulty`, dan `expected_route`. Setelah testset perizinan/sosial masuk, jalankan ulang untuk dapat gambaran 4 domain penuh.

---

## 9. Lampiran — Peta File & Reproduksi

**Entry points**
- `scripts/preprocess.py` — raw → cleaned pickle.
- `scripts/build_vectorstore.py` — chunk + embed → Chroma.
- `scripts/ask.py` — query CLI (`--mode naive|enhanced|agentic`).
- `app.py` — UI Streamlit.

**Kode inti**
- `src/ragtrial/sources/<domain>/{preprocess,chunk,capability}.py` — logika per domain.
- `src/ragtrial/rag/{naive,enhanced,agentic}.py` — 3 mode; `result.py` — kontrak `RagResult`.
- `src/ragtrial/pipeline/{intent,route}.py` — intent gate + semantic/LLM router.
- `src/ragtrial/vectorstore/builder.py`, `src/ragtrial/llm.py` — Chroma builder & konfigurasi Gemini.

**Eval**
- `eval/{run_eval,run_intent_eval,eval_core,analyze}.py`, `eval/testset.json`, `eval/intent_testset.json`, `eval/results/`.

**Reproduksi vector store (contoh sosial)**
```bash
python scripts/preprocess.py --source sosial
python scripts/build_vectorstore.py --source sosial
```

**Verifikasi jumlah chunk**
```python
import sqlite3
sqlite3.connect("data/vector_stores/sosial/chroma.sqlite3")\
    .execute("select count(*) from embeddings").fetchone()  # → 2433
```
