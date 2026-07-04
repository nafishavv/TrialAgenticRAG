# Laporan Data, Chunking & Rekomendasi Evaluasi

**Proyek:** RAG Layanan Publik Kabupaten Batang
**Tanggal:** 2026-07-05
**Status dokumen:** sumber data ter-update (menggantikan angka `PROGRESS.md` yang stale). Evaluasi 3 mode **sudah dijalankan** — ringkasan di §7, analisis penuh di [`docs/EVAL_REPORT.md`](EVAL_REPORT.md).

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
  - **Panjang & banyak** (sosial, 2.433 chunk = ~91% total): chunk per pasal/penjelasan/naratif → dominan di index, berpotensi mendominasi hasil retrieval lintas domain bila tidak ada routing.
- **Implikasi retrieval:** karena volume sosial jauh lebih besar, **routing/intent gating** (enhanced & agentic) lebih krusial daripada di mode naive yang fan-out ke semua collection. Terbukti di eval: sosial jadi domain **paling sulit** (recall@5 0.30–0.64) sementara domain atomik (opd/perizinan) mendekati sempurna — lihat §7.

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

## 7. Status & Hasil Evaluasi

> **Evaluasi 3 mode sudah selesai dijalankan.** Bagian ini merangkum status & angka kunci; analisis lengkap (per-domain, per-query_type, catatan arsitektur, anggaran waktu) ada di **[`docs/EVAL_REPORT.md`](EVAL_REPORT.md)**.

### 7.1 Test set — synthetic, grounded, terverifikasi
Test set lama (60 soal, dukcapil+opd, banyak `TODO_FILL`) sudah **digantikan** oleh test set synthetic 4-domain yang di-generate & diverifikasi otomatis:
- **`eval/testset.json` — 198 soal**, **0 `TODO_FILL`** (full ground-truth). Distribusi route: `sosial 70`, `dukcapil 55`, `perizinan 29`, `opd 25`, `both 10`, `none 9`. Query type: lexical_exact 58, paraphrase 43, semantic 33, analytical 20, multi_chunk 16, cross_store 10, negation_edge 9, out_of_scope 9. Difficulty: easy 105 / medium 92 / hard 1.
- **`eval/intent_testset.json` — 40 soal** (≈20 valid / 20 invalid; subtype chitchat & out-of-scope).
- **Pipeline generate** (`eval/gen_testset.py` + paket `eval/generation/`): LLM hanya menulis field bahasa-natural (question/answer/facts); **route, gold_chunks, query_type, chunk_scope ditetapkan deterministik dari chunk yang dipilih** (bukan ditebak LLM) — sampling stratified per domain, dedup Jaccard antar-pertanyaan.
- **Verifikasi** (`eval/verify_testset.py`): gate keras **gold-id existence** (tiap `gold_chunks` wajib resolve ke chunk nyata di domainnya), plus cek schema, konsistensi route↔gold, dan distribusi. `gen_testset` menolak menulis file yang tak lolos verifikasi.
- **Gold-id chunk-precise, 4 domain:** dukcapil `page:<page_start>`, opd `nomor:<nomor>`, perizinan `id:<perizinan_id>`, sosial `id:<id>#pasal:<n>` / `#penjelasan:<n>` / `#preamble[.part]` / `#narr:<k>`.

### 7.2 Metrik & harness (sudah terpasang)
- `eval/run_eval.py` — retrieval + routing + answer-quality; `eval/run_intent_eval.py` — intent VALID/INVALID; `eval/eval_core.py` — metrik + LLM-judge; `eval/analyze.py` — agregasi & breakdown.
- **Retrieval:** hit@k, recall@k, precision@k, MRR. **Routing/Intent:** accuracy, confusion matrix, macro-F1, recall_valid/invalid, refusal. **Answer-quality (LLM-judge gemini temp 0):** fact_recall, faithfulness, answer_relevance, false_refusal. **Sistem:** latency per-stage, p50/p95.
- Hasil tersimpan di `eval/results/` (`summary_<mode>.json`, `per_query_<mode>.json`, `intent_<mode>.json`, `verify_report.json`).

### 7.3 Angka kunci (ringkas — detail di EVAL_REPORT.md)

| Metrik | naive | enhanced | agentic |
|---|---|---|---|
| retrieval recall@5 | **0.81** | 0.62 | 0.65 |
| routing accuracy | — | 0.72 | **0.92** |
| faithfulness | 0.86 | 0.87 | **0.95** |
| fact_recall | 0.65 | 0.64 | **0.77** |
| recall_invalid (tolak OOS) | 0.00 | **1.00** | **1.00** |
| latency total mean (s) ↓ | 6.35 | 14.57 | **5.32** |

Baca cepat: **naive** menang retrieval murni karena fan-out tapi tak bisa menolak out-of-scope; **agentic** paling seimbang (routing, answer-quality, latency terbaik); **enhanced** titik terlemah di tengah karena reranker masih stub & hybrid dormant (biaya HyDE tak terbayar). Domain **sosial** paling sulit (recall@5 0.30–0.64), domain atomik (opd/perizinan) hampir sempurna.

### 7.4 Gap/limitasi yang tersisa
1. **Answer-quality baru atas subset ~60/198** soal (rate-limit embedding free-tier); retrieval/routing/intent sudah penuh 198.
2. **Difficulty sangat timpang** — hanya **1 soal `hard`** (105 easy / 92 medium). Test set belum menantang di ujung atas.
3. **Test set synthetic** (LLM-generated, grounded + terverifikasi gold-id), belum ada subset kurasi manusia penuh sebagai anchor.
4. **28 dokumen sosial di-skip** (scan) → sebagian corpus hukum belum terwakili di index maupun eval.

---

## 8. Rekomendasi — Semua yang Bisa Ditambahkan/Dilakukan Terkait Data

Karena baseline eval sudah ada (§7), rekomendasi difokuskan pada **data**: kelengkapan, kualitas, cakupan, granularitas, freshness, dan bagaimana perbaikan data itu diukur. Diurutkan per tema, dengan **prioritas** (P1 = high-ROI/segera, P2 = menengah, P3 = lanjutan) dan **sinyal eval** yang memotivasinya.

### 8.1 Kelengkapan & cakupan corpus
- **[P1] Selamatkan 28 dokumen sosial yang di-skip** (`sosial_skipped.json`, scan/`<300` char/hal) via **OCR** (Tesseract-ind / PyMuPDF+OCR / cloud OCR). Ini bagian corpus hukum yang sekarang **tak ada di index maupun eval** — berpotensi menambah ratusan chunk pasal. Ukur dampak: recall@5 domain sosial sebelum/sesudah.
- **[P1] Integrasikan 2 PDF `unprocessed/`** (`PERDA NOMOR 1 TAHUN 2019`, `Analisis-dan-evaluasi-hukum-2025`) — masukkan ke domain sosial (tambahkan ke `metadata.json` + rebuild). Arsitektur sudah siap.
- **[P2] Perluas cakupan perizinan** — baru **34 izin** dari SIPUAS; crawl ulang untuk jenis izin yang belum tercakup (kategori usaha/lingkungan/pendidikan). Sumber tunggal & sempit = mudah out-of-coverage untuk pertanyaan warga nyata.
- **[P2] Perkaya dukcapil** — hanya 1 sumber (Buku Saku 2023). Tambah SOP/persyaratan terbaru bila ada revisi; cek apakah edisi 2024/2025 tersedia (data bisa usang).
- **[P3] Domain baru** yang sudah disinggung di roadmap: **pajak** & **hukum** umum (`sources/pajak/`, `sources/hukum/`) — 1 folder + 1 baris registry, lalu generate soal eval-nya lewat `gen_testset`.

### 8.2 Kualitas & granularitas (dipicu sinyal eval)
> Sosial adalah domain **tersulit** (recall@5 0.30–0.64) padahal ~91% index. Ini murni soal karakteristik data → berikut yang bisa dilakukan **di sisi data/chunk**:
- **[P1] Near-duplicate pasal antar-perda** menurunkan recall (pasal berbunyi mirip di banyak peraturan → embedding tumpang-tindih, chunk yang benar kalah ranking). Mitigasi data: (a) **MMR / diversity re-ranking** saat retrieve; (b) sertakan **judul+nomor+tahun peraturan di dalam `page_content`** tiap chunk pasal (bukan cuma metadata) agar embedding membawa konteks pembeda; (c) dedup chunk identik lintas dokumen.
- **[P2] Granularitas chunk-precise (pasal vs penjelasan)** kadang terlalu halus → gold pasal & penjelasan-nya bersaing. Uji **parent-document / small-to-big retrieval**: retrieve di level pasal, tapi kirim konteks selingkung (pasal + penjelasannya) ke LLM.
- **[P2] Chunk `ayat`-level** untuk pasal panjang (banyak ayat) sebagai opsi granularitas; ukur trade-off recall vs precision.
- **[P3] Tinjau dokumen `charspaced`** (text-layer rusak, 4 dok 2014–2016) secara manual sampel — pastikan rekonstruksi `fix_charspacing()` tak menelan nomor pasal; ini sumber error segmentasi paling halus.
- **[P3] Tangani tabel/lampiran** di PDF hukum & perizinan (saat ini masuk narrative/di-`_enforce_max`) — ekstraksi tabel terstruktur bisa memperbaiki jawaban syarat/biaya.

### 8.3 Metadata enrichment (buka fitur retrieval baru)
- **[P1] Filter berbasis metadata** — manfaatkan `status` (Berlaku/Tidak Berlaku), `tahun`, `bidang`, `tipe_dokumen` untuk **pre-filter** di Chroma (mis. default hanya "Berlaku", kecuali user tanya sejarah). Data & field sudah ada, tinggal dipakai di query. Kritis untuk layanan publik: **jangan jawab pakai peraturan yang sudah dicabut** tanpa peringatan.
- **[P2] Normalisasi & lengkapi metadata** — pastikan `tahun`/`nomor` konsisten (untuk sorting "peraturan terbaru"), tambah `tanggal_ditetapkan`/`tanggal_dicabut` bila tersedia di JDIH.
- **[P2] Timestamp freshness** — perizinan punya `crawl_date`; tambahkan juga untuk domain lain + tampilkan "data per <tanggal>" agar jawaban jujur soal kebaruan.

### 8.4 Freshness & maintenance data
- **[P2] Jadwalkan re-scrape** SIPUAS (perizinan) & JDIH (sosial) berkala; deteksi peraturan baru/dicabut → rebuild incremental vector store.
- **[P3] Versioning corpus** — simpan snapshot `metadata.json` + hash agar perubahan data bisa dilacak dan eval bisa dibandingkan antar-versi corpus.

### 8.5 Data untuk evaluasi (test set & judge)
- **[P1] Tambah soal `hard`** — sekarang hanya **1** dari 198. Naikkan porsi analytical/multi_chunk/negation_edge & lintas-tahun (mis. "peraturan mana yang mencabut Perda X?") lewat generator; ini yang membedakan kualitas antar-mode di ujung atas.
- **[P1] Perbesar subset answer-quality** dari ~60 → seluruh 198 saat kuota memungkinkan (atau pakai tier berbayar) agar fact_recall/faithfulness statistik solid.
- **[P2] Anchor kurasi manusia** — review manual 20–30 soal (terutama sosial) sebagai gold "emas" untuk mengkalibrasi test set synthetic & LLM-judge.
- **[P2] Perbanyak `cross_store` & `both`** (sekarang 10) — pertanyaan realistis warga sering lintas domain (mis. "izin + OPD mana yang mengurus"); ini kelemahan recall semua mode.
- **[P3] Tambah RAGAS** sebagai pembanding answer-quality (context_precision/recall, faithfulness) di atas judge custom — Gemini yang sudah terpasang bisa jadi evaluator; berguna untuk kredibilitas akademis metrik.
- **[P3] Regenerasi test set** wajib tiap kali chunking berubah (gold-id bisa bergeser): `gen_testset` → `verify_testset` (gate gold-id existence) sudah mengotomasi ini.

### 8.6 Cara menjalankan / mengukur ulang (setelah perbaikan data)

```bash
# rebuild index setelah data/chunk berubah
python scripts/preprocess.py --source <domain>
python scripts/build_vectorstore.py --source <domain>
# regenerate + verifikasi test set (gold-id bisa bergeser)
python -m eval.gen_testset --domain <domain> --fresh
python -m eval.verify_testset
# re-run eval 3 mode + breakdown, bandingkan delta vs EVAL_REPORT.md
python -m eval.run_eval --systems naive enhanced agentic --k 5 --no-judge --sleep 2
python -m eval.analyze --systems naive enhanced agentic \
    --breakdown expected_route difficulty query_type --save
```

Fokus baca **delta recall@5 per domain** (khususnya sosial) sebelum/sesudah tiap perubahan data — itu ukuran ROI paling langsung. Anggaran waktu 1 siklus ~2.5–3 jam free-tier (rincian di [EVAL_REPORT.md §9](EVAL_REPORT.md)).

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
- Harness: `eval/{run_eval,run_intent_eval,eval_core,analyze}.py`.
- Generasi test set: `eval/gen_testset.py` + paket `eval/generation/{chunks,generators,dedup,llm_client,prompts,schema}.py`; validasi: `eval/verify_testset.py`.
- Data & hasil: `eval/testset.json` (198 soal), `eval/intent_testset.json` (40), `eval/results/` (`summary_*`, `per_query_*`, `intent_*`, `verify_report.json`).
- Laporan hasil: [`docs/EVAL_REPORT.md`](EVAL_REPORT.md).

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
