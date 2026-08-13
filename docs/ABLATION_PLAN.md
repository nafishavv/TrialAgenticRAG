# Ablation Study — Architectural Audit & Implementation Roadmap

_Audit tanggal 2026-08-03. Basis: pembacaan langsung `src/ragtrial/**`, `eval/**`, dan hasil eval yang sudah ada (`eval/results*`, `docs/EVAL_REPORT.md`, `docs/EVAL_FINDINGS.md`). **Tidak ada fitur yang diasumsikan** — semua klaim di bawah menunjuk file/baris nyata._

**Verifikasi runtime (2026-08-04):** tiga klaim struktural paling penting di dokumen ini (pool `k_candidates` no-op, retry sleep dalam region terukur, ekuivalensi `search_all`) sudah dijalankan sungguhan, bukan cuma dibaca kodenya. Lihat blok "✅ Diverifikasi runtime" di masing-masing bagian terkait (§1.1, §6 P0, §6 R2).

---

## 0. Ringkasan eksekutif

| Temuan | Konsekuensi untuk ablation |
|---|---|
| Enhanced RAG sudah ber-arsitektur **Stage ABC + config dataclass + registry dict + PRESETS** | Hampir semua ablation Enhanced = **1 entry dict**, nol kode |
| Eval runner sudah punya addressing `enhanced@<preset>` ([run_eval.py:217-219](../eval/run_eval.py#L217-L219)) | Sel ablation baru langsung jalan lewat CLI |
| Judge sudah **decoupled + cached by answer-hash** ([run_judge.py](../eval/run_judge.py)) | Re-run analisis ~gratis; biaya nyata cuma fase generate |
| Agentic **tidak punya config object sama sekali** — semua konstanta hardcoded + state di module global | Semua ablation Agentic butuh refactor kecil dulu (kategori B) |
| Ablation 2×2 retrieval-stack **sudah pernah jalan tapi hanya n=30** dari 202 soal | Prioritas #1 = konfirmasi ke n=202, bukan eksperimen baru |
| Chunking terikat ke `gold_chunks` di testset | Ablation chunking = **merusak ground truth**, kategori C, tidak direkomendasikan |
| `_with_retry` tidur 30s **di dalam region yang diukur** ([llm.py:29-44](../src/ragtrial/llm.py#L29-L44)) | Angka latency mean saat ini terkontaminasi — wajib dibereskan sebelum klaim latency apa pun |

---

## 1. Audit implementasi per arsitektur

### 1.1 Substrat bersama (controlled variables)

Ini yang bikin perbandingan antar-tier sah — ketiganya membaca objek yang sama.

| Komponen | Lokasi | Status | Catatan modularitas |
|---|---|---|---|
| Unified vector store (1 koleksi Chroma, semua domain, field `domain`) | [vectorstore/store.py](../src/ragtrial/vectorstore/store.py) | ✅ Implemented | Singleton `unified_store` di baris 161. Dipakai naive/enhanced/agentic. **Coupling: singleton module-level** — parameter `fetch_k`/`weights` tidak bisa diubah per-run tanpa menyentuh objek global |
| Dense search | [store.py:120-127](../src/ragtrial/vectorstore/store.py#L120-L127) | ✅ | `similarity_search_by_vector`, filter opsional `where={"domain": ...}` |
| Hybrid (BM25 + dense, RRF) | [store.py:129-144](../src/ragtrial/vectorstore/store.py#L129-L144) | ✅ | BM25 di-cache **per filter-key** (global / per-domain) → IDF domain-lokal. Bobot `(0.5, 0.5)` di konstruktor |
| Cross-encoder rerank | [pipeline/rerank.py:53-76](../src/ragtrial/pipeline/rerank.py#L53-L76) | ✅ | `rerank_documents()` framework-agnostic, **dishare enhanced + agentic** → tidak ada drift. Model via `$RERANK_MODEL`, default `BAAI/bge-reranker-base` |
| LLM & embeddings | [llm.py](../src/ragtrial/llm.py) | ✅ | `gemini-2.5-flash` temp 0.1; `gemini-embedding-2` 768-dim. Singleton module-level |
| Sub-timing instrumentation | [store.py:87-144](../src/ragtrial/vectorstore/store.py#L87-L144) | ✅ | `t_embed_query / t_search / t_bm25 / t_fuse` diisi in-place, dipakai ketiga tier |
| Kontrak hasil `RagResult` + `decisions` | [result.py](../src/ragtrial/result.py) | ✅ | Normalisasi lintas-tier — eval jadi system-agnostic |

**Constraint penting yang ditemukan:** pada jalur hybrid, pool dense **selalu** `self.fetch_k` (=10), bukan `k` ([store.py:121-123](../src/ragtrial/vectorstore/store.py#L121-L123)), dan `bm25.k = fetch_k` juga. Jadi `k_candidates > 10` **tidak berpengaruh pada hybrid** — fusi hanya melihat 10+10 kandidat lalu dipotong `[:k]`. Setiap studi "kedalaman pool kandidat" harus memperbaiki ini dulu, kalau tidak hasilnya menyesatkan.

> **✅ Diverifikasi runtime (2026-08-04).** Dijalankan `unified_store.search()` sungguhan dengan query yang sama pada `k=10` vs `k=30`, diinstrumentasi tepat di titik fusi (sebelum `[:k]` dipotong):
> ```
> k_requested=10: dense_docs=10  bm25_docs=10  fused_pool_before_slice=19  final_after_slice=10
> k_requested=30: dense_docs=10  bm25_docs=10  fused_pool_before_slice=19  final_after_slice=19
> ```
> `dense_docs`/`bm25_docs` tetap 10/10 di kedua kasus; pool gabungan selalu ≤19 (union dari 10+10 dengan overlap) — meminta k=30 hanya mengembalikan apa yang sudah ada di pool, tidak pernah lebih dari `2×fetch_k`. Klaim terbukti, bukan dugaan.

---

### 1.2 Naive RAG

Satu fungsi, 78 baris: [rag/naive.py](../src/ragtrial/rag/naive.py).

| Komponen | Implemented? | Lokasi | Modular? | Bisa on/off? | Coupling |
|---|---|---|---|---|---|
| Dense retrieve global top-k | ✅ | [naive.py:33](../src/ragtrial/rag/naive.py#L33) | Panggilan langsung ke singleton | `k` adalah parameter fungsi (default 5) — **tapi tidak terekspos** ke `modes.py` maupun `run_eval` | Terikat `unified_store` |
| Context stuffing polos | ✅ | [naive.py:23-25](../src/ragtrial/rag/naive.py#L23-L25) | `_stuff()` private | ❌ hardcoded | Sengaja tanpa header sumber (menjaga "kenaifan") |
| Generate | ✅ | [naive.py:37](../src/ragtrial/rag/naive.py#L37) | `PROMPT_NAIVE` | ❌ | — |
| Intent gate / router / rewrite / rerank / hybrid | ❌ **sengaja tidak ada** | — | — | — | Ini *definisi* baseline |

**Penilaian:** monolitik tapi sengaja dan kecil. Satu-satunya lever nyata = `k`. Rutenya di [modes.py:33](../src/ragtrial/modes.py#L33) dan [run_eval.py:213](../eval/run_eval.py#L213) sama-sama membuang `k` → naive selalu k=5.

---

### 1.3 Enhanced RAG — **paling siap ablation**

Pipeline tetap, dirakit dari dataclass: [rag/enhanced.py](../src/ragtrial/rag/enhanced.py) + [pipeline/](../src/ragtrial/pipeline/).

Urutan stage (dari `build_enhanced`, [enhanced.py:130-141](../src/ragtrial/rag/enhanced.py#L130-L141)):
`intent → route → rewrite → retrieve → rerank → generate`

| Komponen | Implemented? | Lokasi | Registry / switch | Modular? | Coupling |
|---|---|---|---|---|---|
| **Intent gate** (VALID/INVALID, embedding-based, bukan LLM) | ✅ | [pipeline/intent.py](../src/ragtrial/pipeline/intent.py) | `INTENT_GATES = {none, semantic}` (baris 147) | Sangat — `None` berarti stage tidak di-append | Longgar. Downstream cuma baca `state.intent` ([retrieve.py:38](../src/ragtrial/pipeline/retrieve.py#L38), [generate.py:46](../src/ragtrial/pipeline/generate.py#L46), [rewrite.py:44](../src/ragtrial/pipeline/rewrite.py#L44)) |
| **Domain router** | ✅ 3 varian | [pipeline/route.py](../src/ragtrial/pipeline/route.py) | `ROUTERS = {none, semantic, llm}` (baris 139) | Sangat | `state.route` → filter domain di retrieve. Default `none` (global) |
| **Query rewriter** | ⚠️ 2 dari 3 | [pipeline/rewrite.py](../src/ragtrial/pipeline/rewrite.py) | `REWRITERS = {passthrough, hyde}` (baris 66) | Sangat | `MultiQueryRewriter` **stub, `raise NotImplementedError`, sengaja tidak diregistrasi** (baris 54-69) |
| **Retrieval strategy** | ✅ | [pipeline/retrieve.py](../src/ragtrial/pipeline/retrieve.py) | `strategy="dense"\|"hybrid"`, `k` | Sangat | Meneruskan ke `unified_store.search` |
| **Reranker** | ✅ | [pipeline/rerank.py](../src/ragtrial/pipeline/rerank.py) | `RERANKERS = {none, cross_encoder}` (baris 126) | Sangat | `NoopReranker` tetap memotong ke `top_n` → pool generator terkontrol di kedua kondisi ✅ |
| **Generate** | ✅ | [pipeline/generate.py](../src/ragtrial/pipeline/generate.py) | 4 prompt dipilih dari route+intent+docs | Sedang | Pemilihan prompt hardcoded if/elif; prompt sendiri ada di `rag/prompts.py` |
| **Config & presets** | ✅ | [enhanced.py:31-51](../src/ragtrial/rag/enhanced.py#L31-L51) | `EnhancedRAGConfig` 7 field + 5 `PRESETS` | Sangat | `PRESETS` dibaca langsung oleh runner eval |

**Penilaian:** ini teladan desain untuk ablation. Registry dict + dataclass + assembler berarti sel ablation baru = satu baris di `PRESETS`. Config tersimpan ke `meta["config"]` ([enhanced.py:107](../src/ragtrial/rag/enhanced.py#L107)) sehingga tiap record hasil membawa identitas kondisinya sendiri — bagus untuk reprodusibilitas tesis.

**Coupling tersisa (kecil):** `k_candidates` diteruskan sebagai `k` ke retrieve, tapi karena batasan `fetch_k` di §1.1, di mode hybrid nilainya tidak benar-benar mengubah kedalaman pool.

---

### 1.4 Agentic RAG — **fungsional, tapi tidak ada permukaan konfigurasi**

Loop tool-calling LangGraph: [rag/agentic.py](../src/ragtrial/rag/agentic.py).

| Komponen | Implemented? | Lokasi | Bisa on/off? | Coupling |
|---|---|---|---|---|
| Orkestrator LLM (graph agent↔tools) | ✅ | [agentic.py:206-222](../src/ragtrial/rag/agentic.py#L206-L222) | ❌ | `agentic_app` **module global**, di-compile sekali per proses (`_ensure_app`) |
| Tool per domain (= routing implisit) | ✅ | [agentic.py:60-69](../src/ragtrial/rag/agentic.py#L60-L69) | ❌ | Dibangun dari `SEARCHABLE_CAPABILITIES`; **tidak ada tool global/tanpa filter** |
| Retrieval di dalam tool | ✅ | [agentic.py:168-170](../src/ragtrial/rag/agentic.py#L168-L170) | ❌ | `strategy="hybrid"` **hardcoded**, `domain=domain` selalu difilter |
| Rerank per tool call | ✅ | [agentic.py:175](../src/ragtrial/rag/agentic.py#L175) | ❌ | Selalu aktif; `RERANK_TOP_N=5` konstanta modul |
| Sinyal self-correction (`[LOW RELEVANCE SIGNAL]`) | ✅ | [agentic.py:178-188](../src/ragtrial/rag/agentic.py#L178-L188) | ❌ | Ambang `RERANK_SUFFICIENCY=0.30` konstanta modul (baris 47) |
| Batas iterasi | ✅ | [agentic.py:198-203](../src/ragtrial/rag/agentic.py#L198-L203) | ❌ | `MAX_ITERATIONS=5` konstanta modul |
| Intent handling | ✅ tapi **implisit** | [agentic.py:85-95](../src/ragtrial/rag/agentic.py#L85-L95) (system prompt) | ❌ | Menyatu dalam prompt; tidak bisa dimatikan tanpa mengedit teks prompt |
| Config object / presets | ❌ **tidak ada** | — | — | Tidak ada `AgenticRAGConfig`, tidak ada `build_agentic()` |

**Penilaian:** agentic bekerja dan trace-nya kaya (`meta.steps` mencatat tool, query, n_docs, top_score, weak) — observabilitasnya bagus. Tapi **empat lever paling menarik secara riset** (routing on/off, rerank on/off, ambang sufficiency, batas iterasi) semuanya konstanta modul + state global. Ablation apa pun di sini menuntut refactor terlebih dahulu.

---

### 1.5 Infrastruktur evaluasi

| Bagian | Lokasi | Kesiapan |
|---|---|---|
| Runner generate | [eval/run_eval.py](../eval/run_eval.py) | ✅ Checkpoint atomik tiap soal + `--resume` + `--limit` + `--ids` + `--sleep`. **Addressing `enhanced@<preset>`** (baris 217-219) |
| Metrik retrieval | [eval_core.py:63-101](../eval/eval_core.py#L63-L101) | ✅ hit@k, recall@k, precision@k, MRR |
| Metrik routing & intent | [eval_core.py:112-162](../eval/eval_core.py#L112-L162) | ✅ confusion matrix, per-class P/R/F1, macro-F1 |
| Judge (faithfulness / relevance / refusal) | [eval/run_judge.py](../eval/run_judge.py) | ✅ **Decoupled + cache persisten by answer-hash** → jawaban tak berubah = 0 call |
| Agregasi & breakdown | [eval/analyze.py](../eval/analyze.py) | ✅ `--systems` bebas, `--breakdown query_type difficulty ...`, `--save` |
| Testset | [eval/testset.json](../eval/testset.json) | ✅ **202 soal** — route: sosial 61 / dukcapil 60 / perizinan 31 / opd 25 / both 16 / none 9; difficulty easy 93 / medium 91 / hard 18; 8 `query_type` |
| Hasil ablation terdahulu | `eval/results/SUMMARY_ablation.txt` | ⚠️ **n=30 saja** untuk 4 sel 2×2 |

**Kesimpulan infrastruktur: eval bukan bottleneck.** Untuk Enhanced, semua sudah siap hari ini.

---

## 2. Penilaian kesiapan ablation

**A = siap (config saja) · B = refactor kecil · C = perubahan arsitektural**

### Kategori A — siap dijalankan sekarang

| Komponen | Cara mengaktifkan | Alasan |
|---|---|---|
| Hybrid vs dense (Enhanced) | `PRESETS` sudah ada | Field `retrieval` diteruskan langsung ke store |
| Rerank on/off (Enhanced) | `PRESETS` sudah ada | `RERANKERS` registry; Noop tetap trim ke `top_n` → variabel terkontrol |
| Intent gate on/off | preset `no_intent` sudah ada | `INTENT_GATES["none"] = None` → stage tidak dirakit |
| Tipe router (none/semantic/llm) | tambah 2 entry `PRESETS` | Ketiga kelas sudah diimplementasi & teruji impor |
| HyDE vs passthrough | tambah 1 entry `PRESETS` | `HyDERewriter` lengkap, sudah menangani skip saat intent invalid |
| `top_n` (5 → 3/8/10) | tambah entry `PRESETS` | Field dataclass murni |
| Model reranker (bge-base vs v2-m3) | env `$RERANK_MODEL` | Cache di-key `(model, max_length)`; artefak `results_bgebase/` menunjukkan ini sudah pernah dipakai |
| Perbandingan tier naive/enhanced/agentic | sudah dilakukan | — |

### Kategori B — refactor kecil

| Komponen | Yang perlu diubah | Estimasi |
|---|---|---|
| `k` pada Naive | Teruskan `--k-retrieval` di `run_eval` ke `ask_naive`; sekarang dibuang | ~5 baris |
| Kedalaman pool kandidat (`k_candidates`) | `fetch_k` harus mengikuti `k` di jalur hybrid ([store.py:121-123](../src/ragtrial/vectorstore/store.py#L121-L123), 131-138) | ~10 baris |
| Bobot RRF hybrid (0.5/0.5) | `weights` adalah atribut instance pada singleton; ekspos lewat `RetrieveStage` + `EnhancedRAGConfig` | ~15 baris |
| **Ablation Agentic** (routing off, rerank off, max_iter, ambang sufficiency) | Buat `AgenticRAGConfig` + `build_agentic()`; ubah global `_TOOLS`/`_llm_with_tools`/`agentic_app` jadi state instance; tambah tool `search_all` (domain=None); tambah `agentic@<preset>` di runner | ~120 baris, 2 file |
| Kontaminasi timing retry | Kecualikan `time.sleep` backoff dari region terukur, atau catat `n_retries` per query | ~15 baris |
| MultiQuery rewriter | `run()` belum ada **dan** retrieve harus fan-out multi-query lalu union | ~60 baris, batas B/C |

### Kategori C — perubahan arsitektural (dan/atau merusak ground truth)

| Komponen | Kenapa berat |
|---|---|
| Strategi/ukuran chunking | Tiap domain punya chunker bespoke (pasal-aware untuk sosial, QA-aware untuk dukcapil, 1-record-per-chunk untuk opd/perizinan). Mengubahnya menuntut preprocess ulang + **embed ulang seluruh korpus** + rebuild unified. **Fatal: `gold_chunks` di testset di-key ke `page:` / `nomor:` / `id:`+pasal — ID gold jadi tidak valid, 202 soal harus dilabeli ulang manual** |
| Ganti model embedding | Embed ulang 4 domain (kena rate-limit Gemini) + rebuild unified. Kode kecil, biaya besar. Gold ID selamat (chunk tak berubah), jadi lebih layak daripada chunking — tapi tetap mahal |
| Multi-agent / tool SQL / web search | Node graph baru + capability baru |
| Reranker berbasis LLM | Komponen baru |

---

## 3. Studi ablation yang direkomendasikan

Diurutkan menurut (nilai riset ÷ usaha). Empat teratas semuanya menjawab pertanyaan yang **sudah dibangkitkan oleh data kalian sendiri** di `EVAL_FINDINGS.md` — ini kekuatan besar untuk bab pembahasan: temuan → hipotesis → uji terencana.

---

### R1 — Konfirmasi anomali hybrid pada n=202 ⭐ prioritas tertinggi

**Motivasi.** Temuan A2 kalian: hybrid **menurunkan** recall (dense_only 0.750 → +hybrid 0.728; rerank_no_hybrid 0.850 → +hybrid 0.761), berlawanan dengan literatur umum. Hipotesisnya: korpus hukum Indonesia penuh pasal near-duplicate, BM25 menarik "cocok kata tapi salah pasal" dan RRF menyuntikkan noise itu ke ranking akhir. Tapi ini **n=30** — terlalu kecil untuk dipertahankan di sidang.

**Dampak yang diharapkan.** Kalau bertahan di n=202, ini kontribusi paling orisinal di tesis: bukti bahwa retrieval hybrid **corpus-dependent**, bukan peningkatan universal, dengan mekanisme yang bisa dinamai. Juga langsung mengubah default sistem ke dense+rerank.

**Effort implementasi: Low (nol kode).** Keempat preset sudah ada.

**Biaya runtime.** 4 sel × 202 soal. Pakai `--no-judge` → hanya 1 embed + 1 generate per soal; ±25-35 menit per sel, ±2 jam total sekali jalan (checkpoint + `--resume` melindungi). Fase judge menyusul dan sebagian besar cache-hit.

**Perubahan kode:** tidak ada. Sebelum menyimpulkan, perbaiki dulu kontaminasi timing (§4 B-item) kalau ingin mengklaim latency; untuk klaim **recall** tidak perlu.

---

### R2 — Ablation routing Agentic (isolasi "kontrol LLM" dari "filter domain") ⭐ nilai riset tertinggi

**Motivasi.** Temuan M1 kalian: agentic punya recall **terendah** (0.695 < naive 0.781 < enhanced 0.792), dan tersangkanya adalah routing (`both` recall hanya 0.38 — agent pilih satu sisi). Tapi desain saat ini **membaurkan dua variabel**: agentic = kontrol-LLM **dan** filter-domain sekaligus. Selama keduanya menyatu, kalian tidak bisa menyatakan mana penyebab kehilangan recall. Ini persis butir [P2] di `EVAL_FINDINGS.md` §E.

**Dampak yang diharapkan.** Memisahkan sebab. Jika `agentic@global` (tool tanpa filter domain) memulihkan recall ke ~0.78, terbukti **routing** yang merugikan, bukan orkestrasi LLM — mengubah kesimpulan tesis dari "agentic lebih buruk" menjadi "filter domain agresif berbahaya pada korpus lintas-domain, terlepas dari siapa yang memutuskan". Jauh lebih tajam.

**Effort implementasi: Medium** (satu-satunya rekomendasi non-trivial, tapi terbayar).

**Biaya runtime.** 2 sel × 202. Agentic ~2 call LLM/soal → ±40-50 menit per sel.

**Perubahan kode:**
- `rag/agentic.py`: `AgenticRAGConfig` (field: `routing: "per_domain"|"global"`, `reranker: bool`, `max_iterations: int`, `sufficiency: float`, `k_per_tool`, `top_n`); pindahkan `_TOOLS`/`_llm_with_tools`/`agentic_app` dari global ke atribut kelas `AgenticRAG`; tambah tool `search_all` yang memanggil `unified_store.search(..., domain=None)`; parametrikan `_system_prompt` untuk mendeskripsikan tool yang aktif.
- `eval/run_eval.py`: cabang `agentic@<preset>`, sejajar `_mk_enhanced`.
- `modes.py`: tetap; `ask_agentic()` jadi pembungkus tipis atas config default (kompatibel mundur).

---

### R3 — Intent gate on/off (Enhanced)

**Motivasi.** Intent gate adalah komponen "abstention" — sumbu di mana enhanced/agentic mengalahkan naive telak (recall_invalid 1.00 vs 0.00). Untuk chatbot layanan publik, menolak dengan benar adalah persyaratan keselamatan, bukan kemewahan. Ablation mengukur harganya: berapa banyak soal valid yang **salah ditolak** demi kemampuan menolak yang out-of-scope?

**Dampak.** Kuantifikasi trade-off abstention dengan angka, bukan narasi. Berpasangan alami dengan metrik `false_refusal` yang sudah ada.

**Effort: Low (nol kode)** — preset `no_intent` sudah ada.

**Biaya runtime.** 1 sel × 202 (±30 menit). Perlu fase judge (butuh `refused`), tapi 193 dari 202 soal non-none memakai cache lama kalau jawabannya tak berubah.

**Perubahan kode:** tidak ada. Pastikan `analyze.py` melaporkan `recall_invalid`/`false_refusal` untuk sel ini (sudah ada di `intent_eval`).

---

### R4 — Kedalaman pool kandidat & `top_n` (berapa banyak yang perlu dilihat reranker?)

**Motivasi.** Temuan A3 kalian: enhanced **lebih buruk** dari naive di soal easy (0.84 vs 0.88) — hipotesisnya cross-encoder mendemosikan chunk benar yang sudah di peringkat atas. Dan A1: rerank hanya **mengurutkan ulang** pool, tidak **memperluasnya** — kalau gold tidak masuk kandidat, rerank tak bisa menyelamatkan. Keduanya prediksi langsung tentang kedalaman pool. Menaikkan `k_candidates` 10 → 20/30 menguji keduanya sekaligus.

**Dampak.** Membedakan "retrieval gagal" dari "ranking gagal" — pemisahan diagnostik yang benar-benar berguna, dan penjelasan yang jauh lebih baik daripada "rerank membantu".

**Effort: Low-Medium** — butuh perbaikan `fetch_k` dulu (§2 kategori B, ~10 baris), lalu murni preset.

**Biaya runtime.** 3 sel × 202. Rerank warm ~1.4s untuk 10 dok, tumbuh ~linear → k=30 menambah ~3s/soal. ±35-45 menit per sel.

**Perubahan kode:** `store.py` — jadikan pool dense/BM25 pada jalur hybrid mengikuti `k` yang diminta (mis. `fetch_k = max(self.fetch_k, k)`), lalu tambah preset `pool20`, `pool30`. **Catatan validitas: ini juga berarti hasil hybrid n=30 sebelumnya berjalan dengan pool efektif 10 — sebut ini di bab limitasi.**

---

### R5 — HyDE / rewriting kueri

**Motivasi.** Korpus kalian adalah teks hukum formal; kueri pengguna informal ("cara bikin KTP"). Itu justru kasus buku-teks untuk HyDE — jembatani jurang gaya bahasa dengan mengambil berdasarkan *passage hipotetis* alih-alih pertanyaan. Belum pernah diuji sama sekali (`rewrite_rate=0.000` di semua run).

**Dampak.** Sedang. Bisa mengangkat recall pada `query_type = paraphrase` (41 soal) dan `semantic` (29 soal). Nilai jujurnya tetap ada meski null result: "rewriting tidak membantu ketika reranker sudah menskor terhadap kueri asli."

**Effort: Low (nol kode)** — `HyDERewriter` lengkap; hanya tambah `PRESETS["hyde"]`.

**Biaya runtime.** +1 call LLM per soal (±2x biaya generate). 1 sel × 202, ±45 menit.

**Perubahan kode:** satu entry `PRESETS`. Perhatikan `_n_llm_calls` sudah menghitung HyDE ([enhanced.py:82-83](../src/ragtrial/rag/enhanced.py#L82-L83)) — biaya terlaporkan dengan benar.

---

### R6 — Tipe router dalam Enhanced (none vs semantic vs llm)

**Motivasi.** Melengkapi R2 dari sisi berlawanan. R2 menanyakan "bagaimana kalau agentic berhenti me-routing?"; R6 menanyakan "bagaimana kalau enhanced mulai me-routing?". Bersama-sama keduanya membentuk **studi routing 2×2 yang bersih**, terpisah dari orkestrasi — dan itu argumen inti tesis kalian tentang author-time vs inference-time control.

**Dampak.** Tinggi kalau dipasangkan dengan R2; sedang kalau berdiri sendiri. Juga membandingkan routing embedding (murah) vs routing LLM (mahal) pada tugas keputusan identik.

**Effort: Low (nol kode)** — `ROUTERS` sudah punya ketiganya.

**Biaya runtime.** 2 sel × 202. `router="llm"` menambah 1 call LLM/soal.

**Perubahan kode:** dua entry `PRESETS`. **Satu peringatan:** `run_eval` hanya menghitung metrik routing untuk `system == "agentic"` ([run_eval.py:97-98](../eval/run_eval.py#L97-L98)). Untuk menskor akurasi routing enhanced, longgarkan syarat itu menjadi "record punya route non-global" (~3 baris).

---

## 4. Yang **tidak** perlu di-ablation

| Komponen | Kenapa tidak sepadan |
|---|---|
| **Strategi/ukuran chunking** | Bukan sekadar mahal — **merusak eksperimen**. `gold_chunks` di-key ke identitas chunk (`page:40`, `nomor:1.a`, `id:X` + suffix pasal). Re-chunk = seluruh 202 soal harus dilabeli ulang secara manual, dan hasil sebelum/sesudah **tidak lagi sebanding**. Biaya berbulan-bulan untuk temuan yang lebih layak jadi tesis tersendiri |
| **Ganti model embedding** | Butuh embed ulang penuh dengan rate-limit Gemini, dan hasilnya "model X > model Y pada korpus saya" — temuan *benchmarking*, bukan *arsitektural*. Tesis kalian tentang perbandingan arsitektur RAG; ini menggeser topik. (Kalau tetap mau, ini paling murah di antara kategori C karena gold ID selamat.) |
| **Wording prompt** | Ruang variasi tak berbatas, tidak ada variabel bebas berprinsip, hasil tidak akan bereproduksi lintas versi model. Reviewer akan mempertanyakan validitasnya — dan mereka benar |
| **Temperature / max_tokens LLM** | Hyperparameter tuning, bukan ablation. Temp sudah 0.1 (mendekati deterministik); efeknya akan tenggelam dalam noise pada n=202 |
| **MultiQuery rewriter** | Butuh implementasi baru (run() + fan-out retrieve + union), padahal **R5 sudah mencakup sumbu "apakah rewriting membantu?"** dengan nol kode. Effort tinggi untuk temuan yang tumpang-tindih |
| **Fine-tuning bobot RRF (0.5/0.5)** | Kalau R1 mengonfirmasi bahwa hybrid merugikan, mengoptimalkan bobotnya adalah menyelamatkan komponen yang datanya bilang harus dibuang. Grid search hyperparameter, bukan wawasan arsitektural. *Pengecualian:* satu titik `weights=(0.2, 0.8)` boleh sebagai catatan kaki pendukung mekanisme A2 — tapi jangan jadikan studi tersendiri |
| **BM25 tanpa dense** | Bukan sistem yang realistis untuk di-deploy; tidak ada yang akan mengirimnya. Sel ablation harus berupa konfigurasi yang benar-benar mungkin dipilih |
| **Model/prompt judge** | Ini instrumen pengukuran, bukan objek studi. Mengubahnya membatalkan komparabilitas semua run lain. (Cross-check RAGAS pada subset — butir [P3] kalian — sudah cukup sebagai validasi instrumen) |
| **`MAX_ITERATIONS` agentic** | Data kalian menunjukkan `avg_iterations` ≈ 2 dengan batas 5 — **batasnya tidak pernah mengikat**. Meng-ablasi parameter yang tidak aktif menghasilkan garis datar. (`RERANK_SUFFICIENCY` **berbeda** — itu mengikat dan menarik, tapi sertakan saja sebagai preset gratis begitu refactor R2 selesai) |

---

## 5. Peringkat prioritas

| # | Komponen / studi | Nilai riset | Effort | Biaya runtime | Risiko | Rekomendasi |
|---|---|---|---|---|---|---|
| 1 | **Hybrid vs dense × rerank pada n=202** (R1) | ⭐⭐⭐⭐⭐ | **Nol** (preset ada) | ~2 jam, 4 sel | Rendah — anomali bisa hilang di n besar, tapi itu sendiri hasil yang layak dilaporkan | **Kerjakan pertama.** Kandidat kontribusi orisinal |
| 2 | **Ablation routing Agentic** (R2) | ⭐⭐⭐⭐⭐ | Medium (~120 baris) | ~1,5 jam, 2 sel | Sedang — satu-satunya yang menyentuh kode produksi; jaga default tetap sama | **Kerjakan kedua.** Memisahkan variabel yang saat ini terbaur |
| 3 | **Intent gate on/off** (R3) | ⭐⭐⭐⭐ | **Nol** (preset ada) | ~30 mnt, 1 sel | Sangat rendah | Kerjakan — murah, langsung menyentuh keselamatan layanan publik |
| 4 | **Kedalaman pool kandidat / `top_n`** (R4) | ⭐⭐⭐⭐ | Low (~10 baris) | ~2 jam, 3 sel | Rendah — tapi wajib sebut pool efektif=10 di run lama | Kerjakan — memisahkan kegagalan retrieval dari kegagalan ranking |
| 5 | **Tipe router dalam Enhanced** (R6) | ⭐⭐⭐⭐ | Low (~3 baris di runner) | ~1 jam, 2 sel | Rendah | Kerjakan **bersama R2** — berdua membentuk studi routing 2×2 |
| 6 | **HyDE / rewriting** (R5) | ⭐⭐⭐ | **Nol** (preset ada) | ~45 mnt, 1 sel | Rendah | Kerjakan kalau waktu memungkinkan — null result tetap bisa dilaporkan |
| 7 | **Ambang sufficiency agentic** (0.30) | ⭐⭐⭐ | Nol *setelah* R2 | ~1,5 jam, 2 sel | Rendah | Ikut menumpang R2 — menjelaskan over-refusal M5 |
| 8 | **Model reranker** (base vs v2-m3) | ⭐⭐ | Nol (env var) | ~1 jam (v2-m3 ~7× lebih lambat) | Rendah | Opsional — catatan kaki, sebagian sudah ada di `results_bgebase/` |
| 9 | **`k` pada Naive** | ⭐⭐ | Low (~5 baris) | ~30 mnt | Rendah | Opsional — memperkuat baseline, bukan temuan |
| 10 | Bobot RRF | ⭐ | Low | ~1 jam | Rendah | Lewati (kecuali 1 titik penunjang A2) |
| 11 | Ganti model embedding | ⭐⭐ | High (embed ulang) | Berjam-jam + kuota | Sedang | Lewati — benchmarking, bukan arsitektur |
| 12 | Strategi chunking | ⭐⭐⭐ | **Very High** | Berhari-hari + relabel | **Tinggi — merusak ground truth** | **Jangan.** Sebut di future work |

---

## 6. Rencana implementasi konkret (4 teratas)

### Prasyarat P0 — integritas pengukuran (kerjakan sebelum klaim latency apa pun)

**Masalah.** `_with_retry` memanggil `time.sleep(30)` lalu ×1.5 ([llm.py:29-44](../src/ragtrial/llm.py#L29-L44)) **di dalam** region yang diukur `Pipeline.run` ([base.py:66-70](../src/ragtrial/pipeline/base.py#L66-L70)). Buktinya ada di data kalian sendiri: `SUMMARY_ablation.txt` sel `rerank_no_hybrid` melaporkan `intent mean=5.43s` dengan `p50=0.53s` dan **`p95=0.85s`**. Mean di atas p95 secara matematis hanya mungkin karena outlier ekstrem — yaitu backoff retry, bukan waktu komputasi.

**Perbaikan.** `src/ragtrial/llm.py` — akumulasi total waktu tidur per pemanggilan dan ekspos (mis. lewat contextvar atau dict yang dikembalikan), lalu kurangkan dari timing stage; **atau** minimal catat `n_retries` per query ke `timings` agar record yang terkontaminasi bisa disaring saat analisis. Opsi kedua lebih murah dan cukup.

**Dampak:** semua angka latency di §M2/M3/M4 `EVAL_FINDINGS.md` menjadi bisa dipertahankan. Tanpa ini, gunakan **median saja** dan katakan demikian secara eksplisit.

> **✅ Diverifikasi runtime (2026-08-04).** Dipicu error `503 UNAVAILABLE` palsu lewat `invoke_with_retry` **asli** (bukan tiruan/mock terpisah — jalur kode produksi persis), sleep dipersingkat 30s→0.5s hanya agar tes cepat selesai:
> ```
> [retry] transient API error; attempt 1/5, wait 30s (503 UNAVAILABLE...)
> [retry] transient API error; attempt 2/5, wait 45s (503 UNAVAILABLE...)
> measured stage dt = 1.00s  (includes 2 retry sleeps)
> ```
> `time.sleep` terpanggil **sebelum** boundary `t0/dt` ditutup — persis pola timing `Pipeline.run` ([base.py:66-70](../src/ragtrial/pipeline/base.py#L66-L70)). Di produksi (tanpa dipersingkat), 2 retry seperti ini akan menambah **75 detik** ke `dt` stage yang sama — besarannya cocok dengan anomali mean(5.43s) > p95(0.85s) yang dikutip dari `SUMMARY_ablation.txt`. Mekanisme kontaminasi terbukti, bukan dugaan.

---

### R1 — Hybrid × rerank pada n=202 (nol kode)

- **File yang diubah:** tidak ada.
- **Flag konfigurasi cukup?** Ya, sepenuhnya. Keempat sel ada di `PRESETS` ([enhanced.py:45-51](../src/ragtrial/rag/enhanced.py#L45-L51)).
- **Skrip eval sudah mendukung?** Ya — `run_eval` menerjemahkan `enhanced@<preset>` lewat `_mk_enhanced`, `analyze.py --systems` menerima nama sembarang, `run_judge` bekerja per nama sistem.
- **Urutan jalan:** fase generate `--no-judge` untuk keempat sel (checkpoint per soal; pakai `--resume` bila terputus) → fase judge → `analyze --save --breakdown query_type difficulty expected_route`.
- **Yang wajib dilaporkan:** breakdown `expected_route` — hipotesis A2 memprediksi kerugian BM25 **terkonsentrasi di `sosial`** (61 soal, korpus pasal near-duplicate) dan minimal di `opd`/`perizinan` (chunk atomik). Kalau polanya muncul, mekanismenya terkonfirmasi, bukan sekadar angka agregat.
- **Arsipkan** `eval/results/` yang sekarang ke `eval/results_n30/` dulu — jangan timpa bukti n=30.

### R2 — Ablation routing Agentic (refactor sesungguhnya)

- **File yang diubah:**
  - `src/ragtrial/rag/agentic.py` — perubahan utama:
    1. `@dataclass AgenticRAGConfig`: `routing`, `reranker`, `max_iterations`, `sufficiency`, `k_per_tool`, `top_n` (menggantikan konstanta baris 44-48).
    2. Kelas `AgenticRAG` yang memiliki `tools` / `llm_with_tools` / `app` sendiri — memindahkan global `_TOOLS`, `_llm_with_tools`, `agentic_app` (baris 72-73, 207) ke instance. **Ini pekerjaan mekanis inti;** node saat ini menutup atas global, jadi jadikan node sebagai method atau closure atas `self`.
    3. Tool `search_all` untuk `routing="global"`: memanggil `unified_store.search(query, k, strategy, domain=None)` — jalur ini **sudah didukung** store ([store.py:66](../src/ragtrial/vectorstore/store.py#L66) mengembalikan `None` untuk domain falsy).
    4. `_system_prompt(capabilities, config)` — deskripsikan tool yang benar-benar tersedia; instruksi domain-abbreviation (baris 96-98) hanya relevan pada mode per-domain.
    5. `ask_agentic()` tetap sebagai pembungkus tipis atas config default → `modes.py`, server, dan CLI tidak berubah.
  - `eval/run_eval.py` — cabang `agentic@<preset>` sejajar `_mk_enhanced` (baris 204-221).
- **Flag konfigurasi cukup?** Tidak untuk yang pertama — refactor state global→instance adalah prasyarat. Setelah itu, ya: semua ablation agentic berikutnya jadi entry preset.
- **Skrip eval sudah mendukung?** Metrik ya (routing sudah dinilai untuk agentic). Hanya addressing preset yang kurang. Satu catatan: `store_correct` ([eval_core.py:322-328](../eval/eval_core.py#L322-L328)) membandingkan `source_used` dengan `expected_route`; di mode global `source_used` akan menjadi label global — putuskan lebih dulu apakah sel itu dinilai routing-nya (saran: tidak — sengaja tidak me-routing, sama seperti enhanced sekarang).
- **Verifikasi regresi:** jalankan `agentic@default` pada ~20 soal dan bandingkan dengan `per_query_agentic.json` yang ada. Jika refactor benar-benar netral-perilaku, `retrieved_ids` harus cocok.

> **✅ Diverifikasi runtime (2026-08-04) — dasar ekuivalensi `search_all`.** Query yang sama dijalankan lewat `ask_naive()` sungguhan vs panggilan langsung `unified_store.search(domain=None)`:
> ```
> naive (via ask_naive): ['sosial:22', 'dukcapil:36', 'dukcapil:36', 'dukcapil:35', 'dukcapil:39']
> direct (domain=None):   ['sosial:22', 'dukcapil:36', 'dukcapil:36', 'dukcapil:35', 'dukcapil:39']
> IDENTICAL: True
> ```
> Jadi `search_all` yang diusulkan **akan** mereproduksi jalur global naive/enhanced dengan tepat — bukan asumsi. **Catatan koreksi penting dari tes ini:** kesetaraan di atas berlaku untuk `strategy="dense"` (yang dipakai `ask_naive`). Saat dicoba `strategy="hybrid", domain=None` (strategi yang dipakai tool per-domain agentic saat ini), hasilnya **berbeda urutan/isi** dari varian dense — wajar karena strategi berbeda, tapi berarti: kalau `search_all` dibuat dengan `strategy="hybrid"` (konsisten dengan tool domain lain di agentic.py), ia TIDAK akan identik dengan `ask_naive` (dense) — hanya identik dengan varian global-hybrid Enhanced (preset `router="none"`, `retrieval="hybrid"`). Saat menulis §R2, definisikan eksplisit `search_all` memakai `strategy="hybrid"` (konsisten dengan tool domain lain), dan bandingkan hasilnya ke **Enhanced global**, bukan ke Naive.

### R3 — Intent gate on/off (nol kode)

- **File yang diubah:** tidak ada. Preset `no_intent` sudah ada ([enhanced.py:50](../src/ragtrial/rag/enhanced.py#L50)).
- **Skrip eval sudah mendukung?** Ya. Untuk sumbu intent, `eval/run_intent_eval_full.py` berjalan di atas 202 soal penuh dan memanggil `intent_eval` — tapi registry-nya hardcoded ke `naive`/`enhanced`/`agentic` (baris 117-126), **tanpa** addressing preset. Tambahkan cabang yang sama seperti R2, atau turunkan saja label intent dari `per_query_enhanced@no_intent.json` (setiap record membawa `decisions`, dan `decisions.intent` merekam retrieve-vs-direct) — **jalur kedua tidak butuh kode sama sekali**; pakai itu.
- **Metrik utama:** `recall_invalid` (9 soal none), `false_refusal` pada 193 soal valid, dan delta faithfulness/relevance.

### R4 — Kedalaman pool kandidat

- **File yang diubah:** `src/ragtrial/vectorstore/store.py` — di `search()`, jalur hybrid harus memakai pool sebesar `k` yang diminta, bukan `self.fetch_k` tetap: perbaiki fetch dense (baris 121-123), `bm25.k` (baris 80, sekarang di-set saat build dan **di-cache** — jadi ubah `r.k` per pemanggilan, jangan hanya saat konstruksi), dan `search_kwargs` dense retriever (baris 136-138).
- **Kemudian** tambah `PRESETS`: `pool20` (`k_candidates=20`), `pool30` (`k_candidates=30`), dan opsional `topn3` / `topn10`.
- **Flag konfigurasi cukup?** Setelah perbaikan store, ya.
- **Jebakan:** `_bm25` di-cache per filter-key, bukan per-k. Menyetel `r.k` pada instance ter-cache akan **bocor lintas pemanggilan** kalau dua sel berbagi proses. Karena `run_eval` menjalankan sistem secara berurutan dalam satu proses, setel `r.k` tepat sebelum tiap `invoke` — jangan sekali saat build.
- **Skrip eval sudah mendukung?** Ya, tanpa perubahan; `--k` pada `run_eval` adalah k untuk *metrik* dan tetap 5 agar recall@5 tetap sebanding lintas sel.

---

## 7. Urutan eksekusi yang disarankan

1. **P0** instrumentasi retry (~15 baris) + arsipkan `eval/results/` → `eval/results_n30/`
2. **R1** empat sel pada n=202 — nol kode, nilai tertinggi
3. **R3** ikut menumpang (nol kode, satu sel lagi)
4. **R4** perbaikan store, lalu sel pool
5. **R2 + R6** bersama-sama sebagai studi routing 2×2 — satu-satunya blok yang benar-benar butuh koding
6. **R5** HyDE kalau waktu masih ada

Tiga langkah pertama **tidak butuh satu baris kode pun** dan sudah mencakup dua dari tiga temuan terkuat kalian. Kalau waktu tesis mepet, R1+R3 saja sudah merupakan bab ablation yang layak dipertahankan.

---

## 8. Catatan validitas untuk ditulis di bab limitasi

- Sel ablation sebelumnya berjalan pada **n=30**, bukan 202 — jangan campur angkanya dalam satu tabel tanpa keterangan.
- Run hybrid sebelum perbaikan R4 memakai **pool efektif 10**, sehingga `k_candidates=10` kebetulan benar tapi tidak pernah benar-benar teruji sebagai variabel.
- Latency mean terkontaminasi backoff retry sampai P0 dikerjakan — laporkan **median** kecuali sudah diperbaiki.
- Faithfulness mengukur *grounding ke chunk yang di-retrieve*, bukan kebenaran (temuan A4 kalian sendiri) — tidak ada sel ablation yang bisa memperbaiki ini; correctness butuh evaluasi expert.
