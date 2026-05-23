# Refactor: 3-Way RAG Split (naive / enhanced / agentic)

**Branch:** `refactor/package-layout` · **Commits:** `#1`–`#8` (`refactor split #N`)
**Tujuan:** memecah 2 mode RAG (naive-combined + router statis) menjadi **3 mode
yang jelas batasnya** dan menyiapkan struktur agar skala naik mulus — lebih banyak
data, banyak domain heterogen, multi-file per domain, integrasi tool (text-to-sql /
web search), dan multi-agent — tanpa menyusahkan eksperimen & penggantian framework eval.

Dokumen ini = **tracking lengkap**: plan, perubahan per tahap, dan **alasan** tiap keputusan.

---

## 1. Tiga mode — prinsip pembeda: *siapa yang mengontrol alur*

| Mode | Kontrol alur | Komposisi | Untuk apa |
|---|---|---|---|
| **naive** | Tetap, paling minimal | 1 collection gabungan → dense top-k → stuff → 1 LLM call | Baseline jujur, lantai pembanding |
| **enhanced** | Tetap, **developer** yang desain | `rewrite → route → retrieve → rerank → generate` (pipeline fixed) | Kualitas terkontrol; semua query lewat jalur sama, **tanpa** keputusan-alur oleh LLM |
| **agentic** | **Dinamis, LLM** yang putuskan | tool-calling loop: pilih tool, iterasi, retry, skip | Query kompleks/multi-domain; LLM menalar langkahnya sendiri |

> **Insight kunci:** mode "agentic" lama sebenarnya **router statis** (1 klasifikasi LLM
> lalu graph tetap). Menurut definisi di atas itu **bukan agentic** — tidak ada iterasi
> / kontrol-alur oleh LLM. Maka ia **dipensiun**: logika routing-nya disimpan sebagai
> opsi `LLMRouter` di enhanced, dan slot agentic diisi tool-calling loop yang sebenarnya.

---

## 2. Keputusan desain (terkunci) + alasan

| Keputusan | Pilihan | Alasan |
|---|---|---|
| Definisi **naive** | 1 collection `_unified`, dense-only | Baseline harus benar-benar minimal; data terpisah per domain digabung jadi satu pool agar "naive" = textbook naive |
| Build `_unified` | **Copy vektor** dari store per-domain (bukan re-embed) | Gratis, instan, nol rate-limit — vektor dokumen sudah ada |
| Konfigurasi **enhanced** | **Dataclass Python** `EnhancedRAGConfig` | Type-safe, enak buat eval sweep; ganti komponen = ganti 1 field |
| Router default enhanced | **SemanticRouter** (embedding, bukan LLM) | Sesuai definisi enhanced (classifier embedding, bukan LLM call); murah |
| Router lama (LLM) | Disimpan jadi opsi `LLMRouter` | Tidak ada kerja yang dibuang; berguna untuk ablation routing |
| **agentic** | Rebuild jadi LangGraph tool-calling loop | Satu-satunya yang memenuhi definisi "LLM mengontrol alur" |
| Granularitas tool agentic | **Cara 1 (lean)**: `search_<domain>` per kapabilitas | LLM = domain-selector dengan memilih tool; rewrite/iterasi muncul natural. Trace tetap transparan via `meta.steps` (tanpa tool ekstra) |
| Penataan source | **Co-locate per domain**: `sources/<domain>/` | 1 domain = banyak file/subfolder; tambah/hapus domain = 1 folder |
| Pengetahuan per-source | Method `Capability` (`gold_id`, `citation_hint`, `format_header`) | Tambah domain tidak perlu ngedit prompts/eval (dulu hardcoded) |
| Kontrak output | **`RagResult`** seragam untuk 3 mode | eval/chat/UI cukup konsumsi 1 kontrak → framework eval bisa diganti tanpa sentuh pipeline |

---

## 3. Arsitektur akhir

```
src/ragtrial/
├── config.py                 # path absolut, UNIFIED_*; load_env()
├── llm.py                    # singleton llm + embeddings, make_judge_llm()
├── result.py                 # RagResult — kontrak output 3 mode
│
├── capabilities/             # abstraksi retrieval + (future) tools
│   ├── base.py               # Capability ABC (invoke, gold_id, citation_hint, format_header) + format_context
│   ├── vector_source.py      # VectorSourceCapability (Chroma; dense/hybrid; strategy override per-call)
│   └── registry.py           # CAPABILITIES / SEARCHABLE_CAPABILITIES (dari SOURCES)
│
├── sources/                  # co-located per DOMAIN
│   ├── base.py               # Source dataclass + save_docs
│   ├── __init__.py           # SOURCES = {dukcapil, opd}  ← satu tempat daftar domain
│   ├── dukcapil/{preprocess,chunk,capability}.py + __init__.py(build_documents)
│   └── opd/{preprocess,chunk,capability}.py        + __init__.py(build_documents)
│
├── pipeline/                 # stage komposabel untuk ENHANCED
│   ├── base.py               # Stage ABC + RagState + Pipeline(runner)
│   ├── rewrite.py            # REWRITERS: passthrough | hyde* | multiquery*
│   ├── route.py              # ROUTERS:   none | semantic | llm
│   ├── retrieve.py           # RetrieveStage (route-aware, dense/hybrid)
│   ├── rerank.py             # RERANKERS: none | cross_encoder*
│   └── generate.py           # GenerateStage
│
├── vectorstore/
│   ├── builder.py            # build Chroma per domain (batched + retry)
│   └── unified.py            # build `_unified` dengan copy vektor
│
├── rag/
│   ├── prompts.py            # PROMPT_NAIVE/SINGLE/COMBINED/NONE/ROUTER + builder dari registry
│   ├── naive.py              # ask_naive()      → RagResult
│   ├── enhanced.py           # EnhancedRAGConfig + build_enhanced() → RagResult
│   └── agentic.py            # ask_agentic()    → RagResult (LangGraph agent↔tools)
│
└── chat/session.py           # ChatSession(mode=…) multi-turn di atas mode mana pun

scripts/  ask.py (--mode), preprocess.py, build_vectorstore.py (--source …|unified)
eval/     run_eval.py (registry sistem, konsumsi RagResult), eval_core.py, analyze.py
```

`*` = stub bertanda TODO (raise `NotImplementedError`); slot siap diisi.

**Alur per mode:**
- **naive:** `question → _unified.similarity_search(k) → PROMPT_NAIVE → LLM`
- **enhanced:** `RagState` lewat `Pipeline([rewrite, route, retrieve, rerank, generate])`
- **agentic:** `agent ⇄ tools` loop sampai LLM berhenti memanggil tool (cap `MAX_ITERATIONS`)

---

## 4. Changelog per tahap (alasan singkat)

- **`#1` de-hardcode registry** — `Capability.gold_id()`/`citation_hint()`; prompt sitasi
  & label routing eval diturunkan dari registry. *Kenapa:* tambah domain ke-3 tak boleh
  butuh ngedit prompts/eval.
- **`#2` co-locate sources per domain** — `sources/<domain>/{preprocess,chunk,capability}`;
  `Source` + `SOURCES`; `build_documents()` menelusuri `data/raw/<domain>/` (multi-file).
  *Kenapa:* skala domain heterogen; 1 domain bisa banyak file. Parity 258→150 / 61→61.
- **`#3` RagResult + pipeline stages** — kontrak output seragam + `Stage`/`Pipeline` + factory
  dict tiap stage. *Kenapa:* fondasi enhanced + decoupling eval. SemanticRouter diimplement;
  HyDE/MultiQuery/CrossEncoder = stub.
- **`#4` naive + unified** — `_unified` via copy vektor; `ask_naive()`. *Kenapa:* baseline jujur,
  tanpa biaya re-embed.
- **`#5` enhanced** — `EnhancedRAGConfig` + `build_enhanced()`; default semantic+dense; preset
  `fanout_hybrid` (≈ naive_combined lama) & `llm_router_hybrid` (≈ agentic lama). Threshold
  semantic 0.65 (floor similarity Gemini tinggi).
- **`#6` agentic rebuild** — LangGraph tool-calling loop, `search_<domain>`, iterasi/retry/skip,
  `meta.steps` trace. *Kenapa:* agentic sejati = LLM kontrol alur.
- **`#7` wiring** — `ChatSession(mode=)`, `scripts/ask.py --mode`, eval registry sistem + konsumsi
  RagResult; hapus `naive_combined.py` (jadi preset) & `_migrate_notebooks.py`.
- **`#8` docs** — dokumen ini + refresh README/PROGRESS.

---

## 5. Cara menambah (extensibility)

### Tambah DOMAIN data baru (mis. `pajak`)
1. Taruh file di `data/raw/pajak/…` (boleh banyak file/subfolder).
2. Buat `src/ragtrial/sources/pajak/`:
   - `preprocess.py` → `preprocess(path) -> List[Document]` (handler per tipe file)
   - `chunk.py` → `chunk_for_vectorstore(docs) -> List[Document]`
   - `capability.py` → `pajak_capability = VectorSourceCapability(name="pajak", …, gold_id_fn=…, citation=…)`
   - `__init__.py` → `build_documents()` + `source = Source(...)`
3. Daftarkan di `sources/__init__.py` (`_ALL` list).
4. `uv run python scripts/preprocess.py --source pajak`
5. `uv run python scripts/build_vectorstore.py --source pajak`
6. `uv run python scripts/build_vectorstore.py --source unified`  (refresh naive)

**Nol perubahan** di registry, prompts, eval, ketiga mode RAG, atau router — semua auto-pickup.

### Tambah STAGE enhanced (mis. HyDE / cross-encoder rerank)
Implement subclass `Stage` di module stage terkait, tambah 1 entri ke factory dict
(`REWRITERS`/`ROUTERS`/`RERANKERS`). Selesai — `build_enhanced` langsung bisa pakai
`EnhancedRAGConfig(rewriter="hyde", reranker="cross_encoder")`.

### Tambah TOOL non-vector (text-to-sql / web)
Implement `Capability` ABC (`invoke()` balikin `List[Document]`), daftarkan di
`capabilities/registry.py`. Otomatis jadi tool agentic (`search_<name>`) dan ikut
enhanced fan-out.

---

## 6. Known limitations & future

- **SemanticRouter "both" (multi-domain) lemah** — embedding tak bisa menalar "butuh 2 sumber";
  threshold 0.65 perlu di-tune ulang seiring domain bertambah. Query multi-domain → andalkan **agentic**.
- **BM25 (hybrid) materialize semua dokumen di memori** tiap init — perlu indeks persisten untuk korpus besar.
- **Chunker dukcapil kalibrasi 1 PDF** (page-range) — file kependudukan baru butuh handler sendiri.
- **Stub belum diisi:** HyDE, MultiQuery, CrossEncoderReranker, SQL/web tools, multi-agent.
- **Eval routing untuk "both"** kasar; metrik bisa diganti karena semua mode lewat `RagResult`.
- Notebook `notebooks/reports/` masih impor API lama — laporan historis, sengaja beku.

---

## 7. Commit map

| # | Commit | Isi |
|---|---|---|
| #1 | `5a15651` | de-hardcode registry |
| #2 | `b3d97e8` | co-locate sources per domain |
| #3 | `e8e1222` | RagResult + pipeline stages |
| #4 | `616dcda` | naive + unified collection |
| #5 | `c701012` | enhanced (config-driven) |
| #6 | `962f796` | agentic tool-calling loop |
| #7 | `1da802d` | wiring chat/CLI/eval, hapus modul lama |
| #8 | (this) | docs |
