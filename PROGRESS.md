# Progress Report — RAG Layanan Publik Kab. Batang

**Update terakhir**: 2026-05-24 (3-way RAG split selesai: naive / enhanced / agentic)

---

## 1. Data Sources

| File | Status | Notes |
|---|---|---|
| `data/raw/dukcapil/Buku-Saku-Dafduk-Capil-2023.pdf` (282 hal) | ✅ End-to-end | 258 cleaned docs → 150 chunks → Chroma `dukcapil_qa` |
| `data/raw/opd/Nama dan Alamat OPD Kab Batang.pdf` (3 hal) | ✅ End-to-end | 61 records (1 doc = 1 OPD) → Chroma `opd_directory` |
| `data/raw/unprocessed/PERDA NOMOR 1 TAHUN 2019.pdf` | ❌ Belum diproses | Arsitektur siap (`sources/<domain>/`) |
| `data/raw/unprocessed/Analisis-dan-evaluasi-hukum-no-3-tahun-2025.pdf` | ❌ Belum diproses | — |

Ketiga mode pakai collection per-domain yang sama (211 vektor = 150 dukcapil + 61 opd). Naive fan-out ke semua collection lalu merge global top-k di memori — tak ada store terpisah.

---

## 2. Arsitektur (post 3-way split)

Lihat **[docs/REFACTOR_3WAY.md](docs/REFACTOR_3WAY.md)** untuk desain + alasan lengkap.

```
src/ragtrial/
├── result.py        ← RagResult (kontrak output 3 mode; eval/chat/UI konsumsi ini)
├── capabilities/    ← Capability ABC + VectorSourceCapability + registry
├── sources/<domain>/← co-located preprocess + chunk + capability (1 domain = banyak file)
├── pipeline/        ← stage komposabel enhanced (rewrite/route/retrieve/rerank/generate)
├── vectorstore/     ← builder Chroma per-domain
├── rag/             ← naive.py | enhanced.py | agentic.py | prompts.py
└── chat/session.py  ← ChatSession(mode=…)
```

**Tambah domain** = 1 folder `sources/<domain>/` + 1 baris di `SOURCES`. **Tambah stage**
= 1 class + 1 entri factory dict. **Tambah tool** = implement `Capability` + register.

---

## 3. Status mode

### A. naive — END-TO-END ✅
[rag/naive.py](src/ragtrial/rag/naive.py) — fan-out dense ke semua collection per-domain,
merge global top-k, `PROMPT_NAIVE` (stuff polos, tanpa header per-source). Baseline jujur.

### B. enhanced — END-TO-END ✅
[rag/enhanced.py](src/ragtrial/rag/enhanced.py) — `build_enhanced(EnhancedRAGConfig)` rakit
`Pipeline([route, rewrite, retrieve, rerank, generate])` (route dulu → klasifikasi
pertanyaan asli). Routing live: KTP→dukcapil(0.88), Pariwisata→opd(0.82), off-topic→none(0.63).
Preset `fanout_hybrid`, `llm_router_hybrid`. HyDE rewriter live. Stub: MultiQuery, CrossEncoderReranker.

### C. agentic — END-TO-END ✅
[rag/agentic.py](src/ragtrial/rag/agentic.py) — LangGraph `agent ⇄ tools`. Tool `search_<domain>`
per kapabilitas; LLM pilih/iterasi/retry/skip; tiap langkah → `meta.steps`. `MAX_ITERATIONS=5`.

### D. Eval ✅
[eval/run_eval.py](eval/run_eval.py) — registry sistem `{naive, enhanced, agentic}`, konsumsi
`RagResult`; routing dihitung untuk enhanced & agentic. Metrik di [eval/eval_core.py](eval/eval_core.py)
(gold-id & label routing diturunkan dari registry).

---

## 4. Decisions & insights

| Topik | Decision | Reasoning |
|---|---|---|
| 3 mode | naive / enhanced / agentic | Pembeda = siapa kontrol alur (lihat docs) |
| naive | fan-out ke collection per-domain, merge global top-k, dense | Baseline minimal + database identik antar mode |
| Config enhanced | dataclass Python | Type-safe, enak eval sweep |
| Router default | SemanticRouter (embedding) | Sesuai spec enhanced; LLMRouter jadi opsi |
| agentic | tool-calling loop (rebuild) | Router statis lama bukan agentic |
| Source layout | co-locate per domain | 1 domain = banyak file; skala heterogen |
| Pengetahuan per-source | method Capability | Tambah domain tanpa ngedit prompts/eval |
| Output | RagResult seragam | Framework eval bisa diganti tanpa sentuh pipeline |
| Embedding | Gemini `gemini-embedding-2` 768d | Multilingual BI |

---

## 5. To-Do

### Short-term
- [ ] Run full eval `--systems naive enhanced agentic` untuk baseline kuantitatif 3 mode
- [ ] Tune threshold SemanticRouter di eval set (sekarang 0.65, heuristik)
- [ ] Tambah domain PERDA (`sources/perda/`, chunking pasal/ayat)

### Medium-term
- [ ] Isi stub: CrossEncoder reranker, MultiQuery (HyDE ✅)
- [ ] `SqlToolCapability` (text-to-sql) → otomatis jadi tool agentic
- [ ] BM25 indeks persisten (sekarang materialize semua doc di memori)
- [ ] Scale up testset (sekarang kecil)

### Future
- [ ] Multi-agent (tambah node agent di graph agentic)
- [ ] FastAPI backend (UI sudah dipisah via ChatSession)

---

## 6. Refactor history

- **Repackage (Tahap 1–4)**: src/ package + Capability registry + reorg + nbstripout.
- **3-way split (`#1`–`#8`)**: de-hardcode → co-locate sources → RagResult+pipeline →
  naive+unified → enhanced → agentic rebuild → wiring → docs.
  Detail + alasan: **[docs/REFACTOR_3WAY.md](docs/REFACTOR_3WAY.md)**.
