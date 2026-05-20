# Progress Report — RAG Dukcapil & OPD Kab. Batang

**Update terakhir**: 2026-05-20 (refactor Tahap 1+2 selesai, codebase repackaged)

---

## 1. Data Sources

| File | Status | Notes |
|---|---|---|
| `data/raw/dukcapil/Buku-Saku-Dafduk-Capil-2023.pdf` (282 hal) | ✅ End-to-end | 258 cleaned docs → 150 chunks → Chroma `dukcapil_qa` |
| `data/raw/opd/Nama dan Alamat OPD Kab Batang.pdf` (3 hal) | ✅ End-to-end | 61 records (1 doc = 1 OPD) → Chroma `opd_directory` |
| `data/raw/unprocessed/PERDA NOMOR 1 TAHUN 2019.pdf` | ❌ Belum diproses | Belum diputuskan apakah masuk pipeline RAG |
| `data/raw/unprocessed/Analisis-dan-evaluasi-hukum-no-3-tahun-2025.pdf` | ❌ Belum diproses | Belum diputuskan apakah masuk pipeline RAG |

---

## 2. Arsitektur (post-refactor)

```
src/ragtrial/
├── capabilities/      ← REGISTRY: tambah source/tool = 1 file config + register
├── preprocessing/     ← raw PDF → cleaned Documents (per source)
├── chunking/          ← cleaned Documents → embedding chunks (per source)
├── vectorstore/       ← generic Chroma builder + retry
└── rag/               ← naive_combined, agentic (iterate registry)

scripts/               ← CLI: preprocess.py, build_vectorstore.py, ask.py
notebooks/             ← exploration/, reports/, archive/
eval/                  ← run_eval, analyze, testset, results
```

**Capability registry**: `naive_combined.ask_main()` fan out ke semua capability;
`agentic.ask_agentic()` router classify ke `<capability.name>` / `both` / `none`.
Tambah PERDA = bikin `instances/perda.py` + register → otomatis ke-pickup, zero
perubahan pipeline. Details: [docs/REFACTOR_TAHAP1.md](docs/REFACTOR_TAHAP1.md).

---

## 3. Pipeline status

### A. Dukcapil — END-TO-END ✅

**Preprocessing** ([src/ragtrial/preprocessing/dukcapil.py](src/ragtrial/preprocessing/dukcapil.py))
- PyMuPDFLoader → filter cover/ToC pages (282 → 258) → regex cleanup → section tag
- Sections: Kata Pengantar / BAB I / BAB II (Q&A) / BAB III

**Chunking** ([src/ragtrial/chunking/dukcapil.py](src/ragtrial/chunking/dukcapil.py))
- BAB II Q&A-aware (regex per nomor pertanyaan): 139 chunks
- Narrative (RecursiveCharacterTextSplitter, chunk=1200, overlap=200): 11 chunks
- **Total: 150 chunks**

**Vector store**: Chroma `dukcapil_qa` di `data/vector_stores/dukcapil/`,
Gemini `gemini-embedding-2` 768d.

**Variants** ([notebooks/exploration/rag_chat_dukcapil.ipynb](notebooks/exploration/rag_chat_dukcapil.ipynb)):
V1 Dense / V2 Dense+Rerank / V3 Hybrid / V4 Hybrid+Rerank.
Latency: V3 ~4.4s = paling balanced (V2/V4 reranker bottleneck 70-230s).

### B. OPD — END-TO-END ✅

**Preprocessing** ([src/ragtrial/preprocessing/opd.py](src/ragtrial/preprocessing/opd.py))
- pdfplumber table extract → merge continuation rows → parse alamat/email/telp → infer tipe
- Output schema: `{nomor, nama_opd, parent_opd, tipe, alamat, email, no_telp, page}`
- Quality assertions: main 1-43 lengkap, distribusi tipe OK, ga ada alamat kosong

**Vector store**: Chroma `opd_directory` di `data/vector_stores/opd/`,
no chunking (61 records atomic).

**Variants** ([notebooks/exploration/rag_chat_opd.ipynb](notebooks/exploration/rag_chat_opd.ipynb)):
V1 Dense / V2 BM25 / V3 Hybrid (reranker di-skip — 61 docs terlalu kecil).

### C. Naive Combined (registry fan-out) — END-TO-END ✅

[src/ragtrial/rag/naive_combined.py](src/ragtrial/rag/naive_combined.py) —
`ask_main()` iterate `SEARCHABLE_CAPABILITIES`, hybrid retrieve per store
(k_per_source=4 → 8 docs), generate dgn `PROMPT_COMBINED`.

### D. Agentic (LangGraph routed) — END-TO-END ✅

[src/ragtrial/rag/agentic.py](src/ragtrial/rag/agentic.py) — router LLM call
classify ke `<cap.name>` / `both` / `none`, dispatch ke node generic
(`retrieve_single` / `retrieve_all` / `skip_retrieve`) → generate.

### E. Comparison ✅

[notebooks/reports/compare_agentic_vs_naive.ipynb](notebooks/reports/compare_agentic_vs_naive.ipynb)
— 15 query × 4 pipeline (agentic, naive_dukcapil, naive_opd, naive_combined).
**Verdict**: di dataset kecil (150+61 docs), naive_combined ~1.3s lebih cepat
(router overhead 2.5s > ekstra retrieve cost). Agentic menang besar di `none` query.

---

## 4. Decisions & insights

| Topik | Decision | Reasoning |
|---|---|---|
| Chunking BAB II Dukcapil | Q&A-aware regex | Tiap Q+A self-contained, jawaban lintas halaman |
| Chunking OPD | No chunking | Data atomic, ga ada narasi |
| Embedding | Gemini `gemini-embedding-2` 768d | Multilingual BI |
| Vector store per source | Pisah | Schema metadata + chunking strategy beda |
| Reranker untuk OPD | Skip | 61 docs terlalu kecil, latency ga worth |
| Format export | Pickle + JSON | PKL preserve Document, JSON readable |
| Default RAG mode | Agentic | Lebih ekspressif walaupun naive combined sedikit lebih cepat di dataset ini |

---

## 5. To-Do

### Immediate
- [ ] **Tahap 4 (remeh)**: nbstripout setup, `.gitignore` cleanup
- [ ] **Run full eval** (`uv run python -m eval.run_eval --systems agentic naive`) untuk dapat baseline metrik kuantitatif post-refactor

### Short-term
- [ ] **Decision: PERDA + Analisis-hukum** — masuk pipeline? Kalau ya, butuh
      strategy chunking baru (legal doc — pasal/ayat split). Architecture sudah
      siap (tinggal `instances/perda.py` + chunking + register).
- [ ] **Out-of-scope handling**: test lebih banyak edge case selain "resep nasi goreng"
- [ ] **Reranker latency investigation** Dukcapil V2/V4 (~70-230s)

### Medium-term
- [ ] **Text-to-SQL capability** (rencana awal user) — implement `SqlToolCapability`
      yang inherit `Capability` ABC, otomatis ke-route oleh agentic
- [ ] **Conversation memory / multi-turn chat** — sekarang masih single-turn
- [ ] **Evaluasi kuantitatif** — testset sudah ada (`eval/testset.json`),
      tinggal scale up (sekarang 27 query)

### Open questions
1. **Scope final RAG**: cuma 2 dokumen atau ditambah PERDA + analisis hukum?
2. **Output akhir**: notebook eksperimen, atau jadi aplikasi (Streamlit/FastAPI)?
3. **Variant final Dukcapil**: V3 (Hybrid) tampak paling balanced — setuju pakai sebagai default?

---

## 6. Refactor history

- **Tahap 1** (commit `13661b2`): src/ragtrial package + Capability registry.
  Detail: [docs/REFACTOR_TAHAP1.md](docs/REFACTOR_TAHAP1.md).
- **Tahap 2** (commit `39eebed`): reorg data/, notebooks/, scripts/, function-ify preprocessing.
- **Tahap 3** (this commit): polish — README proper, PLAN archived, PROGRESS refreshed.
- **Tahap 4** (pending): nbstripout, .gitignore cleanup.
- **Tahap 5** (future): SqlToolCapability + tambah data source baru.
