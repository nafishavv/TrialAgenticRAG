# RAGTrial — Refactor Notes (Local, Not Committed)

Dokumentasi lengkap proses refactor 2026-05-20. Sengaja **tidak di-commit**
(stay untracked di git) sebagai catatan pribadi.

**Branch**: `refactor/package-layout`
**Base**: `main` @ `313e54a`
**4 commit local-only, belum di-push ke remote.**

---

## 0. Initial Assessment (sebelum refactor)

Sebelum mulai, aku audit codebase dan identify 15 issue diurut krusial → remeh:

### 🔴 Krusial (akan menghambat scale-up)

1. **Tidak ada source registry** — `rag_chat_main.py` dan `agentic_rag.py` hardcode 2 vector store. Tambah PERDA = duplikasi ~80 LOC di dua module. Single biggest bottleneck.
2. **Code & notebook bercampur di folder yang sama** — `notebook/rag_chat_main.py`, `agenticrag/agentic_rag.py` itu Python module, bukan notebook. Akibatnya `eval/run_eval.py` punya `os.chdir` hack tengah eksekusi.
3. **Path handling rapuh** — `load_dotenv("../.env")`, `persist_directory="../data/dukcapil_vector_store"` di tiap module. Silent fail kalau run dari cwd berbeda.
4. **Duplikasi besar antar module** — LLM/embeddings/vectorstore/BM25/format_context diulang di `agentic_rag.py` dan `rag_chat_main.py`. ~80 LOC redundan.

### 🟠 Penting

5. **Naming asimetris** — `preprocessing.ipynb` (cuma dukcapil!) vs `preprocessing_opd.ipynb`. Implicit "dukcapil" di file tanpa suffix.
6. **`notebook/` jadi kitchen sink** — 10 file campur (preprocess, build, rag_chat 4-variant, main, stale `dukcapil.ipynb` 1.8MB, `document.ipynb` 969KB).
7. **Vector store layout jelek** — 3 folder sejajar di root `data/`, plus legacy `data/vector_store/` belum dihapus.
8. **File stale tertinggal** — legacy vector store, sample text files, old notebooks, `main.py` placeholder.

### 🟡 Polish

9. README.md cuma 1 baris (`# TrialAgenticRAG`)
10. `PLAN_*.md` di root padahal sudah selesai dieksekusi
11. ~~`langchain_classic.retrievers` deprecated~~ ← **asesmen salah**, ini lokasi kanonik di langchain 1.x
12. `PROGRESS.md` outdated

### 🟢 Remeh

13. `data/pdf/` campur (OPD + 2 PDF unprocessed)
14. `agenticrag/comparison_results.csv` di-commit, harusnya di `eval/results/`
15. Notebook output ke-commit, diff noisy. `nbstripout` solution.

---

## 1. Plan & Decision (sebelum eksekusi)

Aku tanya user 4 keputusan kunci sebelum eksekusi:

| Keputusan | Pilihan user |
|---|---|
| Layout package | `src/ragtrial/` (standard Python src layout) |
| File stale | Archive ke `notebooks/archive/` (bukan hapus total) |
| Future data | Campuran (narasi + structured), **plus rencana text-to-sql tool integration** |
| Eksekusi | Bertahap, satu tahap satu commit |

Yang paling mengubah arsitektur: **rencana text-to-sql**. Mengubah desain dari
"Source registry" (vector-only) → "**Capability registry**" (generic ABC yang
unify vector stores + future tools).

Plan dibagi 5 tahap:
- **Tahap 1** — Arsitektur (src/, Capability ABC, refactor module)
- **Tahap 2** — File hygiene (reorg data/, notebooks/, function-ify, CLI)
- **Tahap 3** — Polish (README, PROGRESS, archive PLAN)
- **Tahap 4** — Remeh (nbstripout, .gitignore)
- **Tahap 5** — Future (SqlToolCapability, tambah source baru)

---

## 2. Tahap 1 — src/ragtrial package + Capability registry

**Commit**: `13661b2`
**Doc**: [docs/REFACTOR_TAHAP1.md](docs/REFACTOR_TAHAP1.md)

### Yang dibuat

| File | Isi |
|---|---|
| `src/ragtrial/config.py` | Absolute paths via `Path(__file__)`, `load_env()` |
| `src/ragtrial/llm.py` | Singleton `llm` + `embeddings`, `make_judge_llm()` |
| `src/ragtrial/capabilities/base.py` | `Capability` ABC + `format_context()` (dispatch header per source) |
| `src/ragtrial/capabilities/vector_source.py` | `VectorSourceCapability` (Chroma + BM25 hybrid, lazy init) |
| `src/ragtrial/capabilities/instances/{dukcapil,opd}.py` | Config-only instances |
| `src/ragtrial/capabilities/__init__.py` | `CAPABILITIES` registry |
| `src/ragtrial/rag/prompts.py` | Semua PROMPT + `build_router_prompt()` generator dari registry |
| `src/ragtrial/rag/naive_combined.py` | `ask_main()` iterate registry |
| `src/ragtrial/rag/agentic.py` | LangGraph: route → retrieve_single/all/skip → generate |

### Yang dimodifikasi

- `pyproject.toml` — tambah `[build-system]` + `[tool.hatch.build.targets.wheel] packages = ["src/ragtrial"]`
- `eval/eval_core.py` — hapus dotenv duplikat, pakai `ragtrial.llm.make_judge_llm`
- `eval/run_eval.py` — **hapus `os.chdir` hack** + `sys.path` manipulation; langsung `from ragtrial.rag.* import *`
- `notebook/rag_chat_main.py` & `agenticrag/agentic_rag.py` — diubah jadi thin shim (back-compat)

### Win utama

Tambah source baru (e.g. PERDA) = 1 file config + 1 entry registry. **Zero perubahan** di `naive_combined.py`, `agentic.py`, `prompts.py`. Router auto-pickup category baru.

### Smoke tests

- Import semua module ✅
- Retrieval per capability ✅
- `ask_main()` end-to-end ✅ (4.86s, 6 docs, sumber benar)
- `ask_agentic()` 4 routes (dukcapil/opd/both/none) ✅ semua correct
- `python -m eval.run_eval` ✅ jalan tanpa chdir

---

## 3. Tahap 2 — Reorg data/, notebooks/, function-ify

**Commit**: `39eebed`

### Yang dipindahkan

```
data/cleaned_docs.pkl              → data/processed/dukcapil.pkl
data/cleaned_opd_docs.pkl          → data/processed/opd.pkl
data/dukcapil_pdf/<file>.pdf       → data/raw/dukcapil/<file>.pdf
data/pdf/Nama dan Alamat OPD...    → data/raw/opd/...
data/pdf/PERDA...                  → data/raw/unprocessed/PERDA...
data/pdf/Analisis-hukum...         → data/raw/unprocessed/Analisis-hukum...
data/dukcapil_vector_store/        → data/vector_stores/dukcapil/
data/opd_vector_store/             → data/vector_stores/opd/

notebook/preprocessing.ipynb       → notebooks/exploration/preprocessing_dukcapil.ipynb
notebook/preprocessing_opd.ipynb   → notebooks/exploration/preprocessing_opd.ipynb
notebook/build_vectorstore.ipynb   → notebooks/exploration/build_vectorstore_dukcapil.ipynb
notebook/build_vectorstore_opd.ipynb → notebooks/exploration/build_vectorstore_opd.ipynb
notebook/rag_chat.ipynb            → notebooks/exploration/rag_chat_dukcapil.ipynb
notebook/rag_chat_opd.ipynb        → notebooks/exploration/rag_chat_opd.ipynb
notebook/rag_chat_main.ipynb       → notebooks/reports/rag_chat_main.ipynb
agenticrag/2-agentic_router.ipynb  → notebooks/exploration/agentic_router.ipynb
agenticrag/3-compare_*.ipynb       → notebooks/reports/compare_agentic_vs_naive.ipynb
agenticrag/comparison_results.csv  → eval/results/comparison_results.csv

notebook/dukcapil.ipynb            → notebooks/archive/dukcapil.ipynb
notebook/document.ipynb            → notebooks/archive/document.ipynb
agenticrag/1-agenticrag.ipynb      → notebooks/archive/agenticrag_v1.ipynb
```

### Yang dihapus

- `main.py` (placeholder)
- `data/text_files/{ml_sample,py_sample}.txt`
- `data/vector_store/` (legacy chroma)
- Shim `notebook/rag_chat_main.py`, `agenticrag/agentic_rag.py` (notebook udah pakai import dari ragtrial.*)
- Folder kosong `notebook/`, `agenticrag/`

### Yang dibuat baru

| File | Isi |
|---|---|
| `src/ragtrial/preprocessing/__init__.py` | `save_docs()` shared helper (pkl + json + round-trip verify) |
| `src/ragtrial/preprocessing/dukcapil.py` | `load_pdf`, `filter_pages`, `clean_page`, `tag_section`, `preprocess()` |
| `src/ragtrial/preprocessing/opd.py` | `extract_raw_rows`, `merge_continuation_rows`, parsers, `preprocess()` |
| `src/ragtrial/chunking/dukcapil.py` | Q&A regex split (BAB II) + narrative split |
| `src/ragtrial/chunking/opd.py` | Identity (no chunking) |
| `src/ragtrial/vectorstore/builder.py` | Generic Chroma build dengan batched embedding + retry |
| `scripts/preprocess.py` | CLI `--source dukcapil\|opd\|all` |
| `scripts/build_vectorstore.py` | CLI `--source dukcapil\|opd\|all` |
| `scripts/ask.py` | CLI `"pertanyaan" --mode naive\|agentic` |
| `scripts/_migrate_notebooks.py` | One-shot script: rewrite import lama di 2 notebook `notebooks/reports/` ke `ragtrial.*` |
| `src/ragtrial/config.py` | Updated path constants ke layout baru |

### Smoke tests

- `python scripts/preprocess.py --source all` → 258 dukcapil + 61 opd docs ✅
- Chunking parity → 150 dukcapil (139 Q&A + 11 narrative) + 61 opd ✅ (match baseline PROGRESS.md)
- `python scripts/ask.py "..."` ✅ correct answer via agentic route
- `python -m eval.run_eval --limit 2 --no-judge` ✅ jalan dari root tanpa chdir

---

## 4. Tahap 3 — Polish

**Commit**: `5242064`

### Yang dilakukan

1. **`README.md`** — rewrite proper:
   - Setup (uv pip install, .env)
   - Quick start (3 CLI scripts)
   - Eval (run + analyze)
   - Struktur folder lengkap dengan annotation
   - **Worked example tambah source baru** (PERDA, 7 langkah)
   - Worked example tambah tool baru (SqlToolCapability)

2. **`PROGRESS.md`** — refresh total:
   - Reflect post-refactor file paths
   - Section "Arsitektur (post-refactor)" baru
   - Section "Refactor history" dengan commit hashes
   - Decisions table updated

3. **`PLAN_*.md` archive**:
   - `PLAN_opd_integration.md` → `docs/archive/`
   - `PLAN_preprocessing_opd.md` → `docs/archive/`

### Yang DI-SKIP

- **`langchain_classic` → `langchain`** import — asesmen Tahap 0 salah. Di langchain 1.x, `langchain.retrievers` udah dihapus. `langchain_classic` itu lokasi kanonik untuk `EnsembleRetriever` di versi sekarang. Skip, no change.

---

## 5. Tahap 4 — Remeh

**Commit**: `1a03d68`

### Yang dilakukan

1. **`nbstripout` setup**:
   - Install via `uv pip install nbstripout`
   - Register sebagai git filter via `uv run nbstripout --install --attributes .gitattributes`
   - `.gitattributes` baru: `*.ipynb filter=nbstripout`
   - Strip 6700+ baris cell outputs dari semua notebook yang udah committed

2. **`.gitignore` expanded**:
   ```
   .ipynb_checkpoints/
   .vscode/, .idea/
   eval/results/per_query_*.json
   eval/results/summary_*.json
   .DS_Store, Thumbs.db
   ```

3. **`pyproject.toml`** — tambah dev deps:
   ```toml
   [dependency-groups]
   dev = ["nbstripout>=0.9.1", "matplotlib>=3.10.0", "pandas>=2.2.0"]
   ```
   (matplotlib + pandas dipakai di compare notebook)

4. **`README.md`** — note setup nbstripout untuk fresh clone

### Catatan teknis nbstripout

- `.gitattributes` di-commit (semua clone tahu untuk pakai filter)
- Tapi command filter (`python -m nbstripout`) ada di `.git/config` lokal — **harus di-run sekali per clone**
- Cara verify filter aktif: `git config --get filter.nbstripout.clean` harus return path ke python

---

## 6. Tahap 5 — Future (BELUM DIEKSEKUSI)

Ini direncanakan tapi belum dimulai. Yang dimaksud:

### A. Tambah `SqlToolCapability`

Implement `Capability` ABC untuk text-to-sql. Skeleton:

```python
# src/ragtrial/capabilities/sql_tool.py
class SqlToolCapability(Capability):
    name: str
    description: str
    db_path: str

    def invoke(self, query: str, k: int = 5) -> List[Document]:
        sql = self._llm_to_sql(query)        # Gemini call
        rows = self._execute_sql(sql)        # sqlite/duckdb
        return self._tag([
            Document(
                page_content=self._format_row(r),
                metadata={"sql": sql, "row_index": i, **r}
            )
            for i, r in enumerate(rows[:k])
        ])

    def format_header(self, doc, idx):
        return f"[SQL {idx}] query={doc.metadata['sql'][:60]}..."
```

Register di `CAPABILITIES` → agentic router otomatis dapat category baru tanpa ngubah graph.

### B. Tambah data source baru

Misal PERDA. Sudah punya PDF di `data/raw/unprocessed/`. Worked example
lengkap ada di [README.md](README.md). Effort estimasi: ~1-2 jam untuk
preprocessing + chunking strategy yang tepat untuk legal doc (pasal/ayat split).

### C. Possibly: Streamlit / FastAPI wrapper

Belum diputuskan. Open question di [PROGRESS.md](PROGRESS.md).

### D. Conversation memory / multi-turn

Sekarang masih single-turn. Butuh chat history handling di LangGraph state.

---

## 7. State akhir branch `refactor/package-layout`

```
1a03d68 refactor tahap 4 - nbstripout + .gitignore polish
5242064 refactor tahap 3 - polish README, PROGRESS, archive plan docs
39eebed refactor tahap 2 - reorg data/, notebooks/, scripts/, function-ify preprocessing
13661b2 refactor tahap 1 - src/ragtrial package + capability registry
313e54a (main) udah bikin agentic rag dan comparison pake 2 source
```

**Status remote**: belum di-push. Local-only.

### Pilihan selanjutnya

1. **Merge ke main**: `git checkout main && git merge refactor/package-layout`
2. **Push branch + bikin PR**: `git push -u origin refactor/package-layout`
3. **Lanjut Tahap 5** di branch yang sama atau branch baru

---

## 8. Yang aku salah / skip

Catat untuk referensi pribadi:

1. **`langchain_classic` deprecated** — salah. Itu lokasi kanonik di langchain 1.x.
2. **Vector store .bin/.sqlite3 modifications** — selama refactor, file ini sering ke-touch karena Chroma open/close. Berkali-kali muncul di `git status`. Sengaja skip dari commit (pre-existing dirty state, bukan dari refactor work).
3. **Smoke test eval per_query files** — beberapa kali overwrite baseline lama saat smoke test. Sekarang udah di-gitignore.
4. **`uv sync` removed matplotlib + pandas** — pertama kali sync setelah tambah dev-deps, uv considered mereka extras. Fix: add ke `dev` group explicitly.
5. **Windows file locks** — saat move vector_store dir, ada Python kernel yang masih hold sqlite handle. User harus restart Jupyter dulu.
6. **Notebooks at new paths broke old imports** — `notebooks/reports/*.ipynb` import `from rag_chat_main import ...` dan `from agentic_rag import ...`. Aku tulis `scripts/_migrate_notebooks.py` untuk rewrite ke `from ragtrial.rag.* import ...`. Script masih ada di repo (sekali jalan, bisa dihapus nanti).

---

## 9. File ini

Sengaja **tidak di-commit** (gak di `git add`-in). Sebagai catatan local-only.
Kalau mau di-commit: `git add REFACTOR_NOTES.md` lalu commit dengan pesan apa
saja. Kalau mau tidak pernah muncul di tracked git status, tambahin ke
`.gitignore`: `REFACTOR_NOTES.md`.
