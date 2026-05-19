# Refactor Tahap 1 — Package Layout + Capability Registry

**Branch**: `refactor/package-layout`
**Tanggal**: 2026-05-20
**Status**: ✅ selesai, smoke test passed

---

## Tujuan Tahap 1

Memecah bottleneck arsitektur paling parah: setiap penambahan data source baru
sebelumnya butuh duplikasi ~80 LOC di dua module (`agentic_rag.py` +
`rag_chat_main.py`). Sekarang penambahan source = 1 file config + 1 entry di
registry.

---

## File baru — `src/ragtrial/`

| File | Isi |
|---|---|
| `src/ragtrial/__init__.py` | Package marker |
| `src/ragtrial/config.py` | Absolute paths (`PROJECT_ROOT`, `DUKCAPIL_VECTOR_STORE`, dll) + `load_env()` |
| `src/ragtrial/llm.py` | Singleton `llm` & `embeddings` (Gemini); `make_judge_llm()` factory |
| `src/ragtrial/capabilities/base.py` | `Capability` ABC + `format_context()` |
| `src/ragtrial/capabilities/vector_source.py` | `VectorSourceCapability` (Chroma + BM25 hybrid, lazy init) |
| `src/ragtrial/capabilities/instances/dukcapil.py` | Config dukcapil (config-only, no logic) |
| `src/ragtrial/capabilities/instances/opd.py` | Config opd |
| `src/ragtrial/capabilities/__init__.py` | `CAPABILITIES` registry |
| `src/ragtrial/rag/prompts.py` | Semua prompt template + `build_router_prompt()` generator |
| `src/ragtrial/rag/naive_combined.py` | `ask_main()` — iterate registry, fan-out retrieve |
| `src/ragtrial/rag/agentic.py` | `ask_agentic()` + LangGraph generic dispatch |

## File yang dimodifikasi

| File | Perubahan |
|---|---|
| `pyproject.toml` | Tambah build system (hatchling) + package path `src/ragtrial` |
| `eval/eval_core.py` | Hapus dotenv duplikat, pakai `ragtrial.llm.make_judge_llm` |
| `eval/run_eval.py` | **Hapus `os.chdir` hack & `sys.path` manipulation**; import langsung `from ragtrial.rag.* import *` |
| `notebook/rag_chat_main.py` | Diubah jadi **thin shim** yang re-export dari `ragtrial.rag.naive_combined` (back-compat untuk notebook lama) |
| `agenticrag/agentic_rag.py` | Diubah jadi **thin shim** yang re-export dari `ragtrial.rag.agentic` (back-compat) |

## File yang TIDAK berubah (sengaja)

- Notebooks (`*.ipynb`): masih bisa import via shim, ga perlu re-run kecuali kamu mau verify.
- Data PDF, vector store binary, pickle/json hasil preprocessing: tidak disentuh.
- `PROGRESS.md`, `PLAN_*.md`, `README.md`: cleanup di Tahap 3.
- Eval testset & results: tidak disentuh.

---

## API surface — cara pakai package baru

### Tambah source baru (worked example)

Misal mau tambah PERDA:

```python
# src/ragtrial/capabilities/instances/perda.py
from ragtrial.capabilities.vector_source import VectorSourceCapability
from ragtrial.config import DATA_DIR

perda_capability = VectorSourceCapability(
    name="perda",
    description="Peraturan Daerah Kab. Batang — pasal, ayat, sanksi.",
    collection_name="perda_articles",
    persist_directory=DATA_DIR / "perda_vector_store",
    router_examples=["Apa sanksi Perda parkir liar?"],
    strategy="hybrid",
)
```

```python
# src/ragtrial/capabilities/__init__.py
from ragtrial.capabilities.instances.perda import perda_capability

CAPABILITIES = {
    dukcapil_capability.name: dukcapil_capability,
    opd_capability.name: opd_capability,
    perda_capability.name: perda_capability,  # ← satu baris
}
```

**Itu doang.** Router agentic auto-pickup category baru, naive combined auto-include
di fan-out, prompt auto-rendered dari description. **Tidak ada perubahan di**
`naive_combined.py`, `agentic.py`, `prompts.py`.

### Tambah tool (non-vector, mis. text-to-sql)

Implement `Capability` ABC dari [`base.py`](../src/ragtrial/capabilities/base.py):
- `invoke(query, k) -> List[Document]` — Document yang page_content-nya hasil
  formatted SQL row, metadata bawa info struktural.
- `format_header(doc, idx) -> str` — custom header utk prompt.
- Register di `CAPABILITIES`.

### Entry points

```python
from ragtrial.rag.naive_combined import ask_main
from ragtrial.rag.agentic import ask_agentic

ask_main("syarat KTP?")                     # iterate semua searchable
ask_agentic("alamat Dispar?")               # router classify → single dispatch
```

---

## Hal-hal arsitektur penting

### 1. Lazy initialization di `VectorSourceCapability`

Chroma + BM25 di-init pertama kali `invoke()` dipanggil, bukan saat import. Tujuan:
- Import package tidak buka semua DB (cepet, low memory)
- Bisa swap embedding model tanpa break import

Konsekuensi: kalau notebook lama akses `vs_dukcapil` (handle Chroma raw), shim
explicitly call `_ensure_initialized()` supaya handle tersedia.

### 2. `_source` metadata tag

Setiap doc yang lewat `Capability.invoke()` di-stamp `metadata._source = capability.name`.
Ini dipakai `format_context()` untuk dispatch header formatter per source. Existing
prompts (`[Sumber: ...]`) tetap konsisten.

### 3. Router prompt di-generate dari registry

`build_router_prompt()` di [prompts.py](../src/ragtrial/rag/prompts.py) build
categories block + few-shot examples dari `cap.description` + `cap.router_examples`.
Tidak ada hardcoded category list lagi.

### 4. Path handling

Semua path lewat `ragtrial.config` → absolute, resolved dari `__file__`. Boleh
run dari cwd manapun. `eval/run.py` jadi 1-liner import sekarang.

---

## Smoke test results

| Test | Result |
|---|---|
| Import semua module | ✅ |
| `CAPABILITIES["dukcapil"].invoke("syarat KTP", k=2)` | ✅ tagged `_source=dukcapil` |
| `CAPABILITIES["opd"].invoke("Disdukcapil", k=2)` | ✅ retrieved Disdukcapil row |
| `ask_main(...)` end-to-end (Gemini call) | ✅ 4.86s, 6 docs, sumber benar |
| `ask_agentic(...)` 4 routes | ✅ semua route correct (`dukcapil/opd/both/none`) |
| `python -m eval.run_eval --limit 2 --no-judge` | ✅ jalan tanpa chdir |
| `python -m eval.analyze` | ✅ aggregator OK |

---

## ⚠️ Yang HARUS kamu lakukan manual

### Wajib sebelum lanjut Tahap 2

**Tidak ada yang wajib.** Tahap 1 sudah end-to-end working. Kamu bisa langsung
lanjut Tahap 2.

### Optional (kalau mau verify sendiri)

1. **Re-run notebook lama untuk verify shim**:
   - `notebook/rag_chat_main.ipynb`
   - `agenticrag/2-agentic_router.ipynb`
   - `agenticrag/3-compare_agentic_vs_naive.ipynb`

   Mereka import dari `rag_chat_main` / `agentic_rag` (shim) → seharusnya jalan
   tanpa perubahan. Kalau ada cell yang break, kemungkinan akses internal symbol
   yang nggak aku re-export. Kasih tau aku biar ku-tambah ke shim.

2. **Run full eval** (lebih lama, butuh ~5-10 menit + Gemini API):
   ```
   uv run python -m eval.run_eval --systems agentic naive
   uv run python -m eval.analyze --systems agentic naive --breakdown query_type difficulty
   ```
   Bandingkan dengan baseline lama untuk pastikan ga ada regresi.

### Tahap selanjutnya (yang akan aku kerjakan saat kamu kasih sinyal)

**Tahap 2 — File hygiene**
- Reorganize `data/` ke `raw/{dukcapil,opd,unprocessed}/`, `processed/`, `vector_stores/{dukcapil,opd}/`
- Hapus `data/vector_store/` legacy
- Archive stale notebooks ke `notebooks/archive/`: `dukcapil.ipynb`, `document.ipynb`, `1-agenticrag.ipynb`
- Hapus `main.py`, `data/text_files/`
- Rename + pindahin notebook hidup ke `notebooks/{exploration,reports}/`
- Function-ify logic preprocessing dari notebook → `src/ragtrial/preprocessing/{dukcapil,opd}.py`
- Bikin CLI scripts (`scripts/preprocess.py`, `scripts/build_vectorstore.py`, `scripts/ask.py`)

Catatan: setelah Tahap 2 yang reorganize `data/`, path di `config.py` perlu update
(satu file, satu kali edit) dan vector store akan dipindahkan fisiknya. Sebelum
itu jalan, jangan lupa backup atau verify chroma masih readable di lokasi baru.

**Tahap 3 — Polish**: README, archive PLAN_*.md, langchain_classic import, PROGRESS.md refresh.

**Tahap 4 — Remeh**: comparison_results.csv pindah, nbstripout, .gitignore.

**Tahap 5 — Setelah refactor**: tambah `SqlToolCapability` untuk text-to-sql, tambah source data baru.
