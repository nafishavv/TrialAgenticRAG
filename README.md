# RAGTrial

Agentic & naive RAG untuk layanan publik Kab. Batang. Sumber data saat ini:
- **Dukcapil** — Buku Saku administrasi kependudukan (KTP, KK, akta, dll)
- **OPD** — Direktori Organisasi Perangkat Daerah (alamat, telepon, email)

Arsitektur pakai **Capability registry** — tambah sumber data baru atau tool
(text-to-sql, web search, dll) tanpa menyentuh pipeline RAG-nya.

---

## Setup

```bash
# Install package (editable)
uv pip install -e .

# .env di root project harus berisi:
#   GEMINI_API_KEY=...   (atau GOOGLE_API_KEY=...)
```

## Quick start

```bash
# Tanya sesuatu (default: agentic)
uv run python scripts/ask.py "Apa syarat KTP elektronik?"

# Pakai pipeline naive (semua source di-query sekaligus)
uv run python scripts/ask.py "Alamat Disdukcapil?" --mode naive

# (Re-)preprocess raw PDF -> data/processed/<source>.{pkl,json}
uv run python scripts/preprocess.py --source dukcapil
uv run python scripts/preprocess.py --source all

# (Re-)build vector store dari data/processed
uv run python scripts/build_vectorstore.py --source opd
```

## Evaluation

```bash
# Run full eval (~5-10 menit, butuh Gemini API quota)
uv run python -m eval.run_eval --systems agentic naive

# Smoke test (5 query, no LLM judge)
uv run python -m eval.run_eval --systems agentic naive --limit 5 --no-judge

# Aggregate hasil + breakdown per dimensi
uv run python -m eval.analyze --systems agentic naive \
    --breakdown query_type difficulty
```

---

## Struktur

```
RAGTrial/
├── src/ragtrial/                  # package importable
│   ├── config.py                  # absolute paths + load_env()
│   ├── llm.py                     # singleton llm + embeddings
│   ├── capabilities/              # registry generic (vector source + future tools)
│   │   ├── base.py                # Capability ABC + format_context
│   │   ├── vector_source.py       # VectorSourceCapability (Chroma + hybrid)
│   │   └── instances/{dukcapil,opd}.py
│   ├── preprocessing/{dukcapil,opd}.py  # raw PDF -> cleaned Documents
│   ├── chunking/{dukcapil,opd}.py       # cleaned Documents -> embedding chunks
│   ├── vectorstore/builder.py     # generic Chroma builder + retry
│   └── rag/
│       ├── prompts.py             # PROMPT_COMBINED, PROMPT_SINGLE, ROUTER_PROMPT
│       ├── naive_combined.py      # ask_main() — iterate registry, fan-out
│       └── agentic.py             # ask_agentic() — LangGraph router + dispatch
├── scripts/                       # thin CLI wrappers
│   ├── preprocess.py
│   ├── build_vectorstore.py
│   └── ask.py
├── notebooks/
│   ├── exploration/               # per-source experiments (4 variant, dll)
│   ├── reports/                   # demo + comparison notebooks
│   └── archive/                   # stale notebook, kept untuk referensi
├── eval/                          # evaluation harness (run_eval, analyze)
├── data/
│   ├── raw/{dukcapil,opd,unprocessed}/    # source PDFs
│   ├── processed/<source>.{pkl,json}       # cleaned Documents
│   └── vector_stores/<source>/             # Chroma persistent dir
├── docs/                          # REFACTOR_TAHAP1.md, archive/
└── pyproject.toml, README.md, PROGRESS.md
```

---

## Menambah sumber data baru (worked example)

Misal mau tambah **PERDA Kab. Batang**:

### 1. Letakkan raw PDF
```
data/raw/perda/<file>.pdf
```

### 2. Tulis preprocessing
```python
# src/ragtrial/preprocessing/perda.py
from pathlib import Path
from typing import List
from langchain_core.documents import Document

def preprocess(pdf_path: Path) -> List[Document]:
    # ...load + clean...
    return docs
```

### 3. Tulis chunking strategy
```python
# src/ragtrial/chunking/perda.py
def chunk_for_vectorstore(docs):
    # split per pasal, atau RecursiveCharacterTextSplitter, dll
    return chunks
```

### 4. Buat capability instance
```python
# src/ragtrial/capabilities/instances/perda.py
from ragtrial.capabilities.vector_source import VectorSourceCapability
from ragtrial.config import VECTOR_STORE_DIR

perda_capability = VectorSourceCapability(
    name="perda",
    description="Peraturan Daerah Kab. Batang — pasal, ayat, sanksi.",
    collection_name="perda_articles",
    persist_directory=VECTOR_STORE_DIR / "perda",
    router_examples=["Apa sanksi parkir liar menurut Perda?"],
    strategy="hybrid",
)
```

### 5. Daftarkan
```python
# src/ragtrial/capabilities/__init__.py
from ragtrial.capabilities.instances.perda import perda_capability

CAPABILITIES = {
    dukcapil_capability.name: dukcapil_capability,
    opd_capability.name: opd_capability,
    perda_capability.name: perda_capability,  # ← satu baris
}
```

### 6. Tambah ke CLI dispatch
```python
# scripts/preprocess.py + scripts/build_vectorstore.py
PIPELINES["perda"] = (preprocess_perda, RAW_DIR / "perda" / "...pdf", ...)
```

### 7. Run
```bash
uv run python scripts/preprocess.py --source perda
uv run python scripts/build_vectorstore.py --source perda
uv run python scripts/ask.py "Apa sanksi parkir liar menurut Perda?"
```

**Itu doang.** `naive_combined`, `agentic`, dan router prompt auto-pickup —
tidak ada perubahan di pipeline RAG.

---

## Menambah tool (non-vector, mis. text-to-sql)

Implement `Capability` ABC di [src/ragtrial/capabilities/base.py](src/ragtrial/capabilities/base.py):

```python
from ragtrial.capabilities.base import Capability
from langchain_core.documents import Document

class SqlToolCapability(Capability):
    name = "sql_perda"
    description = "Query database Perda untuk statistik / aggregation."
    searchable = True

    def invoke(self, query: str, k: int = 5) -> list[Document]:
        sql = self._llm_to_sql(query)
        rows = self._execute(sql)
        return self._tag([
            Document(page_content=self._format_row(r), metadata={"sql": sql, **r})
            for r in rows[:k]
        ])

    def format_header(self, doc, idx):
        return f"[SQL {idx}] query={doc.metadata['sql'][:60]}"
```

Daftarkan di `CAPABILITIES` — agentic router otomatis dapat category baru.

---

## Catatan

- **Vector store dipisah per source** — schema metadata beda, chunking strategy
  beda. Combined retrieval di-query-time via registry, bukan via merged store.
- **Path absolute via `Path(__file__)`** — semua script jalan dari cwd manapun.
- **`uv run python -m eval.run_eval`** harus jalan dari project root.
- **PROGRESS.md** = state report; **docs/REFACTOR_TAHAP1.md** = refactor history.
