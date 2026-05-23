# RAGTrial

RAG layanan publik Kab. Batang dengan **3 mode**: naive, enhanced, agentic.
Sumber data saat ini:
- **Dukcapil** — Buku Saku administrasi kependudukan (KTP, KK, akta, dll)
- **OPD** — Direktori Organisasi Perangkat Daerah (alamat, telepon, email)

Arsitektur dibangun untuk **skala naik**: domain heterogen (banyak file per domain),
komponen RAG yang bisa di-swap, dan integrasi tool (text-to-sql / web search) di masa depan.
Detail desain + alasan: **[docs/REFACTOR_3WAY.md](docs/REFACTOR_3WAY.md)**.

---

## Tiga mode

| Mode | Apa | Kontrol alur |
|---|---|---|
| **naive** | 1 collection gabungan → dense top-k → stuff → 1 LLM call | Tetap, minimal (baseline) |
| **enhanced** | pipeline fixed `rewrite → route → retrieve → rerank → generate`, di-config | Developer yang desain |
| **agentic** | tool-calling loop: LLM pilih tool, iterasi, retry, skip | Dinamis, LLM yang putuskan |

---

## Setup

```bash
uv sync && uv pip install -e .

# Register nbstripout git filter (sekali per clone)
uv run nbstripout --install --attributes .gitattributes

# .env di root project:  GEMINI_API_KEY=...  (atau GOOGLE_API_KEY=...)
```

## Quick start

```bash
# Tanya (default: enhanced)
uv run python scripts/ask.py "Apa syarat KTP elektronik?"

# Pilih mode
uv run python scripts/ask.py "Alamat Disdukcapil?" --mode naive
uv run python scripts/ask.py "Urus pindah domisili, ke dinas mana?" --mode agentic

# Chat multi-turn (pilih mode juga)
uv run python scripts/ask.py --chat --mode enhanced

# Pipeline data: preprocess → build store → refresh unified (untuk naive)
uv run python scripts/preprocess.py --source all
uv run python scripts/build_vectorstore.py --source all
uv run python scripts/build_vectorstore.py --source unified

# UI
uv run streamlit run app.py
```

## Enhanced — swap komponen lewat config

```python
from ragtrial.rag.enhanced import build_enhanced, EnhancedRAGConfig

cfg = EnhancedRAGConfig(rewriter="passthrough", router="semantic",
                        retrieval="dense", reranker="none")
rag = build_enhanced(cfg)
result = rag.ask("Apa syarat KTP elektronik?")   # -> RagResult
```
Ganti komponen = ganti 1 field. Preset siap pakai: `fanout_hybrid`, `llm_router_hybrid`
(lihat `PRESETS`). Stub menunggu diisi: `hyde`, `multiquery`, `cross_encoder`.

## Evaluation

```bash
uv run python -m eval.run_eval --systems naive enhanced agentic
uv run python -m eval.run_eval --systems enhanced --limit 5 --no-judge   # smoke
uv run python -m eval.analyze --systems naive enhanced agentic --breakdown query_type difficulty
```
Semua mode mengembalikan `RagResult`, jadi metrik bersifat system-agnostic.

---

## Struktur (ringkas)

```
src/ragtrial/
├── result.py                 # RagResult — kontrak output 3 mode
├── capabilities/             # Capability ABC + VectorSourceCapability + registry
├── sources/<domain>/         # co-located: preprocess + chunk + capability per domain
├── pipeline/                 # stage komposabel enhanced (rewrite/route/retrieve/rerank/generate)
├── vectorstore/              # builder per domain + unified (copy vektor)
├── rag/                      # naive.py | enhanced.py | agentic.py | prompts.py
└── chat/session.py           # ChatSession(mode=…)
scripts/  ask.py · preprocess.py · build_vectorstore.py
eval/     run_eval.py · eval_core.py · analyze.py
data/     raw/<domain>/ · processed/<domain>.{pkl,json} · vector_stores/<domain>/ + _unified/
```

---

## Menambah domain data baru (worked example: `pajak`)

```bash
# 1. taruh file (boleh banyak / nested)
data/raw/pajak/<...>.pdf
```
```python
# 2. src/ragtrial/sources/pajak/preprocess.py
def preprocess(pdf_path) -> list[Document]: ...        # handler per tipe file

# 3. src/ragtrial/sources/pajak/chunk.py
def chunk_for_vectorstore(docs) -> list[Document]: ... # mis. split per pasal

# 4. src/ragtrial/sources/pajak/capability.py
pajak_capability = VectorSourceCapability(
    name="pajak", description="Pajak daerah Kab. Batang ...",
    collection_name="pajak", persist_directory=VECTOR_STORE_DIR / "pajak",
    router_examples=["Berapa tarif PBB?"], strategy="hybrid",
    gold_id_fn=..., citation="sebutkan pasal/ayat.",
)

# 5. src/ragtrial/sources/pajak/__init__.py
def build_documents() -> list[Document]: ...           # walk data/raw/pajak/
source = Source(name="pajak", raw_dir=..., processed_pkl=..., capability=pajak_capability,
                build_documents=build_documents, chunk=chunk_for_vectorstore)
```
```python
# 6. daftarkan di src/ragtrial/sources/__init__.py
from ragtrial.sources.pajak import source as pajak_source
_ALL = [dukcapil_source, opd_source, pajak_source]     # ← satu baris
```
```bash
# 7. build
uv run python scripts/preprocess.py --source pajak
uv run python scripts/build_vectorstore.py --source pajak
uv run python scripts/build_vectorstore.py --source unified
```
**Itu doang.** Nol perubahan di registry/prompts/eval/ketiga mode RAG — semua auto-pickup.

## Menambah tool non-vector (mis. text-to-sql)

Implement `Capability` ABC (`invoke()` balikin `List[Document]`), daftarkan di
`capabilities/registry.py`. Otomatis jadi tool agentic (`search_<name>`) dan ikut enhanced fan-out.

---

## Catatan

- **Vector store dipisah per domain** (schema/chunking beda); naive query `_unified` (copy vektor lintas domain).
- **Path absolut via `Path(__file__)`** — script jalan dari cwd manapun; eval dari project root.
- **`docs/REFACTOR_3WAY.md`** = desain + alasan + cara extend; **PROGRESS.md** = state report.
