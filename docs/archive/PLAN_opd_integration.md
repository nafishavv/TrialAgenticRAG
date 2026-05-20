# Plan: JSON Export + OPD Vector Store & RAG Chat

## Context

Saat ini ada 2 preprocessing notebook yang menghasilkan pickle:
- [notebook/preprocessing.ipynb](notebook/preprocessing.ipynb) → `data/cleaned_docs.pkl` (258 halaman buku saku Dukcapil)
- [notebook/preprocessing_opd.ipynb](notebook/preprocessing_opd.ipynb) → `data/cleaned_opd_docs.pkl` (61 record OPD Kab Batang)

Dua kebutuhan:
1. **JSON export** untuk kedua preprocessing — supaya bisa di-inspect manual & language-agnostic (selain pickle yang sudah ada). Isi JSON harus identik dengan pickle (1:1).
2. **OPD belum masuk ke vector store** dan belum ada mekanisme retrieval/chat. Vector store dukcapil sudah jalan di [notebook/build_vectorstore.ipynb](notebook/build_vectorstore.ipynb) → `data/dukcapil_vector_store/` (collection `dukcapil_qa`) dan retrieval di [notebook/rag_chat.ipynb](notebook/rag_chat.ipynb) sudah punya 4 variant (V1–V4).

**Decisions (sudah dikonfirmasi):**
- OPD vector store **dipisah** dari Dukcapil → `data/opd_vector_store/`, collection `opd_directory`.
- Bikin notebook baru `rag_chat_opd.ipynb` dulu (eksperimen terpisah); unified router-based chat ditunda untuk milestone berikutnya.
- Format JSON = list of `{"page_content": str, "metadata": dict}` — 1:1 dengan Document yang dipickle.

---

## Scope

### 1. JSON Export — `preprocessing.ipynb`

Tambah 1 cell baru SETELAH cell `export-pickle`:

```python
import json

JSON_PATH = Path("../data/cleaned_docs.json")
with open(JSON_PATH, "w", encoding="utf-8") as f:
    json.dump(
        [{"page_content": d.page_content, "metadata": d.metadata} for d in cleaned_docs],
        f, ensure_ascii=False, indent=2,
    )
print(f"Saved {len(cleaned_docs)} docs → {JSON_PATH.resolve()}")

# Verifikasi PKL == JSON
with open(JSON_PATH, "r", encoding="utf-8") as f:
    json_docs = json.load(f)
assert len(json_docs) == len(cleaned_docs)
for i, (d, j) in enumerate(zip(cleaned_docs, json_docs)):
    assert d.page_content == j["page_content"], f"page_content mismatch at idx {i}"
    assert d.metadata == j["metadata"], f"metadata mismatch at idx {i}"
print("✓ PKL ↔ JSON content identical")
```

### 2. JSON Export — `preprocessing_opd.ipynb`

Cell baru setelah cell `export-pickle` — pola sama, ganti variable & path:
- `cleaned_docs` → `cleaned_opd_docs`
- `cleaned_docs.json` → `cleaned_opd_docs.json`

### 3. Bikin `notebook/build_vectorstore_opd.ipynb` (NEW)

Pola mirror [notebook/build_vectorstore.ipynb](notebook/build_vectorstore.ipynb), disederhanakan:
- **Tidak perlu chunking** — tiap record OPD sudah atomic (1 row = 1 Document, ~157 chars rata-rata). Langsung embed.
- **Tidak perlu Q&A splitter / narrative splitter**.

Cell structure:
1. Intro markdown — tujuan & output path.
2. Load — `pickle.load("../data/cleaned_opd_docs.pkl")`. Print count & distribusi `tipe`.
3. Load env + init embeddings — copy dari build_vectorstore.ipynb (`GoogleGenerativeAIEmbeddings`, `models/gemini-embedding-2`, `task_type="retrieval_document"`, `output_dimensionality=768`).
4. Wipe & build — `VS_PATH = Path("../data/opd_vector_store")`; `shutil.rmtree` if exists; `Chroma.from_documents(documents=cleaned_opd_docs, embedding=embeddings, collection_name="opd_directory", persist_directory=str(VS_PATH))`. Cuma 61 docs → 1 batch cukup.
5. Sanity check — 3 query: `"alamat dinas pariwisata"`, `"nomor telp kecamatan batang"`, `"email bagian kesra"`. Print top-3 hasil.

### 4. Bikin `notebook/rag_chat_opd.ipynb` (NEW)

Pola mirror [notebook/rag_chat.ipynb](notebook/rag_chat.ipynb), disesuaikan sifat OPD (lookup-style, 61 docs):

**Retrieval variants** (3 — reranker overkill untuk 61 docs):
- **V1 — Dense (baseline)**: `vectorstore.similarity_search(query, k=k)`
- **V2 — BM25**: pure keyword. Cocok karena query biasanya menyebut nama OPD/lokasi exact.
- **V3 — Hybrid (BM25 + Dense, RRF)**: `EnsembleRetriever` sama seperti V3 di rag_chat.ipynb.

**Prompt template** — adaptasi dari rag_chat.ipynb:
```
Kamu adalah asisten direktori OPD (Organisasi Perangkat Daerah) Kab Batang.
Jawab HANYA berdasarkan konteks di bawah. Kalau OPD yang ditanya tidak ada,
jawab: "OPD tidak ditemukan dalam direktori."
Format sumber: [Sumber: <nama_opd>, nomor <nomor>]
```

**format_context** — sertakan `nama_opd`, `parent_opd`, `tipe` di header sumber.

**Test harness** — sample queries:
- `"Alamat dan nomor telp Sekretariat Daerah"`
- `"Email Bagian Kesejahteraan Rakyat"`
- `"Dinas apa saja yang ada di Kab Batang?"`
- `"Kecamatan Pecalungan alamatnya dimana?"`
- `"Apakah ada RSUD di Batang?"`
- `"Nomor telp Disdukcapil"` (test paraphrase — nama lengkap di buku: "Dinas Kependudukan dan Pencatatan Sipil")

Loop variants × queries, print sources + answer. Tidak perlu latency benchmark (data kecil → semua cepat).

---

## Files

**Modified:**
- [notebook/preprocessing.ipynb](notebook/preprocessing.ipynb) — insert 1 cell setelah `export-pickle`.
- [notebook/preprocessing_opd.ipynb](notebook/preprocessing_opd.ipynb) — insert 1 cell setelah `export-pickle`.

**Created:**
- [notebook/build_vectorstore_opd.ipynb](notebook/build_vectorstore_opd.ipynb)
- [notebook/rag_chat_opd.ipynb](notebook/rag_chat_opd.ipynb)

**Generated artifacts (runtime):**
- `data/cleaned_docs.json`
- `data/cleaned_opd_docs.json`
- `data/opd_vector_store/` (ChromaDB)

---

## Reused Patterns

- Embeddings init → copy dari build_vectorstore.ipynb cell `init-embeddings`.
- LLM init → copy dari rag_chat.ipynb cell `init-llm`.
- `EnsembleRetriever` hybrid pattern → copy dari rag_chat.ipynb cell `v3-setup`.
- `chat()` & `format_context()` → adaptasi dari rag_chat.ipynb cell `chat-fn`.

---

## Verification

1. **JSON export sanity** — assertion (PKL == JSON) di cell baru harus pass. Inspect 1-2 record manual.
2. **Build OPD VS** — jalankan end-to-end. Cek `vectorstore._collection.count() == 61`. Sanity search 3 query → top-1 harus relevant.
3. **RAG chat OPD** — jalankan test harness. Inspeksi manual: query `"Nomor telp Disdukcapil"` (paraphrase) harus return record dengan `nama_opd = "Dinas Kependudukan dan Pencatatan Sipil"` di top-3 untuk V1 (dense) dan V3 (hybrid); V2 (BM25) mungkin miss karena "Disdukcapil" ≠ keyword exact. Catat ini sebagai observasi.
