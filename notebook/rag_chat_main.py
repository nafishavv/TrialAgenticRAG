"""Naive Combined RAG — query DUA vector store (dukcapil + opd) sekaligus, tanpa router.

Tujuan: baseline apple-to-apple buat compare vs agentic RAG.
- Retrieve hybrid (BM25 + dense) dari masing-masing store, ambil k_per_store=4 → total 8 docs.
- Tag _source di metadata supaya format_context bisa render header benar.
- Generate pakai PROMPT_COMBINED yang generic (tidak force struktur dua-bagian).

Path-relative: assume caller cwd = sibling dir (mis. notebook/ atau agenticrag/),
sama dgn pola agentic_rag.py.
"""

import os
import time
from typing import Any, Dict, List

from dotenv import load_dotenv

load_dotenv("../.env")
api_key = os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")
assert api_key, "GEMINI_API_KEY tidak ditemukan di .env"
os.environ["GOOGLE_API_KEY"] = api_key

from langchain_chroma import Chroma
from langchain_classic.retrievers import EnsembleRetriever
from langchain_community.retrievers import BM25Retriever
from langchain_core.documents import Document
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings

# ============ LLM + Embeddings ============
llm = ChatGoogleGenerativeAI(
    model="gemini-2.5-flash",
    temperature=0.1,
    max_tokens=1024,
)

embeddings = GoogleGenerativeAIEmbeddings(
    model="models/gemini-embedding-2",
    task_type="retrieval_query",
    output_dimensionality=768,
)

# ============ Vector Stores ============
vs_dukcapil = Chroma(
    collection_name="dukcapil_qa",
    embedding_function=embeddings,
    persist_directory="../data/dukcapil_vector_store",
)

vs_opd = Chroma(
    collection_name="opd_directory",
    embedding_function=embeddings,
    persist_directory="../data/opd_vector_store",
)

# Materialize ke memori untuk BM25 (langchain_community BM25Retriever in-memory)
_raw_d = vs_dukcapil.get()
dukcapil_docs_all = [
    Document(page_content=doc, metadata=meta)
    for doc, meta in zip(_raw_d["documents"], _raw_d["metadatas"])
]
_raw_o = vs_opd.get()
opd_docs_all = [
    Document(page_content=doc, metadata=meta)
    for doc, meta in zip(_raw_o["documents"], _raw_o["metadatas"])
]

# ============ Hybrid Retrievers per Store ============
_bm25_d = BM25Retriever.from_documents(dukcapil_docs_all)
_bm25_d.k = 10
_dense_d = vs_dukcapil.as_retriever(search_kwargs={"k": 10})
_ensemble_d = EnsembleRetriever(retrievers=[_bm25_d, _dense_d], weights=[0.5, 0.5])

_bm25_o = BM25Retriever.from_documents(opd_docs_all)
_bm25_o.k = 10
_dense_o = vs_opd.as_retriever(search_kwargs={"k": 10})
_ensemble_o = EnsembleRetriever(retrievers=[_bm25_o, _dense_o], weights=[0.5, 0.5])


def retrieve_dukcapil_hybrid(query: str, k: int = 4) -> List[Document]:
    return _ensemble_d.invoke(query)[:k]


def retrieve_opd_hybrid(query: str, k: int = 4) -> List[Document]:
    return _ensemble_o.invoke(query)[:k]


def retrieve_combined(query: str, k_per_store: int = 4) -> List[Document]:
    """Fetch top-k dari KEDUA store sekaligus, tag _source, concat."""
    docs_d = retrieve_dukcapil_hybrid(query, k=k_per_store)
    docs_o = retrieve_opd_hybrid(query, k=k_per_store)
    for d in docs_d:
        d.metadata = {**d.metadata, "_source": "dukcapil"}
    for d in docs_o:
        d.metadata = {**d.metadata, "_source": "opd"}
    return docs_d + docs_o


# ============ Prompt ============
PROMPT_COMBINED = """Kamu asisten layanan publik Kab. Batang. Kamu punya akses ke DUA sumber konteks:
- Buku Saku Dukcapil — prosedur/syarat administrasi kependudukan (KTP, KK, akta kelahiran/kematian, NIK, pindah domisili, KITAP, dll).
- Direktori OPD Kab. Batang — alamat kantor, nomor telepon, email, struktur dinas/bagian/kecamatan/kelurahan.

ATURAN:
1. Jawab HANYA berdasarkan konteks. Jangan tambah info dari pengetahuan umum.
2. Pilih sumber yang relevan dengan pertanyaan. Abaikan konteks yang tidak relevan (jangan paksakan dipakai).
3. Untuk info OPD, sebutkan sumber: [Sumber: <nama_opd>, nomor <nomor>]
4. Untuk info Dukcapil, sebutkan sumber kalau ada section/halaman di header konteks.
5. Kalau pertanyaan hanya butuh satu jenis info (mis. cuma alamat, atau cuma prosedur), jawab langsung TANPA memaksa struktur dua-bagian.
6. Kalau pertanyaan butuh DUA-DUANYA (prosedur + kontak), berikan keduanya secara ringkas.
7. Kalau informasi tidak ada di konteks manapun, jawab: "Maaf, informasi tidak ditemukan dalam buku saku Dukcapil maupun direktori OPD."
8. Bahasa Indonesia jelas & ringkas.

KONTEKS:
{context}

PERTANYAAN: {question}

JAWABAN:"""


def format_context_combined(docs: List[Document]) -> str:
    parts = []
    for i, d in enumerate(docs, 1):
        m = d.metadata or {}
        src = m.get("_source", "")
        if src == "opd":
            parent = (
                f" (bagian dari {m['parent_opd']})" if m.get("parent_opd") else ""
            )
            header = (
                f"[OPD {i}: {m.get('nama_opd', '?')}{parent}, "
                f"nomor {m.get('nomor', '?')}, tipe {m.get('tipe', '?')}]"
            )
        elif src == "dukcapil":
            section = m.get("section", "?")
            page = m.get("page", "?")
            header = f"[Dukcapil {i}: section={section}, hal={page}]"
        else:
            header = f"[Sumber {i}]"
        parts.append(f"{header}\n{d.page_content}")
    return "\n\n---\n\n".join(parts)


# ============ Entry Point ============
def ask_main(
    question: str, k_per_store: int = 4, verbose: bool = True
) -> Dict[str, Any]:
    """Naive combined RAG: retrieve dari 2 store, no router, generate dgn PROMPT_COMBINED."""
    t0 = time.perf_counter()

    t_r0 = time.perf_counter()
    docs = retrieve_combined(question, k_per_store=k_per_store)
    t_retrieve = time.perf_counter() - t_r0

    t_g0 = time.perf_counter()
    ctx = format_context_combined(docs)
    answer = llm.invoke(
        PROMPT_COMBINED.format(context=ctx, question=question)
    ).content
    t_generate = time.perf_counter() - t_g0

    total = time.perf_counter() - t0
    result = {
        "question": question,
        "source_used": "combined",
        "documents": docs,
        "answer": answer,
        "timings": {
            "route": 0.0,
            "retrieve": t_retrieve,
            "generate": t_generate,
            "total": total,
        },
    }

    if verbose:
        t = result["timings"]
        n_d = sum(1 for d in docs if d.metadata.get("_source") == "dukcapil")
        n_o = sum(1 for d in docs if d.metadata.get("_source") == "opd")
        print(f"Q: {question}")
        print(f"   Docs: {n_d} dukcapil + {n_o} opd = {len(docs)} total")
        print(
            f"   Timing — retrieve: {t['retrieve']:.2f}s | "
            f"generate: {t['generate']:.2f}s | TOTAL: {t['total']:.2f}s"
        )
        print(
            f"   Answer: {answer[:400]}{'...' if len(answer) > 400 else ''}\n"
        )
    return result
