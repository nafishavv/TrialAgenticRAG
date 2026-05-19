"""Agentic RAG helper — shared between notebooks."""

import os
import time
import json
from typing import Dict, List, Any
from dotenv import load_dotenv

load_dotenv("../.env")
api_key = os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")
assert api_key, "GEMINI_API_KEY tidak ditemukan di .env"
os.environ["GOOGLE_API_KEY"] = api_key

from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_community.retrievers import BM25Retriever
from langchain_classic.retrievers import EnsembleRetriever
from langgraph.graph import StateGraph, END
from typing import TypedDict, Literal

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

raw_opd = vs_opd.get()
opd_docs_all = [
    Document(page_content=doc, metadata=meta)
    for doc, meta in zip(raw_opd["documents"], raw_opd["metadatas"])
]

# ============ Retrievers ============
def retrieve_dukcapil(query: str, k: int = 5) -> List[Document]:
    return vs_dukcapil.similarity_search(query, k=k)

_bm25_opd = BM25Retriever.from_documents(opd_docs_all)
_bm25_opd.k = 10
_dense_opd = vs_opd.as_retriever(search_kwargs={"k": 10})
_ensemble_opd = EnsembleRetriever(
    retrievers=[_bm25_opd, _dense_opd],
    weights=[0.5, 0.5],
)

def retrieve_opd(query: str, k: int = 5) -> List[Document]:
    return _ensemble_opd.invoke(query)[:k]

# ============ Router ============
ROUTER_PROMPT = """Kamu adalah router untuk chatbot layanan publik Kab. Batang.
Tugasmu: klasifikasi pertanyaan user ke SATU dari empat kategori berikut.

KATEGORI:
- "dukcapil"  → pertanyaan PROSEDUR / SYARAT / aturan administrasi kependudukan (KTP, KK, akta kelahiran, akta kematian, pindah domisili, NIK, KITAP, dll). User pengen tahu CARA atau ATURAN.
- "opd"       → pertanyaan tentang DIREKTORI OPD Kab. Batang: alamat kantor, nomor telepon, email, struktur dinas, daftar bagian/sub-dinas. User pengen INFO KONTAK / LOKASI dinas.
- "both"      → pertanyaan yang butuh DUA-DUANYA: prosedur kependudukan + kontak/alamat dinas yang ngurus (mis. "cara bikin KTP dan ke dinas mana").
- "none"      → di luar cakupan (resep masak, olahraga, hiburan, dll).

CONTOH:
Q: "Apa syarat penerbitan KTP elektronik?"  → {{"route":"dukcapil","reason":"prosedur/syarat KTP"}}
Q: "Alamat Dinas Pariwisata Batang?"        → {{"route":"opd","reason":"alamat kantor dinas"}}
Q: "Mau urus akta kelahiran, ke mana dan apa syaratnya?" → {{"route":"both","reason":"prosedur + lokasi kantor"}}
Q: "Resep nasi goreng?"                     → {{"route":"none","reason":"off-topic"}}
Q: "Nomor telepon Sekretariat Daerah?"      → {{"route":"opd","reason":"kontak dinas"}}
Q: "Bagaimana cara pindah domisili?"        → {{"route":"dukcapil","reason":"prosedur pindah domisili"}}

ATURAN OUTPUT:
- Jawab HANYA JSON valid satu baris: {{"route":"<kategori>","reason":"<alasan singkat>"}}
- Tidak ada teks lain, tidak ada markdown, tidak ada code fence.

Pertanyaan: {question}
JSON:"""

def route_query(question: str) -> Dict[str, str]:
    """Returns {'route': 'dukcapil|opd|both|none', 'reason': str}"""
    prompt = ROUTER_PROMPT.format(question=question)
    raw = llm.invoke(prompt).content.strip()
    if raw.startswith("```"):
        raw = raw.strip("`")
        if raw.startswith("json"):
            raw = raw[4:]
        raw = raw.strip()
    try:
        parsed = json.loads(raw)
        route = parsed.get("route", "none")
        if route not in {"dukcapil", "opd", "both", "none"}:
            route = "none"
        return {"route": route, "reason": parsed.get("reason", "")}
    except Exception as e:
        return {"route": "none", "reason": f"parse_error: {e}; raw={raw[:80]}"}

# ============ Prompts ============
PROMPT_DUKCAPIL = """Kamu asisten administrasi kependudukan (Dukcapil) Kab. Batang.

ATURAN:
1. Jawab HANYA berdasarkan konteks. Jangan tambah info dari pengetahuan umum.
2. Kalau jawaban tidak ada di konteks, jawab: "Maaf, informasi tidak ditemukan di buku saku Dukcapil."
3. Bahasa Indonesia jelas & ringkas.

KONTEKS:
{context}

PERTANYAAN: {question}

JAWABAN:"""

PROMPT_OPD = """Kamu asisten direktori OPD (Organisasi Perangkat Daerah) Kab. Batang.

ATURAN:
1. Jawab HANYA berdasarkan konteks.
2. Kalau OPD tidak ada, jawab: "OPD tidak ditemukan dalam direktori."
3. Sebutkan sumber: [Sumber: <nama_opd>, nomor <nomor>]
4. Bahasa Indonesia jelas & ringkas.

KONTEKS:
{context}

PERTANYAAN: {question}

JAWABAN:"""

PROMPT_BOTH = """Kamu asisten layanan publik Kab. Batang. Pertanyaan user butuh info dari DUA sumber:
- Buku Saku Dukcapil (prosedur/syarat administrasi kependudukan)
- Direktori OPD Kab. Batang (alamat & kontak dinas)

ATURAN:
1. Jawab HANYA berdasarkan konteks. Bagi jawaban jadi dua bagian: (a) Prosedur/Syarat, (b) Kontak Dinas.
2. Untuk bagian OPD, sebutkan sumber: [Sumber: <nama_opd>, nomor <nomor>]
3. Kalau salah satu info tidak ada, sebutkan terus terang.
4. Bahasa Indonesia jelas & ringkas.

KONTEKS:
{context}

PERTANYAAN: {question}

JAWABAN:"""

PROMPT_NONE = """Pertanyaan user di luar cakupan chatbot layanan publik Kab. Batang (Dukcapil & direktori OPD).
Jawab dengan sopan bahwa pertanyaan ini di luar cakupan, dan tawarkan bantuan terkait kependudukan atau direktori OPD.
Bahasa Indonesia singkat.

PERTANYAAN: {question}

JAWABAN:"""

def format_context(docs: List[Document]) -> str:
    parts = []
    for i, d in enumerate(docs, 1):
        m = d.metadata or {}
        src = m.get("_source", "")
        if src == "opd" or "nama_opd" in m:
            parent = f" (bagian dari {m['parent_opd']})" if m.get("parent_opd") else ""
            header = f"[OPD {i}: {m.get('nama_opd','?')}{parent}, nomor {m.get('nomor','?')}, tipe {m.get('tipe','?')}]"
        elif src == "dukcapil":
            header = f"[Dukcapil {i}]"
        else:
            header = f"[Sumber {i}]"
        parts.append(f"{header}\n{d.page_content}")
    return "\n\n---\n\n".join(parts)

# ============ Agentic Graph (State + Nodes) ============
RouteType = Literal["dukcapil", "opd", "both", "none"]

class AgentState(TypedDict):
    question: str
    route: RouteType
    route_reason: str
    documents: List[Document]
    source_used: str
    answer: str
    timings: Dict[str, float]

def node_route(state: AgentState) -> AgentState:
    t0 = time.perf_counter()
    res = route_query(state["question"])
    dt = time.perf_counter() - t0
    timings = dict(state.get("timings") or {})
    timings["route"] = dt
    return {
        **state,
        "route": res["route"],
        "route_reason": res["reason"],
        "timings": timings,
    }

def node_retrieve_dukcapil(state: AgentState) -> AgentState:
    t0 = time.perf_counter()
    docs = retrieve_dukcapil(state["question"], k=5)
    dt = time.perf_counter() - t0
    timings = dict(state.get("timings") or {})
    timings["retrieve"] = dt
    return {**state, "documents": docs, "source_used": "dukcapil", "timings": timings}

def node_retrieve_opd(state: AgentState) -> AgentState:
    t0 = time.perf_counter()
    docs = retrieve_opd(state["question"], k=5)
    dt = time.perf_counter() - t0
    timings = dict(state.get("timings") or {})
    timings["retrieve"] = dt
    return {**state, "documents": docs, "source_used": "opd", "timings": timings}

def node_retrieve_both(state: AgentState) -> AgentState:
    t0 = time.perf_counter()
    docs_d = retrieve_dukcapil(state["question"], k=4)
    docs_o = retrieve_opd(state["question"], k=4)
    for d in docs_d:
        d.metadata = {**d.metadata, "_source": "dukcapil"}
    for d in docs_o:
        d.metadata = {**d.metadata, "_source": "opd"}
    docs = docs_d + docs_o
    dt = time.perf_counter() - t0
    timings = dict(state.get("timings") or {})
    timings["retrieve"] = dt
    return {**state, "documents": docs, "source_used": "both", "timings": timings}

def node_skip_retrieve(state: AgentState) -> AgentState:
    timings = dict(state.get("timings") or {})
    timings["retrieve"] = 0.0
    return {**state, "documents": [], "source_used": "none", "timings": timings}

def node_generate(state: AgentState) -> AgentState:
    t0 = time.perf_counter()
    q = state["question"]
    src = state["source_used"]
    docs = state.get("documents") or []
    ctx = format_context(docs) if docs else ""

    if src == "dukcapil":
        prompt = PROMPT_DUKCAPIL.format(context=ctx, question=q)
    elif src == "opd":
        prompt = PROMPT_OPD.format(context=ctx, question=q)
    elif src == "both":
        prompt = PROMPT_BOTH.format(context=ctx, question=q)
    else:
        prompt = PROMPT_NONE.format(question=q)

    answer = llm.invoke(prompt).content
    dt = time.perf_counter() - t0
    timings = dict(state.get("timings") or {})
    timings["generate"] = dt
    return {**state, "answer": answer, "timings": timings}

def branch_on_route(state: AgentState) -> str:
    return state["route"]

workflow = StateGraph(AgentState)
workflow.add_node("route", node_route)
workflow.add_node("retrieve_dukcapil", node_retrieve_dukcapil)
workflow.add_node("retrieve_opd", node_retrieve_opd)
workflow.add_node("retrieve_both", node_retrieve_both)
workflow.add_node("skip_retrieve", node_skip_retrieve)
workflow.add_node("generate", node_generate)

workflow.set_entry_point("route")
workflow.add_conditional_edges(
    "route",
    branch_on_route,
    {
        "dukcapil": "retrieve_dukcapil",
        "opd":      "retrieve_opd",
        "both":     "retrieve_both",
        "none":     "skip_retrieve",
    },
)
for node in ["retrieve_dukcapil", "retrieve_opd", "retrieve_both", "skip_retrieve"]:
    workflow.add_edge(node, "generate")
workflow.add_edge("generate", END)

agentic_app = workflow.compile()

def ask_agentic(question: str, verbose: bool = True) -> Dict[str, Any]:
    """Single entry point: run agentic pipeline, return state + total latency."""
    initial: AgentState = {
        "question": question,
        "route": "none",
        "route_reason": "",
        "documents": [],
        "source_used": "",
        "answer": "",
        "timings": {},
    }
    t0 = time.perf_counter()
    result = agentic_app.invoke(initial)
    total = time.perf_counter() - t0
    result["timings"]["total"] = total

    if verbose:
        t = result["timings"]
        print(f"Q: {question}")
        print(f"   Route: {result['route']}  ({result['route_reason']})")
        print(f"   Source used: {result['source_used']}  | docs: {len(result['documents'])}")
        print(f"   Timing — route: {t.get('route',0):.2f}s | retrieve: {t.get('retrieve',0):.2f}s | generate: {t.get('generate',0):.2f}s | TOTAL: {t['total']:.2f}s")
        print(f"   Answer: {result['answer'][:400]}{'...' if len(result['answer'])>400 else ''}\n")
    return result
