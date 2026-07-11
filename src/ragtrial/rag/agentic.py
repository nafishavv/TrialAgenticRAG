"""Agentic RAG — LLM orchestrator that decides its own steps (tool-calling loop).

Unlike enhanced (a fixed pipeline), here the LLM controls the flow: it chooses
which domain tool(s) to call, may call several, may re-search with a reformulated
query after seeing weak results, or skip retrieval entirely for out-of-scope
questions — then answers when it judges the context sufficient.

Tools (Cara 1, lean): one `search_<domain>` per searchable capability. The LLM
*is* the domain selector — picking a tool = selecting a domain. Every decision is
logged to `meta.steps` (tool + query arg + n_docs), so agent behavior (rewrite /
multi-domain / skip) is fully inspectable from the trace, without extra tools.

Graph:  agent ──tool_calls?──> tools ──> agent ... ──no tool_calls / max_iter──> END

Future tools (SQL, web search) register as Capabilities -> become agent tools
automatically. Multi-agent would add more agent nodes to this graph.
"""

from __future__ import annotations

import time
from typing import Any, Dict, List

from langchain_core.documents import Document
from langchain_core.messages import (
    AIMessage,
    HumanMessage,
    SystemMessage,
    ToolMessage,
)
from langchain_core.tools import StructuredTool
from langgraph.graph import END, StateGraph
from pydantic import BaseModel, Field
from typing_extensions import TypedDict

from ragtrial.capabilities import format_context
from ragtrial.capabilities.registry import CAPABILITIES, SEARCHABLE_CAPABILITIES
from ragtrial.llm import llm
from ragtrial.pipeline.rerank import rerank_documents
from ragtrial.rag.prompts import service_categories_block
from ragtrial.result import RagResult
from ragtrial.vectorstore.store import unified_store

MAX_ITERATIONS = 5
K_PER_TOOL = 10          # candidate pool per tool call (into rerank)
RERANK_TOP_N = 5         # docs kept per tool call after rerank (matches enhanced top_n)
RERANK_SUFFICIENCY = 0.30  # top rerank score below this -> inject [LOW RELEVANCE SIGNAL] so the agent may re-search
_TOOL_PREFIX = "search_"


# ============ Tools (schema only — executed manually to capture Documents) ============
class _SearchArgs(BaseModel):
    query: str = Field(description="Kueri pencarian (boleh ditulis ulang dari pertanyaan asli).")


def _noop(query: str) -> str:  # never actually invoked; we execute in the tools node
    return ""


def _build_tools(capabilities) -> List[StructuredTool]:
    return [
        StructuredTool.from_function(
            func=_noop,
            name=f"{_TOOL_PREFIX}{name}",
            description=f"Cari informasi di domain '{name}'. {cap.description}",
            args_schema=_SearchArgs,
        )
        for name, cap in capabilities.items()
    ]


_TOOLS: list | None = None
_llm_with_tools = None


def _system_prompt(capabilities) -> str:
    tool_lines = "\n".join(
        f"- {_TOOL_PREFIX}{name}: {cap.description}" for name, cap in capabilities.items()
    )
    return (
        "You are a public-service assistant for Kabupaten Batang. You can help with:\n"
        f"{service_categories_block()}\n\n"
        "You have per-domain search tools:\n"
        f"{tool_lines}\n\n"
        "USER INTENT — decide YOURSELF whether retrieval (a tool call) is needed:\n"
        "1. Public-service questions (permits, civil registration, taxes, law, agency contacts/addresses,\n"
        "   etc.) → ALWAYS call a relevant tool first (you may call more than one), EVEN if you doubt the\n"
        "   data exists — do not assume coverage yourself; let the tool result decide. For FACTUAL service\n"
        "   answers, answer ONLY from tool results — do not invent from general knowledge.\n"
        "2. Greetings / questions about your identity / what services you offer (e.g. \"halo\", \"siapa\n"
        "   kamu?\", \"bisa bantu apa?\") → do NOT call a tool. Answer directly and warmly: say you are a\n"
        "   public-service chatbot for Kab. Batang, list the service categories above, then offer help.\n"
        "3. Questions outside public services (e.g. product prices, national figures, recipes, general\n"
        "   topics) → do NOT call a tool. Decline politely: \"Maaf, saya hanya bisa membantu dengan\n"
        "   informasi layanan publik Kabupaten Batang.\", then point back to the service categories.\n"
        "4. When calling search_opd for a specific agency, ALWAYS use the full official name, not an\n"
        "   abbreviation. E.g. 'Dukcapil' → 'Dinas Kependudukan dan Pencatatan Sipil', 'Dinkes' → 'Dinas\n"
        "   Kesehatan', 'BPKAD' → 'Badan Pengelolaan Keuangan dan Aset Daerah', etc.\n\n"
        "SUFFICIENCY CHECK — after each tool call, BEFORE answering, judge the results:\n"
        "1. If the tool results ALREADY contain enough relevant information to answer → ANSWER NOW. Do NOT\n"
        "   search again. (Re-searching when the evidence is already there wastes time — prefer a fast\n"
        "   answer.)\n"
        "2. If the results are irrelevant / do not answer the question / carry a [LOW RELEVANCE SIGNAL]\n"
        "   → SEARCH AGAIN ONCE, diagnosing WHY it failed:\n"
        "     - wrong domain?         → call a different, more appropriate domain tool\n"
        "     - query too narrow/broad/informal? → rewrite the query (more specific / official terms)\n"
        "     - unsure of the domain?  → search across several domains\n"
        "   Limit: AT MOST 1-2 retries.\n"
        "3. If after retrying the results are STILL inadequate → do NOT keep forcing it. Answer with what\n"
        "   you have, or say honestly: \"Maaf, informasi tidak ditemukan dalam sumber yang tersedia.\"\n"
        "   Principle: re-search ONLY to fix a clearly failed retrieval — never 'just in case'.\n\n"
        "Respond in polite, clear, concise Bahasa Indonesia. Cite the source when relevant."
    )


# ============ State ============
class AgentState(TypedDict):
    messages: List[Any]
    documents: List[Document]
    steps: List[Dict[str, Any]]
    iterations: int
    t_agent: float
    t_tools: float
    t_rerank: float


# ============ Nodes ============
def _node_agent(state: AgentState) -> AgentState:
    t0 = time.perf_counter()
    ai = _llm_with_tools.invoke(state["messages"])
    dt = time.perf_counter() - t0
    return {
        **state,
        "messages": state["messages"] + [ai],
        "iterations": state["iterations"] + 1,
        "t_agent": state["t_agent"] + dt,
    }


def _node_tools(state: AgentState) -> AgentState:
    last: AIMessage = state["messages"][-1]
    new_msgs: List[Any] = []
    docs: List[Document] = list(state["documents"])
    steps: List[Dict[str, Any]] = list(state["steps"])
    t_tools = state["t_tools"]
    t_rerank = state["t_rerank"]

    for call in last.tool_calls:
        name = call["name"]
        query = (call.get("args") or {}).get("query", "")
        domain = name[len(_TOOL_PREFIX):] if name.startswith(_TOOL_PREFIX) else name

        if domain not in SEARCHABLE_CAPABILITIES:
            content = f"Unknown tool {name}."
            steps.append({"tool": name, "query": query, "n_docs": 0, "error": "unknown_tool"})
        else:
            # Routing = domain FILTER on the unified index (hybrid), then rerank.
            t0 = time.perf_counter()
            found = unified_store.search(query, k=K_PER_TOOL, strategy="hybrid", domain=domain)
            t_tools += time.perf_counter() - t0
            t0 = time.perf_counter()
            found, scores = rerank_documents(query, found, top_n=RERANK_TOP_N)
            t_rerank += time.perf_counter() - t0
            top = scores[0] if scores else 0.0
            weak = top < RERANK_SUFFICIENCY
            docs.extend(found)
            content = format_context(found, CAPABILITIES) if found else "No results found."
            if weak:
                # Self-correction signal: low relevance -> the agent may re-search
                # (different domain / reformulated query / no filter). See system prompt.
                content += (
                    f"\n\n[LOW RELEVANCE SIGNAL: top score {top:.2f} < {RERANK_SUFFICIENCY}. "
                    "These results may be inadequate. If this is not the answer being sought, "
                    "SEARCH AGAIN: try a different domain, reformulated keywords, or drop the domain filter.]"
                )
            steps.append({"tool": name, "query": query, "n_docs": len(found),
                          "top_score": round(top, 3), "weak": weak})

        new_msgs.append(ToolMessage(content=content, tool_call_id=call["id"]))

    return {**state, "messages": state["messages"] + new_msgs, "documents": docs,
            "steps": steps, "t_tools": t_tools, "t_rerank": t_rerank}


def _should_continue(state: AgentState) -> str:
    last = state["messages"][-1]
    has_calls = isinstance(last, AIMessage) and bool(last.tool_calls)
    if has_calls and state["iterations"] < MAX_ITERATIONS:
        return "tools"
    return END


# ============ Graph (lazy) ============
agentic_app = None


def _ensure_app() -> None:
    global _TOOLS, _llm_with_tools, agentic_app
    if agentic_app is not None:
        return
    _TOOLS = _build_tools(SEARCHABLE_CAPABILITIES)
    _llm_with_tools = llm.bind_tools(_TOOLS)
    _workflow = StateGraph(AgentState)
    _workflow.add_node("agent", _node_agent)
    _workflow.add_node("tools", _node_tools)
    _workflow.set_entry_point("agent")
    _workflow.add_conditional_edges("agent", _should_continue, {"tools": "tools", END: END})
    _workflow.add_edge("tools", "agent")
    agentic_app = _workflow.compile()


def _text(msg: AIMessage) -> str:
    """Flatten an AIMessage content to plain text.

    With bind_tools, Gemini returns content as a list of blocks (text + a thinking
    'signature'), not a string. Concatenate the text parts.
    """
    c = msg.content
    if isinstance(c, str):
        return c
    if isinstance(c, list):
        parts = []
        for b in c:
            if isinstance(b, str):
                parts.append(b)
            elif isinstance(b, dict) and b.get("text"):
                parts.append(b["text"])
        return "".join(parts)
    return str(c)


def _dedup(docs: List[Document]) -> List[Document]:
    seen, out = set(), []
    for d in docs:
        key = ((d.metadata or {}).get("_source", ""), d.page_content[:120])
        if key not in seen:
            seen.add(key)
            out.append(d)
    return out


def _source_used(steps: List[Dict[str, Any]]) -> str:
    domains = []
    for s in steps:
        dom = s["tool"][len(_TOOL_PREFIX):] if s["tool"].startswith(_TOOL_PREFIX) else s["tool"]
        if s.get("n_docs", 0) > 0 and dom not in domains:
            domains.append(dom)
    if not domains:
        return "none"
    return domains[0] if len(domains) == 1 else "both"


def ask_agentic(question: str, verbose: bool = True) -> RagResult:
    _ensure_app()
    init: AgentState = {
        "messages": [SystemMessage(_system_prompt(SEARCHABLE_CAPABILITIES)),
                     HumanMessage(question)],
        "documents": [],
        "steps": [],
        "iterations": 0,
        "t_agent": 0.0,
        "t_tools": 0.0,
        "t_rerank": 0.0,
    }
    t0 = time.perf_counter()
    final = agentic_app.invoke(init)
    total = time.perf_counter() - t0

    answer = ""
    for msg in reversed(final["messages"]):
        if isinstance(msg, AIMessage):
            txt = _text(msg)
            if txt.strip():
                answer = txt
                break
    if not answer:
        answer = "Maaf, tidak bisa menyelesaikan jawaban."

    docs = _dedup(final["documents"])
    source_used = _source_used(final["steps"])

    # Normalized routing for the execution log: domain(s) the agent actually used.
    routed: List[str] = []
    for s in final["steps"]:
        dom = s["tool"][len(_TOOL_PREFIX):] if s["tool"].startswith(_TOOL_PREFIX) else s["tool"]
        if s.get("n_docs", 0) > 0 and dom not in routed:
            routed.append(dom)
    routing = "none" if not routed else (routed[0] if len(routed) == 1 else routed)
    rewrote = any((s.get("query") or "").strip() != question.strip() for s in final["steps"])

    result = RagResult(
        question=question,
        answer=answer,
        documents=docs,
        query=question,
        route=source_used,
        source_used=source_used,
        mode="agentic",
        timings={
            "route": 0.0,
            "retrieve": final["t_tools"],
            "rerank": final["t_rerank"],
            "generate": final["t_agent"],
            "total": total,
        },
        meta={
            "steps": final["steps"],
            "iterations": final["iterations"],
            # Agent's implicit intent decision: called a tool = needed retrieval.
            "intent": "valid" if final["steps"] else "invalid",
        },
        decisions={
            "intent": "retrieve" if final["steps"] else "direct",
            "rewrite": rewrote,
            "routing": routing,                       # domain | [domains] | "none"
            "retrieval": "hybrid" if final["steps"] else "none",
            "rerank": bool(final["steps"]),
            "iterations": final["iterations"],
        },
    )

    if verbose:
        print(f"Q: {question}")
        for i, s in enumerate(final["steps"], 1):
            print(f"   step {i}: {s['tool']}(query={s['query']!r}) -> {s.get('n_docs', 0)} docs")
        if not final["steps"]:
            print("   (no tool calls — skipped retrieval)")
        t = result.timings
        print(
            f"   source_used: {source_used} | iters: {final['iterations']} | docs: {len(docs)}"
        )
        print(
            f"   Timing — retrieve: {t['retrieve']:.2f}s | "
            f"generate: {t['generate']:.2f}s | TOTAL: {t['total']:.2f}s"
        )
        print(f"   Answer: {answer[:400]}{'...' if len(answer) > 400 else ''}\n")
    return result
