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
from ragtrial.result import RagResult

MAX_ITERATIONS = 5
K_PER_TOOL = 5
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


_TOOLS = _build_tools(SEARCHABLE_CAPABILITIES)
_llm_with_tools = llm.bind_tools(_TOOLS)


def _system_prompt(capabilities) -> str:
    tool_lines = "\n".join(
        f"- {_TOOL_PREFIX}{name}: {cap.description}" for name, cap in capabilities.items()
    )
    return (
        "Kamu asisten layanan publik Kab. Batang. Kamu punya tools pencarian per domain:\n"
        f"{tool_lines}\n\n"
        "CARA KERJA:\n"
        "1. Putuskan sendiri domain mana yang relevan, lalu panggil tool-nya. Boleh lebih dari satu.\n"
        "2. Kalau hasil kurang relevan, kamu boleh memanggil tool lagi dengan query yang ditulis ulang.\n"
        "3. Kalau pertanyaan di luar cakupan semua domain, JANGAN panggil tool apa pun — langsung tolak dengan sopan.\n"
        "4. Jawab HANYA berdasarkan hasil tool. Jangan menambah info dari pengetahuan umum.\n"
        "5. Kalau info tidak ditemukan, katakan: \"Maaf, informasi tidak ditemukan dalam sumber yang tersedia.\"\n"
        "6. Bahasa Indonesia jelas & ringkas. Sebutkan sumbernya kalau relevan."
    )


# ============ State ============
class AgentState(TypedDict):
    messages: List[Any]
    documents: List[Document]
    steps: List[Dict[str, Any]]
    iterations: int
    t_agent: float
    t_tools: float


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

    for call in last.tool_calls:
        name = call["name"]
        query = (call.get("args") or {}).get("query", "")
        domain = name[len(_TOOL_PREFIX):] if name.startswith(_TOOL_PREFIX) else name
        cap = SEARCHABLE_CAPABILITIES.get(domain)

        if cap is None:
            content = f"Tool {name} tidak dikenal."
            steps.append({"tool": name, "query": query, "n_docs": 0, "error": "unknown_tool"})
        else:
            t0 = time.perf_counter()
            found = cap.invoke(query, k=K_PER_TOOL)
            t_tools += time.perf_counter() - t0
            docs.extend(found)
            content = format_context(found, CAPABILITIES) if found else "Tidak ada hasil."
            steps.append({"tool": name, "query": query, "n_docs": len(found)})

        new_msgs.append(ToolMessage(content=content, tool_call_id=call["id"]))

    return {**state, "messages": state["messages"] + new_msgs, "documents": docs,
            "steps": steps, "t_tools": t_tools}


def _should_continue(state: AgentState) -> str:
    last = state["messages"][-1]
    has_calls = isinstance(last, AIMessage) and bool(last.tool_calls)
    if has_calls and state["iterations"] < MAX_ITERATIONS:
        return "tools"
    return END


# ============ Graph ============
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
    init: AgentState = {
        "messages": [SystemMessage(_system_prompt(SEARCHABLE_CAPABILITIES)),
                     HumanMessage(question)],
        "documents": [],
        "steps": [],
        "iterations": 0,
        "t_agent": 0.0,
        "t_tools": 0.0,
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
            "generate": final["t_agent"],
            "total": total,
        },
        meta={"steps": final["steps"], "iterations": final["iterations"]},
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
