"""Agentic RAG — router classifies query into a capability.name (or 'both'/'none'),
then dispatches to a generic retrieve node and finally generate.

Capabilities are read from the registry at module load. Adding a new capability
extends the router categories and dispatch automatically — no graph rewiring.
"""

from __future__ import annotations

import json
import time
from typing import Any, Dict, List, Literal

from langchain_core.documents import Document
from langgraph.graph import END, StateGraph
from typing_extensions import TypedDict

from ragtrial.capabilities import format_context
from ragtrial.capabilities.registry import CAPABILITIES, SEARCHABLE_CAPABILITIES
from ragtrial.llm import llm
from ragtrial.rag.prompts import (
    PROMPT_COMBINED,
    PROMPT_NONE,
    PROMPT_SINGLE,
    build_citation_rules,
    build_router_prompt,
    build_sources_brief,
)

# ============ Route values: every capability name + "both" + "none" ============
_ROUTE_VALUES = set(SEARCHABLE_CAPABILITIES.keys()) | {"both", "none"}


def route_query(question: str) -> Dict[str, str]:
    """Classify question into one of: <capability.name> | 'both' | 'none'."""
    prompt = build_router_prompt(question, SEARCHABLE_CAPABILITIES)
    raw = llm.invoke(prompt).content.strip()
    if raw.startswith("```"):
        raw = raw.strip("`")
        if raw.startswith("json"):
            raw = raw[4:]
        raw = raw.strip()
    try:
        parsed = json.loads(raw)
        route = parsed.get("route", "none")
        if route not in _ROUTE_VALUES:
            route = "none"
        return {"route": route, "reason": parsed.get("reason", "")}
    except Exception as e:
        return {"route": "none", "reason": f"parse_error: {e}; raw={raw[:80]}"}


# ============ State ============
class AgentState(TypedDict):
    question: str
    route: str
    route_reason: str
    documents: List[Document]
    source_used: str
    answer: str
    timings: Dict[str, float]


# ============ Nodes ============
def node_route(state: AgentState) -> AgentState:
    t0 = time.perf_counter()
    res = route_query(state["question"])
    dt = time.perf_counter() - t0
    timings = {**(state.get("timings") or {}), "route": dt}
    return {**state, "route": res["route"], "route_reason": res["reason"], "timings": timings}


def node_retrieve_single(state: AgentState) -> AgentState:
    t0 = time.perf_counter()
    cap = SEARCHABLE_CAPABILITIES[state["route"]]
    docs = cap.invoke(state["question"], k=5)
    dt = time.perf_counter() - t0
    timings = {**(state.get("timings") or {}), "retrieve": dt}
    return {**state, "documents": docs, "source_used": cap.name, "timings": timings}


def node_retrieve_all(state: AgentState) -> AgentState:
    """For 'both': fan out across every searchable capability."""
    t0 = time.perf_counter()
    docs: List[Document] = []
    for cap in SEARCHABLE_CAPABILITIES.values():
        docs.extend(cap.invoke(state["question"], k=4))
    dt = time.perf_counter() - t0
    timings = {**(state.get("timings") or {}), "retrieve": dt}
    return {**state, "documents": docs, "source_used": "both", "timings": timings}


def node_skip_retrieve(state: AgentState) -> AgentState:
    timings = {**(state.get("timings") or {}), "retrieve": 0.0}
    return {**state, "documents": [], "source_used": "none", "timings": timings}


def node_generate(state: AgentState) -> AgentState:
    t0 = time.perf_counter()
    q = state["question"]
    src = state["source_used"]
    docs = state.get("documents") or []
    ctx = format_context(docs, CAPABILITIES) if docs else ""

    if src == "both":
        prompt = PROMPT_COMBINED.format(
            sources_brief=build_sources_brief(SEARCHABLE_CAPABILITIES),
            citation_rules=build_citation_rules(SEARCHABLE_CAPABILITIES),
            context=ctx,
            question=q,
        )
    elif src in SEARCHABLE_CAPABILITIES:
        prompt = PROMPT_SINGLE.format(
            source_description=SEARCHABLE_CAPABILITIES[src].description,
            context=ctx,
            question=q,
        )
    else:  # none
        prompt = PROMPT_NONE.format(question=q)

    answer = llm.invoke(prompt).content
    dt = time.perf_counter() - t0
    timings = {**(state.get("timings") or {}), "generate": dt}
    return {**state, "answer": answer, "timings": timings}


def branch_on_route(state: AgentState) -> str:
    r = state["route"]
    if r == "both":
        return "retrieve_all"
    if r == "none":
        return "skip_retrieve"
    return "retrieve_single"


# ============ Graph ============
_workflow = StateGraph(AgentState)
_workflow.add_node("route", node_route)
_workflow.add_node("retrieve_single", node_retrieve_single)
_workflow.add_node("retrieve_all", node_retrieve_all)
_workflow.add_node("skip_retrieve", node_skip_retrieve)
_workflow.add_node("generate", node_generate)

_workflow.set_entry_point("route")
_workflow.add_conditional_edges(
    "route",
    branch_on_route,
    {
        "retrieve_single": "retrieve_single",
        "retrieve_all": "retrieve_all",
        "skip_retrieve": "skip_retrieve",
    },
)
for node in ["retrieve_single", "retrieve_all", "skip_retrieve"]:
    _workflow.add_edge(node, "generate")
_workflow.add_edge("generate", END)

agentic_app = _workflow.compile()


def ask_agentic(question: str, verbose: bool = True) -> Dict[str, Any]:
    """Run agentic pipeline, return result dict with timings."""
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
    result["timings"]["total"] = time.perf_counter() - t0

    if verbose:
        t = result["timings"]
        print(f"Q: {question}")
        print(f"   Route: {result['route']}  ({result['route_reason']})")
        print(
            f"   Source used: {result['source_used']}  | docs: {len(result['documents'])}"
        )
        print(
            f"   Timing — route: {t.get('route', 0):.2f}s | "
            f"retrieve: {t.get('retrieve', 0):.2f}s | "
            f"generate: {t.get('generate', 0):.2f}s | "
            f"TOTAL: {t['total']:.2f}s"
        )
        print(
            f"   Answer: {result['answer'][:400]}"
            f"{'...' if len(result['answer']) > 400 else ''}\n"
        )
    return result
