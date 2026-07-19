"""Trace record builders — ChatTurnResult -> (basic, full) dicts.

Documents are serialized to plain dicts (never pickled objects). The FINAL doc
list on RagResult is post-rerank, so `retrieved_chunks` in the full tier IS the
reranked set; per-doc scores come from metadata['_rerank_score'] (stamped by
rerank_documents) and pre-rerank candidate pools are not retained by design.
"""

from __future__ import annotations

from datetime import datetime
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Tuple

from langchain_core.documents import Document

from ragtrial.result import sources_in

if TYPE_CHECKING:
    from ragtrial.chat.session import ChatTurnResult

SCHEMA_VERSION = 1


def _serialize_doc(doc: Document, rank: int) -> Dict[str, Any]:
    m = dict(doc.metadata or {})
    domain = m.pop("_source", "")
    score = m.pop("_rerank_score", None)
    return {
        "rank": rank,
        "domain": domain,
        "score": score,
        "metadata": m,
        "text": doc.page_content,
    }


def _cited_sources(docs: List[Document]) -> List[Dict[str, str]]:
    """Deduped [{label, domain}] in first-appearance order, via Capability.citation_label."""
    try:
        from ragtrial.capabilities.registry import CAPABILITIES
    except Exception:
        CAPABILITIES = {}
    out: List[Dict[str, str]] = []
    seen: set = set()
    for d in docs:
        domain = (d.metadata or {}).get("_source", "")
        cap = CAPABILITIES.get(domain)
        label = cap.citation_label(d) if cap else (domain.title() or "Sumber")
        if label not in seen:
            seen.add(label)
            out.append({"label": label, "domain": domain})
    return out


def build_trace_records(
    turn: "ChatTurnResult", *, client: str, error: Optional[str] = None
) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    """Build the (basic, full) record pair for one turn. Pure function, no I/O."""
    res = turn.result
    ts = datetime.now().astimezone().isoformat()
    decisions = res.decisions or {}

    basic: Dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "ts": ts,
        "session_id": turn.session_id,
        "query_id": turn.query_id,
        "client": client,
        "mode": res.mode or "",
        "latency_seconds": round(turn.timings.get("total", 0.0), 4),
        "timings": {
            **{k: round(v, 4) for k, v in res.timings.items() if isinstance(v, (int, float))},
            "rewrite_followup": round(turn.timings.get("rewrite_followup", 0.0), 4),
            "pipeline_total": round(turn.timings.get("pipeline_total", 0.0), 4),
            "total": round(turn.timings.get("total", 0.0), 4),
        },
        "original_query": turn.original_query,
        "effective_query": turn.effective_query,
        "retrieval_query": res.query,
        "rewrite_followup_applied": turn.rewrite_applied,
        "intent": decisions.get("intent", ""),
        "route": res.route,
        "decisions": decisions,
        "n_documents": len(res.documents),
        "reranked": bool(decisions.get("rerank", False)),
        "source_used": res.source_used,
        "sources": sources_in(res.documents),
        "cited_sources": _cited_sources(res.documents),
        "answer": res.answer,
        "error": error,
    }

    full: Dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "ts": ts,
        "session_id": turn.session_id,
        "query_id": turn.query_id,
        "mode": res.mode or "",
        "retrieved_chunks": [_serialize_doc(d, i) for i, d in enumerate(res.documents, 1)],
        "meta": res.meta,
        "error": error,
    }
    return basic, full
