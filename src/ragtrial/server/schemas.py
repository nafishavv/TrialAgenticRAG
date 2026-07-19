"""Pydantic request/response models for the chat API.

ChatResponse is intentionally minimal: answer, deduped source labels, total
latency, ids. No chunks, no routing, no rewritten queries — that is trace-only
data (see ragtrial.tracing).
"""

from __future__ import annotations

from typing import List, Literal, Optional

from pydantic import BaseModel, Field

from ragtrial.modes import DEFAULT_MODE


class ChatRequest(BaseModel):
    message: str = Field(min_length=1, max_length=2000)
    session_id: Optional[str] = None
    """None on the first message; afterwards echo back what the server returned."""
    mode: Literal["naive", "enhanced", "agentic"] = DEFAULT_MODE


class SourceItem(BaseModel):
    label: str
    """Human-readable citation label, e.g. 'Buku Saku Adminduk Kab. Batang 2023, hal. 40'."""
    domain: str


class ChatResponse(BaseModel):
    session_id: str
    query_id: str
    answer: str
    sources: List[SourceItem]
    latency_seconds: float
    mode: str


class ErrorBody(BaseModel):
    code: Literal["rate_limited", "internal", "invalid_request"]
    message: str
    """Polite Indonesian text — the UI shows it verbatim."""


class HealthResponse(BaseModel):
    status: Literal["ok", "starting", "degraded"]
    ready: bool
    n_documents: Optional[int] = None
    api_key_present: bool
