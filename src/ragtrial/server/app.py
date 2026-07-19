"""FastAPI app — chat API + static frontend host.

Run:  uv run python scripts/serve.py            (recommended on Windows)
      uv run uvicorn ragtrial.server.app:app

Design notes:
- /api/chat is a SYNC handler on purpose: FastAPI runs it in a threadpool, so
  the blocking Gemini/Chroma/reranker calls never stall the event loop.
- Warmup happens in a background daemon thread at startup (Chroma + BM25 +
  cross-encoder + intent router take 10-30 s cold); /api/health reports it.
- Same-origin by design (the static UI is served by this same app), so no CORS
  middleware. Add CORSMiddleware here if the frontend is ever hosted apart.
- Singletons (unified_store, reranker, LLM clients) are shared across the
  threadpool; fine at prototype scale — do not add locks preemptively.
"""

from __future__ import annotations

import logging
import threading
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles

from ragtrial.config import WEB_DIR
from ragtrial.modes import MODES
from ragtrial.server import meta as ui_meta
from ragtrial.server.schemas import (
    ChatRequest,
    ChatResponse,
    ErrorBody,
    HealthResponse,
    SourceItem,
)
from ragtrial.server.sessions import SessionStore

log = logging.getLogger(__name__)

_RATE_LIMIT_MARKERS = ("429", "RESOURCE_EXHAUSTED", "ResourceExhausted", "rate limit")

MSG_RATE_LIMITED = "Sistem sedang sibuk. Mohon coba lagi dalam beberapa saat."
MSG_INTERNAL = "Maaf, terjadi kendala teknis. Silakan coba lagi."


def _warmup(app: FastAPI) -> None:
    """Preload the heavy stack so the first user request isn't 30 s cold."""
    try:
        from ragtrial.pipeline.rerank import DEFAULT_RERANK_MODEL, _get_cross_encoder
        from ragtrial.vectorstore.store import unified_store

        unified_store.search("warmup", k=1, strategy="hybrid")   # Chroma + BM25
        _get_cross_encoder(DEFAULT_RERANK_MODEL)                 # cross-encoder
        app.state.ready = True
        log.info("warmup selesai — sistem siap")
    except Exception:
        log.exception("warmup gagal — /api/health akan melapor degraded")


@asynccontextmanager
async def lifespan(app: FastAPI):
    app.state.ready = False
    app.state.sessions = SessionStore()
    threading.Thread(target=_warmup, args=(app,), daemon=True, name="warmup").start()
    yield


app = FastAPI(title="Layanan Informasi Publik Kab. Batang", lifespan=lifespan)


def _error(status: int, code: str, message: str) -> JSONResponse:
    return JSONResponse(
        status_code=status,
        content={"error": ErrorBody(code=code, message=message).model_dump()},
    )


def _sources_of(turn) -> list[SourceItem]:
    """Deduped citation labels (first-appearance order) via Capability.citation_label."""
    from ragtrial.capabilities.registry import CAPABILITIES

    items: list[SourceItem] = []
    seen: set[str] = set()
    for doc in turn.documents:
        domain = (doc.metadata or {}).get("_source", "")
        cap = CAPABILITIES.get(domain)
        label = cap.citation_label(doc) if cap else (domain.title() or "Sumber")
        if label not in seen:
            seen.add(label)
            items.append(SourceItem(label=label, domain=domain))
    return items


@app.post("/api/chat", response_model=ChatResponse)
def chat(req: ChatRequest) -> ChatResponse | JSONResponse:
    session = app.state.sessions.get_or_create(req.session_id, req.mode)
    try:
        turn = session.ask(req.message)
    except Exception as e:  # error trace already written inside ChatSession
        text = str(e)
        if any(m in text for m in _RATE_LIMIT_MARKERS):
            return _error(429, "rate_limited", MSG_RATE_LIMITED)
        log.exception("chat gagal (session=%s)", session.session_id)
        return _error(500, "internal", MSG_INTERNAL)
    return ChatResponse(
        session_id=turn.session_id,
        query_id=turn.query_id,
        answer=turn.answer,
        sources=_sources_of(turn),
        latency_seconds=round(turn.timings["total"], 2),
        mode=turn.mode or session.mode,
    )


@app.get("/api/meta")
def get_meta() -> dict:
    return {
        "app_name": "Layanan Informasi Publik Kabupaten Batang",
        "domains": ui_meta.DOMAINS,
        "popular_questions": ui_meta.POPULAR_QUESTIONS,
        "modes": list(MODES),
    }


@app.get("/api/health", response_model=HealthResponse)
def health() -> HealthResponse:
    import os

    ready = bool(getattr(app.state, "ready", False))
    n_docs = None
    if ready:
        try:
            from ragtrial.vectorstore.store import unified_store
            n_docs = unified_store.count()
        except Exception:
            pass
    return HealthResponse(
        status="ok" if ready else "starting",
        ready=ready,
        n_documents=n_docs,
        api_key_present=bool(os.getenv("GOOGLE_API_KEY") or os.getenv("GEMINI_API_KEY")),
    )


# Static frontend — mounted LAST so /api/* wins. html=True serves index.html at /.
if WEB_DIR.is_dir():
    app.mount("/", StaticFiles(directory=str(WEB_DIR), html=True), name="web")
