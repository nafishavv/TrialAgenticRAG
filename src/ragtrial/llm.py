"""Shared LLM + Embeddings singletons.

Import these instead of constructing new instances — keeps token usage predictable
and avoids re-initializing the Gemini client per module.
"""

from __future__ import annotations

import time

from langchain_core.embeddings import Embeddings
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings

from ragtrial.config import load_env

load_env()

LLM_MODEL: str = "gemini-2.5-flash"
EMBEDDING_MODEL: str = "models/gemini-embedding-2"
EMBEDDING_DIM: int = 768

# Transient Gemini/network failures surface as these substrings. They are NOT
# bugs — a short backoff usually clears them. Over a full eval (198 Q x 3 tiers x
# several calls) such blips are near-certain, so retrying keeps a momentary 503
# or a dropped connection from turning into a permanently-errored question.
_RETRYABLE = (
    "429", "RESOURCE_EXHAUSTED", "503", "UNAVAILABLE",
    "RemoteProtocolError", "Server disconnected", "ConnectionError",
    "Connection reset", "Connection aborted", "timed out",
)


def _with_retry(fn, *args, max_retries: int = 5, initial_wait: int = 30, **kwargs):
    """Call `fn`, retrying with growing backoff only on transient API errors."""
    wait = initial_wait
    for attempt in range(max_retries + 1):
        try:
            return fn(*args, **kwargs)
        except Exception as e:  # noqa: BLE001 — inspect message for retryable codes
            msg = str(e)
            if any(code in msg for code in _RETRYABLE) and attempt < max_retries:
                print(f"  [retry] transient API error; attempt {attempt + 1}/{max_retries}, "
                      f"wait {wait}s ({msg[:70]}...)", flush=True)
                time.sleep(wait)
                wait = int(wait * 1.5)
            else:
                raise
    raise RuntimeError("unreachable")


def invoke_with_retry(runnable, arg):
    """`runnable.invoke(arg)` with transient-error retry. Returns the raw result
    (e.g. an AIMessage), so callers keep using `.content` as before."""
    return _with_retry(runnable.invoke, arg)


class _RetryingEmbeddings(Embeddings):
    """Transparent proxy that adds transient-error retry to an Embeddings backend.

    All tiers share one embeddings singleton (imported here), and Chroma calls its
    `embed_query`/`embed_documents` internally — so wrapping the singleton once makes
    every embedding call (retrieval, intent gate, router) resilient without touching
    call sites."""

    def __init__(self, inner: Embeddings) -> None:
        self._inner = inner

    def embed_documents(self, texts):
        return _with_retry(self._inner.embed_documents, texts)

    def embed_query(self, text):
        return _with_retry(self._inner.embed_query, text)


llm = ChatGoogleGenerativeAI(
    model=LLM_MODEL,
    temperature=0.0,
    max_tokens=4096,
)

embeddings: Embeddings = _RetryingEmbeddings(
    GoogleGenerativeAIEmbeddings(
        model=EMBEDDING_MODEL,
        task_type="retrieval_query",
        output_dimensionality=EMBEDDING_DIM,
    )
)


def make_judge_llm(temperature: float = 0.0, max_tokens: int = 256) -> ChatGoogleGenerativeAI:
    """Construct a separate LLM client tuned for eval (deterministic, short)."""
    return ChatGoogleGenerativeAI(
        model=LLM_MODEL,
        temperature=temperature,
        max_tokens=max_tokens,
    )
