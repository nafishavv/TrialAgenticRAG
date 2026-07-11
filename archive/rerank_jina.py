"""ARCHIVED — Jina Reranker API backend (NOT wired into the live pipeline).

Kept for reference only. Jina was trialed once as a fast GPU-hosted reranker
alternative; its result fed the reranker-choice comparison
(implementation-revision-plan.md §9.3): ~0.7s/call (~10x faster than local
bge-v2-m3) but PAID (free-trial balance runs out after a handful of calls) and
network-dependent. We chose the local bge-reranker-base default instead
(free, reproducible, recall-equal in the hybrid stack), so this backend was
removed from `src/ragtrial/pipeline/rerank.py`.

To resurrect: copy `_rerank_jina` back into rerank.py, restore a `backend`
param on `rerank_documents`, set RERANK_BACKEND=jina + a valid JINA_API_KEY.
"""

from __future__ import annotations

import os

JINA_RERANK_URL = "https://api.jina.ai/v1/rerank"
DEFAULT_JINA_MODEL = "jina-reranker-v2-base-multilingual"


def rerank_jina(query: str, docs, top_n, model: str = None):
    """Rerank via the Jina Reranker API (remote GPU). Needs JINA_API_KEY.

    Returns (reranked_docs, scores) aligned — same contract as the local
    cross-encoder `rerank_documents`, so it was a drop-in backend swap.
    """
    import requests

    key = os.getenv("JINA_API_KEY")
    if not key:
        raise RuntimeError("RERANK_BACKEND=jina but JINA_API_KEY is not set (.env).")
    model = model or os.getenv("JINA_RERANK_MODEL", DEFAULT_JINA_MODEL)
    resp = requests.post(
        JINA_RERANK_URL,
        headers={"Authorization": f"Bearer {key}", "Content-Type": "application/json"},
        json={
            "model": model,
            "query": query,
            "documents": [d.page_content for d in docs],
            "top_n": top_n or len(docs),
        },
        timeout=60,
    )
    resp.raise_for_status()
    results = resp.json()["results"]
    return [docs[r["index"]] for r in results], [float(r["relevance_score"]) for r in results]
