"""UnifiedVectorStore — the single retrieval backend over the unified collection.

ONE Chroma collection spanning all domains (built by vectorstore/unified.py). All
three RAG tiers retrieve through this:
  - naive     : dense, global (no filter)
  - enhanced  : hybrid, global (no filter) -> rerank
  - agentic   : hybrid, domain-FILTERED (routing = metadata filter, tool per domain)

`domain=None` searches the whole corpus; `domain="sosial"` restricts via Chroma
`where={"domain": ...}`. For hybrid, BM25 is built per filter-key (global or a
single domain) and cached, so a domain-filtered hybrid search has domain-local
IDF — i.e. it matches what a physically separate collection would give.
"""

from __future__ import annotations

import time
from typing import Dict, List, Literal, Optional, Tuple

from langchain_chroma import Chroma
from langchain_classic.retrievers import EnsembleRetriever
from langchain_community.retrievers import BM25Retriever
from langchain_core.documents import Document

from ragtrial.config import UNIFIED_COLLECTION, UNIFIED_VECTOR_STORE
from ragtrial.llm import embeddings

Strategy = Literal["dense", "hybrid"]
_ALL_KEY = "__all__"


class UnifiedVectorStore:
    def __init__(
        self,
        collection_name: str = UNIFIED_COLLECTION,
        persist_directory=UNIFIED_VECTOR_STORE,
        fetch_k: int = 10,
        weights: Tuple[float, float] = (0.5, 0.5),
    ):
        self.collection_name = collection_name
        self.persist_directory = persist_directory
        self.fetch_k = fetch_k
        self.weights = weights
        self._vs: Optional[Chroma] = None
        self._all_docs: Optional[List[Document]] = None
        self._bm25: Dict[str, BM25Retriever] = {}

    def _ensure_vs(self) -> None:
        if self._vs is None:
            self._vs = Chroma(
                collection_name=self.collection_name,
                embedding_function=embeddings,
                persist_directory=str(self.persist_directory),
            )

    def _load_all_docs(self) -> None:
        if self._all_docs is None:
            self._ensure_vs()
            raw = self._vs.get()
            self._all_docs = [
                Document(page_content=d, metadata=m)
                for d, m in zip(raw["documents"], raw["metadatas"])
            ]

    @staticmethod
    def _where(domain: Optional[str]) -> Optional[dict]:
        return {"domain": domain} if domain else None

    def _bm25_for(self, domain: Optional[str]) -> BM25Retriever:
        """Cached BM25 over the whole corpus (domain=None) or one domain's docs."""
        key = domain or _ALL_KEY
        if key not in self._bm25:
            self._load_all_docs()
            docs = (
                self._all_docs
                if domain is None
                else [d for d in self._all_docs if (d.metadata or {}).get("domain") == domain]
            )
            r = BM25Retriever.from_documents(docs)
            r.k = self.fetch_k
            self._bm25[key] = r
        return self._bm25[key]

    # Sub-timing keys populated into the caller-provided `timings` dict by search().
    TIMING_KEYS = ("t_embed_query", "t_search", "t_bm25", "t_fuse")

    def search(
        self,
        query: str,
        k: int = 5,
        strategy: Strategy = "hybrid",
        domain: Optional[str] = None,
        timings: Optional[Dict[str, float]] = None,
    ) -> List[Document]:
        """Top-k docs. `domain=None` = global; else filtered to that domain.

        If `timings` (a dict) is passed, it is populated in place with per-stage
        sub-timings under the keys in `TIMING_KEYS`:
          - t_embed_query : query-embedding API call (rawan 429)
          - t_search      : pure Chroma dense vector search
          - t_bm25        : BM25 build/query (0.0 for dense-only)
          - t_fuse        : RRF fusion (0.0 for dense-only)
        Existing callers that don't pass `timings` are unaffected. Results are
        byte-identical to the previous EnsembleRetriever path (same RRF, same
        weights, same ordering) — only the internals are timed.
        """
        self._ensure_vs()
        where = self._where(domain)
        t = timings if timings is not None else {}
        # Init all keys so callers always see the full schema (0.0 where n/a).
        for key in self.TIMING_KEYS:
            t.setdefault(key, 0.0)

        # Embed the query ONCE (dense path would otherwise embed inside Chroma),
        # so t_embed_query is cleanly isolated from the vector search itself.
        t0 = time.perf_counter()
        qvec = embeddings.embed_query(query)
        t["t_embed_query"] = time.perf_counter() - t0

        t0 = time.perf_counter()
        dense_docs = self._vs.similarity_search_by_vector(
            qvec, k=(k if strategy == "dense" else self.fetch_k), filter=where
        )
        t["t_search"] = time.perf_counter() - t0

        if strategy == "dense":
            return dense_docs[:k]

        # hybrid: domain-local BM25 + domain-filtered dense, fused by RRF.
        t0 = time.perf_counter()
        bm25 = self._bm25_for(domain)          # first call pays the (cached) build
        bm25_docs = bm25.invoke(query)
        t["t_bm25"] = time.perf_counter() - t0

        t0 = time.perf_counter()
        dense = self._vs.as_retriever(
            search_kwargs={"k": self.fetch_k, **({"filter": where} if where else {})}
        )
        ensemble = EnsembleRetriever(retrievers=[bm25, dense], weights=list(self.weights))
        # Fuse the ALREADY-retrieved lists directly (same RRF the ensemble's own
        # .invoke would run) so BM25 + dense are not re-queried.
        fused = ensemble.weighted_reciprocal_rank([bm25_docs, dense_docs])
        t["t_fuse"] = time.perf_counter() - t0
        return fused[:k]

    def search_with_scores(
        self, query: str, k: int = 5, domain: Optional[str] = None
    ) -> List[Tuple[Document, float]]:
        """Dense top-k with distance scores (lower = closer)."""
        self._ensure_vs()
        pairs = self._vs.similarity_search_with_score(query, k=k, filter=self._where(domain))
        return [(d, float(s)) for d, s in pairs]

    def count(self) -> int:
        """Number of chunks in the unified collection (health/readiness probe)."""
        self._ensure_vs()
        return self._vs._collection.count()


# Module-level singleton — all tiers share one store (one BM25 cache, one client).
unified_store = UnifiedVectorStore()
