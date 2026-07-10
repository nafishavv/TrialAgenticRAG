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

    def search(
        self,
        query: str,
        k: int = 5,
        strategy: Strategy = "hybrid",
        domain: Optional[str] = None,
    ) -> List[Document]:
        """Top-k docs. `domain=None` = global; else filtered to that domain."""
        self._ensure_vs()
        where = self._where(domain)
        if strategy == "dense":
            return self._vs.similarity_search(query, k=k, filter=where)
        # hybrid: domain-local BM25 + domain-filtered dense, fused by RRF.
        bm25 = self._bm25_for(domain)
        dense = self._vs.as_retriever(
            search_kwargs={"k": self.fetch_k, **({"filter": where} if where else {})}
        )
        ensemble = EnsembleRetriever(retrievers=[bm25, dense], weights=list(self.weights))
        return ensemble.invoke(query)[:k]

    def search_with_scores(
        self, query: str, k: int = 5, domain: Optional[str] = None
    ) -> List[Tuple[Document, float]]:
        """Dense top-k with distance scores (lower = closer)."""
        self._ensure_vs()
        pairs = self._vs.similarity_search_with_score(query, k=k, filter=self._where(domain))
        return [(d, float(s)) for d, s in pairs]


# Module-level singleton — all tiers share one store (one BM25 cache, one client).
unified_store = UnifiedVectorStore()
