"""VectorSourceCapability — Chroma + hybrid BM25/dense retrieval.

Default strategy is `hybrid` (EnsembleRetriever, BM25 + dense, RRF). Set
`strategy="dense"` for plain similarity search (recommended for stores that are
too small for BM25 to be meaningful, or where reranker is preferred).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, List, Literal, Optional

from langchain_chroma import Chroma
from langchain_classic.retrievers import EnsembleRetriever
from langchain_community.retrievers import BM25Retriever
from langchain_core.documents import Document

from ragtrial.capabilities.base import Capability
from ragtrial.llm import embeddings

Strategy = Literal["dense", "hybrid"]


@dataclass
class VectorSourceCapability(Capability):
    """Capability backed by a Chroma collection (+ optional BM25 hybrid)."""

    name: str
    description: str
    collection_name: str
    persist_directory: Path
    router_examples: List[str] = field(default_factory=list)
    strategy: Strategy = "hybrid"
    fetch_k: int = 10
    weights: tuple[float, float] = (0.5, 0.5)
    header_formatter: Optional[Callable[[Document, int], str]] = None
    gold_id_fn: Optional[Callable[[Document], Optional[str]]] = None
    citation: Optional[str] = None
    label_formatter: Optional[Callable[[Document], str]] = None
    searchable: bool = True

    # Lazy-initialized
    _vectorstore: Optional[Chroma] = field(default=None, init=False, repr=False)
    _retriever: Optional[object] = field(default=None, init=False, repr=False)
    _all_docs: Optional[List[Document]] = field(default=None, init=False, repr=False)

    def _ensure_vs(self) -> None:
        if self._vectorstore is None:
            self._vectorstore = Chroma(
                collection_name=self.collection_name,
                embedding_function=embeddings,
                persist_directory=str(self.persist_directory),
            )

    def _ensure_hybrid(self) -> None:
        """Lazily build the BM25+dense ensemble (materializes all docs for BM25)."""
        self._ensure_vs()
        if self._retriever is not None:
            return
        raw = self._vectorstore.get()
        self._all_docs = [
            Document(page_content=doc, metadata=meta)
            for doc, meta in zip(raw["documents"], raw["metadatas"])
        ]
        bm25 = BM25Retriever.from_documents(self._all_docs)
        bm25.k = self.fetch_k
        dense = self._vectorstore.as_retriever(search_kwargs={"k": self.fetch_k})
        self._retriever = EnsembleRetriever(
            retrievers=[bm25, dense],
            weights=list(self.weights),
        )

    def invoke(
        self, query: str, k: int = 5, strategy: Optional[Strategy] = None, **kwargs
    ) -> List[Document]:
        """Retrieve top-k. `strategy` overrides the configured default per call
        (enhanced RAG uses `dense`; the capability default may be `hybrid`)."""
        effective = strategy or self.strategy
        self._ensure_vs()
        if effective == "dense":
            docs = self._vectorstore.similarity_search(query, k=k)
        else:
            self._ensure_hybrid()
            docs = self._retriever.invoke(query)[:k]
        return self._tag(docs)

    def search_with_scores(self, query: str, k: int = 5) -> List[tuple[Document, float]]:
        """Dense top-k with distance scores (lower = closer). Used by naive RAG to
        merge a global top-k across collections without a pre-copied unified store."""
        self._ensure_vs()
        pairs = self._vectorstore.similarity_search_with_score(query, k=k)
        return [(self._tag([d])[0], float(s)) for d, s in pairs]

    def format_header(self, doc: Document, idx: int) -> str:
        if self.header_formatter is not None:
            return self.header_formatter(doc, idx)
        return super().format_header(doc, idx)

    def gold_id(self, doc: Document) -> Optional[str]:
        if self.gold_id_fn is not None:
            return self.gold_id_fn(doc)
        return super().gold_id(doc)

    def citation_hint(self) -> Optional[str]:
        return self.citation

    def citation_label(self, doc: Document) -> str:
        if self.label_formatter is not None:
            return self.label_formatter(doc)
        return super().citation_label(doc)
