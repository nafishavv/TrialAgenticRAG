"""Capability ABC — unified interface for retrieval sources and (future) tools.

A Capability is something the agentic router can dispatch to. Concrete subclasses:
  - VectorSourceCapability  (Chroma + hybrid BM25/dense)
  - (future) SqlToolCapability, WebSearchCapability, etc.

invoke() returns List[Document] so existing prompt formatting and downstream
chains work unchanged. Structured tools (SQL) wrap their results in
Document(page_content=formatted_row, metadata=...) — the page_content is what
the LLM sees as context.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import List

from langchain_core.documents import Document


class Capability(ABC):
    """Anything the router/RAG pipeline can dispatch to."""

    name: str
    """Registry key + value used as `metadata._source` tag on returned Documents."""

    description: str
    """One-line description shown to the router LLM for classification."""

    router_examples: List[str]
    """Few-shot example queries that route to this capability."""

    searchable: bool = True
    """If True, naive-combined RAG includes this in the union-retrieve fan-out."""

    @abstractmethod
    def invoke(self, query: str, k: int = 5) -> List[Document]:
        """Run the capability, return top-k Documents tagged with `_source`."""
        ...

    def format_header(self, doc: Document, idx: int) -> str:
        """Render the bracketed header line that precedes a chunk in the prompt.

        Subclasses override for source-specific formatting (e.g. OPD shows
        `nama_opd` + `nomor`, Dukcapil shows section + page).
        """
        return f"[{self.name.title()} {idx}]"

    def _tag(self, docs: List[Document]) -> List[Document]:
        """Stamp `metadata._source = self.name` on each doc (idempotent)."""
        for d in docs:
            d.metadata = {**(d.metadata or {}), "_source": self.name}
        return docs


def format_context(docs: List[Document], capabilities: dict[str, Capability]) -> str:
    """Render a list of retrieved docs as numbered context blocks.

    Dispatches header formatting based on each doc's `_source` tag back to the
    owning capability. Unknown sources fall back to a generic header.
    """
    parts = []
    for i, d in enumerate(docs, 1):
        src = (d.metadata or {}).get("_source", "")
        cap = capabilities.get(src)
        header = cap.format_header(d, i) if cap else f"[Sumber {i}]"
        parts.append(f"{header}\n{d.page_content}")
    return "\n\n---\n\n".join(parts)
