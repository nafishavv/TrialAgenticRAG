"""Capability registry — derived from the source registry.

CAPABILITIES is built from `ragtrial.sources.SOURCES`, so registering a domain
there automatically exposes its capability to every RAG pipeline. Future
non-source tools (SQL, web search) get appended here directly.
"""

from __future__ import annotations

from ragtrial.capabilities.base import Capability
from ragtrial.sources import SOURCES

CAPABILITIES: dict[str, Capability] = {
    name: src.capability for name, src in SOURCES.items()
}

# Future tools (not backed by a Source) would be added here, e.g.:
#   CAPABILITIES["sql"] = sql_tool_capability

SEARCHABLE_CAPABILITIES: dict[str, Capability] = {
    name: cap for name, cap in CAPABILITIES.items() if cap.searchable
}
