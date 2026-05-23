"""Capability layer — generic abstraction for retrieval AND tools.

Add a new data source / tool by:
  1. (data source) create `sources/<domain>/` and register it in
     `ragtrial.sources` — its capability is picked up automatically.
  2. (tool) implement a Capability subclass and add it in
     `ragtrial.capabilities.registry`.

This package's __init__ exposes ONLY the ABCs + format_context. The populated
registry lives in `ragtrial.capabilities.registry` (import CAPABILITIES from
there) — keeping it out of __init__ avoids a cycle, since `sources/*` import
the Capability classes back from this package.
"""

from __future__ import annotations

from ragtrial.capabilities.base import Capability, format_context
from ragtrial.capabilities.vector_source import VectorSourceCapability

__all__ = [
    "Capability",
    "VectorSourceCapability",
    "format_context",
]
