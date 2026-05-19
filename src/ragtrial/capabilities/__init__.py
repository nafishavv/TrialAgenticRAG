"""Capability registry — generic abstraction for retrieval AND tools.

Add a new data source / tool by:
  1. Implementing a Capability subclass (see base.py).
  2. Constructing an instance in instances/<name>.py.
  3. Registering it in CAPABILITIES below.

Naive-combined RAG and agentic router both iterate this registry — no callsite
changes needed when a new capability is added.
"""

from __future__ import annotations

from ragtrial.capabilities.base import Capability, format_context
from ragtrial.capabilities.instances.dukcapil import dukcapil_capability
from ragtrial.capabilities.instances.opd import opd_capability

CAPABILITIES: dict[str, Capability] = {
    dukcapil_capability.name: dukcapil_capability,
    opd_capability.name: opd_capability,
}

SEARCHABLE_CAPABILITIES: dict[str, Capability] = {
    name: cap for name, cap in CAPABILITIES.items() if cap.searchable
}

__all__ = ["Capability", "CAPABILITIES", "SEARCHABLE_CAPABILITIES", "format_context"]
