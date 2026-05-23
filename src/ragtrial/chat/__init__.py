"""Conversation layer — multi-turn chat over any of the 3 RAG modes.

Public API:
    ChatSession  — stateful wrapper; pick mode=naive|enhanced|agentic
    Turn         — single chat entry (role + content + timestamp)
    rewrite_query — standalone-query rewriter (exposed for testing)
"""

from ragtrial.chat.rewriter import rewrite_query
from ragtrial.chat.session import ChatSession, Turn

__all__ = ["ChatSession", "Turn", "rewrite_query"]
