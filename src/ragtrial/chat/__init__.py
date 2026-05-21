"""Conversation layer — multi-turn chat on top of naive-combined RAG.

Public API:
    ChatSession  — stateful wrapper around ask_main()
    Turn         — single chat entry (role + content + timestamp)
    rewrite_query — standalone-query rewriter (exposed for testing)
"""

from ragtrial.chat.rewriter import rewrite_query
from ragtrial.chat.session import ChatSession, Turn

__all__ = ["ChatSession", "Turn", "rewrite_query"]
