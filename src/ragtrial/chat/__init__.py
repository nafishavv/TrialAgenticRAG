"""Conversation layer — multi-turn chat over any of the 3 RAG modes.

Public API:
    ChatSession    — stateful wrapper; pick mode=naive|enhanced|agentic
    ChatTurnResult — typed envelope for one processed turn (wraps RagResult)
    Turn           — single chat entry (role + content + timestamp)
    rewrite_query  — standalone-query rewriter (exposed for testing)
"""

from ragtrial.chat.rewriter import rewrite_query
from ragtrial.chat.session import ChatSession, ChatTurnResult, Turn

__all__ = ["ChatSession", "ChatTurnResult", "Turn", "rewrite_query"]
