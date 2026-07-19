"""In-memory session store: {session_id: ChatSession} with TTL + LRU cap.

Good enough for a single-process research prototype — sessions hold only
trimmed text history (a few KB each). Thread-safe via one lock because FastAPI
runs sync endpoints in a threadpool. An unknown/expired session_id silently
becomes a fresh session; the client just keeps using whatever id the server
returns.
"""

from __future__ import annotations

import threading
import time
from typing import Dict, Optional, Tuple

from ragtrial.chat import ChatSession


class SessionStore:
    def __init__(self, ttl_seconds: float = 3600.0, max_sessions: int = 200):
        self.ttl_seconds = ttl_seconds
        self.max_sessions = max_sessions
        self._lock = threading.Lock()
        self._sessions: Dict[str, Tuple[ChatSession, float]] = {}

    def get_or_create(self, session_id: Optional[str], mode: str) -> ChatSession:
        now = time.time()
        with self._lock:
            self._sweep(now)
            entry = self._sessions.get(session_id) if session_id else None
            if entry is not None:
                session = entry[0]
            else:
                session = ChatSession(mode=mode, client="web")
            session.set_mode(mode)
            self._sessions[session.session_id] = (session, now)
            self._evict()
            return session

    def _sweep(self, now: float) -> None:
        expired = [sid for sid, (_, last) in self._sessions.items()
                   if now - last > self.ttl_seconds]
        for sid in expired:
            del self._sessions[sid]

    def _evict(self) -> None:
        while len(self._sessions) > self.max_sessions:
            oldest = min(self._sessions, key=lambda sid: self._sessions[sid][1])
            del self._sessions[oldest]
