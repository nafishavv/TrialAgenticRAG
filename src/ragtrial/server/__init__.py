"""FastAPI presentation layer — serves the static web UI + a minimal chat API.

Everything RAG goes through ragtrial.chat.ChatSession (the shared service
layer); this package only maps HTTP <-> ChatTurnResult and keeps per-browser
sessions. Nothing debug crosses the wire: the response carries answer, source
labels, latency, and ids — the research detail lives in data/traces/.
"""
