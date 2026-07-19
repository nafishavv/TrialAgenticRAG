"""Tracing — automatic per-query research trace persistence (JSONL).

Two tiers, linked by query_id:
  - BASIC  (data/traces/traces-YYYY-MM-DD.jsonl): compact one-liner per query —
    ids, timings, queries, decisions, sources, answer. ~1-2 KB. Always on.
  - FULL   (data/traces/full/traces-full-YYYY-MM-DD.jsonl): the heavy payload —
    retrieved chunk texts + scores + complete meta (incl. agentic step trace).

Level via $RAGTRIAL_TRACE = off | basic | full (default: full — this is the
thesis research data). The single write hook lives in ChatSession.ask(), so
web + CLI are traced automatically; eval bypasses ChatSession and keeps its
own per-query checkpoints.
"""

from ragtrial.tracing.schema import build_trace_records
from ragtrial.tracing.writer import TraceWriter, default_writer

__all__ = ["TraceWriter", "default_writer", "build_trace_records"]
