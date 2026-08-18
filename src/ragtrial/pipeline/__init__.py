"""Composable stages for Enhanced RAG.

Primitives: Stage (ABC), RagState (working state), Pipeline (ordered runner).
Stage libraries each expose a factory dict so `rag/enhanced.py` can assemble a
pipeline from an EnhancedRAGConfig:

    REWRITERS  rewrite.py   passthrough | hyde | multiquery*
    RERANKERS  rerank.py    none | cross_encoder*
    (retrieve + generate are single stages, parameterized not selected)

(* = stub, raises NotImplementedError until filled in)
"""

from __future__ import annotations

from ragtrial.pipeline.base import Pipeline, RagState, Stage
from ragtrial.pipeline.generate import GenerateStage
from ragtrial.pipeline.intent import INTENT_GATES, IntentStage
from ragtrial.pipeline.rerank import RERANKERS
from ragtrial.pipeline.retrieve import RetrieveStage
from ragtrial.pipeline.rewrite import REWRITERS

__all__ = [
    "Stage",
    "RagState",
    "Pipeline",
    "REWRITERS",
    "RERANKERS",
    "INTENT_GATES",
    "IntentStage",
    "RetrieveStage",
    "GenerateStage",
]
