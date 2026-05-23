"""Composable stages for Enhanced RAG.

Primitives: Stage (ABC), RagState (working state), Pipeline (ordered runner).
Stage libraries each expose a factory dict so `rag/enhanced.py` can assemble a
pipeline from an EnhancedRAGConfig:

    REWRITERS  rewrite.py   passthrough | hyde* | multiquery*
    ROUTERS    route.py     none | semantic | llm
    RERANKERS  rerank.py    none | cross_encoder*
    (retrieve + generate are single stages, parameterized not selected)

(* = stub, raises NotImplementedError until filled in)
"""

from __future__ import annotations

from ragtrial.pipeline.base import Pipeline, RagState, Stage
from ragtrial.pipeline.generate import GenerateStage
from ragtrial.pipeline.rerank import RERANKERS
from ragtrial.pipeline.retrieve import RetrieveStage
from ragtrial.pipeline.rewrite import REWRITERS
from ragtrial.pipeline.route import ROUTERS

__all__ = [
    "Stage",
    "RagState",
    "Pipeline",
    "REWRITERS",
    "ROUTERS",
    "RERANKERS",
    "RetrieveStage",
    "GenerateStage",
]
