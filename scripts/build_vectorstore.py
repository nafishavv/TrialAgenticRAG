"""Build a Chroma vector store for a domain: load processed pkl -> chunk -> embed -> persist.

Usage:
    uv run python scripts/build_vectorstore.py --source dukcapil
    uv run python scripts/build_vectorstore.py --source opd
    uv run python scripts/build_vectorstore.py --source all
    uv run python scripts/build_vectorstore.py --source unified  # copy per-domain vectors -> _unified (no re-embed)
"""

from __future__ import annotations

import argparse
import pickle
from pathlib import Path

from ragtrial.sources import SOURCES
from ragtrial.vectorstore.builder import build_vectorstore


def run_one(name: str, no_wipe: bool) -> None:
    src = SOURCES[name]
    cap = src.capability
    print(f"=== Building vector store: {name} ===")
    print(f"  Processed pkl  : {src.processed_pkl}")
    print(f"  Collection     : {cap.collection_name}")
    print(f"  Persist dir    : {cap.persist_directory}")

    with open(src.processed_pkl, "rb") as f:
        docs = pickle.load(f)
    print(f"  Loaded {len(docs)} cleaned docs")

    chunks = src.chunk(docs)
    print(f"  After chunking : {len(chunks)} chunks")

    build_vectorstore(
        chunks=chunks,
        collection_name=cap.collection_name,
        persist_directory=Path(cap.persist_directory),
        wipe=not no_wipe,
    )


def run_unified() -> None:
    """Build the unified naive-RAG collection by copying per-domain vectors."""
    from ragtrial.capabilities.registry import SEARCHABLE_CAPABILITIES
    from ragtrial.vectorstore.unified import build_unified

    print("=== Building unified collection (copy vectors, no re-embed) ===")
    build_unified(SEARCHABLE_CAPABILITIES)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--source", choices=list(SOURCES) + ["all", "unified"], required=True)
    ap.add_argument(
        "--no-wipe",
        action="store_true",
        help="Do not wipe existing vector store before building (append mode)",
    )
    args = ap.parse_args()

    if args.source == "unified":
        run_unified()
        return

    names = list(SOURCES) if args.source == "all" else [args.source]
    for name in names:
        run_one(name, args.no_wipe)


if __name__ == "__main__":
    main()
