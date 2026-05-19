"""Build a Chroma vector store for a source: load processed pkl -> chunk -> embed -> persist.

Usage:
    uv run python scripts/build_vectorstore.py --source dukcapil
    uv run python scripts/build_vectorstore.py --source opd
    uv run python scripts/build_vectorstore.py --source all
"""

from __future__ import annotations

import argparse
import pickle
from pathlib import Path

from ragtrial.capabilities.instances.dukcapil import dukcapil_capability
from ragtrial.capabilities.instances.opd import opd_capability
from ragtrial.chunking import dukcapil as chunk_dukcapil
from ragtrial.chunking import opd as chunk_opd
from ragtrial.config import DUKCAPIL_PROCESSED_PKL, OPD_PROCESSED_PKL
from ragtrial.vectorstore.builder import build_vectorstore

PIPELINES = {
    "dukcapil": (dukcapil_capability, chunk_dukcapil.chunk_for_vectorstore, DUKCAPIL_PROCESSED_PKL),
    "opd": (opd_capability, chunk_opd.chunk_for_vectorstore, OPD_PROCESSED_PKL),
}


def run_one(source: str, no_wipe: bool) -> None:
    cap, chunker, pkl_path = PIPELINES[source]
    print(f"=== Building vector store: {source} ===")
    print(f"  Processed pkl  : {pkl_path}")
    print(f"  Collection     : {cap.collection_name}")
    print(f"  Persist dir    : {cap.persist_directory}")

    with open(pkl_path, "rb") as f:
        docs = pickle.load(f)
    print(f"  Loaded {len(docs)} cleaned docs")

    chunks = chunker(docs)
    print(f"  After chunking : {len(chunks)} chunks")

    build_vectorstore(
        chunks=chunks,
        collection_name=cap.collection_name,
        persist_directory=Path(cap.persist_directory),
        wipe=not no_wipe,
    )


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--source", choices=list(PIPELINES) + ["all"], required=True)
    ap.add_argument(
        "--no-wipe",
        action="store_true",
        help="Do not wipe existing vector store before building (append mode)",
    )
    args = ap.parse_args()

    sources = list(PIPELINES) if args.source == "all" else [args.source]
    for src in sources:
        run_one(src, args.no_wipe)


if __name__ == "__main__":
    main()
