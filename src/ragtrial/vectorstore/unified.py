"""Build the `_unified` collection for naive RAG by COPYING existing vectors.

Naive RAG needs one collection spanning all domains. Instead of re-embedding
(API cost + rate limits), we read the document embeddings already stored in each
per-domain collection and write them into a single unified collection. Free and
instant. Each copied doc gets `_source=<domain>` stamped so eval/format helpers
can still attribute it.

Re-run after rebuilding any per-domain store so the unified copy stays in sync.
"""

from __future__ import annotations

import shutil
from pathlib import Path
from typing import Dict

import chromadb

from ragtrial.capabilities.base import Capability
from ragtrial.config import UNIFIED_COLLECTION, UNIFIED_VECTOR_STORE


def build_unified(capabilities: Dict[str, Capability], wipe: bool = True) -> int:
    """Copy all vector-source collections into one unified collection.

    Skips capabilities that aren't backed by a Chroma collection (e.g. tools).
    Returns total vectors copied.
    """
    dest = Path(UNIFIED_VECTOR_STORE)
    if wipe and dest.exists():
        shutil.rmtree(dest)
        print(f"Wiped {dest}")
    dest.mkdir(parents=True, exist_ok=True)

    client = chromadb.PersistentClient(path=str(dest))
    uni = client.get_or_create_collection(UNIFIED_COLLECTION)

    total = 0
    for name, cap in capabilities.items():
        coll_name = getattr(cap, "collection_name", None)
        persist_dir = getattr(cap, "persist_directory", None)
        if not coll_name or not persist_dir:
            print(f"  skip {name} (no Chroma collection)")
            continue

        src = chromadb.PersistentClient(path=str(persist_dir)).get_collection(coll_name)
        data = src.get(include=["embeddings", "documents", "metadatas"])
        ids = data["ids"]
        if not ids:
            print(f"  skip {name} (empty)")
            continue

        uni.add(
            ids=[f"{name}:{i}" for i in ids],
            embeddings=[list(e) for e in data["embeddings"]],
            documents=data["documents"],
            metadatas=[{**(m or {}), "_source": name} for m in data["metadatas"]],
        )
        total += len(ids)
        print(f"  copied {len(ids)} vectors from {name} ({coll_name})")

    print(f"Done: {total} vectors -> unified collection '{UNIFIED_COLLECTION}' @ {dest}")
    return total
