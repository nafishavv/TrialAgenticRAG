"""Generic Chroma vector store builder with batched embedding + rate-limit retry.

Used by `scripts/build_vectorstore.py` to (re)build a capability's vector store
from cleaned chunks. Handles Gemini's 100/min rate limit via exponential backoff.
"""

from __future__ import annotations

import shutil
import time
from pathlib import Path
from typing import List

from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_google_genai import GoogleGenerativeAIEmbeddings

from ragtrial.llm import EMBEDDING_DIM, EMBEDDING_MODEL


def _embeddings_for_indexing() -> GoogleGenerativeAIEmbeddings:
    """Indexing uses task_type='retrieval_document' (vs 'retrieval_query' at query time)."""
    return GoogleGenerativeAIEmbeddings(
        model=EMBEDDING_MODEL,
        task_type="retrieval_document",
        output_dimensionality=EMBEDDING_DIM,
    )


def build_vectorstore(
    chunks: List[Document],
    collection_name: str,
    persist_directory: Path,
    batch_size: int = 40,
    wipe: bool = True,
    initial_wait: int = 70,
    max_retries: int = 5,
    inter_batch_delay: float = 0.0,
) -> int:
    """Embed `chunks` and persist into Chroma. Returns total embedded count.

    Args:
      chunks: Documents ready for embedding (already chunked appropriately).
      collection_name: Chroma collection name.
      persist_directory: Absolute path to where Chroma should write.
      batch_size: Chunks per embedding API call (40 = safe under 100/min limit).
      wipe: If True, wipe `persist_directory` before building (fresh rebuild).
      initial_wait: Seconds to wait on first 429 (grows ×1.5 per retry).
      max_retries: Max retries per batch on rate-limit errors.
      inter_batch_delay: Proactive pause (s) between batches to respect low RPM/TPM
        free-tier limits. Default 0 = no pause (preserves behavior for small domains);
        large corpora on rate-limited models should set this (e.g. 6).
    """
    persist_directory = Path(persist_directory)
    if wipe and persist_directory.exists():
        shutil.rmtree(persist_directory)
        print(f"Wiped {persist_directory}")
    persist_directory.mkdir(parents=True, exist_ok=True)

    embeddings = _embeddings_for_indexing()
    vectorstore: Chroma | None = None

    total_batches = (len(chunks) - 1) // batch_size + 1
    for batch_idx, i in enumerate(range(0, len(chunks), batch_size)):
        batch = chunks[i : i + batch_size]
        batch_num = batch_idx + 1
        print(
            f"Embedding batch {batch_num}/{total_batches} ({len(batch)} chunks)...",
            end=" ",
            flush=True,
        )

        wait = initial_wait
        for attempt in range(max_retries + 1):
            try:
                if vectorstore is None:
                    vectorstore = Chroma.from_documents(
                        documents=batch,
                        embedding=embeddings,
                        collection_name=collection_name,
                        persist_directory=str(persist_directory),
                    )
                else:
                    vectorstore.add_documents(batch)
                print("OK")
                break
            except Exception as e:
                msg = str(e)
                if ("429" in msg or "RESOURCE_EXHAUSTED" in msg or "503" in msg or "UNAVAILABLE" in msg) and attempt < max_retries:
                    print(f"\n  Rate limited / unavailable. Retry {attempt + 1}/{max_retries} in {wait}s...", flush=True)
                    time.sleep(wait)
                    wait = int(wait * 1.5)
                else:
                    raise

        # Proactive pacing: pause between batches (skip after the last one).
        if inter_batch_delay and batch_num < total_batches:
            time.sleep(inter_batch_delay)

    assert vectorstore is not None, "No chunks were embedded"
    count = vectorstore._collection.count()
    assert count == len(chunks), f"Mismatch: stored {count} vs input {len(chunks)}"
    print(f"\nDone: {count} chunks embedded & persisted to {persist_directory}")
    return count
