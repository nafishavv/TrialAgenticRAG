"""Source descriptor + shared persistence helper.

A `Source` co-locates everything about one domain: where its raw files live,
how to turn them into cleaned Documents (multi-file aware), how to chunk them,
and the Capability that serves them at query time. The registry
(`ragtrial.capabilities.registry`) and the build/preprocess CLIs all iterate
the `SOURCES` dict assembled in `ragtrial.sources.__init__`.

Adding a domain = create `sources/<domain>/` with preprocess.py + chunk.py +
capability.py, expose a `source` Source in its __init__, and register it.
"""

from __future__ import annotations

import json
import pickle
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, List

from langchain_core.documents import Document

from ragtrial.capabilities.base import Capability


@dataclass
class Source:
    """Everything about one domain, in one place."""

    name: str
    raw_dir: Path
    """Directory holding this domain's raw files (may contain many / nested)."""
    processed_pkl: Path
    """Where `build_documents()` output is persisted (pkl; json sibling too)."""
    capability: Capability
    """Query-time retrieval capability backing this domain's collection."""
    build_documents: Callable[[], List[Document]]
    """Walk `raw_dir`, parse every file, return cleaned Documents."""
    chunk: Callable[[List[Document]], List[Document]]
    """Turn cleaned Documents into embed-ready chunks."""


def save_docs(docs: List[Document], out_base: Path) -> None:
    """Persist docs to `<out_base>.pkl` and `<out_base>.json`, verify round-trip.

    `out_base` is a path WITHOUT extension (e.g. `data/processed/dukcapil`).
    """
    out_base.parent.mkdir(parents=True, exist_ok=True)

    pkl_path = out_base.with_suffix(".pkl")
    with open(pkl_path, "wb") as f:
        pickle.dump(docs, f)

    json_path = out_base.with_suffix(".json")
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(
            [{"page_content": d.page_content, "metadata": d.metadata} for d in docs],
            f,
            ensure_ascii=False,
            indent=2,
        )

    # Round-trip verification
    with open(json_path, "r", encoding="utf-8") as f:
        json_docs = json.load(f)
    assert len(json_docs) == len(docs), "Count mismatch between pkl and json"
    for i, (d, j) in enumerate(zip(docs, json_docs)):
        assert d.page_content == j["page_content"], f"page_content mismatch at idx {i}"
        assert d.metadata == j["metadata"], f"metadata mismatch at idx {i}"

    print(f"Saved {len(docs)} docs -> {pkl_path.name} + {json_path.name}")
