"""Preprocessing modules — extract clean Documents from raw PDFs.

Each source has its own submodule with a `preprocess(input_pdf, output_dir)`
entry point that returns the cleaned `list[Document]` AND writes pkl+json
alongside it. The orchestrating CLI lives at `scripts/preprocess.py`.
"""

from __future__ import annotations

import json
import pickle
from pathlib import Path
from typing import List

from langchain_core.documents import Document


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
