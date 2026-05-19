"""One-shot migration script: rewrite legacy imports in moved notebooks
to use the new ragtrial.* package. Run once after the Tahap 2 reorg.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

NEW_RAG_CHAT_MAIN_IMPORT = '''from ragtrial.capabilities import CAPABILITIES, format_context as _format_context
from ragtrial.llm import embeddings, llm
from ragtrial.rag.naive_combined import ask_main, retrieve_combined
from ragtrial.rag.prompts import PROMPT_COMBINED

# Force eager init so legacy notebook cells can grab the raw Chroma / docs handles.
CAPABILITIES["dukcapil"]._ensure_initialized()
CAPABILITIES["opd"]._ensure_initialized()
vs_dukcapil = CAPABILITIES["dukcapil"]._vectorstore
vs_opd = CAPABILITIES["opd"]._vectorstore
dukcapil_docs_all = CAPABILITIES["dukcapil"]._all_docs
opd_docs_all = CAPABILITIES["opd"]._all_docs

def retrieve_dukcapil_hybrid(q, k=4):
    return CAPABILITIES["dukcapil"].invoke(q, k=k)

def retrieve_opd_hybrid(q, k=4):
    return CAPABILITIES["opd"].invoke(q, k=k)

def format_context_combined(docs):
    return _format_context(docs, CAPABILITIES)

print(f"✓ Module loaded")
print(f"   Dukcapil docs in-memory: {len(dukcapil_docs_all)}")
print(f"   OPD docs in-memory:      {len(opd_docs_all)}")
'''

NEW_COMPARE_IMPORT = '''from ragtrial.capabilities import CAPABILITIES, format_context as _format_context
from ragtrial.llm import embeddings, llm
from ragtrial.rag.agentic import agentic_app, ask_agentic
from ragtrial.rag.naive_combined import ask_main as ask_naive_combined
from ragtrial.rag.prompts import (
    PROMPT_COMBINED,
    PROMPT_NONE,
    PROMPT_SINGLE,
)

# Legacy aliases for the per-source prompts (now unified to PROMPT_SINGLE).
PROMPT_DUKCAPIL = PROMPT_SINGLE
PROMPT_OPD = PROMPT_SINGLE
PROMPT_BOTH = PROMPT_COMBINED

CAPABILITIES["dukcapil"]._ensure_initialized()
CAPABILITIES["opd"]._ensure_initialized()
vs_dukcapil = CAPABILITIES["dukcapil"]._vectorstore
vs_opd = CAPABILITIES["opd"]._vectorstore
opd_docs_all = CAPABILITIES["opd"]._all_docs

def retrieve_dukcapil(q, k=5):
    return CAPABILITIES["dukcapil"].invoke(q, k=k)

def retrieve_opd(q, k=5):
    return CAPABILITIES["opd"].invoke(q, k=k)

def format_context(docs):
    return _format_context(docs, CAPABILITIES)

print("✓ Agentic + Naive Combined pipelines imported via ragtrial package")
'''


def _set_source(cell: dict, new_src: str) -> None:
    cell["source"] = [line + "\n" for line in new_src.rstrip().split("\n")]
    if cell["source"]:
        # Drop trailing newline on last line, jupyter convention
        cell["source"][-1] = cell["source"][-1].rstrip("\n")
    cell["outputs"] = []
    cell["execution_count"] = None


def patch(path: Path, marker: str, new_src: str) -> None:
    nb = json.loads(path.read_text(encoding="utf-8"))
    patched = 0
    for cell in nb["cells"]:
        if cell["cell_type"] != "code":
            continue
        src = "".join(cell["source"]) if isinstance(cell["source"], list) else cell["source"]
        if marker in src:
            _set_source(cell, new_src)
            patched += 1
    if patched == 0:
        print(f"  [warn] no cell matched in {path.name}")
        return
    path.write_text(json.dumps(nb, ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"  patched {patched} cell(s) in {path.name}")


def main() -> None:
    print("Migrating notebooks to ragtrial.* imports")
    patch(
        ROOT / "notebooks" / "reports" / "rag_chat_main.ipynb",
        marker="from rag_chat_main import",
        new_src=NEW_RAG_CHAT_MAIN_IMPORT,
    )
    patch(
        ROOT / "notebooks" / "reports" / "compare_agentic_vs_naive.ipynb",
        marker="from agentic_rag import",
        new_src=NEW_COMPARE_IMPORT,
    )
    print("Done.")


if __name__ == "__main__":
    main()
