"""Preprocess raw files into cleaned Documents (pickle + json).

Walks each domain's raw dir (multi-file aware) via its Source.build_documents().

Usage:
    uv run python scripts/preprocess.py --source dukcapil
    uv run python scripts/preprocess.py --source opd
    uv run python scripts/preprocess.py --source all
"""

from __future__ import annotations

import argparse

from ragtrial.sources import SOURCES, save_docs


def run_one(name: str) -> None:
    src = SOURCES[name]
    print(f"=== Preprocessing: {name} ===")
    print(f"  Raw dir : {src.raw_dir}")
    docs = src.build_documents()
    print(f"  Cleaned {len(docs)} docs")
    save_docs(docs, src.processed_pkl.with_suffix(""))


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--source",
        choices=list(SOURCES) + ["all"],
        required=True,
        help="Domain to preprocess",
    )
    args = ap.parse_args()

    names = list(SOURCES) if args.source == "all" else [args.source]
    for name in names:
        run_one(name)


if __name__ == "__main__":
    main()
