"""Preprocess raw PDFs into cleaned Documents (pickle + json).

Usage:
    uv run python scripts/preprocess.py --source dukcapil
    uv run python scripts/preprocess.py --source opd
    uv run python scripts/preprocess.py --source all
"""

from __future__ import annotations

import argparse

from ragtrial.config import (
    DUKCAPIL_PROCESSED_PKL,
    DUKCAPIL_RAW_PDF,
    OPD_PROCESSED_PKL,
    OPD_RAW_PDF,
)
from ragtrial.preprocessing import save_docs
from ragtrial.preprocessing.dukcapil import preprocess as preprocess_dukcapil
from ragtrial.preprocessing.opd import preprocess as preprocess_opd

PIPELINES = {
    "dukcapil": (preprocess_dukcapil, DUKCAPIL_RAW_PDF, DUKCAPIL_PROCESSED_PKL),
    "opd": (preprocess_opd, OPD_RAW_PDF, OPD_PROCESSED_PKL),
}


def run_one(source: str) -> None:
    fn, raw_pdf, out_pkl = PIPELINES[source]
    print(f"=== Preprocessing: {source} ===")
    print(f"  Input : {raw_pdf}")
    docs = fn(raw_pdf)
    print(f"  Cleaned {len(docs)} docs")
    save_docs(docs, out_pkl.with_suffix(""))


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--source",
        choices=list(PIPELINES) + ["all"],
        required=True,
        help="Source to preprocess",
    )
    args = ap.parse_args()

    sources = list(PIPELINES) if args.source == "all" else [args.source]
    for src in sources:
        run_one(src)


if __name__ == "__main__":
    main()
