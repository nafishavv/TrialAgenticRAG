"""One-off question CLI: route a question through naive-combined or agentic RAG.

Usage:
    uv run python scripts/ask.py "Apa syarat KTP elektronik?"
    uv run python scripts/ask.py "Alamat Disdukcapil?" --mode agentic
    uv run python scripts/ask.py "..." --mode naive --quiet
"""

from __future__ import annotations

import argparse


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("question", help="The question to answer")
    ap.add_argument(
        "--mode",
        choices=["naive", "agentic"],
        default="agentic",
        help="Pipeline to use (default: agentic)",
    )
    ap.add_argument("--quiet", action="store_true", help="Suppress verbose pipeline logs")
    args = ap.parse_args()

    if args.mode == "agentic":
        from ragtrial.rag.agentic import ask_agentic as ask
    else:
        from ragtrial.rag.naive_combined import ask_main as ask

    result = ask(args.question, verbose=not args.quiet)
    if args.quiet:
        print(result["answer"])


if __name__ == "__main__":
    main()
