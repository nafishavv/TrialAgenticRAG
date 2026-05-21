"""Question CLI: one-shot OR interactive chat mode.

One-shot (single question, no history):
    uv run python scripts/ask.py "Apa syarat KTP elektronik?"
    uv run python scripts/ask.py "Alamat Disdukcapil?" --mode agentic
    uv run python scripts/ask.py "..." --mode naive --quiet

Interactive chat (multi-turn, with conversation memory):
    uv run python scripts/ask.py --chat
    uv run python scripts/ask.py --chat --no-rewrite
    uv run python scripts/ask.py --chat --max-turns 10

Chat commands: 'exit'/'quit' to leave, 'reset' to clear history.
"""

from __future__ import annotations

import argparse
import sys


def _run_oneshot(args: argparse.Namespace) -> None:
    if args.mode == "agentic":
        from ragtrial.rag.agentic import ask_agentic as ask
    else:
        from ragtrial.rag.naive_combined import ask_main as ask

    result = ask(args.question, verbose=not args.quiet)
    if args.quiet:
        print(result["answer"])


def _run_chat(args: argparse.Namespace) -> None:
    from ragtrial.chat import ChatSession

    session = ChatSession(
        max_history_turns=args.max_turns,
        rewrite_followups=not args.no_rewrite,
    )

    print(
        f"[Chat mode - naive RAG | max_turns={args.max_turns} | "
        f"rewrite={'on' if not args.no_rewrite else 'off'}]"
    )
    print("Commands: 'exit'/'quit' = keluar, 'reset' = hapus history\n")

    while True:
        try:
            user_input = input("You: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\n[Bye]")
            return

        if not user_input:
            continue
        if user_input.lower() in {"exit", "quit"}:
            print("[Bye]")
            return
        if user_input.lower() == "reset":
            session.reset()
            print("[History cleared]\n")
            continue

        try:
            result = session.ask(user_input)
        except Exception as e:
            print(f"[Error: {e}]\n", file=sys.stderr)
            continue

        t = result["timings"]
        rewrite_note = ""
        if result["rewritten_query"] != result["original_query"]:
            rewrite_note = f' [rewritten: "{result["rewritten_query"]}"]'

        print(f"Bot: {result['answer']}")
        print(
            f"     [rewrite={t['rewrite']:.2f}s | retrieve={t['retrieve']:.2f}s | "
            f"generate={t['generate']:.2f}s | total={t['total']:.2f}s | "
            f"docs={len(result['documents'])}]{rewrite_note}\n"
        )


def main() -> None:
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    ap.add_argument(
        "question",
        nargs="?",
        help="Question for one-shot mode (omit when using --chat)",
    )
    ap.add_argument(
        "--mode",
        choices=["naive", "agentic"],
        default="agentic",
        help="One-shot pipeline (default: agentic). Ignored in --chat mode (always naive).",
    )
    ap.add_argument("--quiet", action="store_true", help="One-shot: print answer only")
    ap.add_argument("--chat", action="store_true", help="Enter interactive chat REPL")
    ap.add_argument(
        "--max-turns",
        type=int,
        default=5,
        help="Chat: max history turns to keep (default: 5)",
    )
    ap.add_argument(
        "--no-rewrite",
        action="store_true",
        help="Chat: disable LLM query rewriting for follow-ups",
    )
    args = ap.parse_args()

    if args.chat:
        _run_chat(args)
    else:
        if not args.question:
            ap.error("question is required (or use --chat for interactive mode)")
        _run_oneshot(args)


if __name__ == "__main__":
    main()
