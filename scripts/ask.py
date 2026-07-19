"""Question CLI: one-shot OR interactive chat, across all 3 RAG modes.

One-shot (single question, no history):
    uv run python scripts/ask.py "Apa syarat KTP elektronik?"
    uv run python scripts/ask.py "Alamat Disdukcapil?" --mode agentic
    uv run python scripts/ask.py "..." --mode naive --quiet

Interactive chat (multi-turn, with conversation memory):
    uv run python scripts/ask.py --chat
    uv run python scripts/ask.py --chat --mode agentic
    uv run python scripts/ask.py --chat --no-rewrite --max-turns 10

Modes: naive (fan-out dense baseline) | enhanced (semantic+dense pipeline) |
       agentic (LLM tool-calling loop). Chat commands: 'exit'/'quit', 'reset'.

Every query is traced to data/traces/ (see ragtrial.tracing); --no-trace disables.
"""

from __future__ import annotations

import argparse
import sys

from ragtrial.modes import DEFAULT_MODE, MODES


def _make_session(args: argparse.Namespace, rewrite: bool):
    from ragtrial.chat import ChatSession

    kwargs = {"trace_writer": None} if args.no_trace else {}
    return ChatSession(
        mode=args.mode,
        max_history_turns=args.max_turns,
        rewrite_followups=rewrite,
        client="cli",
        **kwargs,
    )


def _print_stats(turn) -> None:
    t = turn.timings
    pt = turn.result.timings
    rewrite_note = ""
    if turn.rewrite_applied:
        rewrite_note = f' [rewritten: "{turn.effective_query}"]'
    print(
        f"     [src={turn.source_used} | rewrite={t['rewrite_followup']:.2f}s | "
        f"retrieve={pt.get('retrieve', 0.0):.2f}s | generate={pt.get('generate', 0.0):.2f}s | "
        f"total={t['total']:.2f}s | docs={len(turn.documents)}]{rewrite_note}\n"
    )


def _run_oneshot(args: argparse.Namespace) -> None:
    session = _make_session(args, rewrite=False)
    turn = session.ask(args.question, verbose=not args.quiet)
    if args.quiet:
        print(turn.answer)


def _run_chat(args: argparse.Namespace) -> None:
    session = _make_session(args, rewrite=not args.no_rewrite)

    print(
        f"[Chat mode - {args.mode} RAG | max_turns={args.max_turns} | "
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
            turn = session.ask(user_input)
        except Exception as e:
            print(f"[Error: {e}]\n", file=sys.stderr)
            continue

        print(f"Bot: {turn.answer}")
        _print_stats(turn)


def main() -> None:
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    ap.add_argument("question", nargs="?", help="Question for one-shot mode (omit with --chat)")
    ap.add_argument("--mode", choices=MODES, default=DEFAULT_MODE, help=f"RAG pipeline (default: {DEFAULT_MODE})")
    ap.add_argument("--quiet", action="store_true", help="One-shot: print answer only")
    ap.add_argument("--chat", action="store_true", help="Enter interactive chat REPL")
    ap.add_argument("--max-turns", type=int, default=5, help="Chat: max history turns (default: 5)")
    ap.add_argument("--no-rewrite", action="store_true", help="Chat: disable follow-up query rewriting")
    ap.add_argument("--no-trace", action="store_true", help="Disable trace persistence for this run")
    args = ap.parse_args()

    if args.chat:
        _run_chat(args)
    else:
        if not args.question:
            ap.error("question is required (or use --chat for interactive mode)")
        _run_oneshot(args)


if __name__ == "__main__":
    main()
