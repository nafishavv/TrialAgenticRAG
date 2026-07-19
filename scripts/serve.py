"""Launch the web app (FastAPI + static frontend).

    uv run python scripts/serve.py
    uv run python scripts/serve.py --port 8080
    uv run python scripts/serve.py --reload     # dev only; flaky with torch on Windows

Then open http://127.0.0.1:8000
"""

from __future__ import annotations

import argparse

import uvicorn


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=8000)
    ap.add_argument("--reload", action="store_true",
                    help="Auto-reload on code changes (dev; can be flaky with torch)")
    args = ap.parse_args()

    uvicorn.run(
        "ragtrial.server.app:app",
        host=args.host,
        port=args.port,
        reload=args.reload,
    )


if __name__ == "__main__":
    main()
