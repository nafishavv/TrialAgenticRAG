"""TraceWriter — append-only JSONL sink for query traces.

Sync append by design: records are a few KB and a local append is sub-ms;
background threads would only add failure modes at thesis scale. write() must
NEVER raise — a lost trace beats a lost answer.
"""

from __future__ import annotations

import json
import logging
import os
import subprocess
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Optional

from ragtrial import __version__
from ragtrial.config import PROJECT_ROOT, TRACES_DIR, TRACES_FULL_DIR

log = logging.getLogger(__name__)

VALID_LEVELS = ("off", "basic", "full")


def _git_sha() -> str:
    try:
        return subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            cwd=PROJECT_ROOT, capture_output=True, text=True, timeout=5,
        ).stdout.strip()
    except Exception:
        return ""


class TraceWriter:
    """Appends basic (and optionally full) trace records to daily JSONL files."""

    def __init__(
        self,
        directory: Optional[Path] = None,
        full_directory: Optional[Path] = None,
        level: Optional[str] = None,
    ):
        self.directory = directory or TRACES_DIR
        self.full_directory = full_directory or TRACES_FULL_DIR
        lvl = (level or os.getenv("RAGTRIAL_TRACE", "full")).lower()
        if lvl not in VALID_LEVELS:
            log.warning("RAGTRIAL_TRACE=%r tidak dikenal, pakai 'full'", lvl)
            lvl = "full"
        self.level = lvl
        self._git_sha = _git_sha()

    def write(self, basic: Dict[str, Any], full: Optional[Dict[str, Any]] = None) -> None:
        """Persist one query's records. Never raises."""
        try:
            if self.level == "off":
                return
            day = datetime.now().strftime("%Y-%m-%d")
            basic = {**basic, "app_version": __version__, "git_sha": self._git_sha}
            self._append(self.directory / f"traces-{day}.jsonl", basic)
            if self.level == "full" and full is not None:
                self._append(self.full_directory / f"traces-full-{day}.jsonl", full)
        except Exception:
            log.warning("trace write failed", exc_info=True)

    @staticmethod
    def _append(path: Path, record: Dict[str, Any]) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "a", encoding="utf-8") as f:
            f.write(json.dumps(record, ensure_ascii=False, default=str) + "\n")


default_writer = TraceWriter()
"""Shared writer used by ChatSession unless a session overrides it."""
