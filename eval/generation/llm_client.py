"""Generation LLM client: proactive pacing + rate-limit retry.

Wraps `make_judge_llm` (gemini-2.5-flash, chat quota — separate from the heavily
throttled embedding quota). Retry logic mirrors
`ragtrial/vectorstore/builder.py`: back off on 429/RESOURCE_EXHAUSTED/503/
UNAVAILABLE with a growing wait. A DryRunLLM lets the whole pipeline run without
any API calls for plumbing tests.
"""

from __future__ import annotations

import time
from typing import Protocol

from langchain_google_genai import ChatGoogleGenerativeAI

from ragtrial.llm import LLM_MODEL

_RETRYABLE = ("429", "RESOURCE_EXHAUSTED", "503", "UNAVAILABLE")


class LLMClient(Protocol):
    def generate(self, prompt: str) -> str: ...


class GenLLM:
    """Paced, retrying wrapper around the chat LLM."""

    def __init__(
        self,
        temperature: float = 0.5,
        max_tokens: int = 2048,
        inter_call_delay: float = 4.0,
        initial_wait: int = 30,
        max_retries: int = 5,
    ) -> None:
        self._llm = ChatGoogleGenerativeAI(
            model=LLM_MODEL,
            temperature=temperature,
            max_tokens=max_tokens,
            thinking_budget=0,  # gemini-2.5: disable reasoning so JSON isn't truncated
        )
        self.inter_call_delay = inter_call_delay
        self.initial_wait = initial_wait
        self.max_retries = max_retries

    def generate(self, prompt: str) -> str:
        if self.inter_call_delay:
            time.sleep(self.inter_call_delay)
        wait = self.initial_wait
        for attempt in range(self.max_retries + 1):
            try:
                return self._llm.invoke(prompt).content
            except Exception as e:  # noqa: BLE001 — inspect message for retryable codes
                msg = str(e)
                if any(code in msg for code in _RETRYABLE) and attempt < self.max_retries:
                    print(f"  Rate limited / unavailable. Retry {attempt + 1}/{self.max_retries} in {wait}s...",
                          flush=True)
                    time.sleep(wait)
                    wait = int(wait * 1.5)
                else:
                    raise
        raise RuntimeError("unreachable")


class DryRunLLM:
    """Deterministic templated output — no API. For --dry-run plumbing tests."""

    def __init__(self) -> None:
        self._n = 0

    def generate(self, prompt: str) -> str:
        self._n += 1
        is_none = "DI LUAR CAKUPAN" in prompt
        if is_none:
            return ('{"question": "DRYRUN out-of-scope #%d?", '
                    '"notes": "dryrun none"}' % self._n)
        return (
            '{"question": "DRYRUN pertanyaan #%d?", "difficulty": "medium", '
            '"expected_answer": "DRYRUN jawaban ringkas.", '
            '"expected_facts": ["DRYRUN fakta 1", "DRYRUN fakta 2"], '
            '"notes": "dryrun generated"}' % self._n
        )
