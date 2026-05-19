"""Shared LLM + Embeddings singletons.

Import these instead of constructing new instances — keeps token usage predictable
and avoids re-initializing the Gemini client per module.
"""

from __future__ import annotations

from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings

from ragtrial.config import load_env

load_env()

LLM_MODEL: str = "gemini-2.5-flash"
EMBEDDING_MODEL: str = "models/gemini-embedding-2"
EMBEDDING_DIM: int = 768

llm = ChatGoogleGenerativeAI(
    model=LLM_MODEL,
    temperature=0.1,
    max_tokens=1024,
)

embeddings = GoogleGenerativeAIEmbeddings(
    model=EMBEDDING_MODEL,
    task_type="retrieval_query",
    output_dimensionality=EMBEDDING_DIM,
)


def make_judge_llm(temperature: float = 0.0, max_tokens: int = 256) -> ChatGoogleGenerativeAI:
    """Construct a separate LLM client tuned for eval (deterministic, short)."""
    return ChatGoogleGenerativeAI(
        model=LLM_MODEL,
        temperature=temperature,
        max_tokens=max_tokens,
    )
