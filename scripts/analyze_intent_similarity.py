"""Diagnostic: semantic separation of the intent reference set (VALID vs INVALID).

Embeds every utterance in `intent_reference_set.json` with the SAME embeddings
singleton the semantic router uses (`ragtrial.llm.embeddings` — Gemini Embedding 2,
768-d, task_type=retrieval_query) and reports cosine similarity within and across
the two classes:

    V-V : distinct VALID pairs        (20 -> 190 pairs)
    V-I : every VALID x INVALID pair  (20 x 20 -> 400 pairs)
    I-I : distinct INVALID pairs      (20 -> 190 pairs)
    delta = mean(V-V) - mean(V-I)

This only characterizes the reference set; it does not evaluate the router and
does not declare the set good or bad. No thresholds, no outlier removal.

Usage:
    uv run python scripts/analyze_intent_similarity.py
    uv run python scripts/analyze_intent_similarity.py --refresh-cache
    uv run python scripts/analyze_intent_similarity.py --input path/to/set.json
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import statistics
import sys
from dataclasses import dataclass
from datetime import datetime
from itertools import combinations, product
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Tuple

from ragtrial.config import PROJECT_ROOT

DEFAULT_INPUT: Path = PROJECT_ROOT / "intent_reference_set.json"
RESULTS_DIR: Path = PROJECT_ROOT / "results" / "intent_similarity"
REPORT_PATH: Path = RESULTS_DIR / "intent_similarity_report.txt"
PAIRS_CSV_PATH: Path = RESULTS_DIR / "intent_similarity_pairs.csv"
CACHE_PATH: Path = RESULTS_DIR / "embeddings_cache.json"

EXPECTED_VALID = 20
EXPECTED_INVALID = 20


class AnalysisError(RuntimeError):
    """Fatal, user-facing problem — reported without a traceback."""


# ============ data model ============
@dataclass(frozen=True)
class Utterance:
    id: str
    question: str
    label: str  # "valid" | "invalid"


@dataclass(frozen=True)
class Pair:
    pair_type: str  # "V-V" | "V-I" | "I-I"
    a: Utterance
    b: Utterance
    similarity: float


@dataclass(frozen=True)
class Summary:
    pair_type: str
    count: int
    mean: float
    median: float
    stdev: float
    minimum: float
    maximum: float
    p25: float
    p75: float
    p95: float


# ============ input ============
def load_reference_set(path: Path) -> Tuple[List[Utterance], List[Utterance]]:
    """Read the reference set and validate shape + expected sizes."""
    if not path.exists():
        raise AnalysisError(
            f"Input file not found: {path}\n"
            "Expected the intent reference set at the project root "
            "(or pass --input <path>)."
        )
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as e:
        raise AnalysisError(f"Invalid JSON in {path}: {e}") from e

    if not isinstance(raw, dict):
        raise AnalysisError(f"{path}: top level must be a JSON object with 'valid'/'invalid' keys.")

    def parse(key: str, label: str) -> List[Utterance]:
        items = raw.get(key)
        if items is None:
            raise AnalysisError(f"{path}: missing required key '{key}'.")
        if not isinstance(items, list):
            raise AnalysisError(f"{path}: '{key}' must be a list, got {type(items).__name__}.")
        out: List[Utterance] = []
        for i, item in enumerate(items):
            if not isinstance(item, dict):
                raise AnalysisError(f"{path}: {key}[{i}] must be an object with 'id' and 'question'.")
            qid = item.get("id")
            question = item.get("question")
            if not isinstance(qid, str) or not qid.strip():
                raise AnalysisError(f"{path}: {key}[{i}] has a missing or empty 'id'.")
            if not isinstance(question, str) or not question.strip():
                raise AnalysisError(f"{path}: {key}[{i}] (id={qid!r}) has a missing or empty 'question'.")
            out.append(Utterance(id=qid.strip(), question=question.strip(), label=label))
        return out

    valid = parse("valid", "valid")
    invalid = parse("invalid", "invalid")

    if len(valid) != EXPECTED_VALID or len(invalid) != EXPECTED_INVALID:
        raise AnalysisError(
            f"Unexpected reference set size: {len(valid)} VALID + {len(invalid)} INVALID "
            f"(expected {EXPECTED_VALID} + {EXPECTED_INVALID}). Refusing to continue — "
            "pair counts and the report would not be comparable across runs."
        )

    ids = [u.id for u in valid + invalid]
    dupes = sorted({i for i in ids if ids.count(i) > 1})
    if dupes:
        raise AnalysisError(f"{path}: duplicate ids: {', '.join(dupes)}")

    return valid, invalid


# ============ embeddings ============
def _cache_key(utt: Utterance, model: str, dim: int) -> str:
    """Cache identity = id + model/config + exact text, so edited text never reuses
    a stale vector."""
    return f"{model}|{dim}|{utt.id}|{utt.question}"


def _load_cache(path: Path) -> Dict[str, List[float]]:
    if not path.exists():
        return {}
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        print(f"  [cache] ignoring unreadable cache at {path}", flush=True)
        return {}
    entries = data.get("entries") if isinstance(data, dict) else None
    return entries if isinstance(entries, dict) else {}


def _save_cache(path: Path, entries: Dict[str, List[float]], model: str, dim: int) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "_meta": {
            "model": model,
            "dimension": dim,
            "key_format": "model|dimension|id|question",
            "updated": datetime.now().isoformat(timespec="seconds"),
        },
        "entries": entries,
    }
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def get_embeddings(
    utterances: Sequence[Utterance],
    *,
    cache_path: Path = CACHE_PATH,
    use_cache: bool = True,
) -> Tuple[Dict[str, List[float]], str, int]:
    """Embed every utterance with the project's shared embeddings singleton.

    Returns (vectors_by_id, model_name, dimension). Reuses `ragtrial.llm.embeddings`
    — the exact object `IntentStage`'s GeminiEncoder wraps — so the geometry here
    matches the router's, and calls it via `embed_documents` like the router does.
    """
    try:
        from ragtrial.llm import EMBEDDING_DIM, EMBEDDING_MODEL, embeddings
    except AssertionError as e:  # load_env() asserts on a missing key
        raise AnalysisError(
            f"Could not initialize the Gemini client: {e}\n"
            "Set GEMINI_API_KEY in .env at the project root."
        ) from e
    except ImportError as e:
        raise AnalysisError(
            f"Could not import ragtrial.llm: {e}\n"
            "Run through the project environment, e.g. `uv run python scripts/analyze_intent_similarity.py`."
        ) from e

    cache = _load_cache(cache_path) if use_cache else {}
    vectors: Dict[str, List[float]] = {}
    pending: List[Utterance] = []

    for utt in utterances:
        hit = cache.get(_cache_key(utt, EMBEDDING_MODEL, EMBEDDING_DIM))
        if isinstance(hit, list) and hit:
            vectors[utt.id] = [float(x) for x in hit]
        else:
            pending.append(utt)

    print(f"  Cache: {len(vectors)} reused, {len(pending)} to embed", flush=True)

    if pending:
        try:
            fresh = embeddings.embed_documents([u.question for u in pending])
        except Exception as e:  # noqa: BLE001 — surface the API's own message
            raise AnalysisError(
                f"Embedding API call failed: {e}\n"
                "No vectors were substituted — rerun once the API is reachable."
            ) from e

        if len(fresh) != len(pending):
            raise AnalysisError(
                f"Embedding API returned {len(fresh)} vectors for {len(pending)} inputs."
            )
        for utt, vec in zip(pending, fresh):
            if not vec or any(v is None for v in vec):
                raise AnalysisError(f"Embedding API returned an empty vector for {utt.id}.")
            vectors[utt.id] = [float(x) for x in vec]
            cache[_cache_key(utt, EMBEDDING_MODEL, EMBEDDING_DIM)] = vectors[utt.id]

        if use_cache:
            _save_cache(cache_path, cache, EMBEDDING_MODEL, EMBEDDING_DIM)

    dims = {len(v) for v in vectors.values()}
    if len(dims) != 1:
        raise AnalysisError(f"Inconsistent embedding dimensions across utterances: {sorted(dims)}")
    actual_dim = dims.pop()
    if actual_dim != EMBEDDING_DIM:
        print(
            f"  [warn] embedding dimension {actual_dim} != configured {EMBEDDING_DIM}",
            flush=True,
        )

    return vectors, EMBEDDING_MODEL, actual_dim


# ============ math ============
def cosine_similarity(a: Sequence[float], b: Sequence[float]) -> float:
    """Plain cosine similarity — no normalization or rescaling of the result."""
    if len(a) != len(b):
        raise AnalysisError(f"Cannot compare vectors of different length: {len(a)} vs {len(b)}")
    dot = sum(x * y for x, y in zip(a, b))
    na = math.sqrt(sum(x * x for x in a))
    nb = math.sqrt(sum(y * y for y in b))
    if na == 0.0 or nb == 0.0:
        raise AnalysisError("Zero-magnitude embedding encountered; cosine similarity undefined.")
    return dot / (na * nb)


def compute_pairwise_similarities(
    valid: Sequence[Utterance],
    invalid: Sequence[Utterance],
    vectors: Dict[str, List[float]],
) -> List[Pair]:
    """All V-V, V-I and I-I pairs. Unordered pairs appear once; no self-pairs."""
    pairs: List[Pair] = []

    def add(pair_type: str, combos: Iterable[Tuple[Utterance, Utterance]]) -> None:
        for a, b in combos:
            pairs.append(Pair(pair_type, a, b, cosine_similarity(vectors[a.id], vectors[b.id])))

    add("V-V", combinations(valid, 2))
    add("V-I", product(valid, invalid))
    add("I-I", combinations(invalid, 2))
    return pairs


def _percentile(sorted_values: Sequence[float], pct: float) -> float:
    """Linear-interpolation percentile on an already-sorted sequence."""
    if not sorted_values:
        raise AnalysisError("Cannot compute a percentile of an empty sample.")
    if len(sorted_values) == 1:
        return sorted_values[0]
    pos = (len(sorted_values) - 1) * pct
    lo = math.floor(pos)
    hi = math.ceil(pos)
    if lo == hi:
        return sorted_values[int(pos)]
    return sorted_values[lo] + (sorted_values[hi] - sorted_values[lo]) * (pos - lo)


def summarize_similarities(pairs: Sequence[Pair], pair_type: str) -> Summary:
    values = sorted(p.similarity for p in pairs if p.pair_type == pair_type)
    if not values:
        raise AnalysisError(f"No pairs of type {pair_type} to summarize.")
    return Summary(
        pair_type=pair_type,
        count=len(values),
        mean=statistics.fmean(values),
        median=statistics.median(values),
        stdev=statistics.stdev(values) if len(values) > 1 else 0.0,
        minimum=values[0],
        maximum=values[-1],
        p25=_percentile(values, 0.25),
        p75=_percentile(values, 0.75),
        p95=_percentile(values, 0.95),
    )


def get_top_cross_class_pairs(pairs: Sequence[Pair], top_n: int = 10) -> List[Pair]:
    """Highest-similarity V-I pairs — near-boundary cases an average can hide."""
    cross = [p for p in pairs if p.pair_type == "V-I"]
    return sorted(cross, key=lambda p: p.similarity, reverse=True)[:top_n]


# ============ output ============
def _fmt_summary(s: Summary) -> str:
    return (
        f"Pairs: {s.count}\n"
        f"Mean: {s.mean:.4f}\n"
        f"Median: {s.median:.4f}\n"
        f"Std: {s.stdev:.4f}\n"
        f"Min: {s.minimum:.4f}\n"
        f"Max: {s.maximum:.4f}\n"
        f"P25: {s.p25:.4f}\n"
        f"P75: {s.p75:.4f}\n"
        f"P95: {s.p95:.4f}\n"
    )


def build_report(
    *,
    input_path: Path,
    model: str,
    dim: int,
    n_valid: int,
    n_invalid: int,
    summaries: Dict[str, Summary],
    top_pairs: Sequence[Pair],
    timestamp: str,
) -> str:
    sep = "-" * 40
    lines: List[str] = [
        "=" * 40,
        "INTENT REFERENCE SEMANTIC SIMILARITY",
        "=" * 40,
        "",
        f"Generated: {timestamp}",
        f"Input file: {input_path}",
        f"Embedding model: {model}",
        f"Embedding dimension: {dim}",
        "",
        "Reference set:",
        f"VALID: {n_valid}",
        f"INVALID: {n_invalid}",
        "",
    ]
    for pair_type, title in (("V-V", "V-V SIMILARITY"), ("V-I", "V-I SIMILARITY"), ("I-I", "I-I SIMILARITY")):
        lines += [sep, title, sep, _fmt_summary(summaries[pair_type]).rstrip(), ""]

    delta = summaries["V-V"].mean - summaries["V-I"].mean
    lines += [
        sep,
        "DELTA",
        sep,
        f"V-V mean: {summaries['V-V'].mean:.4f}",
        f"V-I mean: {summaries['V-I'].mean:.4f}",
        f"Delta: {delta:+.4f}",
        "",
        sep,
        f"TOP {len(top_pairs)} V-I PAIRS",
        sep,
    ]
    for i, p in enumerate(top_pairs, 1):
        lines += [
            f"{i}. cosine = {p.similarity:.4f}",
            f"   VALID   [{p.a.id}] {p.a.question}",
            f"   INVALID [{p.b.id}] {p.b.question}",
            "",
        ]

    lines += [
        sep,
        "INTERPRETATION",
        sep,
        "If V-V mean similarity exceeds V-I mean similarity, the reference classes",
        "show average semantic separation under this embedding model.",
        "",
        "The V-I pairs listed above are the closest cross-class neighbours; high",
        "values there may indicate potential class-boundary cases, even when the",
        "averages separate cleanly.",
        "",
        "These are descriptive statistics of the reference set only. They do not",
        "prove that the semantic router will classify unseen queries correctly.",
        "",
    ]
    return "\n".join(lines)


def save_results(
    *,
    report: str,
    pairs: Sequence[Pair],
    report_path: Path = REPORT_PATH,
    csv_path: Path = PAIRS_CSV_PATH,
) -> None:
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(report, encoding="utf-8")

    with csv_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(
            ["pair_type", "id_a", "question_a", "id_b", "question_b", "cosine_similarity"]
        )
        for p in pairs:
            writer.writerow(
                [p.pair_type, p.a.id, p.a.question, p.b.id, p.b.question, f"{p.similarity:.6f}"]
            )


# ============ entry point ============
def main() -> int:
    parser = argparse.ArgumentParser(
        description="Measure cosine similarity within/across the intent reference classes."
    )
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT, help="reference set JSON")
    parser.add_argument("--top-n", type=int, default=10, help="how many top V-I pairs to report")
    parser.add_argument(
        "--refresh-cache",
        action="store_true",
        help="ignore cached embeddings and re-embed everything",
    )
    args = parser.parse_args()

    timestamp = datetime.now().isoformat(timespec="seconds")
    print("=== Intent reference similarity ===")
    print(f"  Timestamp : {timestamp}")
    print(f"  Input     : {args.input}")

    valid, invalid = load_reference_set(args.input)
    print(f"  VALID     : {len(valid)}")
    print(f"  INVALID   : {len(invalid)}")

    vectors, model, dim = get_embeddings(
        valid + invalid, use_cache=not args.refresh_cache
    )
    print(f"  Model     : {model} ({dim}-d)")

    pairs = compute_pairwise_similarities(valid, invalid, vectors)
    expected = {
        "V-V": len(valid) * (len(valid) - 1) // 2,
        "V-I": len(valid) * len(invalid),
        "I-I": len(invalid) * (len(invalid) - 1) // 2,
    }
    summaries = {pt: summarize_similarities(pairs, pt) for pt in ("V-V", "V-I", "I-I")}
    for pt in ("V-V", "V-I", "I-I"):
        actual = summaries[pt].count
        flag = "" if actual == expected[pt] else "  <-- MISMATCH"
        print(f"  {pt} pairs : expected {expected[pt]}, actual {actual}{flag}")
    mismatched = [pt for pt in expected if summaries[pt].count != expected[pt]]
    if mismatched:
        raise AnalysisError(f"Pair-count mismatch for {', '.join(mismatched)} — aborting.")

    top_pairs = get_top_cross_class_pairs(pairs, args.top_n)
    report = build_report(
        input_path=args.input,
        model=model,
        dim=dim,
        n_valid=len(valid),
        n_invalid=len(invalid),
        summaries=summaries,
        top_pairs=top_pairs,
        timestamp=timestamp,
    )
    save_results(report=report, pairs=pairs)

    print()
    print(report)
    print(f"Report saved : {REPORT_PATH}")
    print(f"Pairs saved  : {PAIRS_CSV_PATH}")
    print(f"Cache        : {CACHE_PATH}")
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except AnalysisError as err:
        print(f"\nERROR: {err}", file=sys.stderr)
        sys.exit(1)
