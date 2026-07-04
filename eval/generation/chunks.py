"""Load real chunks, compute gold-ids, and draw stratified samples.

Chunks are NOT stored in the processed pkl (that holds page-level cleaned docs).
The embed-ready chunks that gold-ids address are produced on the fly by
`SOURCES[domain].chunk(docs)` — exactly what `scripts/build_vectorstore.py` feeds
the embedder. Replicating that here guarantees our gold-ids match what the
retriever returns at eval time.
"""

from __future__ import annotations

import pickle
import random
import sys
from collections import defaultdict
from dataclasses import dataclass
from typing import Dict, List, Optional

from langchain_core.documents import Document

from ragtrial.capabilities.registry import CAPABILITIES
from ragtrial.sources import SOURCES

_CHUNK_CACHE: Dict[str, List[Document]] = {}


def _warn(msg: str) -> None:
    print(f"  [warn] {msg}", file=sys.stderr, flush=True)


def load_domain_chunks(domain: str) -> List[Document]:
    """`pickle.load(processed_pkl)` -> `src.chunk(...)`. Cached per process."""
    if domain in _CHUNK_CACHE:
        return _CHUNK_CACHE[domain]
    src = SOURCES[domain]
    with open(src.processed_pkl, "rb") as f:
        docs = pickle.load(f)
    chunks = src.chunk(docs)
    _CHUNK_CACHE[domain] = chunks
    return chunks


def gold_id_of(domain: str, chunk: Document) -> Optional[str]:
    """Prefixed gold-id via the owning capability (e.g. 'sosial:id:foo#pasal:2')."""
    return CAPABILITIES[domain].gold_id(chunk)


def valid_gold_ids(domain: str) -> set[str]:
    """Set of all prefixed gold-ids for a domain — the existence oracle for verify."""
    out: set[str] = set()
    for c in load_domain_chunks(domain):
        gid = gold_id_of(domain, c)
        if gid:
            out.add(gid)
    return out


def all_valid_gold_ids() -> set[str]:
    out: set[str] = set()
    for d in CAPABILITIES:
        out |= valid_gold_ids(d)
    return out


@dataclass
class SampleUnit:
    """A chunk chosen to ground a single-chunk question."""

    domain: str
    chunk: Document
    gold_id: str            # prefixed
    target_query_type: str
    stratum_key: str


# ---- stratification ----
def _stratum_key(domain: str, chunk: Document) -> str:
    m = chunk.metadata or {}
    if domain in ("sosial", "dukcapil"):
        return str(m.get("chunk_type", "?"))
    if domain == "perizinan":
        return str(m.get("kategori", "?"))
    if domain == "opd":
        return str(m.get("tipe", "?"))
    return "?"


# Minimum chunks to force-include from specific strata (sosial diversity).
_MINIMUMS: Dict[str, Dict[str, int]] = {
    "sosial": {"penjelasan_pasal": 8, "preamble": 6, "penjelasan_preamble": 2, "narrative": 6},
}

# Per-domain target query-type mix (single-chunk). Fractions, normalized to n.
_QTYPE_MIX: Dict[str, List[tuple]] = {
    "sosial": [("lexical_exact", 0.25), ("paraphrase", 0.25), ("semantic", 0.20),
               ("analytical", 0.15), ("negation_edge", 0.15)],
    "dukcapil": [("lexical_exact", 0.30), ("paraphrase", 0.30), ("semantic", 0.25),
                 ("analytical", 0.15)],
    "perizinan": [("lexical_exact", 0.30), ("paraphrase", 0.30), ("semantic", 0.20),
                  ("analytical", 0.20)],
    "opd": [("lexical_exact", 0.45), ("paraphrase", 0.35), ("semantic", 0.20)],
}


def _qtype_plan(domain: str, n: int) -> List[str]:
    mix = _QTYPE_MIX.get(domain, [("lexical_exact", 0.5), ("paraphrase", 0.5)])
    plan: List[str] = []
    for qt, frac in mix:
        plan += [qt] * round(n * frac)
    # pad / trim to exactly n
    filler = mix[0][0]
    while len(plan) < n:
        plan.append(filler)
    return plan[:n]


def _is_negation_chunk(chunk: Document) -> bool:
    m = chunk.metadata or {}
    if m.get("status") == "Tidak Berlaku":
        return True
    txt = (chunk.page_content or "").lower()
    return any(w in txt for w in ("dilarang", "larangan", "sanksi", "tidak boleh", "dikecualikan"))


def stratified_sample(domain: str, n: int, seed: int = 42) -> List[SampleUnit]:
    """Pick `n` distinct chunks (by gold-id) spread across strata, with query-types.

    Sampling is without replacement and degrades with a warning (not a crash) when
    a domain has fewer usable chunks than requested.
    """
    rng = random.Random(f"{seed}:{domain}")
    chunks = [c for c in load_domain_chunks(domain) if gold_id_of(domain, c)]
    rng.shuffle(chunks)

    by_stratum: Dict[str, List[Document]] = defaultdict(list)
    for c in chunks:
        by_stratum[_stratum_key(domain, c)].append(c)

    picked: List[Document] = []
    used: set[str] = set()

    def take(chunk: Document) -> bool:
        gid = gold_id_of(domain, chunk)
        if not gid or gid in used:
            return False
        used.add(gid)
        picked.append(chunk)
        return True

    # 1) force minimums from named strata
    for stratum, mn in _MINIMUMS.get(domain, {}).items():
        cnt = 0
        for c in by_stratum.get(stratum, []):
            if cnt >= mn or len(picked) >= n:
                break
            if take(c):
                cnt += 1

    # 2) round-robin the rest across strata for diversity
    keys = list(by_stratum.keys())
    rng.shuffle(keys)
    cursors = {k: 0 for k in keys}
    while len(picked) < n:
        progressed = False
        for k in keys:
            if len(picked) >= n:
                break
            lst = by_stratum[k]
            while cursors[k] < len(lst):
                c = lst[cursors[k]]
                cursors[k] += 1
                if take(c):
                    progressed = True
                    break
        if not progressed:
            _warn(f"{domain}: only {len(picked)}/{n} unique chunks available")
            break

    return _assign_query_types(domain, picked, rng)


def _assign_query_types(domain: str, picked: List[Document], rng: random.Random) -> List[SampleUnit]:
    n = len(picked)
    plan = _qtype_plan(domain, n)
    assigned: List[Optional[str]] = [None] * n
    free = list(range(n))

    # Put negation_edge slots onto chunks that actually express a limit/sanction.
    neg = plan.count("negation_edge")
    if neg:
        pref = [i for i in free if _is_negation_chunk(picked[i])]
        rng.shuffle(pref)
        for i in pref[:neg]:
            assigned[i] = "negation_edge"
            free.remove(i)
        placed = min(neg, len(pref))
        plan = [q for q in plan if q != "negation_edge"] + ["negation_edge"] * (neg - placed)

    rng.shuffle(free)
    for i, q in zip(free, plan):
        assigned[i] = q

    return [
        SampleUnit(
            domain=domain,
            chunk=picked[i],
            gold_id=gold_id_of(domain, picked[i]),  # type: ignore[arg-type]
            target_query_type=assigned[i] or "paraphrase",
            stratum_key=_stratum_key(domain, picked[i]),
        )
        for i in range(n)
    ]
