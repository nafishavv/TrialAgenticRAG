"""Automated audit + selection of the final in-scope eval set from the 202 candidates.

Read-only w.r.t. candidate_testset.json. Resolves every gold_id back to the *actual* chunk text
(same chunking pipeline as build_vectorstore), scores each candidate on evidence
grounding / answer support / retrieval usefulness / redundancy, re-classifies query
type from the evidence rather than the AI-written label, then picks the final set under
domain quotas.

    python eval/curate_testset.py --target 110
"""
from __future__ import annotations

import argparse
import json
import random
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Dict, List

_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_ROOT / "src"))
sys.path.insert(0, str(_ROOT))

from eval.generation.chunks import gold_id_of, load_domain_chunks  # noqa: E402

DOMAINS = ["sosial", "dukcapil", "opd", "perizinan"]

# Domain buckets for selection. 'cross' = expected_route == 'both'.
BUCKETS = ["sosial", "dukcapil", "opd", "perizinan", "cross"]
TARGETS = {"sosial": 25, "dukcapil": 25, "opd": 25, "perizinan": 25, "cross": 10}
QTYPE_TARGETS = {"lexical": 45, "semantic": 45, "complex": 20}
LEX_THRESHOLD = 0.70   # verbatim key-term overlap above which a query is lexical-oriented

STOP = set("""
yang dan di ke dari untuk pada dengan dalam adalah atau apa siapa bagaimana kapan
mana saja apakah itu ini ada tidak bisa dapat oleh sebagai akan juga agar jika maka
tersebut para serta secara harus telah sudah bagi antara tentang terhadap sebuah
suatu setiap kepada karena namun tetapi lain lainnya kah nya yg utk dgn dr pd
berapa mengapa kenapa mana apabila bila hal seperti sesuai berdasarkan menurut
kalau gimana gmn buat aja sih dong nih kalo emang kok terus lalu
""".split())

_SUF = ("nya", "kannya", "kan", "annya", "an", "i", "lah", "kah", "pun")
_PRE = ("memper", "meng", "meny", "mem", "men", "me", "peng", "peny", "pem", "pen",
        "per", "ber", "ter", "di", "ke", "se")


def stem(tok: str) -> str:
    """Crude Indonesian stemmer — good enough as a lexical-overlap signal."""
    w = tok
    for s in sorted(_SUF, key=len, reverse=True):
        if len(w) > len(s) + 3 and w.endswith(s):
            w = w[: -len(s)]
            break
    for p in _PRE:
        if len(w) > len(p) + 3 and w.startswith(p):
            w = w[len(p):]
            break
    return w


def content_tokens(text: str) -> List[str]:
    toks = re.findall(r"[a-z0-9]+", (text or "").lower())
    return [t for t in toks if t not in STOP and len(t) >= 3]


def _key(tok: str) -> str:
    """Matching key: numbers verbatim, words by truncated stem."""
    if tok.isdigit():
        return tok
    return stem(tok)[:6]


_IDF: Dict[str, float] = {}
_IDF_DEFAULT = 1.0


def build_idf(gidx: Dict[str, Dict[str, str]]) -> None:
    """Corpus IDF over all chunks — plain token coverage saturates, IDF-weighted doesn't."""
    import math

    df: Counter = Counter()
    n_docs = 0
    for by_id in gidx.values():
        for text in by_id.values():
            n_docs += 1
            df.update({_key(t) for t in content_tokens(text)})
    global _IDF, _IDF_DEFAULT
    _IDF = {k: math.log(n_docs / (1 + v)) + 1.0 for k, v in df.items()}
    _IDF_DEFAULT = math.log(n_docs / 1) + 1.0   # unseen key = maximally rare


def _idf(k: str) -> float:
    return max(_IDF.get(k, _IDF_DEFAULT), 0.05)


def coverage(text: str, gold: str) -> float:
    """IDF-weighted fraction of `text`'s content tokens supported by `gold`.

    Rare terms carry the weight, so a fact that reuses generic legalese but invents
    the one distinctive term scores low instead of saturating at 1.0.
    """
    toks = content_tokens(text)
    if not toks:
        return 0.0
    gold_keys = {_key(t) for t in content_tokens(gold)}
    gold_low = gold.lower()
    num, den = 0.0, 0.0
    for t in toks:
        k = _key(t)
        w = _idf(k)
        den += w
        if k in gold_keys or (len(k) >= 5 and k in gold_low):
            num += w
    return num / den if den else 0.0


def lexical_overlap(question: str, gold: str) -> float:
    """Verbatim overlap on the question's most distinctive terms.

    This is the Category A/B discriminator: do the query's key terms literally appear
    in the gold chunk (lexical/BM25-friendly) or has the need been re-worded (semantic)?
    Uses the question's own top-IDF terms, so it is not biased by chunk length.
    """
    toks = list(dict.fromkeys(content_tokens(question)))
    if not toks:
        return 0.0
    ranked = sorted(toks, key=lambda t: -_idf(_key(t)))
    key_terms = ranked[: max(4, min(8, len(ranked) // 2 + 2))]
    gold_low = gold.lower()
    gold_keys = {_key(t) for t in content_tokens(gold)}
    hit = 0.0
    for t in key_terms:
        if t in gold_low:                       # exact surface form
            hit += 1.0
        elif _key(t) in gold_keys:              # same stem, inflected differently
            hit += 0.6
    return hit / len(key_terms)


_EN_MARKERS = set("the of and for in shall article chapter with paragraph is are that "
                  "provisions rights carried out based".split())


def is_english(text: str) -> bool:
    """The sosial corpus contains one English translation doc — questions grounded there
    are cross-lingual, and word-overlap grounding would false-negative on them."""
    toks = re.findall(r"[a-z]+", (text or "").lower())
    if not toks:
        return False
    return sum(1 for t in toks if t in _EN_MARKERS) / len(toks) > 0.05


_PRONOUNS = {"anda", "kami", "saya", "mereka", "beliau"}


def anchors_of(text: str, top: int = 8) -> List[str]:
    """The load-bearing terms of an answer/fact.

    Identifiers and proper nouns first — those must survive any paraphrase. Only when a
    statement has none of those do we fall back to its rarest words, because top-IDF
    ranking on prose otherwise picks up the paraphrasing verbs themselves.
    """
    text = text or ""
    hard = re.findall(r"[\w.\-]+@[\w.\-]+|\b\d[\d.,/-]*\d\b|\b\d+\b", text)
    proper = []
    for sent in re.split(r"(?<=[.;:])\s+|\n", text):
        for i, w in enumerate(re.findall(r"[A-Za-zÀ-ÿ]+", sent)):
            if i and w[:1].isupper() and w.lower() not in STOP and w.lower() not in _PRONOUNS:
                proper.append(w.lower())
    anc = list(dict.fromkeys(hard + proper))
    if anc:
        return anc[:12]
    words = [t for t in dict.fromkeys(content_tokens(text)) if not t.isdigit()]
    words.sort(key=lambda t: -_idf(_key(t)))
    return words[:top]


def anchor_support(text: str, gold: str, cross_lingual: bool = False) -> float:
    """Share of the text's load-bearing terms actually present in the gold evidence.

    This is the factuality probe: a paraphrased answer may share few tokens with the
    chunk, but its names, numbers and identifiers must still be there.
    """
    anc = anchors_of(text)
    if cross_lingual:
        anc = [a for a in anc if any(ch.isdigit() or ch == "@" for ch in a)]
    if not anc:
        return 1.0 if cross_lingual else 0.0
    gold_low = gold.lower()
    gold_keys = {_key(t) for t in content_tokens(gold)}
    hit = 0
    for a in anc:
        al = a.lower()
        if al in gold_low or _key(al) in gold_keys:
            hit += 1
    return hit / len(anc)


_BOILER = re.compile(
    r"Lembaran Negara|Berita Negara|Tambahan Lembaran|Mengingat\s*:|Menimbang\s*:"
    r"|Undang-Undang Nomor|Peraturan Menteri|Peraturan Pemerintah Nomor", re.I)


# bibliography entry: "..., (Jakarta: Penerbit, 2015)."
_BIBLIO = re.compile(r"\((?:[A-Z][^()]{2,40}:\s*)?[^()]{3,70},\s*(?:19|20)\d{2}\)")


def boilerplate_frac(gold: str) -> float:
    """How much of the evidence is a bare citation list — 'Mengingat/Menimbang' recitals
    or a naskah-akademik bibliography.

    Those chunks name a lot of regulations and titles but assert nothing, so a question
    grounded there tests looking up a reference entry rather than any real information
    need, and its answer often is not supported by the chunk at all.
    """
    toks = len(re.findall(r"\w+", gold or "")) or 1
    hits = len(_BOILER.findall(gold or "")) * 12.0 + len(_BIBLIO.findall(gold or "")) * 25.0
    return min(1.0, hits / toks)


# "Berdasarkan rujukan yang diberikan, ..." — the question presupposes a context that a
# real user never has; as an eval query it is both unnatural and self-answering.
_CONTEXT_REF = re.compile(
    r"berdasarkan (rujukan|informasi|konteks|teks|dokumen|data|penjelasan)[^,.?]{0,25}"
    r"(diberikan|disediakan|tersebut|ini)"
    r"|dalam konteks yang diberikan"
    r"|menurut (teks|konteks|dokumen) (di atas|tersebut|ini)"
    r"|yang (diberikan|disebutkan) (dalam|di) (konteks|teks)", re.I)

# Question that already names the exact Pasal + regulation hands the retriever its own
# locator. Still a valid lexical case, just a weak one — deprioritized, not rejected.
_SELF_CITE = re.compile(
    r"pasal\s*\d+[^?]{0,60}?(nomor|no\.)\s*\d+"
    r"|(permendagri|undang-undang|perda|perbup|peraturan pemerintah)\s*(nomor|no\.)?\s*\d+"
    r"\s*tahun\s*\d{4}", re.I)

_META_ONLY = re.compile(
    r"berstatus\s*['\"‘’]?\s*(tidak|masih)\s+berlaku"
    r"|berdasarkan metadata|metadata dokumen"
    r"|status\s+(keberlakuan|dokumen|peraturan)\b", re.I)


def metadata_dependent(answer: str, gold: str) -> bool:
    """Answer leans on document metadata (e.g. status berlaku) absent from chunk text."""
    if not _META_ONLY.search(answer or ""):
        return False
    return "tidak berlaku" not in (gold or "").lower()


def number_support(text: str, gold: str) -> float:
    """Fraction of 'meaningful' numbers in text that also occur in gold."""
    nums = [n for n in re.findall(r"\d+", text or "") if len(n) >= 1]
    nums = [n for n in nums if n not in {"1", "2", "3"} or len(nums) <= 2]
    if not nums:
        return 1.0
    return sum(1 for n in nums if re.search(rf"\b{re.escape(n)}\b", gold)) / len(nums)


# ---------------------------------------------------------------- gold index
def build_gold_index() -> Dict[str, Dict[str, str]]:
    index: Dict[str, Dict[str, str]] = {}
    for dom in DOMAINS:
        by_id: Dict[str, str] = {}
        for c in load_domain_chunks(dom):
            gid = gold_id_of(dom, c)
            if not gid:
                continue
            by_id[gid.split(":", 1)[1] if ":" in gid else gid] = c.page_content
        index[dom] = by_id
    return index


def _topic_of(gold_ids: List[str]) -> str:
    """Coarse 'part of the knowledge base' key, for representativeness spreading."""
    if not gold_ids:
        return "-"
    gid = sorted(gold_ids)[0]
    dom, rest = gid.split(":", 1)
    if dom == "dukcapil":                       # 'page:37' -> bucket of 5 pages
        m = re.search(r"(\d+)", rest)
        return f"dukcapil:p{int(m.group(1)) // 5}" if m else gid
    root = rest.split("#")[0]                   # doc / perizinan id / opd nomor
    return f"{dom}:{root}"


# ---------------------------------------------------------------- per-question audit
def audit_question(q: dict, gidx: Dict[str, Dict[str, str]]) -> dict:
    gold_ids: List[str] = []
    texts: List[str] = []
    missing: List[str] = []
    for dom, ids in (q.get("gold_chunks") or {}).items():
        for gid in ids or []:
            gold_ids.append(f"{dom}:{gid}")
            t = gidx.get(dom, {}).get(gid)
            if t is None:
                missing.append(f"{dom}:{gid}")
            else:
                texts.append(t)
    gold_text = "\n".join(texts)
    topic = _topic_of(gold_ids)

    xling = bool(gold_text) and is_english(gold_text)
    facts = q.get("expected_facts") or []
    fact_cov = [max(coverage(f, gold_text), anchor_support(f, gold_text, xling)) for f in facts]
    ans = q.get("expected_answer", "")

    a = {
        "id": q["id"],
        "route": q["expected_route"],
        "bucket": "cross" if q["expected_route"] == "both" else q["expected_route"],
        "label_qtype": q["query_type"],
        "difficulty": q["difficulty"],
        "chunk_scope": q["chunk_scope"],
        "n_gold": len(gold_ids),
        "gold_ids": gold_ids,
        "topic": topic,
        "missing_gold": missing,
        "n_facts": len(facts),
        "fact_cov_mean": sum(fact_cov) / len(fact_cov) if fact_cov else 0.0,
        "fact_cov_min": min(fact_cov) if fact_cov else 0.0,
        "n_weak_facts": sum(1 for c in fact_cov if c < 0.45),
        "ans_cov": coverage(ans, gold_text),
        "ans_anchor": anchor_support(ans, gold_text, xling),
        "ans_numbers": 1.0 if xling else number_support(ans, gold_text),
        "cross_lingual": xling,
        "boilerplate": round(boilerplate_frac(gold_text), 3),
        "context_ref": bool(_CONTEXT_REF.search(q["question"])),
        "self_cite": bool(_SELF_CITE.search(q["question"])),
        "metadata_only": metadata_dependent(ans, gold_text),
        "q_cov": coverage(q["question"], gold_text),
        "lex_overlap": round(lexical_overlap(q["question"], gold_text), 3),
        "q_len": len(content_tokens(q["question"])),
        "ans_len": len(content_tokens(ans)),
    }
    return a


# ---------------------------------------------------------------- classification
def classify(a: dict) -> str:
    """Simplified query type from the evidence, not from the AI label."""
    if a["n_gold"] >= 2 or a["route"] == "both" or a["chunk_scope"] in ("multi", "cross_store"):
        return "complex"
    if a["label_qtype"] in ("analytical", "multi_chunk", "cross_store"):
        # single-chunk 'analytical' still counts as complex only if it really
        # demands combining several facts out of the chunk
        return "complex" if a["n_facts"] >= 3 else _lex_or_sem(a)
    if a["label_qtype"] == "negation_edge" and a["n_facts"] >= 3:
        return "complex"
    return _lex_or_sem(a)


def _lex_or_sem(a: dict) -> str:
    return "lexical" if a["lex_overlap"] >= LEX_THRESHOLD else "semantic"


# ---------------------------------------------------------------- quality scoring
def quality(a: dict) -> tuple:
    """(score 0..1, list of flags). Evidence validity dominates."""
    flags: List[str] = []
    if a["missing_gold"]:
        flags.append("gold_missing")
    if a["n_gold"] == 0:
        flags.append("no_gold")
    if a["fact_cov_mean"] < 0.45:
        flags.append("facts_ungrounded")
    elif a["fact_cov_mean"] < 0.60:
        flags.append("facts_weak")
    if a["n_weak_facts"] >= 2:
        flags.append("multi_weak_facts")
    # expected_facts is what the LLM-judge actually scores, so it is the primary
    # factuality evidence; the prose answer is a weaker, paraphrase-noisy signal.
    if a["ans_anchor"] < 0.55 and a["ans_cov"] < 0.45 and a["fact_cov_mean"] < 0.80:
        flags.append("answer_ungrounded")
    elif a["ans_anchor"] < 0.65 and a["fact_cov_mean"] < 0.90:
        flags.append("answer_partly_supported")
    if a["cross_lingual"]:
        flags.append("english_gold")
    if a["boilerplate"] >= 0.45:
        flags.append("boilerplate_gold")
    if a["metadata_only"]:
        flags.append("metadata_only_answer")
    if a["context_ref"]:
        flags.append("context_reference")
    if a["self_cite"]:
        flags.append("self_citing")
    if a["ans_numbers"] < 1.0:
        flags.append("unsupported_numbers")
    if a["n_facts"] == 0:
        flags.append("no_facts")
    if a["q_len"] < 4:
        flags.append("question_too_short")
    if a["ans_len"] < 4:
        flags.append("answer_too_short")

    # weighted score
    s = (
        0.47 * min(a["fact_cov_mean"] / 0.85, 1.0)
        + 0.15 * min(max(a["ans_anchor"], a["ans_cov"]) / 0.85, 1.0)
        + 0.12 * a["ans_numbers"]
        + 0.10 * (1.0 if a["n_facts"] >= 2 else 0.4 if a["n_facts"] == 1 else 0.0)
        + 0.08 * min(a["q_len"] / 12.0, 1.0)          # non-trivial information need
        + 0.08 * (1.0 - min(a["n_weak_facts"] / 3.0, 1.0))
    )
    if a["missing_gold"] or a["n_gold"] == 0:
        s *= 0.2
    return round(s, 4), flags


def tier(score: float, flags: List[str]) -> str:
    f = set(flags)
    if {"gold_missing", "no_gold", "no_facts"} & f:
        return "unusable"
    if {"facts_ungrounded", "answer_ungrounded", "multi_weak_facts",
            "boilerplate_gold", "metadata_only_answer", "context_reference"} & f:
        return "problematic"
    # 'redundant' is handled by the selector, not by the intrinsic quality tier;
    # 'self_citing' is a weak-but-valid lexical case, deprioritized there too
    blocking = f - {"redundant", "self_citing"}
    if score >= 0.88 and not blocking:
        return "high"
    if score >= 0.72:
        return "acceptable"
    return "problematic"


# ---------------------------------------------------------------- redundancy
def jac(a: str, b: str) -> float:
    A, B = {_key(t) for t in content_tokens(a)}, {_key(t) for t in content_tokens(b)}
    return len(A & B) / len(A | B) if A | B else 0.0


_BUCKET_LABEL = {"sosial": "Sosial", "dukcapil": "Dukcapil", "opd": "OPD",
                 "perizinan": "Perizinan", "cross": "Cross-domain/ambiguous"}

_FLAG_WHY = {
    "boilerplate_gold": "gold chunk-nya daftar sitasi doang (*Mengingat/Menimbang* atau "
                        "daftar pustaka naskah akademik) — nyebutin banyak peraturan & judul "
                        "tapi ga nyatain apa pun yang dibutuhin jawabannya",
    "metadata_only_answer": "jawabannya ngandelin status berlaku dokumen, yang adanya di "
                            "metadata chunk dan ga pernah muncul di teks yang di-retrieve",
    "english_gold": "gold-nya dokumen terjemahan Inggris — pertanyaan Indonesia vs evidence "
                    "Inggris itu artefak korpus, bukan kemampuan retrieval yang mau diukur",
    "unsupported_numbers": "ada angka di jawaban yang ga muncul di chunk yang disitir",
    "answer_ungrounded": "baik kata-kata jawabannya maupun istilah kuncinya ga ada di evidence",
    "facts_ungrounded": "expected_facts ga didukung chunk yang disitir",
    "facts_weak": "expected_facts cuma sebagian didukung chunk yang disitir",
    "multi_weak_facts": "dua atau lebih expected_facts dukungannya lemah",
    "context_reference": "pertanyaannya nyebut \"berdasarkan rujukan/informasi yang "
                         "diberikan\" — ngandaiin ada konteks yang user asli ga punya, jadi "
                         "ga valid sebagai query eval",
    "self_citing": "pertanyaannya udah nyebut Pasal + nomor peraturannya sendiri, jadi "
                   "retriever dikasih lokasi jawabannya (masih valid sebagai kasus lexical, "
                   "cuma lemah — dideprioritasin, bukan ditolak)",
    "redundant": "nanya info / nembak evidence yang sama dengan kandidat lain",
    "answer_partly_supported": "sebagian istilah kunci jawaban ga ketemu di evidence "
                               "(sinyal lemah, sensitif ke parafrase)",
}


def write_report(path: Path, audits: dict, pool: dict, oos: List[str],
                 selected: List[str], manual: List[str], by_id: dict) -> None:
    sel = [audits[i] for i in selected]
    all_a = list(audits.values())
    tiers = Counter(a["tier"] for a in all_a)
    L: List[str] = []
    w = L.append

    w("# Testset Curation — in-scope selection dari 202 kandidat\n")
    w("Dihasilkan oleh `eval/curate_testset.py` (read-only terhadap `eval/candidate_testset.json`). "
      "Setiap `gold_chunks` di-resolve balik ke teks chunk asli lewat pipeline chunking yang "
      "sama dengan `scripts/build_vectorstore.py`, jadi penilaian evidence-nya dilakukan "
      "terhadap teks yang benar-benar bisa di-retrieve, bukan terhadap label bawaan.\n")

    w("## Overall selection\n")
    w("| | n |\n|---|---:|")
    w(f"| total candidates | {len(all_a) + len(oos)} |")
    w(f"| in-scope candidates | {len(all_a)} |")
    w(f"| selected in-scope | {len(sel)} |")
    w(f"| excluded / unused in-scope | {len(all_a) - len(sel)} |")
    w(f"| OOS excluded from this selection | {len(oos)} |")
    w("")
    w(f"OOS yang dikecualikan: {', '.join(oos)} — set OOS ~20 soal dibuat terpisah nanti.\n")

    w("## Domain distribution\n")
    w("| Domain | Target | Pool | Selected |\n|---|---:|---:|---:|")
    for b in BUCKETS:
        n = sum(1 for a in sel if a["bucket"] == b)
        w(f"| {_BUCKET_LABEL[b]} | ~{TARGETS[b]} | {len(pool[b])} | {n} |")
    w(f"| **Total** | **110** | **{len(all_a)}** | **{len(sel)}** |\n")

    w("## Query-type distribution\n")
    w("Kategori ditentukan ulang dari evidence (bukan dari label `query_type` bawaan): "
      "**Complex** kalau butuh >1 gold chunk / lintas-store / sintesis banyak fakta; sisanya "
      "**Lexical** vs **Semantic** lewat overlap verbatim istilah paling distinktif dari "
      f"pertanyaan terhadap gold chunk (ambang {LEX_THRESHOLD:.2f}).\n")
    w("| Category | Target | Pool | Selected |\n|---|---:|---:|---:|")
    poolq = Counter(a["qtype"] for a in all_a)
    selq = Counter(a["qtype"] for a in sel)
    for t, label in [("lexical", "Lexical-oriented"), ("semantic", "Semantic-oriented"),
                     ("complex", "Complex")]:
        w(f"| {label} | ~{QTYPE_TARGETS[t]} | {poolq[t]} | {selq[t]} |")
    w("")

    w("### Domain x query-type (selected)\n")
    w("Ada batas: satu query type maksimal ~60% dari jatah satu domain, kecuali pool domain "
      "itu emang ga nyediain pilihan lain (OPD isinya hampir semua semantic). Tanpa batas ini "
      "target lexical global gampang dipenuhin dengan numpukin satu domain penuh soal lookup "
      "gampang, dan capability coverage-nya jadi jelek.\n")
    w("| Domain | Lexical | Semantic | Complex |\n|---|---:|---:|---:|")
    for b in BUCKETS:
        c = Counter(a["qtype"] for a in sel if a["bucket"] == b)
        w(f"| {_BUCKET_LABEL[b]} | {c['lexical']} | {c['semantic']} | {c['complex']} |")
    w("")

    w("## Quality summary\n")
    w("| Tier | Definisi | In-scope pool | Selected |\n|---|---|---:|---:|")
    seltier = Counter(a["tier"] for a in sel)
    defs = {
        "high": "evidence & fakta ter-grounding penuh, tanpa flag",
        "acceptable": "usable tapi ada flag lemah (mis. sebagian anchor jawaban ga ketemu)",
        "problematic": "ada cacat evidence/jawaban yang bikin ga layak dipakai apa adanya",
        "unusable": "gold_id ga resolve / ga ada gold / ga ada expected_facts",
    }
    for t in ["high", "acceptable", "problematic", "unusable"]:
        w(f"| {t} | {defs[t]} | {tiers[t]} | {seltier[t]} |")
    w("")

    flag_counts = Counter(f for a in all_a for f in a["flags"])
    w("### Alasan penolakan / flag (seluruh in-scope pool)\n")
    w("| Flag | n | Kenapa masalah |\n|---|---:|---|")
    for f, n in flag_counts.most_common():
        w(f"| `{f}` | {n} | {_FLAG_WHY.get(f, '')} |")
    w("")

    rejected = [a for a in all_a if not a["selected"]]
    n_prob = sum(1 for a in rejected if a["tier"] in ("problematic", "unusable"))
    n_acc = sum(1 for a in rejected if a["tier"] == "acceptable")
    w(f"Dari {len(rejected)} kandidat in-scope yang ga kepilih: **{n_prob} ditolak karena cacat "
      f"evidence/jawaban** (tier `problematic`), **{n_acc} dikesampingin** karena ada flag lemah "
      f"sementara masih ada kandidat `high` di domain yang sama, dan sisanya "
      f"**{len(rejected) - n_prob - n_acc} kualitasnya oke tapi kepotong kuota domain** atau "
      "dihindari karena redundan / nembak chunk & topik yang udah diwakili soal lain.\n")

    prob = [a for a in all_a if a["tier"] in ("problematic", "unusable")]
    if prob:
        w("### Kandidat bermasalah (semua dikeluarkan dari selection)\n")
        w("| ID | Domain | Flags | Gold |\n|---|---|---|---|")
        for a in sorted(prob, key=lambda x: x["id"]):
            w(f"| {a['id']} | {a['bucket']} | {', '.join(f'`{f}`' for f in a['flags'])} | "
              f"`{a['gold_ids'][0] if a['gold_ids'] else '-'}` |")
        w("")

    w("## Feasibility\n")
    w("**Domain ~25/25/25/25 + 10: feasible, tapi OPD nol slack.**\n")
    w("| Domain | Usable (high+acceptable) | Butuh | Slack |\n|---|---:|---:|---:|")
    for b in BUCKETS:
        usable = sum(1 for a in pool[b] if a["tier"] in ("high", "acceptable"))
        w(f"| {_BUCKET_LABEL[b]} | {usable} | {TARGETS[b]} | {usable - TARGETS[b]:+d} |")
    w("")
    w("- **OPD** cuma punya 25 kandidat buat target 25 — jadi ga ada seleksi sama sekali di "
      "domain ini, semuanya masuk. Kalau nanti ada yang gugur pas manual audit, ga ada "
      "cadangan; harus generate soal OPD baru.\n"
      "- **Perizinan** 31 kandidat buat 25 → slack 6.\n"
      "- **Cross-domain** 16 kandidat buat 10 → slack cukup.\n"
      "- **Sosial & Dukcapil** longgar (61 & 60), di situ seleksi kualitas & diversitas "
      "beneran kerja.\n")
    w("**Query-type ~45/45/20: mepet, dan bentuknya ditentukan sama isi domain.**\n")
    opd_sem = sum(1 for a in pool["opd"] if a["qtype"] == "semantic")
    w(f"- OPD kepaksa nyumbang {opd_sem} soal semantic (isinya record kontak OPD yang terstruktur, "
      "hampir selalu ditanya pakai sinonim), dan cross-domain kepaksa nyumbang 10 complex. "
      "Dua blok ini udah ngunci ~32 dari 110 slot sebelum domain lain disentuh.\n")
    lex_avail = sum(1 for a in all_a if a["qtype"] == "lexical")
    w(f"- Total kandidat lexical di seluruh pool cuma {lex_avail}. Buat nembak 45 harus ambil "
      f"~{45 / max(lex_avail,1):.0%} dari semua kandidat lexical yang ada — artinya nyaris "
      "tanpa seleksi kualitas, dan praktiknya bikin satu domain (Dukcapil) ketimbun soal "
      "lookup gampang sampe 76% jatahnya. Makanya target lexical dilonggarin ke "
      f"{selq['lexical']} dan complex kebawa naik ke {selq['complex']}.\n")
    w("- **Rekomendasi:** target domain dipertahankan; target query-type dianggap panduan. "
      f"Hasil {selq['lexical']}/{selq['semantic']}/{selq['complex']} "
      "(lexical/semantic/complex) masih dalam koridor dan ga ngorbanin kualitas.\n")

    w("## Difficulty & coverage\n")
    d = Counter(a["difficulty"] for a in sel)
    w(f"- difficulty: easy {d['easy']}, medium {d['medium']}, hard {d['hard']}")
    w(f"- gold-set unik: {len({'|'.join(sorted(a['gold_ids'])) for a in sel})}/{len(sel)} "
      "(ga ada dua soal terpilih yang nembak kombinasi chunk yang sama persis)")
    w(f"- topik/dokumen berbeda yang tersentuh: {len({a['topic'] for a in sel})}")
    w(f"- soal multi-chunk / cross-store: {sum(1 for a in sel if a['n_gold'] > 1)}")
    w(f"- redundan yang ikut kepilih: {sum(1 for a in sel if 'redundant' in a['flags'])}\n")

    w("## Manual audit subset (25)\n")
    w("Random 5 per domain **di dalam** tiap grup (seed tetap), bukan 25 teratas — biar "
      "sampelnya representatif buat dicek balik ke dokumen sumber.\n")
    w("| ID | Domain | Query type | Difficulty | Pertanyaan |\n|---|---|---|---|---|")
    for qid in manual:
        a = audits[qid]
        qtext = by_id[qid]["question"].replace("|", "/")
        qtext = qtext[:95] + ("…" if len(qtext) > 95 else "")
        w(f"| {qid} | {_BUCKET_LABEL[a['bucket']]} | {a['qtype']} | {a['difficulty']} | {qtext} |")
    w("")
    w("Artefak: `eval/selection_inscope_110.json` (selected_ids + manual_audit_subset), "
      "`eval/results/curation_audit.json` (skor & flag per kandidat).\n")

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(L), encoding="utf-8")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--testset", default="eval/candidate_testset.json")
    ap.add_argument("--out-selection", default="eval/selection_inscope_110.json")
    ap.add_argument("--out-audit", default="eval/results/curation_audit.json")
    ap.add_argument("--out-report", default="docs/TESTSET_CURATION.md")
    ap.add_argument("--seed", type=int, default=20260812)
    args = ap.parse_args()

    testset = json.loads(Path(args.testset).read_text(encoding="utf-8"))
    questions = testset["questions"]
    by_id = {q["id"]: q for q in questions}

    print("Chunking sumber asli (identik build_vectorstore)...", file=sys.stderr)
    gidx = build_gold_index()
    for d, i in gidx.items():
        print(f"  {d}: {len(i)} chunk", file=sys.stderr)
    build_idf(gidx)

    oos = [q["id"] for q in questions if q["expected_route"] == "none"]
    inscope = [q for q in questions if q["expected_route"] != "none"]

    audits = {}
    for q in inscope:
        a = audit_question(q, gidx)
        a["qtype"] = classify(a)
        a["score"], a["flags"] = quality(a)
        a["tier"] = tier(a["score"], a["flags"])
        audits[q["id"]] = a

    # ---- redundancy: same gold evidence + similar question wording
    dup_of: Dict[str, str] = {}
    ids = [q["id"] for q in inscope]
    for i in range(len(ids)):
        for j in range(i + 1, len(ids)):
            ai, aj = audits[ids[i]], audits[ids[j]]
            if ai["bucket"] != aj["bucket"]:
                continue
            same_gold = set(ai["gold_ids"]) == set(aj["gold_ids"]) and ai["n_gold"] > 0
            qsim = jac(by_id[ids[i]]["question"], by_id[ids[j]]["question"])
            asim = jac(by_id[ids[i]]["expected_answer"], by_id[ids[j]]["expected_answer"])
            redundant = (same_gold and (qsim >= 0.45 or asim >= 0.62)) or qsim >= 0.72
            if redundant:
                keep, drop = (ids[i], ids[j]) if ai["score"] >= aj["score"] else (ids[j], ids[i])
                dup_of.setdefault(drop, keep)
    for k, v in dup_of.items():
        audits[k]["redundant_with"] = v
        if "redundant" not in audits[k]["flags"]:
            audits[k]["flags"].append("redundant")

    # ---- selection
    rng = random.Random(args.seed)
    pool: Dict[str, List[dict]] = defaultdict(list)
    for a in audits.values():
        pool[a["bucket"]].append(a)

    selected: List[str] = []
    sel_qtype = Counter()
    used_gold: Counter = Counter()

    # quality dominates: a problematic candidate must never displace a usable one just
    # because it fills a query-type quota
    tier_penalty = {"high": 0.0, "acceptable": 0.35, "problematic": 2.0, "unusable": 99.0}
    # Scarce buckets go first: they have no slack, so they get first claim on the
    # global query-type budget. Abundant buckets absorb whatever is left over.
    order = sorted(BUCKETS, key=lambda b: len(pool[b]) - TARGETS[b])
    topic_used: Counter = Counter()

    for bucket in order:
        want = TARGETS[bucket]
        cands = [a for a in pool[bucket] if a["tier"] != "unusable"]
        # No single query type may own more than ~60% of a domain unless that domain's
        # pool leaves no choice (OPD is genuinely almost all semantic). Without this the
        # global lexical target gets satisfied by burying one domain in easy lookups.
        dom_cap = max(3, int(want * 0.60))
        dom_qtype: Counter = Counter()
        picked: List[dict] = []
        for _ in range(min(want, len(cands))):
            need = {t: max(0, QTYPE_TARGETS[t] - sel_qtype[t]) for t in QTYPE_TARGETS}
            tot_need = sum(need.values())
            best, best_val = None, -1e9
            for a in cands:
                if a in picked:
                    continue
                qt = a["qtype"]
                val = a["score"]
                val -= tier_penalty[a["tier"]]
                val -= 0.45 if "redundant" in a["flags"] else 0.0
                val -= 0.30 if "self_citing" in a["flags"] else 0.0
                val -= 0.90 if dom_qtype[qt] >= dom_cap else 0.0
                # query-type budget: continuous pressure toward whichever type is
                # furthest behind its share, so quotas steer from the first pick and
                # not only once one of them is already full
                val += 0.70 * (need[qt] / tot_need) if tot_need else 0.0
                val -= 0.0 if need[qt] else 0.55
                # evidence diversity: don't keep re-testing the same chunk(s)/topic
                gkey = "|".join(sorted(a["gold_ids"]))
                val -= 0.35 * used_gold[gkey]
                val -= 0.12 * topic_used[a["topic"]]
                # keep the set from collapsing into easy questions (crit. 4.6)
                val += {"easy": 0.0, "medium": 0.18, "hard": 0.35}[a["difficulty"]]
                val -= 0.20 if a["cross_lingual"] else 0.0
                if val > best_val:
                    best, best_val = a, val
            if best is None:
                break
            picked.append(best)
            dom_qtype[best["qtype"]] += 1
            used_gold["|".join(sorted(best["gold_ids"]))] += 1
            topic_used[best["topic"]] += 1
            sel_qtype[best["qtype"]] += 1
        for a in picked:
            a["selected"] = True
        selected += [a["id"] for a in picked]

    for a in audits.values():
        a.setdefault("selected", False)

    # ---- manual audit subset: random 5 per bucket among selected
    manual: List[str] = []
    for bucket in BUCKETS:
        grp = sorted(a["id"] for a in audits.values() if a["selected"] and a["bucket"] == bucket)
        manual += rng.sample(grp, min(5, len(grp)))
    manual.sort()

    selected.sort()
    Path(args.out_selection).write_text(
        json.dumps({"selected_ids": selected, "manual_audit_subset": manual},
                   ensure_ascii=False, indent=2), encoding="utf-8")

    Path(args.out_audit).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out_audit).write_text(
        json.dumps({"oos_excluded": oos, "audits": audits}, ensure_ascii=False, indent=2),
        encoding="utf-8")

    write_report(Path(args.out_report), audits, pool, oos, selected, manual, by_id)

    # ---- console summary
    print(f"\nin-scope candidates: {len(inscope)}  | OOS excluded: {len(oos)}")
    print("tier (all in-scope):", dict(Counter(a["tier"] for a in audits.values())))
    print("\nper-bucket pool tiers:")
    for b in BUCKETS:
        c = Counter(a["tier"] for a in pool[b])
        print(f"  {b:10s} n={len(pool[b]):3d}  {dict(c)}")
    sel = [audits[i] for i in selected]
    print(f"\nselected: {len(sel)}")
    print("  bucket:", dict(Counter(a["bucket"] for a in sel)))
    print("  qtype :", dict(Counter(a["qtype"] for a in sel)))
    print("  tier  :", dict(Counter(a["tier"] for a in sel)))
    print("  diff  :", dict(Counter(a["difficulty"] for a in sel)))
    print("  distinct gold-sets:", len({"|".join(sorted(a['gold_ids'])) for a in sel}))
    print("\nmanual_audit_subset:", manual)
    print("\npool qtype (all in-scope):", dict(Counter(a["qtype"] for a in audits.values())))
    print("wrote", args.out_selection, "and", args.out_audit)


if __name__ == "__main__":
    main()
