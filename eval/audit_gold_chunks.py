"""Read-only manual-audit dump: soal x chunk asli (vector-store-identik).

Gak manggil LLM, gak nulis ke candidate_testset.json. Cuma baca chunk lewat pipeline
chunking yang sama persis dipakai buat build vectorstore (eval/generation/chunks.py),
terus tulis side-by-side ke satu file txt biar gampang di-scroll & dibandingin manual.

Usage: python eval/audit_gold_chunks.py [--domain sosial] [--out eval/results/audit_dump.txt]
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_ROOT / "src"))
sys.path.insert(0, str(_ROOT))

from eval.generation.chunks import gold_id_of, load_domain_chunks  # noqa: E402

DOMAINS = ["sosial", "dukcapil", "opd", "perizinan"]


def build_gold_index(domains: list[str]) -> dict[str, dict[str, str]]:
    """domain -> {gold_id (unprefixed, tanpa 'domain:'): chunk text}"""
    index: dict[str, dict[str, str]] = {}
    for dom in domains:
        chunks = load_domain_chunks(dom)
        by_id: dict[str, str] = {}
        for c in chunks:
            gid = gold_id_of(dom, c)
            if not gid:
                continue
            # gold_id_of balikin bentuk prefixed "domain:id..."; testset nyimpen tanpa prefix domain
            unprefixed = gid.split(":", 1)[1] if ":" in gid else gid
            by_id[unprefixed] = c.page_content
        index[dom] = by_id
    return index


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--testset", default="eval/candidate_testset.json")
    ap.add_argument("--out", default="eval/results/audit_dump.txt")
    ap.add_argument("--domain", choices=DOMAINS, help="filter cuma 1 domain")
    ap.add_argument("--id", help="filter cuma 1 question id (mis. SO004)")
    args = ap.parse_args()

    testset = json.loads(Path(args.testset).read_text(encoding="utf-8"))
    questions = testset["questions"]

    print("Loading & chunking sumber asli (sama kayak build_vectorstore)...", file=sys.stderr)
    gold_index = build_gold_index(DOMAINS)
    for dom, idx in gold_index.items():
        print(f"  {dom}: {len(idx)} chunk ter-index", file=sys.stderr)

    lines: list[str] = []
    n_ok, n_missing = 0, 0

    for q in questions:
        if args.id and q["id"] != args.id:
            continue
        gold_chunks = q.get("gold_chunks", {})
        if args.domain and not gold_chunks.get(args.domain):
            continue
        if not any(gold_chunks.values()):
            continue  # skip soal 'none'/out_of_scope, ga ada gold chunk

        lines.append("=" * 100)
        lines.append(f"ID: {q['id']}  |  route: {q['expected_route']}  |  difficulty: {q['difficulty']}  |  query_type: {q['query_type']}")
        lines.append(f"PERTANYAAN: {q['question']}")
        lines.append(f"NOTES: {q.get('notes', '')}")
        lines.append("-" * 100)
        lines.append(f"EXPECTED_ANSWER: {q['expected_answer']}")
        lines.append("EXPECTED_FACTS:")
        for f in q.get("expected_facts", []):
            lines.append(f"  - {f}")
        lines.append("-" * 100)

        for dom, ids in gold_chunks.items():
            for gid in ids:
                text = gold_index.get(dom, {}).get(gid)
                lines.append(f"[CHUNK ASLI] domain={dom}  gold_id={gid}")
                if text is None:
                    lines.append("  !!! GOLD_ID TIDAK DITEMUKAN DI CHUNK HASIL BUILD !!!")
                    n_missing += 1
                else:
                    lines.append(text)
                    n_ok += 1
                lines.append("")
        lines.append("")

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text("\n".join(lines), encoding="utf-8")

    print(f"Ditulis ke {out_path}", file=sys.stderr)
    print(f"gold_id ketemu: {n_ok}, gold_id HILANG: {n_missing}", file=sys.stderr)


if __name__ == "__main__":
    main()
