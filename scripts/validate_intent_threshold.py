"""Tune the semantic-router similarity threshold on a held-out validation set.

Mechanism evaluated (the *intended* Enhanced semantic router), Top-K FIXED at 5:

    1. embed the query with the project's embeddings singleton
    2. cosine-compare it against ALL 40 reference utterances
    3. take the GLOBAL top-5 references (one ranking, not top-5 per class)
    4. group those 5 by their label and average each class that appears
    5. the class with the higher mean wins; winning_score = that mean
    6. predict the winning class if winning_score >= threshold, else INVALID

A class absent from the global top-5 gets score `None`, never 0.0 — the present
class then wins by default.

Only the threshold is tuned here. Top-K, the reference set, and the validation
set are held constant, and the final test sets (eval/main_testset.json,
eval/intent_testset.json) are never used for selection — they are read only for a
leakage diagnostic, which aborts the run if it finds an overlap.

NOTE on the production gate: `ragtrial.pipeline.intent.IntentStage` currently
routes through the `semantic-router` library with its own hardcoded utterance
lists and its own aggregation. This script deliberately implements the
specified global-top-5 mechanism against `intent_reference_set.json` instead, so
its numbers describe the intended router, not today's IntentStage. It shares the
embedding model with IntentStage (`ragtrial.llm.embeddings`), so the vector
geometry is identical.

Reads only; writes only under results/intent_similarity/.

Usage:
    uv run python scripts/validate_intent_threshold.py
    uv run python scripts/validate_intent_threshold.py --refresh-cache
    uv run python scripts/validate_intent_threshold.py --skip-leakage-check
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

# `scripts/` is not a package; running this file puts its own directory on
# sys.path, so the sibling analysis module imports directly. Reusing it keeps
# loading, embedding (incl. the shared cache) and cosine identical across the
# two intent tools.
sys.path.insert(0, str(Path(__file__).resolve().parent))

from analyze_intent_similarity import (  # noqa: E402
    AnalysisError,
    Utterance,
    cosine_similarity,
    get_embeddings,
    load_reference_set,
)

from ragtrial.config import PROJECT_ROOT  # noqa: E402

REFERENCE_PATH: Path = PROJECT_ROOT / "intent_reference_set.json"
VALIDATION_PATH: Path = PROJECT_ROOT / "intent_validation_set.json"
MAIN_TESTSET_PATH: Path = PROJECT_ROOT / "eval" / "main_testset.json"
INTENT_TESTSET_PATH: Path = PROJECT_ROOT / "eval" / "intent_testset.json"

RESULTS_DIR: Path = PROJECT_ROOT / "results" / "intent_similarity"
PREDICTIONS_CSV: Path = RESULTS_DIR / "threshold_validation_predictions.csv"
GRID_CSV: Path = RESULTS_DIR / "threshold_grid_results.csv"
REPORT_PATH: Path = RESULTS_DIR / "threshold_validation_report.txt"

VALID = "VALID"
INVALID = "INVALID"
POSITIVE_CLASS = VALID  # VALID = positive, INVALID = rejection/OOS class

TOP_K = 5  # FIXED — never tuned in this script
THRESHOLD_MIN_CENTS = 40  # 0.40
THRESHOLD_MAX_CENTS = 80  # 0.80
THRESHOLD_STEP_CENTS = 1  # 0.01


# ============ data model ============
@dataclass(frozen=True)
class Neighbour:
    """One reference utterance in a query's global top-K."""

    reference_id: str
    label: str  # "VALID" | "INVALID"
    similarity: float


@dataclass(frozen=True)
class QueryScores:
    """Threshold-independent scoring of one validation query.

    Computed once; every threshold in the grid reuses it unchanged.
    """

    query_id: str
    question: str
    ground_truth: str  # "VALID" | "INVALID"
    neighbours: Tuple[Neighbour, ...]  # global top-K, similarity desc
    valid_score: Optional[float]
    invalid_score: Optional[float]
    winning_class: str
    winning_score: float

    def predict(self, threshold: float) -> str:
        """Winning class if it clears the threshold, else the rejection class."""
        return self.winning_class if self.winning_score >= threshold else INVALID


@dataclass(frozen=True)
class GridRow:
    """Metrics for one threshold. Positive class = VALID."""

    threshold: float
    accuracy: float
    precision: float
    recall: float
    f1: float
    macro_f1: float
    invalid_recall: float
    invalid_f1: float
    tp: int
    tn: int
    fp: int
    fn: int


# ============ input ============
def load_validation_set(path: Path) -> List[Utterance]:
    """Load the validation set (40 VALID + 40 INVALID) and check shape + duplicates.

    Does NOT use load_reference_set so it can accept the larger size.
    """
    if not path.exists():
        raise AnalysisError(f"Validation set not found: {path}")
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

    # Validation set can be any size; just check for duplicates.
    utterances = list(valid) + list(invalid)

    ids = [u.id for u in utterances]
    dupes = sorted({i for i in ids if ids.count(i) > 1})
    if dupes:
        raise AnalysisError(f"{path}: duplicate ids: {', '.join(dupes)}")

    seen: Dict[str, str] = {}
    q_dupes: List[str] = []
    for u in utterances:
        key = u.question.strip().lower()
        if key in seen:
            q_dupes.append(f"{seen[key]} == {u.id}: {u.question!r}")
        else:
            seen[key] = u.id
    if q_dupes:
        raise AnalysisError(
            f"Validation set ({path.name}): duplicate questions:\n  " + "\n  ".join(q_dupes)
        )

    return utterances


def assert_no_reference_overlap(
    reference: Sequence[Utterance], validation: Sequence[Utterance]
) -> None:
    """FAIL LOUDLY on any exact question shared by the two sets.

    A validation query that also sits in the reference set is scored against
    itself, which inflates every metric — the threshold would be tuned on a
    self-similarity artefact.
    """
    ref_by_question = {u.question.strip().lower(): u for u in reference}
    hits = [
        (u, ref_by_question[u.question.strip().lower()])
        for u in validation
        if u.question.strip().lower() in ref_by_question
    ]
    if hits:
        lines = [f"  {v.id} == {r.id}: {v.question!r}" for v, r in hits]
        raise AnalysisError(
            "Validation set overlaps the reference set — threshold tuning aborted.\n"
            f"{len(hits)} overlapping question(s):\n" + "\n".join(lines)
        )

    ids = {u.id for u in reference} & {u.id for u in validation}
    if ids:
        raise AnalysisError(
            "Reference and validation sets share IDs (embedding cache keys would "
            f"collide): {', '.join(sorted(ids))}"
        )


def _read_testset_questions(path: Path) -> List[Tuple[str, str]]:
    """(id, question) pairs from a final test set. Missing file -> empty list."""
    if not path.exists():
        return []
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as e:
        raise AnalysisError(f"Invalid JSON in {path}: {e}") from e
    items = raw.get("questions") if isinstance(raw, dict) else None
    if not isinstance(items, list):
        return []
    out: List[Tuple[str, str]] = []
    for item in items:
        if isinstance(item, dict):
            qid, question = item.get("id"), item.get("question")
            if isinstance(qid, str) and isinstance(question, str):
                out.append((qid, question))
    return out


def check_testset_leakage(validation: Sequence[Utterance]) -> List[str]:
    """DIAGNOSTIC ONLY: exact validation/test-set question overlap.

    The final test sets are never used to score or select a threshold. They are
    read here purely to prove the validation set is disjoint from them; any hit
    is returned so the caller can stop before selecting a threshold.
    """
    findings: List[str] = []
    val_by_question = {u.question.strip().lower(): u for u in validation}
    for path in (MAIN_TESTSET_PATH, INTENT_TESTSET_PATH):
        for qid, question in _read_testset_questions(path):
            hit = val_by_question.get(question.strip().lower())
            if hit is not None:
                findings.append(f"{hit.id} == {path.name}/{qid}: {question!r}")
    return findings


# ============ router mechanism ============
def score_query(
    query: Utterance,
    reference: Sequence[Utterance],
    vectors: Dict[str, List[float]],
    *,
    top_k: int = TOP_K,
) -> QueryScores:
    """Global top-K over ALL references, then per-class means. No self-similarity:
    reference and validation sets are disjoint (enforced upstream).

    Ties in similarity are broken by reference ID so the ranking is deterministic
    across runs. An exact tie between the two class means resolves to INVALID —
    the conservative choice for a rejection gate (documented in the report).
    """
    scored = sorted(
        (
            Neighbour(
                reference_id=ref.id,
                label=VALID if ref.label == "valid" else INVALID,
                similarity=cosine_similarity(vectors[query.id], vectors[ref.id]),
            )
            for ref in reference
        ),
        key=lambda n: (-n.similarity, n.reference_id),
    )
    neighbours = tuple(scored[:top_k])

    def class_mean(label: str) -> Optional[float]:
        sims = [n.similarity for n in neighbours if n.label == label]
        return sum(sims) / len(sims) if sims else None

    valid_score = class_mean(VALID)
    invalid_score = class_mean(INVALID)

    if valid_score is None and invalid_score is None:
        raise AnalysisError(f"{query.id}: empty top-{top_k} — no references to compare against.")
    if invalid_score is None:
        winning_class, winning_score = VALID, valid_score
    elif valid_score is None:
        winning_class, winning_score = INVALID, invalid_score
    elif valid_score > invalid_score:
        winning_class, winning_score = VALID, valid_score
    else:
        winning_class, winning_score = INVALID, invalid_score

    return QueryScores(
        query_id=query.id,
        question=query.question,
        ground_truth=VALID if query.label == "valid" else INVALID,
        neighbours=neighbours,
        valid_score=valid_score,
        invalid_score=invalid_score,
        winning_class=winning_class,
        winning_score=float(winning_score),
    )


# ============ metrics ============
def _ratio(numerator: int, denominator: int) -> float:
    """0.0 for an undefined ratio — an absent class scores 0, never crashes."""
    return numerator / denominator if denominator else 0.0


def _f1(precision: float, recall: float) -> float:
    return 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0


def evaluate_threshold(scores: Sequence[QueryScores], threshold: float) -> GridRow:
    """Confusion matrix + metrics at one threshold. Positive class = VALID."""
    tp = fn = tn = fp = 0
    for s in scores:
        predicted = s.predict(threshold)
        if s.ground_truth == VALID:
            if predicted == VALID:
                tp += 1
            else:
                fn += 1
        else:
            if predicted == VALID:
                fp += 1
            else:
                tn += 1

    total = tp + tn + fp + fn
    accuracy = _ratio(tp + tn, total)

    precision = _ratio(tp, tp + fp)
    recall = _ratio(tp, tp + fn)
    f1 = _f1(precision, recall)

    invalid_precision = _ratio(tn, tn + fn)
    invalid_recall = _ratio(tn, tn + fp)
    invalid_f1 = _f1(invalid_precision, invalid_recall)

    return GridRow(
        threshold=threshold,
        accuracy=accuracy,
        precision=precision,
        recall=recall,
        f1=f1,
        macro_f1=(f1 + invalid_f1) / 2,  # unweighted — both classes count equally
        invalid_recall=invalid_recall,
        invalid_f1=invalid_f1,
        tp=tp,
        tn=tn,
        fp=fp,
        fn=fn,
    )


def threshold_candidates() -> List[float]:
    """0.40 .. 0.80 step 0.01, built from integers to avoid float drift."""
    return [
        c / 100
        for c in range(THRESHOLD_MIN_CENTS, THRESHOLD_MAX_CENTS + 1, THRESHOLD_STEP_CENTS)
    ]


def select_best(grid: Sequence[GridRow]) -> GridRow:
    """Highest Macro-F1; ties -> highest INVALID recall; still tied -> lower threshold."""
    if not grid:
        raise AnalysisError("Empty threshold grid — nothing to select.")
    return sorted(grid, key=lambda r: (-r.macro_f1, -r.invalid_recall, r.threshold))[0]


# ============ output ============
def _fmt_score(value: Optional[float]) -> str:
    return "None" if value is None else f"{value:.4f}"


def save_predictions_csv(
    scores: Sequence[QueryScores], best: GridRow, path: Path = PREDICTIONS_CSV
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    header = [
        "query_id",
        "question",
        "ground_truth",
        "valid_score",
        "invalid_score",
        "winning_class",
        "winning_score",
        "selected_threshold",
        "predicted_class",
        "correct",
    ]
    for i in range(1, TOP_K + 1):
        header += [f"top{i}_reference_id", f"top{i}_reference_label", f"top{i}_similarity"]

    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(header)
        for s in scores:
            predicted = s.predict(best.threshold)
            row: List[object] = [
                s.query_id,
                s.question,
                s.ground_truth,
                _fmt_score(s.valid_score),
                _fmt_score(s.invalid_score),
                s.winning_class,
                f"{s.winning_score:.4f}",
                f"{best.threshold:.2f}",
                predicted,
                predicted == s.ground_truth,
            ]
            for i in range(TOP_K):
                if i < len(s.neighbours):
                    n = s.neighbours[i]
                    row += [n.reference_id, n.label, f"{n.similarity:.6f}"]
                else:
                    row += ["", "", ""]
            writer.writerow(row)


def save_grid_csv(grid: Sequence[GridRow], path: Path = GRID_CSV) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(
            ["threshold", "accuracy", "precision", "recall", "f1", "macro_f1",
             "tp", "tn", "fp", "fn"]
        )
        for r in grid:
            writer.writerow([
                f"{r.threshold:.2f}",
                f"{r.accuracy:.4f}",
                f"{r.precision:.4f}",
                f"{r.recall:.4f}",
                f"{r.f1:.4f}",
                f"{r.macro_f1:.4f}",
                r.tp, r.tn, r.fp, r.fn,
            ])


def _confusion_block(best: GridRow) -> List[str]:
    return [
        "Positive class: VALID   (INVALID = rejection / out-of-scope class)",
        "",
        f"{'':<18}{'Pred VALID':>14}{'Pred INVALID':>16}",
        f"{'Actual VALID':<18}{best.tp:>14}{best.fn:>16}",
        f"{'Actual INVALID':<18}{best.fp:>14}{best.tn:>16}",
        "",
        f"TP = {best.tp}   FN = {best.fn}   FP = {best.fp}   TN = {best.tn}",
    ]


def build_report(
    *,
    grid: Sequence[GridRow],
    best: GridRow,
    scores: Sequence[QueryScores],
    n_ref_valid: int,
    n_ref_invalid: int,
    n_val_valid: int,
    n_val_invalid: int,
    model: str,
    dim: int,
    timestamp: str,
    leakage_note: str,
) -> str:
    sep = "-" * 40
    correct = sum(1 for s in scores if s.predict(best.threshold) == s.ground_truth)
    incorrect = len(scores) - correct

    lines: List[str] = [
        "=" * 40,
        "SEMANTIC ROUTER THRESHOLD VALIDATION",
        "=" * 40,
        "",
        f"Generated: {timestamp}",
        f"Reference set: {REFERENCE_PATH}",
        f"Validation set: {VALIDATION_PATH}",
        f"Embedding model: {model}",
        f"Embedding dimension: {dim}",
        f"Global Top-K: {TOP_K} (fixed — not tuned)",
        "Similarity: cosine",
        "",
        "Reference:",
        f"VALID = {n_ref_valid}",
        f"INVALID = {n_ref_invalid}",
        "",
        "Validation:",
        f"VALID = {n_val_valid}",
        f"INVALID = {n_val_invalid}",
        "",
        "Threshold search:",
        f"{THRESHOLD_MIN_CENTS / 100:.2f}-{THRESHOLD_MAX_CENTS / 100:.2f}",
        f"step = {THRESHOLD_STEP_CENTS / 100:.2f}",
        f"candidates = {len(grid)}",
        "Primary metric = Macro-F1 (unweighted mean of VALID F1 and INVALID F1)",
        "",
        f"Test-set leakage diagnostic: {leakage_note}",
        "  (eval/main_testset.json and eval/intent_testset.json were NOT used to",
        "   score or select the threshold.)",
        "",
        sep,
        "THRESHOLD RESULTS",
        sep,
        f"{'thresh':>7}{'acc':>9}{'prec':>9}{'rec':>9}{'f1':>9}{'macroF1':>10}"
        f"{'TP':>5}{'TN':>5}{'FP':>5}{'FN':>5}",
    ]
    for r in grid:
        marker = "  <-- selected" if r.threshold == best.threshold else ""
        lines.append(
            f"{r.threshold:>7.2f}{r.accuracy:>9.4f}{r.precision:>9.4f}{r.recall:>9.4f}"
            f"{r.f1:>9.4f}{r.macro_f1:>10.4f}{r.tp:>5}{r.tn:>5}{r.fp:>5}{r.fn:>5}{marker}"
        )

    lines += [
        "",
        sep,
        "BEST THRESHOLD",
        sep,
        f"Selected threshold: {best.threshold:.2f}",
        f"Macro-F1: {best.macro_f1:.4f}",
        f"Accuracy: {best.accuracy:.4f}",
        f"Precision: {best.precision:.4f}   (VALID)",
        f"Recall: {best.recall:.4f}   (VALID)",
        f"F1: {best.f1:.4f}   (VALID)",
        f"INVALID F1: {best.invalid_f1:.4f}",
        "",
        "Tie-break order: highest Macro-F1, then highest INVALID recall, then the",
        "lower threshold.",
        "",
        sep,
        "CONFUSION MATRIX",
        sep,
    ]
    lines += _confusion_block(best)

    lines += [
        "",
        sep,
        "VALIDATION SUMMARY",
        sep,
        f"Number of validation queries: {len(scores)}",
        f"Correct: {correct}",
        f"Incorrect: {incorrect}",
        "",
        f"VALID recall: {best.recall:.4f}   ({best.tp}/{best.tp + best.fn})",
        f"INVALID recall: {best.invalid_recall:.4f}   ({best.tn}/{best.tn + best.fp})",
        "",
        sep,
        "MISCLASSIFIED QUERIES",
        sep,
    ]

    misses = [s for s in scores if s.predict(best.threshold) != s.ground_truth]
    if not misses:
        lines.append("None — every validation query was classified correctly.")
    for s in misses:
        lines += [
            f"[{s.query_id}] {s.question}",
            f"  ground truth   : {s.ground_truth}",
            f"  predicted      : {s.predict(best.threshold)}",
            f"  valid_score    : {_fmt_score(s.valid_score)}",
            f"  invalid_score  : {_fmt_score(s.invalid_score)}",
            f"  winning_class  : {s.winning_class}",
            f"  winning_score  : {s.winning_score:.4f}"
            f"  (threshold {best.threshold:.2f})",
            "  top-5 references:",
        ]
        for rank, n in enumerate(s.neighbours, 1):
            lines.append(
                f"    {rank}. {n.reference_id:<24} {n.label:<8} {n.similarity:.4f}"
            )
        lines.append("")

    lines += [
        sep,
        "NOTES",
        sep,
        "Only the threshold was tuned. Top-K stayed fixed at 5, the global top-5 was",
        "taken before grouping by class, and a class absent from the top-5 scored",
        "None rather than 0.0.",
        "",
        "An exact tie between the two class means resolves to INVALID (conservative",
        "for a rejection gate); with float cosine values this effectively never fires.",
        "",
        "These metrics describe the validation set only. Final performance must be",
        "measured on the untouched test sets.",
        "",
    ]
    return "\n".join(lines)


# ============ entry point ============
def main() -> int:
    parser = argparse.ArgumentParser(
        description="Tune the semantic-router threshold on intent_validation_set.json (Top-K fixed at 5)."
    )
    parser.add_argument(
        "--refresh-cache",
        action="store_true",
        help="ignore cached embeddings and re-embed everything",
    )
    parser.add_argument(
        "--skip-leakage-check",
        action="store_true",
        help="skip the validation-vs-test-set overlap diagnostic",
    )
    args = parser.parse_args()

    timestamp = datetime.now().isoformat(timespec="seconds")
    print("=== Semantic router threshold validation ===")
    print(f"  Timestamp  : {timestamp}")
    print(f"  Reference  : {REFERENCE_PATH}")
    print(f"  Validation : {VALIDATION_PATH}")

    try:
        ref_valid, ref_invalid = load_reference_set(REFERENCE_PATH)
        reference = list(ref_valid) + list(ref_invalid)
    except AnalysisError as e:
        raise AnalysisError(f"reference set: {e}") from e

    validation = load_validation_set(VALIDATION_PATH)
    n_ref_valid = sum(1 for u in reference if u.label == "valid")
    n_val_valid = sum(1 for u in validation if u.label == "valid")
    print(f"  Reference  : {n_ref_valid} VALID + {len(reference) - n_ref_valid} INVALID")
    print(f"  Validation : {n_val_valid} VALID + {len(validation) - n_val_valid} INVALID")

    assert_no_reference_overlap(reference, validation)
    print("  Overlap    : none between reference and validation sets")

    if args.skip_leakage_check:
        leakage_note = "skipped (--skip-leakage-check)"
        print("  Leakage    : diagnostic skipped")
    else:
        findings = check_testset_leakage(validation)
        if findings:
            raise AnalysisError(
                "Validation questions also appear in the final test sets — threshold "
                "selection stopped so the test sets stay untouched by tuning.\n"
                + "\n".join(f"  {f}" for f in findings)
            )
        leakage_note = "clean (no validation question found in either final test set)"
        print("  Leakage    : clean vs main_testset.json / intent_testset.json")

    vectors, model, dim = get_embeddings(
        list(reference) + list(validation), use_cache=not args.refresh_cache
    )
    print(f"  Model      : {model} ({dim}-d)")

    scores = [score_query(q, reference, vectors) for q in validation]
    grid = [evaluate_threshold(scores, t) for t in threshold_candidates()]
    best = select_best(grid)

    save_grid_csv(grid)
    save_predictions_csv(scores, best)
    report = build_report(
        grid=grid,
        best=best,
        scores=scores,
        n_ref_valid=n_ref_valid,
        n_ref_invalid=len(reference) - n_ref_valid,
        n_val_valid=n_val_valid,
        n_val_invalid=len(validation) - n_val_valid,
        model=model,
        dim=dim,
        timestamp=timestamp,
        leakage_note=leakage_note,
    )
    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    REPORT_PATH.write_text(report, encoding="utf-8")

    print()
    print("=" * 52)
    print("SUMMARY")
    print("=" * 52)
    print(f"  Validation set size : {len(validation)}")
    print(f"  Reference set size  : {len(reference)}")
    print(f"  Top-K               : {TOP_K} (fixed)")
    print(f"  Selected threshold  : {best.threshold:.2f}")
    print(f"  Best Macro-F1       : {best.macro_f1:.4f}")
    print(f"  Accuracy            : {best.accuracy:.4f}")
    print(f"  Precision (VALID)   : {best.precision:.4f}")
    print(f"  Recall (VALID)      : {best.recall:.4f}")
    print(f"  F1 (VALID)          : {best.f1:.4f}")
    print()
    for line in _confusion_block(best):
        print(f"  {line}" if line else "")
    print()
    print(f"  Report      : {REPORT_PATH}")
    print(f"  Grid CSV    : {GRID_CSV}")
    print(f"  Per-query   : {PREDICTIONS_CSV}")
    print("  (full threshold table is in the report and grid CSV)")
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except AnalysisError as err:
        print(f"\nERROR: {err}", file=sys.stderr)
        sys.exit(1)
