"""
compare_gate_csv.py -- is a regenerated gate CSV the SAME RESULT as the
committed one, when the two were produced on DIFFERENT machines?

WHY THIS EXISTS (Phase 8B.3 follow-up)
--------------------------------------
The two regression-gate CSVs are checked for byte identity after every run:

    data/adversarial_escalation_results.csv   (test_adversarial_escalation.py)
    data/ablation_baseline_results.csv        (run_ablation_study.py --mode baseline)

On the machine that produced them, byte identity is the right test and it stays
the LOCAL gate ritual: the same arithmetic on the same CPU gives the same bytes,
so any moved byte is a moved result.

On a DIFFERENT machine it is the wrong test. Both files write their floats at 6
decimal places, and an embedding similarity is a float32 BGE pass plus a FAISS
inner product whose last digits depend on the machine's BLAS kernel and SIMD
width (2.384e-07 in a Linux container, ~7e-07 on a GitHub runner). CLAUDE.md
recorded that the CSVs survived a platform change "by luck, not by
construction" -- adv_05 sat 0.0020 of a 6th-decimal unit from a rounding
boundary. The luck ran out on the first gates.yml run after 8B.3: five
adversarial rows (adv_01, 02, 04, 05, 07) differed by exactly one unit in the
6th decimal of rag_similarity, while every decision column, and every
tier1_confidence, was identical. The gate itself passed 9/9.

So on a runner this script replaces byte identity with a comparison of the
right KIND, the same move Phase 8B.2 made for the goldens:

  * every non-numeric and decision column (ids, categories, text, expected and
    actual escalation, pass_fail, predictions, tier used, correct) EXACTLY;
  * the header, the row count and the row order EXACTLY;
  * an empty numeric cell must be empty on both sides;
  * the two float columns within a tolerance DERIVED from their arithmetic,
    never fitted to a machine -- see TOLERANCES below;
  * the max delta per float column is printed on every run, pass or fail.

A flipped decision can never hide inside a tolerance here, because the decision
columns themselves are compared exactly.

THE REFERENCE is the committed blob (`git show HEAD:<path>`), CRLF normalised
-- Python's csv module writes '\\r\\n' on every platform while git stores LF,
the Phase 8B.1 lesson. The working-tree file is the freshly regenerated one.

Run from the project root, after the gate script has rewritten the file:

    python src/experiments/compare_gate_csv.py data/adversarial_escalation_results.csv
    python src/experiments/compare_gate_csv.py data/ablation_baseline_results.csv

Exit code 0 means the regenerated file is the same result. Nothing is written.
"""

from __future__ import annotations

import argparse
import csv
import io
import os
import subprocess
import sys

PROJECT_ROOT = os.path.dirname(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from src.agent.config import settings  # noqa: E402

# --------------------------------------------------------------------------- #
# TOLERANCES -- derived, one per KIND of number.
# --------------------------------------------------------------------------- #
# 6-dp ROUNDING, ON BOTH SIDES. Both writers round to 6 dp (the adversarial
# script with f"{v:.6f}", the ablation with round(x, 6)), so the committed and
# the regenerated value each carry up to 5e-7 of rounding error. Two rounded
# values of the SAME underlying number can therefore differ by a full 1e-6 --
# one unit in the last place -- whenever the true values straddle a rounding
# boundary. 5e-7 would be the budget for comparing a rounded value against an
# UNROUNDED one (which is what verify_deployment.py does); here it would fail a
# float64 tier1_confidence that differs by 1e-15 but happens to straddle a
# boundary.
ROUNDING_BOTH_SIDES = 1e-6

# tier1_confidence: float64 TF-IDF + LogReg against a committed vocabulary
# (Phase 8B.3). Measured cross-platform worst case over 54 golden tickets is
# 1.17e-15; 1e-12 is the same float64 budget test_pipeline_parity.py uses.
FLOAT64_BUDGET = 1e-12

# rag_similarity / top_similarity: a float32 inner product of two
# L2-normalised n-dim vectors. Higham, *Accuracy and Stability of Numerical
# Algorithms* (2nd ed.) section 3.1: |fl(x.y) - x.y| <= gamma_n |x||y| with
# gamma_n = n*u / (1 - n*u), u = 2^-24. For n = 768 this is 4.577846e-05.
# STATED LIMIT: it bounds the inner product only and does not model the BGE
# forward pass; it is used because it is derived and covers every observed
# cross-platform delta (2.384e-07 container, ~7e-07 runner) with room.
FLOAT32_UNIT_ROUNDOFF = 2.0 ** -24


def gamma_n(n: int, u: float = FLOAT32_UNIT_ROUNDOFF) -> float:
    return n * u / (1.0 - n * u)


FLOAT32_DOT_BOUND = gamma_n(settings.models.embedding_dim)

CONF_TOLERANCE = ROUNDING_BOTH_SIDES + FLOAT64_BUDGET
SIM_TOLERANCE = ROUNDING_BOTH_SIDES + FLOAT32_DOT_BOUND

# Which columns are floats, and which tolerance each carries. Every column NOT
# listed here is compared exactly. A file not listed here is refused rather
# than compared under a guessed schema.
SCHEMAS = {
    "adversarial_escalation_results.csv": {
        "key": ("id",),
        "floats": {"tier1_confidence": "conf", "rag_similarity": "sim"},
    },
    "ablation_baseline_results.csv": {
        "key": ("section", "key"),
        "floats": {"tier1_conf": "conf", "top_similarity": "sim"},
    },
}
TOLERANCE_BY_KIND = {"conf": CONF_TOLERANCE, "sim": SIM_TOLERANCE}


class CompareError(Exception):
    """A setup problem (unknown file, unreadable reference) -- not a mismatch."""


def _read_rows(text: str) -> list[dict]:
    text = text.replace("\r\n", "\n")
    return list(csv.DictReader(io.StringIO(text, newline="")))


def committed_text(rel_path: str, rev: str = "HEAD") -> str:
    """The committed blob of rel_path, as text. Raises CompareError on failure."""
    git_path = rel_path.replace(os.sep, "/")
    try:
        out = subprocess.run(
            ["git", "show", f"{rev}:{git_path}"],
            cwd=PROJECT_ROOT, capture_output=True, check=False)
    except FileNotFoundError as exc:
        raise CompareError("git is not on PATH; the reference is the committed "
                           "blob, so git is required.") from exc
    if out.returncode != 0:
        raise CompareError(
            f"could not read {rev}:{git_path} from git:\n    "
            + out.stderr.decode("utf-8", "replace").strip())
    return out.stdout.decode("utf-8")


def compare(reference_text: str, candidate_text: str, file_name: str) -> dict:
    """Compare two CSV texts under the schema for file_name.

    Returns {"failures": [str], "max_delta": {column: (delta, row_key)},
    "rows": int, "bytes_identical": bool}. Pure: no I/O.
    """
    if file_name not in SCHEMAS:
        raise CompareError(
            f"no comparison schema for {file_name!r}; known: "
            f"{', '.join(sorted(SCHEMAS))}. Add one rather than guess which "
            "columns are floats.")
    schema = SCHEMAS[file_name]
    floats = schema["floats"]

    ref = _read_rows(reference_text)
    new = _read_rows(candidate_text)
    failures: list[str] = []
    max_delta = {col: (0.0, None) for col in floats}

    ref_header = list(ref[0].keys()) if ref else []
    new_header = list(new[0].keys()) if new else []
    if ref_header != new_header:
        failures.append(f"header differs:\n    committed   {ref_header}\n"
                        f"    regenerated {new_header}")
        return {"failures": failures, "max_delta": max_delta,
                "rows": len(new), "bytes_identical": False}
    for col in list(floats) + list(schema["key"]):
        if col not in ref_header:
            raise CompareError(f"schema names column {col!r}, which "
                               f"{file_name} does not have")

    if len(ref) != len(new):
        failures.append(f"row count differs: committed {len(ref)}, "
                        f"regenerated {len(new)}")

    for i, (a, b) in enumerate(zip(ref, new)):
        row_key = "/".join(a[k] for k in schema["key"])
        for col in ref_header:
            va, vb = a[col], b[col]
            if col not in floats:
                if va != vb:
                    failures.append(f"row {i} ({row_key}) {col}: exact "
                                    f"mismatch {va!r} -> {vb!r}")
                continue
            if va == "" or vb == "":
                if va != vb:
                    failures.append(f"row {i} ({row_key}) {col}: empty on one "
                                    f"side only {va!r} -> {vb!r}")
                continue
            try:
                delta = abs(float(va) - float(vb))
            except ValueError:
                failures.append(f"row {i} ({row_key}) {col}: not a number "
                                f"{va!r} -> {vb!r}")
                continue
            if delta > max_delta[col][0]:
                max_delta[col] = (delta, row_key)
            tol = TOLERANCE_BY_KIND[floats[col]]
            if delta > tol:
                failures.append(f"row {i} ({row_key}) {col}: {va} -> {vb}, "
                                f"|delta| {delta:.3e} > tolerance {tol:.3e}")

    bytes_identical = (reference_text.replace("\r\n", "\n")
                       == candidate_text.replace("\r\n", "\n"))
    return {"failures": failures, "max_delta": max_delta, "rows": len(new),
            "bytes_identical": bytes_identical}


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(
        description="Cross-platform comparison of a regenerated gate CSV "
                    "against its committed blob.")
    parser.add_argument("path", help="gate CSV, relative to the project root")
    parser.add_argument("--rev", default="HEAD",
                        help="git revision holding the reference (default HEAD)")
    args = parser.parse_args(argv)

    rel_path = os.path.relpath(os.path.join(PROJECT_ROOT, args.path),
                               PROJECT_ROOT)
    abs_path = os.path.join(PROJECT_ROOT, rel_path)
    file_name = os.path.basename(rel_path)

    try:
        if not os.path.isfile(abs_path):
            raise CompareError(f"{rel_path} does not exist -- run the gate "
                               "script that writes it first.")
        with open(abs_path, "r", encoding="utf-8", newline="") as fh:
            candidate = fh.read()
        result = compare(committed_text(rel_path, args.rev), candidate,
                         file_name)
    except CompareError as exc:
        print(f"[ERROR] {exc}")
        return 2

    floats = SCHEMAS[file_name]["floats"]
    print(f"[compare] {rel_path} vs {args.rev}: {result['rows']} rows")
    print(f"          byte-identical (CRLF-normalised): "
          f"{result['bytes_identical']}  (informational off-machine)")
    for col, (delta, row_key) in result["max_delta"].items():
        tol = TOLERANCE_BY_KIND[floats[col]]
        print(f"          max |delta| {col:<17} {delta:.3e} "
              f"(row {row_key})  tolerance {tol:.3e}")

    if result["failures"]:
        print(f"[FAIL] {len(result['failures'])} mismatch(es):")
        for line in result["failures"]:
            print(f"    - {line}")
        return 1
    print("[PASS] every exact column identical; every float inside its "
          "derived tolerance.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
