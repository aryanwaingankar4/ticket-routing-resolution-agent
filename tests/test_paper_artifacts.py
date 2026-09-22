"""Golden parity for the PAPER (Phase 8A).

tests/goldens/*.json pin the routing decisions. This file pins the write-up:
it rebuilds paper/ into a temporary directory and fails if any NUMBERS.md
value, table CSV or figure-data CSV has moved.

Why it exists: Phase 7A recorded two domain AUCs that were computed in an
interactive session, quoted at a gate, and never committed. They do not
reproduce. A number in the paper that nothing regenerates is a number that can
stop being true without anyone noticing.

PDF and PNG bytes are deliberately NOT compared. Rendering is not what carries
a number, and a rasteriser upgrade should not fail the build; the data behind
every figure is compared instead.
"""

from __future__ import annotations

import ast
import hashlib
import json
import os
import subprocess
import sys

import pytest

PROJECT_ROOT = os.path.abspath(
    os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
PAPER_DIR = os.path.join(PROJECT_ROOT, "paper")
BUILDER = os.path.join(PROJECT_ROOT, "src", "experiments",
                       "build_paper_artifacts.py")

pytestmark = pytest.mark.skipif(
    not os.path.isdir(PAPER_DIR),
    reason="paper/ has not been built yet; run "
           "python src/experiments/build_paper_artifacts.py")


def _sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


# RECONCILIATION.md is an audit OF THE FOUR PROJECT DOCUMENTS, not a
# published number. Those documents legitimately change every session, so
# comparing its bytes would fail the build for a README edit that moved
# nothing. Its meaningful invariant -- zero anchored mismatches -- is asserted
# separately, against the freshly rebuilt copy.
NOT_BYTE_COMPARED = {"RECONCILIATION.md"}


def _comparable_files(root):
    """Everything whose bytes must not move: values and data, never pixels."""
    out = {}
    for dirpath, _dirnames, filenames in os.walk(root):
        for name in filenames:
            if name.endswith((".pdf", ".png")):
                continue
            full = os.path.join(dirpath, name)
            rel_path = os.path.relpath(full, root).replace(os.sep, "/")
            if rel_path in NOT_BYTE_COMPARED:
                continue
            out[rel_path] = full
    return out


@pytest.fixture(scope="module")
def rebuilt(tmp_path_factory):
    out = tmp_path_factory.mktemp("paper_parity")
    result = subprocess.run(
        [sys.executable, BUILDER, "--out", str(out), "--no-render",
         "--force"],
        cwd=PROJECT_ROOT, capture_output=True, text=True)
    assert result.returncode == 0, (
        "build_paper_artifacts.py failed:\n"
        f"{result.stdout[-4000:]}\n{result.stderr[-4000:]}")
    return str(out)


# ---------------------------------------------------------------------------
# The parity check itself
# ---------------------------------------------------------------------------
@pytest.mark.slow
def test_numbers_md_is_byte_identical(rebuilt):
    """Every published number, unchanged."""
    committed = os.path.join(PAPER_DIR, "NUMBERS.md")
    fresh = os.path.join(rebuilt, "NUMBERS.md")
    assert _sha256(committed) == _sha256(fresh), (
        "paper/NUMBERS.md no longer matches what the committed result files "
        "produce. Either a result file changed (check PROVENANCE.json) or "
        "the builder changed. Rebuild deliberately with --force; do not "
        "edit NUMBERS.md.")


@pytest.mark.slow
def test_every_table_and_figure_datum_is_byte_identical(rebuilt):
    committed = _comparable_files(PAPER_DIR)
    fresh = _comparable_files(rebuilt)

    missing = sorted(set(committed) - set(fresh))
    added = sorted(set(fresh) - set(committed))
    assert not missing, f"the rebuild no longer produces: {missing}"
    assert not added, f"the rebuild produces new files not committed: {added}"

    drifted = [name for name in sorted(committed)
               if _sha256(committed[name]) != _sha256(fresh[name])]
    assert not drifted, (
        "these paper artifacts changed without a deliberate rebuild: "
        f"{drifted}")


@pytest.mark.slow
def test_the_documents_still_agree_with_their_source_files(rebuilt):
    """The reconciliation audit, run live against the CURRENT documents.

    This is the check that would have caught the Phase 7A drift: a figure a
    document states that the committed file no longer produces.
    """
    with open(os.path.join(rebuilt, "RECONCILIATION.md"), "r",
              encoding="utf-8") as fh:
        report = fh.read()

    assert "**0 anchored mismatch(es).**" in report, (
        "a project document now states a value that its source file does not "
        "produce. Open paper/RECONCILIATION.md, find the MISMATCH row, and "
        "fix the DOCUMENT -- never the result file.")

    stale_anchors = [line for line in report.splitlines()
                     if "anchor absent from every document" in line]
    assert not stale_anchors, (
        "these anchors no longer match anywhere in the documents, so they "
        "have stopped checking anything: " + "; ".join(stale_anchors))


@pytest.mark.slow
def test_source_files_still_hash_to_what_the_paper_was_built_from():
    """Distinguishes 'the builder changed' from 'a result file changed'.

    Those are different bugs and a parity failure should say which one it is.
    """
    with open(os.path.join(PAPER_DIR, "PROVENANCE.json"), "r",
              encoding="utf-8") as fh:
        provenance = json.load(fh)

    drifted = []
    for rel_path, expected in sorted(provenance["sources"].items()):
        full = os.path.join(PROJECT_ROOT, rel_path)
        assert os.path.isfile(full), (
            f"{rel_path} is named in paper/PROVENANCE.json but no longer "
            f"exists")
        if _sha256(full) != expected:
            drifted.append(rel_path)
    assert not drifted, (
        "these RESULT FILES changed since paper/ was built, so the paper is "
        f"stale rather than the builder being wrong: {drifted}")


# ---------------------------------------------------------------------------
# The structural rules -- fast, no rebuild needed
# ---------------------------------------------------------------------------
def test_no_emit_call_passes_a_numeric_literal():
    """The rule that makes Phase 8A structural rather than aspirational.

    A hard-coded number in an emit() call is a retyped number, which is
    exactly what this phase exists to abolish. Strings are allowed (verdicts
    such as "blocked_auc_ge_0.95" come from the source files as text).
    """
    with open(BUILDER, "r", encoding="utf-8") as fh:
        tree = ast.parse(fh.read(), filename=BUILDER)

    offenders = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        name = node.func.id if isinstance(node.func, ast.Name) else (
            node.func.attr if isinstance(node.func, ast.Attribute) else "")
        if name != "emit":
            continue
        value = None
        if len(node.args) >= 2:
            value = node.args[1]
        for kw in node.keywords:
            if kw.arg == "value":
                value = kw.value
        if value is None:
            continue
        literal = value
        if isinstance(literal, ast.UnaryOp) and isinstance(
                literal.op, (ast.USub, ast.UAdd)):
            literal = literal.operand
        if isinstance(literal, ast.Constant) and isinstance(
                literal.value, (int, float, complex)) and not isinstance(
                    literal.value, bool):
            first = node.args[0]
            label = (first.value if isinstance(first, ast.Constant)
                     else f"line {node.lineno}")
            offenders.append(f"{label} (line {node.lineno})")

    assert not offenders, (
        "these emit() calls pass a hard-coded number instead of reading it "
        f"from a committed source file: {offenders}")


def test_do_not_cite_literals_do_not_reappear():
    """A retired number must not creep back into the paper surface."""
    with open(os.path.join(PAPER_DIR, "NUMBERS.md"), "r",
              encoding="utf-8") as fh:
        numbers_md = fh.read()

    marker = "\n## DO NOT CITE\n"
    assert marker in numbers_md, "NUMBERS.md lost its DO NOT CITE section"
    numbers_body = numbers_md.split(marker)[0]

    sys.path.insert(0, PROJECT_ROOT)
    from src.experiments.build_paper_artifacts import DO_NOT_CITE

    literals = [lit for item in DO_NOT_CITE for lit in item["literals"]]
    assert literals, "the do-not-cite list has no literals left to check"

    # Only the PROSE surface is checked. A data CSV is generated numbers, not
    # a citation, and a retired literal can occur inside one by coincidence --
    # 0.6706 is a real risk-coverage threshold as well as a retired AUC bound.
    offenders = []
    for dirpath, _dirnames, filenames in os.walk(PAPER_DIR):
        for name in filenames:
            if not name.endswith((".md", ".tex", ".txt")):
                continue
            full = os.path.join(dirpath, name)
            rel_path = os.path.relpath(full, PAPER_DIR).replace(os.sep, "/")
            if rel_path == "RECONCILIATION.md":
                # The audit's job is to REPORT where retired literals still
                # appear in the lab notebook, so it is allowed to name them.
                continue
            with open(full, "r", encoding="utf-8", errors="replace") as fh:
                text = fh.read()
            if rel_path == "NUMBERS.md":
                text = numbers_body
            for lit in literals:
                if lit in text:
                    offenders.append(f"{rel_path}: {lit!r}")

    assert not offenders, (
        "retired numbers reappeared in the paper surface outside the "
        f"do-not-cite section: {offenders}")


def test_case_study_axes_carry_no_confidence_interval():
    """Pre-registered case-study axes report counts, never a rate with a CI.

    Phase 6C pre-registered that the 2-positive axis is reported as a count.
    A Wilson interval on two observations would look like a measurement.
    """
    with open(os.path.join(PAPER_DIR, "NUMBERS.md"), "r",
              encoding="utf-8") as fh:
        lines = fh.read().splitlines()

    offenders = []
    for line in lines:
        if "`case-study`" in line and "95% CI" in line:
            offenders.append(line.strip()[:120])
    assert not offenders, (
        f"a case-study axis was given a confidence interval: {offenders}")


def test_emitter_refuses_a_ci_on_a_case_study_axis():
    """The same rule, enforced at the choke point rather than in the output."""
    sys.path.insert(0, PROJECT_ROOT)
    from src.experiments.build_paper_artifacts import Emitter, TAG_CASE_STUDY

    emitter = Emitter()
    with pytest.raises(ValueError, match="case-study"):
        emitter.emit("X.test", "2/2", "README.md", command="none",
                     tags=[TAG_CASE_STUDY], ci=(0.1, 0.9))


def test_emitter_refuses_a_value_with_no_committed_source():
    sys.path.insert(0, PROJECT_ROOT)
    from src.experiments.build_paper_artifacts import Emitter

    emitter = Emitter()
    with pytest.raises(ValueError, match="does not exist"):
        emitter.emit("X.test", "1/2", "data/there_is_no_such_file.csv",
                     command="none")


def test_emitter_tags_a_difference_inside_its_own_noise_band():
    sys.path.insert(0, PROJECT_ROOT)
    from src.experiments.build_paper_artifacts import (
        Emitter, TAG_WITHIN_BAND)

    emitter = Emitter()
    emitter.emit("X.small", 0.01, "README.md", command="none",
                 band=0.05, diff=0.01)
    emitter.emit("X.large", 0.90, "README.md", command="none",
                 band=0.05, diff=0.90)
    tagged = {n.id: n.tags for n in emitter.numbers}
    assert TAG_WITHIN_BAND in tagged["X.small"]
    assert TAG_WITHIN_BAND not in tagged["X.large"]


def test_statistics_agree_with_the_repos_own_implementations():
    """Rule 6, as a test: a second independent derivation of each statistic."""
    sys.path.insert(0, PROJECT_ROOT)
    from src.experiments.build_paper_artifacts import cross_check_statistics

    report = cross_check_statistics()
    assert report["wilson_checks"] > 0
    assert report["mcnemar_checks"] > 0


def test_paper_records_the_frozen_production_gates():
    """Phases 5-9 are measurement only; the paper surface must say so."""
    with open(os.path.join(PAPER_DIR, "PROVENANCE.json"), "r",
              encoding="utf-8") as fh:
        provenance = json.load(fh)

    gates = provenance["production_gates"]
    assert gates["cascade_confidence_threshold"] == 0.50
    assert gates["rag_similarity_threshold"] == 0.67
    assert gates["clustering_resolution_similarity_threshold"] == 0.80
    assert gates["conformal_enabled"] is False
    assert gates["drift_enabled"] is False


def test_the_phase_8a1_sources_exist_and_are_readable():
    """Phase 8A.1 gave four documented numbers a committed source.

    If one of these files goes missing, emit() will refuse the number and the
    build fails -- but it fails deep inside a table builder. Checking here
    says plainly which file is gone.
    """
    expected = {
        "data/baseline_tfidf_benchmark14.csv":
            "TF-IDF + LogReg on the 14-ticket benchmark",
        "data/baseline_tfidf_indistribution.csv":
            "the TF-IDF baseline's in-distribution metrics",
        "data/cascade_threshold_sweep.csv":
            "the cascade threshold-by-target-accuracy sweep",
        "data/cascade_calibration_attempts.csv":
            "the three cascade calibration attempts",
        "data/rag_self_retrieval_check.csv":
            "the in-domain self-retrieval contamination check",
        "data/distilbert_finetune_metrics.csv":
            "the DistilBERT per-epoch metrics on both benchmarks",
    }
    missing = [f"{path} ({what})" for path, what in expected.items()
               if not os.path.isfile(os.path.join(PROJECT_ROOT, path))]
    assert not missing, (
        "these Phase 8A.1 result files are gone, so the numbers they source "
        f"have no committed source again: {missing}")


def test_the_previously_unsourced_numbers_are_now_emitted():
    """The point of Phase 8A.1, pinned.

    Each id below replaced an entry on 8A's 'no committed source' list. If one
    disappears, a number has quietly gone back to being unsourced.
    """
    with open(os.path.join(PAPER_DIR, "NUMBERS.md"), "r",
              encoding="utf-8") as fh:
        text = fh.read()

    for number_id in ("T1.tfidf_baseline.benchmark14",
                      "T1.tfidf_baseline.in_distribution_accuracy",
                      "T1.distilbert.benchmark14",
                      "T1.distilbert.benchmark45",
                      "T4.sweep.threshold.target70",
                      "T4.calibration.attempt3.threshold",
                      "T5.self_retrieval_rate"):
        assert number_id in text, (
            f"{number_id} is no longer in NUMBERS.md -- a number that Phase "
            f"8A.1 sourced has lost its source")

    # The 45-ticket DistilBERT score is a FIRST measurement, not a
    # reproduction, and must stay labelled as one.
    for line in text.splitlines():
        if line.startswith("| `T1.distilbert.benchmark45`"):
            assert "new-measurement" in line, (
                "T1.distilbert.benchmark45 lost its new-measurement tag; it "
                "has never been measured before and must not read as a "
                "reproduction")
            break
    else:
        raise AssertionError("T1.distilbert.benchmark45 row not found")


def test_the_remaining_no_source_gap_is_only_the_35_ticket_set():
    """8A listed five numbers with no committed source; 8A.1 closed four.

    What remains is cascade calibration attempt 2, whose 35-ticket set was
    never committed and cannot be re-run. If anything else joins that list,
    it should be a deliberate decision, not a silent regression.
    """
    with open(os.path.join(PAPER_DIR, "NUMBERS.md"), "r",
              encoding="utf-8") as fh:
        text = fh.read()

    marker = "## Numbers in the documents with NO committed source"
    assert marker in text, "NUMBERS.md lost its no-committed-source section"
    section = text.split(marker, 1)[1].split("\n## ", 1)[0]
    rows = [line for line in section.splitlines()
            if line.startswith("| ") and "|---" not in line
            and not line.startswith("| number |")]
    assert len(rows) == 1, (
        "expected exactly one remaining unsourced number (cascade "
        f"calibration attempt 2); found {len(rows)}:\n" + "\n".join(rows))
    assert "35" in rows[0], (
        "the one remaining unsourced number should be the 35-ticket "
        f"calibration attempt; found: {rows[0]}")


def test_every_clause_group_in_numbers_md_carries_its_clause():
    """A number in a clause group may never be lifted out of its clause."""
    sys.path.insert(0, PROJECT_ROOT)
    from src.experiments.build_paper_artifacts import CLAUSES

    with open(os.path.join(PAPER_DIR, "NUMBERS.md"), "r",
              encoding="utf-8") as fh:
        text = fh.read()

    for group_id in CLAUSES:
        assert f"### Clause group `{group_id}`" in text, (
            f"clause group {group_id} is missing from NUMBERS.md")
        first_sentence = CLAUSES[group_id].split(".")[0].replace("\n", " ")
        assert first_sentence.split(" -- ")[0][:40] in text.replace("\n> ",
                                                                   " "), (
            f"clause group {group_id} lost its clause text")
