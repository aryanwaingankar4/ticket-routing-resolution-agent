"""
Phase 7B helpers, verified against CONSTRUCTED ground truth.

7B's output is a table of coverage gaps and a set of split sizes -- numbers
that stay in plausible ranges whatever goes wrong upstream. A misaligned
embedding cache, a near-duplicate spanning train and calibration, or a
component count off by one would all produce a table that looks entirely
normal. So the structural helpers are checked on small inputs whose duplicate,
split and neighbour structure is built by hand rather than eyeballed from a
run.

Two bugs found during 7B's own development are pinned here: a DictWriter that
dropped rows whose key sets differed (which would have silently deleted every
BLOCKED and skipped configuration from the Design B file), and a split that
was disjoint by row while sharing de-duplication components.

No network, no model loading, no external data. The expensive paths (BGE,
FAISS over 28k rows, the bootstrap) are covered by the scripts' own in-run
rule-6 cross-checks against brute force.
"""

from __future__ import annotations

import csv
import os

import numpy as np
import pandas as pd
import pytest

from src.experiments.run_external_conformal_shift import (
    build_texts,
    dedup_components,
    stratified_split,
    verify_split,
    write_csv,
)


# --------------------------------------------------------------------------- #
# Text composition -- must match exactly what 7A encoded
# --------------------------------------------------------------------------- #
def test_build_texts_joins_subject_and_body_with_one_space():
    df = pd.DataFrame({"subject": ["Printer down"], "body": ["It is offline"]})
    assert build_texts(df) == ["Printer down It is offline"]


def test_build_texts_treats_a_missing_subject_as_empty_not_nan():
    df = pd.DataFrame({"subject": [np.nan], "body": ["body only"]})
    texts = build_texts(df)
    assert texts == [" body only"]
    assert "nan" not in texts[0]


def test_build_texts_preserves_row_order():
    df = pd.DataFrame({"subject": ["a", "b", "c"], "body": ["1", "2", "3"]})
    assert build_texts(df) == ["a 1", "b 2", "c 3"]


# --------------------------------------------------------------------------- #
# De-duplication -- constructed component structure
# --------------------------------------------------------------------------- #
def _unit(vectors):
    arr = np.asarray(vectors, dtype=np.float32)
    return arr / np.linalg.norm(arr, axis=1, keepdims=True)


def test_dedup_finds_one_component_per_identical_group():
    # Three pairs of identical vectors in three orthogonal directions.
    emb = _unit([[1, 0, 0], [1, 0, 0],
                 [0, 1, 0], [0, 1, 0],
                 [0, 0, 1], [0, 0, 1]])
    comp, reps = dedup_components(emb)
    assert len(set(comp.tolist())) == 3
    assert len(reps) == 3
    assert comp[0] == comp[1] and comp[2] == comp[3] and comp[4] == comp[5]
    assert comp[0] != comp[2]


def test_dedup_leaves_distinct_vectors_as_singletons():
    emb = _unit(np.eye(5))
    comp, reps = dedup_components(emb)
    assert len(set(comp.tolist())) == 5
    assert len(reps) == 5


def test_dedup_is_transitive_through_a_chain():
    """A--B and B--C above threshold puts A and C in one component.

    Union-find on connected components is deliberate, not an approximation of
    pairwise grouping: the production clustering does the same thing, and a
    chain is exactly how a near-duplicate group grows.
    """
    a = [1.0, 0.0]
    b = [0.97, 0.243]      # cos(a, b) ~ 0.97
    c = [0.88, 0.475]      # cos(b, c) ~ 0.97, cos(a, c) ~ 0.88
    emb = _unit([a, b, c])
    cos_ac = float(emb[0] @ emb[2])
    assert cos_ac < 0.95, "the test's own premise: A and C are NOT neighbours"

    comp, _reps = dedup_components(emb)
    assert comp[0] == comp[1] == comp[2]


def test_dedup_representative_choice_is_deterministic():
    emb = _unit([[1, 0, 0], [1, 0, 0], [0, 1, 0]])
    first = dedup_components(emb)[1]
    second = dedup_components(emb)[1]
    assert np.array_equal(first, second)


# --------------------------------------------------------------------------- #
# Stratified split
# --------------------------------------------------------------------------- #
def test_split_is_disjoint_and_complete():
    labels = np.array(["a"] * 50 + ["b"] * 30 + ["c"] * 20)
    train, cal = stratified_split(labels, 0.6)
    assert len(set(train.tolist()) & set(cal.tolist())) == 0
    assert len(train) + len(cal) == len(labels)


def test_split_holds_the_fraction_within_one_row_per_class():
    labels = np.array(["a"] * 50 + ["b"] * 30 + ["c"] * 20)
    train, _cal = stratified_split(labels, 0.6)
    for value, total in (("a", 50), ("b", 30), ("c", 20)):
        got = int(np.sum(labels[train] == value))
        assert abs(got - int(total * 0.6)) <= 1


def test_split_is_reproducible_under_the_same_seed():
    labels = np.array(["a"] * 20 + ["b"] * 20)
    first = stratified_split(labels, 0.6, seed=42)
    second = stratified_split(labels, 0.6, seed=42)
    assert np.array_equal(first[0], second[0])
    assert np.array_equal(first[1], second[1])


def test_split_differs_under_a_different_seed():
    labels = np.array(["a"] * 40 + ["b"] * 40)
    a = stratified_split(labels, 0.6, seed=1)[0]
    b = stratified_split(labels, 0.6, seed=2)[0]
    assert not np.array_equal(a, b)


def test_a_class_too_small_to_split_still_reaches_calibration():
    """With one row, floor(1 * 0.6) = 0, so it must land in calibration.

    Silently dropping a rare class would shrink the label space the conformal
    quantile sees without any count changing visibly.
    """
    labels = np.array(["a"] * 10 + ["rare"])
    train, cal = stratified_split(labels, 0.6)
    assert "rare" in labels[cal].tolist()
    assert len(train) + len(cal) == len(labels)


# --------------------------------------------------------------------------- #
# Split integrity -- the guard that a row-disjoint split can still fail
# --------------------------------------------------------------------------- #
def test_verify_split_accepts_a_clean_split():
    comp = np.array([0, 1, 2, 3])
    verify_split(np.array([0, 1]), np.array([2, 3]), comp, 4)


def test_verify_split_rejects_a_component_spanning_the_split(monkeypatch):
    """Row-disjoint but component-shared: the failure the guard exists for."""
    comp = np.array([0, 0, 1, 2])   # rows 0 and 1 are near-duplicates
    with pytest.raises(SystemExit):
        verify_split(np.array([0]), np.array([1, 2, 3]), comp, 3)


def test_verify_split_rejects_shared_rows():
    comp = np.array([0, 1, 2, 3])
    with pytest.raises(SystemExit):
        verify_split(np.array([0, 1]), np.array([1, 2, 3]), comp, 4)


def test_verify_split_rejects_an_incomplete_cover():
    comp = np.array([0, 1, 2, 3])
    with pytest.raises(SystemExit):
        verify_split(np.array([0]), np.array([1]), comp, 4)


# --------------------------------------------------------------------------- #
# CSV writing -- ragged rows must survive
# --------------------------------------------------------------------------- #
def test_write_csv_keeps_rows_whose_keys_differ(tmp_path):
    """A BLOCKED Design B row carries fewer fields than a completed one.

    The first implementation raised on it, and the tempting fix -- writing
    only the first row's keys -- would have dropped the blocked and skipped
    configurations entirely. Those rows ARE the degeneracy rule's output.
    """
    path = os.path.join(str(tmp_path), "ragged.csv")
    rows = [
        {"queue": "Billing", "status": "BLOCKED_AUC_GE_0.95"},
        {"queue": "IT", "status": "ok", "coverage_gap": -0.02},
    ]
    write_csv(path, rows)

    with open(path, newline="", encoding="utf-8") as fh:
        out = list(csv.DictReader(fh))

    assert len(out) == 2
    assert out[0]["status"] == "BLOCKED_AUC_GE_0.95"
    assert out[0]["coverage_gap"] == ""
    assert out[1]["coverage_gap"] == "-0.02"


def test_write_csv_refuses_an_empty_result(tmp_path):
    with pytest.raises(SystemExit):
        write_csv(os.path.join(str(tmp_path), "empty.csv"), [])
