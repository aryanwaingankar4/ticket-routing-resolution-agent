"""
Tests for src/experiments/compare_gate_csv.py -- the cross-platform gate-CSV
comparison gates.yml uses in place of byte identity on a Linux runner.

Fast: reads the two committed CSVs and edits copies in memory. No model load.

The cases that matter most:
  * the ACTUAL runner failure (one unit in the 6th decimal of rag_similarity
    on adv_01/02/04/05/07) must pass -- that is the false alarm being removed;
  * any change to a decision column must still fail -- a tolerance must never
    be able to hide a flipped decision;
  * a real similarity move well past float32 must still fail.
"""

import csv
import io
import os

import pytest

from src.experiments import compare_gate_csv as cg

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ADV = "adversarial_escalation_results.csv"
ABL = "ablation_baseline_results.csv"


def _text(name):
    with open(os.path.join(ROOT, "data", name), encoding="utf-8",
              newline="") as fh:
        return fh.read()


def _edit(text, fn):
    """Apply fn(rows) to the parsed rows and re-serialise as 6-dp CSV text."""
    rows = list(csv.DictReader(io.StringIO(text.replace("\r\n", "\n"))))
    header = list(rows[0].keys())
    fn(rows)
    out = io.StringIO()
    writer = csv.DictWriter(out, fieldnames=header, lineterminator="\n")
    writer.writeheader()
    writer.writerows(rows)
    return out.getvalue()


def _bump(value, units):
    return f"{float(value) + units * 1e-6:.6f}"


@pytest.mark.parametrize("name", [ADV, ABL])
def test_committed_csv_matches_itself_exactly(name):
    text = _text(name)
    result = cg.compare(text, text, name)
    assert result["failures"] == []
    assert result["bytes_identical"]
    assert all(d == 0.0 for d, _ in result["max_delta"].values())


@pytest.mark.parametrize("name", [ADV, ABL])
def test_schema_names_real_columns(name):
    header = next(csv.reader(io.StringIO(_text(name))))
    schema = cg.SCHEMAS[name]
    for col in list(schema["floats"]) + list(schema["key"]):
        assert col in header


def test_the_actual_runner_failure_now_passes():
    ids = {"adv_01", "adv_02", "adv_04", "adv_05", "adv_07"}

    def fn(rows):
        for r in rows:
            if r["id"] in ids:
                r["rag_similarity"] = _bump(r["rag_similarity"], +1)

    text = _text(ADV)
    result = cg.compare(text, _edit(text, fn), ADV)
    assert result["failures"] == []
    assert not result["bytes_identical"]
    delta, _ = result["max_delta"]["rag_similarity"]
    assert delta == pytest.approx(1e-6, abs=1e-12)


def test_tier1_conf_straddling_a_rounding_boundary_passes():
    # One unit in the 6th place is what two roundings of the same float64
    # value can produce; 5e-7 alone would fail this.
    def fn(rows):
        rows[0]["tier1_confidence"] = _bump(rows[0]["tier1_confidence"], +1)

    text = _text(ADV)
    assert cg.compare(text, _edit(text, fn), ADV)["failures"] == []


def test_tier1_conf_two_units_off_fails():
    def fn(rows):
        rows[0]["tier1_confidence"] = _bump(rows[0]["tier1_confidence"], +2)

    text = _text(ADV)
    assert len(cg.compare(text, _edit(text, fn), ADV)["failures"]) == 1


def test_similarity_beyond_float32_bound_fails():
    def fn(rows):
        rows[3]["rag_similarity"] = _bump(rows[3]["rag_similarity"], +100)

    text = _text(ADV)
    assert len(cg.compare(text, _edit(text, fn), ADV)["failures"]) == 1


@pytest.mark.parametrize("column,new_value", [
    ("pass_fail", "FAIL"),
    ("actual_escalate", "False"),
    ("predicted_category", "Network"),
    ("id", "adv_99"),
])
def test_any_decision_or_id_change_fails(column, new_value):
    def fn(rows):
        rows[0][column] = new_value

    text = _text(ADV)
    assert cg.compare(text, _edit(text, fn), ADV)["failures"]


@pytest.mark.parametrize("column,new_value", [
    ("predicted", "Network"),
    ("correct", "False"),
    ("tier_used", "1"),
])
def test_ablation_decision_change_fails(column, new_value):
    def fn(rows):
        rows[0][column] = new_value

    text = _text(ABL)
    assert cg.compare(text, _edit(text, fn), ABL)["failures"]


def test_ablation_similarity_one_unit_passes():
    def fn(rows):
        for r in rows:
            if r["top_similarity"]:
                r["top_similarity"] = _bump(r["top_similarity"], -1)

    text = _text(ABL)
    assert cg.compare(text, _edit(text, fn), ABL)["failures"] == []


def test_row_order_and_count_are_exact():
    text = _text(ADV)
    swapped = _edit(text, lambda rows: rows.reverse())
    assert cg.compare(text, swapped, ADV)["failures"]
    dropped = _edit(text, lambda rows: rows.pop())
    assert cg.compare(text, dropped, ADV)["failures"]


def test_empty_numeric_cell_must_be_empty_on_both_sides():
    def fn(rows):
        rows[0]["top_similarity"] = "0.500000"  # classification row: empty

    text = _text(ABL)
    assert cg.compare(text, _edit(text, fn), ABL)["failures"]


def test_crlf_only_difference_is_identical():
    text = _text(ADV).replace("\r\n", "\n")
    result = cg.compare(text, text.replace("\n", "\r\n"), ADV)
    assert result["failures"] == [] and result["bytes_identical"]


def test_unknown_file_is_refused():
    with pytest.raises(cg.CompareError):
        cg.compare("a\n1\n", "a\n1\n", "some_other.csv")


def test_float32_bound_agrees_with_the_other_two_copies():
    # gamma_n is written down in two other places; the three must agree, or one
    # of them has drifted.
    from src.service.verify_deployment import FLOAT32_DOT_BOUND
    from tests.test_pipeline_parity import TOL_FLOAT32

    assert cg.FLOAT32_DOT_BOUND == pytest.approx(4.577846e-05, abs=1e-11)
    assert cg.FLOAT32_DOT_BOUND == pytest.approx(FLOAT32_DOT_BOUND, abs=1e-11)
    assert cg.FLOAT32_DOT_BOUND == pytest.approx(TOL_FLOAT32, abs=1e-11)
