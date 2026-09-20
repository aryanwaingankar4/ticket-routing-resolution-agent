"""
Phase 5A regression tests for the template-contamination diagnostic.

WHY THIS FILE EXISTS
--------------------
`calibrate_conformal.report_contamination_structure()` is the measurement
behind Phase 1's Finding 2 ("the in-domain calibration set cannot be
de-contaminated"). It originally grouped rows by `scenario_id` ALONE, but
scenario_id is an index WITHIN a category -- data/generate_dataset.py:634 says
so in its own comment -- so the grouping merged seven categories' templates
into one and reported 12 templates of ~430 rows where there are 66 of ~62.

That is this project's recurring bug class in its purest form: a key that is
internally consistent, produces a plausible number, and is wrong for its
context. It ran clean for weeks and reached published results.

So these tests pin both halves:
  1. the FIX  -- templates are counted per (category, scenario_id);
  2. the BUG  -- and a scenario_id-only grouping would give a different,
     smaller answer, asserted explicitly so a revert fails the build rather
     than quietly restoring the old numbers.

Offline, no models, no quota.
"""

from __future__ import annotations

import json
import os

import pandas as pd
import pytest

from src.agent.config import settings
from src.experiments.calibrate_conformal import report_contamination_structure

CALIBRATION_JSON = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "data", "calibration_tickets_paraphrased.json")

# The corrected, published values (Phase 5A). Independently re-derived four
# ways for the template count and two ways each for the others before being
# written here.
EXPECTED_TEMPLATES = 66
EXPECTED_TOUCHED = 62
EXPECTED_MEDIAN_ROWS = 62
EXPECTED_SURVIVING = 210
EXPECTED_DATASET_ROWS = 4000


def _synthetic_frame():
    """Two categories that reuse the same scenario_id values.

    Four real templates: (A,0) (A,1) (B,0) (B,1). A scenario_id-only grouping
    sees two. That gap is the bug.
    """
    rows = []
    ticket_id = 0
    for category in ("Alpha", "Beta"):
        for scenario_id in (0, 1):
            for _ in range(5):
                rows.append({"id": ticket_id, "category": category,
                             "scenario_id": scenario_id})
                ticket_id += 1
    return pd.DataFrame(rows)


def test_templates_are_keyed_by_category_and_scenario_id():
    """The compound key is used, and the old key would disagree."""
    df = _synthetic_frame()
    # One calibration ticket, from (Alpha, 0) only.
    calibration = [{"id": int(df.iloc[0]["id"])}]

    result = report_contamination_structure(df, calibration)

    pairs = df.drop_duplicates(["category", "scenario_id"]).shape[0]
    assert pairs == 4
    assert result["total_templates"] == pairs
    assert result["template_key"] == "category+scenario_id"

    # The bug, pinned: scenario_id alone collapses Alpha's and Beta's
    # templates together. If anyone reverts the grouping, total_templates
    # becomes this number and the assertion above fails -- but assert the
    # inequality too, so the intent survives even if the fixture changes.
    assert df["scenario_id"].nunique() == 2
    assert result["total_templates"] > df["scenario_id"].nunique()

    # Only (Alpha, 0) is touched, so its 5 rows are excluded and the other
    # three templates' 15 rows survive. Under the old key, (Beta, 0) would
    # have been wrongly excluded too, leaving 10.
    assert result["calibration_templates"] == 1
    assert result["rows_surviving_template_exclusion"] == 15


def test_touched_templates_never_span_categories():
    """A calibration ticket must not mark another category's template.

    This is the specific failure the old key produced: a ticket from
    (Alpha, 0) also excluded (Beta, 0), because they share a scenario_id.
    """
    df = _synthetic_frame()
    calibration = [{"id": int(df.iloc[0]["id"])}]

    result = report_contamination_structure(df, calibration)

    beta_rows = int((df["category"] == "Beta").sum())
    surviving = result["rows_surviving_template_exclusion"]
    assert surviving >= beta_rows, (
        "every Beta row must survive exclusion of an Alpha template; "
        "a lower count means the grouping is merging categories")


@pytest.mark.skipif(not os.path.isfile(CALIBRATION_JSON),
                    reason="175-ticket calibration set not present")
def test_published_diagnostic_numbers_on_the_real_dataset():
    """Pin Phase 1 Finding 2's corrected diagnostic, as test_config pins
    calibrated constants: if a published number drifts, the build fails."""
    df = pd.read_csv(settings.models.dataset_path)
    with open(CALIBRATION_JSON, encoding="utf-8") as fh:
        calibration = json.load(fh)

    result = report_contamination_structure(df, calibration)

    assert result == {
        "template_key": "category+scenario_id",
        "total_templates": EXPECTED_TEMPLATES,
        "calibration_templates": EXPECTED_TOUCHED,
        "median_rows_per_template": EXPECTED_MEDIAN_ROWS,
        "rows_surviving_template_exclusion": EXPECTED_SURVIVING,
        "dataset_rows": EXPECTED_DATASET_ROWS,
    }

    # The conclusion the numbers carry: template-level exclusion destroys the
    # training set (210 of 4000 rows), while row-level exclusion leaves the
    # template essentially intact (~61 siblings, ~1.6% removed).
    assert result["rows_surviving_template_exclusion"] < 0.1 * len(df)
    assert result["median_rows_per_template"] - 1 >= 60
