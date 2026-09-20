"""
Phase 5B tests: the tier2-only ablation mode, the second evaluation set, and
the paired cascade-vs-Tier-2 comparison.

WHY THIS FILE EXISTS
--------------------
`--mode tier2-only` is implemented as a cascade threshold set ABOVE any
attainable Tier-1 confidence, so that run_cascade routes every ticket to
Tier-2 without a second copy of the routing logic existing anywhere. That is
efficient and it is also exactly this project's recurring bug class waiting to
happen: a sentinel value that is internally consistent, produces a plausible
accuracy, and would be silently wrong if the comparison in run_cascade ever
changed from `<` to `<=`, or if Tier-1 confidences stopped being bounded by 1.

So the sentinel is pinned from both ends here:
  1. the threshold genuinely exceeds every attainable Tier-1 confidence, and
  2. run_cascade really does route 100% of tickets to Tier-2 at that value,
     asserted against a real (tiny) Tier-1 rather than by reading the code.

The deployment175 loader and the McNemar analysis carry the same shape of
risk -- a wrong-but-plausible count -- so their failure modes are pinned too.

Offline: no BGE, no FAISS, no Gemini. The one Tier-1 fitted here is a 20-row
toy, not the production artifact, because the point is the ROUTING RULE and
not the model.
"""

from __future__ import annotations

import csv
import json
import os

import numpy as np
import pytest

from src.classification.train_cascade import run_cascade
from src.experiments import compare_cascade_vs_tier2 as cmp_mod
from src.experiments import run_ablation_study as abl


# --------------------------------------------------------------------------
# Tiny stand-ins, so the routing rule can be tested without production models
# --------------------------------------------------------------------------
def _toy_tier1():
    """A real TF-IDF + LogReg pair, fitted on 20 toy rows in milliseconds."""
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.linear_model import LogisticRegression

    texts, labels = [], []
    for _ in range(10):
        texts.append("vpn connection drops at the office network")
        labels.append("Network")
        texts.append("database query is slow and the table is locked")
        labels.append("Database")

    vec = TfidfVectorizer()
    X = vec.fit_transform(texts)
    clf = LogisticRegression(max_iter=1000)
    clf.fit(X, labels)
    return vec, clf


class _StubTier2:
    """Always answers 'Storage', so a Tier-2 answer is unmistakable."""

    def predict(self, embeddings):
        return np.array(["Storage"] * len(embeddings))


class _StubEmbedder:
    def encode(self, texts, **kwargs):
        return np.zeros((len(texts), 4), dtype=np.float32)


TOY_TEXTS = [
    "vpn connection drops at the office network",
    "database query is slow and the table is locked",
    "something entirely unlike either training example whatsoever",
]


# --------------------------------------------------------------------------
# The sentinel threshold
# --------------------------------------------------------------------------
def test_tier2_only_threshold_exceeds_any_attainable_confidence():
    """Tier-1 confidence is a max over a predict_proba row, so it is <= 1.0.

    The sentinel must sit strictly above that ceiling, or tier2-only would
    silently let a maximally-confident Tier-1 keep a ticket.
    """
    assert abl.TIER2_ONLY_THRESHOLD > 1.0


def test_tier2_only_routes_every_ticket_to_tier2():
    """The routing rule itself, against a real Tier-1 -- not by reading code."""
    vec, clf = _toy_tier1()
    result = run_cascade(
        TOY_TEXTS, vec, clf, _StubTier2(), _StubEmbedder(),
        abl.TIER2_ONLY_THRESHOLD,
    )

    assert result["n_tier2"] == len(TOY_TEXTS)
    assert result["n_tier1"] == 0
    assert result["tiers"] == [2] * len(TOY_TEXTS)
    # Every answer is the stub's, so nothing leaked through from Tier-1.
    assert list(result["preds"]) == ["Storage"] * len(TOY_TEXTS)


def test_no_cascade_threshold_routes_nothing_to_tier2():
    """The opposite end of the same rule, so the pair is pinned together."""
    vec, clf = _toy_tier1()
    result = run_cascade(
        TOY_TEXTS, vec, clf, _StubTier2(), _StubEmbedder(), 0.0,
    )
    assert result["n_tier2"] == 0
    assert result["tiers"] == [1] * len(TOY_TEXTS)
    assert "Storage" not in list(result["preds"])


def test_tier2_only_guard_refuses_a_partial_routing(monkeypatch):
    """If the sentinel ever stopped working, the mode must FAIL, not report.

    A wrong-but-plausible accuracy is the failure this project keeps hitting,
    so run_tier2_only asserts rather than warns.
    """
    rows = [
        {"index": 0, "text": "a", "expected": "Network",
         "predicted": "Network", "tier1_conf": 0.9, "tier_used": 1,
         "correct": True},
        {"index": 1, "text": "b", "expected": "Storage",
         "predicted": "Storage", "tier1_conf": 0.1, "tier_used": 2,
         "correct": True},
    ]
    monkeypatch.setattr(
        abl, "evaluate_classification",
        lambda *a, **k: (rows, 1.0, 2, 2, 1),
    )

    class _Art:
        tier1_vectorizer = tier1_classifier = tier2_classifier = None
        embedder = None

    with pytest.raises(SystemExit):
        abl.run_tier2_only({}, _Art())


# --------------------------------------------------------------------------
# Output filenames -- the published CSVs must never be the target
# --------------------------------------------------------------------------
def test_benchmark45_keeps_the_historical_filenames(tmp_path, monkeypatch):
    monkeypatch.setattr(abl, "DATA_DIR", str(tmp_path))
    path = abl.write_csv("baseline", ["index"], [{"index": 0}])
    assert os.path.basename(path) == "ablation_baseline_results.csv"


def test_other_sets_get_their_own_filenames(tmp_path, monkeypatch):
    """Rule 4: a new result gets a NEW filename, never an overwrite."""
    monkeypatch.setattr(abl, "DATA_DIR", str(tmp_path))
    path = abl.write_csv(
        "baseline", ["index"], [{"index": 0}], eval_set="deployment175"
    )
    assert os.path.basename(path) == "ablation_baseline_results_deployment175.csv"


def test_modes_and_sets_are_registered():
    assert "tier2-only" in abl.VALID_MODES
    assert set(abl.VALID_SETS) == {"benchmark45", "deployment175"}
    assert abl.DEFAULT_EVAL_SET == "benchmark45"


# --------------------------------------------------------------------------
# The deployment175 loader
# --------------------------------------------------------------------------
def test_deployment175_loads_the_real_set():
    records = abl.load_deployment175_set()
    assert len(records) == 175
    assert all(r["text"].strip() and r["expected"].strip() for r in records)


def _write_json(tmp_path, payload):
    path = tmp_path / "deployment_calibration_tickets.json"
    path.write_text(json.dumps(payload), encoding="utf-8")
    return str(path)


def test_deployment175_rejects_a_wrong_length_set(tmp_path, monkeypatch):
    payload = [{"text": "t", "expected": "Network"}] * 174
    monkeypatch.setattr(
        abl, "DEPLOYMENT175_JSON_PATH", _write_json(tmp_path, payload)
    )
    with pytest.raises(SystemExit):
        abl.load_deployment175_set()


def test_deployment175_rejects_a_malformed_record(tmp_path, monkeypatch):
    """A blank label must fail loudly, not be silently dropped or scored."""
    payload = [{"text": "t", "expected": "Network"} for _ in range(175)]
    payload[7] = {"text": "t", "expected": "   "}
    monkeypatch.setattr(
        abl, "DEPLOYMENT175_JSON_PATH", _write_json(tmp_path, payload)
    )
    with pytest.raises(SystemExit):
        abl.load_deployment175_set()


def test_no_rag_refuses_an_evaluation_set():
    """--mode no-rag never runs classification, so --set would be a lie."""
    with pytest.raises(SystemExit):
        abl.main(["--mode", "no-rag", "--set", "deployment175"])


# --------------------------------------------------------------------------
# The paired comparison
# --------------------------------------------------------------------------
def _pair(index, expected, cascade_pred, tier2_pred, tier_used, conf=0.6):
    casc = {"index": index, "text": "t%d" % index, "expected": expected,
            "predicted": cascade_pred, "tier1_conf": conf,
            "tier_used": tier_used, "correct": cascade_pred == expected}
    t2 = {"index": index, "text": "t%d" % index, "expected": expected,
          "predicted": tier2_pred, "tier1_conf": conf, "tier_used": 2,
          "correct": tier2_pred == expected}
    return casc, t2


def test_mcnemar_matches_a_hand_computed_table():
    """b=1, c=3 -> two-sided exact binomial p = 2 * P(X <= 1 | n=4, 0.5).

    P(X<=1) = (1 + 4) / 16 = 5/16, so p = 10/16 = 0.625.
    """
    from scipy.stats import binomtest

    pairs = [
        _pair(0, "Network", "Network", "Storage", 1),      # b
        _pair(1, "Network", "Storage", "Network", 1),      # c
        _pair(2, "Network", "Storage", "Network", 1),      # c
        _pair(3, "Network", "Storage", "Network", 1),      # c
        _pair(4, "Network", "Network", "Network", 2),      # both right
        _pair(5, "Network", "Storage", "Storage", 2),      # both wrong
    ]
    res = cmp_mod.analyse(pairs, binomtest)

    assert (res["b"], res["c"]) == (1, 3)
    assert res["both_right"] == 1 and res["both_wrong"] == 1
    assert res["p_value"] == pytest.approx(0.625)


def test_mcnemar_reports_one_when_nothing_is_discordant():
    from scipy.stats import binomtest

    pairs = [_pair(i, "Network", "Network", "Network", 2) for i in range(5)]
    res = cmp_mod.analyse(pairs, binomtest)
    assert (res["b"], res["c"]) == (0, 0)
    assert res["p_value"] == 1.0


def test_a_discordant_pair_answered_by_tier2_is_impossible():
    """Structural check: if the cascade escalated, both runs used the SAME
    Tier-2 prediction, so they cannot disagree. A CSV that says otherwise is
    stale, and must fail rather than produce a number."""
    from scipy.stats import binomtest

    pairs = [_pair(0, "Network", "Storage", "Network", 2)]
    with pytest.raises(SystemExit):
        cmp_mod.analyse(pairs, binomtest)


def test_pairing_is_rejected_when_the_tickets_disagree():
    """Matching indices are not proof the two CSVs describe the same tickets."""
    casc = [{"index": 0, "text": "one", "expected": "Network",
             "predicted": "Network", "tier1_conf": 0.6, "tier_used": 1,
             "correct": True}]
    t2 = [{"index": 0, "text": "DIFFERENT", "expected": "Network",
           "predicted": "Network", "tier1_conf": 0.6, "tier_used": 2,
           "correct": True}]
    with pytest.raises(SystemExit):
        cmp_mod.pair_rows(casc, t2)


def test_internally_inconsistent_csv_is_refused(tmp_path, monkeypatch):
    """The 'correct' column is recomputed from predicted vs expected."""
    monkeypatch.setattr(cmp_mod, "DATA_DIR", str(tmp_path))
    monkeypatch.setitem(cmp_mod.EVAL_SET_SIZES, "benchmark45", 1)

    path = tmp_path / "ablation_baseline_results.csv"
    with open(path, "w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(
            fh,
            fieldnames=["index", "text", "expected", "predicted",
                        "tier1_conf", "tier_used", "correct"],
        )
        writer.writeheader()
        writer.writerow({"index": 0, "text": "t", "expected": "Network",
                         "predicted": "Storage", "tier1_conf": 0.6,
                         "tier_used": 1, "correct": True})

    with pytest.raises(SystemExit):
        cmp_mod.load_classification_rows("baseline", "benchmark45")
