"""
Phase 6A tests: the deferral-rule risk-coverage comparison.

WHY THIS FILE EXISTS
--------------------
6A's whole result rests on one claim from algebra: that each deferral rule's
ranking can be reduced to a scalar built from the top two probabilities --

    confidence -> p1        lac -> -p2        aps -> p1 + p2

If that algebra is wrong, every curve, AURC and bootstrap interval in the phase
is wrong while remaining perfectly self-consistent, which is exactly this
project's recurring bug class. So the central test does not check the algebra
against itself: it brute-forces real prediction sets through
`conformal.predict_sets()` over a grid of quantiles and asserts the scalar rule
says "accept" **iff** the real set is a singleton.

Also pinned:

- **Ties.** A rule that cannot separate two tickets must not be credited as if
  it could. `risk_coverage_curve` cuts only between tie-groups; a rule that
  assigns every ticket the same score must collapse to a single operating
  point, not a flattering ordering produced by argsort's stability.
- **The oracle.** Excess AURC is measured against it, so an oracle that is not
  actually optimal would silently shift every excess figure.
- **The published endpoint.** At coverage 1.0 the risk must equal the plain
  error rate -- the independent recount that would catch probabilities which
  are not the ones the published results came from.

Offline: no artifacts, no models, no network. Pure functions over arrays.
"""

from __future__ import annotations

import numpy as np
import pytest

from src.agent import conformal
from src.experiments import compare_deferral_rules as dr


# --------------------------------------------------------------------------
# The central check: the scalar rules against real prediction sets
# --------------------------------------------------------------------------
def _random_probabilities(n, k, seed):
    rng = np.random.default_rng(seed)
    raw = rng.random((n, k)) ** 3      # skewed, so p1 varies widely
    return raw / raw.sum(axis=1, keepdims=True)


@pytest.mark.parametrize("score_function,rule", [("lac", "lac"),
                                                 ("aps", "aps")])
def test_scalar_rule_matches_brute_force_prediction_sets(score_function, rule):
    """accept(scalar) must equal |predict_sets(q)| == 1, for every q.

    This is the test that makes 6A trustworthy: it pins the module docstring's
    algebra against the real implementation rather than against my derivation.
    """
    classes = ["A", "B", "C", "D", "E", "F", "G"]
    probs = _random_probabilities(200, len(classes), seed=42)
    scores = dr.deferral_score(probs, rule)

    ordered = np.sort(probs, axis=1)[:, ::-1]
    p1, p2 = ordered[:, 0], ordered[:, 1]

    for q in np.linspace(0.0, 1.0, 101):
        calibration = conformal.ConformalCalibration(
            alpha=0.1, score_function=score_function, classes=classes,
            quantile=float(q), n_calibration=100,
        )
        sets = conformal.predict_sets(probs, calibration)
        actual_singleton = np.array([len(s) == 1 for s in sets])

        if rule == "lac":
            # singleton while p2 < 1-q <= p1  ->  scalar -p2 > -(1-q)
            predicted = (p2 < 1.0 - q) & (1.0 - q <= p1)
            assert np.array_equal(
                np.asarray(scores > -(1.0 - q)) & (1.0 - q <= p1),
                predicted), "scalar reconstruction disagrees at q={}".format(q)
        else:
            # singleton while p1 <= q < p1+p2
            predicted = (p1 <= q) & (q < p1 + p2)
            assert np.array_equal(
                (p1 <= q) & (q < np.asarray(scores)),
                predicted), "scalar reconstruction disagrees at q={}".format(q)

        assert np.array_equal(actual_singleton, predicted), (
            "predict_sets disagrees with the derived singleton condition "
            "at q={q} for {sf}".format(q=q, sf=score_function))


def test_the_singleton_region_is_an_interval_not_a_half_line():
    """The non-monotonicity the plan flagged: empty sets are deferrals too.

    If this ever became a half-line, sweeping alpha WOULD trace the curve
    monotonically and the write-up's caveat would be wrong.
    """
    classes = ["A", "B", "C"]
    probs = np.array([[0.7, 0.2, 0.1]])
    singleton_at = []
    for q in np.linspace(0.0, 1.0, 201):
        cal = conformal.ConformalCalibration(
            alpha=0.1, score_function="lac", classes=classes,
            quantile=float(q), n_calibration=100)
        singleton_at.append(len(conformal.predict_sets(probs, cal)[0]) == 1)

    # Empty at small q, singleton in the middle, larger at big q.
    assert not singleton_at[0], "expected an empty set at q=0"
    assert any(singleton_at), "expected a singleton region"
    assert not singleton_at[-1], "expected a non-singleton set at q=1"


# --------------------------------------------------------------------------
# Risk-coverage mechanics
# --------------------------------------------------------------------------
def test_perfect_ranking_has_zero_risk_until_the_errors_start():
    correct = np.array([True] * 8 + [False] * 2)
    scores = np.array([1.0] * 8 + [0.0] * 2) - np.arange(10) * 1e-6
    cov, risk, _ = dr.risk_coverage_curve(scores, correct)
    assert risk[0] == 0.0
    assert cov[-1] == pytest.approx(1.0)
    assert risk[-1] == pytest.approx(0.2)


def test_full_coverage_risk_equals_the_plain_error_rate():
    """The independent recount that guards the published endpoint."""
    rng = np.random.default_rng(7)
    correct = rng.random(45) < 0.7333          # ~33/45
    scores = rng.random(45)
    cov, risk, _ = dr.risk_coverage_curve(scores, correct)
    assert cov[-1] == pytest.approx(1.0)
    assert risk[-1] == pytest.approx(1.0 - correct.mean())


def test_a_rule_with_no_information_collapses_to_one_operating_point():
    """Ties must not hand a rule a free ordering."""
    correct = np.array([True, False, True, False])
    scores = np.zeros(4)
    cov, risk, _ = dr.risk_coverage_curve(scores, correct)
    assert len(cov) == 1
    assert cov[0] == pytest.approx(1.0)
    assert risk[0] == pytest.approx(0.5)


def test_oracle_aurc_is_optimal_and_beats_every_random_ranking():
    rng = np.random.default_rng(3)
    correct = rng.random(60) < 0.75
    oracle = dr.oracle_aurc(correct)
    for seed in range(20):
        s = np.random.default_rng(seed).random(60)
        cov, risk, _ = dr.risk_coverage_curve(s, correct)
        assert dr.aurc(cov, risk) >= oracle - 1e-12


def test_oracle_aurc_is_zero_when_everything_is_correct():
    assert dr.oracle_aurc(np.ones(10, dtype=bool)) == pytest.approx(0.0)


def test_risk_at_coverage_reports_the_coverage_actually_achieved():
    """At n=45 the grid is coarse; quoting an unachievable coverage would
    describe an operating point that does not exist."""
    cov = np.array([0.25, 0.50, 0.75, 1.00])
    risk = np.array([0.0, 0.1, 0.2, 0.3])
    r, actual = dr.risk_at_coverage(cov, risk, 0.60)
    assert actual == pytest.approx(0.75)
    assert r == pytest.approx(0.2)


# --------------------------------------------------------------------------
# The paired bootstrap
# --------------------------------------------------------------------------
def test_bootstrap_of_a_rule_against_itself_is_exactly_zero():
    rng = np.random.default_rng(11)
    correct = rng.random(50) < 0.7
    scores = rng.random(50)
    mean, lo, hi = dr.paired_bootstrap(scores, scores, correct, dr._aurc_stat,
                                       n=200)
    assert mean == pytest.approx(0.0)
    assert lo == pytest.approx(0.0)
    assert hi == pytest.approx(0.0)


def test_bootstrap_detects_a_genuinely_better_rule():
    correct = np.array([True] * 30 + [False] * 20)
    good = np.concatenate([np.linspace(1.0, 0.6, 30),
                           np.linspace(0.4, 0.0, 20)])
    bad = np.concatenate([np.linspace(0.0, 0.4, 30),
                          np.linspace(0.6, 1.0, 20)])
    mean, lo, hi = dr.paired_bootstrap(good, bad, correct, dr._aurc_stat,
                                       n=500)
    assert mean < 0            # lower AURC is better
    assert hi < 0              # interval excludes zero


def test_bootstrap_is_reproducible_under_the_fixed_seed():
    rng = np.random.default_rng(5)
    correct = rng.random(40) < 0.7
    a, b = rng.random(40), rng.random(40)
    first = dr.paired_bootstrap(a, b, correct, dr._aurc_stat, n=300)
    second = dr.paired_bootstrap(a, b, correct, dr._aurc_stat, n=300)
    assert first == second


# --------------------------------------------------------------------------
# The rules themselves
# --------------------------------------------------------------------------
def test_every_rule_is_a_function_of_the_top_two_probabilities_only():
    """Permuting the tail must not move any rule's score."""
    probs = np.array([[0.5, 0.3, 0.12, 0.08]])
    shuffled = np.array([[0.5, 0.3, 0.08, 0.12]])
    for rule in dr.RULES:
        assert dr.deferral_score(probs, rule) == pytest.approx(
            dr.deferral_score(shuffled, rule))


def test_confidence_and_lac_induce_different_orderings():
    """If they agreed, 6A would be comparing a rule against itself."""
    probs = np.array([
        [0.60, 0.39, 0.01],     # higher p1, but a close runner-up
        [0.55, 0.25, 0.20],     # lower  p1, but a clearly beaten runner-up
    ])
    conf = dr.deferral_score(probs, "confidence")
    lac = dr.deferral_score(probs, "lac")
    # Confidence prefers row 0 (0.60 > 0.55); LAC prefers row 1, because its
    # runner-up is further behind (-0.25 > -0.39). This is the whole reason
    # the two rules are worth comparing.
    assert np.argmax(conf) == 0
    assert np.argmax(lac) == 1


def test_unknown_rule_is_rejected():
    with pytest.raises(ValueError):
        dr.deferral_score(np.array([[0.5, 0.5]]), "not-a-rule")
