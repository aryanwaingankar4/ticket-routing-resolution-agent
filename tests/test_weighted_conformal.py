"""
Weighted split conformal (Phase 6B) correctness.

The headline test here is `test_uniform_weights_reproduce_the_order_statistic`.
Phase 6B's whole result rests on comparing a weighted quantile against the
published UNWEIGHTED one, so if the weighted path did not collapse to the
unweighted path at uniform weights, every "weighting changed the coverage"
reading would be confounded with an implementation difference. The equality is
exact rather than approximate by construction -- both sides reduce to the rank
ceil((n+1)(1-alpha)) computed from the same float expression -- so it is
asserted with `==`, not `pytest.approx`.

The second load-bearing test is `test_test_point_atom_is_not_dropped`. Omitting
the test point's own weight from the normalisation is the standard
implementation error in weighted conformal: it shrinks q_hat, shrinks the sets,
and pushes coverage below nominal -- failing silently in the reassuring
direction, exactly like this project's other conformal failure mode.

These tests are fast: pure numpy, no models, no marker needed.
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from src.agent import conformal as cp

CLASSES = ["A", "B", "C", "D"]
RNG_SEED = 42
ALPHAS = [0.20, 0.10, 0.05, 0.01]


# --------------------------------------------------------------------------- #
# The equivalence Phase 6B's comparison depends on
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("alpha", ALPHAS)
@pytest.mark.parametrize("n", [19, 45, 100, 175, 176, 500])
def test_uniform_weights_reproduce_the_order_statistic(alpha, n):
    """Uniform weights must give the unweighted quantile EXACTLY.

    Not approximately. 6B reads the difference between these two numbers as
    its result, so any gap here would be indistinguishable from a finding.
    """
    rng = np.random.default_rng(RNG_SEED)
    scores = rng.random(n)

    unweighted = cp.conformal_quantile(scores, alpha)
    weighted = cp.weighted_conformal_quantile(
        scores, np.ones(n), alpha, test_weight=1.0)

    assert weighted == unweighted


@pytest.mark.parametrize("alpha", ALPHAS)
def test_uniform_weights_agree_on_the_degenerate_case(alpha):
    """When n is too small to certify alpha, both paths must return +inf.

    The weighted path reaches it a different way -- the calibration mass never
    reaches 1 - alpha, rather than a rank exceeding n -- so the agreement is
    worth pinning.
    """
    n = cp.min_calibration_size(alpha) - 1
    if n < 1:
        pytest.skip(f"alpha={alpha} admits no n below the minimum")

    rng = np.random.default_rng(RNG_SEED)
    scores = rng.random(n)

    assert not math.isfinite(cp.conformal_quantile(scores, alpha))
    assert not math.isfinite(
        cp.weighted_conformal_quantile(scores, np.ones(n), alpha, 1.0))


def test_constant_nonunit_weights_also_reproduce_the_order_statistic():
    """Only the RELATIVE weights matter; a constant rescaling changes nothing.

    w_i = w_test = 7.5 is still uniform, so the quantile must not move.
    """
    rng = np.random.default_rng(RNG_SEED)
    scores = rng.random(175)

    for alpha in ALPHAS:
        expected = cp.conformal_quantile(scores, alpha)
        got = cp.weighted_conformal_quantile(
            scores, np.full(175, 7.5), alpha, test_weight=7.5)
        assert got == expected


# --------------------------------------------------------------------------- #
# The atom at infinity
# --------------------------------------------------------------------------- #
def test_test_point_atom_is_not_dropped():
    """A larger test weight must push q_hat UP, never leave it unchanged.

    The test point's weight sits on an atom at +infinity, so raising it raises
    the total the cumulative calibration mass is measured against, which can
    only move the selected score later in the sort. Dropping the atom is the
    classic bug; this is the test that would catch it.
    """
    rng = np.random.default_rng(RNG_SEED)
    scores = np.sort(rng.random(100))
    weights = np.ones(100)

    small = cp.weighted_conformal_quantile(scores, weights, 0.10, 0.1)
    large = cp.weighted_conformal_quantile(scores, weights, 0.10, 50.0)

    assert large >= small
    assert large > small, "the atom at +inf is being ignored"


def test_test_weight_must_be_finite_and_non_negative():
    scores = np.linspace(0.0, 1.0, 50)
    weights = np.ones(50)

    for bad in (-1.0, float("inf"), float("nan")):
        with pytest.raises(ValueError):
            cp.weighted_conformal_quantile(scores, weights, 0.10, bad)


# --------------------------------------------------------------------------- #
# Weighting does what it is supposed to do
# --------------------------------------------------------------------------- #
def test_upweighting_high_scores_raises_the_quantile():
    """Concentrating weight on the nonconforming tail must widen the sets.

    This is the mechanism 6B is testing: if the target distribution looks like
    the hard end of the calibration set, the weighted quantile should rise to
    cover it.
    """
    scores = np.linspace(0.0, 1.0, 200)

    uniform = cp.weighted_conformal_quantile(
        scores, np.ones(200), 0.10, 1.0)
    tail_heavy = cp.weighted_conformal_quantile(
        scores, np.where(scores > 0.8, 10.0, 1.0), 0.10, 1.0)

    assert tail_heavy > uniform


def test_zero_weight_points_are_ignored():
    """A zero-weighted calibration point must not influence the quantile.

    Equivalent to having dropped it, which is what a density ratio of 0 means.
    """
    scores = np.array([0.1, 0.2, 0.3, 0.4, 0.5, 0.95, 0.99])
    weights = np.array([1.0, 1.0, 1.0, 1.0, 1.0, 0.0, 0.0])

    with_zeros = cp.weighted_conformal_quantile(scores, weights, 0.20, 1.0)
    without = cp.weighted_conformal_quantile(
        scores[:5], np.ones(5), 0.20, 1.0)

    assert with_zeros == without


def test_ties_in_scores_are_handled_as_one_block():
    """Tied scores contribute their whole mass before the comparison."""
    scores = np.array([0.5, 0.5, 0.5, 0.5, 0.9])
    q = cp.weighted_conformal_quantile(scores, np.ones(5), 0.20, 1.0)
    assert q == 0.9


# --------------------------------------------------------------------------- #
# Effective sample size
# --------------------------------------------------------------------------- #
def test_effective_sample_size_equals_n_for_uniform_weights():
    for n in (10, 175, 1000):
        assert cp.effective_sample_size(np.ones(n)) == pytest.approx(float(n))
        # A constant rescaling must not change it either.
        assert cp.effective_sample_size(np.full(n, 3.7)) == pytest.approx(
            float(n))


def test_effective_sample_size_collapses_under_a_dominant_weight():
    w = np.ones(175)
    w[0] = 10_000.0
    assert cp.effective_sample_size(w) < 2.0


def test_effective_sample_size_is_the_kish_formula():
    """Second, independent derivation -- rule 6 applied to a helper.

    Computed here from the mean and variance identity
    n_eff = n / (1 + CV^2) rather than from (sum w)^2 / sum(w^2).
    """
    rng = np.random.default_rng(RNG_SEED)
    w = rng.lognormal(mean=0.0, sigma=0.8, size=500)

    n = w.size
    cv_squared = float(np.var(w) / (np.mean(w) ** 2))
    expected = n / (1.0 + cv_squared)

    assert cp.effective_sample_size(w) == pytest.approx(expected, rel=1e-9)


def test_effective_sample_size_rejects_empty_and_negative():
    with pytest.raises(ValueError):
        cp.effective_sample_size([])
    with pytest.raises(ValueError):
        cp.effective_sample_size([1.0, -1.0])
    with pytest.raises(ValueError):
        cp.effective_sample_size([0.0, 0.0])


# --------------------------------------------------------------------------- #
# Prediction sets
# --------------------------------------------------------------------------- #
def _synthetic(n, n_classes=4, difficulty=2.0, seed=RNG_SEED):
    rng = np.random.default_rng(seed)
    logits = rng.normal(scale=difficulty, size=(n, n_classes))
    exp = np.exp(logits - logits.max(axis=1, keepdims=True))
    probs = exp / exp.sum(axis=1, keepdims=True)
    labels = [CLASSES[rng.choice(n_classes, p=probs[i])] for i in range(n)]
    return probs, labels


@pytest.mark.parametrize("score_function", ["lac", "aps"])
def test_uniform_weighted_sets_match_unweighted_sets(score_function):
    """The whole weighted path, not just the quantile, must collapse.

    6B's degeneracy rule compares weighted against unweighted prediction sets,
    so the two must be identical at uniform weights or that comparison is
    meaningless.
    """
    cal_probs, cal_labels = _synthetic(175, seed=1)
    test_probs, _ = _synthetic(45, seed=2)

    calibration = cp.calibrate(cal_probs, cal_labels, CLASSES, alpha=0.10,
                               score_function=score_function)
    unweighted = cp.predict_sets(test_probs, calibration)

    cal_scores = cp.true_label_scores(
        cal_probs, cal_labels, CLASSES, score_function)
    weighted = cp.weighted_predict_sets(
        test_probs, cal_scores, np.ones(175), np.ones(45), 0.10, CLASSES,
        score_function=score_function)

    assert weighted == unweighted


def test_true_label_scores_match_what_calibrate_uses():
    """Independent check that the exposed scores are the calibrated ones.

    If these diverged, the weighted quantile would be taken over a different
    score population than the published unweighted one -- this project's
    recurring bug class, in its conformal form.
    """
    cal_probs, cal_labels = _synthetic(200, seed=7)

    for score_function in ("lac", "aps"):
        scores = cp.true_label_scores(
            cal_probs, cal_labels, CLASSES, score_function)
        calibration = cp.calibrate(cal_probs, cal_labels, CLASSES, alpha=0.10,
                                   score_function=score_function)
        assert cp.conformal_quantile(scores, 0.10) == calibration.quantile


def test_weighted_sets_preserve_empty_sets():
    """An empty set stays empty, for the same reason as the unweighted path."""
    probs = np.array([[0.25, 0.25, 0.25, 0.25]])
    cal_scores = np.zeros(50)  # every calibration point perfectly conforming

    sets = cp.weighted_predict_sets(
        probs, cal_scores, np.ones(50), np.ones(1), 0.10, CLASSES)

    assert sets == [[]]


def test_weighted_sets_validate_their_shapes():
    probs, labels = _synthetic(10)
    cal_scores = np.zeros(50)

    with pytest.raises(ValueError):
        cp.weighted_predict_sets(
            probs, cal_scores, np.ones(50), np.ones(3), 0.10, CLASSES)
    with pytest.raises(ValueError):
        cp.weighted_predict_sets(
            probs, cal_scores, np.ones(49), np.ones(10), 0.10, CLASSES)


# --------------------------------------------------------------------------- #
# The guarantee, on data where the shift is known
# --------------------------------------------------------------------------- #
def test_vectorised_quantiles_match_the_scalar_function():
    """The batched path must equal the scalar one EXACTLY.

    `weighted_predict_sets` sorts once and reuses the cumulative sum rather
    than recomputing a quantile per test point. That is an arithmetic identity,
    not an approximation -- and a silent divergence between two internally
    consistent implementations of the same quantity is this project's recurring
    bug class, so it is pinned rather than assumed.
    """
    rng = np.random.default_rng(RNG_SEED)
    scores = rng.random(175)
    weights = rng.lognormal(0.0, 1.0, 175)
    test_weights = rng.lognormal(0.0, 1.0, 45)

    for alpha in ALPHAS:
        batched = cp._weighted_quantiles(scores, weights, alpha, test_weights)
        scalar = [
            cp.weighted_conformal_quantile(scores, weights, alpha, float(t))
            for t in test_weights
        ]
        assert list(batched) == scalar


def test_weighting_recovers_coverage_under_a_known_covariate_shift():
    """The property 6B is testing for, on data where it provably holds.

    A genuine covariate shift: P(Y|X) is IDENTICAL in both draws and only P(X)
    moves, which is exactly the condition weighted conformal is proved under.
    The model never sees x and always predicts class A confidently, while the
    probability that the truth is NOT class A rises with x. Calibration is
    drawn at x ~ N(0,1) and test at x ~ N(1,1), so the model is confidently
    wrong far more often on the test draw and unweighted conformal undercovers.
    The true density ratio exp(x - 1/2) is supplied.

    This is the analogue of test_conformal.py's exchangeable-data test. Without
    it, a null result on the real data could not be told apart from a broken
    implementation -- which is the distinction 6B's entire verdict rests on.
    """
    rng = np.random.default_rng(RNG_SEED)
    n, mu, alpha = 6000, 1.0, 0.10

    x_cal = rng.normal(0.0, 1.0, n)
    x_test = rng.normal(mu, 1.0, n)

    def draw(x):
        # The model is blind to x: it always says class A, confidently.
        logits = rng.normal(scale=0.3, size=(x.size, 4))
        logits[:, 0] += 3.0
        exp = np.exp(logits - logits.max(axis=1, keepdims=True))
        probs = exp / exp.sum(axis=1, keepdims=True)
        # P(Y|X), fixed across both draws.
        p_not_a = 1.0 / (1.0 + np.exp(-(x - 1.0)))
        idx = np.where(rng.random(x.size) < p_not_a,
                       rng.integers(1, 4, x.size), 0)
        return probs, [CLASSES[i] for i in idx]

    cal_probs, cal_labels = draw(x_cal)
    test_probs, test_labels = draw(x_test)

    cal_scores = cp.true_label_scores(cal_probs, cal_labels, CLASSES, "lac")

    unweighted_cov = cp.coverage(
        cp.predict_sets(
            test_probs,
            cp.calibrate(cal_probs, cal_labels, CLASSES, alpha=alpha)),
        test_labels)

    w_cal = np.exp(mu * x_cal - mu * mu / 2.0)
    w_test = np.exp(mu * x_test - mu * mu / 2.0)
    weighted_cov = cp.coverage(
        cp.weighted_predict_sets(
            test_probs, cal_scores, w_cal, w_test, alpha, CLASSES),
        test_labels)

    # The shift must actually bite, or the test proves nothing.
    assert unweighted_cov < 1 - alpha - 0.02, (
        f"shift did not bite: unweighted coverage {unweighted_cov:.3f}")
    # And reweighting must repair most of it. The residual gap is the known
    # finite-sample cost of weighting (Barber et al. 2022), not a bug.
    assert weighted_cov > unweighted_cov + 0.03, (
        f"weighting did not help: {unweighted_cov:.3f} -> {weighted_cov:.3f}")
    assert weighted_cov >= 1 - alpha - 0.025, (
        f"weighted coverage {weighted_cov:.3f} did not recover")
