"""
Conformal prediction correctness.

The headline test here is `test_coverage_holds_on_exchangeable_data`: it
verifies the actual mathematical guarantee on synthetic data drawn so that
exchangeability provably holds. If that fails, the implementation is wrong --
independently of any dataset, model, or threshold in this project.

That distinction matters because on the REAL data coverage is expected to
come in below nominal (the calibration set is paraphrases of training rows,
and the deployment benchmark is a different distribution). Without a test on
data where the assumption genuinely holds, there would be no way to tell an
implementation bug apart from the exchangeability violation we are trying to
measure.

These tests are fast: pure numpy, no models, no marker needed.
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from src.agent import conformal as cp

CLASSES = ["A", "B", "C", "D"]
RNG_SEED = 42


def _synthetic(n, n_classes=4, difficulty=2.0, seed=RNG_SEED):
    """Draw (probabilities, labels) where the label is sampled FROM the
    predicted distribution.

    Sampling the label from the same row of probabilities that is handed to
    the predictor is what makes calibration and test points exchangeable by
    construction -- the guarantee should hold here to within Monte-Carlo
    noise regardless of how good or bad the "model" is.
    """
    rng = np.random.default_rng(seed)
    logits = rng.normal(scale=difficulty, size=(n, n_classes))
    exp = np.exp(logits - logits.max(axis=1, keepdims=True))
    probs = exp / exp.sum(axis=1, keepdims=True)
    labels = [
        CLASSES[rng.choice(n_classes, p=probs[i])] for i in range(n)
    ]
    return probs, labels


# --------------------------------------------------------------------------- #
# The guarantee
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("alpha", [0.20, 0.10, 0.05])
@pytest.mark.parametrize("score_function", ["lac", "aps"])
def test_coverage_holds_on_exchangeable_data(alpha, score_function):
    """Empirical coverage must average to 1-alpha across calibration draws.

    AVERAGED OVER INDEPENDENT CALIBRATION SETS, deliberately. The conformal
    guarantee is MARGINAL over both the calibration draw and the test draw --
    it is not conditional on the calibration set you happen to hold. For any
    single calibration set, coverage is itself a random quantity that can sit
    either side of 1-alpha.

    That distinction is not pedantic here. Every test point scored against one
    calibration set shares that set, so their outcomes are correlated through
    it: the spread of coverage is governed by n_calibration, NOT by how many
    test points you evaluate. Measured directly for the novelty case below,
    the standard deviation across calibration draws was 0.0114 at alpha=0.20
    and n_cal=2000, against a naive binomial estimate of 0.0063 -- nearly
    double. A test that fixes one calibration seed and tightens the tolerance
    on test-sample size alone will flake, and worse, will look like a real
    coverage violation when it does.

    The same caveat governs how the real-data results must be read: with a
    175-ticket calibration set, one standard deviation on coverage is roughly
    2 percentage points, so a small shortfall is not by itself evidence of an
    exchangeability violation.
    """
    empirical = []
    for trial in range(12):
        cal_probs, cal_labels = _synthetic(1000, seed=100 + trial)
        test_probs, test_labels = _synthetic(2000, seed=500 + trial)

        calibration = cp.calibrate(
            cal_probs, cal_labels, CLASSES, alpha=alpha,
            score_function=score_function,
        )
        sets = cp.predict_sets(test_probs, calibration)
        empirical.append(cp.coverage(sets, test_labels))

    mean_coverage = float(np.mean(empirical))
    assert abs(mean_coverage - (1 - alpha)) < 0.015, (
        f"{score_function} at alpha={alpha}: mean coverage over "
        f"{len(empirical)} calibration draws was {mean_coverage:.4f}, "
        f"nominal {1 - alpha:.4f}"
    )


@pytest.mark.parametrize("alpha", [0.20, 0.10])
def test_mondrian_gives_per_class_coverage(alpha):
    """Class-conditional coverage should hold for EVERY class, not just on
    average -- that is the whole reason to pay Mondrian's sample-size cost."""
    cal_probs, cal_labels = _synthetic(4000, seed=3)
    test_probs, test_labels = _synthetic(4000, seed=4)

    calibration = cp.calibrate(
        cal_probs, cal_labels, CLASSES, alpha=alpha, mondrian=True,
    )
    sets = cp.predict_sets(test_probs, calibration)

    for c in CLASSES:
        idx = [i for i, y in enumerate(test_labels) if y == c]
        per_class = cp.coverage([sets[i] for i in idx],
                                [test_labels[i] for i in idx])
        assert per_class >= (1 - alpha) - 0.05, (
            f"class {c}: coverage {per_class:.4f} vs nominal {1 - alpha:.4f}"
        )


# --------------------------------------------------------------------------- #
# The quantile: exact rank, not an interpolated estimate
# --------------------------------------------------------------------------- #
def test_quantile_is_the_exact_order_statistic():
    scores = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]  # n = 9
    # ceil((9+1) * 0.9) = 9  -> the 9th smallest
    assert cp.conformal_quantile(scores, 0.10) == pytest.approx(0.9)
    # ceil((9+1) * 0.8) = 8  -> the 8th smallest
    assert cp.conformal_quantile(scores, 0.20) == pytest.approx(0.8)
    # ceil((9+1) * 0.5) = 5  -> the 5th smallest
    assert cp.conformal_quantile(scores, 0.50) == pytest.approx(0.5)


def test_quantile_differs_from_interpolated_numpy_quantile():
    """Guards the specific bug this implementation exists to avoid.

    np.quantile interpolates between order statistics; the conformal
    guarantee depends on the exact ceil((n+1)(1-alpha)) rank. They disagree,
    and the disagreement is silent.
    """
    scores = list(np.linspace(0.0, 1.0, 20))
    exact = cp.conformal_quantile(scores, 0.10)
    interpolated = float(np.quantile(scores, 0.90))
    assert exact != pytest.approx(interpolated)


def test_degenerate_when_calibration_too_small_for_alpha():
    """n too small for alpha => +inf => every set is all labels.

    Honest, not an error: a set containing every label does satisfy the
    coverage guarantee, it just conveys nothing. Callers must be able to see
    that state rather than mistake it for a confident wide set.
    """
    scores = [0.1, 0.2, 0.3]  # n = 3; ceil(4 * 0.99) = 4 > 3
    assert math.isinf(cp.conformal_quantile(scores, 0.01))

    probs, labels = _synthetic(3, seed=5)
    calibration = cp.calibrate(probs, labels, CLASSES, alpha=0.01)
    assert calibration.is_degenerate
    assert cp.predict_sets(probs, calibration) == [CLASSES] * 3


def test_min_calibration_size_matches_reality():
    for alpha in (0.20, 0.10, 0.05, 0.01):
        n = cp.min_calibration_size(alpha)
        assert math.isinf(cp.conformal_quantile(list(np.linspace(0, 1, n - 1)),
                                                alpha))
        assert math.isfinite(cp.conformal_quantile(
            list(np.linspace(0, 1, n)), alpha))


def test_alpha_outside_unit_interval_raises():
    for bad in (0.0, 1.0, -0.1, 1.5):
        with pytest.raises(ValueError, match="alpha"):
            cp.conformal_quantile([0.1, 0.2], bad)


# --------------------------------------------------------------------------- #
# Set-size behaviour
# --------------------------------------------------------------------------- #
def test_sets_grow_monotonically_as_alpha_shrinks():
    """A stricter guarantee can never produce a smaller set."""
    cal_probs, cal_labels = _synthetic(500, seed=6)
    test_probs, _ = _synthetic(200, seed=7)

    previous = None
    for alpha in (0.30, 0.20, 0.10, 0.05):
        calibration = cp.calibrate(cal_probs, cal_labels, CLASSES, alpha=alpha)
        sizes = np.array(
            [len(s) for s in cp.predict_sets(test_probs, calibration)]
        )
        if previous is not None:
            assert np.all(sizes >= previous), (
                f"set sizes shrank when alpha tightened to {alpha}"
            )
        previous = sizes


def test_aps_sets_are_at_least_as_large_as_lac():
    cal_probs, cal_labels = _synthetic(800, seed=8)
    test_probs, _ = _synthetic(400, seed=9)

    mean_size = {}
    for fn in ("lac", "aps"):
        calibration = cp.calibrate(cal_probs, cal_labels, CLASSES,
                                   alpha=0.10, score_function=fn)
        sets = cp.predict_sets(test_probs, calibration)
        mean_size[fn] = float(np.mean([len(s) for s in sets]))

    assert mean_size["aps"] >= mean_size["lac"]


def test_empty_sets_are_preserved_not_backfilled():
    """An empty set is the clearest OOD signal the method produces.

    Backfilling it with the argmax would convert "I have never seen anything
    like this" into a confident wrong answer -- the exact failure the
    escalation gates exist to prevent.
    """
    calibration = cp.ConformalCalibration(
        alpha=0.10, score_function="lac", classes=CLASSES, quantile=0.0,
    )
    uniform = np.full((1, len(CLASSES)), 0.25)
    assert cp.predict_sets(uniform, calibration) == [[]]


# --------------------------------------------------------------------------- #
# Input validation -- fail loud, per project convention
# --------------------------------------------------------------------------- #
def test_class_column_mismatch_raises():
    probs, labels = _synthetic(10, seed=10)
    with pytest.raises(ValueError, match="classes"):
        cp.calibrate(probs, labels, ["A", "B"], alpha=0.10)


def test_unknown_calibration_label_raises():
    probs, labels = _synthetic(10, seed=11)
    labels = list(labels)
    labels[0] = "Nonexistent"
    with pytest.raises(ValueError, match="does not know"):
        cp.calibrate(probs, labels, CLASSES, alpha=0.10)


def test_scores_are_valid_probabilities_transform():
    probs, _ = _synthetic(50, seed=12)
    lac = cp.lac_scores(probs)
    aps = cp.aps_scores(probs)

    assert np.all(lac >= 0) and np.all(lac <= 1)
    assert np.all(aps >= 0) and np.all(aps <= 1 + 1e-9)
    # APS at the argmax equals that label's own probability.
    top = np.argmax(probs, axis=1)
    for i, j in enumerate(top):
        assert aps[i, j] == pytest.approx(probs[i, j])


# --------------------------------------------------------------------------- #
# Novelty detection
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("alpha", [0.20, 0.10, 0.05])
def test_p_values_are_superuniform_on_inliers(alpha):
    """The novelty guarantee: at most alpha of genuine inliers get flagged.

    Averaged over independent calibration draws, for the reason documented on
    test_coverage_holds_on_exchangeable_data -- the guarantee is marginal, and
    a single calibration draw legitimately lands either side of alpha.

    Theory gives P(p <= alpha) = floor(alpha*(n+1))/(n+1), which is just below
    alpha; the assertion allows a small Monte-Carlo margin above it.
    """
    rates = []
    for trial in range(12):
        rng = np.random.default_rng(200 + trial)
        cal = rng.normal(size=2000)
        test = rng.normal(size=2000)
        rates.append(float(np.mean(cp.conformal_p_values(cal, test) <= alpha)))

    mean_rate = float(np.mean(rates))
    assert mean_rate <= alpha + 0.01, (
        f"alpha={alpha}: mean false-positive rate over {len(rates)} "
        f"calibration draws was {mean_rate:.4f}"
    )


def test_p_values_detect_genuine_outliers():
    rng = np.random.default_rng(RNG_SEED)
    cal = rng.normal(size=1000)
    outliers = rng.normal(loc=6.0, size=200)

    p = cp.conformal_p_values(cal, outliers)
    assert float(np.mean(p <= 0.10)) > 0.95


def test_p_values_are_bounded_and_never_zero():
    """Minimum attainable p-value is 1/(n+1) -- a small calibration set
    cannot certify a small alpha, and must not pretend to."""
    cal = list(np.linspace(0, 1, 10))
    p = cp.conformal_p_values(cal, [99.0, -99.0])
    assert p.min() == pytest.approx(1.0 / 11.0)
    assert p.max() == pytest.approx(1.0)


def test_p_value_orientation_is_higher_means_more_nonconforming():
    """Guards a sign error that would invert the whole gate.

    Retrieval similarity must be negated before use; if it were not, the
    novelty detector would escalate the most relevant tickets and pass the
    least relevant ones.
    """
    similarities_in_domain = [0.80, 0.85, 0.90, 0.95]
    cal_scores = [-s for s in similarities_in_domain]

    off_topic = cp.conformal_p_values(cal_scores, [-0.20])[0]
    on_topic = cp.conformal_p_values(cal_scores, [-0.93])[0]
    assert off_topic < on_topic
