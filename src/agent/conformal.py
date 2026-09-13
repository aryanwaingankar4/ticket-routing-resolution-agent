"""
Split conformal prediction -- distribution-free, finite-sample guarantees.

WHY THIS EXISTS
---------------
This project's three gates are calibrated but not *guaranteed*. The cascade
threshold (0.50) and the RAG similarity threshold (0.67) were each derived by
sweeping a metric and picking a defensible point; neither carries a statement
of the form "the true answer is in here at least 90% of the time."

Split conformal does carry exactly that statement, and it needs no assumption
that the underlying model is well calibrated -- which matters here, because
both cascade tiers are measurably UNDERconfident (ECE 0.1122 / 0.0992).
Conformal absorbs that miscalibration automatically: it only ever compares a
new score against the empirical distribution of calibration scores.

THE GUARANTEE, AND ITS ONE ASSUMPTION
-------------------------------------
For calibration scores s_1..s_n drawn exchangeably with a test point:

    q_hat = the ceil((n+1)(1-alpha))-th smallest calibration score
    C(x)  = { y : s(x, y) <= q_hat }
    =>      1 - alpha  <=  P(Y in C(X))  <=  1 - alpha + 1/(n+1)

The upper bound matters as much as the lower one: conformal is not
conservative by accident, it is tight to within 1/(n+1).

EXCHANGEABILITY is the whole assumption, and it is where this breaks in
practice. If the calibration set is not drawn like deployment traffic, the
guarantee is nominal only. This project has two documented violations:

  1. The 175-ticket calibration set consists of paraphrases of rows the
     models were TRAINED on, so calibration scores are optimistically low,
     q_hat is too small, sets are too tight, and coverage lands BELOW
     nominal -- failing silently in the reassuring direction.
  2. The deployment-like benchmark (plain English, non-template) scores ~71%
     where in-distribution data scores ~100%. Calibrating on one and
     deploying on the other breaks exchangeability outright.

Both are measured rather than assumed -- see
src/experiments/calibrate_conformal.py.

IMPLEMENTATION NOTES
--------------------
numpy only. The quantile is computed as an exact ORDER STATISTIC (sort, then
index), not via np.quantile -- interpolating quantile estimators silently
break the finite-sample guarantee, which depends on the exact
ceil((n+1)(1-alpha)) rank. Getting this wrong is the single most common
conformal implementation bug, and it fails quietly.

Deterministic throughout. The randomized variant of APS gives exact rather
than conservative coverage, but this project fixes seed 42 everywhere for
reproducibility, and a reviewer re-running a calibration must get the same
number twice.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Literal, Sequence

import numpy as np

ScoreFunction = Literal["lac", "aps"]


# --------------------------------------------------------------------------- #
# Nonconformity scores
# --------------------------------------------------------------------------- #
def lac_scores(probabilities: np.ndarray) -> np.ndarray:
    """LAC / THR score: s(x, y) = 1 - p_hat(y | x).

    Returns an (n_samples, n_classes) matrix of scores -- the score for
    EVERY candidate label, not just the true one, because prediction-set
    construction needs all of them.

    Produces the smallest possible average set size, at the cost of poorer
    class-conditional coverage (easy classes get systematically over-covered
    and hard ones under-covered). Pair with mondrian=True if per-class
    coverage matters.
    """
    probabilities = np.asarray(probabilities, dtype=np.float64)
    return 1.0 - probabilities


def aps_scores(probabilities: np.ndarray) -> np.ndarray:
    """APS score: cumulative probability mass down to and including y.

    s(x, y) = sum of p_hat(y') over all y' with p_hat(y') >= p_hat(y)

    Larger sets than LAC, but adapts set size to genuine difficulty: an easy
    ticket gets a singleton, a genuinely ambiguous one gets a wider set
    rather than a confidently wrong singleton. For a triage system deciding
    whether a human is needed, that adaptivity is the point.

    Deterministic (non-randomized) variant -- see module docstring.
    """
    probabilities = np.asarray(probabilities, dtype=np.float64)

    # Sort each row descending, cumulative-sum, then scatter back so that
    # position (i, y) holds the cumulative mass at label y's rank.
    order = np.argsort(-probabilities, axis=1, kind="stable")
    sorted_probs = np.take_along_axis(probabilities, order, axis=1)
    cumulative = np.cumsum(sorted_probs, axis=1)

    scores = np.empty_like(probabilities)
    np.put_along_axis(scores, order, cumulative, axis=1)
    return scores


def score_matrix(probabilities: np.ndarray,
                 score_function: ScoreFunction = "lac") -> np.ndarray:
    if score_function == "lac":
        return lac_scores(probabilities)
    if score_function == "aps":
        return aps_scores(probabilities)
    raise ValueError(
        f"Unknown score_function {score_function!r}; expected 'lac' or 'aps'."
    )


# --------------------------------------------------------------------------- #
# The quantile -- the part that must be exactly right
# --------------------------------------------------------------------------- #
def conformal_quantile(scores: Sequence[float], alpha: float) -> float:
    """Return q_hat: the ceil((n+1)(1-alpha))-th smallest score.

    Deliberately an exact order statistic rather than np.quantile. An
    interpolating quantile lands between two order statistics and voids the
    finite-sample guarantee, with no visible symptom.

    When ceil((n+1)(1-alpha)) > n the requested rank does not exist -- the
    calibration set is too small to certify that alpha. Returns +inf, which
    makes C(x) the full label set: the honest answer, since a set containing
    every label trivially satisfies the coverage guarantee while conveying
    no information. Callers should surface that rather than hide it.
    """
    if not 0.0 < alpha < 1.0:
        raise ValueError(f"alpha must be in (0, 1); got {alpha!r}")

    scores = np.asarray(scores, dtype=np.float64).ravel()
    n = scores.size
    if n == 0:
        raise ValueError("Cannot calibrate on an empty score set.")

    rank = math.ceil((n + 1) * (1.0 - alpha))
    if rank > n:
        return float("inf")

    return float(np.sort(scores)[rank - 1])


def min_calibration_size(alpha: float) -> int:
    """Smallest n for which a finite q_hat exists at this alpha.

    ceil((n+1)(1-alpha)) <= n  holds first at n = ceil(1/alpha) - 1.
    At alpha=0.01 that is 99 calibration points; at alpha=0.05, 19.
    Useful for telling a user *why* their sets came back as all labels.
    """
    return math.ceil(1.0 / alpha) - 1


# --------------------------------------------------------------------------- #
# Calibrated predictor
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class ConformalCalibration:
    """A fitted conformal predictor: quantile(s) plus full provenance."""

    alpha: float
    score_function: ScoreFunction
    classes: list[str]
    quantile: float | None = None
    class_quantiles: dict[str, float] = field(default_factory=dict)
    mondrian: bool = False
    n_calibration: int = 0
    n_per_class: dict[str, int] = field(default_factory=dict)

    @property
    def is_degenerate(self) -> bool:
        """True when no finite quantile exists, so every set is all labels."""
        if self.mondrian:
            return any(not math.isfinite(q)
                       for q in self.class_quantiles.values())
        return self.quantile is None or not math.isfinite(self.quantile)

    def to_dict(self) -> dict:
        return {
            "alpha": self.alpha,
            "score_function": self.score_function,
            "classes": list(self.classes),
            "quantile": self.quantile,
            "class_quantiles": dict(self.class_quantiles),
            "mondrian": self.mondrian,
            "n_calibration": self.n_calibration,
            "n_per_class": dict(self.n_per_class),
            "is_degenerate": self.is_degenerate,
        }


def calibrate(probabilities: np.ndarray,
              labels: Sequence[str],
              classes: Sequence[str],
              alpha: float,
              score_function: ScoreFunction = "lac",
              mondrian: bool = False) -> ConformalCalibration:
    """Fit a conformal predictor on held-out calibration data.

    `probabilities` is (n_calibration, n_classes), column order matching
    `classes` -- which must be the estimator's own `classes_` order. A
    mismatch here silently scores the wrong label and is invisible
    downstream, so it is validated.

    `mondrian=True` fits one quantile per class, giving class-conditional
    rather than marginal coverage. Costs sample size: each class's quantile
    sees only its own calibration points.
    """
    probabilities = np.asarray(probabilities, dtype=np.float64)
    classes = list(classes)
    labels = list(labels)

    if probabilities.ndim != 2:
        raise ValueError(
            f"probabilities must be 2-D (n_samples, n_classes); got shape "
            f"{probabilities.shape}"
        )
    if probabilities.shape[1] != len(classes):
        raise ValueError(
            f"probabilities has {probabilities.shape[1]} columns but "
            f"{len(classes)} classes were given. Column order must match the "
            f"estimator's classes_ exactly."
        )
    if len(labels) != probabilities.shape[0]:
        raise ValueError(
            f"Got {len(labels)} labels for {probabilities.shape[0]} rows."
        )

    unknown = sorted(set(labels) - set(classes))
    if unknown:
        raise ValueError(
            f"Calibration labels contain categories the model does not know: "
            f"{unknown}"
        )

    index_of = {c: i for i, c in enumerate(classes)}
    all_scores = score_matrix(probabilities, score_function)
    true_scores = np.array(
        [all_scores[i, index_of[y]] for i, y in enumerate(labels)],
        dtype=np.float64,
    )

    n_per_class = {c: int(sum(1 for y in labels if y == c)) for c in classes}

    if not mondrian:
        return ConformalCalibration(
            alpha=alpha,
            score_function=score_function,
            classes=classes,
            quantile=conformal_quantile(true_scores, alpha),
            mondrian=False,
            n_calibration=len(labels),
            n_per_class=n_per_class,
        )

    class_quantiles: dict[str, float] = {}
    for c in classes:
        mask = np.array([y == c for y in labels], dtype=bool)
        if not mask.any():
            # No calibration data for this class: cannot certify it at all.
            class_quantiles[c] = float("inf")
            continue
        class_quantiles[c] = conformal_quantile(true_scores[mask], alpha)

    return ConformalCalibration(
        alpha=alpha,
        score_function=score_function,
        classes=classes,
        class_quantiles=class_quantiles,
        mondrian=True,
        n_calibration=len(labels),
        n_per_class=n_per_class,
    )


def predict_sets(probabilities: np.ndarray,
                 calibration: ConformalCalibration) -> list[list[str]]:
    """Return one prediction set per row.

    An EMPTY set is a real, meaningful outcome, not an error: it means every
    label is more nonconforming than anything seen during calibration, i.e. a
    strong out-of-distribution signal. It is never silently backfilled with
    the argmax -- doing so would convert the system's clearest "I have never
    seen anything like this" signal into a confident wrong answer, which is
    the exact failure this project's escalation gates exist to prevent.
    """
    probabilities = np.asarray(probabilities, dtype=np.float64)
    if probabilities.ndim == 1:
        probabilities = probabilities.reshape(1, -1)

    if probabilities.shape[1] != len(calibration.classes):
        raise ValueError(
            f"probabilities has {probabilities.shape[1]} columns but the "
            f"calibration covers {len(calibration.classes)} classes."
        )

    all_scores = score_matrix(probabilities, calibration.score_function)
    sets: list[list[str]] = []

    for row in all_scores:
        if calibration.mondrian:
            chosen = [
                c for i, c in enumerate(calibration.classes)
                if row[i] <= calibration.class_quantiles.get(c, float("inf"))
            ]
        else:
            q = calibration.quantile
            q = float("inf") if q is None else q
            chosen = [
                c for i, c in enumerate(calibration.classes) if row[i] <= q
            ]
        sets.append(chosen)

    return sets


def coverage(prediction_sets: Sequence[Sequence[str]],
             labels: Sequence[str]) -> float:
    """Fraction of points whose true label is in its prediction set."""
    if len(prediction_sets) != len(labels):
        raise ValueError(
            f"Got {len(prediction_sets)} sets for {len(labels)} labels."
        )
    if not labels:
        raise ValueError("Cannot compute coverage over zero points.")
    hits = sum(1 for s, y in zip(prediction_sets, labels) if y in s)
    return hits / len(labels)


# --------------------------------------------------------------------------- #
# Conformal novelty detection (the RAG gate, reframed)
# --------------------------------------------------------------------------- #
def conformal_p_values(calibration_scores: Sequence[float],
                       test_scores: Sequence[float]) -> np.ndarray:
    """Marginal conformal p-values for a one-class / novelty test.

        p(x) = (1 + #{i : s_i >= s(x)}) / (n + 1)

    Under exchangeability with the calibration inliers, p(x) is (super)uniform
    on [0, 1], so thresholding at alpha gives a false-positive rate <= alpha.
    Applied to the RAG gate this means: escalating when p <= alpha wrongly
    escalates at most alpha of genuinely in-domain tickets -- a guarantee the
    hand-tuned 0.67 similarity threshold does not have.

    This is why the framing matters: the gate's job is detecting out-of-domain
    input, and conformal novelty detection calibrates on INLIERS ONLY by
    construction. The 45-ticket OOD set is therefore freed to be pure
    evaluation data, measuring detection power rather than setting the
    threshold it is then judged against.

    Pass scores oriented so that HIGHER means MORE nonconforming. For
    retrieval similarity that means negating it: s = -similarity.
    """
    cal = np.asarray(calibration_scores, dtype=np.float64).ravel()
    test = np.asarray(test_scores, dtype=np.float64).ravel()
    if cal.size == 0:
        raise ValueError("Cannot compute p-values with no calibration scores.")

    # Count calibration scores >= each test score. searchsorted on the sorted
    # array is O(log n) per test point instead of O(n).
    cal_sorted = np.sort(cal)
    n_ge = cal.size - np.searchsorted(cal_sorted, test, side="left")
    return (1.0 + n_ge) / (cal.size + 1.0)
