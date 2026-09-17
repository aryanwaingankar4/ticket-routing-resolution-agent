"""
Drift detection over persisted decision records -- MEASUREMENT ONLY.

WHY THIS EXISTS
---------------
This project's recurring bug class is a value or artifact that is wrong for
its context, internally consistent, and therefore silent. artifacts.py's
guards catch that at LOAD time. This module is the population-level detector
for the same class: an artifact that is wrong but still loadable, or a world
that has moved away from what the gates were calibrated on, shows up as a
shift in the gates' own inputs before it shows up as a visibly wrong answer.

Nothing here gates routing. `settings.drift.enabled` is False, and the report
carries statistics and p-values only -- never an alarm. Choosing a window
size or an alarm threshold is Phase 4B's measured output, not a default.

PURE BY DESIGN
--------------
No file I/O, no models. Records arrive already parsed; the reference arrives
already loaded (artifacts.load_drift_reference). An escalation-adjacent
detector that needs BGE and FAISS to test is one nobody tests -- the same
reason the orchestrator's gates are pure functions.

THE THREE SIGNALS, KEPT SEPARATE
--------------------------------
A. Retrieval novelty (primary). Each record's top-1 similarity becomes a
   conformal p-value against the 175-ticket in-domain reference, reusing
   conformal.conformal_p_values() verbatim. Under exchangeability p is
   superuniform, so drift piles mass at low p.

   The false-alarm rate is alpha only MARGINALLY -- averaged over calibration
   draws. For the single fixed reference in use, the per-ticket rate of
   p <= alpha is itself a random variable ~ Beta(l, n+1-l), l = floor((n+1)a),
   whose spread (~0.023 at n=175, a=0.10) matches a 200-ticket window's own
   sampling spread. So two tests are reported: MARGINAL against l/(n+1), and
   CALIBRATION-CONDITIONAL against the Beta upper bound held with confidence
   1 - delta. Phase 4B measures both null rates; neither is assumed.

B. The rates the thesis rests on: escalation rate, Tier-1 share, category
   mix. Reported, NOT claimed as calibrated -- the reference proportions are
   estimated from n=175, and a binomial against an estimated rate ignores
   that estimation noise.

C. Config fingerprint. A window with a fingerprint other than the expected
   one is a DEPLOYMENT FAULT, not a distribution shift, and is reported in
   its own field so it can never be averaged into a shift verdict.
"""

from __future__ import annotations

import json
import math
from collections import Counter
from typing import Any, Iterable, Sequence

import numpy as np
from pydantic import (BaseModel, ConfigDict, Field, field_validator,
                      model_validator)

from src.agent.config import settings
from src.agent.conformal import conformal_p_values
from src.agent.errors import ArtifactError

# Kept in step with logging_setup.DECISION_SCHEMA_VERSION; the test suite
# asserts they agree. Imported rather than duplicated would pull logging_setup
# into this pure module, so it is pinned by test instead.
SUPPORTED_RECORD_SCHEMA_VERSION = 1
REFERENCE_SCHEMA_VERSION = 1

_RETRIEVAL_AGENT = "retrieval"
_STEP_OK = "ok"


class _Model(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid",
                              protected_namespaces=())


# --------------------------------------------------------------------------- #
# Inputs                                                                       #
# --------------------------------------------------------------------------- #
class DecisionRecord(BaseModel):
    """The fields of a persisted pipeline_decision record drift reads.

    Extra fields are ignored (the record also carries latency, reason, etc.),
    but every field read here is required: a record missing one is rejected,
    never defaulted -- a defaulted top_similarity of 0.0 would register as
    maximal novelty.
    """

    model_config = ConfigDict(frozen=True, extra="ignore")

    schema_version: int
    category: str
    tier: int
    top_similarity: float
    escalated: bool
    config_fingerprint: str
    agent_status: dict[str, str]

    @field_validator("schema_version")
    @classmethod
    def _known_schema(cls, value: int) -> int:
        if value != SUPPORTED_RECORD_SCHEMA_VERSION:
            raise ValueError(
                f"decision record schema_version {value} is not supported "
                f"(this detector reads version "
                f"{SUPPORTED_RECORD_SCHEMA_VERSION})"
            )
        return value

    @field_validator("tier")
    @classmethod
    def _known_tier(cls, value: int) -> int:
        if value not in (1, 2):
            raise ValueError(f"tier must be 1 or 2, got {value}")
        return value

    @field_validator("agent_status")
    @classmethod
    def _has_retrieval_step(cls, value: dict[str, str]) -> dict[str, str]:
        if _RETRIEVAL_AGENT not in value:
            raise ValueError(
                "agent_status has no 'retrieval' entry, so whether "
                "top_similarity was measured cannot be determined"
            )
        return value

    @property
    def retrieval_ran(self) -> bool:
        return self.agent_status[_RETRIEVAL_AGENT] == _STEP_OK

    @classmethod
    def from_json_line(cls, line: str) -> "DecisionRecord":
        return cls.model_validate(json.loads(line))


class DriftReference(_Model):
    """The in-domain distribution the gates were calibrated against.

    Built offline by src/experiments/build_drift_reference.py from the
    175-ticket calibration set -- the same set behind the RAG gate's
    conformal novelty result.
    """

    schema_version: int
    embedding_model: str
    embedding_dim: int
    index_ntotal: int
    index_sha256: str
    source_sha256: str
    config_fingerprint: str
    n: int = Field(gt=0)

    # Signal A. `similarity_scores` is the NON-SELF top-1 similarity: a live
    # ticket is never its own index row, so this is the exchangeable analogue.
    # The self-inclusive scores are kept for audit only.
    similarity_scores: list[float]
    similarity_scores_with_self: list[float]

    # Signal B.
    n_escalated: int
    n_tier1: int
    category_counts: dict[str, int]

    # Audit trail from the builder (source path, the published rates it
    # reproduced). Read by humans, never by the detector.
    provenance: dict[str, Any] = Field(default_factory=dict)

    @field_validator("schema_version")
    @classmethod
    def _known_schema(cls, value: int) -> int:
        if value != REFERENCE_SCHEMA_VERSION:
            raise ValueError(
                f"drift reference schema_version {value} is not supported"
            )
        return value

    @model_validator(mode="after")
    def _internally_consistent(self) -> "DriftReference":
        lengths = {
            "similarity_scores": len(self.similarity_scores),
            "similarity_scores_with_self":
                len(self.similarity_scores_with_self),
            "sum(category_counts)": sum(self.category_counts.values()),
        }
        bad = {k: v for k, v in lengths.items() if v != self.n}
        if bad or not (0 <= self.n_escalated <= self.n) \
                or not (0 <= self.n_tier1 <= self.n):
            raise ValueError(
                f"drift reference is internally inconsistent with n={self.n}: "
                f"{bad or 'escalated/tier1 counts out of range'}"
            )
        return self

    @property
    def escalation_rate(self) -> float:
        return self.n_escalated / self.n

    @property
    def tier1_share(self) -> float:
        return self.n_tier1 / self.n

    def check_compatible(self, *, embedding_model: str, embedding_dim: int,
                         index_ntotal: int | None = None,
                         index_sha256: str | None = None) -> None:
        """Raise if this reference was built against different artifacts.

        Deliberately NOT a full config-fingerprint check: the fingerprint
        changes on any unrelated config edit (the Phase 1 conformal artifact
        already carries a stale one), which would make the guard cry wolf
        until someone deleted it. What invalidates a similarity reference is
        the encoder and the index, so those are what is checked.
        """
        problems = []
        if embedding_model != self.embedding_model:
            problems.append(f"embedding model: reference "
                            f"{self.embedding_model!r}, live "
                            f"{embedding_model!r}")
        if embedding_dim != self.embedding_dim:
            problems.append(f"embedding dim: reference {self.embedding_dim}, "
                            f"live {embedding_dim}")
        if index_ntotal is not None and index_ntotal != self.index_ntotal:
            problems.append(f"index size: reference {self.index_ntotal}, "
                            f"live {index_ntotal}")
        if index_sha256 is not None and index_sha256 != self.index_sha256:
            problems.append("FAISS index file hash differs from the one the "
                            "reference was built against")
        if problems:
            raise ArtifactError(
                "STALE DRIFT REFERENCE:\n"
                + "\n".join(f"    - {p}" for p in problems)
                + "\n\n  Similarities measured against a different encoder "
                "or index are not\n  comparable, so every drift p-value would "
                "be wrong with no error.\n"
                "  Rebuild it from the project root with:\n"
                "      python src/experiments/build_drift_reference.py"
            )


# --------------------------------------------------------------------------- #
# Outputs                                                                      #
# --------------------------------------------------------------------------- #
class NoveltySignal(_Model):
    """Signal A. Statistics only; no alarm."""

    n_scored: int
    n_excluded_retrieval_skipped: int
    alpha: float
    flagged: int
    flagged_rate: float
    # Exact marginal P(p <= alpha) = floor((n+1)alpha)/(n+1), which is <= alpha.
    marginal_null_rate: float
    marginal_binomial_p: float
    conditional_delta: float
    conditional_null_rate_bound: float
    conditional_binomial_p: float
    # Secondary read. Conformal p-values are discrete (multiples of 1/(n+1)),
    # so KS against continuous U[0,1] is approximate and grows sensitive to
    # that discreteness in large windows.
    ks_statistic: float
    ks_p: float


class RateTest(_Model):
    count: int
    n: int
    observed_rate: float
    reference_rate: float
    # True when the reference rate is exactly 0 or 1. A binomial against such
    # a rate returns p = 0 the moment one ticket differs, which is not
    # evidence of drift, so the p-value is left None rather than reported.
    # The in-domain reference escalates 0/175 tickets, so the escalation test
    # is degenerate on it.
    degenerate_reference: bool
    binomial_p_two_sided: float | None


class CategoryMixTest(_Model):
    observed_counts: dict[str, int]
    expected_counts: dict[str, float]
    # Categories seen in the window but absent from the reference. When any
    # exist the chi-square is undefined and left None rather than invented.
    unknown_categories: list[str]
    chi_square: float | None
    p_value: float | None
    min_expected: float
    small_expected_counts: bool


class RateSignals(_Model):
    """Signal B. Reported, not claimed as calibrated."""

    escalation: RateTest
    tier1_share: RateTest
    category_mix: CategoryMixTest


class DeploymentSignal(_Model):
    """Signal C. A deployment fault, never a distribution shift."""

    expected_fingerprint: str
    fingerprint_counts: dict[str, int]
    fingerprint_mismatch: bool


class DriftReport(_Model):
    n_records: int
    insufficient_data: bool
    novelty: NoveltySignal | None
    rates: RateSignals | None
    deployment: DeploymentSignal


# --------------------------------------------------------------------------- #
# Statistics                                                                   #
# --------------------------------------------------------------------------- #
def marginal_null_rate(n_calibration: int, alpha: float) -> float:
    """Exact marginal probability that a conformal p-value is <= alpha."""
    _check_alpha(alpha)
    return math.floor((n_calibration + 1) * alpha) / (n_calibration + 1)


def conditional_null_rate_bound(n_calibration: int, alpha: float,
                                delta: float) -> float:
    """Upper bound on P(p <= alpha | calibration set), with prob >= 1-delta.

    Conditional on the calibration draw, the rate is Beta(l, n+1-l) with
    l = floor((n+1)alpha) (Vovk 2012, training-conditional validity). Its
    1-delta quantile bounds the rate the fixed reference actually delivers.
    """
    _check_alpha(alpha)
    if not 0.0 < delta < 1.0:
        raise ValueError(f"delta must be in (0, 1), got {delta}")
    from scipy.stats import beta as beta_dist

    l = math.floor((n_calibration + 1) * alpha)
    if l == 0:
        return 0.0
    return float(beta_dist.ppf(1.0 - delta, l, n_calibration + 1 - l))


def _check_alpha(alpha: float) -> None:
    if not 0.0 < alpha < 1.0:
        raise ValueError(f"alpha must be in (0, 1), got {alpha}")


def _binomial_greater(k: int, n: int, p: float) -> float:
    from scipy.stats import binomtest

    return float(binomtest(k, n, p, alternative="greater").pvalue)


def _binomial_two_sided(k: int, n: int, p: float) -> float:
    from scipy.stats import binomtest

    return float(binomtest(k, n, p, alternative="two-sided").pvalue)


def novelty_signal(similarities: Sequence[float],
                   reference: DriftReference,
                   alpha: float, delta: float,
                   n_excluded: int = 0) -> NoveltySignal:
    """Signal A over a non-empty sequence of top-1 similarities."""
    if len(similarities) == 0:
        raise ValueError("novelty_signal needs at least one similarity")
    from scipy.stats import kstest

    # Higher must mean MORE nonconforming, so similarity is negated -- the
    # same orientation calibrate_conformal.py uses for the RAG gate.
    p_values = conformal_p_values(
        [-s for s in reference.similarity_scores],
        [-s for s in similarities],
    )
    n = len(p_values)
    flagged = int(np.sum(p_values <= alpha))
    marginal = marginal_null_rate(reference.n, alpha)
    conditional = conditional_null_rate_bound(reference.n, alpha, delta)
    ks = kstest(p_values, "uniform")

    return NoveltySignal(
        n_scored=n,
        n_excluded_retrieval_skipped=n_excluded,
        alpha=alpha,
        flagged=flagged,
        flagged_rate=flagged / n,
        marginal_null_rate=marginal,
        marginal_binomial_p=_binomial_greater(flagged, n, marginal),
        conditional_delta=delta,
        conditional_null_rate_bound=conditional,
        conditional_binomial_p=_binomial_greater(flagged, n, conditional),
        ks_statistic=float(ks.statistic),
        ks_p=float(ks.pvalue),
    )


def _rate_test(count: int, n: int, reference_rate: float) -> RateTest:
    degenerate = reference_rate in (0.0, 1.0)
    return RateTest(
        count=count, n=n, observed_rate=count / n,
        reference_rate=reference_rate,
        degenerate_reference=degenerate,
        binomial_p_two_sided=(None if degenerate else
                              _binomial_two_sided(count, n, reference_rate)),
    )


def category_mix_test(categories: Iterable[str],
                      reference: DriftReference) -> CategoryMixTest:
    observed = Counter(categories)
    n = sum(observed.values())
    if n == 0:
        raise ValueError("category_mix_test needs at least one record")

    ref_total = sum(reference.category_counts.values())
    names = sorted(set(reference.category_counts) | set(observed))
    expected = {
        c: n * reference.category_counts.get(c, 0) / ref_total for c in names
    }
    observed_full = {c: observed.get(c, 0) for c in names}
    unknown = sorted(c for c in observed if reference.category_counts.get(c, 0)
                     == 0)
    known_expected = [v for c, v in expected.items() if c not in unknown]
    min_expected = min(known_expected) if known_expected else 0.0

    chi_square = p_value = None
    if not unknown:
        from scipy.stats import chisquare

        result = chisquare(
            [observed_full[c] for c in names],
            [expected[c] for c in names],
        )
        chi_square, p_value = float(result.statistic), float(result.pvalue)

    return CategoryMixTest(
        observed_counts=observed_full,
        expected_counts=expected,
        unknown_categories=unknown,
        chi_square=chi_square,
        p_value=p_value,
        min_expected=min_expected,
        small_expected_counts=min_expected < 5.0,
    )


def deployment_signal(records: Sequence[DecisionRecord],
                      expected_fingerprint: str) -> DeploymentSignal:
    counts = dict(Counter(r.config_fingerprint for r in records))
    return DeploymentSignal(
        expected_fingerprint=expected_fingerprint,
        fingerprint_counts=counts,
        fingerprint_mismatch=any(fp != expected_fingerprint for fp in counts),
    )


# --------------------------------------------------------------------------- #
# The entry point                                                              #
# --------------------------------------------------------------------------- #
def detect(records: Sequence[DecisionRecord],
           reference: DriftReference,
           expected_fingerprint: str,
           alpha: float | None = None,
           delta: float | None = None) -> DriftReport:
    """Compute every signal over one window of decision records.

    `alpha` and `delta` default to settings.drift. `expected_fingerprint` is
    passed in rather than read from config so a caller auditing an old window
    can state what it expected -- and so this function stays pure.
    """
    alpha = settings.drift.alpha if alpha is None else alpha
    delta = settings.drift.conditional_delta if delta is None else delta
    _check_alpha(alpha)

    records = list(records)
    deployment = deployment_signal(records, expected_fingerprint)

    if not records:
        return DriftReport(n_records=0, insufficient_data=True, novelty=None,
                           rates=None, deployment=deployment)

    # A record whose retrieval step was skipped (the filing gate) carries
    # top_similarity 0.0 because nothing was measured. Scoring it would
    # register maximal novelty, so it is excluded and counted instead.
    scored = [r.top_similarity for r in records if r.retrieval_ran]
    n_excluded = len(records) - len(scored)
    novelty = (novelty_signal(scored, reference, alpha, delta, n_excluded)
               if scored else None)

    n = len(records)
    rates = RateSignals(
        escalation=_rate_test(sum(r.escalated for r in records), n,
                              reference.escalation_rate),
        tier1_share=_rate_test(sum(r.tier == 1 for r in records), n,
                               reference.tier1_share),
        category_mix=category_mix_test((r.category for r in records),
                                       reference),
    )

    return DriftReport(
        n_records=n,
        insufficient_data=novelty is None,
        novelty=novelty,
        rates=rates,
        deployment=deployment,
    )
