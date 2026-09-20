"""
Drift detector (Phase 4A) -- pure functions, no models.

What these pin down is the list of ways a drift detector goes quietly wrong:
a p-value against the wrong null, an unmeasured similarity scored as maximal
novelty, a deployment fault averaged into a shift verdict, a degenerate
reference producing p = 0 on the first ticket, and a stale reference accepted
against a rebuilt index.
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from src.agent import drift
from src.agent.errors import ArtifactError

FP = "aaaaaaaaaaaa"
N_REF = 175
REF_SCORES = [float(x) for x in np.linspace(0.60, 0.95, N_REF)]
CATEGORY_COUNTS = {"Database": 100, "Network": 50, "Storage": 25}


def _reference(**overrides) -> drift.DriftReference:
    fields = dict(
        schema_version=drift.REFERENCE_SCHEMA_VERSION,
        embedding_model="BAAI/bge-base-en-v1.5",
        embedding_dim=768,
        index_ntotal=4000,
        index_sha256="f" * 64,
        source_sha256="e" * 64,
        config_fingerprint=FP,
        n=N_REF,
        similarity_scores=REF_SCORES,
        similarity_scores_with_self=REF_SCORES,
        n_escalated=35,
        n_tier1=70,
        category_counts=CATEGORY_COUNTS,
    )
    fields.update(overrides)
    return drift.DriftReference(**fields)


def _record(sim=0.8, *, retrieval="ok", escalated=False, tier=2,
            category="Database", fingerprint=FP, **overrides):
    fields = dict(
        schema_version=drift.SUPPORTED_RECORD_SCHEMA_VERSION,
        category=category, tier=tier, top_similarity=sim,
        escalated=escalated, config_fingerprint=fingerprint,
        agent_status={"classification": "ok", "retrieval": retrieval,
                      "resolution": "skipped"},
        ticket_id="t1", latency_ms=12.0,  # extra fields are ignored
    )
    fields.update(overrides)
    return drift.DecisionRecord(**fields)


def _hand_binomial_greater(k, n, p):
    return sum(math.comb(n, i) * p**i * (1 - p)**(n - i)
               for i in range(k, n + 1))


# --------------------------------------------------------------------------- #
# Null rates                                                                   #
# --------------------------------------------------------------------------- #
def test_marginal_null_rate_is_the_exact_attainable_rate():
    assert drift.marginal_null_rate(175, 0.10) == 17 / 176
    assert drift.marginal_null_rate(175, 0.10) <= 0.10


def test_conditional_bound_exceeds_marginal_and_tightens_with_n():
    """The fixed-reference correction: large at n=175, vanishing as n grows."""
    small = drift.conditional_null_rate_bound(175, 0.10, 0.10)
    large = drift.conditional_null_rate_bound(100_000, 0.10, 0.10)
    assert small > drift.marginal_null_rate(175, 0.10)
    assert small > large > 0.10 - 1e-3
    assert small - 0.10 > 0.02, "the correction is not negligible at n=175"


def test_conditional_bound_is_the_quantile_of_the_realised_rate():
    """Phase 4B check 1: the Beta law behind the 0.126 bound, by simulation.

    For continuous exchangeable scores the realised false-alarm rate of a
    fixed calibration set is 1 - U_(n+1-l), which is Beta(l, n+1-l). Its mean
    must be the marginal rate and its 90th percentile the conditional bound.
    This verifies the maths and the code, not this project's data.
    """
    n, alpha, delta = 175, 0.10, 0.10
    l = math.floor((n + 1) * alpha)
    rng = np.random.default_rng(42)
    cal = np.sort(rng.random((5_000, n)), axis=1)
    realised = 1.0 - cal[:, n - l]
    assert abs(realised.mean() - drift.marginal_null_rate(n, alpha)) < 0.002
    bound = drift.conditional_null_rate_bound(n, alpha, delta)
    assert abs(bound - 0.1259) < 5e-4
    assert abs(np.quantile(realised, 1 - delta) - bound) < 0.004


def test_conditional_bound_is_zero_when_no_p_value_can_reach_alpha():
    assert drift.conditional_null_rate_bound(50, 0.01, 0.10) == 0.0


@pytest.mark.parametrize("alpha", [0.0, 1.0, -0.1])
def test_invalid_alpha_raises(alpha):
    with pytest.raises(ValueError):
        drift.marginal_null_rate(175, alpha)


# --------------------------------------------------------------------------- #
# Signal A                                                                     #
# --------------------------------------------------------------------------- #
def test_novelty_binomial_matches_hand_computation():
    ref = _reference()
    sims = [0.50] * 7 + [0.90] * 43  # 7 below every reference score
    signal = drift.novelty_signal(sims, ref, alpha=0.10, delta=0.10)
    assert signal.flagged == 7
    expected = _hand_binomial_greater(7, 50, 17 / 176)
    assert signal.marginal_binomial_p == pytest.approx(expected, rel=1e-9)


def test_in_reference_window_is_not_novel():
    ref = _reference()
    signal = drift.novelty_signal(REF_SCORES, ref, alpha=0.10, delta=0.10)
    assert signal.marginal_binomial_p > 0.2
    assert signal.conditional_binomial_p > signal.marginal_binomial_p


def test_shifted_window_is_novel():
    ref = _reference()
    signal = drift.novelty_signal([0.55] * 60, ref, alpha=0.10, delta=0.10)
    assert signal.flagged == 60
    assert signal.marginal_binomial_p < 1e-30
    assert signal.conditional_binomial_p < 1e-20
    assert signal.ks_p < 1e-10


def test_skipped_retrieval_is_excluded_not_scored_as_zero():
    """top_similarity 0.0 on a skipped step means 'not measured'."""
    ref = _reference()
    records = [_record(0.8)] * 10 + [_record(0.0, retrieval="skipped")] * 5
    report = drift.detect(records, ref, expected_fingerprint=FP)
    assert report.novelty.n_scored == 10
    assert report.novelty.n_excluded_retrieval_skipped == 5
    assert report.novelty.flagged == 0
    assert report.n_records == 15


def test_all_skipped_is_insufficient_not_novel():
    ref = _reference()
    records = [_record(0.0, retrieval="skipped")] * 5
    report = drift.detect(records, ref, expected_fingerprint=FP)
    assert report.insufficient_data is True
    assert report.novelty is None


def test_empty_window_is_insufficient():
    report = drift.detect([], _reference(), expected_fingerprint=FP)
    assert report.insufficient_data is True
    assert report.novelty is None and report.rates is None
    assert report.deployment.fingerprint_mismatch is False


# --------------------------------------------------------------------------- #
# Signal B                                                                     #
# --------------------------------------------------------------------------- #
def test_rate_signals_against_reference():
    ref = _reference()  # escalation 0.2, tier-1 share 0.4
    records = ([_record(escalated=True, tier=1)] * 20
               + [_record(escalated=False, tier=2)] * 80)
    rates = drift.detect(records, ref, expected_fingerprint=FP).rates
    assert rates.escalation.observed_rate == 0.20
    assert rates.escalation.degenerate_reference is False
    assert rates.escalation.binomial_p_two_sided > 0.5
    assert rates.tier1_share.observed_rate == 0.20
    assert rates.tier1_share.binomial_p_two_sided < 1e-4


def test_zero_reference_rate_is_degenerate_not_p_zero():
    """The real in-domain reference escalates 0/175."""
    ref = _reference(n_escalated=0)
    rates = drift.detect([_record(escalated=True)], ref,
                         expected_fingerprint=FP).rates
    assert rates.escalation.degenerate_reference is True
    assert rates.escalation.binomial_p_two_sided is None
    assert rates.escalation.fisher_p_two_sided is None


def _hand_fisher_two_sided(a, b, c, d):
    """Sum of hypergeometric tables no more probable than the observed one."""
    row1, col1, total = a + b, a + c, a + b + c + d

    def prob(x):
        return (math.comb(col1, x) * math.comb(total - col1, row1 - x)
                / math.comb(total, row1))

    observed = prob(a)
    lo, hi = max(0, row1 - (total - col1)), min(row1, col1)
    return sum(prob(x) for x in range(lo, hi + 1)
               if prob(x) <= observed * (1 + 1e-7))


def test_fisher_matches_hand_computation():
    ref = _reference()  # 35/175 escalated
    records = ([_record(escalated=True)] * 18
               + [_record(escalated=False)] * 32)
    esc = drift.detect(records, ref, expected_fingerprint=FP).rates.escalation
    assert esc.fisher_p_two_sided == pytest.approx(
        _hand_fisher_two_sided(18, 32, 35, 140), rel=1e-9)


def test_fisher_is_more_conservative_than_binomial_against_estimate():
    """The binomial treats 35/175 as the true rate; Fisher does not."""
    ref = _reference()
    records = ([_record(escalated=True)] * 40
               + [_record(escalated=False)] * 110)
    esc = drift.detect(records, ref, expected_fingerprint=FP).rates.escalation
    assert esc.fisher_p_two_sided > esc.binomial_p_two_sided


def test_rate_reference_defaults_to_the_novelty_reference():
    ref = _reference()
    records = [_record(escalated=True)] * 5 + [_record()] * 20
    implicit = drift.detect(records, ref, expected_fingerprint=FP)
    explicit = drift.detect(records, ref, expected_fingerprint=FP,
                            rate_reference=ref)
    assert implicit == explicit


def test_rate_reference_scores_signal_b_only():
    """Signal A must not move when only the rate reference changes."""
    novelty_ref = _reference(n_escalated=0)
    rate_ref = _reference(n_escalated=39, n_tier1=33,
                          similarity_scores=[0.1] * N_REF,
                          similarity_scores_with_self=[0.1] * N_REF)
    records = [_record(sim=0.7, escalated=True)] * 10 + [_record()] * 40
    alone = drift.detect(records, novelty_ref, expected_fingerprint=FP)
    split = drift.detect(records, novelty_ref, expected_fingerprint=FP,
                         rate_reference=rate_ref)
    assert split.novelty == alone.novelty
    assert alone.rates.escalation.degenerate_reference is True
    assert split.rates.escalation.degenerate_reference is False
    assert split.rates.escalation.reference_rate == 39 / 175
    assert split.rates.tier1_share.reference_rate == 33 / 175


def test_category_mix_matching_reference():
    ref = _reference()
    records = ([_record(category="Database")] * 40
               + [_record(category="Network")] * 20
               + [_record(category="Storage")] * 10)
    mix = drift.detect(records, ref, expected_fingerprint=FP).rates.category_mix
    assert mix.p_value == pytest.approx(1.0)
    assert mix.unknown_categories == []
    assert mix.small_expected_counts is False


def test_unknown_category_leaves_chi_square_undefined():
    ref = _reference()
    records = [_record(category="Database"), _record(category="Printers")]
    mix = drift.detect(records, ref, expected_fingerprint=FP).rates.category_mix
    assert mix.unknown_categories == ["Printers"]
    assert mix.chi_square is None and mix.p_value is None


def test_small_window_flags_small_expected_counts():
    mix = drift.detect([_record()] * 4, _reference(),
                       expected_fingerprint=FP).rates.category_mix
    assert mix.small_expected_counts is True


# --------------------------------------------------------------------------- #
# Signal C                                                                     #
# --------------------------------------------------------------------------- #
def test_fingerprint_mismatch_is_separate_from_novelty():
    ref = _reference()
    same = [_record(0.8)] * 20
    mixed = [_record(0.8)] * 10 + [_record(0.8, fingerprint="b" * 12)] * 10
    a = drift.detect(same, ref, expected_fingerprint=FP)
    b = drift.detect(mixed, ref, expected_fingerprint=FP)
    assert a.deployment.fingerprint_mismatch is False
    assert b.deployment.fingerprint_mismatch is True
    assert b.deployment.fingerprint_counts == {FP: 10, "b" * 12: 10}
    assert a.novelty == b.novelty


# --------------------------------------------------------------------------- #
# Record and reference validation                                             #
# --------------------------------------------------------------------------- #
def test_record_schema_version_is_in_step_with_the_logger():
    from src.agent.logging_setup import DECISION_SCHEMA_VERSION

    assert drift.SUPPORTED_RECORD_SCHEMA_VERSION == DECISION_SCHEMA_VERSION


@pytest.mark.parametrize("overrides", [
    {"schema_version": 2},
    {"tier": 3},
    {"agent_status": {"classification": "ok"}},
])
def test_malformed_record_raises(overrides):
    with pytest.raises(ValueError):
        _record(**overrides)


def test_record_missing_similarity_raises_not_defaults():
    fields = _record().model_dump()
    del fields["top_similarity"]
    with pytest.raises(ValueError):
        drift.DecisionRecord.model_validate(fields)


def test_inconsistent_reference_raises():
    with pytest.raises(ValueError):
        _reference(similarity_scores=REF_SCORES[:-1])
    with pytest.raises(ValueError):
        _reference(n_tier1=N_REF + 1)


@pytest.mark.parametrize("live, fragment", [
    ({"embedding_model": "all-MiniLM-L6-v2"}, "embedding model"),
    ({"embedding_dim": 384}, "embedding dim"),
    ({"index_ntotal": 3999}, "index size"),
    ({"index_sha256": "0" * 64}, "hash"),
])
def test_stale_reference_is_refused(live, fragment):
    ok = dict(embedding_model="BAAI/bge-base-en-v1.5", embedding_dim=768,
              index_ntotal=4000, index_sha256="f" * 64)
    _reference().check_compatible(**ok)
    with pytest.raises(ArtifactError, match=fragment):
        _reference().check_compatible(**{**ok, **live})


# --------------------------------------------------------------------------- #
# The committed reference against the live artifacts                          #
# --------------------------------------------------------------------------- #
@pytest.mark.slow
def test_committed_reference_matches_live_artifacts(artifacts):
    """A rebuilt index or swapped encoder must invalidate the reference."""
    from src.agent.artifacts import load_drift_reference

    reference = load_drift_reference(artifacts)
    assert reference.n == 175
    assert reference.index_ntotal == artifacts.index.ntotal


@pytest.mark.slow
def test_committed_rate_reference_is_non_degenerate_and_live(artifacts):
    """Phase 4B: Signal B's reference exists so the escalation test is not
    None. 39/175 was counted two ways when it was built."""
    from src.agent.artifacts import load_drift_reference

    reference = load_drift_reference(artifacts, source="deployment")
    assert reference.n == 175
    assert reference.n_escalated == 39
    assert 0 < reference.escalation_rate < 1
    assert reference.provenance["escalation_counted_two_ways"][
        "escalated_direct_below_threshold"] == 39


def test_unknown_reference_source_is_refused():
    from src.agent.artifacts import load_drift_reference

    with pytest.raises(ValueError, match="source"):
        load_drift_reference(source="rolling")
