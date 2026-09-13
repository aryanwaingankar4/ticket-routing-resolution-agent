"""
Schema validation.

These models replace five mutually incompatible ad-hoc dict shapes. The point
of typing them is that a name drift (tier1_conf vs tier1_confidence,
top_similarity vs rag_similarity) becomes a validation error at the boundary
instead of a silently-missing key three layers downstream.
"""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from src.agent import schemas as S


def test_combined_text_is_the_single_join_convention():
    t = S.TicketIn(title="VPN drops", description="every ten minutes")
    assert t.combined_text == "VPN drops every ten minutes"


def test_combined_text_handles_title_only():
    """Both fixed benchmark sets carry only a single text field."""
    assert S.TicketIn(title="Help").combined_text == "Help"
    assert S.TicketIn().combined_text == ""


def test_confidence_bounds_are_enforced():
    with pytest.raises(ValidationError):
        S.ClassificationResult(
            category="Network", confidence=1.4, tier=S.Tier.TIER1_TFIDF,
            tier1_pred="Network", tier1_conf=0.5,
        )


def test_unknown_fields_are_rejected():
    """extra='forbid' catches a renamed field instead of ignoring it."""
    with pytest.raises(ValidationError):
        S.ClassificationResult(
            category="Network", confidence=0.9, tier=S.Tier.TIER1_TFIDF,
            tier1_pred="Network", tier1_conf=0.5,
            tier1_confidence=0.5,          # the old, drifted spelling
        )


def test_top_similarity_uses_the_highest_score():
    r = S.RetrievalResult(retrieved=[
        S.RetrievedTicket(id=1, similarity=0.91),
        S.RetrievedTicket(id=2, similarity=0.44),
    ])
    assert r.top_similarity == 0.91


def test_top_similarity_on_empty_retrieval_is_zero():
    assert S.RetrievalResult().top_similarity == 0.0


def test_status_and_reason_are_enums_not_free_strings():
    assert S.ResolutionStatus.AUTO_RESOLVED.value == "auto_resolved"
    assert S.EscalationReason.LOW_RETRIEVAL_SIMILARITY.value == (
        "low_retrieval_similarity"
    )
    with pytest.raises(ValueError):
        S.ResolutionStatus("resolved_automatically")


def test_pipeline_result_round_trips():
    result = S.PipelineResult(
        ticket=S.TicketIn(title="Disk full", ticket_id="t9"),
        classification=S.ClassificationResult(
            category="Storage", confidence=0.88,
            tier=S.Tier.TIER1_TFIDF, tier1_pred="Storage", tier1_conf=0.88,
        ),
        retrieval=S.RetrievalResult(retrieved=[
            S.RetrievedTicket(id=3, similarity=0.77),
        ]),
        decision=S.EscalationDecision(
            escalated=False, reason=S.EscalationReason.NONE,
            threshold_applied=0.67, observed_value=0.77,
        ),
        status=S.ResolutionStatus.AUTO_RESOLVED,
        config_fingerprint="abc123def456",
    )
    restored = S.PipelineResult.model_validate(
        result.model_dump(mode="json")
    )
    assert restored.top_similarity == 0.77
    assert restored.escalated is False
    assert restored.classification.tier is S.Tier.TIER1_TFIDF
