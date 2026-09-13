"""
The 9-ticket adversarial escalation set -- this project's primary regression
gate, now as real tests.

Each ticket asserts the human-escalation gate fires (or correctly does not)
on inputs designed to break it: off-topic questions, vague multi-category
complaints, near-empty text, and unusual register. Six of the nine would
otherwise receive a confidently fabricated resolution.

Two of the nine are CONTROL cases that must NOT escalate (adv_03, adv_05,
adv_09) -- they guard against the opposite failure, a gate so tight that
genuinely answerable tickets get pushed to a human unnecessarily.

Gemini is never called: the escalation decision is fully determined before
any LLM call.
"""

from __future__ import annotations

import pytest

from src.agent import pipeline
from src.agent.config import settings
from src.agent.schemas import EscalationReason, TicketIn

pytestmark = pytest.mark.slow


def _ids(tickets):
    return [t["id"] for t in tickets]


def test_adversarial_set_is_intact(adversarial_tickets):
    """The set is a fixed reference point and must not drift."""
    assert len(adversarial_tickets) == 9
    assert _ids(adversarial_tickets) == [f"adv_{i:02d}" for i in range(1, 10)]


def test_all_nine_escalate_as_expected(adversarial_tickets, artifacts):
    """The headline regression gate: 9/9."""
    failures = []
    for t in adversarial_tickets:
        result = pipeline.run(
            TicketIn(title=t["text"], ticket_id=t["id"]),
            artifacts=artifacts, generate_resolution=False, emit_log=False,
        )
        if result.escalated != bool(t["expected_escalate"]):
            failures.append(
                f"{t['id']}: expected escalate="
                f"{t['expected_escalate']}, got {result.escalated} "
                f"(similarity {result.top_similarity:.4f} vs threshold "
                f"{settings.rag.similarity_threshold})"
            )
    assert not failures, "Adversarial escalation regressions:\n" + "\n".join(
        failures
    )


def test_escalations_cite_the_similarity_gate(adversarial_tickets, artifacts):
    """An escalation must record WHY, not just that it happened."""
    for t in adversarial_tickets:
        if not t["expected_escalate"]:
            continue
        result = pipeline.run(
            TicketIn(title=t["text"], ticket_id=t["id"]),
            artifacts=artifacts, generate_resolution=False, emit_log=False,
        )
        assert result.decision.reason in (
            EscalationReason.LOW_RETRIEVAL_SIMILARITY,
            EscalationReason.NO_RETRIEVAL_RESULTS,
        )
        assert result.decision.observed_value < (
            settings.rag.similarity_threshold
        )


def test_off_topic_tickets_never_reach_the_llm(adversarial_tickets,
                                               artifacts):
    """The weather and pizza tickets are the clearest demonstration that the
    gate prevents fabricated output."""
    off_topic = [t for t in adversarial_tickets
                 if t["category_type"] == "off_topic"]
    assert off_topic, "expected off-topic tickets in the adversarial set"

    for t in off_topic:
        result = pipeline.run(
            TicketIn(title=t["text"], ticket_id=t["id"]),
            artifacts=artifacts, generate_resolution=False, emit_log=False,
        )
        assert result.escalated is True
        assert result.suggestion is None
