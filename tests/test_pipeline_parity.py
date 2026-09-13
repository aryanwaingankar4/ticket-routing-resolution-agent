"""
Parity against the pre-refactor goldens -- the safety net for Phase 0.

The consolidation of four duplicate pipelines into src/agent/ is required to
be BEHAVIOUR-PRESERVING. Every routing decision, tier assignment, confidence
and similarity score must reproduce exactly what the old code produced.

If one of these fails, the refactor introduced a behavioural change. That is
a bug, not an improvement -- the published results were measured on the old
behaviour, and silently shifting them would be exactly the failure mode this
project exists to guard against.

Goldens are regenerated with:
    venv\\Scripts\\python.exe tests/capture_goldens.py
"""

from __future__ import annotations

import pytest

from src.agent import pipeline
from src.agent.schemas import TicketIn

pytestmark = pytest.mark.slow

# Similarity and confidence are floats from the same deterministic ops, so
# they should match to well within float noise.
TOL = 1e-9


def _run(text, artifacts, ticket_id=None):
    return pipeline.run(
        TicketIn(title=text, ticket_id=ticket_id),
        artifacts=artifacts, generate_resolution=False, emit_log=False,
    )


def test_goldens_were_captured_under_current_config(adversarial_golden):
    """Goldens are only meaningful if captured at the live thresholds."""
    from src.agent.config import settings

    manifest = adversarial_golden["manifest"]
    assert manifest["similarity_threshold"] == (
        settings.rag.similarity_threshold
    )
    assert manifest["cascade_confidence_threshold"] == (
        settings.cascade.confidence_threshold
    )
    assert manifest["embedding_model"] == settings.models.embedding_model


def test_adversarial_parity(adversarial_tickets, adversarial_golden,
                            artifacts):
    golden = {r["id"]: r for r in adversarial_golden["rows"]}
    diffs = []

    for t in adversarial_tickets:
        want = golden[t["id"]]
        got = _run(t["text"], artifacts, t["id"])

        if got.classification.category != want["predicted_category"]:
            diffs.append(f"{t['id']} category: {got.classification.category}"
                         f" != {want['predicted_category']}")
        if int(got.classification.tier) != want["tier"]:
            diffs.append(f"{t['id']} tier: {int(got.classification.tier)}"
                         f" != {want['tier']}")
        if abs(got.top_similarity - want["rag_similarity"]) > TOL:
            diffs.append(f"{t['id']} similarity: {got.top_similarity!r}"
                         f" != {want['rag_similarity']!r}")
        if got.escalated != want["actual_escalate"]:
            diffs.append(f"{t['id']} escalated: {got.escalated}"
                         f" != {want['actual_escalate']}")

    assert not diffs, "Parity broken vs pre-refactor goldens:\n" + "\n".join(
        diffs
    )


def test_benchmark_parity(benchmark_tickets, benchmark_golden, artifacts):
    """All 45 tickets, every field, exact."""
    golden = {r["index"]: r for r in benchmark_golden["rows"]}
    diffs = []

    for i, t in enumerate(benchmark_tickets):
        want = golden[i]
        got = _run(t["text"], artifacts)

        if got.classification.category != want["predicted_category"]:
            diffs.append(f"idx {i} category: {got.classification.category}"
                         f" != {want['predicted_category']}")
        if int(got.classification.tier) != want["tier"]:
            diffs.append(f"idx {i} tier: {int(got.classification.tier)}"
                         f" != {want['tier']}")
        if abs(got.classification.tier1_conf
               - want["tier1_confidence"]) > TOL:
            diffs.append(f"idx {i} tier1_conf mismatch")
        if abs(got.top_similarity - want["rag_similarity"]) > TOL:
            diffs.append(f"idx {i} similarity mismatch")

    assert not diffs, "Parity broken vs pre-refactor goldens:\n" + "\n".join(
        diffs
    )


def test_benchmark_accuracy_unchanged(benchmark_tickets, artifacts):
    """32/45 is the measured BGE cascade baseline.

    Note this is NOT the 33/45 in the README's comparison table -- that is
    pure Tier-2 (frozen BGE + LogReg) with no cascade in front of it. The
    end-to-end cascade scores 32/45 because Tier-1 resolves four tickets
    itself at the 0.50 gate.
    """
    correct = sum(
        1 for t in benchmark_tickets
        if _run(t["text"], artifacts).classification.category == t["expected"]
    )
    assert correct == 32, f"benchmark accuracy moved: {correct}/45"
