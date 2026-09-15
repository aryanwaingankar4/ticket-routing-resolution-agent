"""
Agent boundaries and orchestration (Phase 3B).

The restructure is only worth anything if the boundaries are real. These
tests check the three properties that distinguish an agent architecture from
renamed functions:

  * each agent DECLARES what it needs, and the declaration is checked against
    Artifacts rather than trusted;
  * the orchestrator owns every routing decision, and those decisions can be
    exercised without loading a model;
  * what ran, and what deliberately did not, is recorded.

The gate tests deliberately load nothing. An escalation policy that can only
be tested by standing up BGE and FAISS is a policy nobody will test.
"""

from __future__ import annotations

import dataclasses

import pytest

from src.agent import agents as agents_mod
from src.agent import orchestrator, pipeline
from src.agent.artifacts import Artifacts
from src.agent.config import settings
from src.agent.errors import ArtifactError
from src.agent.schemas import (
    ClassificationResult,
    EscalationReason,
    RetrievalResult,
    RetrievedTicket,
    StepStatus,
    Tier,
    TicketIn,
)

AGENT_CLASSES = (
    agents_mod.ClassificationAgent,
    agents_mod.RetrievalAgent,
    agents_mod.ResolutionAgent,
)


def _decision_view(result):
    """Everything that decides a ticket's fate, and nothing that varies run
    to run (latency, and the traces that carry it)."""
    return {
        "category": result.classification.category,
        "tier": int(result.classification.tier),
        "confidence": result.classification.confidence,
        "tier1_conf": result.classification.tier1_conf,
        "top_similarity": result.top_similarity,
        "escalated": result.escalated,
        "reason": result.decision.reason.value,
        "threshold": result.decision.threshold_applied,
        "status": result.status.value,
    }


# --------------------------------------------------------------------------- #
# The contract. No artifacts needed.                                           #
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("agent_cls", AGENT_CLASSES)
def test_every_agent_declares_a_name_and_real_dependencies(agent_cls):
    """`requires` is the agent boundary written down. It must be accurate."""
    known = {f.name for f in dataclasses.fields(Artifacts)}

    assert isinstance(agent_cls.name, str) and agent_cls.name
    assert isinstance(agent_cls.requires, tuple) and agent_cls.requires
    assert not set(agent_cls.requires) - known, (
        f"{agent_cls.__name__} declares dependencies that are not fields of "
        f"Artifacts: {sorted(set(agent_cls.requires) - known)}"
    )
    assert callable(agent_cls.run)


def test_agent_names_are_distinct():
    """Traces and logs are keyed by name, so a collision would silently merge
    two agents' records into one."""
    names = [cls.name for cls in AGENT_CLASSES]
    assert len(set(names)) == len(names)


def test_a_typo_in_requires_fails_loudly():
    """A misspelled dependency must not degrade into 'no dependencies'.

    Checked before any artifact is touched, so it fires even with nothing
    loaded -- which is the point: it is a code defect, not a missing file.
    """
    class BrokenAgent(agents_mod._BaseAgent):
        name = "broken"
        requires = ("tier1_vectorizor",)          # note the typo

    with pytest.raises(ArtifactError, match="fields of Artifacts"):
        BrokenAgent(None)


# --------------------------------------------------------------------------- #
# Routing policy, exercised without loading a model.                           #
# --------------------------------------------------------------------------- #
def test_rag_gate_uses_the_calibrated_threshold_not_a_literal():
    threshold = settings.rag.similarity_threshold

    below = RetrievalResult(retrieved=[
        RetrievedTicket(id=1, similarity=threshold - 0.01),
    ])
    at = RetrievalResult(retrieved=[
        RetrievedTicket(id=1, similarity=threshold),
    ])

    decision = orchestrator.rag_gate(below)
    assert decision is not None
    assert decision.escalated is True
    assert decision.reason is EscalationReason.LOW_RETRIEVAL_SIMILARITY
    assert decision.threshold_applied == threshold

    assert orchestrator.rag_gate(at) is None, (
        "the gate is >= threshold, not > threshold"
    )


def test_rag_gate_escalates_when_nothing_was_retrieved():
    decision = orchestrator.rag_gate(RetrievalResult())
    assert decision is not None
    assert decision.reason is EscalationReason.NO_RETRIEVAL_RESULTS
    assert decision.observed_value == 0.0


def test_filing_gate_stays_off_by_default():
    """Only the batch adapter enables it; the live demo never did."""
    hopeless = ClassificationResult(
        category="Network", confidence=0.01, tier=Tier.TIER1_TFIDF,
        tier1_pred="Network", tier1_conf=0.01,
    )
    assert settings.cascade.filing_gate_enabled is False
    assert orchestrator.filing_gate(hopeless) is None


# --------------------------------------------------------------------------- #
# Behaviour against the real artifacts.                                        #
# --------------------------------------------------------------------------- #
@pytest.mark.slow
def test_agents_satisfy_the_shared_contract(artifacts):
    for agent_cls in (agents_mod.ClassificationAgent,
                      agents_mod.RetrievalAgent):
        assert isinstance(agent_cls(artifacts), agents_mod.TicketAgent)


@pytest.mark.slow
def test_classification_and_retrieval_need_no_gemini_client(artifacts):
    """The evaluation path loads artifacts with require_gemini=False and must
    keep working with no API key and no quota."""
    assert artifacts.gemini_client is None
    agents_mod.ClassificationAgent(artifacts)
    agents_mod.RetrievalAgent(artifacts)


@pytest.mark.slow
def test_resolution_agent_refuses_to_construct_without_its_dependency(
    artifacts,
):
    """Prove the dependency guard fires rather than merely existing."""
    with pytest.raises(ArtifactError, match="gemini_client"):
        agents_mod.ResolutionAgent(artifacts)


@pytest.mark.slow
def test_facade_and_orchestrator_agree(benchmark_tickets, artifacts):
    """pipeline.run() must stay a transparent delegate.

    Every consumer in the project imports the façade; if it ever diverged
    from the orchestrator it would be the fifth pipeline copy.
    """
    for ticket in benchmark_tickets[:10]:
        payload = TicketIn(title=ticket["text"])
        through_facade = pipeline.run(
            payload, artifacts=artifacts, generate_resolution=False,
            emit_log=False,
        )
        through_orchestrator = orchestrator.run(
            payload, artifacts=artifacts, generate_resolution=False,
            emit_log=False,
        )
        assert _decision_view(through_facade) == _decision_view(
            through_orchestrator
        )


@pytest.mark.slow
def test_escalated_ticket_records_a_skipped_resolution_step(
    adversarial_tickets, artifacts,
):
    """The trace is the positive evidence that no LLM call was made.

    An absent resolution step would be ambiguous -- it could mean the agent
    was skipped, or that tracing missed it. SKIPPED with a reason is not.
    """
    escalated = None
    for ticket in adversarial_tickets:
        result = orchestrator.run(
            TicketIn(title=ticket["text"], ticket_id=ticket["id"]),
            artifacts=artifacts, generate_resolution=False, emit_log=False,
        )
        if result.escalated:
            escalated = result
            break

    assert escalated is not None, "no adversarial ticket escalated"
    assert [s.agent for s in escalated.steps] == [
        "classification", "retrieval", "resolution",
    ]

    by_agent = {s.agent: s for s in escalated.steps}
    assert by_agent["classification"].status is StepStatus.OK
    assert by_agent["retrieval"].status is StepStatus.OK
    assert by_agent["resolution"].status is StepStatus.SKIPPED
    assert by_agent["resolution"].detail == escalated.decision.reason.value


@pytest.mark.slow
def test_disabled_generation_is_recorded_not_silent(
    benchmark_tickets, artifacts,
):
    for ticket in benchmark_tickets:
        result = orchestrator.run(
            TicketIn(title=ticket["text"]), artifacts=artifacts,
            generate_resolution=False, emit_log=False,
        )
        if result.escalated:
            continue
        resolution = {s.agent: s for s in result.steps}["resolution"]
        assert resolution.status is StepStatus.SKIPPED
        assert resolution.detail == "generation_disabled"
        return

    pytest.fail("no benchmark ticket cleared the RAG gate")


@pytest.mark.slow
def test_every_agent_appears_in_every_trace(benchmark_tickets, artifacts):
    """Three agents, three steps, always -- whether they ran or not."""
    for ticket in benchmark_tickets[:10]:
        result = orchestrator.run(
            TicketIn(title=ticket["text"]), artifacts=artifacts,
            generate_resolution=False, emit_log=False,
        )
        assert {s.agent for s in result.steps} == {
            cls.name for cls in AGENT_CLASSES
        }
