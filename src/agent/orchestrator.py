"""
The orchestrator: it owns the sequence, and it owns every routing decision.

WHY THIS EXISTS
---------------
Phase 0 collapsed four copies of the pipeline into one function. That function
was already an orchestrator in effect, but the agents it coordinated were not
agents -- they were free functions it called, and the gates were `if`
statements interleaved with them.

Phase 3B separates the two halves. agents.py holds units that declare what
they need and do one thing. This module holds the part that decides: which
agent runs, in what order, and whether the ticket goes to a human instead.
No agent reads a threshold; no agent knows what runs after it.

That separation is not cosmetic. The escalation policy is the claim this
project rests on, and it now sits in one file, as two small pure functions
(_filing_gate, _rag_gate) that can be read and tested without loading a model.

THE TWO GATES
-------------
Gate 1 (cascade, threshold 0.50) is INSIDE the classification agent -- it
selects which model answers and never escalates to a human, so it is the
agent's business rather than the orchestrator's.

Gate 2 (RAG similarity, threshold 0.67) is the orchestrator's. Below it, the
resolution agent is never constructed and Gemini is never called -- not called
and discarded. A forced-choice classifier can never refuse to emit a category,
but the retrieval-confidence gate can refuse to emit a fabricated resolution.

THE FILING GATE
---------------
process_ticket_batch.py applies a third gate on final classification
confidence before filing a ticket; the other consumers do not. It stays
config-controlled (settings.cascade.filing_gate_enabled) and defaults OFF,
preserving live-demo and adversarial-test behaviour exactly.

TRACES
------
Every step appends a StepTrace, including the ones that did not run. A
resolution step marked SKIPPED is the positive evidence that the RAG gate held
and no LLM call was made -- more useful after the fact than its absence would
be, and the per-agent history drift detection will read later.

PARITY
------
This is a restructure. The branch order, the thresholds, the statuses and the
values in PipelineResult are unchanged from pipeline.run(), and the goldens
prove it. The one thing that moved: the resolution agent is constructed inside
the try/except that already mapped a failed draft onto LLM_CALL_FAILED, so a
missing Gemini client is recorded with error_kind "ArtifactError" where it used
to say "LLMError". Same status, same escalation, same gate -- only the name of
the exception on an unreachable-in-practice path differs.
"""

from __future__ import annotations

import time

from src.agent.agents import (
    ClassificationAgent,
    ResolutionAgent,
    RetrievalAgent,
)
from src.agent.artifacts import Artifacts, load_artifacts
from src.agent.config import config_fingerprint, settings
from src.agent.errors import ArtifactError, LLMError
from src.agent.logging_setup import log_decision
from src.agent.schemas import (
    ClassificationResult,
    EscalationDecision,
    EscalationReason,
    PipelineResult,
    ResolutionStatus,
    RetrievalResult,
    StepStatus,
    StepTrace,
    TicketIn,
)


# --------------------------------------------------------------------------- #
# Routing policy. Pure, model-free, and readable side by side on purpose.      #
# Each returns an EscalationDecision to escalate, or None to continue.         #
# --------------------------------------------------------------------------- #
def _filing_gate(
    classification: ClassificationResult,
) -> EscalationDecision | None:
    """Batch-processing behaviour: refuse to file a low-confidence ticket."""
    if not settings.cascade.filing_gate_enabled:
        return None

    threshold = settings.cascade.filing_confidence_threshold
    if classification.confidence >= threshold:
        return None

    return EscalationDecision(
        escalated=True,
        reason=EscalationReason.LOW_CLASSIFICATION_CONFIDENCE,
        threshold_applied=threshold,
        observed_value=classification.confidence,
    )


def _rag_gate(retrieval: RetrievalResult) -> EscalationDecision | None:
    """The human-escalation gate. Below it, the LLM is never called."""
    threshold = settings.rag.similarity_threshold

    if not retrieval.retrieved:
        return EscalationDecision(
            escalated=True,
            reason=EscalationReason.NO_RETRIEVAL_RESULTS,
            threshold_applied=threshold,
            observed_value=0.0,
        )

    if retrieval.top_similarity < threshold:
        return EscalationDecision(
            escalated=True,
            reason=EscalationReason.LOW_RETRIEVAL_SIMILARITY,
            threshold_applied=threshold,
            observed_value=retrieval.top_similarity,
        )

    return None


# --------------------------------------------------------------------------- #
# Step bookkeeping.                                                            #
# --------------------------------------------------------------------------- #
def _elapsed_ms(started: float) -> float:
    return (time.perf_counter() - started) * 1000.0


def _timed(steps: list[StepTrace], name: str, fn, *args, **kwargs):
    """Run one agent operation, recording how it went either way."""
    started = time.perf_counter()
    try:
        result = fn(*args, **kwargs)
    except Exception as exc:
        steps.append(StepTrace(
            agent=name, status=StepStatus.FAILED,
            latency_ms=_elapsed_ms(started), detail=type(exc).__name__,
        ))
        raise
    steps.append(StepTrace(
        agent=name, status=StepStatus.OK, latency_ms=_elapsed_ms(started),
    ))
    return result


def _skipped(steps: list[StepTrace], name: str, reason: str) -> None:
    """Record an agent that deliberately did not run, and why."""
    steps.append(StepTrace(
        agent=name, status=StepStatus.SKIPPED, latency_ms=0.0, detail=reason,
    ))


# --------------------------------------------------------------------------- #
# The sequence.                                                                #
# --------------------------------------------------------------------------- #
def run(ticket: TicketIn,
        artifacts: Artifacts | None = None,
        generate_resolution: bool = True,
        emit_log: bool = True) -> PipelineResult:
    """Run one ticket end to end.

    `generate_resolution=False` stops after the escalation decision. Every
    evaluation path in this project uses that mode: the decision is fully
    determined before any LLM call, so routing benchmarks need no API key and
    burn no quota.
    """
    started = time.perf_counter()
    steps: list[StepTrace] = []

    if artifacts is None:
        artifacts = load_artifacts(require_gemini=generate_resolution)

    classification_agent = ClassificationAgent(artifacts)
    retrieval_agent = RetrievalAgent(artifacts)

    # ---- Classification ----------------------------------------------------
    classification = _timed(
        steps, classification_agent.name, classification_agent.run, ticket,
    )

    # ---- Optional filing gate (batch-processing behaviour) ----------------
    filing = _filing_gate(classification)
    if filing is not None:
        _skipped(steps, retrieval_agent.name, filing.reason.value)
        _skipped(steps, ResolutionAgent.name, filing.reason.value)
        return _finish(
            ticket, classification, RetrievalResult(), filing, None,
            ResolutionStatus.ESCALATED, started, steps, emit_log,
        )

    # ---- Retrieval ---------------------------------------------------------
    retrieval = _timed(
        steps, retrieval_agent.name, retrieval_agent.run, ticket,
    )

    # ---- Gate 2: does a human need to see this? ---------------------------
    escalation = _rag_gate(retrieval)
    if escalation is not None:
        _skipped(steps, ResolutionAgent.name, escalation.reason.value)
        return _finish(
            ticket, classification, retrieval, escalation, None,
            ResolutionStatus.NEEDS_HUMAN_RESOLUTION, started, steps, emit_log,
        )

    decision = EscalationDecision(
        escalated=False,
        reason=EscalationReason.NONE,
        threshold_applied=settings.rag.similarity_threshold,
        observed_value=retrieval.top_similarity,
    )

    # ---- Resolution --------------------------------------------------------
    if not generate_resolution:
        _skipped(steps, ResolutionAgent.name, "generation_disabled")
        return _finish(
            ticket, classification, retrieval, decision, None,
            ResolutionStatus.AUTO_RESOLVED, started, steps, emit_log,
        )

    try:
        resolution_agent = ResolutionAgent(artifacts)
        suggestion = _timed(
            steps, resolution_agent.name, resolution_agent.run,
            ticket, retrieval,
        )
    except (LLMError, ArtifactError) as exc:
        failed = EscalationDecision(
            escalated=True,
            reason=EscalationReason.LLM_CALL_FAILED,
            threshold_applied=settings.rag.similarity_threshold,
            observed_value=retrieval.top_similarity,
        )
        return _finish(
            ticket, classification, retrieval, failed, None,
            ResolutionStatus.LLM_CALL_FAILED, started, steps, emit_log,
            error_kind=type(exc).__name__, error_message=str(exc),
        )

    return _finish(
        ticket, classification, retrieval, decision, suggestion,
        ResolutionStatus.AUTO_RESOLVED, started, steps, emit_log,
    )


def _finish(ticket, classification, retrieval, decision, suggestion,
            status, started, steps, emit_log, error_kind=None,
            error_message=None) -> PipelineResult:
    result = PipelineResult(
        ticket=ticket,
        classification=classification,
        retrieval=retrieval,
        decision=decision,
        suggestion=suggestion,
        status=status,
        config_fingerprint=config_fingerprint(),
        latency_ms=_elapsed_ms(started),
        error_kind=error_kind,
        error_message=error_message,
        steps=steps,
    )
    if emit_log:
        log_decision(result)
    return result
