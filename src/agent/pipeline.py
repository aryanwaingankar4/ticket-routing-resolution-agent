"""
The pipeline: classify -> retrieve -> decide -> (maybe) resolve.

This is the single implementation that replaces four copies:
    streamlit_app.run_pipeline()                     (line 567)
    process_ticket_batch.process()                   (line 292)
    test_adversarial.run_ticket_through_pipeline()   (line 620)
    suggest_resolution.suggest_resolution_for_ticket()  (line 514)

THE TWO GATES
-------------
Gate 1 (cascade, threshold 0.50) selects WHICH MODEL answers. It never
escalates to a human; it escalates from the cheap model to the strong one.

Gate 2 (RAG similarity, threshold 0.67) decides whether a human is needed.
Below it, the Gemini call is skipped ENTIRELY -- not made and discarded.
That is the whole point: a forced-choice classifier can never refuse to emit
a category, but the retrieval-confidence gate can refuse to emit a
fabricated resolution.

THE FILING GATE
---------------
process_ticket_batch.py applied a third gate on final classification
confidence before filing a ticket; the other three copies did not. Rather
than silently imposing one behaviour on all callers, that gate is
config-controlled (settings.cascade.filing_gate_enabled) and defaults OFF,
preserving the live-demo and adversarial-test behaviour exactly. The batch
adapter turns it on.
"""

from __future__ import annotations

import time

from src.agent.artifacts import Artifacts, load_artifacts
from src.agent.classifier import classify
from src.agent.config import config_fingerprint, settings
from src.agent.errors import LLMError
from src.agent.logging_setup import log_decision
from src.agent.retriever import retrieve
from src.agent.schemas import (
    EscalationDecision,
    EscalationReason,
    PipelineResult,
    ResolutionStatus,
    TicketIn,
)


def run(ticket: TicketIn,
        artifacts: Artifacts | None = None,
        generate_resolution: bool = True,
        emit_log: bool = True) -> PipelineResult:
    """Run one ticket end to end.

    `generate_resolution=False` stops after the escalation decision. Every
    evaluation path in this project uses that mode: the decision is fully
    determined before any LLM call, so routing benchmarks need no API key
    and burn no quota.
    """
    started = time.perf_counter()

    if artifacts is None:
        artifacts = load_artifacts(require_gemini=generate_resolution)

    text = ticket.combined_text

    # ---- Gate 1: which model answers --------------------------------------
    classification = classify(text, artifacts)

    # ---- Optional filing gate (batch-processing behaviour) ----------------
    if (settings.cascade.filing_gate_enabled
            and classification.confidence
            < settings.cascade.filing_confidence_threshold):
        decision = EscalationDecision(
            escalated=True,
            reason=EscalationReason.LOW_CLASSIFICATION_CONFIDENCE,
            threshold_applied=settings.cascade.filing_confidence_threshold,
            observed_value=classification.confidence,
        )
        return _finish(
            ticket, classification, _empty_retrieval(), decision,
            None, ResolutionStatus.ESCALATED, started, emit_log,
        )

    # ---- Retrieval ---------------------------------------------------------
    retrieval = retrieve(text, artifacts)
    top_similarity = retrieval.top_similarity
    threshold = settings.rag.similarity_threshold

    # ---- Gate 2: does a human need to see this? ---------------------------
    if not retrieval.retrieved:
        decision = EscalationDecision(
            escalated=True,
            reason=EscalationReason.NO_RETRIEVAL_RESULTS,
            threshold_applied=threshold,
            observed_value=0.0,
        )
        return _finish(
            ticket, classification, retrieval, decision, None,
            ResolutionStatus.NEEDS_HUMAN_RESOLUTION, started, emit_log,
        )

    if top_similarity < threshold:
        decision = EscalationDecision(
            escalated=True,
            reason=EscalationReason.LOW_RETRIEVAL_SIMILARITY,
            threshold_applied=threshold,
            observed_value=top_similarity,
        )
        return _finish(
            ticket, classification, retrieval, decision, None,
            ResolutionStatus.NEEDS_HUMAN_RESOLUTION, started, emit_log,
        )

    decision = EscalationDecision(
        escalated=False,
        reason=EscalationReason.NONE,
        threshold_applied=threshold,
        observed_value=top_similarity,
    )

    # ---- Resolution --------------------------------------------------------
    if not generate_resolution:
        return _finish(
            ticket, classification, retrieval, decision, None,
            ResolutionStatus.AUTO_RESOLVED, started, emit_log,
        )

    from src.agent.resolver import generate

    try:
        suggestion = generate(ticket, retrieval, artifacts)
    except LLMError as exc:
        failed = EscalationDecision(
            escalated=True,
            reason=EscalationReason.LLM_CALL_FAILED,
            threshold_applied=threshold,
            observed_value=top_similarity,
        )
        return _finish(
            ticket, classification, retrieval, failed, None,
            ResolutionStatus.LLM_CALL_FAILED, started, emit_log,
            error_kind=type(exc).__name__, error_message=str(exc),
        )

    return _finish(
        ticket, classification, retrieval, decision, suggestion,
        ResolutionStatus.AUTO_RESOLVED, started, emit_log,
    )


def _empty_retrieval():
    from src.agent.schemas import RetrievalResult

    return RetrievalResult(retrieved=[])


def _finish(ticket, classification, retrieval, decision, suggestion,
            status, started, emit_log, error_kind=None,
            error_message=None) -> PipelineResult:
    result = PipelineResult(
        ticket=ticket,
        classification=classification,
        retrieval=retrieval,
        decision=decision,
        suggestion=suggestion,
        status=status,
        config_fingerprint=config_fingerprint(),
        latency_ms=(time.perf_counter() - started) * 1000.0,
        error_kind=error_kind,
        error_message=error_message,
    )
    if emit_log:
        log_decision(result)
    return result
