"""
Typed schemas for every pipeline stage.

WHY THIS EXISTS
---------------
Before this module the repo contained zero dataclasses, NamedTuples or
TypedDicts -- every stage passed ad-hoc dicts, and five mutually incompatible
shapes had drifted apart. The same value was called `tier1_conf` in
streamlit_app.py and `tier1_confidence` in test_adversarial_escalation.py;
`top_similarity` in one and `rag_similarity` in the other; `escalated` in one
and `actual_escalate` in the other. Resolution status was a free string
("auto_resolved", "needs_human_resolution", "gemini_call_failed") with no
enumeration, so a typo could never be caught.

These models give one name per concept, validated at every boundary.

NAMING NOTE
-----------
Where the old names disagreed, the streamlit_app.py spelling wins, since that
is the live demo path and the one the adversarial regression test compares
against. RetrievedTicket adopts the existing canonical shape from
suggest_resolution.py:346-354 verbatim.
"""

from __future__ import annotations

from enum import Enum

from pydantic import BaseModel, ConfigDict, Field


class _Model(BaseModel):
    model_config = ConfigDict(extra="forbid", protected_namespaces=())


class Tier(int, Enum):
    """Which cascade tier produced the classification."""

    TIER1_TFIDF = 1
    TIER2_EMBEDDING = 2


class EscalationReason(str, Enum):
    """Why a ticket was routed to a human instead of being auto-resolved.

    Replaces the free-string statuses in process_ticket_batch.py:70-81.
    """

    NONE = "none"
    LOW_RETRIEVAL_SIMILARITY = "low_retrieval_similarity"
    LOW_CLASSIFICATION_CONFIDENCE = "low_classification_confidence"
    NO_RETRIEVAL_RESULTS = "no_retrieval_results"
    LLM_CALL_FAILED = "llm_call_failed"


class ResolutionStatus(str, Enum):
    AUTO_RESOLVED = "auto_resolved"
    ESCALATED = "escalated"
    NEEDS_HUMAN_RESOLUTION = "needs_human_resolution"
    LLM_CALL_FAILED = "llm_call_failed"


class TicketIn(_Model):
    """A ticket entering the pipeline.

    `description` is optional because several fixed benchmark sets carry only
    a single `text` field (the 9-ticket adversarial set and the 45-ticket
    expanded benchmark both do). `combined_text` is the single definition of
    how title and description are joined -- previously duplicated in three
    places plus two inline f-strings.
    """

    title: str = ""
    description: str = ""
    ticket_id: str | None = None

    @property
    def combined_text(self) -> str:
        return f"{self.title} {self.description}".strip()


class ClassificationResult(_Model):
    category: str
    confidence: float = Field(ge=0.0, le=1.0)
    tier: Tier
    tier1_pred: str
    tier1_conf: float = Field(ge=0.0, le=1.0)


class RetrievedTicket(_Model):
    """One retrieved neighbour.

    Field-for-field the canonical shape already produced by
    suggest_resolution.retrieve_similar_tickets() (lines 346-354).
    """

    id: str | int
    title: str = ""
    description: str = ""
    category: str = ""
    resolution: str = ""
    priority: str = ""
    similarity: float


class RetrievalResult(_Model):
    retrieved: list[RetrievedTicket] = Field(default_factory=list)

    @property
    def top_similarity(self) -> float:
        return self.retrieved[0].similarity if self.retrieved else 0.0


class EscalationDecision(_Model):
    escalated: bool
    reason: EscalationReason = EscalationReason.NONE
    threshold_applied: float
    observed_value: float


class ResolutionSuggestion(_Model):
    text: str
    llm_model: str
    grounded_ticket_ids: list[str | int] = Field(default_factory=list)


class StepStatus(str, Enum):
    """What happened to one agent on one request."""

    OK = "ok"
    SKIPPED = "skipped"
    FAILED = "failed"


class StepTrace(_Model):
    """One agent's participation in a single request.

    The orchestrator records these so a decision can be read back as a
    sequence of agent actions rather than as one opaque outcome. `SKIPPED` is
    as informative as `OK` here: a resolution step marked skipped is the
    positive evidence that the LLM was never called, which is the property the
    RAG gate exists to provide.

    logging_setup.py was written in Phase 1 on the basis that decision records
    are "the raw input for drift detection". These are the per-agent half of
    that record.
    """

    agent: str
    status: StepStatus
    latency_ms: float
    detail: str | None = None


class ConformalPrediction(_Model):
    """A conformal prediction set for one ticket.

    An EMPTY prediction_set is meaningful, not an error: every label was more
    nonconforming than anything seen during calibration, i.e. a strong
    out-of-distribution signal. It is never backfilled with the argmax.

    `alpha` records the nominal error rate the set was built at. The coverage
    guarantee holds only if calibration and deployment data are exchangeable;
    on this project's data that holds for Tier-2 but demonstrably not for
    Tier-1, so the tier is recorded alongside.
    """

    prediction_set: list[str] = Field(default_factory=list)
    alpha: float
    quantile: float | None = None
    score_function: str = "lac"
    mondrian: bool = False
    tier: Tier | None = None

    @property
    def set_size(self) -> int:
        return len(self.prediction_set)

    @property
    def is_singleton(self) -> bool:
        return len(self.prediction_set) == 1

    @property
    def is_empty(self) -> bool:
        return not self.prediction_set


class PipelineResult(_Model):
    """The complete outcome of one ticket through the pipeline.

    `config_fingerprint` is stamped on every result so a stored row always
    carries the exact thresholds and model identity that produced it -- making
    a stale-artifact mismatch detectable after the fact instead of silent.
    """

    ticket: TicketIn
    classification: ClassificationResult
    retrieval: RetrievalResult
    decision: EscalationDecision
    suggestion: ResolutionSuggestion | None = None
    status: ResolutionStatus
    config_fingerprint: str
    latency_ms: float | None = None

    # Populated only when the LLM call failed. `error_kind` is the exception
    # class name from errors.py ("RateLimitError", "AuthError", ...), so the
    # UI and the batch writer can both branch on it without re-parsing the
    # message text -- which is how three near-identical error ladders came to
    # exist in the first place.
    error_kind: str | None = None
    error_message: str | None = None

    # Populated only when settings.conformal.enabled. Default None keeps
    # results byte-identical to the Phase 0 goldens.
    conformal: ConformalPrediction | None = None

    # Per-agent trace, populated by the orchestrator. Default-empty for the
    # same reason `conformal` defaults to None: stored results and the Phase 0
    # goldens must be unaffected by a field being added here.
    steps: list[StepTrace] = Field(default_factory=list)

    @property
    def escalated(self) -> bool:
        return self.decision.escalated

    @property
    def top_similarity(self) -> float:
        return self.retrieval.top_similarity
