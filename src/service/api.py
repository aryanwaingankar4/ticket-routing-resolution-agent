"""
The HTTP surface: the agents, the policy, and the whole pipeline over FastAPI.

WHY THIS EXISTS
---------------
Phase 3A made Tier-1 a persisted artifact so a per-request service would not
refit a model at boot. Phase 3B gave each agent declared dependencies and put
every routing decision in one orchestrator. This is what that was for: the
agents become reachable individually, so an external workflow (Phase 3D's n8n)
can drive them and show the routing visually.

WHAT THE SERVICE DOES NOT DO
----------------------------
`/triage` does NOT call the per-agent endpoints over HTTP. It calls
pipeline.run() in-process. Two reasons, both load-bearing:

  * golden parity must not depend on a running server;
  * network hops would add failure modes to the measured path, and the
    numbers in this project's results are measured on that path.

The per-agent endpoints exist so something else can orchestrate the same
agents. /triage is the fast path, and it is the same code the test suite and
every experiment run through.

THE POLICY ENDPOINT
-------------------
/policy/rag-gate returns the escalation decision for a retrieval result. It
exists so an external workflow can branch on a decision COMPUTED HERE, in the
code the goldens and the adversarial gate cover, rather than re-implementing
"is the similarity below 0.67" in a workflow tool where nothing would test it.
n8n gets its visible IF-branch; it never holds a threshold.

THE FAILURE BOUNDARY
--------------------
A library may crash. A service may not: one malformed ticket must not take
down the process.

/triage catches AgentError -- the package's own base class -- and answers HTTP
200 with an escalation envelope. 200 rather than 5xx because the request WAS
handled correctly: the ticket needs a human. A 500 would tell the caller to
retry, which is wrong for a ticket that will fail the same way next time.

Anything that is NOT an AgentError returns 500 and is logged. Converting an
unknown bug into a routine escalation would hide it behind a plausible
response, and this project's recurring failure mode is precisely the wrong
answer that looks reasonable. The boundary catches KNOWN failures only.

Note the asymmetry with the per-agent endpoints: those return 503 on failure,
because a caller who asked for retrieval specifically wants to know retrieval
failed. "Needs a human" is only a meaningful answer for a whole ticket.

ARTIFACTS, MEMORY AND CONCURRENCY
---------------------------------
Artifacts preload on startup, so the first request does not pay BGE's ~60s and
a stale artifact stops the server instead of surfacing mid-request.

The service loads artifacts ONE way only, with require_gemini=False, and
attaches a Gemini client later via dataclasses.replace() if a resolution is
actually requested. That is deliberate: load_artifacts is lru_cache(maxsize=2)
keyed on require_gemini, so calling it both ways in one process would load BGE
TWICE. Sharing the same embedder and index through a replaced dataclass avoids
that entirely.

Endpoints are sync `def`, so Starlette runs them in a threadpool, and one
module-level lock serialises agent work. The embedder and index are shared
state and this is a research demo, not a throughput exercise: one ticket at a
time, on purpose.

Run it from the project root:

    uvicorn src.service.api:app --port 8000
"""

from __future__ import annotations

import dataclasses
import os
import threading
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException, Query
from pydantic import BaseModel, ConfigDict, Field

from src.agent import pipeline
from src.agent.agents import (
    ClassificationAgent,
    ResolutionAgent,
    RetrievalAgent,
)
from src.agent.artifacts import Artifacts, build_gemini_client, load_artifacts
from src.agent.config import PROJECT_ROOT, config_fingerprint, settings
from src.agent.errors import (
    AgentError,
    ClassificationError,
    LLMError,
    RetrievalError,
)
from src.agent.logging_setup import get_logger
from src.agent.orchestrator import rag_gate
from src.agent.schemas import (
    ClassificationResult,
    EscalationDecision,
    PipelineResult,
    ResolutionSuggestion,
    RetrievalResult,
    TicketIn,
)

# Module state. Populated on lifespan startup, never at import -- importing
# this module must not load a model.
_ARTIFACTS: Artifacts | None = None
_ARTIFACTS_WITH_GEMINI: Artifacts | None = None
_TIER1_MANIFEST: dict | None = None
_LOCK = threading.Lock()


# --------------------------------------------------------------------------- #
# Response models.                                                             #
# --------------------------------------------------------------------------- #
class _Model(BaseModel):
    model_config = ConfigDict(extra="forbid", protected_namespaces=())


class HealthResponse(_Model):
    """What is actually loaded, not merely whether the process is up.

    config_fingerprint is the useful field: it is the same value stamped onto
    every PipelineResult, so a deployment serving stale artifacts can be spotted
    by comparing this against the fingerprint on a stored result.
    """

    status: str
    config_fingerprint: str
    embedding_model: str
    embedding_dim: int
    index_ntotal: int
    metadata_rows: int
    cascade_confidence_threshold: float
    rag_similarity_threshold: float
    tier1_dataset_sha256: str
    tier1_fitted_rows: int
    resolution_available: bool


class AgentFailure(_Model):
    """A known agent failure, reported rather than raised."""

    escalated: bool = True
    requires_human: bool = True
    failed_agent: str | None = None
    error_kind: str
    error_message: str


class TriageResponse(_Model):
    ok: bool
    result: PipelineResult | None = None
    failure: AgentFailure | None = None


class RagGateRequest(_Model):
    """Takes /agents/retrieve's output verbatim, so the two compose."""

    retrieval: RetrievalResult


class RagGateResponse(_Model):
    escalate: bool
    decision: EscalationDecision | None = Field(
        default=None,
        description="Populated only when escalate is true.",
    )
    threshold_applied: float
    observed_value: float


# --------------------------------------------------------------------------- #
# Startup.                                                                     #
# --------------------------------------------------------------------------- #
def _gemini_key_present() -> bool:
    try:
        from dotenv import load_dotenv

        load_dotenv(PROJECT_ROOT / ".env")
    except ImportError:  # pragma: no cover - dotenv is a declared dependency
        pass
    return bool(os.getenv("GEMINI_API_KEY", "").strip())


def _read_tier1_manifest() -> dict:
    import joblib

    bundle = joblib.load(settings.models.tier1_classifier_path)
    return bundle.get("manifest") or {}


def startup() -> None:
    """Load once, verify, and report. Raises ArtifactError to stop the server.

    Deliberately not swallowed: a service that boots on stale artifacts and
    serves confident wrong answers is worse than one that refuses to boot.
    """
    global _ARTIFACTS, _TIER1_MANIFEST

    _ARTIFACTS = load_artifacts(require_gemini=False)
    _TIER1_MANIFEST = _read_tier1_manifest()

    get_logger().info(
        "service_ready",
        extra={
            "config_fingerprint": config_fingerprint(),
            "embedding_model": settings.models.embedding_model,
            "index_ntotal": int(_ARTIFACTS.index.ntotal),
            "resolution_available": _gemini_key_present(),
        },
    )


@asynccontextmanager
async def lifespan(app: FastAPI):
    startup()
    yield


app = FastAPI(
    title="IT ticket triage agent",
    version="3c",
    summary="Cascade classification, FAISS retrieval, and a calibrated "
            "human-escalation gate.",
    lifespan=lifespan,
)


# --------------------------------------------------------------------------- #
# Accessors.                                                                   #
# --------------------------------------------------------------------------- #
def _artifacts() -> Artifacts:
    if _ARTIFACTS is None:
        raise HTTPException(
            status_code=503,
            detail="Artifacts are not loaded. The service was started "
                   "without its lifespan running.",
        )
    return _ARTIFACTS


def _artifacts_with_gemini() -> Artifacts:
    """Artifacts plus a Gemini client, SHARING the already-loaded models.

    dataclasses.replace() rather than load_artifacts(require_gemini=True),
    which would load a second copy of BGE into memory.
    """
    global _ARTIFACTS_WITH_GEMINI

    if _ARTIFACTS_WITH_GEMINI is not None:
        return _ARTIFACTS_WITH_GEMINI

    if not _gemini_key_present():
        raise HTTPException(
            status_code=503,
            detail="Resolution drafting is unavailable: GEMINI_API_KEY is "
                   "not set. Classification, retrieval and the escalation "
                   "gate work without it.",
        )

    _ARTIFACTS_WITH_GEMINI = dataclasses.replace(
        _artifacts(), gemini_client=build_gemini_client(),
    )
    return _ARTIFACTS_WITH_GEMINI


def _failed_agent(exc: Exception) -> str | None:
    """Which agent a known failure came from, by exception type."""
    if isinstance(exc, ClassificationError):
        return ClassificationAgent.name
    if isinstance(exc, RetrievalError):
        return RetrievalAgent.name
    if isinstance(exc, LLMError):
        return ResolutionAgent.name
    return None


# --------------------------------------------------------------------------- #
# Endpoints.                                                                   #
# --------------------------------------------------------------------------- #
@app.get("/health", response_model=HealthResponse)
def health() -> HealthResponse:
    artifacts = _artifacts()
    manifest = _TIER1_MANIFEST or {}
    return HealthResponse(
        status="ok",
        config_fingerprint=config_fingerprint(),
        embedding_model=settings.models.embedding_model,
        embedding_dim=settings.models.embedding_dim,
        index_ntotal=int(artifacts.index.ntotal),
        metadata_rows=len(artifacts.metadata),
        cascade_confidence_threshold=settings.cascade.confidence_threshold,
        rag_similarity_threshold=settings.rag.similarity_threshold,
        tier1_dataset_sha256=str(manifest.get("dataset_sha256", "")),
        tier1_fitted_rows=int(manifest.get("fitted_rows", 0)),
        resolution_available=_gemini_key_present(),
    )


@app.post("/agents/classify", response_model=ClassificationResult)
def classify(ticket: TicketIn) -> ClassificationResult:
    try:
        with _LOCK:
            return ClassificationAgent(_artifacts()).run(ticket)
    except AgentError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc


@app.post("/agents/retrieve", response_model=RetrievalResult)
def retrieve(ticket: TicketIn) -> RetrievalResult:
    try:
        with _LOCK:
            return RetrievalAgent(_artifacts()).run(ticket)
    except AgentError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc


@app.post("/agents/resolve", response_model=ResolutionSuggestion)
def resolve(ticket: TicketIn,
            retrieval: RetrievalResult) -> ResolutionSuggestion:
    """SPENDS GEMINI QUOTA. Free tier is 15 requests/minute, 500/day.

    Never reached by /triage unless generate_resolution is explicitly true.
    """
    artifacts = _artifacts_with_gemini()
    try:
        with _LOCK:
            return ResolutionAgent(artifacts).run(ticket, retrieval)
    except AgentError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc


@app.post("/policy/rag-gate", response_model=RagGateResponse)
def policy_rag_gate(request: RagGateRequest) -> RagGateResponse:
    """The escalation decision, computed by the tested code.

    Loads nothing: the gate is a pure function of a retrieval result and the
    calibrated threshold, which is exactly why it can be trusted over HTTP.
    """
    decision = rag_gate(request.retrieval)
    return RagGateResponse(
        escalate=decision is not None,
        decision=decision,
        threshold_applied=settings.rag.similarity_threshold,
        observed_value=request.retrieval.top_similarity,
    )


@app.post("/triage", response_model=TriageResponse)
def triage(
    ticket: TicketIn,
    generate_resolution: bool = Query(
        default=False,
        description="Draft a resolution with Gemini. Defaults to false: the "
                    "free tier allows 500 calls/day and a looping workflow "
                    "would drain it, so spending quota must be asked for.",
    ),
) -> TriageResponse:
    artifacts = (_artifacts_with_gemini() if generate_resolution
                 else _artifacts())
    try:
        with _LOCK:
            result = pipeline.run(
                ticket,
                artifacts=artifacts,
                generate_resolution=generate_resolution,
            )
    except AgentError as exc:
        # A known failure. The ticket needs a human; the caller should not
        # retry, so this is a 200 carrying an escalation, not a 5xx.
        get_logger().warning(
            "agent_failure",
            extra={
                "ticket_id": ticket.ticket_id,
                "failed_agent": _failed_agent(exc),
                "error_kind": type(exc).__name__,
            },
        )
        return TriageResponse(
            ok=False,
            failure=AgentFailure(
                failed_agent=_failed_agent(exc),
                error_kind=type(exc).__name__,
                error_message=str(exc),
            ),
        )
    except Exception as exc:
        # NOT laundered into an escalation. An unknown bug must look like one.
        get_logger().exception(
            "unhandled_error",
            extra={"ticket_id": ticket.ticket_id,
                   "error_kind": type(exc).__name__},
        )
        raise HTTPException(
            status_code=500,
            detail=f"Unhandled error: {type(exc).__name__}",
        ) from exc

    return TriageResponse(ok=True, result=result)
