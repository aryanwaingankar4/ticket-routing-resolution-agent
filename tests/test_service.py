"""
The HTTP surface (Phase 3C).

Two things are being established here, and they are gated differently.

PARITY. /triage must return exactly what the library returns. A service that
routes differently from the thing it wraps is a fifth pipeline, and ending
that is what src/agent/ exists for. There is no golden for HTTP, so the test
compares the two directly, ticket by ticket, over both fixed benchmark sets.

THE FAILURE BOUNDARY. A known agent failure must become a 200 with an
escalation -- the ticket needs a human, and telling the caller to retry would
be wrong. An UNKNOWN error must still be a 500. Laundering an unexpected bug
into a routine escalation would hide it behind a plausible response, which is
this project's recurring failure shape exactly.

No test here spends Gemini quota. The resolution endpoint is exercised only on
its refusal path, with the key deliberately masked.
"""

from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

from src.agent import pipeline
from src.agent.config import config_fingerprint, settings
from src.agent.errors import RetrievalError
from src.agent.schemas import TicketIn
from src.service import api as service_api

# Captured at import, before any client starts the lifespan. Test modules are
# imported during collection, so this is the module's state on a cold import.
_ARTIFACTS_AT_IMPORT = service_api._ARTIFACTS

pytestmark = pytest.mark.slow


@pytest.fixture(scope="module")
def client(artifacts):
    """A live app. Depends on the session artifacts fixture so startup reuses
    the already-cached load rather than paying for BGE twice."""
    with TestClient(service_api.app) as test_client:
        yield test_client


def _decision_view(payload: dict) -> dict:
    """The fields that decide a ticket's fate, from a /triage response."""
    return {
        "category": payload["classification"]["category"],
        "tier": payload["classification"]["tier"],
        "confidence": payload["classification"]["confidence"],
        "tier1_conf": payload["classification"]["tier1_conf"],
        "top_similarity": payload["retrieval"]["retrieved"][0]["similarity"]
        if payload["retrieval"]["retrieved"] else 0.0,
        "escalated": payload["decision"]["escalated"],
        "reason": payload["decision"]["reason"],
        "status": payload["status"],
    }


def _library_view(result) -> dict:
    return {
        "category": result.classification.category,
        "tier": int(result.classification.tier),
        "confidence": result.classification.confidence,
        "tier1_conf": result.classification.tier1_conf,
        "top_similarity": result.top_similarity,
        "escalated": result.decision.escalated,
        "reason": result.decision.reason.value,
        "status": result.status.value,
    }


# --------------------------------------------------------------------------- #
# Import hygiene.                                                              #
# --------------------------------------------------------------------------- #
def test_importing_the_service_loads_no_models():
    """Artifacts load on lifespan startup, never at import.

    streamlit_app.py cannot be imported headlessly because it does work at
    import time; the service must not repeat that -- it is why the adversarial
    test once grew its own copy of the cascade.
    """
    assert _ARTIFACTS_AT_IMPORT is None


# --------------------------------------------------------------------------- #
# Health.                                                                      #
# --------------------------------------------------------------------------- #
def test_health_reports_what_is_actually_loaded(client):
    body = client.get("/health").json()

    assert body["status"] == "ok"
    assert body["config_fingerprint"] == config_fingerprint()
    assert body["embedding_model"] == settings.models.embedding_model
    assert body["index_ntotal"] == 4000
    assert body["metadata_rows"] == 4000
    assert body["rag_similarity_threshold"] == (
        settings.rag.similarity_threshold
    )


def test_health_exposes_the_tier1_provenance(client):
    """A deployment serving a stale Tier-1 should be visible over HTTP."""
    body = client.get("/health").json()
    assert len(body["tier1_dataset_sha256"]) == 64
    assert body["tier1_fitted_rows"] == 4000


# --------------------------------------------------------------------------- #
# Parity: the service must not become a second pipeline.                       #
# --------------------------------------------------------------------------- #
def test_triage_matches_the_library_on_the_adversarial_set(
    client, adversarial_tickets, artifacts,
):
    for ticket in adversarial_tickets:
        response = client.post(
            "/triage",
            json={"title": ticket["text"], "ticket_id": ticket["id"]},
        )
        assert response.status_code == 200
        body = response.json()
        assert body["ok"] is True

        expected = pipeline.run(
            TicketIn(title=ticket["text"], ticket_id=ticket["id"]),
            artifacts=artifacts, generate_resolution=False, emit_log=False,
        )
        assert _decision_view(body["result"]) == _library_view(expected), (
            f"{ticket['id']} routed differently over HTTP"
        )


def test_triage_matches_the_library_on_the_benchmark(
    client, benchmark_tickets, artifacts,
):
    for ticket in benchmark_tickets[:15]:
        body = client.post("/triage", json={"title": ticket["text"]}).json()
        expected = pipeline.run(
            TicketIn(title=ticket["text"]), artifacts=artifacts,
            generate_resolution=False, emit_log=False,
        )
        assert _decision_view(body["result"]) == _library_view(expected)


# --------------------------------------------------------------------------- #
# The failure boundary.                                                        #
# --------------------------------------------------------------------------- #
def test_known_agent_failure_returns_an_escalation_not_a_crash(
    client, monkeypatch,
):
    def boom(*args, **kwargs):
        raise RetrievalError("FAISS retrieval failed (simulated).")

    monkeypatch.setattr(service_api.pipeline, "run", boom)

    response = client.post("/triage", json={"title": "anything"})

    assert response.status_code == 200, "a failed ticket is not a server error"
    body = response.json()
    assert body["ok"] is False
    assert body["result"] is None
    assert body["failure"]["failed_agent"] == "retrieval"
    assert body["failure"]["error_kind"] == "RetrievalError"
    assert body["failure"]["escalated"] is True
    assert body["failure"]["requires_human"] is True


def test_unknown_errors_are_not_laundered_into_escalations(
    client, monkeypatch,
):
    """An unexpected bug must look like a bug.

    If this ever returns 200, a genuine defect would be indistinguishable from
    a ticket that legitimately needs a human -- the exact shape of every bug in
    this project's recurring class.
    """
    def boom(*args, **kwargs):
        raise ValueError("a real defect, not an agent failure")

    monkeypatch.setattr(service_api.pipeline, "run", boom)

    response = client.post("/triage", json={"title": "anything"})
    assert response.status_code == 500


# --------------------------------------------------------------------------- #
# Policy over HTTP.                                                            #
# --------------------------------------------------------------------------- #
def test_policy_gate_uses_the_calibrated_threshold(client):
    """n8n branches on this, so it must be the same 0.67 as everything else."""
    threshold = settings.rag.similarity_threshold

    below = client.post("/policy/rag-gate", json={
        "retrieval": {"retrieved": [{"id": 1, "similarity": threshold - 0.01}]},
    }).json()
    assert below["escalate"] is True
    assert below["threshold_applied"] == threshold
    assert below["decision"]["reason"] == "low_retrieval_similarity"

    at = client.post("/policy/rag-gate", json={
        "retrieval": {"retrieved": [{"id": 1, "similarity": threshold}]},
    }).json()
    assert at["escalate"] is False
    assert at["decision"] is None


def test_policy_gate_escalates_on_empty_retrieval(client):
    body = client.post(
        "/policy/rag-gate", json={"retrieval": {"retrieved": []}},
    ).json()
    assert body["escalate"] is True
    assert body["decision"]["reason"] == "no_retrieval_results"


# --------------------------------------------------------------------------- #
# Individual agents, and the quota guard.                                      #
# --------------------------------------------------------------------------- #
def test_agent_endpoints_answer_individually(client):
    classification = client.post(
        "/agents/classify", json={"title": "cannot connect to the vpn"},
    ).json()
    assert classification["category"]
    assert classification["tier"] in (1, 2)

    retrieval = client.post(
        "/agents/retrieve", json={"title": "cannot connect to the vpn"},
    ).json()
    assert len(retrieval["retrieved"]) == settings.rag.top_k


def test_triage_spends_no_quota_by_default(client):
    """generate_resolution defaults to false; the resolution agent must be
    recorded as skipped rather than merely absent."""
    body = client.post("/triage", json={"title": "vpn keeps dropping"}).json()
    steps = {s["agent"]: s for s in body["result"]["steps"]}
    assert steps["resolution"]["status"] == "skipped"


def test_resolution_is_refused_when_no_key_is_configured(client, monkeypatch):
    """Exercises the guard, never the API. Masks the key so a developer with a
    real one in .env cannot accidentally spend quota running the suite."""
    monkeypatch.setattr(service_api, "_gemini_key_present", lambda: False)
    monkeypatch.setattr(service_api, "_ARTIFACTS_WITH_GEMINI", None)

    response = client.post("/agents/resolve", json={
        "ticket": {"title": "printer offline"},
        "retrieval": {"retrieved": [{"id": 1, "similarity": 0.9}]},
    })
    assert response.status_code == 503
    assert "GEMINI_API_KEY" in response.json()["detail"]
