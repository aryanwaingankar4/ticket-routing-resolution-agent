"""
The three agents, and the contract they share.

WHY THIS EXISTS
---------------
Until Phase 3B this project was, in the README's own words, "a sequential
pipeline with confidence-based decision points -- NOT yet a true multi-agent
system". The three stages were free functions with three different
signatures:

    classify(text, artifacts)
    retrieve(text, artifacts, top_k)
    generate(ticket, retrieval, artifacts)

Each took the whole Artifacts blob and reached into whatever it happened to
need. That works in one process and nowhere else: nothing declares what a
stage actually depends on, so nothing can be moved, replaced, or served
independently without reading its body.

WHAT AN AGENT IS HERE
---------------------
An agent is a named unit with DECLARED DEPENDENCIES and one entry point:

    name        what it is called in traces and logs
    requires    the attributes of Artifacts it cannot work without
    run(...)    its single operation, typed per agent

`requires` is the important half. It is the agent boundary written down --
the reason ClassificationAgent and RetrievalAgent are separable is that they
need overlapping but different artifacts, and now they say so. Phase 3C
splits these across HTTP endpoints; that is a mechanical job precisely
because each agent already declares what it needs to be given.

Dependencies are validated at CONSTRUCTION, not on first use. An agent that
cannot work should say so before the orchestrator starts routing a ticket
through it, not halfway down the sequence.

WHAT AN AGENT IS NOT
--------------------
An agent decides nothing about routing. No agent reads a threshold, and no
agent knows whether another agent will run. Every gate lives in
orchestrator.py. That separation is the whole point of the restructure: the
escalation policy this project's thesis rests on must be readable in one
place, not distributed across the things being orchestrated.

The stage implementations themselves are deliberately UNCHANGED. classifier.py,
retriever.py and resolver.py hold parity-critical logic -- the L2-normalise
before search, the calibrated prompt, the retry ladder -- and agents wrap them
rather than absorbing them.
"""

from __future__ import annotations

from dataclasses import fields as dataclass_fields
from typing import Protocol, runtime_checkable

from src.agent.artifacts import Artifacts
from src.agent.classifier import classify
from src.agent.errors import ArtifactError
from src.agent.resolver import generate
from src.agent.retriever import retrieve
from src.agent.schemas import (
    ClassificationResult,
    ResolutionSuggestion,
    RetrievalResult,
    TicketIn,
)


@runtime_checkable
class TicketAgent(Protocol):
    """The shared contract.

    `run` is deliberately not typed here: the three agents take and return
    different things, and flattening that into one envelope would buy
    uniformity at the cost of the typed boundaries schemas.py exists to
    provide. What is uniform -- and what the orchestrator relies on -- is that
    every agent has a name, declares its dependencies, and exposes exactly one
    operation called `run`.
    """

    name: str
    requires: tuple[str, ...]

    def run(self, *args, **kwargs): ...


class _BaseAgent:
    """Dependency declaration and validation, shared by every agent."""

    name: str = ""
    requires: tuple[str, ...] = ()

    def __init__(self, artifacts: Artifacts) -> None:
        known = {f.name for f in dataclass_fields(Artifacts)}
        unknown = [dep for dep in self.requires if dep not in known]
        if unknown:
            raise ArtifactError(
                f"The {self.name!r} agent declares dependencies that are not "
                f"fields of Artifacts: {sorted(unknown)}\n"
                f"  Known fields: {sorted(known)}\n"
                "  This is a typo in the agent's `requires`, not a missing "
                "artifact."
            )

        missing = [dep for dep in self.requires
                   if getattr(artifacts, dep) is None]
        if missing:
            raise ArtifactError(
                f"The {self.name!r} agent cannot run: the loaded artifacts "
                f"are missing {sorted(missing)}.\n"
                "  Load artifacts through artifacts.load_artifacts(); if "
                "'gemini_client' is\n"
                "  missing, it was loaded with require_gemini=False, which "
                "is correct for\n"
                "  evaluation paths but cannot draft a resolution."
            )

        self._artifacts = artifacts

    def __repr__(self) -> str:  # pragma: no cover - diagnostic only
        return f"<{type(self).__name__} name={self.name!r}>"


class ClassificationAgent(_BaseAgent):
    """Cascade classification: Tier-1 TF-IDF, escalating to Tier-2 BGE.

    Needs the embedder because Tier-2 encodes the ticket before classifying
    it -- the cascade's expensive half lives inside this one agent, which is
    why the cascade threshold is not the orchestrator's business.
    """

    name = "classification"
    requires = (
        "tier1_vectorizer",
        "tier1_classifier",
        "tier2_classifier",
        "embedder",
    )

    def run(self, ticket: TicketIn) -> ClassificationResult:
        return classify(ticket.combined_text, self._artifacts)


class RetrievalAgent(_BaseAgent):
    """FAISS retrieval over past tickets."""

    name = "retrieval"
    requires = ("embedder", "index", "metadata", "faiss")

    def run(self, ticket: TicketIn,
            top_k: int | None = None) -> RetrievalResult:
        return retrieve(ticket.combined_text, self._artifacts, top_k=top_k)


class ResolutionAgent(_BaseAgent):
    """Grounded resolution drafting via Gemini.

    Constructed only when a resolution is actually wanted. The normal
    evaluation path loads artifacts with require_gemini=False and never
    reaches this agent, which is what lets routing benchmarks run with no API
    key and no quota.
    """

    name = "resolution"
    requires = ("gemini_client",)

    def run(self, ticket: TicketIn,
            retrieval: RetrievalResult) -> ResolutionSuggestion:
        return generate(ticket, retrieval, self._artifacts)
