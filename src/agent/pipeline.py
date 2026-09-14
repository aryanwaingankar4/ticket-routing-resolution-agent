"""
The public entry point: classify -> retrieve -> decide -> (maybe) resolve.

This is the single implementation that replaced four copies:
    streamlit_app.run_pipeline()                     (line 567)
    process_ticket_batch.process()                   (line 292)
    test_adversarial.run_ticket_through_pipeline()   (line 620)
    suggest_resolution.suggest_resolution_for_ticket()  (line 514)

AS OF PHASE 3B THIS IS A FACADE
-------------------------------
The orchestration itself now lives in orchestrator.py, coordinating the three
agents declared in agents.py. This module stays because it is the name every
consumer already imports -- streamlit_app, test_adversarial_escalation,
process_ticket_batch, build_groundedness_set, run_ablation_study and the test
suite all call pipeline.run(), and CLAUDE.md tells future consumers to do the
same. Keeping the façade meant the agent restructure touched no consumer at
all.

Add new consumers here, the same way. Never re-implement the orchestration:
four divergent copies is what this package exists to have ended.

THE TWO GATES (unchanged, and still the point)
----------------------------------------------
Gate 1 (cascade, threshold 0.50) selects WHICH MODEL answers. It never
escalates to a human; it escalates from the cheap model to the strong one, and
it lives inside the classification agent.

Gate 2 (RAG similarity, threshold 0.67) decides whether a human is needed, and
belongs to the orchestrator. Below it, the Gemini call is skipped ENTIRELY --
not made and discarded. That is the whole point: a forced-choice classifier can
never refuse to emit a category, but the retrieval-confidence gate can refuse
to emit a fabricated resolution.

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

from src.agent.artifacts import Artifacts
from src.agent.orchestrator import run as _orchestrate
from src.agent.schemas import PipelineResult, TicketIn


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
    return _orchestrate(
        ticket,
        artifacts=artifacts,
        generate_resolution=generate_resolution,
        emit_log=emit_log,
    )
