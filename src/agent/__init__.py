"""
The agent package: one pipeline, one loader, one source of truth.

This package consolidates what were previously four independent copies of the
classify -> retrieve -> escalate pipeline (streamlit_app.run_pipeline,
process_ticket_batch.process, test_adversarial_escalation.
run_ticket_through_pipeline, and suggest_resolution.suggest_resolution_for_
ticket) together with three disagreeing model loaders.

Import order note: config imports errors, so errors must not import config.
"""

from __future__ import annotations

__all__ = ["config", "errors", "schemas"]
