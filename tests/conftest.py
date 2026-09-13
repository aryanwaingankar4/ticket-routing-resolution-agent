"""
Shared pytest fixtures.

The two fixed benchmark sets are read-only reference points for this whole
project and are loaded here verbatim -- never regenerated, never mutated.

Artifact loading is session-scoped: BGE plus a fresh Tier-1 fit over 4,000
tickets costs roughly a minute, and every slow test shares one instance.
"""

from __future__ import annotations

import json
import os
import sys

import pytest

_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(_THIS_DIR, os.pardir))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

DATA_DIR = os.path.join(PROJECT_ROOT, "data")
GOLDENS_DIR = os.path.join(_THIS_DIR, "goldens")


def _read_json(path):
    with open(path, "r", encoding="utf-8") as fh:
        return json.load(fh)


@pytest.fixture(scope="session")
def adversarial_tickets():
    """The 9-ticket adversarial escalation set (read-only)."""
    return _read_json(
        os.path.join(DATA_DIR, "adversarial_escalation_tickets.json")
    )


@pytest.fixture(scope="session")
def benchmark_tickets():
    """The 45-ticket expanded generalization benchmark (read-only)."""
    return _read_json(os.path.join(DATA_DIR, "novel_tickets_expanded.json"))


@pytest.fixture(scope="session")
def adversarial_golden():
    return _read_json(
        os.path.join(GOLDENS_DIR, "adversarial_baseline.json")
    )


@pytest.fixture(scope="session")
def benchmark_golden():
    return _read_json(os.path.join(GOLDENS_DIR, "benchmark_baseline.json"))


@pytest.fixture(scope="session")
def artifacts():
    """Production artifacts, loaded once per session, without Gemini.

    require_gemini=False on purpose: every routing assertion in this suite is
    decided before any LLM call, so the suite needs no API key and no quota.
    """
    import logging

    from src.agent.artifacts import load_artifacts
    from src.agent.logging_setup import configure_logging

    configure_logging(level=logging.WARNING)
    return load_artifacts(require_gemini=False)
