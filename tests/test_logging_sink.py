"""
The opt-in decision log sink (Phase 4A).

The first test is the gate criterion: by default nothing is written. Every
evaluation path, the service and the demo must inherit no new file-writing
side effect from Phase 4A.
"""

from __future__ import annotations

import json
import logging

import pytest

from src.agent import logging_setup as ls
from src.agent import schemas as S
from src.agent.errors import ConfigError


@pytest.fixture
def clean_logger():
    """Snapshot and restore the package logger around each test."""
    logger = logging.getLogger(ls.LOGGER_NAME)
    saved_handlers, saved_level = list(logger.handlers), logger.level
    yield logger
    ls.detach_decision_log()
    logger.handlers[:] = saved_handlers
    logger.setLevel(saved_level)


def _result(ticket_id="t9", retrieval_ok=True):
    steps = [S.StepTrace(agent="classification", status=S.StepStatus.OK,
                         latency_ms=1.0)]
    steps.append(S.StepTrace(
        agent="retrieval",
        status=S.StepStatus.OK if retrieval_ok else S.StepStatus.SKIPPED,
        latency_ms=1.0 if retrieval_ok else 0.0))
    return S.PipelineResult(
        ticket=S.TicketIn(title="Disk full", ticket_id=ticket_id),
        classification=S.ClassificationResult(
            category="Storage", confidence=0.88,
            tier=S.Tier.TIER1_TFIDF, tier1_pred="Storage", tier1_conf=0.88,
        ),
        retrieval=S.RetrievalResult(retrieved=[
            S.RetrievedTicket(id=3, similarity=0.77),
        ]),
        decision=S.EscalationDecision(
            escalated=False, reason=S.EscalationReason.NONE,
            threshold_applied=0.67, observed_value=0.77,
        ),
        status=S.ResolutionStatus.AUTO_RESOLVED,
        config_fingerprint="abc123def456",
        steps=steps,
    )


def _decision_handlers(logger):
    return [h for h in logger.handlers
            if isinstance(h, logging.FileHandler)]


def test_default_configuration_writes_nothing(clean_logger, tmp_path,
                                              monkeypatch):
    monkeypatch.chdir(tmp_path)
    ls.configure_logging(level=logging.INFO)
    ls.log_decision(_result())
    assert _decision_handlers(clean_logger) == []
    assert list(tmp_path.iterdir()) == []


def test_sink_writes_one_json_line_per_decision(clean_logger, tmp_path):
    path = tmp_path / "logs" / "decisions.jsonl"
    ls.configure_logging(level=logging.INFO, decision_log_path=path)
    ls.log_decision(_result("a"))
    ls.log_decision(_result("b"))

    lines = path.read_text(encoding="utf-8").splitlines()
    assert [json.loads(line)["ticket_id"] for line in lines] == ["a", "b"]
    assert all(json.loads(line)["schema_version"]
               == ls.DECISION_SCHEMA_VERSION for line in lines)


def test_sink_filters_non_decision_events(clean_logger, tmp_path):
    path = tmp_path / "decisions.jsonl"
    ls.configure_logging(level=logging.INFO, decision_log_path=path)
    ls.get_logger().warning("agent_failure", extra={"ticket_id": "x"})
    ls.get_logger().info("service_started")
    ls.log_decision(_result())
    events = [json.loads(line)["event"]
              for line in path.read_text(encoding="utf-8").splitlines()]
    assert events == [ls.DECISION_EVENT]


def test_repeated_configuration_does_not_duplicate(clean_logger, tmp_path):
    """Streamlit reruns configure_logging on every interaction."""
    path = tmp_path / "decisions.jsonl"
    for _ in range(3):
        ls.configure_logging(level=logging.INFO, decision_log_path=path)
    ls.configure_logging(level=logging.INFO)  # no path: must not detach
    ls.log_decision(_result())
    assert len(path.read_text(encoding="utf-8").splitlines()) == 1
    assert len(_decision_handlers(clean_logger)) == 1


def test_starved_sink_is_refused_before_anything_changes(clean_logger,
                                                         tmp_path):
    """A sink below the logger level would record an empty 'quiet period'."""
    path = tmp_path / "decisions.jsonl"
    level_before = clean_logger.level
    with pytest.raises(ConfigError, match="INFO"):
        ls.configure_logging(level=logging.WARNING, decision_log_path=path)
    assert _decision_handlers(clean_logger) == []
    assert clean_logger.level == level_before
    assert not path.exists()

    ls.configure_logging(level=logging.INFO, decision_log_path=path)
    with pytest.raises(ConfigError):
        ls.configure_logging(level=logging.WARNING)


def test_logged_line_round_trips_as_a_drift_record(clean_logger, tmp_path):
    """The writer and the reader agree on the record shape."""
    from src.agent.drift import DecisionRecord

    path = tmp_path / "decisions.jsonl"
    ls.configure_logging(level=logging.INFO, decision_log_path=path)
    ls.log_decision(_result(retrieval_ok=True))
    ls.log_decision(_result(retrieval_ok=False))

    records = [DecisionRecord.from_json_line(line)
               for line in path.read_text(encoding="utf-8").splitlines()]
    assert records[0].top_similarity == 0.77
    assert records[0].tier == 1 and records[0].category == "Storage"
    assert [r.retrieval_ran for r in records] == [True, False]
