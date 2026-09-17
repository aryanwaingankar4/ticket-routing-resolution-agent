"""
Structured JSON logging for pipeline decisions.

WHY THIS EXISTS
---------------
Two reasons, one immediate and one for later.

Immediately: the pipeline's escalation decisions were previously only visible
as print() output interleaved with progress chatter, which makes them
impossible to audit after the fact. A decision this project's entire thesis
rests on deserves a machine-readable record.

For later: these records are the raw input for drift detection.

PERSISTENCE IS OPT-IN (Phase 4A)
--------------------------------
Until Phase 4A this module attached only a stderr handler, so every record
it emitted evaporated -- while three modules described them as drift
detection's history. `configure_logging(decision_log_path=...)` now adds a
JSONL sink that receives `pipeline_decision` records only, one per line.

It is OFF by default, following the `filing_gate_enabled` precedent: no
script, test, service or demo inherits a new file-writing side effect, and
nothing in the project enables it yet. Enabling it anywhere is a deliberate
decision with its own gate.

Deliberately stdlib-only. `structlog` is a drop-in upgrade if richer context
binding is ever wanted, but it is not worth a dependency for one formatter,
and this project's dependency list is already under-specified.
"""

from __future__ import annotations

import json
import logging
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from src.agent.errors import ConfigError

LOGGER_NAME = "ticket_agent"

DECISION_EVENT = "pipeline_decision"

# Bumped whenever the shape of a decision record changes, so a reader can
# reject a record it does not understand instead of silently mis-parsing it.
DECISION_SCHEMA_VERSION = 1

# Marks the handlers this module owns, so repeated configure_logging() calls
# can recognise them. Type checks are not enough: logging.FileHandler is a
# subclass of logging.StreamHandler.
_ROLE_ATTR = "_ticket_agent_role"
_ROLE_CONSOLE = "console"
_ROLE_DECISIONS = "decision_log"

# Attributes LogRecord always carries; anything else was attached by us via
# `extra=` and belongs in the JSON payload.
_STD_ATTRS = frozenset(
    logging.LogRecord("", 0, "", 0, "", (), None).__dict__.keys()
) | {"message", "asctime", "taskName"}


class JsonFormatter(logging.Formatter):
    """Render each record as a single JSON object, one per line."""

    def format(self, record: logging.LogRecord) -> str:
        payload: dict[str, Any] = {
            "ts": datetime.fromtimestamp(
                record.created, tz=timezone.utc
            ).isoformat(),
            "level": record.levelname,
            "logger": record.name,
            "event": record.getMessage(),
        }

        for key, value in record.__dict__.items():
            if key not in _STD_ATTRS and not key.startswith("_"):
                payload[key] = value

        if record.exc_info:
            payload["exc"] = self.formatException(record.exc_info)

        return json.dumps(payload, default=str, sort_keys=True)


class _DecisionsOnly(logging.Filter):
    """Pass pipeline_decision records; drop progress and failure chatter."""

    def filter(self, record: logging.LogRecord) -> bool:
        return record.getMessage() == DECISION_EVENT


def _handlers_with_role(logger: logging.Logger, role: str):
    return [h for h in logger.handlers if getattr(h, _ROLE_ATTR, None) == role]


def configure_logging(level: int = logging.INFO,
                      stream=None,
                      decision_log_path: str | Path | None = None,
                      ) -> logging.Logger:
    """Configure and return the package logger.

    Idempotent: calling it repeatedly (as Streamlit does on every rerun) will
    not stack duplicate handlers.

    `decision_log_path` attaches the opt-in JSONL decision sink. Default None
    attaches nothing and writes nothing. A later call without a path does not
    detach a sink that is already attached.
    """
    logger = logging.getLogger(LOGGER_NAME)

    # Checked before anything is mutated, so a refused call leaves the logger
    # exactly as it was. A sink that the logger level silently starves would
    # produce an empty history that reads as a quiet period -- the recurring
    # bug class.
    sink_wanted = (decision_log_path is not None
                   or bool(_handlers_with_role(logger, _ROLE_DECISIONS)))
    if sink_wanted and level > logging.INFO:
        raise ConfigError(
            "The decision log sink needs the logger at INFO or below, but "
            f"level {logging.getLevelName(level)} was requested.\n"
            "  pipeline_decision records are INFO, so the sink would stay "
            "empty\n  and an empty history would read as no traffic.\n"
            "  Configure logging at INFO, or detach the sink first with "
            "detach_decision_log()."
        )

    logger.setLevel(level)
    logger.propagate = False

    if not _handlers_with_role(logger, _ROLE_CONSOLE):
        handler = logging.StreamHandler(stream or sys.stderr)
        handler.setFormatter(JsonFormatter())
        setattr(handler, _ROLE_ATTR, _ROLE_CONSOLE)
        logger.addHandler(handler)

    if decision_log_path is not None:
        path = Path(decision_log_path).resolve()
        attached = {
            Path(h.baseFilename).resolve()
            for h in _handlers_with_role(logger, _ROLE_DECISIONS)
        }
        if path not in attached:
            path.parent.mkdir(parents=True, exist_ok=True)
            sink = logging.FileHandler(path, mode="a", encoding="utf-8")
            sink.setFormatter(JsonFormatter())
            sink.addFilter(_DecisionsOnly())
            setattr(sink, _ROLE_ATTR, _ROLE_DECISIONS)
            logger.addHandler(sink)

    return logger


def detach_decision_log() -> None:
    """Remove and close any attached decision sink (tests, shutdown)."""
    logger = logging.getLogger(LOGGER_NAME)
    for handler in _handlers_with_role(logger, _ROLE_DECISIONS):
        logger.removeHandler(handler)
        handler.close()


def ensure_utf8_console() -> None:
    """Make stdout/stderr UTF-8 safe on Windows.

    Several scripts print check marks and em dashes in their reports. On a
    default Windows console (cp1252) those raise UnicodeEncodeError, which
    crashes the run with a raw traceback -- directly against this project's
    convention that scripts fail with clear, actionable messages. It bit the
    adversarial regression gate, which crashed mid-report despite all nine
    tickets actually passing.

    Safe to call repeatedly, and a no-op where stdout is already UTF-8.
    """
    for stream_name in ("stdout", "stderr"):
        stream = getattr(sys, stream_name, None)
        reconfigure = getattr(stream, "reconfigure", None)
        if reconfigure is None:
            continue
        try:
            reconfigure(encoding="utf-8", errors="replace")
        except (ValueError, OSError):  # pragma: no cover - already closed
            pass


def get_logger() -> logging.Logger:
    """Return the package logger, configuring it on first use."""
    logger = logging.getLogger(LOGGER_NAME)
    if not logger.handlers:
        return configure_logging()
    return logger


def log_decision(result, **extra: Any) -> None:
    """Emit one structured record for a completed PipelineResult.

    Field names match the schema exactly so logs and stored results can be
    joined without a translation layer -- the absence of which is precisely
    how `tier1_conf`/`tier1_confidence` and `top_similarity`/`rag_similarity`
    drifted apart in the first place.
    """
    get_logger().info(
        DECISION_EVENT,
        extra={
            "schema_version": DECISION_SCHEMA_VERSION,
            "ticket_id": result.ticket.ticket_id,
            "category": result.classification.category,
            "confidence": result.classification.confidence,
            "tier": int(result.classification.tier),
            "tier1_conf": result.classification.tier1_conf,
            "top_similarity": result.top_similarity,
            "escalated": result.decision.escalated,
            "escalation_reason": result.decision.reason.value,
            "threshold_applied": result.decision.threshold_applied,
            "status": result.status.value,
            "config_fingerprint": result.config_fingerprint,
            "latency_ms": result.latency_ms,
            # Per-agent trace, flattened into two compact maps. A resolution
            # agent recorded as "skipped" is the positive evidence that the
            # RAG gate held and no LLM call was made.
            "agent_status": {s.agent: s.status.value for s in result.steps},
            "agent_latency_ms": {
                s.agent: round(s.latency_ms, 3) for s in result.steps
            },
            **extra,
        },
    )
