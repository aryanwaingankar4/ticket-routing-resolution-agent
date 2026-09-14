"""
Structured JSON logging for pipeline decisions.

WHY THIS EXISTS
---------------
Two reasons, one immediate and one for later.

Immediately: the pipeline's escalation decisions were previously only visible
as print() output interleaved with progress chatter, which makes them
impossible to audit after the fact. A decision this project's entire thesis
rests on deserves a machine-readable record.

For later: these records are the raw input for drift detection. Emitting them
now means that phase has real history to work against instead of starting
from zero.

Deliberately stdlib-only. `structlog` is a drop-in upgrade if richer context
binding is ever wanted, but it is not worth a dependency for one formatter,
and this project's dependency list is already under-specified.
"""

from __future__ import annotations

import json
import logging
import sys
from datetime import datetime, timezone
from typing import Any

LOGGER_NAME = "ticket_agent"

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


def configure_logging(level: int = logging.INFO,
                      stream=None) -> logging.Logger:
    """Configure and return the package logger.

    Idempotent: calling it repeatedly (as Streamlit does on every rerun) will
    not stack duplicate handlers.
    """
    logger = logging.getLogger(LOGGER_NAME)
    logger.setLevel(level)
    logger.propagate = False

    if not logger.handlers:
        handler = logging.StreamHandler(stream or sys.stderr)
        handler.setFormatter(JsonFormatter())
        logger.addHandler(handler)

    return logger


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
        "pipeline_decision",
        extra={
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
