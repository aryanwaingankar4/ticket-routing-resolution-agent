"""
Typed exception hierarchy for the agent package.

WHY THIS EXISTS
---------------
The previous failure convention was `_fail()` -> `sys.exit(1)`
(suggest_resolution.py:137, and near-identical copies elsewhere). That gives
good CLI messages but makes the pipeline untestable and unimportable: a
library cannot kill the host process, and pytest cannot assert on a SystemExit
that carries only a formatted string.

These exceptions preserve the project's convention of clear, actionable
messages -- the message text lives on the exception -- while letting the
library RAISE and the entry point DECIDE how to render. Streamlit shows an
error panel, CLI scripts print and exit, tests assert.

Every message should stay actionable: say what failed, and what the user
should run or check to fix it.
"""

from __future__ import annotations


class AgentError(Exception):
    """Base for every error raised by the agent package."""


class ConfigError(AgentError):
    """Configuration is missing or invalid (e.g. absent GEMINI_API_KEY)."""


class ArtifactError(AgentError):
    """A model, index, or metadata artifact is missing, unreadable, or stale.

    Raised in particular when the FAISS index and its metadata disagree on
    length, or when the embedder's dimensionality does not match the index --
    the exact silent-mismatch class this project has hit repeatedly.
    """


class RetrievalError(AgentError):
    """FAISS retrieval failed."""


class ClassificationError(AgentError):
    """Cascade classification failed."""


class LLMError(AgentError):
    """Base for Gemini call failures."""


class RateLimitError(LLMError):
    """Gemini rate limit hit (free tier: 15 req/min, 500/day)."""


class AuthError(LLMError):
    """Gemini rejected the API key."""


class NetworkError(LLMError):
    """Network-level failure reaching Gemini."""


def classify_llm_exception(exc: Exception) -> LLMError:
    """Map a raw SDK exception onto this hierarchy.

    Consolidates the Gemini error ladder that previously existed in three
    near-identical copies (suggest_resolution.py:440-485,
    streamlit_app.py:629-654, process_ticket_batch.py:185-217).
    """
    text = f"{type(exc).__name__}: {exc}".lower()

    rate_markers = ("429", "rate limit", "quota", "resource_exhausted")
    if any(s in text for s in rate_markers):
        return RateLimitError(
            "Gemini rate limit or quota exceeded.\n"
            "  Free tier allows 15 requests/minute and 500/day.\n"
            "  Wait for the window to reset, or raise the per-call delay "
            "(settings.gemini.call_delay_sec).\n"
            f"  Original error: {exc}"
        )
    auth_markers = ("401", "403", "api key", "unauthenticated",
                    "permission")
    if any(s in text for s in auth_markers):
        return AuthError(
            "Gemini rejected the API key.\n"
            "  Check GEMINI_API_KEY in your .env file.\n"
            "  Get a free key from https://aistudio.google.com/apikey\n"
            f"  Original error: {exc}"
        )
    net_markers = ("timeout", "connection", "network", "dns",
                   "unreachable")
    if any(s in text for s in net_markers):
        return NetworkError(
            "Could not reach the Gemini API.\n"
            "  Check your internet connection and try again.\n"
            f"  Original error: {exc}"
        )
    return LLMError(f"Gemini call failed.\n  Original error: {exc}")
