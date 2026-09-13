"""
Grounded resolution generation via Gemini.

Prompt construction is ported from suggest_resolution.build_llm_prompt()
(line 366) unchanged -- the prompt is a calibrated artifact in its own right,
since the whole point of the RAG layer is that the LLM answers ONLY from
retrieved past resolutions rather than from general knowledge.

The retry/backoff behaviour is taken from
process_ticket_batch.call_gemini_with_retry() (line 220), which was the most
robust of the three error ladders that previously existed. Error
classification now goes through errors.classify_llm_exception(), so the
ladder exists once instead of three times.

Gemini never decides a ticket's category. Classification is entirely the
trained models' job; this module only drafts resolution text.
"""

from __future__ import annotations

import time

from src.agent.artifacts import Artifacts
from src.agent.config import settings
from src.agent.errors import (
    LLMError,
    RateLimitError,
    classify_llm_exception,
)
from src.agent.schemas import (
    ResolutionSuggestion,
    RetrievalResult,
    TicketIn,
)


def build_prompt(ticket: TicketIn, retrieval: RetrievalResult) -> str:
    """Build the grounded-resolution prompt."""
    header = (
        "You are an IT-support assistant that drafts suggested resolutions "
        "for new support tickets.\n\n"
        "IMPORTANT INSTRUCTIONS:\n"
        "- Base your suggestion ONLY on the RETRIEVED PAST TICKETS below. "
        "These are real past fixes from our ticket history.\n"
        "- Do NOT invent a generic answer from outside knowledge. Ground "
        "every step in what the retrieved resolutions actually did.\n"
        "- If the retrieved examples do not fully match the new ticket, say "
        "so explicitly and recommend that a human agent double-check.\n"
        "- Keep the suggestion concise: a few sentences of clear, actionable "
        "steps.\n"
    )

    new_block = (
        "\n================ NEW TICKET (needs a resolution) ================\n"
        f"Title:       {ticket.title}\n"
        f"Description: {ticket.description}\n"
    )

    parts = ["\n================ RETRIEVED PAST TICKETS ================"]
    for rank, r in enumerate(retrieval.retrieved, start=1):
        parts.append(
            f"\n--- Past ticket #{rank} "
            f"(similarity {r.similarity:.3f}, category: {r.category}) ---\n"
            f"Title:       {r.title}\n"
            f"Description: {r.description}\n"
            f"Resolution:  {r.resolution}"
        )

    footer = (
        "\n\n================ YOUR TASK ================\n"
        "Write a concise SUGGESTED RESOLUTION for the NEW TICKET, grounded "
        "in the retrieved past resolutions above. End with a one-line note "
        "stating whether the retrieved examples closely match or whether a "
        "human should verify.\n"
    )

    return header + new_block + "\n".join(parts) + footer


def _call_gemini_once(client, prompt: str) -> str:
    try:
        response = client.models.generate_content(
            model=settings.models.gemini_model,
            contents=prompt,
        )
    except Exception as exc:
        raise classify_llm_exception(exc) from exc

    text = getattr(response, "text", None)
    if not text or not str(text).strip():
        raise LLMError(
            "Gemini returned an empty response. The prompt may have been "
            "blocked by a safety filter, or the model returned no content."
        )
    return str(text).strip()


def generate(ticket: TicketIn, retrieval: RetrievalResult,
             artifacts: Artifacts) -> ResolutionSuggestion:
    """Draft a grounded resolution, retrying on transient rate limits.

    Raises LLMError (or a subclass) if every attempt fails -- the caller
    decides whether that means escalate-to-human or abort.
    """
    if artifacts.gemini_client is None:
        raise LLMError(
            "No Gemini client is loaded. Load artifacts with "
            "require_gemini=True before requesting a resolution."
        )

    prompt = build_prompt(ticket, retrieval)
    last_error: LLMError | None = None

    for attempt in range(settings.gemini.max_retries):
        try:
            text = _call_gemini_once(artifacts.gemini_client, prompt)
            return ResolutionSuggestion(
                text=text,
                llm_model=settings.models.gemini_model,
                grounded_ticket_ids=[r.id for r in retrieval.retrieved],
            )
        except RateLimitError as exc:
            last_error = exc
            if attempt < settings.gemini.max_retries - 1:
                time.sleep(settings.gemini.backoff_base_sec ** (attempt + 1))
        except LLMError as exc:
            # Auth and network failures are not worth retrying blindly.
            raise exc

    raise last_error if last_error else LLMError("Gemini call failed.")
