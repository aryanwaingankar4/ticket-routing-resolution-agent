# src/rag/suggest_resolution.py
"""
RAG query / suggestion layer for the IT-support-ticket routing project.

Given a NEW, unseen ticket (title + description) this script:
  1. Embeds the new ticket with the same BGE model used to build the index.
  2. Searches the FAISS index for the top-K most similar past tickets.
  3. Retrieves their resolutions from the index-aligned metadata.
  4. Sends the new ticket + retrieved resolutions to Gemini, asking it to
     draft a suggested resolution GROUNDED in those past fixes.
  5. Prints the retrieved similar tickets (with similarity scores) and the
     LLM's suggested resolution (or an escalation message for novel tickets).

Run from the project root:
    python src/rag/suggest_resolution.py

-----------------------------------------------------------------------------
IMPORTANT SDK / MODEL NOTE (deliberate, defensible deviation from the brief):

The brief asked for the `google-generativeai` package with a
`gemini-2.0-flash` default. Both are now deprecated:
  - The legacy `google-generativeai` SDK's support ended 2025-11-30; Google's
    current, GA, unified SDK is `google-genai` (imported as `from google
    import genai`).
  - `gemini-2.0-flash` is scheduled for shutdown, so it is a poor long-term
    default.

This file therefore targets the CURRENT unified SDK and a current free-tier
flash model (`gemini-2.5-flash`), exposed as a single MODEL_NAME constant so
renames/deprecations are a one-line change. Install with:

    pip install google-genai

If Google renames the model again, edit MODEL_NAME below and nothing else.
-----------------------------------------------------------------------------
"""

import os
import sys
import json
import random

import numpy as np


# ---------------------------------------------------------------------------
# Reproducibility: keep the project's seeding convention consistent even
# though FAISS retrieval is deterministic given a fixed index.
# ---------------------------------------------------------------------------
random.seed(42)
np.random.seed(42)


# ---------------------------------------------------------------------------
# Configuration constants (near the top so they are easy to change).
# ---------------------------------------------------------------------------
MODEL_NAME = "gemini-flash-lite-latest"         # current free-tier flash model
EMBED_MODEL_NAME = "BAAI/bge-base-en-v1.5"    # MUST match the model used to build index
TOP_K = 5

# Cosine-similarity floor for "is any past ticket actually relevant?".
# 0.35 is a reasonable starting point for MiniLM embeddings of short ticket
# text. This is NOT a tuned value - it should be revisited against real usage
# data (e.g. by inspecting score distributions of true matches vs. novel
# tickets). It doubles as a preview of the upcoming confidence-based Agentic
# escalation layer.
SIMILARITY_THRESHOLD = 0.65

# If True, print the full constructed LLM prompt before sending it to Gemini.
DEBUG = False


# ---------------------------------------------------------------------------
# Path resolution convention (identical to every other script in this
# project): project root is TWO directories up from THIS file's location,
# never relative to the invocation directory. Works on Windows / PowerShell.
#
#   this file:  <root>/src/rag/suggest_resolution.py
#   dirname 1:  <root>/src/rag
#   dirname 2:  <root>/src
#   dirname 3:  <root>            <- project root
# ---------------------------------------------------------------------------
_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(os.path.dirname(_THIS_DIR))

DATA_DIR = os.path.join(PROJECT_ROOT, "data")
# Model-aware artifact filenames - MUST match what build_vector_index.py
# produces for BGE, or this script would load a mismatched (or nonexistent)
# index.
INDEX_PATH = os.path.join(DATA_DIR, "ticket_index_bge-base-en-v1-5.faiss")
METADATA_PATH = os.path.join(DATA_DIR, "ticket_metadata_bge-base-en-v1-5.json")
DOTENV_PATH = os.path.join(PROJECT_ROOT, ".env")


# ---------------------------------------------------------------------------
# Network / offline error detection. Same "network_markers" pattern used
# elsewhere in this project (train_embeddings.py, build_vector_index.py).
# ---------------------------------------------------------------------------
NETWORK_MARKERS = [
    "connection",
    "connectionerror",
    "timed out",
    "timeout",
    "proxy",
    "proxyerror",
    "max retries",
    "failed to establish",
    "huggingface.co",
    "name resolution",
    "temporary failure in name resolution",
    "network is unreachable",
    "getaddrinfo",
    "ssl",
    "certificate",
    "offline",
    "couldn't connect",
    "could not connect",
    "unable to load",
    "no address associated",
]


def _looks_like_network_error(exc):
    """Return True if the exception text matches a known network failure marker."""
    text = f"{type(exc).__name__} {exc}".lower()
    return any(marker in text for marker in NETWORK_MARKERS)


def _fail(message, code=1):
    """Print a clean error message (no traceback) and exit."""
    print(f"\n[ERROR] {message}", file=sys.stderr)
    sys.exit(code)


# ===========================================================================
# 1. LOAD INDEX + METADATA + EMBEDDING MODEL
# ===========================================================================
def load_faiss():
    """Import faiss with a clean install message if it is missing."""
    try:
        import faiss
    except ImportError:
        _fail(
            "faiss (faiss-cpu) is required but not installed.\n"
            "Install it with:\n\n    pip install faiss-cpu\n"
        )
    return faiss


def load_index_and_metadata(faiss):
    """
    Load the FAISS index and the index-aligned metadata, and verify they are
    in sync. Returns (index, metadata_list).
    """
    if not os.path.isfile(INDEX_PATH) or not os.path.isfile(METADATA_PATH):
        missing = [p for p in (INDEX_PATH, METADATA_PATH) if not os.path.isfile(p)]
        _fail(
            "Required RAG artifact(s) not found:\n"
            + "\n".join(f"    - {m}" for m in missing)
            + "\n\nThe vector index has not been built yet (or was removed).\n"
            "Build it first by running, from the project root:\n\n"
            "    python src/rag/build_vector_index.py\n"
        )

    try:
        index = faiss.read_index(INDEX_PATH)
    except Exception as exc:  # noqa: BLE001
        _fail(
            f"Failed to read FAISS index at {INDEX_PATH} "
            f"({type(exc).__name__}: {exc}).\n"
            f"The file may be corrupted - rebuild it with:\n\n"
            f"    python src/rag/build_vector_index.py\n"
        )

    try:
        with open(METADATA_PATH, "r", encoding="utf-8") as f:
            metadata = json.load(f)
    except Exception as exc:  # noqa: BLE001
        _fail(
            f"Failed to read metadata at {METADATA_PATH} "
            f"({type(exc).__name__}: {exc}).\n"
            f"Rebuild it with:\n\n    python src/rag/build_vector_index.py\n"
        )

    if not isinstance(metadata, list):
        _fail(
            f"{METADATA_PATH} is not a JSON list as expected. Rebuild the "
            f"index with:\n\n    python src/rag/build_vector_index.py\n"
        )

    # ---- CRITICAL sync check ---------------------------------------------
    # If ntotal != len(metadata), one artifact was rebuilt without the other.
    # FAISS returns integer positions; if metadata is a different length or
    # order, position i no longer maps to the right ticket and we would return
    # WRONG resolutions with no error. Fail loudly instead.
    if index.ntotal != len(metadata):
        _fail(
            f"INDEX / METADATA OUT OF SYNC:\n"
            f"    FAISS index.ntotal = {index.ntotal}\n"
            f"    len(metadata)      = {len(metadata)}\n\n"
            f"These MUST match, or retrieved resolutions would be misaligned "
            f"(wrong answers, silently). This usually means one file was "
            f"rebuilt and the other wasn't. Rebuild BOTH together with:\n\n"
            f"    python src/rag/build_vector_index.py\n"
        )

    print(f"[load] FAISS index: ntotal={index.ntotal}, dim={index.d}")
    print(f"[load] Metadata entries: {len(metadata)}  (in sync with index)")
    return index, metadata


def load_embedding_model():
    """
    Load SentenceTransformer(BAAI/bge-base-en-v1.5), reusing the project's
    network-error handling pattern. Returns the model.
    """
    try:
        from sentence_transformers import SentenceTransformer
    except ImportError:
        _fail(
            "sentence-transformers is required but not installed.\n"
            "Install it with:\n\n    pip install sentence-transformers\n"
        )

    try:
        print(f"[model] Loading embedding model '{EMBED_MODEL_NAME}' ...")
        model = SentenceTransformer(EMBED_MODEL_NAME)
    except Exception as exc:  # noqa: BLE001
        if _looks_like_network_error(exc):
            _fail(
                f"Failed to download/load the embedding model "
                f"'{EMBED_MODEL_NAME}' due to what looks like a network "
                f"problem:\n    {type(exc).__name__}: {exc}\n\n"
                f"Internet access is needed the FIRST time only, to download "
                f"the model from huggingface.co. Please check:\n"
                f"  - your internet connection is up,\n"
                f"  - any corporate proxy/VPN is configured "
                f"(HTTP_PROXY / HTTPS_PROXY),\n"
                f"  - huggingface.co is reachable.\n"
                f"Once cached locally, future runs work offline."
            )
        _fail(
            f"Failed to load embedding model '{EMBED_MODEL_NAME}': "
            f"{type(exc).__name__}: {exc}"
        )
    return model


# ===========================================================================
# 2. GEMINI SETUP
# ===========================================================================
def setup_gemini_client():
    """
    Load GEMINI_API_KEY from the project-root .env and build a genai.Client.
    The key is never printed or logged. Returns a configured client.
    """
    try:
        from dotenv import load_dotenv
    except ImportError:
        _fail(
            "python-dotenv is required but not installed.\n"
            "Install it with:\n\n    pip install python-dotenv\n"
        )

    try:
        from google import genai
    except ImportError:
        _fail(
            "The Google Gen AI SDK is required but not installed.\n"
            "Install it with:\n\n    pip install google-genai\n\n"
            "(Note: this is the CURRENT unified SDK. The older "
            "'google-generativeai' package is deprecated.)"
        )

    # Load .env from the project root explicitly (not cwd-dependent).
    load_dotenv(dotenv_path=DOTENV_PATH)
    api_key = os.getenv("GEMINI_API_KEY", "").strip()

    if not api_key:
        _fail(
            "GEMINI_API_KEY is missing or empty.\n\n"
            "To fix this:\n"
            f"  1. Create a file named '.env' at the project root:\n"
            f"       {DOTENV_PATH}\n"
            "  2. Add this single line to it (no quotes, no spaces):\n"
            "       GEMINI_API_KEY=your_key_here\n"
            "  3. Get a free API key at:\n"
            "       https://aistudio.google.com/apikey\n\n"
            "No API call will be attempted without a key."
        )

    try:
        client = genai.Client(api_key=api_key)
    except Exception as exc:  # noqa: BLE001
        # Do NOT echo the key or full env in the message.
        _fail(
            f"Failed to initialize the Gemini client "
            f"({type(exc).__name__}: {exc}). Check that GEMINI_API_KEY is a "
            f"valid key from https://aistudio.google.com/apikey"
        )
    return client


# ===========================================================================
# 3. RETRIEVAL
# ===========================================================================
def retrieve_similar_tickets(query_text, model, index, metadata, faiss, top_k=TOP_K):
    """
    Embed the query ticket, search the FAISS index, and return the top_k most
    similar past tickets with their similarity scores.

    Returns a list of dicts sorted by similarity DESCENDING, each containing:
        id, title, description, category, resolution, priority, similarity
    """
    # Embed the query using the SAME text convention (title + " " + desc is
    # already combined by the caller) and the SAME model as the index.
    query_emb = model.encode([query_text], convert_to_numpy=True)
    query_emb = np.asarray(query_emb, dtype=np.float32)
    query_emb = np.ascontiguousarray(query_emb)

    # L2-normalize the query. This MUST match the normalization used when the
    # index was built (build_vector_index.py normalized the corpus), otherwise
    # inner-product search no longer equals cosine similarity and we get
    # plausible-looking but wrong rankings.
    faiss.normalize_L2(query_emb)

    # Guard: don't ask for more neighbours than exist in the index.
    k = min(top_k, index.ntotal)
    scores, indices = index.search(query_emb, k)

    results = []
    for score, idx in zip(scores[0], indices[0]):
        if idx < 0:
            # FAISS returns -1 for empty slots when k > ntotal; skip defensively.
            continue
        entry = metadata[idx]
        results.append(
            {
                "id": entry.get("id"),
                "title": entry.get("title", ""),
                "description": entry.get("description", ""),
                "category": entry.get("category", ""),
                "resolution": entry.get("resolution", ""),
                "priority": entry.get("priority"),
                "similarity": float(score),
            }
        )

    # IndexFlatIP already returns results in descending score order, but sort
    # explicitly/defensively rather than relying on that.
    results.sort(key=lambda r: r["similarity"], reverse=True)
    return results


# ===========================================================================
# 5. PROMPT CONSTRUCTION (separated from the API call for testability)
# ===========================================================================
def build_llm_prompt(query_title, query_description, retrieved):
    """
    Build a readable, structured prompt that instructs Gemini to ground its
    suggested resolution in the retrieved past resolutions.
    """
    header = (
        "You are an IT-support assistant that drafts suggested resolutions for "
        "new support tickets.\n\n"
        "IMPORTANT INSTRUCTIONS:\n"
        "- Base your suggestion ONLY on the RETRIEVED PAST TICKETS below. These "
        "are real past fixes from our ticket history.\n"
        "- Do NOT invent a generic answer from outside knowledge. Ground every "
        "step in what the retrieved resolutions actually did.\n"
        "- If the retrieved examples do not fully match the new ticket, say so "
        "explicitly and recommend that a human agent double-check.\n"
        "- Keep the suggestion concise: a few sentences of clear, actionable "
        "steps.\n"
    )

    new_ticket_block = (
        "\n================ NEW TICKET (needs a resolution) ================\n"
        f"Title:       {query_title}\n"
        f"Description: {query_description}\n"
    )

    examples_block = ["\n================ RETRIEVED PAST TICKETS ================"]
    for rank, r in enumerate(retrieved, start=1):
        examples_block.append(
            f"\n--- Past ticket #{rank} "
            f"(similarity {r['similarity']:.3f}, category: {r['category']}) ---\n"
            f"Title:       {r['title']}\n"
            f"Description: {r['description']}\n"
            f"Resolution:  {r['resolution']}"
        )
    examples_str = "\n".join(examples_block)

    footer = (
        "\n\n================ YOUR TASK ================\n"
        "Write a concise SUGGESTED RESOLUTION for the NEW TICKET, grounded in "
        "the retrieved past resolutions above. End with a one-line note stating "
        "whether the retrieved examples closely match or whether a human should "
        "verify.\n"
    )

    return header + new_ticket_block + examples_str + footer


# ===========================================================================
# 6. GEMINI API CALL with robust error handling
# ===========================================================================
def call_gemini(client, prompt):
    """
    Send the prompt to Gemini and return the generated text.

    Handles the free tier's real failure modes cleanly (rate limit/quota,
    auth, network, blocked/empty responses, and a general fallback) instead
    of crashing with a raw traceback. Returns the response text, or exits
    cleanly via _fail() on a fatal error.
    """
    # Import the SDK's error types so we can distinguish failure modes.
    # ClientError covers 4xx (429 rate limit, 401/403 auth); ServerError 5xx.
    try:
        from google.genai import errors as genai_errors
        APIError = genai_errors.APIError
        ClientError = genai_errors.ClientError
        ServerError = genai_errors.ServerError
    except Exception:  # noqa: BLE001 - if error module shape changes, degrade gracefully
        APIError = ClientError = ServerError = None

    try:
        response = client.models.generate_content(
            model=MODEL_NAME,
            contents=prompt,
        )
    except Exception as exc:  # noqa: BLE001 - we classify below, never raw-crash
        # Rate limit / quota (HTTP 429).
        code = getattr(exc, "code", None)
        exc_text = f"{type(exc).__name__} {exc}".lower()

        is_client_error = ClientError is not None and isinstance(exc, ClientError)
        is_server_error = ServerError is not None and isinstance(exc, ServerError)

        if code == 429 or "resource_exhausted" in exc_text or "quota" in exc_text \
                or "rate limit" in exc_text:
            _fail(
                "Gemini free-tier rate limit / quota hit (HTTP 429).\n"
                "Wait a minute and retry, or check your quota and limits at:\n"
                "    https://aistudio.google.com\n"
                "Free-tier limits are per-minute AND per-day, so a long demo "
                "session can exhaust the daily quota."
            )

        # Auth: invalid/expired key (HTTP 401/403 or UNAUTHENTICATED).
        if code in (401, 403) or "unauthenticated" in exc_text \
                or "permission" in exc_text or "api key" in exc_text \
                or "api_key" in exc_text or "invalid key" in exc_text:
            _fail(
                "Gemini authentication failed - your API key was rejected.\n"
                "Check that GEMINI_API_KEY in your .env is a valid, active key "
                "(no extra spaces/quotes) from:\n"
                "    https://aistudio.google.com/apikey"
            )

        # Network / connectivity.
        if _looks_like_network_error(exc) or is_server_error:
            _fail(
                f"Could not reach the Gemini API "
                f"({type(exc).__name__}: {exc}).\n"
                f"Check your internet connection and any proxy/VPN settings "
                f"(HTTP_PROXY / HTTPS_PROXY), then retry. If it persists, the "
                f"service may be temporarily unavailable."
            )

        # Known API error but not one of the specific cases above.
        if APIError is not None and isinstance(exc, APIError):
            _fail(f"Gemini API error ({type(exc).__name__}): {exc}")

        # General fallback: catch-all so nothing crashes with a raw traceback.
        _fail(f"Unexpected error calling Gemini "
              f"({type(exc).__name__}): {exc}")

    # ---- Empty / blocked response handling --------------------------------
    # Gemini can return no usable text if content was blocked by safety
    # filters or the candidate finished without text. Handle before printing.
    text = getattr(response, "text", None)
    if text is None or not str(text).strip():
        block_info = ""
        feedback = getattr(response, "prompt_feedback", None)
        if feedback is not None:
            block_info = f" (prompt_feedback: {feedback})"
        candidates = getattr(response, "candidates", None)
        finish = ""
        if candidates:
            finish_reason = getattr(candidates[0], "finish_reason", None)
            if finish_reason is not None:
                finish = f" (finish_reason: {finish_reason})"
        return (
            "[Gemini returned an empty or blocked response"
            f"{block_info}{finish}. No suggestion could be generated. "
            "A human agent should handle this ticket.]"
        )

    return str(text).strip()


# ===========================================================================
# ORCHESTRATION: retrieve -> (guard) -> prompt -> LLM
# ===========================================================================
def suggest_resolution_for_ticket(query_title, query_description,
                                   model, index, metadata, faiss, client,
                                   top_k=TOP_K):
    """
    End-to-end for a single ticket. Prints the retrieved tickets and either
    the LLM suggestion or the low-similarity escalation message.
    Returns a dict summarizing what happened (handy for a future API layer).
    """
    query_text = f"{query_title} {query_description}"

    retrieved = retrieve_similar_tickets(
        query_text, model, index, metadata, faiss, top_k=top_k
    )

    # ---- Print retrieved tickets -----------------------------------------
    print("\n  Retrieved similar past tickets (by cosine similarity):")
    if not retrieved:
        print("    (none - the index appears to be empty)")
        print("\n  DECISION: escalate to a human (no candidates retrieved).")
        return {"escalated": True, "reason": "no_candidates", "retrieved": []}

    for rank, r in enumerate(retrieved, start=1):
        print(
            f"    {rank}. [sim {r['similarity']:.3f}] "
            f"(id={r['id']}, {r['category']}) {r['title']}"
        )

    top_sim = retrieved[0]["similarity"]

    # ---- LOW-SIMILARITY GUARD (preview of the Agentic escalation layer) ---
    if top_sim < SIMILARITY_THRESHOLD:
        print(
            f"\n  [LOW-SIMILARITY GUARD] Top similarity {top_sim:.3f} is below "
            f"the threshold {SIMILARITY_THRESHOLD:.2f}."
        )
        print(
            "  No sufficiently similar past ticket was found, so the retrieved "
            "resolutions are NOT a reliable basis for an LLM suggestion."
        )
        print(
            "  DECISION: SKIP the LLM call and ESCALATE TO A HUMAN AGENT.\n"
            "  (This confidence-based skip is a preview of the project's "
            "upcoming Agentic escalation layer.)"
        )
        return {
            "escalated": True,
            "reason": "below_similarity_threshold",
            "top_similarity": top_sim,
            "retrieved": retrieved,
        }

    # ---- Above threshold: build prompt and call the LLM -------------------
    prompt = build_llm_prompt(query_title, query_description, retrieved)

    if DEBUG:
        print("\n  [DEBUG] Full prompt sent to Gemini:")
        print("  " + "-" * 68)
        for line in prompt.splitlines():
            print("  | " + line)
        print("  " + "-" * 68)

    print(f"\n  Calling Gemini ({MODEL_NAME}) for a grounded suggestion ...")
    suggestion = call_gemini(client, prompt)

    print("\n  --- SUGGESTED RESOLUTION (grounded in retrieved tickets) ---")
    for line in suggestion.splitlines():
        print("  " + line)
    print("  ------------------------------------------------------------")

    return {
        "escalated": False,
        "top_similarity": top_sim,
        "retrieved": retrieved,
        "suggestion": suggestion,
    }


# ===========================================================================
# DEMO EXAMPLES + MAIN
# ===========================================================================
# Hardcoded example NEW tickets to demonstrate the pipeline end-to-end without
# user input. Kept clearly separate from the reusable functions above.
#   - The first two SHOULD find good matches (common IT-support scenarios).
#   - The last one deliberately should NOT match anything in an IT-support
#     dataset, to demonstrate the low-similarity escalation guard.
DEMO_TICKETS = [
    {
        "title": "Cannot connect to company VPN from home",
        "description": (
            "Since this morning my VPN client fails to connect when I work "
            "remotely. It gets stuck on 'authenticating' and then times out. "
            "Office network works fine but I need VPN for internal tools."
        ),
    },
    {
        "title": "Forgot password and locked out of my account",
        "description": (
            "I entered the wrong password too many times and now my account is "
            "locked. I need a password reset so I can log back in and access my "
            "email."
        ),
    },
    {
        "title": "What's the weather going to be like tomorrow?",
        "description": (
            "I'm planning a weekend trip to the hills and want to know if it "
            "will rain tomorrow afternoon so I can decide whether to pack an "
            "umbrella."
        ),
    },
]


def run_demo():
    faiss = load_faiss()
    index, metadata = load_index_and_metadata(faiss)
    model = load_embedding_model()
    client = setup_gemini_client()

    print("\n" + "=" * 72)
    print("RAG SUGGESTION DEMO")
    print(f"  embed model : {EMBED_MODEL_NAME}")
    print(f"  LLM model   : {MODEL_NAME}")
    print(f"  top_k       : {TOP_K}   similarity threshold : {SIMILARITY_THRESHOLD}")
    print("=" * 72)

    for i, ticket in enumerate(DEMO_TICKETS, start=1):
        print("\n" + "#" * 72)
        print(f"# EXAMPLE {i} of {len(DEMO_TICKETS)}")
        print("#" * 72)
        print(f"  NEW TICKET title       : {ticket['title']}")
        print(f"  NEW TICKET description : {ticket['description']}")

        suggest_resolution_for_ticket(
            ticket["title"], ticket["description"],
            model, index, metadata, faiss, client, top_k=TOP_K,
        )

    print("\n" + "=" * 72)
    print("DEMO COMPLETE")
    print("=" * 72)


if __name__ == "__main__":
    run_demo()
