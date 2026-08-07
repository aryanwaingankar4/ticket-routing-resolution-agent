# src/classification/generate_expanded_benchmark.py
# ---------------------------------------------------------------------------
# Generate 35 NEW novel benchmark tickets (5 per category x 7 categories) to
# EXTEND the hand-written generalization benchmark from 14 -> 49 tickets.
#
# RELATIONSHIP TO THE OTHER TWO FILES
# -----------------------------------
#   * generalization_test.py  (READ-ONLY, never modified by this script)
#       Owns the canonical NOVEL_TICKETS list: 14 hand-written, deliberately
#       non-technical tickets (2 per category). That list is THE primary
#       accuracy benchmark. This script imports NOVEL_TICKETS purely to (a)
#       mirror its exact dict structure ({"text", "expected"}) and (b) feed
#       Gemini 2-3 real examples per category as few-shot STYLE ANCHORS so the
#       generated tickets match the same plain-spoken register. Importing it is
#       side-effect-free: everything runnable in that file is guarded under
#       `if __name__ == "__main__":`.
#
#   * generate_calibration_set.py  (the working precedent this MIRRORS)
#       Established this project's Gemini-based paraphrase-and-verify pattern:
#       the unified google-genai SDK (`from google import genai`), model
#       "gemini-flash-lite-latest", GEMINI_API_KEY loaded from .env via
#       python-dotenv, a self-consistency check that independently re-asks
#       Gemini to classify the generated text, retry-with-backoff on rate
#       limits, os.path-based path resolution, and seed=42 reproducibility.
#
# CRITICAL DEVIATION FROM THE CALIBRATION SCRIPT
# ----------------------------------------------
#   For the calibration set, self-consistency mismatches were flagged inline
#   but AUTO-KEPT (ground truth was still usable for threshold calibration).
#   For THIS benchmark -- which is the primary ACCURACY metric -- a mismatch
#   must NOT be auto-kept. Any ticket whose independent Gemini classification
#   disagrees with the intended category is written to a SEPARATE review file
#   and EXCLUDED from the clean output until a human manually approves it.
#
# WHAT THIS SCRIPT DOES *NOT* DO
# ------------------------------
#   It does NOT build the combined 49-ticket NOVEL_TICKETS_EXPANDED list. Its
#   only job is to generate + vet the 35 NEW tickets. Merging old + new is a
#   deliberate, separate later step, to be done only after the flagged tickets
#   have been human-reviewed.
#
# OUTPUTS (both under <project_root>/data/):
#   * expanded_benchmark_new_tickets.json
#         Clean, PASSED tickets only, in the EXACT NOVEL_TICKETS shape:
#         [{"text": "...", "expected": "<category>"}, ...]
#   * expanded_benchmark_flagged_for_review.json
#         Tickets whose self-consistency check failed, each showing both the
#         intended category and Gemini's independent guess, for manual review.
#   * .expanded_benchmark_checkpoint.json  (internal, resumable state)
#         Bookkeeping so the run is interruptible/resumable like its precedent.
#         NOT a deliverable; safe to delete once the run is complete.
#
# HOW TO RUN (from the project root):
#     python src/classification/generate_expanded_benchmark.py
#
# This makes MANY sequential Gemini API calls (2 per candidate ticket: one to
# generate, one to independently classify), each separated by a
# GEMINI_CALL_DELAY_SEC delay to respect the free-tier rate limit, so it will
# take several minutes. It is safely interruptible: press Ctrl+C (or on a
# transient crash) and re-run the same command -- it checkpoints after every
# resolved candidate and resumes where it left off.
# ---------------------------------------------------------------------------

import os
import sys
import json
import time
import random

# --- Third-party imports with clear pip-install messages (project convention)
try:
    import numpy as np
except ImportError:
    print("ERROR: numpy is not installed. Install it with:\n"
          "    pip install numpy")
    sys.exit(1)

try:
    from dotenv import load_dotenv
except ImportError:
    print("ERROR: python-dotenv is not installed. Install it with:\n"
          "    pip install python-dotenv")
    sys.exit(1)

try:
    from google import genai
except ImportError:
    print("ERROR: the Google Gen AI SDK is not installed. Install it with:\n"
          "    pip install google-genai\n"
          "(NOTE: this is the new unified SDK, imported as "
          "`from google import genai`.)")
    sys.exit(1)

# ---------------------------------------------------------------------------
# PATH RESOLUTION (project convention: root = two dirs up from this file)
# ---------------------------------------------------------------------------
# This file lives at <root>/src/classification/generate_expanded_benchmark.py,
# so walking up twice from its own directory yields the project root. Using
# os.path.* keeps this correct on Windows too.
_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(os.path.dirname(_THIS_DIR))

# Make the classification package importable regardless of CWD, so that
# `from generalization_test import NOVEL_TICKETS` resolves to the sibling file.
if _THIS_DIR not in sys.path:
    sys.path.insert(0, _THIS_DIR)

# --- Import the canonical benchmark tickets (READ-ONLY reference) -----------
# generalization_test.py guards all executable work under __main__, so this
# import only materializes the NOVEL_TICKETS list literal -- no test is run.
try:
    from generalization_test import NOVEL_TICKETS
except ImportError as exc:
    print("ERROR: could not import NOVEL_TICKETS from generalization_test.py.\n"
          "This script expects generalization_test.py to sit next to it in\n"
          f"    {_THIS_DIR}\n"
          f"Details: {exc}")
    sys.exit(1)

# --- Project-wide reproducibility convention -------------------------------
random.seed(42)
np.random.seed(42)

# ---------------------------------------------------------------------------
# CONFIGURATION
# ---------------------------------------------------------------------------

# How many NEW tickets to generate per category (5 x 7 = 35).
NEW_PER_CATEGORY = 5

# The 7 canonical categories, spelled EXACTLY as in the rest of the project.
# (Mirrors KNOWN_CATEGORIES in generate_calibration_set.py.)
KNOWN_CATEGORIES = [
    "Infrastructure",
    "Application",
    "Security",
    "Database",
    "Storage",
    "Network",
    "Access Management",
]

# How many real NOVEL_TICKETS examples to show Gemini per category as
# few-shot STYLE ANCHORS. Each category has exactly 2 in NOVEL_TICKETS, so we
# cap at whatever is actually available (2-3 as requested).
STYLE_ANCHORS_PER_CATEGORY = 3

# Gemini model. Do NOT use "gemini-2.5-flash" -- that alias is retired and
# 404s for new API users as of mid-2026.
GEMINI_MODEL = "gemini-flash-lite-latest"

# Rate limit: free tier is 15 requests/minute, 500/day. 4.5s between EVERY
# call keeps us comfortably under 15/min. BOTH the generation call and the
# self-consistency classification call each pay this delay separately -- we
# never batch or parallelize in a way that would burst above the limit.
GEMINI_CALL_DELAY_SEC = 4.5

# Retry / backoff tuning (mirrors the calibration script).
MAX_RETRIES = 3
BACKOFF_SCHEDULE = [5, 15, 30]   # seconds to wait before retry 1, 2, 3

# --- Output / checkpoint paths ---------------------------------------------
CLEAN_OUTPUT_JSON = os.path.join(
    PROJECT_ROOT, "data", "expanded_benchmark_new_tickets.json"
)
FLAGGED_OUTPUT_JSON = os.path.join(
    PROJECT_ROOT, "data", "expanded_benchmark_flagged_for_review.json"
)
CHECKPOINT_JSON = os.path.join(
    PROJECT_ROOT, "data", ".expanded_benchmark_checkpoint.json"
)


# ---------------------------------------------------------------------------
# GEMINI CLIENT SETUP (mirrors generate_calibration_set.build_gemini_client)
# ---------------------------------------------------------------------------
def build_gemini_client():
    """Load GEMINI_API_KEY from .env and return an initialized genai client.

    The key is never printed or logged.
    """
    load_dotenv(os.path.join(PROJECT_ROOT, ".env"))
    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        print("ERROR: GEMINI_API_KEY is not set.\n"
              "Add a line like the following to your .env file at the project "
              "root:\n"
              "    GEMINI_API_KEY=your_key_here\n"
              "(Get a key from https://aistudio.google.com/apikey)")
        sys.exit(1)
    try:
        client = genai.Client(api_key=api_key)
    except Exception as exc:  # noqa: BLE001 - surface a clean message, no key
        print("ERROR: failed to initialize the Gemini client. "
              "Check that your GEMINI_API_KEY is valid.\n"
              f"Details: {exc}")
        sys.exit(1)
    return client


def _is_rate_limit_error(exc):
    """Best-effort detection of rate-limit/quota errors from the message/type."""
    marker_text = f"{type(exc).__name__} {exc}".lower()
    return any(m in marker_text
               for m in ("rate limit", "ratelimit", "quota",
                         "429", "resource_exhausted", "resource exhausted"))


# ---------------------------------------------------------------------------
# STYLE ANCHORS: pull real NOVEL_TICKETS examples per category
# ---------------------------------------------------------------------------
def build_style_anchor_map():
    """Group the existing NOVEL_TICKETS by category for few-shot anchoring.

    Returns {category: [example_text, ...]}. Also validates that every
    NOVEL_TICKETS label is one of the known 7 (fail loudly on a typo, matching
    the strictness of generalization_test.validate_expected_labels).
    """
    known = set(KNOWN_CATEGORIES)
    anchors = {cat: [] for cat in KNOWN_CATEGORIES}

    bad = []
    for i, ticket in enumerate(NOVEL_TICKETS):
        cat = ticket.get("expected")
        text = ticket.get("text")
        if cat not in known:
            bad.append((i, cat))
            continue
        if isinstance(text, str) and text.strip():
            anchors[cat].append(text.strip())

    if bad:
        print("ERROR: NOVEL_TICKETS contains expected label(s) outside the "
              "known 7 categories (exact, case-sensitive):")
        for idx, label in bad:
            print(f"    - ticket index {idx}: {label!r}")
        print(f"Known categories: {KNOWN_CATEGORIES}")
        sys.exit(1)

    # Sanity: every category should have at least one anchor to show Gemini.
    empty = [cat for cat in KNOWN_CATEGORIES if not anchors[cat]]
    if empty:
        print("ERROR: no NOVEL_TICKETS style anchors found for category(ies): "
              f"{empty}. Cannot build a faithful style prompt without them.")
        sys.exit(1)

    return anchors


# ---------------------------------------------------------------------------
# PROMPT BUILDERS
# ---------------------------------------------------------------------------
def _build_generation_prompt(category, anchor_examples, already_generated):
    """Build the strict-JSON generation prompt for ONE new ticket.

    * anchor_examples     : real NOVEL_TICKETS texts for this category (style).
    * already_generated   : texts already accepted/seen for this category in
                            THIS run, so Gemini avoids near-duplicates of them.
    """
    anchors_block = "\n".join(
        f'  {n}. "{ex}"' for n, ex in enumerate(anchor_examples, start=1)
    )

    if already_generated:
        avoid_block = "\n".join(
            f'  - "{t}"' for t in already_generated
        )
        avoid_section = (
            "\nYou have ALREADY produced the following tickets for this "
            "category in this session. Your new ticket must describe a "
            "GENUINELY DIFFERENT scenario and must NOT reuse their phrasing "
            "or the same underlying problem:\n"
            f"{avoid_block}\n"
        )
    else:
        avoid_section = ""

    return f"""You are helping build the HELD-OUT accuracy benchmark for an IT \
support ticket classifier. These tickets must read exactly like an ordinary, \
NON-TECHNICAL employee describing a problem to the help desk in their own \
everyday words -- never like a templated or machine-generated log line.

TARGET CATEGORY: {category}

Here are real example tickets that already exist for this category. Match \
their TONE, REGISTER, and plain-spoken everyday style (but NOT their specific \
scenarios):
{anchors_block}
{avoid_section}
Write ONE brand-new IT support ticket for the category "{category}". Hard \
requirements:
  - Plain, conversational, non-technical everyday language -- the voice of a \
regular employee, spoken out loud. First person is natural.
  - No jargon, acronyms, tool names, server names, or log-style phrasing.
  - AVOID the vocabulary and sentence patterns typical of synthetic/templated \
training data (e.g. "authentication failure on primary domain controller", \
"connection to the database instance timed out", "disk utilization exceeded \
threshold"). Describe the SYMPTOM as a person experiences it, not the system \
diagnosis.
  - Describe a scenario that is CLEARLY DISTINCT from every example shown \
above and from any tickets you were told to avoid -- a different situation, \
not a reworded version of one.
  - The problem must still unambiguously belong to the "{category}" category.
  - 1-3 sentences. Natural, varied sentence structure.

Return ONLY a single JSON object, with no markdown, no code fences, and no \
extra commentary, in exactly this shape:
{{"ticket": "<the plain-English ticket text>"}}"""


def _build_classification_prompt(ticket_text):
    """Build the strict-JSON self-consistency classification prompt.

    Mirrors the calibration script's independent-guess step: Gemini is NOT told
    the intended category; it must classify the text cold from the known 7.
    """
    category_list = ", ".join(KNOWN_CATEGORIES)
    return f"""You are an IT support ticket triage assistant. Read the ticket \
below and decide which ONE category it belongs to.

TICKET: {ticket_text}

Choose strictly ONE category from this list, using the exact spelling:
   {category_list}

Return ONLY a single JSON object, with no markdown, no code fences, and no \
extra commentary, in exactly this shape:
{{"category": "<one of the categories above>"}}"""


def _strip_code_fences(text):
    """Remove surrounding ```json ... ``` / ``` ... ``` fences if present."""
    t = text.strip()
    if t.startswith("```"):
        first_newline = t.find("\n")
        if first_newline != -1:
            t = t[first_newline + 1:]
        if t.rstrip().endswith("```"):
            t = t.rstrip()[:-3]
    return t.strip()


# ---------------------------------------------------------------------------
# CORE GEMINI CALL (shared retry/backoff, mirrors the calibration script)
# ---------------------------------------------------------------------------
def _call_gemini_json(client, prompt, temperature):
    """Call Gemini with retry/backoff and parse a single JSON object.

    Returns the parsed dict on success.

    IMPORTANT (per project convention / requirement #7): on failure this
    RAISES a plain Exception rather than calling sys.exit()/os._exit(), so the
    script stays compatible with being wrapped in a retry driver later. This
    is the deliberate difference from the calibration script, which returned
    None and continued; here a failed candidate must not silently vanish.
    """
    last_error = None
    for attempt in range(MAX_RETRIES + 1):
        try:
            response = client.models.generate_content(
                model=GEMINI_MODEL,
                contents=prompt,
                config={
                    "temperature": temperature,
                    "response_mime_type": "application/json",
                },
            )
        except Exception as exc:  # noqa: BLE001
            last_error = exc
            if _is_rate_limit_error(exc) and attempt < MAX_RETRIES:
                wait = BACKOFF_SCHEDULE[min(attempt, len(BACKOFF_SCHEDULE) - 1)]
                print(f"    Rate limit / quota hit (attempt {attempt + 1}"
                      f"/{MAX_RETRIES}). Waiting {wait}s before retry...")
                time.sleep(wait)
                continue
            # Non-retryable, or retries exhausted.
            raise RuntimeError(f"Gemini API call failed: {exc}") from exc

        raw = getattr(response, "text", None)
        if not raw:
            raise RuntimeError(
                "Gemini returned an empty or blocked response (no text)."
            )

        cleaned = _strip_code_fences(raw)
        try:
            parsed = json.loads(cleaned)
        except json.JSONDecodeError as exc:
            raise RuntimeError(
                f"Could not parse Gemini output as JSON. Raw was: {raw!r}"
            ) from exc

        if not isinstance(parsed, dict):
            raise RuntimeError(
                f"Gemini returned JSON that is not an object: {parsed!r}"
            )
        return parsed

    # Retries exhausted on rate-limit path.
    raise RuntimeError(
        f"Gemini API call failed after {MAX_RETRIES} retries. "
        f"Last error: {last_error}"
    )


def generate_one_ticket(client, category, anchor_examples, already_generated):
    """Generate ONE new ticket text for a category. Raises on failure."""
    prompt = _build_generation_prompt(
        category, anchor_examples, already_generated
    )
    parsed = _call_gemini_json(client, prompt, temperature=0.9)
    ticket_text = parsed.get("ticket")
    if not ticket_text or not isinstance(ticket_text, str):
        raise RuntimeError(
            f"Generation JSON missing a valid 'ticket' field: {parsed!r}"
        )
    return ticket_text.strip()


def classify_one_ticket(client, ticket_text):
    """Independently classify a ticket (self-consistency). Raises on failure."""
    prompt = _build_classification_prompt(ticket_text)
    # Low temperature for a stable, deterministic-ish classification decision.
    parsed = _call_gemini_json(client, prompt, temperature=0.0)
    guess = parsed.get("category")
    if not guess or not isinstance(guess, str):
        raise RuntimeError(
            f"Classification JSON missing a valid 'category' field: {parsed!r}"
        )
    return guess.strip()


# ---------------------------------------------------------------------------
# CHECKPOINTING (internal resumable state -- NOT a deliverable)
# ---------------------------------------------------------------------------
def load_checkpoint():
    """Load prior run state, returning (clean_records, flagged_records).

    Each clean record:   {"id", "text", "expected"}
    Each flagged record: {"id", "text", "intended_category", "gemini_guess"}
    (The 'id' is internal bookkeeping for resumability and per-category
    counting; it is stripped from the final deliverable JSON, which must match
    the 2-field NOVEL_TICKETS shape.)
    """
    if not os.path.isfile(CHECKPOINT_JSON):
        return [], []
    try:
        with open(CHECKPOINT_JSON, "r", encoding="utf-8") as fh:
            state = json.load(fh)
    except (json.JSONDecodeError, OSError) as exc:
        print(f"WARNING: could not read checkpoint {CHECKPOINT_JSON} ({exc}). "
              "Starting fresh.")
        return [], []

    if not isinstance(state, dict):
        print(f"WARNING: checkpoint {CHECKPOINT_JSON} has unexpected shape. "
              "Starting fresh.")
        return [], []

    clean = state.get("clean", [])
    flagged = state.get("flagged", [])
    if not isinstance(clean, list) or not isinstance(flagged, list):
        print(f"WARNING: checkpoint {CHECKPOINT_JSON} has unexpected shape. "
              "Starting fresh.")
        return [], []

    print(f"Resuming: checkpoint has {len(clean)} clean and {len(flagged)} "
          "flagged ticket(s) already resolved.")
    return clean, flagged


def save_checkpoint(clean_records, flagged_records):
    """Atomically persist run state so the script is interruptible/resumable."""
    os.makedirs(os.path.dirname(CHECKPOINT_JSON), exist_ok=True)
    tmp_path = CHECKPOINT_JSON + ".tmp"
    payload = {"clean": clean_records, "flagged": flagged_records}
    with open(tmp_path, "w", encoding="utf-8") as fh:
        json.dump(payload, fh, indent=2, ensure_ascii=False)
    os.replace(tmp_path, CHECKPOINT_JSON)


def _atomic_write_json(path, obj):
    """Write JSON atomically (temp file + os.replace)."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    tmp_path = path + ".tmp"
    with open(tmp_path, "w", encoding="utf-8") as fh:
        json.dump(obj, fh, indent=2, ensure_ascii=False)
    os.replace(tmp_path, path)


def write_deliverables(clean_records, flagged_records):
    """Write the two deliverable JSONs in their required shapes.

    * Clean file: EXACT NOVEL_TICKETS shape -> [{"text", "expected"}, ...].
      Internal 'id' bookkeeping is stripped here.
    * Flagged file: review-oriented shape showing both categories.
    """
    clean_out = [
        {"text": r["text"], "expected": r["expected"]}
        for r in clean_records
    ]
    flagged_out = [
        {
            "text": r["text"],
            "intended_category": r["intended_category"],
            "gemini_guess": r["gemini_guess"],
        }
        for r in flagged_records
    ]
    _atomic_write_json(CLEAN_OUTPUT_JSON, clean_out)
    _atomic_write_json(FLAGGED_OUTPUT_JSON, flagged_out)


# ---------------------------------------------------------------------------
# MAIN
# ---------------------------------------------------------------------------
def main():
    print("=" * 75)
    print("Generating EXPANDED benchmark: 5 NEW novel tickets x 7 categories")
    print("Each candidate = 1 generation call + 1 self-consistency call,")
    print(f"every call separated by {GEMINI_CALL_DELAY_SEC}s (free-tier safe).")
    print("Interruptible: Ctrl+C and re-run to resume from the checkpoint.")
    print("=" * 75)

    client = build_gemini_client()
    anchors_by_category = build_style_anchor_map()

    # Trim anchors to the requested few-shot count per category (2-3).
    anchors_by_category = {
        cat: texts[:STYLE_ANCHORS_PER_CATEGORY]
        for cat, texts in anchors_by_category.items()
    }

    # Load any prior progress.
    clean_records, flagged_records = load_checkpoint()

    # Determine how many candidates each category has ALREADY resolved
    # (clean + flagged both count as "resolved" -- we do not re-attempt a
    # flagged candidate automatically; that's for human review).
    def resolved_count(cat):
        c = sum(1 for r in clean_records if r["expected"] == cat)
        f = sum(1 for r in flagged_records if r["intended_category"] == cat)
        return c + f

    # Deterministic category processing order (seeded shuffle for repeatable
    # ordering, per the project's seed=42 convention).
    category_order = list(KNOWN_CATEGORIES)
    random.shuffle(category_order)

    # Global counter used to mint stable internal ids across resumes.
    existing_ids = [r["id"] for r in clean_records] + \
                   [r["id"] for r in flagged_records]
    next_id = (max(existing_ids) + 1) if existing_ids else 1

    try:
        for category in category_order:
            already_done = resolved_count(category)
            if already_done >= NEW_PER_CATEGORY:
                print(f"\n[{category}] already has {already_done} resolved "
                      "candidate(s); skipping (from checkpoint).")
                continue

            print(f"\n[{category}] need "
                  f"{NEW_PER_CATEGORY - already_done} more candidate(s).")

            # Texts we must not near-duplicate: the accepted-clean texts for
            # this category from THIS run so far (flagged ones are excluded
            # from the benchmark anyway, but we still avoid repeating them to
            # keep diversity honest).
            def clean_texts_for(cat):
                return [r["text"] for r in clean_records
                        if r["expected"] == cat]

            def flagged_texts_for(cat):
                return [r["text"] for r in flagged_records
                        if r["intended_category"] == cat]

            while resolved_count(category) < NEW_PER_CATEGORY:
                seen_texts = (clean_texts_for(category)
                              + flagged_texts_for(category))

                # --- Gemini call #1: generate a candidate ------------------
                ticket_text = generate_one_ticket(
                    client, category, anchors_by_category[category], seen_texts
                )
                time.sleep(GEMINI_CALL_DELAY_SEC)  # delay after generation call

                # --- Gemini call #2: independent self-consistency check ----
                gemini_guess = classify_one_ticket(client, ticket_text)
                time.sleep(GEMINI_CALL_DELAY_SEC)  # delay after classify call

                passed = (gemini_guess == category)

                if passed:
                    clean_records.append({
                        "id": next_id,
                        "text": ticket_text,
                        "expected": category,
                    })
                    print(f"  [PASS] (#{next_id}) intended='{category}' "
                          f"gemini='{gemini_guess}'  {ticket_text}")
                else:
                    flagged_records.append({
                        "id": next_id,
                        "text": ticket_text,
                        "intended_category": category,
                        "gemini_guess": gemini_guess,
                    })
                    print(f"  [FLAG] (#{next_id}) intended='{category}' "
                          f"gemini='{gemini_guess}'  -> review file  "
                          f"{ticket_text}")

                next_id += 1

                # Checkpoint after EVERY resolved candidate (pass or flag).
                save_checkpoint(clean_records, flagged_records)

    except KeyboardInterrupt:
        print("\nInterrupted by user (Ctrl+C). Progress is saved in the "
              "checkpoint. Re-run to resume. Writing deliverables from what "
              "has been resolved so far...")

    # --- Always write the two deliverable files from resolved state --------
    write_deliverables(clean_records, flagged_records)

    # --- Terminal summary: per-category clean vs. flagged ------------------
    print("\n" + "=" * 75)
    print("RUN SUMMARY (per category: PASSED clean vs. FLAGGED for review)")
    print("=" * 75)
    print("  {:<20} {:>7} {:>9} {:>9}".format(
        "Category", "Clean", "Flagged", "Target"))
    print("  " + "-" * 48)

    total_clean = 0
    total_flagged = 0
    for cat in KNOWN_CATEGORIES:  # canonical (unshuffled) order for the table
        c = sum(1 for r in clean_records if r["expected"] == cat)
        f = sum(1 for r in flagged_records if r["intended_category"] == cat)
        total_clean += c
        total_flagged += f
        note = "" if c >= NEW_PER_CATEGORY else "  <-- short of target"
        print("  {:<20} {:>7} {:>9} {:>9}{}".format(
            cat, c, f, NEW_PER_CATEGORY, note))

    print("  " + "-" * 48)
    print("  {:<20} {:>7} {:>9} {:>9}".format(
        "TOTAL", total_clean, total_flagged,
        NEW_PER_CATEGORY * len(KNOWN_CATEGORIES)))

    print("\nDeliverables written:")
    print(f"  clean   -> {CLEAN_OUTPUT_JSON}")
    print(f"  flagged -> {FLAGGED_OUTPUT_JSON}")

    if total_flagged:
        print("\nNOTE: flagged tickets are EXCLUDED from the clean benchmark "
              "output. Review them in the flagged file and, if you approve any, "
              "add them manually -- this script never auto-keeps a mismatch.")

    short = [cat for cat in KNOWN_CATEGORIES
             if sum(1 for r in clean_records if r["expected"] == cat)
             < NEW_PER_CATEGORY]
    if short:
        print("\nNOTE: the following categories have FEWER than "
              f"{NEW_PER_CATEGORY} clean tickets because some candidates were "
              "flagged: "
              f"{short}. Re-running will NOT top these up automatically (a "
              "flagged candidate counts as 'resolved'). If you want 5 CLEAN "
              "tickets per category, review/approve flagged ones or delete the "
              "checkpoint entry for those and re-run.")

    print("\nDone. (This script generated + vetted the 35 NEW tickets only. "
          "Merging with the original 14 into NOVEL_TICKETS_EXPANDED is a "
          "separate later step, after you review the flagged file.)")


if __name__ == "__main__":
    main()
