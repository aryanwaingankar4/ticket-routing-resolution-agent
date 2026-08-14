# src/classification/generate_infrastructure_replacement.py
# ---------------------------------------------------------------------------
# Generate 5 NEW "Infrastructure" benchmark tickets to REPLACE a prior batch
# that was generated at the WRONG SCOPE.
#
# WHY THIS SCRIPT EXISTS
# ----------------------
#   A previous run produced 5 "Infrastructure" tickets that described
#   PHYSICAL-BUILDING / FACILITIES problems (flickering lights, stuck doors,
#   burst pipes, elevators, broken heaters). That is the WRONG SCOPE for this
#   project. In this project's dataset (generate_dataset.py's
#   SCENARIOS["Infrastructure"]), every "Infrastructure" scenario is
#   COMPUTE/SERVER focused: high CPU load, a server unresponsive after reboot,
#   memory exhaustion / OOM killer, a Kubernetes node going NotReady, NTP time
#   drift, a scheduled cron job failing, a load balancer dropping healthy
#   backends, disk I/O saturation, and autoscaling not triggering. The entity
#   pool is generic hostnames (PRD-WEB-01, K8S-NODE-14, VM-BATCH-09), NOT
#   offices or buildings.
#
#   The root cause of the bad batch was a generation prompt that only asked for
#   "plain, non-technical, everyday language" WITHOUT anchoring the SCOPE of
#   the word "Infrastructure". This script fixes exactly that: it adds an
#   EXPLICIT compute/server scope anchor AND an explicit NEGATIVE constraint
#   ruling out physical-facilities issues -- while KEEPING the plain,
#   non-technical register. The fix is scope, not technical-vs-plain voice.
#
# RELATIONSHIP TO THE OTHER FILES (mirrors generate_expanded_benchmark.py)
# -----------------------------------------------------------------------
#   * generalization_test.py  (READ-ONLY, never modified here)
#       Owns the canonical NOVEL_TICKETS list. We import it purely to (a)
#       mirror its {"text", "expected"} dict shape and (b) feed Gemini the real
#       Infrastructure examples as few-shot STYLE ANCHORS so tone matches. The
#       import is side-effect-free (all runnable work in that file is guarded
#       under __main__).
#
#   * generate_expanded_benchmark.py  (the precedent this MIRRORS)
#       Same unified google-genai SDK, same model, same GEMINI_API_KEY-from-.env
#       loading, same retry/backoff, same self-consistency-flag-not-auto-keep
#       policy, same atomic writes + resumable checkpoint, same 4.5s delay after
#       EVERY Gemini call, same seed=42 convention.
#
# SELF-CONSISTENCY POLICY (unchanged from the expanded-benchmark precedent)
# ------------------------------------------------------------------------
#   This is the primary ACCURACY benchmark, so a self-consistency mismatch is
#   NEVER auto-kept. After generating each candidate we independently re-ask
#   Gemini (blind to the intended category) to classify it against the known 7
#   categories. Any ticket whose independent guess != "Infrastructure" is
#   written to a SEPARATE review file and EXCLUDED from the clean output until a
#   human approves it.
#
# WHAT THIS SCRIPT DOES *NOT* DO
# ------------------------------
#   It does NOT touch novel_tickets_expanded.json. It only generates + vets the
#   5 replacement tickets and writes them to a NEW file. Merging / swapping the
#   old bad batch for these is a deliberate, separate step to be done by hand
#   AFTER reviewing this output (and the flagged file).
#
# OUTPUTS (under <project_root>/data/):
#   * infrastructure_benchmark_replacement.json
#         Clean, PASSED tickets only, in the EXACT NOVEL_TICKETS shape:
#         [{"text": "...", "expected": "Infrastructure"}, ...]
#   * infrastructure_benchmark_replacement_flagged_for_review.json
#         Tickets whose self-consistency check failed, showing both the
#         intended category and Gemini's independent guess, for manual review.
#   * .infrastructure_benchmark_replacement_checkpoint.json  (internal state)
#         Resumable bookkeeping; NOT a deliverable; safe to delete when done.
#
# HOW TO RUN (from the project root):
#     python src/classification/generate_infrastructure_replacement.py
#
# Makes 2 Gemini calls per candidate (generate + independent classify), each
# separated by GEMINI_CALL_DELAY_SEC. Safely interruptible: Ctrl+C and re-run to
# resume from the checkpoint.
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
_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(os.path.dirname(_THIS_DIR))

# Make the classification package importable regardless of CWD.
if _THIS_DIR not in sys.path:
    sys.path.insert(0, _THIS_DIR)

# --- Import the canonical benchmark tickets (READ-ONLY reference) -----------
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

# This job targets exactly ONE category.
TARGET_CATEGORY = "Infrastructure"

# How many NEW replacement tickets to generate for that category.
NEW_COUNT = 5

# The 7 canonical categories, spelled EXACTLY as in the rest of the project.
KNOWN_CATEGORIES = [
    "Infrastructure",
    "Application",
    "Security",
    "Database",
    "Storage",
    "Network",
    "Access Management",
]

# How many real NOVEL_TICKETS examples to show Gemini as few-shot STYLE
# ANCHORS. Capped at whatever the target category actually has.
STYLE_ANCHORS_COUNT = 3

# Gemini model. Do NOT use "gemini-2.5-flash" -- that alias is retired.
GEMINI_MODEL = "gemini-flash-lite-latest"

# Rate limit: 4.5s between EVERY call (both generation and classification each
# pay this delay separately). Unchanged from the precedent.
GEMINI_CALL_DELAY_SEC = 4.5

# Retry / backoff tuning (mirrors the precedent).
MAX_RETRIES = 3
BACKOFF_SCHEDULE = [5, 15, 30]   # seconds to wait before retry 1, 2, 3

# --- Output / checkpoint paths ---------------------------------------------
CLEAN_OUTPUT_JSON = os.path.join(
    PROJECT_ROOT, "data", "infrastructure_benchmark_replacement.json"
)
FLAGGED_OUTPUT_JSON = os.path.join(
    PROJECT_ROOT, "data",
    "infrastructure_benchmark_replacement_flagged_for_review.json"
)
CHECKPOINT_JSON = os.path.join(
    PROJECT_ROOT, "data",
    ".infrastructure_benchmark_replacement_checkpoint.json"
)


# ---------------------------------------------------------------------------
# GEMINI CLIENT SETUP (mirrors the precedent)
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
# STYLE ANCHORS: pull real NOVEL_TICKETS examples for the target category
# ---------------------------------------------------------------------------
def build_style_anchors():
    """Return the existing NOVEL_TICKETS example texts for TARGET_CATEGORY.

    Also validates that every NOVEL_TICKETS label is one of the known 7 (fail
    loudly on a typo, matching the precedent's strictness).
    """
    known = set(KNOWN_CATEGORIES)

    bad = []
    anchors = []
    for i, ticket in enumerate(NOVEL_TICKETS):
        cat = ticket.get("expected")
        text = ticket.get("text")
        if cat not in known:
            bad.append((i, cat))
            continue
        if cat == TARGET_CATEGORY and isinstance(text, str) and text.strip():
            anchors.append(text.strip())

    if bad:
        print("ERROR: NOVEL_TICKETS contains expected label(s) outside the "
              "known 7 categories (exact, case-sensitive):")
        for idx, label in bad:
            print(f"    - ticket index {idx}: {label!r}")
        print(f"Known categories: {KNOWN_CATEGORIES}")
        sys.exit(1)

    if not anchors:
        print("ERROR: no NOVEL_TICKETS style anchors found for category "
              f"{TARGET_CATEGORY!r}. Cannot build a faithful style prompt "
              "without them.")
        sys.exit(1)

    return anchors


# ---------------------------------------------------------------------------
# SCOPE ANCHOR (the whole point of this script)
# ---------------------------------------------------------------------------
# Positive scope: what "Infrastructure" MEANS in this project (compute/server).
# Negative scope: physical-facilities issues that caused the earlier bad batch.
_INFRA_SCOPE_ANCHOR = """\
SCOPE OF "Infrastructure" IN THIS PROJECT -- READ CAREFULLY.
In this project, "Infrastructure" means COMPUTE / SERVER infrastructure only:
the servers, virtual machines, containers, Kubernetes nodes, and load
balancers that run the company's systems, plus the underlying compute
resources they depend on -- CPU, memory (RAM), disk throughput/space, the
sync of system clocks, scheduled/automated jobs, and automatic scaling of
capacity. Concretely, in-scope problems look like: an internal system or
website being down or extremely slow for everyone; a server that won't come
back up after a restart; systems crashing or freezing because they've run out
of memory; part of the platform being unreachable even though the machines
look "on"; automatic overnight/scheduled jobs not running; timestamps across
systems being wrong or out of sync; things grinding to a halt because storage
is overwhelmed; or the system failing to add capacity automatically under
heavy load.

Do NOT write about physical office issues like lighting, doors, elevators,
plumbing, or heating/cooling systems -- those are OUT OF SCOPE and are NOT
what "Infrastructure" means here. Also avoid personal-device problems (a
single person's laptop/monitor/printer) -- this category is about the shared
servers/compute platform, not one desk's hardware.
"""


# ---------------------------------------------------------------------------
# PROMPT BUILDERS
# ---------------------------------------------------------------------------
def _build_generation_prompt(anchor_examples, already_generated):
    """Build the strict-JSON generation prompt for ONE new Infrastructure ticket.

    Combines: (a) the plain-non-technical register from the precedent, with
    (b) the EXPLICIT compute/server scope anchor + NEGATIVE facilities
    constraint that fixes the wrong-scope mistake.

    * anchor_examples   : real NOVEL_TICKETS Infrastructure texts (style).
    * already_generated : texts already accepted/seen in THIS run, to avoid
                          near-duplicates.
    """
    anchors_block = "\n".join(
        f'  {n}. "{ex}"' for n, ex in enumerate(anchor_examples, start=1)
    )

    if already_generated:
        avoid_block = "\n".join(f'  - "{t}"' for t in already_generated)
        avoid_section = (
            "\nYou have ALREADY produced the following tickets in this "
            "session. Your new ticket must describe a GENUINELY DIFFERENT "
            "compute/server problem and must NOT reuse their phrasing or the "
            "same underlying issue:\n"
            f"{avoid_block}\n"
        )
    else:
        avoid_section = ""

    return f"""You are helping build the HELD-OUT accuracy benchmark for an IT \
support ticket classifier. These tickets must read exactly like an ordinary, \
NON-TECHNICAL employee describing a problem to the help desk in their own \
everyday words -- never like a templated or machine-generated log line.

TARGET CATEGORY: {TARGET_CATEGORY}

{_INFRA_SCOPE_ANCHOR}
Here are real example tickets that already exist for this category. Match \
their TONE, REGISTER, and plain-spoken everyday style (but NOT their specific \
scenarios):
{anchors_block}
{avoid_section}
Write ONE brand-new IT support ticket for the category "{TARGET_CATEGORY}". \
Hard requirements:
  - Plain, conversational, non-technical everyday language -- the voice of a \
regular employee, spoken out loud. First person is natural.
  - No jargon, acronyms, tool names, server names, error codes, or log-style \
phrasing. Describe the SYMPTOM the way a non-technical person NOTICES it, not \
the technical diagnosis. For example, write "the website has been down for \
everyone all morning" rather than "server X is returning 503s"; write "the \
overnight report just never showed up today" rather than "the cron job \
failed".
  - CRITICAL -- STAY IN THE COMPUTE/SERVER SCOPE described above. The ticket \
MUST be about the shared servers/compute platform (systems down or slow for \
everyone, out-of-memory crashes, a server not coming back after a restart, \
scheduled jobs not running, wrong/out-of-sync timestamps, storage overwhelmed, \
capacity not scaling up, part of the platform unreachable). It MUST NOT be \
about physical building/facilities issues (lighting, doors, elevators, \
plumbing, heating/cooling) or a single person's personal device.
  - Describe a scenario that is CLEARLY DISTINCT from every example shown \
above and from any tickets you were told to avoid -- a different situation, \
not a reworded version of one.
  - The problem must unambiguously belong to the "{TARGET_CATEGORY}" category \
as scoped above.
  - 1-3 sentences. Natural, varied sentence structure.

Return ONLY a single JSON object, with no markdown, no code fences, and no \
extra commentary, in exactly this shape:
{{"ticket": "<the plain-English ticket text>"}}"""


def _build_classification_prompt(ticket_text):
    """Build the strict-JSON self-consistency classification prompt.

    Mirrors the precedent's independent-guess step: Gemini is NOT told the
    intended category; it must classify the text cold from the known 7.
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
# CORE GEMINI CALL (shared retry/backoff, mirrors the precedent)
# ---------------------------------------------------------------------------
def _call_gemini_json(client, prompt, temperature):
    """Call Gemini with retry/backoff and parse a single JSON object.

    Returns the parsed dict on success. On failure, RAISES rather than
    sys.exit()/returning None -- so a failed candidate can never silently
    vanish (same deliberate policy as the precedent).
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

    raise RuntimeError(
        f"Gemini API call failed after {MAX_RETRIES} retries. "
        f"Last error: {last_error}"
    )


def generate_one_ticket(client, anchor_examples, already_generated):
    """Generate ONE new Infrastructure ticket text. Raises on failure."""
    prompt = _build_generation_prompt(anchor_examples, already_generated)
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
    (The 'id' is internal bookkeeping; it is stripped from the final clean
    deliverable, which must match the 2-field NOVEL_TICKETS shape.)
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
    print(f"Generating {NEW_COUNT} REPLACEMENT '{TARGET_CATEGORY}' tickets "
          "(correct compute/server scope)")
    print("Each candidate = 1 generation call + 1 self-consistency call,")
    print(f"every call separated by {GEMINI_CALL_DELAY_SEC}s (free-tier safe).")
    print("Interruptible: Ctrl+C and re-run to resume from the checkpoint.")
    print("=" * 75)

    client = build_gemini_client()
    anchor_examples = build_style_anchors()[:STYLE_ANCHORS_COUNT]

    # Load any prior progress.
    clean_records, flagged_records = load_checkpoint()

    def resolved_count():
        # clean + flagged both count as "resolved": we never auto-retry a
        # flagged candidate; that's a human-review decision.
        return len(clean_records) + len(flagged_records)

    # Global counter for stable internal ids across resumes.
    existing_ids = [r["id"] for r in clean_records] + \
                   [r["id"] for r in flagged_records]
    next_id = (max(existing_ids) + 1) if existing_ids else 1

    try:
        already_done = resolved_count()
        if already_done >= NEW_COUNT:
            print(f"\n[{TARGET_CATEGORY}] already has {already_done} resolved "
                  "candidate(s); nothing to generate (from checkpoint).")
        else:
            print(f"\n[{TARGET_CATEGORY}] need {NEW_COUNT - already_done} more "
                  "candidate(s).")

        while resolved_count() < NEW_COUNT:
            # Avoid near-duplicating any text already produced this run
            # (clean + flagged), to keep diversity honest.
            seen_texts = [r["text"] for r in clean_records] + \
                         [r["text"] for r in flagged_records]

            # --- Gemini call #1: generate a candidate ----------------------
            ticket_text = generate_one_ticket(
                client, anchor_examples, seen_texts
            )
            time.sleep(GEMINI_CALL_DELAY_SEC)  # delay after generation call

            # --- Gemini call #2: independent self-consistency check --------
            gemini_guess = classify_one_ticket(client, ticket_text)
            time.sleep(GEMINI_CALL_DELAY_SEC)  # delay after classify call

            passed = (gemini_guess == TARGET_CATEGORY)

            if passed:
                clean_records.append({
                    "id": next_id,
                    "text": ticket_text,
                    "expected": TARGET_CATEGORY,
                })
                print(f"  [PASS] (#{next_id}) intended='{TARGET_CATEGORY}' "
                      f"gemini='{gemini_guess}'  {ticket_text}")
            else:
                flagged_records.append({
                    "id": next_id,
                    "text": ticket_text,
                    "intended_category": TARGET_CATEGORY,
                    "gemini_guess": gemini_guess,
                })
                print(f"  [FLAG] (#{next_id}) intended='{TARGET_CATEGORY}' "
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

    # --- Terminal summary --------------------------------------------------
    total_clean = len(clean_records)
    total_flagged = len(flagged_records)

    print("\n" + "=" * 75)
    print("RUN SUMMARY")
    print("=" * 75)
    print("  {:<20} {:>7} {:>9} {:>9}".format(
        "Category", "Clean", "Flagged", "Target"))
    print("  " + "-" * 48)
    note = "" if total_clean >= NEW_COUNT else "  <-- short of target"
    print("  {:<20} {:>7} {:>9} {:>9}{}".format(
        TARGET_CATEGORY, total_clean, total_flagged, NEW_COUNT, note))

    print("\nDeliverables written:")
    print(f"  clean   -> {CLEAN_OUTPUT_JSON}")
    print(f"  flagged -> {FLAGGED_OUTPUT_JSON}")

    if total_flagged:
        print("\nNOTE: flagged tickets are EXCLUDED from the clean output. "
              "Review them in the flagged file and, if you approve any, add "
              "them manually -- this script never auto-keeps a mismatch.")

    if total_clean < NEW_COUNT:
        print(f"\nNOTE: fewer than {NEW_COUNT} CLEAN tickets because some "
              "candidates were flagged. Re-running will NOT top these up "
              "automatically (a flagged candidate counts as 'resolved'). To "
              "get 5 clean tickets, approve flagged ones or delete the "
              "checkpoint and re-run.")

    print("\nDone. (This script generated + vetted the 5 REPLACEMENT "
          f"'{TARGET_CATEGORY}' tickets only. It did NOT touch "
          "novel_tickets_expanded.json -- swapping the old wrong-scope batch "
          "for these clean ones is a separate manual step, after you review "
          "this output and the flagged file.)")


if __name__ == "__main__":
    main()
