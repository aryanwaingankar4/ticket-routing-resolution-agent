# src/classification/generate_ood_calibration_set.py
# ---------------------------------------------------------------------------
# Generate an OUT-OF-DOMAIN (OOD) calibration set for the RAG similarity
# threshold.
#
# WHY: The existing 175-ticket paraphrased calibration set
# (data/calibration_tickets_paraphrased.json) is entirely IN-DOMAIN -- every
# ticket is a legitimate IT-support issue. A threshold sweep over it can't
# observe escalation behavior in the useful range (every ticket "proceeds").
# To calibrate the escalate/proceed gate we need genuine NEGATIVE-class
# signal: tickets that SHOULD escalate because they aren't real, actionable
# IT issues at all.
#
# WHY HAND-WRITTEN SEEDS (not paraphrases of the 9-ticket adversarial set):
# paraphrasing only the existing 9 risks near-duplicate scenarios clustered
# in the same semantic neighborhoods, which overstates how cleanly separated
# the resulting threshold looks. Instead we hand-write 15 broad OOD seeds
# spanning 4 buckets (weather/errands, food, non-IT departmental, vague
# non-actionable) and paraphrase each into 3 independent variants -> ~45
# tickets.
#
# HOW TO RUN (from the project root, Windows PowerShell + venv):
#     python src\classification\generate_ood_calibration_set.py
#
# NOTE: This makes up to 45 SEQUENTIAL Gemini API calls with a PROACTIVE
# inter-call delay sized for the free-tier 15 req/min cap (~3.5 min total).
# It is safely interruptible: press Ctrl+C at any time (or if it crashes on a
# transient error), then simply re-run the same command -- it checkpoints
# after every successful ticket and resumes, skipping ids already done.
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

# --- Project-wide reproducibility convention -------------------------------
random.seed(42)
np.random.seed(42)

# ---------------------------------------------------------------------------
# CONFIGURATION
# ---------------------------------------------------------------------------

# Gemini model. Do NOT use "gemini-2.5-flash" -- that alias is retired and
# 404s for new API users as of mid-2026. (Same constant as the in-domain
# generator.)
GEMINI_MODEL = "gemini-flash-lite-latest"

# Variants to generate per seed scenario.
VARIANTS_PER_SEED = 3

# Retry / rate-limit tuning.
MAX_RETRIES = 3
BACKOFF_SCHEDULE = [5, 15, 30]   # seconds to wait before retry 1, 2, 3

# PROACTIVE inter-call delay. The free-tier cap is ~15 req/min -> 60/15 = 4.0s
# floor; we use 4.2s for margin. This differs deliberately from the in-domain
# generator's reactive INTER_CALL_DELAY=1s (which leans on 429 backoff): with
# only 45 calls we'd rather never trip the limit in the first place. The
# [5,15,30]s backoff below remains as a second layer of defense.
INTER_CALL_DELAY = 4.2

# ---------------------------------------------------------------------------
# PATH RESOLUTION (project convention: root = two dirs up from this file)
# ---------------------------------------------------------------------------
# This file lives at <root>/src/classification/generate_ood_calibration_set.py,
# so walking up twice from its own directory yields the project root.
_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(os.path.dirname(_THIS_DIR))

OUTPUT_JSON = os.path.join(
    PROJECT_ROOT, "data", "ood_calibration_tickets.json"
)

# ---------------------------------------------------------------------------
# SEED SCENARIOS
# ---------------------------------------------------------------------------
# 15 hand-written OOD seeds across 4 buckets. Each entry is
# (seed_key, bucket, one_line_description). The one-line descriptions are
# derived directly from the bucket descriptions in the task spec -- no new
# scenarios invented.
SEEDS = [
    # Bucket A -- Weather & personal errands (4)
    ("A1", "A",
     "An employee wants to know whether it will rain this coming weekend "
     "because they're planning a trip to the hills and are trying to decide "
     "whether to pack an umbrella."),
    ("A2", "A",
     "An employee is (mistakenly) asking the help desk, as if asking a "
     "coworker, where a good place to grab lunch near the office is."),
    ("A3", "A",
     "An employee wants to know how to book a dentist appointment through "
     "the company's insurance plan."),
    ("A4", "A",
     "An employee wants to know the best route to take to avoid traffic on "
     "their commute home."),

    # Bucket B -- Food, non-IT (3)
    ("B1", "B",
     "An employee is complaining that the office cafeteria has once again "
     "run out of the vegetarian option today."),
    ("B2", "B",
     "An employee is asking whether the snack vending machine on their floor "
     "can be restocked with a specific favorite snack of theirs."),
    ("B3", "B",
     "An employee is asking about catering options for an upcoming team "
     "lunch."),

    # Bucket C -- Non-IT departmental: HR / Finance / Facilities (4)
    ("C1", "C",
     "An employee is asking HR about the details of the maternity/paternity "
     "leave policy."),
    ("C2", "C",
     "An employee is asking Finance about the reimbursement policy for travel "
     "expenses -- a policy question, not a broken system."),
    ("C3", "C",
     "An employee is asking Facilities to fix a broken office chair or to "
     "adjust the temperature in the room."),
    ("C4", "C",
     "An employee is asking HR when the next performance review cycle "
     "starts."),

    # Bucket D -- Vague, non-actionable "it's broken" messages (4)
    ("D1", "D",
     "An employee sends a message that simply says something is broken and "
     "asks for help, with no further detail whatsoever -- no system named, "
     "no symptom described."),
    ("D2", "D",
     "An employee says their 'thing' isn't working again, the same as last "
     "time, with no specifics and no system named."),
    ("D3", "D",
     "An employee sends a one-line generic complaint with no actual request, "
     "just frustration that nothing works around here."),
    ("D4", "D",
     "An employee asks someone to just fix everything because they don't "
     "have time to explain, naming no system and describing no symptom."),
]

# Expected per-bucket counts of SEEDS (for the final validation table).
EXPECTED_BUCKET_SEEDS = {"A": 4, "B": 3, "C": 4, "D": 4}


# ---------------------------------------------------------------------------
# GEMINI CLIENT SETUP
# ---------------------------------------------------------------------------
def build_gemini_client():
    """Load GEMINI_API_KEY from .env and return an initialized genai client.

    The key is never printed or logged. Follows the error-handling patterns in
    src/rag/suggest_resolution.py and generate_calibration_set.py.
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
# RESPONSE-SCHEMA CAPABILITY DETECTION (reliability improvement)
# ---------------------------------------------------------------------------
def detect_response_schema_support():
    """Best-effort check for whether the installed google-genai SDK supports
    passing a `response_schema` in the generate_content config.

    Returns (supported: bool, schema_config_fragment: dict). If supported, the
    fragment is a config dict we can merge into the call; if not, an empty
    dict. Pinning a schema is strictly more reliable than mime-type +
    fence-stripping alone at temperature=0.9.

    We probe the SDK's `types` module for the pieces needed to declare a
    single-string-property object schema. If anything is missing we fall back
    cleanly.
    """
    try:
        from google.genai import types as genai_types
    except Exception:
        return False, {}

    Schema = getattr(genai_types, "Schema", None)
    Type = getattr(genai_types, "Type", None)
    if Schema is None or Type is None:
        return False, {}

    # Type may be an enum with STRING/OBJECT members, or accept plain strings.
    string_type = getattr(Type, "STRING", "STRING")
    object_type = getattr(Type, "OBJECT", "OBJECT")

    try:
        schema = Schema(
            type=object_type,
            properties={"paraphrase": Schema(type=string_type)},
            required=["paraphrase"],
        )
    except Exception:
        return False, {}

    return True, {"response_schema": schema}


# ---------------------------------------------------------------------------
# PROMPT
# ---------------------------------------------------------------------------
def _build_prompt(seed_description, variant_idx, variant_total):
    """Build the strict-JSON paraphrase prompt for one variant of one seed.

    Voice/style conventions mirror generate_calibration_set.py: plain English,
    non-technical employee voice, no reused keywords, natural spoken phrasing.
    The variant instruction asks Gemini to treat this as an INDEPENDENT report
    of the same underlying situation, to push the 3 variants apart. We never
    show Gemini the other variants.
    """
    return f"""You are helping build a test set of messages that ordinary, \
non-technical employees mistakenly send to a company IT help desk, even though \
these messages are NOT real, actionable IT-support problems.

Here is the situation to write about:
{seed_description}

This is variant {variant_idx} of {variant_total}. Write it as if a DIFFERENT \
person independently reported the SAME underlying situation -- so vary the \
phrasing, the length, and the framing from how someone else might have said \
it. Do not produce a near-duplicate of a generic template.

Requirements:
   - Write ONE plain-English paragraph, in the natural spoken voice of an \
employee typing a quick message to the help desk.
   - No technical jargon, acronyms, or system/tool names.
   - Do NOT reuse the exact wording of the situation description above where \
you can express the same idea in everyday words instead.
   - Vary the sentence structure; make it sound natural and spoken.
   - CRITICAL: preserve the UNDERLYING SITUATION and MEANING exactly. Change \
only the SURFACE PHRASING, never the actual thing being reported.

Return ONLY a single JSON object, with no markdown, no code fences, and no \
extra commentary, in exactly this shape:
{{"paraphrase": "<the plain-English paragraph>"}}"""


def _strip_code_fences(text):
    """Remove surrounding ```json ... ``` / ``` ... ``` fences if present."""
    t = text.strip()
    if t.startswith("```"):
        # Drop the first fence line (``` or ```json).
        first_newline = t.find("\n")
        if first_newline != -1:
            t = t[first_newline + 1:]
        # Drop a trailing closing fence.
        if t.rstrip().endswith("```"):
            t = t.rstrip()[:-3]
    return t.strip()


def paraphrase_one_variant(client, seed_description, variant_idx,
                           variant_total, schema_fragment):
    """Call Gemini to paraphrase one variant, with retry/backoff.

    Returns the paraphrase text on success, or None on failure (so the caller
    can log it and continue -- a later re-run fills the gap).
    """
    prompt = _build_prompt(seed_description, variant_idx, variant_total)

    config = {
        "temperature": 0.9,
        "response_mime_type": "application/json",
    }
    # Merge the response_schema fragment if the SDK supports it.
    config.update(schema_fragment)

    for attempt in range(MAX_RETRIES + 1):
        try:
            response = client.models.generate_content(
                model=GEMINI_MODEL,
                contents=prompt,
                config=config,
            )
        except Exception as exc:  # noqa: BLE001
            if _is_rate_limit_error(exc) and attempt < MAX_RETRIES:
                wait = BACKOFF_SCHEDULE[min(attempt, len(BACKOFF_SCHEDULE) - 1)]
                print(f"    Rate limit / quota hit (attempt {attempt + 1}"
                      f"/{MAX_RETRIES}). Waiting {wait}s before retry...")
                time.sleep(wait)
                continue
            # Non-retryable, or retries exhausted -> clean actionable message.
            if _is_rate_limit_error(exc):
                print("    FAILED: rate limit / quota still exceeded after "
                      f"{MAX_RETRIES} retries. Consider slowing down or "
                      "checking your API quota.")
            elif "api key" in str(exc).lower() or "permission" in str(exc).lower():
                print(f"    FAILED: API key / permission problem: {exc}")
            else:
                print(f"    FAILED: Gemini API error: {exc}")
            return None

        # --- Extract text safely (handle empty / blocked responses) --------
        raw = getattr(response, "text", None)
        if not raw:
            print("    FAILED: Gemini returned an empty or blocked response "
                  "(no text). Skipping this variant.")
            return None

        # --- Parse defensively ---------------------------------------------
        cleaned = _strip_code_fences(raw)
        try:
            parsed = json.loads(cleaned)
        except json.JSONDecodeError:
            print("    FAILED: could not parse Gemini output as JSON. "
                  "Skipping this variant.")
            return None

        paraphrase = parsed.get("paraphrase")
        if not paraphrase or not isinstance(paraphrase, str) \
                or not paraphrase.strip():
            print("    FAILED: parsed JSON is missing a valid 'paraphrase' "
                  "field. Skipping this variant.")
            return None

        return paraphrase.strip()

    return None  # Should not be reached, but keeps the contract explicit.


# ---------------------------------------------------------------------------
# RESUMABILITY / CHECKPOINTING
# ---------------------------------------------------------------------------
def _variant_id(seed_key, variant_idx):
    """Deterministic string id for a (seed, variant) pair, prefixed 'ood_'.

    Format: ood_<seedkey>_v<variant>, e.g. 'ood_A1_v1'. Prefixed 'ood_' so it
    can never collide with the integer ids in
    calibration_tickets_paraphrased.json or the 4,000-ticket metadata.
    """
    return f"ood_{seed_key}_v{variant_idx}"


def load_existing_progress():
    """Load any prior output so we can resume, returning (records, done_ids)."""
    if not os.path.isfile(OUTPUT_JSON):
        return [], set()
    try:
        with open(OUTPUT_JSON, "r", encoding="utf-8") as fh:
            records = json.load(fh)
    except (json.JSONDecodeError, OSError) as exc:
        print(f"WARNING: could not read existing progress file {OUTPUT_JSON} "
              f"({exc}). Starting fresh (existing file will be overwritten on "
              "the first successful save).")
        return [], set()

    if not isinstance(records, list):
        print(f"WARNING: existing progress file {OUTPUT_JSON} is not a JSON "
              "list. Starting fresh.")
        return [], set()

    done_ids = {rec["id"] for rec in records if isinstance(rec, dict)
                and "id" in rec}
    print(f"Resuming: found {len(records)} previously generated OOD ticket(s); "
          "these ids will be skipped.")
    return records, done_ids


def save_progress(records):
    """Re-save the full accumulated list to disk (atomic via temp file)."""
    os.makedirs(os.path.dirname(OUTPUT_JSON), exist_ok=True)
    tmp_path = OUTPUT_JSON + ".tmp"
    with open(tmp_path, "w", encoding="utf-8") as fh:
        json.dump(records, fh, indent=2, ensure_ascii=False)
    os.replace(tmp_path, OUTPUT_JSON)  # atomic on the same filesystem


# ---------------------------------------------------------------------------
# NEAR-DUPLICATE HEURISTIC (advisory only)
# ---------------------------------------------------------------------------
def _normalize_for_compare(text):
    """Lowercase + collapse whitespace for a crude near-duplicate check."""
    return " ".join(text.lower().split())


def warn_on_near_identical_variants(records):
    """Advisory check: if the 3 variants of a seed look near-identical, say so
    explicitly rather than silently accepting near-duplicates. Uses a crude
    token-overlap (Jaccard) heuristic -- this is a hint for manual spot-check,
    not a hard gate."""
    by_seed = {}
    for rec in records:
        # id format: ood_<seedkey>_v<variant>
        parts = rec["id"].split("_")
        if len(parts) >= 3:
            seed_key = parts[1]
            by_seed.setdefault(seed_key, []).append(rec["text"])

    flagged_any = False
    for seed_key, texts in sorted(by_seed.items()):
        norm = [set(_normalize_for_compare(t).split()) for t in texts]
        for i in range(len(norm)):
            for j in range(i + 1, len(norm)):
                a, b = norm[i], norm[j]
                if not a or not b:
                    continue
                jac = len(a & b) / len(a | b)
                if jac >= 0.8:
                    flagged_any = True
                    print(f"WARNING: seed '{seed_key}' variants "
                          f"{i + 1} and {j + 1} look near-identical "
                          f"(token Jaccard {jac:.2f}). Consider re-running to "
                          "regenerate more varied phrasing, or spot-check them "
                          "manually.")
    if not flagged_any:
        print("No near-identical variant pairs detected "
              "(token-overlap heuristic).")


# ---------------------------------------------------------------------------
# VALIDATION GATE
# ---------------------------------------------------------------------------
def final_validation(records):
    """Assert output integrity and print a per-bucket count table.

    Returns True if the set is complete (all 15 seeds x 3 variants = 45),
    False if short (so main() can tell the user to re-run).
    """
    # No duplicate ids.
    ids = [r["id"] for r in records]
    assert len(ids) == len(set(ids)), (
        "Validation failed: duplicate 'id' values found in the OOD set."
    )

    # No empty/whitespace text.
    empty = [r["id"] for r in records if not str(r.get("text", "")).strip()]
    assert not empty, (
        f"Validation failed: {len(empty)} record(s) have empty/whitespace "
        f"text. Offending ids: {empty}"
    )

    # Per-bucket counts (derive bucket from the seed key embedded in the id).
    bucket_counts = {"A": 0, "B": 0, "C": 0, "D": 0}
    for r in records:
        parts = r["id"].split("_")
        if len(parts) >= 2 and parts[1]:
            bucket_letter = parts[1][0]
            if bucket_letter in bucket_counts:
                bucket_counts[bucket_letter] += 1

    expected_per_bucket = {
        b: EXPECTED_BUCKET_SEEDS[b] * VARIANTS_PER_SEED
        for b in EXPECTED_BUCKET_SEEDS
    }
    expected_total = sum(expected_per_bucket.values())

    print("\nFinal per-bucket counts in the OOD calibration set:")
    print("  {:<8} {:>7} {:>10}".format("Bucket", "Count", "Expected"))
    print("  " + "-" * 27)
    for b in ("A", "B", "C", "D"):
        print("  {:<8} {:>7} {:>10}".format(
            b, bucket_counts[b], expected_per_bucket[b]))
    print("  " + "-" * 27)
    print("  {:<8} {:>7} {:>10}".format(
        "TOTAL", len(records), expected_total))

    complete = (len(records) == expected_total)
    return complete


# ---------------------------------------------------------------------------
# MAIN
# ---------------------------------------------------------------------------
def main():
    print("=" * 75)
    print("Generating OUT-OF-DOMAIN (OOD) calibration set via Gemini")
    print(f"{len(SEEDS)} seeds x {VARIANTS_PER_SEED} variants = "
          f"{len(SEEDS) * VARIANTS_PER_SEED} target tickets.")
    print("Proactive rate limiting (~4.2s/call); expect a few minutes.")
    print("It is safely interruptible: Ctrl+C and re-run to resume.")
    print("=" * 75)

    client = build_gemini_client()

    schema_supported, schema_fragment = detect_response_schema_support()
    if schema_supported:
        print("NOTE: installed google-genai SDK supports response_schema -- "
              "pinning a JSON schema for stricter, more reliable output.")
    else:
        print("NOTE: installed google-genai SDK does not expose a usable "
              "response_schema -- falling back to response_mime_type + "
              "fence-stripping only.")

    records, done_ids = load_existing_progress()

    total_target = len(SEEDS) * VARIANTS_PER_SEED
    processed = len(done_ids)

    try:
        for seed_key, _bucket, seed_description in SEEDS:
            for variant_idx in range(1, VARIANTS_PER_SEED + 1):
                ticket_id = _variant_id(seed_key, variant_idx)

                if ticket_id in done_ids:
                    continue  # already generated in a prior run

                paraphrase = paraphrase_one_variant(
                    client, seed_description, variant_idx,
                    VARIANTS_PER_SEED, schema_fragment
                )

                # Proactive fixed delay between calls regardless of outcome.
                time.sleep(INTER_CALL_DELAY)

                if paraphrase is None:
                    print(f"[SKIP] {ticket_id} FAILED and was skipped. "
                          "Re-run later to retry it.")
                    continue

                record = {"id": ticket_id, "text": paraphrase}
                records.append(record)
                done_ids.add(ticket_id)
                save_progress(records)  # checkpoint after EVERY success

                processed += 1
                print(f"[{processed}/{total_target}] Generated {ticket_id}")

    except KeyboardInterrupt:
        print("\nInterrupted by user (Ctrl+C). Progress has been saved after "
              "the last completed ticket. Re-run to resume.")
        # Fall through to the summary so a partial run is still useful.

    # --- Summary + advisory near-duplicate check ---------------------------
    print("\n" + "=" * 75)
    print("RUN SUMMARY")
    print("=" * 75)
    print(f"Total generated: {len(records)} / {total_target}")

    if records:
        warn_on_near_identical_variants(records)
        complete = final_validation(records)
        if complete:
            print(f"\nDone. OOD calibration set written to:\n  {OUTPUT_JSON}")
        else:
            short_by = total_target - len(records)
            print(f"\nINCOMPLETE: the set is short by {short_by} ticket(s) "
                  "due to failed/skipped calls.\n"
                  "Re-run the SAME command to fill the gaps -- already-"
                  "generated ids will be skipped and only the missing ones "
                  "will be retried:\n"
                  "    python src\\classification\\generate_ood_calibration_set.py")
    else:
        print("No records were generated, so no validation was performed.")


if __name__ == "__main__":
    main()
