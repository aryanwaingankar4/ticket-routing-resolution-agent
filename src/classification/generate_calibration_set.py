# src/classification/generate_calibration_set.py
# ---------------------------------------------------------------------------
# Generate an out-of-distribution (OOD) calibration set for the cascade
# classifier's confidence threshold.
#
# WHY: The confidence threshold must be calibrated on non-template phrasing,
# not on the in-distribution held-out split (which is 100% accurate in every
# confidence bucket and therefore uninformative). An earlier 35-ticket hand-
# written attempt was too sparse (34/35 collapsed into a single <50% bucket).
# This script paraphrases 175 existing synthetic tickets into plain, non-
# technical language via the Gemini API to get enough density across
# confidence buckets to derive a stable threshold.
#
# HOW TO RUN (from the project root):
#     python src/classification/generate_calibration_set.py
#
# NOTE: This makes 175 SEQUENTIAL Gemini API calls with rate-limit-friendly
# delays, so it will take SEVERAL MINUTES to finish. It is safely
# interruptible: press Ctrl+C at any time (or if it crashes on a transient
# error), then simply re-run the same command -- it checkpoints after every
# successful ticket and will resume where it left off, skipping ids already
# done.
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
    import pandas as pd
except ImportError:
    print("ERROR: pandas is not installed. Install it with:\n"
          "    pip install pandas")
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

# Easy-to-change knob: how many tickets to sample per category.
SAMPLES_PER_CATEGORY = 25

# The 7 canonical categories used across the project. This exact list is also
# handed to Gemini for its self-guess (self-consistency flag only).
KNOWN_CATEGORIES = [
    "Infrastructure",
    "Application",
    "Security",
    "Database",
    "Storage",
    "Network",
    "Access Management",
]

# Required columns in the source CSV.
REQUIRED_COLUMNS = ["id", "title", "description", "category"]

# Gemini model. Do NOT use "gemini-2.5-flash" -- that alias is retired and
# 404s for new API users as of mid-2026.
GEMINI_MODEL = "gemini-flash-lite-latest"

# Retry / rate-limit tuning.
MAX_RETRIES = 3
BACKOFF_SCHEDULE = [5, 15, 30]   # seconds to wait before retry 1, 2, 3
INTER_CALL_DELAY = 1             # small fixed delay between calls (seconds)

# ---------------------------------------------------------------------------
# PATH RESOLUTION (project convention: root = two dirs up from this file)
# ---------------------------------------------------------------------------
# This file lives at <root>/src/classification/generate_calibration_set.py,
# so walking up twice from its own directory yields the project root.
# os.path.abspath + dirname works correctly on Windows too.
_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(os.path.dirname(_THIS_DIR))

INPUT_CSV = os.path.join(PROJECT_ROOT, "data", "synthetic_tickets.csv")
OUTPUT_JSON = os.path.join(
    PROJECT_ROOT, "data", "calibration_tickets_paraphrased.json"
)


# ---------------------------------------------------------------------------
# GEMINI CLIENT SETUP
# ---------------------------------------------------------------------------
def build_gemini_client():
    """Load GEMINI_API_KEY from .env and return an initialized genai client.

    The key is never printed or logged. Follows the error-handling patterns in
    src/rag/suggest_resolution.py.
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
# 1. LOAD + SAMPLE
# ---------------------------------------------------------------------------
def load_and_sample():
    """Load and validate the source CSV, then stratified-sample it.

    Returns a DataFrame of exactly SAMPLES_PER_CATEGORY rows per category
    (fewer if a category has insufficient rows), reproducible via
    random_state=42.
    """
    # --- Validation gates (project convention) -----------------------------
    if not os.path.isfile(INPUT_CSV):
        print(f"ERROR: input file not found: {INPUT_CSV}\n"
              "Make sure you are running this from the project root and that "
              "data/synthetic_tickets.csv exists.")
        sys.exit(1)

    try:
        df = pd.read_csv(INPUT_CSV)
    except Exception as exc:  # noqa: BLE001
        print(f"ERROR: failed to read {INPUT_CSV}.\nDetails: {exc}")
        sys.exit(1)

    missing = [c for c in REQUIRED_COLUMNS if c not in df.columns]
    if missing:
        print(f"ERROR: {INPUT_CSV} is missing required column(s): {missing}\n"
              f"Required columns: {REQUIRED_COLUMNS}")
        sys.exit(1)

    if df[REQUIRED_COLUMNS].isna().any().any():
        na_cols = [c for c in REQUIRED_COLUMNS if df[c].isna().any()]
        print(f"ERROR: {INPUT_CSV} contains NaN values in column(s): {na_cols}\n"
              "Please clean the dataset before generating the calibration set.")
        sys.exit(1)

    # Sanity check: categories in the CSV should be within the known set.
    unknown = sorted(set(df["category"]) - set(KNOWN_CATEGORIES))
    if unknown:
        print("ERROR: the CSV contains categories outside the known 7:\n"
              f"    unexpected: {unknown}\n"
              f"    known:      {KNOWN_CATEGORIES}")
        sys.exit(1)

    # --- Stratified sample: SAMPLES_PER_CATEGORY per category ---------------
    # Cap per-group n at the available count so a small category won't error.
    # NOTE: deliberately avoids groupby(...).apply(...) here -- recent pandas
    # versions (2.2+/3.0) exclude the grouping column from the group passed
    # into the applied function, which silently drops 'category' from the
    # result. Explicit per-category filtering sidesteps that entirely and
    # works the same way regardless of pandas version.
    group_frames = []
    for cat in KNOWN_CATEGORIES:
        subset = df[df["category"] == cat]
        n = min(SAMPLES_PER_CATEGORY, len(subset))
        group_frames.append(subset.sample(n=n, random_state=42))

    sampled = pd.concat(group_frames, ignore_index=True)

    # Warn about any under-filled categories.
    counts = sampled["category"].value_counts()
    for cat in KNOWN_CATEGORIES:
        got = int(counts.get(cat, 0))
        if got < SAMPLES_PER_CATEGORY:
            print(f"WARNING: category '{cat}' only has {got} rows available "
                  f"(requested {SAMPLES_PER_CATEGORY}).")

    expected_total = SAMPLES_PER_CATEGORY * len(KNOWN_CATEGORIES)
    print(f"Sampled {len(sampled)} tickets "
          f"(target was {expected_total} = {SAMPLES_PER_CATEGORY} x "
          f"{len(KNOWN_CATEGORIES)} categories).")

    return sampled


# ---------------------------------------------------------------------------
# 2. PARAPHRASE VIA GEMINI (with retry / backoff)
# ---------------------------------------------------------------------------
def _build_prompt(title, description):
    """Build the strict-JSON paraphrase prompt for a single ticket."""
    category_list = ", ".join(KNOWN_CATEGORIES)
    return f"""You are helping build a test set of IT support tickets phrased \
the way ordinary, non-technical employees describe their problems.

Below is an existing IT support ticket (title + description) written in \
technical language:

TITLE: {title}
DESCRIPTION: {description}

Do TWO things:

1. Rewrite this ticket as a SINGLE plain-English paragraph, in the voice of a \
non-technical employee describing the problem out loud to a help desk. \
Requirements:
   - No technical jargon, acronyms, or system/tool names where they can be \
avoided.
   - Do NOT reuse the exact keywords from the original where you can express \
the same idea in everyday words instead.
   - Vary the sentence structure; make it sound natural and spoken.
   - CRITICAL: preserve the UNDERLYING PROBLEM and MEANING exactly, so the \
original problem is still unambiguous. Change only the SURFACE PHRASING, never \
the actual issue being reported.

2. Independently guess which ONE category this paraphrased problem belongs to, \
choosing strictly from this list (use the exact spelling):
   {category_list}

Return ONLY a single JSON object, with no markdown, no code fences, and no \
extra commentary, in exactly this shape:
{{"paraphrase": "<the plain-English paragraph>", "category": "<one of the \
categories above>"}}"""


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


def paraphrase_one_ticket(client, title, description):
    """Call Gemini to paraphrase one ticket, with retry/backoff.

    Returns (paraphrase_text, gemini_guess) on success, or None on failure
    (so the caller can log the ticket as FAILED and continue).
    """
    prompt = _build_prompt(title, description)

    for attempt in range(MAX_RETRIES + 1):
        try:
            response = client.models.generate_content(
                model=GEMINI_MODEL,
                contents=prompt,
                config={
                    "temperature": 0.9,
                    "response_mime_type": "application/json",
                },
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
                  "(no text). Skipping this ticket.")
            return None

        # --- Parse defensively ---------------------------------------------
        cleaned = _strip_code_fences(raw)
        try:
            parsed = json.loads(cleaned)
        except json.JSONDecodeError:
            print("    FAILED: could not parse Gemini output as JSON. "
                  "Skipping this ticket.")
            return None

        paraphrase = parsed.get("paraphrase")
        guess = parsed.get("category")
        if not paraphrase or not isinstance(paraphrase, str):
            print("    FAILED: parsed JSON is missing a valid 'paraphrase' "
                  "field. Skipping this ticket.")
            return None
        if not guess or not isinstance(guess, str):
            # Missing guess is non-fatal; we just can't flag it meaningfully.
            guess = None
        else:
            guess = guess.strip()

        return paraphrase.strip(), guess

    return None  # Should not be reached, but keeps the contract explicit.


# ---------------------------------------------------------------------------
# 3. RESUMABILITY / CHECKPOINTING
# ---------------------------------------------------------------------------
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

    done_ids = {rec["id"] for rec in records if "id" in rec}
    print(f"Resuming: found {len(records)} previously generated ticket(s); "
          "these ids will be skipped.")
    return records, done_ids


def save_progress(records):
    """Re-save the full accumulated list to disk (atomic-ish via temp file)."""
    os.makedirs(os.path.dirname(OUTPUT_JSON), exist_ok=True)
    tmp_path = OUTPUT_JSON + ".tmp"
    with open(tmp_path, "w", encoding="utf-8") as fh:
        json.dump(records, fh, indent=2, ensure_ascii=False)
    os.replace(tmp_path, OUTPUT_JSON)  # atomic on the same filesystem


# ---------------------------------------------------------------------------
# 7. FINAL VALIDATION GATE
# ---------------------------------------------------------------------------
def final_validation(records):
    """Assert output integrity and print a per-category count table."""
    # Every 'expected' is a known category.
    bad = [r["id"] for r in records if r.get("expected") not in KNOWN_CATEGORIES]
    assert not bad, (
        f"Validation failed: {len(bad)} record(s) have an 'expected' category "
        f"outside the known 7. Offending ids: {bad}"
    )

    # No duplicate ids.
    ids = [r["id"] for r in records]
    assert len(ids) == len(set(ids)), (
        "Validation failed: duplicate 'id' values found in the calibration set."
    )

    # Per-category count table.
    counts = {cat: 0 for cat in KNOWN_CATEGORIES}
    for r in records:
        counts[r["expected"]] += 1

    print("\nFinal per-category counts in the calibration set:")
    print("  {:<20} {:>5}".format("Category", "Count"))
    print("  " + "-" * 26)
    for cat in KNOWN_CATEGORIES:
        print("  {:<20} {:>5}".format(cat, counts[cat]))
    print("  " + "-" * 26)
    print("  {:<20} {:>5}".format("TOTAL", len(records)))


# ---------------------------------------------------------------------------
# MAIN
# ---------------------------------------------------------------------------
def main():
    print("=" * 75)
    print("Generating OOD calibration set (paraphrased tickets via Gemini)")
    print("This makes many sequential API calls and will take several minutes.")
    print("It is safely interruptible: Ctrl+C and re-run to resume.")
    print("=" * 75)

    client = build_gemini_client()
    sampled = load_and_sample()

    records, done_ids = load_existing_progress()

    total_target = len(sampled)
    processed = len(done_ids)  # count already-done toward the running progress

    try:
        for _, row in sampled.iterrows():
            ticket_id = row["id"]
            true_category = row["category"]

            if ticket_id in done_ids:
                continue  # already generated in a prior run

            result = paraphrase_one_ticket(
                client, str(row["title"]), str(row["description"])
            )

            # Fixed courtesy delay between calls regardless of outcome.
            time.sleep(INTER_CALL_DELAY)

            if result is None:
                print(f"[SKIP] Ticket {ticket_id} ({true_category}) "
                      "FAILED and was skipped. Re-run later to retry it.")
                continue

            paraphrase, gemini_guess = result

            flagged = (gemini_guess is not None
                       and gemini_guess != true_category)

            record = {
                "id": ticket_id,
                "text": paraphrase,                 # fed to the classifiers
                "expected": true_category,          # ground truth (never overwritten)
                "original_title": str(row["title"]),
                "original_description": str(row["description"]),
                "flagged": bool(flagged),
                "gemini_guess": gemini_guess if flagged else None,
            }

            records.append(record)
            done_ids.add(ticket_id)
            save_progress(records)  # checkpoint after EVERY success

            processed += 1
            print(f"[{processed}/{total_target}] Paraphrased ticket "
                  f"{ticket_id} ({true_category})"
                  + (f"  [FLAGGED: Gemini guessed '{gemini_guess}']"
                     if flagged else ""))

    except KeyboardInterrupt:
        print("\nInterrupted by user (Ctrl+C). Progress has been saved after "
              "the last completed ticket. Re-run to resume.")
        # Fall through to the summaries below so the partial run is still useful.

    # --- 5. Label-consistency summary --------------------------------------
    flagged_records = [r for r in records if r.get("flagged")]
    total_generated = len(records)
    flagged_rate = (100.0 * len(flagged_records) / total_generated
                    if total_generated else 0.0)

    print("\n" + "=" * 75)
    print("RUN SUMMARY")
    print("=" * 75)
    print(f"Total generated:          {total_generated}")
    print(f"Flagged (label mismatch): {len(flagged_records)} "
          f"({flagged_rate:.1f}%)")

    if flagged_records:
        print("\nFlagged tickets (spot-check these manually -- ground truth is "
              "kept as the ORIGINAL category, Gemini's guess is advisory only):")
        for r in flagged_records:
            print(f"  id={r['id']}  original='{r['expected']}'  "
                  f"gemini_guess='{r['gemini_guess']}'")
    else:
        print("\nNo tickets were flagged for label disagreement.")

    # --- 7. Final validation gate ------------------------------------------
    if total_generated:
        final_validation(records)
        print(f"\nDone. Calibration set written to:\n  {OUTPUT_JSON}")
    else:
        print("\nNo records were generated, so no validation was performed.")


if __name__ == "__main__":
    main()