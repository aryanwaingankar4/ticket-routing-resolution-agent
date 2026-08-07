# src/experiments/retry_failed_resolutions.py
"""
Retry harness for the 15 Day-7 tickets whose Gemini resolution-suggestion
calls failed permanently (resolution_status == "gemini_call_failed") after the
original run's retry delay was too short.

What this script does
----------------------
  1. Loads the FAISS index / metadata / embedding model / Gemini client ONCE
     at startup, reusing the functions already implemented in
     src/rag/suggest_resolution.py (no reimplementation of retrieval or
     prompt-building here).
  2. Scans all 7 data/category_stores/{Category}.csv files for rows whose
     resolution_status is exactly "gemini_call_failed".
  3. For each such ticket: retrieves similar past tickets, builds the same
     grounded prompt, and calls Gemini -- but through a THIN WRAPPER that
     raises a normal Exception on failure instead of sys.exit()ing, so one
     still-failing ticket does not abort the retry of the other 14.
  4. Respects the project's 4.5s inter-call delay (free tier: 15 req/min).
  5. On success: sets resolution_status = "auto_resolved" and
     resolution_text = <new suggestion>, then writes that category's CSV back
     ATOMICALLY (temp file + os.replace), per-category, so partial progress
     survives a later crash.
  6. On failure: leaves the row untouched (still "gemini_call_failed"), logs
     it, and continues.
  7. Prints per-ticket progress and a final per-category summary.

Run from the project root:
    python src/experiments/retry_failed_resolutions.py
"""

import os
import sys
import csv
import time
import tempfile

# ---------------------------------------------------------------------------
# Make src/ importable so we can reuse suggest_resolution.py regardless of the
# invocation directory. This file lives at <root>/src/experiments/, so the
# project root is two directories up, and <root>/src is one directory up.
# ---------------------------------------------------------------------------
_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(os.path.dirname(_THIS_DIR))
SRC_DIR = os.path.join(PROJECT_ROOT, "src")
if SRC_DIR not in sys.path:
    sys.path.insert(0, SRC_DIR)

# Reuse the existing pipeline. We import the module and reference its members
# so that constants (MODEL_NAME, TOP_K, ...) always match the source of truth.
from rag import suggest_resolution as sr  # noqa: E402


# ---------------------------------------------------------------------------
# Configuration specific to this retry run.
# ---------------------------------------------------------------------------
CATEGORY_STORES_DIR = os.path.join(PROJECT_ROOT, "data", "category_stores")

# The 7 category files, exactly as they exist on disk. "Access Management"
# contains a space; os.path.join handles that fine.
CATEGORY_FILES = [
    "Infrastructure",
    "Application",
    "Security",
    "Database",
    "Storage",
    "Network",
    "Access Management",
]

FAILED_STATUS = "gemini_call_failed"   # exact string we retry
SUCCESS_STATUS = "auto_resolved"       # exact string we set on success

# Project rate-limit convention: 4.5s between every Gemini call.
GEMINI_CALL_DELAY_SEC = 4.5

# Columns we depend on (confirmed present in every category CSV).
COL_ID = "batch_ticket_id"
COL_TITLE = "title"
COL_DESCRIPTION = "description"
COL_STATUS = "resolution_status"
COL_RESOLUTION = "resolution_text"


# ===========================================================================
# THIN, NON-EXITING GEMINI WRAPPER
# ===========================================================================
def call_gemini_no_exit(client, prompt):
    """
    Call Gemini with the SAME request shape suggest_resolution.call_gemini()
    uses, but WITHOUT sys.exit() on failure.

    The original call_gemini() classifies errors and then calls _fail(), which
    prints and sys.exit(1)s -- fine for a single-shot demo, fatal for a batch
    retry loop (it would kill all remaining tickets). Here we make the exact
    same client.models.generate_content(model=MODEL_NAME, contents=prompt)
    call, but let any exception propagate as a normal Exception for the caller
    to catch.

    We also reproduce the original's empty/blocked-response handling, but
    treat an empty/blocked response as a FAILURE (raise) rather than returning
    a placeholder string -- for a retry we only want to mark a ticket
    "auto_resolved" when we actually got usable text.

    Returns: the generated suggestion text (non-empty, stripped).
    Raises:  Exception on any API error, or RuntimeError on empty/blocked
             output.
    """
    response = client.models.generate_content(
        model=sr.MODEL_NAME,
        contents=prompt,
    )

    text = getattr(response, "text", None)
    if text is None or not str(text).strip():
        # Mirror the diagnostic detail the original surfaces, but as an error.
        block_info = ""
        feedback = getattr(response, "prompt_feedback", None)
        if feedback is not None:
            block_info = f" prompt_feedback={feedback}"
        finish = ""
        candidates = getattr(response, "candidates", None)
        if candidates:
            finish_reason = getattr(candidates[0], "finish_reason", None)
            if finish_reason is not None:
                finish = f" finish_reason={finish_reason}"
        raise RuntimeError(
            f"Gemini returned an empty or blocked response.{block_info}{finish}"
        )

    return str(text).strip()


# ===========================================================================
# CSV I/O (atomic write, per-category)
# ===========================================================================
def read_category_csv(csv_path):
    """
    Read a category CSV into (fieldnames, rows) where rows is a list of dicts.
    Returns (None, None) if the file does not exist.
    """
    if not os.path.isfile(csv_path):
        return None, None

    with open(csv_path, "r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        fieldnames = reader.fieldnames
        rows = list(reader)
    return fieldnames, rows


def write_category_csv_atomic(csv_path, fieldnames, rows):
    """
    Write rows back to csv_path ATOMICALLY: write to a temp file in the SAME
    directory (so os.replace is a same-filesystem rename), flush+fsync, then
    os.replace() over the original. A crash mid-write leaves the original
    intact.
    """
    target_dir = os.path.dirname(os.path.abspath(csv_path))

    fd, tmp_path = tempfile.mkstemp(
        prefix=".retry_tmp_", suffix=".csv", dir=target_dir
    )
    try:
        # Wrap the low-level fd in a text stream with the same conventions the
        # rest of the project uses (utf-8, newline="" so csv controls EOLs).
        with os.fdopen(fd, "w", encoding="utf-8", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(rows)
            f.flush()
            os.fsync(f.fileno())
        # Atomic swap. On Windows, os.replace() overwrites an existing target.
        os.replace(tmp_path, csv_path)
    except Exception:
        # Best-effort cleanup of the temp file on any failure before replace.
        try:
            if os.path.isfile(tmp_path):
                os.remove(tmp_path)
        except OSError:
            pass
        raise


# ===========================================================================
# CORE RETRY LOGIC
# ===========================================================================
def retry_one_ticket(row, model, index, metadata, faiss, client):
    """
    Retry the Gemini suggestion for a single failed ticket (a row dict).

    Reuses retrieve_similar_tickets() and build_llm_prompt() from
    suggest_resolution.py verbatim, then calls the non-exiting Gemini wrapper.

    Returns the new suggestion text on success.
    Raises Exception on failure (caller catches and continues).
    """
    title = row.get(COL_TITLE, "") or ""
    description = row.get(COL_DESCRIPTION, "") or ""

    # SAME query-text convention as suggest_resolution.py.
    query_text = f"{title} {description}"

    retrieved = sr.retrieve_similar_tickets(
        query_text, model, index, metadata, faiss, top_k=sr.TOP_K
    )
    if not retrieved:
        # No candidates at all -> cannot ground a suggestion. Treat as failure
        # so the row stays "gemini_call_failed" for a human to look at.
        raise RuntimeError("No candidates retrieved from the index.")

    prompt = sr.build_llm_prompt(title, description, retrieved)
    suggestion = call_gemini_no_exit(client, prompt)
    return suggestion


def process_category(category, model, index, metadata, faiss, client,
                     first_call_state):
    """
    Process one category CSV end-to-end:
      - find all rows with resolution_status == FAILED_STATUS,
      - retry each (with the 4.5s delay honored between every Gemini call),
      - update successful rows in memory,
      - write the CSV back atomically ONCE, after this category's retries.

    first_call_state is a single-element list used as a mutable flag so the
    inter-call delay is applied BETWEEN calls (never before the very first
    call of the whole run).

    Returns (succeeded, failed) counts for this category.
    """
    csv_path = os.path.join(CATEGORY_STORES_DIR, f"{category}.csv")

    fieldnames, rows = read_category_csv(csv_path)
    if fieldnames is None:
        print(f"\n[{category}] SKIP -- file not found: {csv_path}")
        return 0, 0

    # Identify the failed rows (by index so we mutate the exact row objects).
    failed_indices = [
        i for i, r in enumerate(rows)
        if (r.get(COL_STATUS, "") or "").strip() == FAILED_STATUS
    ]

    print(f"\n{'=' * 72}")
    print(f"[{category}] {len(failed_indices)} failed ticket(s) to retry "
          f"(file: {os.path.basename(csv_path)})")
    print(f"{'=' * 72}")

    if not failed_indices:
        print(f"[{category}] Nothing to do.")
        return 0, 0

    succeeded = 0
    failed = 0

    for n, row_idx in enumerate(failed_indices, start=1):
        row = rows[row_idx]
        ticket_id = row.get(COL_ID, "<unknown-id>")
        title = row.get(COL_TITLE, "") or ""

        print(f"\n  [{category} {n}/{len(failed_indices)}] "
              f"ticket_id={ticket_id!r}")
        print(f"    title: {title[:70]}")

        # Honor the 4.5s delay BETWEEN Gemini calls (skip before the very
        # first call of the entire run).
        if first_call_state[0]:
            first_call_state[0] = False
        else:
            time.sleep(GEMINI_CALL_DELAY_SEC)

        try:
            suggestion = retry_one_ticket(
                row, model, index, metadata, faiss, client
            )
        except Exception as exc:  # noqa: BLE001 - one ticket must not kill batch
            failed += 1
            print(f"    -> STILL FAILED ({type(exc).__name__}): {exc}")
            print(f"    -> leaving resolution_status = {FAILED_STATUS!r} "
                  f"(unchanged)")
            continue

        # Success: update this row in memory.
        row[COL_STATUS] = SUCCESS_STATUS
        row[COL_RESOLUTION] = suggestion
        succeeded += 1
        print(f"    -> SUCCESS: resolution_status -> {SUCCESS_STATUS!r}")
        print(f"    -> suggestion ({len(suggestion)} chars): "
              f"{suggestion[:100].replace(chr(10), ' ')}...")

    # ---- Persist this category's progress atomically, once ----------------
    if succeeded > 0:
        try:
            write_category_csv_atomic(csv_path, fieldnames, rows)
            print(f"\n  [{category}] wrote {succeeded} update(s) atomically -> "
                  f"{os.path.basename(csv_path)}")
        except Exception as exc:  # noqa: BLE001
            # If the write itself fails, the on-disk file is untouched (atomic
            # swap never happened). Report loudly but do NOT crash the run.
            print(f"\n  [{category}] ERROR writing CSV "
                  f"({type(exc).__name__}: {exc}). On-disk file left "
                  f"UNCHANGED; these {succeeded} success(es) were NOT saved.")
    else:
        print(f"\n  [{category}] no successful retries -> CSV left unchanged.")

    return succeeded, failed


# ===========================================================================
# MAIN
# ===========================================================================
def main():
    print("=" * 72)
    print("RETRY FAILED RESOLUTIONS  (Day-7 'gemini_call_failed' tickets)")
    print("=" * 72)
    print(f"  category stores dir : {CATEGORY_STORES_DIR}")
    print(f"  retry status match  : {FAILED_STATUS!r}")
    print(f"  success status set  : {SUCCESS_STATUS!r}")
    print(f"  LLM model           : {sr.MODEL_NAME}")
    print(f"  embed model         : {sr.EMBED_MODEL_NAME}")
    print(f"  top_k               : {sr.TOP_K}")
    print(f"  inter-call delay    : {GEMINI_CALL_DELAY_SEC}s")

    if not os.path.isdir(CATEGORY_STORES_DIR):
        print(f"\n[ERROR] Category stores directory not found:\n"
              f"    {CATEGORY_STORES_DIR}", file=sys.stderr)
        sys.exit(1)

    # ---- Load the heavy artifacts ONCE, reusing suggest_resolution.py -----
    faiss = sr.load_faiss()
    index, metadata = sr.load_index_and_metadata(faiss)
    model = sr.load_embedding_model()
    client = sr.setup_gemini_client()

    # Mutable single-element flag: True until the first Gemini call is made,
    # so we don't sleep before the very first call of the whole run.
    first_call_state = [True]

    # Per-category tallies for the final summary table.
    per_category = {}   # category -> {"succeeded": int, "failed": int}
    total_succeeded = 0
    total_failed = 0

    for category in CATEGORY_FILES:
        succeeded, failed = process_category(
            category, model, index, metadata, faiss, client, first_call_state
        )
        per_category[category] = {"succeeded": succeeded, "failed": failed}
        total_succeeded += succeeded
        total_failed += failed

    # ---- Final summary ----------------------------------------------------
    total_attempted = total_succeeded + total_failed

    print("\n" + "=" * 72)
    print("RETRY SUMMARY")
    print("=" * 72)
    header = f"  {'Category':<20} {'attempted':>10} {'succeeded':>10} {'still_failed':>13}"
    print(header)
    print("  " + "-" * (len(header) - 2))
    for category in CATEGORY_FILES:
        s = per_category[category]["succeeded"]
        f = per_category[category]["failed"]
        a = s + f
        # Only print rows that actually had failed tickets, but keep zero-rows
        # visible too so the table maps 1:1 to the 7 category files.
        print(f"  {category:<20} {a:>10} {s:>10} {f:>13}")
    print("  " + "-" * (len(header) - 2))
    print(f"  {'TOTAL':<20} {total_attempted:>10} "
          f"{total_succeeded:>10} {total_failed:>13}")
    print("=" * 72)

    print(f"\nRetried {total_attempted} ticket(s): "
          f"{total_succeeded} now auto_resolved, "
          f"{total_failed} still gemini_call_failed.")
    if total_failed > 0:
        print("Re-run this script to retry the remaining failures (already "
              "auto_resolved tickets are skipped automatically).")
    print("DONE.")


if __name__ == "__main__":
    main()
