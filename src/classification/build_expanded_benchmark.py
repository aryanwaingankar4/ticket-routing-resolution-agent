# src/classification/build_expanded_benchmark.py
#
# Small, final merge step: combine the original 14 hand-written NOVEL_TICKETS
# (from generalization_test.py, read-only, never modified) with the 32 clean
# tickets from data/expanded_benchmark_new_tickets.json (generated +
# self-consistency-checked by generate_expanded_benchmark.py) into a single
# NOVEL_TICKETS_EXPANDED list of 46 tickets total.
#
# The 3 flagged tickets (data/expanded_benchmark_flagged_for_review.json) are
# intentionally EXCLUDED -- accepted as a legitimate manual-review outcome,
# not auto-approved just to hit a round number.
#
# Output: data/novel_tickets_expanded.json -- a flat list of
# {"text": ..., "expected": ...} dicts, 46 total, matching the exact
# NOVEL_TICKETS shape so it can be loaded the same way elsewhere.
#
# This does NOT modify generalization_test.py. NOVEL_TICKETS_EXPANDED is a
# separate, additive benchmark -- the original 14-ticket NOVEL_TICKETS stays
# untouched as the primary benchmark.

import os
import sys
import json

_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(os.path.dirname(_THIS_DIR))

if _THIS_DIR not in sys.path:
    sys.path.insert(0, _THIS_DIR)

try:
    from generalization_test import NOVEL_TICKETS
except ImportError as exc:
    print(f"ERROR: could not import NOVEL_TICKETS from generalization_test.py.\n"
          f"Details: {exc}")
    sys.exit(1)

NEW_TICKETS_JSON = os.path.join(PROJECT_ROOT, "data", "expanded_benchmark_new_tickets.json")
OUTPUT_JSON = os.path.join(PROJECT_ROOT, "data", "novel_tickets_expanded.json")

KNOWN_CATEGORIES = [
    "Infrastructure", "Application", "Security", "Database",
    "Storage", "Network", "Access Management",
]


def main():
    if not os.path.isfile(NEW_TICKETS_JSON):
        print(f"ERROR: {NEW_TICKETS_JSON} not found. Run "
              "generate_expanded_benchmark.py first.")
        sys.exit(1)

    with open(NEW_TICKETS_JSON, "r", encoding="utf-8") as f:
        new_tickets = json.load(f)

    print(f"Original NOVEL_TICKETS: {len(NOVEL_TICKETS)} tickets")
    print(f"New clean tickets:      {len(new_tickets)} tickets")

    combined = list(NOVEL_TICKETS) + list(new_tickets)

    # Sanity: every ticket must have a valid category label.
    bad = [t for t in combined if t.get("expected") not in KNOWN_CATEGORIES]
    if bad:
        print(f"ERROR: {len(bad)} ticket(s) have an invalid/missing "
              f"'expected' category. First bad entry: {bad[0]}")
        sys.exit(1)

    # Print per-category counts for a quick sanity check.
    counts = {}
    for t in combined:
        counts[t["expected"]] = counts.get(t["expected"], 0) + 1
    print("\nPer-category counts in NOVEL_TICKETS_EXPANDED:")
    for cat in KNOWN_CATEGORIES:
        print(f"  {cat:<20} {counts.get(cat, 0)}")
    print(f"  {'TOTAL':<20} {len(combined)}")

    with open(OUTPUT_JSON, "w", encoding="utf-8") as f:
        json.dump(combined, f, indent=2, ensure_ascii=False)

    print(f"\nWrote {len(combined)} total tickets to:\n  {OUTPUT_JSON}")
    print("\nNOTE: generalization_test.py was NOT modified. NOVEL_TICKETS "
          "(14 tickets) remains the primary benchmark. "
          "novel_tickets_expanded.json is a separate, additive benchmark "
          "you can load wherever you want to re-evaluate against the larger set.")


if __name__ == "__main__":
    main()