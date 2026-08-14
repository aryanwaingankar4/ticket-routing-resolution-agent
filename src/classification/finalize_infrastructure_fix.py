# src/classification/finalize_infrastructure_fix.py
#
# Final step in fixing the Infrastructure category-scope mismatch.
#
# Combines:
#   1. data/novel_tickets_expanded.json (currently 41 tickets: the 2 original
#      hand-written Infrastructure tickets + 39 other-category tickets,
#      after the 5 wrong-scope Infrastructure tickets were already stripped
#      by fix_infrastructure_benchmark.py)
#   2. The 2 CLEAN replacement Infrastructure tickets from
#      data/infrastructure_benchmark_replacement.json (passed self-
#      consistency automatically)
#   3. 2 MANUALLY-APPROVED tickets from
#      data/infrastructure_benchmark_replacement_flagged_for_review.json
#      (Gemini's self-consistency check disagreed, but a human reviewer
#      judged them genuinely correct on inspection):
#        - "core customer portal unreachable... status page looks green"
#          (classic infra monitoring-gap scenario; Gemini guessed
#          Application, human judged Infrastructure)
#        - "clocks across internal applications completely out of whack"
#          (directly matches the NTP-drift scenario in generate_dataset.py;
#          Gemini guessed Application, human judged Infrastructure --
#          this one looks like a genuine Gemini miss, not a boundary case)
#   The 3rd flagged ticket ("storage drives totally maxed out") is
#   DELIBERATELY EXCLUDED -- on review this is genuinely Storage-scoped
#   (matches the Storage category's actual quota/capacity scenarios), not a
#   legitimate Infrastructure ticket. Confirmed as a correct Gemini flag,
#   not a false flag.
#
# Result: Infrastructure goes from 2 -> 6 tickets (2 original + 2 auto-clean
# + 2 manually-approved), giving a final 45-ticket benchmark.
#
# Run from the project root:
#     python src/classification/finalize_infrastructure_fix.py

import os
import sys
import json

_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(os.path.dirname(_THIS_DIR))

BENCHMARK_PATH = os.path.join(PROJECT_ROOT, "data", "novel_tickets_expanded.json")
REPLACEMENT_CLEAN_PATH = os.path.join(
    PROJECT_ROOT, "data", "infrastructure_benchmark_replacement.json"
)
REPLACEMENT_FLAGGED_PATH = os.path.join(
    PROJECT_ROOT, "data",
    "infrastructure_benchmark_replacement_flagged_for_review.json"
)

# Distinctive substrings identifying the 2 flagged tickets to MANUALLY
# APPROVE despite Gemini's self-consistency disagreement (human-reviewed,
# judged genuinely correct -- see module docstring for reasoning).
MANUALLY_APPROVED_SUBSTRINGS = [
    "core customer portal has been completely unreachable",
    "clocks across our internal applications are completely out of whack",
]


def main():
    for path in (BENCHMARK_PATH, REPLACEMENT_CLEAN_PATH, REPLACEMENT_FLAGGED_PATH):
        if not os.path.isfile(path):
            print(f"ERROR: required file not found: {path}")
            sys.exit(1)

    with open(BENCHMARK_PATH, "r", encoding="utf-8") as f:
        benchmark = json.load(f)
    with open(REPLACEMENT_CLEAN_PATH, "r", encoding="utf-8") as f:
        replacement_clean = json.load(f)
    with open(REPLACEMENT_FLAGGED_PATH, "r", encoding="utf-8") as f:
        replacement_flagged = json.load(f)

    print(f"Current benchmark:        {len(benchmark)} tickets")
    print(f"Clean replacements:       {len(replacement_clean)} tickets")
    print(f"Flagged replacements:     {len(replacement_flagged)} tickets")

    # --- Select the manually-approved flagged tickets -----------------------
    approved = []
    for entry in replacement_flagged:
        text = entry.get("text", "")
        matched = next(
            (s for s in MANUALLY_APPROVED_SUBSTRINGS if s in text), None
        )
        if matched is not None:
            approved.append({"text": text, "expected": "Infrastructure"})
            print(f"\n  [APPROVED] matched on {matched!r}")
            print(f"    text: {text}")
            print(f"    (Gemini's self-consistency guess was: "
                  f"{entry.get('gemini_guess')!r})")

    if len(approved) != len(MANUALLY_APPROVED_SUBSTRINGS):
        print(
            f"\nWARNING: expected to manually-approve exactly "
            f"{len(MANUALLY_APPROVED_SUBSTRINGS)} flagged ticket(s), but "
            f"matched {len(approved)}. Check the flagged file content "
            "before trusting this merge."
        )

    # --- Combine everything --------------------------------------------------
    final = list(benchmark) + list(replacement_clean) + approved

    # Sanity: no exact-text duplicates.
    seen_texts = set()
    duplicates = []
    for t in final:
        if t["text"] in seen_texts:
            duplicates.append(t["text"])
        seen_texts.add(t["text"])
    if duplicates:
        print(f"\nWARNING: {len(duplicates)} duplicate ticket text(s) found "
              "after merge -- please review manually:")
        for d in duplicates:
            print(f"    - {d}")

    # Per-category counts.
    counts = {}
    for t in final:
        counts[t["expected"]] = counts.get(t["expected"], 0) + 1
    print("\nFinal per-category counts:")
    for cat in sorted(counts):
        print(f"  {cat:<20} {counts[cat]}")
    print(f"  {'TOTAL':<20} {len(final)}")

    with open(BENCHMARK_PATH, "w", encoding="utf-8") as f:
        json.dump(final, f, indent=2, ensure_ascii=False)

    print(f"\nWrote final corrected benchmark ({len(final)} tickets) to:\n  "
          f"{BENCHMARK_PATH}")
    print("\nDone. Infrastructure category-scope mismatch is now resolved: "
          "every Infrastructure ticket in the benchmark is genuinely "
          "compute/server-scoped, matching generate_dataset.py's actual "
          "SCENARIOS definitions.")


if __name__ == "__main__":
    main()