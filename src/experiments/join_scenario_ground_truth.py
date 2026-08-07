# src/experiments/join_scenario_ground_truth.py
#
# Small, targeted join: attach the ground-truth `scenario_id` (from
# data/synthetic_tickets.csv) onto the Day 7 batch-intake output in
# data/category_stores/{Category}.csv, via original_id -> id.
#
# This gives us EXACT, construction-based ground truth for "do these two
# tickets share the same underlying resolution scenario" -- no LLM judgment,
# no manual labeling. Two rows with the same (category, scenario_id) are, by
# definition, the same underlying fix (just different entity names).
#
# Output: a single combined CSV across all 7 categories, at
# data/category_stores_with_scenario_id.csv, with columns:
#   batch_ticket_id, original_id, category, scenario_id, resolution_status,
#   resolution_text
# (only auto_resolved rows are kept, since those are the ones we clustered
# in the exploratory clustering script -- escalated/failed rows have no
# settled resolution text to cluster on)
#
# This is a diagnostic/prep step, not a final feature. Read-only against
# both source file sets; writes ONE new combined output file.

import os
import sys
import pandas as pd

CATEGORIES = [
    "Infrastructure",
    "Application",
    "Security",
    "Database",
    "Storage",
    "Network",
    "Access Management",
]

_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(_THIS_DIR, os.pardir, os.pardir))
CATEGORY_STORES_DIR = os.path.join(PROJECT_ROOT, "data", "category_stores")
SYNTHETIC_CSV = os.path.join(PROJECT_ROOT, "data", "synthetic_tickets.csv")
OUTPUT_CSV = os.path.join(PROJECT_ROOT, "data", "category_stores_with_scenario_id.csv")


def main():
    if not os.path.exists(SYNTHETIC_CSV):
        print(f"ERROR: could not find {SYNTHETIC_CSV}")
        sys.exit(1)

    synth = pd.read_csv(SYNTHETIC_CSV)
    if "scenario_id" not in synth.columns:
        print("ERROR: synthetic_tickets.csv has no 'scenario_id' column. "
              "Did you regenerate it with the patched generate_dataset.py?")
        sys.exit(1)

    # Ground-truth lookup: id -> scenario_id (id is globally unique already).
    id_to_scenario = synth.set_index("id")["scenario_id"]

    combined_frames = []
    print("=" * 70)
    print("Joining scenario_id ground truth onto category_stores data")
    print("=" * 70)

    for category in CATEGORIES:
        csv_path = os.path.join(CATEGORY_STORES_DIR, category + ".csv")
        if not os.path.exists(csv_path):
            print(f"  [skip] {category}: file not found.")
            continue

        df = pd.read_csv(csv_path)

        required = ["batch_ticket_id", "original_id", "resolution_status",
                    "resolution_text"]
        missing = [c for c in required if c not in df.columns]
        if missing:
            print(f"  [skip] {category}: missing column(s) {missing}.")
            continue

        before = len(df)
        df = df[df["resolution_status"] == "auto_resolved"].copy()
        df = df[df["resolution_text"].notna() & (df["resolution_text"].str.strip() != "")]
        after = len(df)

        # Join scenario_id via original_id -> synthetic_tickets.csv's id.
        df["scenario_id"] = df["original_id"].map(id_to_scenario)

        unmatched = df["scenario_id"].isna().sum()
        if unmatched:
            print(f"  [warn] {category}: {unmatched} row(s) had no matching "
                  "original_id in synthetic_tickets.csv (dropping these -- "
                  "no ground truth available).")
            df = df[df["scenario_id"].notna()].copy()

        df["scenario_id"] = df["scenario_id"].astype(int)
        df["category"] = category

        print(f"  [ok] {category}: {before} rows -> {after} auto_resolved "
              f"with text -> {len(df)} with matched scenario_id "
              f"({df['scenario_id'].nunique()} distinct scenarios present)")

        combined_frames.append(
            df[["batch_ticket_id", "original_id", "category", "scenario_id",
                "resolution_status", "resolution_text"]]
        )

    if not combined_frames:
        print("\nERROR: no category files produced any usable rows.")
        sys.exit(1)

    combined = pd.concat(combined_frames, ignore_index=True)
    combined.to_csv(OUTPUT_CSV, index=False)

    print("\n" + "=" * 70)
    print(f"Wrote {len(combined)} total rows to:\n  {OUTPUT_CSV}")
    print("=" * 70)
    print("\nPer-category scenario_id distribution (how many tickets share "
          "each ground-truth scenario -- this previews how much real "
          "clustering signal exists per category):")
    for category in CATEGORIES:
        sub = combined[combined["category"] == category]
        if sub.empty:
            continue
        counts = sub["scenario_id"].value_counts().sort_index()
        print(f"\n  {category} ({len(sub)} rows, "
              f"{sub['scenario_id'].nunique()} distinct scenarios):")
        print("    " + ", ".join(f"scenario {sid}: {cnt} tickets"
                                  for sid, cnt in counts.items()))


if __name__ == "__main__":
    main()