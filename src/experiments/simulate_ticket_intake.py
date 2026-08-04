"""
simulate_ticket_intake.py

Simulate a realistic batch of incoming tickets with a Zipf-like skewed
distribution across the 7 categories, drawn EXCLUSIVELY from the held-out
test split (never from training rows). Produces:

    data/batch_intake/incoming_tickets_batch.csv

Run from project root:
    python -m src.experiments.simulate_ticket_intake
    (or) python src/experiments/simulate_ticket_intake.py
"""

import os
import sys

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

# --------------------------------------------------------------------------- #
# Configuration
# --------------------------------------------------------------------------- #
RANDOM_SEED = 42                 # Fixed everywhere in this project.
BATCH_SIZE = 500                 # Configurable batch size for the simulation.
TEST_SIZE = 0.2                  # Must match the split used everywhere else.

CATEGORIES = [
    "Infrastructure",
    "Application",
    "Security",
    "Database",
    "Storage",
    "Network",
    "Access Management",
]

# --------------------------------------------------------------------------- #
# Path resolution: project root = two directories up from this script's dir.
# (Matches train_embeddings.py's os.path.* pattern.)
#   this file: <root>/src/experiments/simulate_ticket_intake.py
#   -> dirname = <root>/src/experiments
#   -> two dirs up = <root>
# --------------------------------------------------------------------------- #
_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(_SCRIPT_DIR, "..", ".."))

DATA_DIR = os.path.join(PROJECT_ROOT, "data")
SOURCE_CSV = os.path.join(DATA_DIR, "synthetic_tickets.csv")
BATCH_DIR = os.path.join(DATA_DIR, "batch_intake")
OUTPUT_CSV = os.path.join(BATCH_DIR, "incoming_tickets_batch.csv")


def _fail(message: str) -> None:
    """Print a clear, actionable error and exit non-zero (no raw traceback)."""
    print(f"\n[ERROR] {message}\n", file=sys.stderr)
    sys.exit(1)


def load_source_dataframe() -> pd.DataFrame:
    """Load the full synthetic dataset, failing clearly if it is missing."""
    if not os.path.isfile(SOURCE_CSV):
        _fail(
            "Could not find the source dataset at:\n"
            f"    {SOURCE_CSV}\n"
            "Expected data/synthetic_tickets.csv relative to the project root.\n"
            "Generate it first (e.g. `python -m src.data.generate_dataset`) "
            "before running this simulation."
        )

    df = pd.read_csv(SOURCE_CSV)

    # Basic schema guard so downstream code fails with a clear message
    # rather than a KeyError deep inside sampling.
    required_cols = {"id", "title", "description", "category"}
    missing = required_cols - set(df.columns)
    if missing:
        _fail(
            f"{SOURCE_CSV} is missing required column(s): {sorted(missing)}.\n"
            f"Found columns: {list(df.columns)}.\n"
            "Expected at least: id, title, description, category."
        )
    return df


def get_held_out_split(df: pd.DataFrame) -> pd.DataFrame:
    """
    Recreate the EXACT train_test_split used everywhere else in the project so
    the 'unseen' claim is genuinely true.

    test_size=0.2, random_state=42, stratify=y  ->  the 800-row held-out set.
    """
    X = df
    y = df["category"]

    _, X_test = train_test_split(
        X,
        test_size=TEST_SIZE,
        random_state=RANDOM_SEED,
        stratify=y,
    )
    # X_test carries all original columns, indexed back into df.
    return X_test.reset_index(drop=True)


def compute_zipf_targets(rng: np.random.RandomState) -> dict:
    """
    Assign a randomized rank (1..7) to the 7 categories, then set target volume
    proportional to 1/rank, normalized to sum to BATCH_SIZE exactly.

    Returns an ordered dict: {category_name: target_volume}, ordered by rank
    (rank 1 = highest volume) for clear printing.
    """
    # Randomize which category gets which rank (per-run, but reproducible
    # under the fixed seed) rather than hardcoding a category order.
    ranked_categories = list(CATEGORIES)
    rng.shuffle(ranked_categories)  # in-place shuffle -> ranked_categories[0] is rank 1

    ranks = np.arange(1, len(ranked_categories) + 1)  # 1..7
    weights = 1.0 / ranks
    weights = weights / weights.sum()

    raw_volumes = weights * BATCH_SIZE
    volumes = np.floor(raw_volumes).astype(int)

    # Absorb the rounding remainder into the LAST-ranked category so the total
    # is exactly BATCH_SIZE.
    remainder = BATCH_SIZE - int(volumes.sum())
    volumes[-1] += remainder

    targets = {cat: int(vol) for cat, vol in zip(ranked_categories, volumes)}

    print("=" * 64)
    print(f"Zipf-like target distribution (BATCH_SIZE = {BATCH_SIZE})")
    print("=" * 64)
    print(f"{'Rank':<6}{'Category':<20}{'Target volume':>14}")
    print("-" * 64)
    for rank, cat in enumerate(ranked_categories, start=1):
        print(f"{rank:<6}{cat:<20}{targets[cat]:>14}")
    print("-" * 64)
    print(f"{'':<6}{'TOTAL':<20}{sum(targets.values()):>14}")
    print("=" * 64)

    return targets


def redistribute_shortfall(targets: dict, pool_sizes: dict) -> dict:
    """
    Cap each target at its available held-out pool size, and redistribute any
    shortfall proportionally among categories that still have spare capacity.

    Returns the adjusted {category: final_target} dict whose sum matches the
    original total wherever total pool capacity allows.
    """
    original_total = sum(targets.values())
    final = dict(targets)

    # First pass: cap at pool size, accumulate shortfall.
    shortfall = 0
    for cat in final:
        if final[cat] > pool_sizes[cat]:
            deficit = final[cat] - pool_sizes[cat]
            print(
                f"[WARNING] Category '{cat}': held-out pool has only "
                f"{pool_sizes[cat]} tickets but target is {final[cat]}. "
                f"Capping at {pool_sizes[cat]} and redistributing {deficit}."
            )
            shortfall += deficit
            final[cat] = pool_sizes[cat]

    # Redistribute shortfall proportionally among categories with spare room.
    # Iterate because a redistribution pass may itself hit new caps.
    while shortfall > 0:
        spare = {c: pool_sizes[c] - final[c] for c in final if pool_sizes[c] > final[c]}
        total_spare = sum(spare.values())
        if total_spare == 0:
            print(
                f"[WARNING] Cannot fully redistribute {shortfall} ticket(s): "
                "no category has remaining held-out capacity. "
                f"Final batch will be {original_total - shortfall} tickets."
            )
            break

        spare_weight_total = sum(spare.values())
        # Proportional allocation of the shortfall across spare capacity.
        allocated = 0
        # Sort for deterministic assignment order.
        spare_items = sorted(spare.items(), key=lambda kv: kv[0])
        additions = {}
        for cat, room in spare_items:
            share = int(np.floor(shortfall * (room / spare_weight_total)))
            share = min(share, room)
            additions[cat] = share
            allocated += share

        # Distribute any residual (from flooring) one-by-one to categories
        # that still have room, deterministically.
        residual = shortfall - allocated
        for cat, room in spare_items:
            if residual <= 0:
                break
            if additions.get(cat, 0) < room:
                additions[cat] = additions.get(cat, 0) + 1
                residual += 0
                allocated += 1
                residual = shortfall - allocated

        for cat, add in additions.items():
            final[cat] += add

        new_shortfall = shortfall - allocated
        if new_shortfall == shortfall:
            # No progress possible; avoid infinite loop.
            print(
                f"[WARNING] Redistribution stalled with {shortfall} ticket(s) "
                "unassigned; capacity exhausted."
            )
            break
        shortfall = new_shortfall

    return final


def sample_batch(held_out: pd.DataFrame, targets: dict) -> pd.DataFrame:
    """
    Sample the target number of tickets per category (without replacement,
    random_state=42) from the held-out split, handling pool shortfalls.
    """
    pool_sizes = {
        cat: int((held_out["category"] == cat).sum()) for cat in CATEGORIES
    }
    final_targets = redistribute_shortfall(targets, pool_sizes)

    sampled_frames = []
    for cat in CATEGORIES:
        n = final_targets[cat]
        if n <= 0:
            continue
        pool = held_out[held_out["category"] == cat]
        take = min(n, len(pool))
        sampled_frames.append(
            pool.sample(n=take, replace=False, random_state=RANDOM_SEED)
        )

    if not sampled_frames:
        _fail(
            "No tickets could be sampled for the batch. "
            "Check that the held-out split actually contains all 7 categories."
        )

    batch = pd.concat(sampled_frames, ignore_index=True)

    # Shuffle so tickets aren't grouped by category (realistic arrival order).
    batch = batch.sample(frac=1.0, random_state=RANDOM_SEED).reset_index(drop=True)

    return batch, final_targets


def build_output(batch: pd.DataFrame) -> pd.DataFrame:
    """Assemble the final output frame with the required column layout."""
    out = pd.DataFrame(
        {
            "batch_ticket_id": range(1, len(batch) + 1),
            "original_id": batch["id"].values,
            "title": batch["title"].values,
            "description": batch["description"].values,
            # TRUE label retained for evaluation ONLY. The column name makes it
            # explicit that this must NEVER be fed into the classifier.
            "ground_truth_category": batch["category"].values,
        }
    )
    return out


def print_summary(final_targets: dict, output_df: pd.DataFrame) -> None:
    """Print a category / target / actual summary table (generate_dataset style)."""
    actual_counts = output_df["ground_truth_category"].value_counts().to_dict()

    print("\n" + "=" * 64)
    print("BATCH INTAKE SUMMARY")
    print("=" * 64)
    print(f"{'Category':<20}{'Target':>12}{'Actual':>12}")
    print("-" * 64)
    total_target = 0
    total_actual = 0
    for cat in CATEGORIES:
        tgt = final_targets.get(cat, 0)
        act = int(actual_counts.get(cat, 0))
        total_target += tgt
        total_actual += act
        print(f"{cat:<20}{tgt:>12}{act:>12}")
    print("-" * 64)
    print(f"{'TOTAL':<20}{total_target:>12}{total_actual:>12}")
    print("=" * 64)


def main() -> None:
    rng = np.random.RandomState(RANDOM_SEED)

    df = load_source_dataframe()
    held_out = get_held_out_split(df)

    print(f"Loaded {len(df)} total tickets; held-out split = {len(held_out)} rows.")

    targets = compute_zipf_targets(rng)
    batch, final_targets = sample_batch(held_out, targets)
    output_df = build_output(batch)

    os.makedirs(BATCH_DIR, exist_ok=True)
    output_df.to_csv(OUTPUT_CSV, index=False)

    print_summary(final_targets, output_df)
    print(f"\nWrote {len(output_df)} tickets to:\n    {OUTPUT_CSV}\n")


if __name__ == "__main__":
    main()
