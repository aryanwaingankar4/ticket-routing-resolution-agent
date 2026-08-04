"""
generate_skewed_datasets.py
===========================
Class-imbalance experiment — dataset generation stage.

Down-samples the "Access Management" category from the EXISTING
data/synthetic_tickets.csv to a series of target counts, holding all other
six categories at their full existing volume. Writes one skewed CSV per level
to data/skewed/.

Conventions (project-wide):
  - Paths resolved via os.path relative to THIS script's location, walking up
    to the project root. Never assumes a fixed cwd, never hardcodes absolutes.
  - random_state=42 everywhere for reproducibility.
  - Defensive, actionable error messages (no raw tracebacks) on missing input.
  - Idempotent: safe to re-run; overwrites files in data/skewed/.

Run:
    python src/experiments/generate_skewed_datasets.py
"""

import os
import sys

import numpy as np
import pandas as pd

# --------------------------------------------------------------------------- #
# Reproducibility
# --------------------------------------------------------------------------- #
RANDOM_STATE = 42
np.random.seed(RANDOM_STATE)

# --------------------------------------------------------------------------- #
# Path resolution — relative to this script, walk up to project root
#   .../ticket-routing-agent/src/experiments/generate_skewed_datasets.py
#   project root = two levels up from this file's directory
# --------------------------------------------------------------------------- #
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, os.pardir, os.pardir))

DATA_DIR = os.path.join(PROJECT_ROOT, "data")
SOURCE_CSV = os.path.join(DATA_DIR, "synthetic_tickets.csv")
SKEWED_DIR = os.path.join(DATA_DIR, "skewed")

# --------------------------------------------------------------------------- #
# Experiment constants
# --------------------------------------------------------------------------- #
TARGET_CATEGORY = "Access Management"

ALL_CATEGORIES = [
    "Infrastructure",
    "Application",
    "Security",
    "Database",
    "Storage",
    "Network",
    "Access Management",
]

# 571 = full baseline (no skew) -> written unchanged for a clean level-0 point.
SKEW_LEVELS = [571, 500, 200, 100, 50]

# Column that must never contain nulls/empties after sampling. The sampling
# operation itself cannot introduce nulls, but we validate anyway as a gate.
REQUIRED_TEXT_COLUMN_CANDIDATES = ["text", "ticket_text", "description", "body"]
CATEGORY_COLUMN_CANDIDATES = ["category", "label", "Category"]


def _fail(message: str) -> None:
    """Print a clear, actionable error and exit non-zero (no traceback)."""
    print("\n" + "=" * 75)
    print("ERROR — cannot continue")
    print("=" * 75)
    print(message)
    print("=" * 75 + "\n")
    sys.exit(1)


def _resolve_column(df: pd.DataFrame, candidates, purpose: str) -> str:
    """Find the first candidate column present in df, or fail clearly."""
    for c in candidates:
        if c in df.columns:
            return c
    _fail(
        f"Could not find the {purpose} column in {SOURCE_CSV}.\n"
        f"Looked for any of: {candidates}\n"
        f"Actual columns present: {list(df.columns)}\n"
        f"Fix: ensure the dataset uses one of the expected column names, or add\n"
        f"the correct name to the candidate list in this script."
    )


def load_source() -> pd.DataFrame:
    if not os.path.isfile(SOURCE_CSV):
        _fail(
            f"Source dataset not found:\n    {SOURCE_CSV}\n\n"
            f"This experiment samples from the EXISTING dataset; it never\n"
            f"regenerates tickets. Make sure the base dataset exists first.\n\n"
            f"Expected location (relative to project root):\n"
            f"    data/synthetic_tickets.csv"
        )
    try:
        df = pd.read_csv(SOURCE_CSV)
    except Exception as exc:  # noqa: BLE001 - surface a clean message
        _fail(f"Failed to read {SOURCE_CSV}:\n    {exc}")

    if df.empty:
        _fail(f"Source dataset {SOURCE_CSV} is empty.")
    return df


def validate_level(
    result: pd.DataFrame,
    original_counts: pd.Series,
    target_n: int,
    category_col: str,
    text_col: str,
) -> None:
    """
    Validation gate — run BEFORE writing each CSV. Fails loudly on any issue
    so a bad CSV is never written.
    """
    result_counts = result[category_col].value_counts()

    # 1) Access Management must equal N exactly.
    am_count = int(result_counts.get(TARGET_CATEGORY, 0))
    if am_count != target_n:
        _fail(
            f"[level {target_n}] Validation failed: '{TARGET_CATEGORY}' count "
            f"is {am_count}, expected exactly {target_n}."
        )

    # 2) No other category's count changed from the original.
    for cat in ALL_CATEGORIES:
        if cat == TARGET_CATEGORY:
            continue
        orig = int(original_counts.get(cat, 0))
        now = int(result_counts.get(cat, 0))
        if now != orig:
            _fail(
                f"[level {target_n}] Validation failed: category '{cat}' count "
                f"changed from {orig} (original) to {now} (skewed). Only "
                f"'{TARGET_CATEGORY}' should change."
            )

    # 3) No null / empty required fields introduced.
    if result[category_col].isnull().any():
        _fail(f"[level {target_n}] Validation failed: null values in "
              f"category column '{category_col}'.")
    if result[text_col].isnull().any():
        _fail(f"[level {target_n}] Validation failed: null values in "
              f"text column '{text_col}'.")
    empty_text = (result[text_col].astype(str).str.strip() == "").sum()
    if empty_text > 0:
        _fail(f"[level {target_n}] Validation failed: {empty_text} empty "
              f"string(s) in text column '{text_col}'.")


def main() -> None:
    print("=" * 75)
    print("SKEWED DATASET GENERATION  —  class-imbalance experiment")
    print("=" * 75)
    print(f"Project root : {PROJECT_ROOT}")
    print(f"Source CSV   : {SOURCE_CSV}")
    print(f"Output dir   : {SKEWED_DIR}")
    print(f"Target cat   : {TARGET_CATEGORY}")
    print(f"Skew levels  : {SKEW_LEVELS}")
    print(f"random_state : {RANDOM_STATE}")
    print("-" * 75)

    df = load_source()
    category_col = _resolve_column(df, CATEGORY_COLUMN_CANDIDATES, "category")
    text_col = _resolve_column(df, REQUIRED_TEXT_COLUMN_CANDIDATES, "ticket text")

    # Sanity check the category values match the expected vocabulary exactly.
    present = set(df[category_col].dropna().unique())
    expected = set(ALL_CATEGORIES)
    unexpected = present - expected
    missing = expected - present
    if unexpected:
        _fail(
            f"Dataset contains unexpected category values not in the agreed "
            f"vocabulary:\n    {sorted(unexpected)}\n"
            f"Expected exactly: {ALL_CATEGORIES}"
        )
    if missing:
        _fail(
            f"Dataset is missing expected categories:\n    {sorted(missing)}\n"
            f"Expected exactly: {ALL_CATEGORIES}"
        )

    original_counts = df[category_col].value_counts()
    am_available = int(original_counts.get(TARGET_CATEGORY, 0))
    print(f"Detected category column : '{category_col}'")
    print(f"Detected text column     : '{text_col}'")
    print(f"'{TARGET_CATEGORY}' rows available : {am_available}")
    print("Original per-category counts:")
    for cat in ALL_CATEGORIES:
        print(f"    {cat:<20} {int(original_counts.get(cat, 0))}")
    print("-" * 75)

    # Guard: cannot sample more AM rows than exist.
    for n in SKEW_LEVELS:
        if n > am_available:
            _fail(
                f"Requested skew level {n} exceeds available "
                f"'{TARGET_CATEGORY}' rows ({am_available}). Sampling without "
                f"replacement is impossible. Adjust SKEW_LEVELS."
            )

    # Ensure output dir exists (idempotent).
    os.makedirs(SKEWED_DIR, exist_ok=True)

    # Split once: everything NOT Access Management stays fixed across levels.
    other_rows = df[df[category_col] != TARGET_CATEGORY].copy()
    am_rows_all = df[df[category_col] == TARGET_CATEGORY].copy()

    # --- Summary table accumulator ---
    summary = []

    for n in SKEW_LEVELS:
        print(f"\n>>> Level {n}  (sampling '{TARGET_CATEGORY}' down to {n})")

        if n == am_available:
            # Full baseline: keep AM unchanged (still deterministic).
            am_sample = am_rows_all.copy()
            print(f"    Full baseline — using all {n} AM rows unchanged.")
        else:
            am_sample = am_rows_all.sample(
                n=n, replace=False, random_state=RANDOM_STATE
            )
            print(f"    Sampled {n} AM rows without replacement "
                  f"(random_state={RANDOM_STATE}).")

        result = pd.concat([other_rows, am_sample], ignore_index=True)

        # Validation gate BEFORE writing.
        validate_level(result, original_counts, n, category_col, text_col)

        out_path = os.path.join(SKEWED_DIR, f"synthetic_tickets_am{n}.csv")
        result.to_csv(out_path, index=False)

        total_rows = len(result)
        am_written = int(result[category_col].value_counts().get(TARGET_CATEGORY, 0))
        matches = "YES" if am_written == n else "NO"
        print(f"    Wrote {out_path}")
        print(f"    total rows = {total_rows} | AM rows = {am_written} | "
              f"matches target {n}: {matches}")

        summary.append(
            {
                "level": n,
                "file": os.path.relpath(out_path, PROJECT_ROOT),
                "total_rows": total_rows,
                "am_rows": am_written,
                "matches_target": matches,
            }
        )

    # --- Per-level summary table ---
    print("\n" + "=" * 75)
    print("SUMMARY")
    print("=" * 75)
    header = f"{'level':>6} | {'total':>6} | {'AM rows':>7} | {'match':>5} | file"
    print(header)
    print("-" * len(header))
    for row in summary:
        print(
            f"{row['level']:>6} | {row['total_rows']:>6} | {row['am_rows']:>7} | "
            f"{row['matches_target']:>5} | {row['file']}"
        )
    print("=" * 75)
    print("All skewed datasets generated and validated successfully.")
    print(f"Next step:\n    python src/experiments/run_imbalance_sweep.py")


if __name__ == "__main__":
    main()
