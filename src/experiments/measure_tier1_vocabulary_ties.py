# src/experiments/measure_tier1_vocabulary_ties.py
"""
Phase 9A -- a committed writer for the Tier-1 vocabulary tie-break finding.

Phase 8B.3 found that TfidfVectorizer(max_features=5000) does not choose its
vocabulary from the data alone. sklearn's `_limit_features` ranks terms by
corpus COUNT and keeps the top 5,000 with `(-tfs).argsort()`, an UNSTABLE
quicksort, so every term tied at the cut is placed by the sort's tie-break --
and numpy dispatches SIMD sorts by CPU. The figures (4,240 above the cut,
11,834 tied, 760 slots) were recorded in the docs and in code comments, but no
committed file produced them, so the paper could not cite them. This script is
that file's writer.

WHAT IT COUNTS, AND WHY IT IS PLATFORM-INDEPENDENT. It builds the COUNT matrix
with exactly the vectorizer configuration every Tier-1 fit uses (unigrams +
bigrams, English stop words, no max_features) and reads three integers off the
sorted count vector: how many terms sit strictly above the 5,000th count, how
many share that count, and how many of the 5,000 slots the tied terms compete
for. Counts are integers and the question "how many terms have count c" does
not depend on sort order -- only WHICH tied terms survive does. So this writer
reproduces on any machine, even though the vocabulary it describes did not.

Two corpora, matching the two kinds of fit in the repository:
  full4000   -- all 4,000 rows: the production Tier-1 (before 8B.3 fixed it)
                and generalization_test.py's full4000 arm.
  split3200  -- the seed-42 stratified 80/20 training split, which is what
                train_baseline_tfidf.py, generalization_test.py's split arm and
                run_imbalance_sweep.py-style fits see.

FATAL unless the full4000 figures equal the ones recorded in 8B.3. A mismatch
means the recorded finding was wrong, and it is reported, never smoothed over.

Offline: no model load, no Gemini, no Ollama. Refuses to overwrite its output
without --force.

Usage (from the project root):
    python src/experiments/measure_tier1_vocabulary_ties.py
"""

import argparse
import json
import os
import sys

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split

THIS_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(os.path.dirname(THIS_DIR))

DATASET_PATH = os.path.join(PROJECT_ROOT, "data", "synthetic_tickets.csv")
OUTPUT_PATH = os.path.join(PROJECT_ROOT, "data", "tier1_vocabulary_ties.json")

MAX_FEATURES = 5000
NGRAM_RANGE = (1, 2)
STOP_WORDS = "english"
RANDOM_STATE = 42
TEST_SIZE = 0.2

# Recorded in Phase 8B.3 (PROJECT_STATUS.md, train_cascade.train_tier1's
# docstring). Checked, not used: the file below is written from the counts.
RECORDED_8B3_FULL = {"above_cut": 4240, "tied_at_cut": 11834,
                     "slots_for_tied": 760}
RECORDED_8B3_SPLIT_SLOTS = 950


def load_texts():
    df = pd.read_csv(DATASET_PATH)
    texts = (df["title"].fillna("").astype(str) + " "
             + df["description"].fillna("").astype(str))
    return texts, df["category"].astype(str), len(df)


def tie_profile(texts):
    counts = CountVectorizer(ngram_range=NGRAM_RANGE, stop_words=STOP_WORDS)
    X = counts.fit_transform(texts)
    tfs = np.asarray(X.sum(axis=0)).ravel()
    ranked = np.sort(tfs)[::-1]
    cut_count = int(ranked[MAX_FEATURES - 1])
    above = int((tfs > cut_count).sum())
    tied = int((tfs == cut_count).sum())
    slots = MAX_FEATURES - above
    return {
        "n_documents": int(X.shape[0]),
        "terms_total": int(tfs.size),
        "max_features": MAX_FEATURES,
        "count_at_cut": cut_count,
        "above_cut": above,
        "tied_at_cut": tied,
        "slots_for_tied": slots,
        "share_of_features_set_by_tie_break": slots / MAX_FEATURES,
    }


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[1])
    parser.add_argument("--force", action="store_true",
                        help="overwrite data/tier1_vocabulary_ties.json")
    args = parser.parse_args(argv)

    if os.path.exists(OUTPUT_PATH) and not args.force:
        print(f"[ERROR] {os.path.relpath(OUTPUT_PATH, PROJECT_ROOT)} exists. "
              f"Re-run with --force to overwrite it.")
        return 1

    texts, labels, n_rows = load_texts()
    full = tie_profile(texts)
    train_texts, _, _, _ = train_test_split(
        texts, labels, test_size=TEST_SIZE, random_state=RANDOM_STATE,
        stratify=labels)
    split = tie_profile(train_texts)

    mismatches = [
        f"full4000 {key}: measured {full[key]}, recorded {value}"
        for key, value in RECORDED_8B3_FULL.items() if full[key] != value]
    if split["slots_for_tied"] != RECORDED_8B3_SPLIT_SLOTS:
        mismatches.append(
            f"split3200 slots_for_tied: measured {split['slots_for_tied']}, "
            f"recorded {RECORDED_8B3_SPLIT_SLOTS}")
    if mismatches:
        print("[FATAL] The recorded 8B.3 tie figures do not reproduce:")
        for line in mismatches:
            print(f"        {line}")
        return 2

    result = {
        "description": (
            "Count-matrix tie profile at TfidfVectorizer(max_features=5000, "
            "ngram_range=(1,2), stop_words='english'). Terms tied at the cut "
            "are ordered by an unstable sort, so 'slots_for_tied' features are "
            "chosen by the sort's tie-break rather than by the corpus."),
        "dataset": "data/synthetic_tickets.csv",
        "dataset_rows": n_rows,
        "split": {"test_size": TEST_SIZE, "random_state": RANDOM_STATE,
                  "stratify": "category"},
        "full4000": full,
        "split3200": split,
    }
    with open(OUTPUT_PATH, "w", encoding="utf-8", newline="\n") as fh:
        json.dump(result, fh, indent=2, sort_keys=True)
        fh.write("\n")

    for arm, prof in (("full4000", full), ("split3200", split)):
        print(f"{arm:<10} docs={prof['n_documents']:>5}  "
              f"above={prof['above_cut']:>5}  tied={prof['tied_at_cut']:>6}  "
              f"slots={prof['slots_for_tied']:>4}  "
              f"share={prof['share_of_features_set_by_tie_break']:.4f}")
    print(f"Wrote {os.path.relpath(OUTPUT_PATH, PROJECT_ROOT)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
