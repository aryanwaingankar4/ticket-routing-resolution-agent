"""
train_baseline_tfidf.py
=======================

Baseline ticket classifier for the IT support routing agent.

WHY TF-IDF + LOGISTIC REGRESSION AS THE FIRST BASELINE
------------------------------------------------------
This is deliberately the *simplest thing that could possibly work*, and it
exists to produce a NUMBER, not to be the final production classifier.

  * TF-IDF is a purely lexical, "bag-of-n-grams" representation. It counts
    which words/2-word phrases appear and weights them by how discriminative
    they are across categories. It has NO understanding of meaning: to TF-IDF,
    "cannot log in" and "unable to authenticate" are completely unrelated
    strings because they share no tokens.

  * Logistic Regression on top of TF-IDF is a fast, well-understood linear
    model. No GPU, no model download, no network, trains in well under a second.

  * The whole point of a baseline is comparison. Before we can justify the
    complexity/cost of a semantic-embeddings classifier (sentence-transformers,
    etc.), we need a defensible, evidence-based answer to "why embeddings over
    keyword methods?". That answer must be a measured accuracy gap, not an
    opinion. This file produces the in-distribution number; generalization_test.py
    produces the number that actually matters.

A KNOWN TRAP with this baseline: our dataset is template-generated synthetic
data. If a category's templates reuse the same vocabulary, TF-IDF can score
near-perfectly by MEMORIZING template tokens, which looks like success but is
actually test-set leakage / lack of generalization. This script prints a loud
warning when that happens.

Run from the project root:
    python src/classification/train_baseline_tfidf.py
"""

import os
import sys

import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    classification_report,
    confusion_matrix,
)

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
RANDOM_STATE = 42          # FIXED so every run is reproducible.
TEST_SIZE = 0.2
MIN_CATEGORIES = 7
EXPECTED_COLUMNS = ["id", "title", "description", "category", "resolution", "priority"]
NEAR_PERFECT_THRESHOLD = 0.97


# ---------------------------------------------------------------------------
# Path resolution
# ---------------------------------------------------------------------------
def resolve_csv_path():
    """
    Resolve data/synthetic_tickets.csv relative to THIS script's own location,
    not the directory the script happens to be invoked from.

    Layout assumed:
        <project_root>/
            data/synthetic_tickets.csv
            src/classification/train_baseline_tfidf.py   <- this file

    So project root is two directories up from this file.
    """
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.abspath(os.path.join(script_dir, "..", ".."))
    return os.path.join(project_root, "data", "synthetic_tickets.csv")


# ---------------------------------------------------------------------------
# Loading + validation
# ---------------------------------------------------------------------------
def load_and_validate(csv_path):
    """
    Load the CSV and validate it BEFORE any training happens, so problems are
    reported clearly and up front rather than as a cryptic sklearn traceback.
    Returns a validated DataFrame.
    """
    if not os.path.exists(csv_path):
        raise FileNotFoundError(
            f"\nDataset not found at:\n    {csv_path}\n\n"
            "This baseline expects data/synthetic_tickets.csv to already exist.\n"
            "Run the dataset generator first (e.g. the synthetic data script that\n"
            "produces data/synthetic_tickets.csv), then re-run this file."
        )

    try:
        df = pd.read_csv(csv_path)
    except Exception as exc:  # noqa: BLE001 - we want a clear message on any read failure
        raise RuntimeError(
            f"Failed to read CSV at {csv_path}. Underlying error: {exc}"
        ) from exc

    # --- column check -------------------------------------------------------
    missing_columns = [c for c in EXPECTED_COLUMNS if c not in df.columns]
    if missing_columns:
        print("ERROR: The dataset is missing required column(s):")
        for col in missing_columns:
            print(f"    - {col}")
        print(f"\nExpected columns: {EXPECTED_COLUMNS}")
        print(f"Found columns:    {list(df.columns)}")
        sys.exit(1)

    # --- NaN check on the columns we actually use ---------------------------
    used_columns = ["title", "description", "category"]
    nan_report = {col: int(df[col].isna().sum()) for col in used_columns}
    if any(count > 0 for count in nan_report.values()):
        print("ERROR: NaN / missing values found in required columns:")
        for col, count in nan_report.items():
            if count > 0:
                print(f"    - {col}: {count} missing value(s)")
        print("\nClean these rows in the dataset before training.")
        sys.exit(1)

    # --- category count check ----------------------------------------------
    unique_categories = sorted(df["category"].unique().tolist())
    if len(unique_categories) < MIN_CATEGORIES:
        print(
            f"ERROR: Expected at least {MIN_CATEGORIES} unique categories, "
            f"but found only {len(unique_categories)}:"
        )
        for cat in unique_categories:
            print(f"    - {cat}")
        sys.exit(1)

    return df


def validate_stratification(y):
    """
    train_test_split(stratify=y) requires every class to have at least 2 rows
    (one for train, one for test). Check this explicitly and fail with a clear
    message rather than letting sklearn raise a cryptic ValueError mid-run.
    """
    counts = y.value_counts()
    # With test_size=0.2 a class needs enough rows that its test allocation is
    # at least 1. The hard floor for stratify to work at all is 2 rows/class.
    too_small = counts[counts < 2]
    if len(too_small) > 0:
        print("ERROR: Some categories have too few rows to stratify the split:")
        for cat, count in too_small.items():
            print(f"    - {cat}: {count} row(s) (need at least 2)")
        print("\nAdd more examples for these categories before training.")
        sys.exit(1)


# ---------------------------------------------------------------------------
# Reporting helpers
# ---------------------------------------------------------------------------
def print_labeled_confusion_matrix(y_true, y_pred, labels):
    """
    Print a confusion matrix with category NAMES on both axes, not bare indices.
    Rows = true category, columns = predicted category.
    """
    cm = confusion_matrix(y_true, y_pred, labels=labels)

    # Short display names keep columns aligned but readable.
    short = [lab[:10] for lab in labels]
    col_width = max(10, max(len(s) for s in short)) + 2
    row_label_width = max(len(lab) for lab in labels) + 2

    print("\nConfusion matrix (rows = TRUE category, cols = PREDICTED category):\n")

    # Header row
    header = " " * row_label_width + "".join(s.rjust(col_width) for s in short)
    print(header)

    for i, lab in enumerate(labels):
        row_cells = "".join(str(cm[i][j]).rjust(col_width) for j in range(len(labels)))
        print(lab.ljust(row_label_width) + row_cells)

    print("\nColumn key (abbreviation -> full category):")
    for s, lab in zip(short, labels):
        if s != lab:
            print(f"    {s:<12} -> {lab}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    csv_path = resolve_csv_path()
    print("=" * 75)
    print("BASELINE CLASSIFIER: TF-IDF + Logistic Regression")
    print("=" * 75)
    print(f"Dataset: {csv_path}\n")

    df = load_and_validate(csv_path)

    # Combine title + description as the classifier input text.
    # We fill nothing here because NaNs were already rejected in validation.
    X_text = (df["title"].astype(str) + " " + df["description"].astype(str))
    y = df["category"].astype(str)

    print(f"Loaded {len(df)} tickets across {y.nunique()} categories.")
    print("Rows per category:")
    for cat, count in y.value_counts().sort_index().items():
        print(f"    {cat:<20} {count}")
    print()

    # Guard the stratified split before calling sklearn.
    validate_stratification(y)

    # Stratified split so every category is represented proportionally in both
    # train and test. Fixed random_state => reproducible numbers.
    X_train, X_test, y_train, y_test = train_test_split(
        X_text,
        y,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
        stratify=y,
    )

    # -----------------------------------------------------------------------
    # The deliberately-simple baseline.
    #
    # TfidfVectorizer:
    #   max_features=5000  -> cap vocabulary to the 5000 most informative terms
    #   ngram_range=(1,2)  -> unigrams + bigrams, so short phrases like
    #                         "log in" or "disk full" become single features
    #   stop_words="english" -> drop uninformative words ("the", "is", ...)
    #
    # This is a purely LEXICAL representation. It matches on the surface form of
    # words. That is exactly the property we want to expose as a limitation in
    # generalization_test.py.
    # -----------------------------------------------------------------------
    vectorizer = TfidfVectorizer(
        max_features=5000,
        ngram_range=(1, 2),
        stop_words="english",
    )
    X_train_vec = vectorizer.fit_transform(X_train)
    X_test_vec = vectorizer.transform(X_test)

    clf = LogisticRegression(max_iter=1000)
    clf.fit(X_train_vec, y_train)

    y_pred = clf.predict(X_test_vec)

    # -----------------------------------------------------------------------
    # Metrics
    # -----------------------------------------------------------------------
    labels = sorted(y.unique().tolist())

    accuracy = accuracy_score(y_test, y_pred)
    macro_f1 = f1_score(y_test, y_pred, average="macro")
    weighted_f1 = f1_score(y_test, y_pred, average="weighted")

    print("-" * 75)
    print("IN-DISTRIBUTION RESULTS (held-out 20% of the same synthetic data)")
    print("-" * 75)
    print(f"Accuracy    : {accuracy:.4f}")
    print(f"Macro F1    : {macro_f1:.4f}")
    print(f"Weighted F1 : {weighted_f1:.4f}")

    print("\nPer-category classification report:\n")
    print(classification_report(y_test, y_pred, labels=labels, digits=4, zero_division=0))

    print_labeled_confusion_matrix(y_test, y_pred, labels)

    # -----------------------------------------------------------------------
    # The critical warning about near-perfect accuracy.
    # -----------------------------------------------------------------------
    if accuracy > NEAR_PERFECT_THRESHOLD:
        print("\n" + "!" * 75)
        print("WARNING: NEAR-PERFECT ACCURACY (> {:.0%}) DETECTED".format(NEAR_PERFECT_THRESHOLD))
        print("!" * 75)
        print(
            "This is a RED FLAG, not a success.\n\n"
            "This dataset is template-generated synthetic data. When a linear\n"
            "TF-IDF model scores this high, the most likely explanation is that\n"
            "it MEMORIZED template vocabulary that leaks between the train and\n"
            "test splits - not that it learned to genuinely generalize.\n\n"
            "In-distribution accuracy here is therefore misleading. The number\n"
            "that actually matters is produced by:\n\n"
            "    python src/classification/generalization_test.py\n\n"
            "which tests the model on hand-written tickets phrased in words that\n"
            "never appear in the training templates."
        )
        print("!" * 75)
    else:
        print(
            "\nNote: accuracy is below the near-perfect threshold, but this is still\n"
            "IN-DISTRIBUTION performance. Run generalization_test.py to measure\n"
            "whether the model handles genuinely novel phrasing."
        )

    print("\nDone.")


if __name__ == "__main__":
    main()
