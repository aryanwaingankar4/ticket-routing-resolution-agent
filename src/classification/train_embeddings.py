"""
train_embeddings.py

Embeddings-based counterpart to train_baseline_tfidf.py.

Replaces the TF-IDF representation with sentence-transformer embeddings
("all-MiniLM-L6-v2") while keeping everything else identical:
    - input text = title + " " + description
    - LogisticRegression(max_iter=1000) on top
    - train_test_split(test_size=0.2, random_state=42, stratify=y)

This isolates the effect of the *representation* (embeddings vs TF-IDF).

Run from the project root:
    python src/classification/train_embeddings.py
"""

import os
import sys
import random

import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    classification_report,
    confusion_matrix,
)

# ---------------------------------------------------------------------------
# Reproducibility (embeddings are deterministic given a fixed model, but this
# keeps any downstream sampling reproducible - matches the dataset generator).
# ---------------------------------------------------------------------------
random.seed(42)
np.random.seed(42)

# ---------------------------------------------------------------------------
# Shared model constant. Hardcoded so this file and the generalization test
# are GUARANTEED to use the identical model. A mismatch here would silently
# invalidate the comparison.
# ---------------------------------------------------------------------------
MODEL_NAME = "all-MiniLM-L6-v2"

REQUIRED_COLUMNS = ["id", "title", "description", "category", "resolution", "priority"]
MIN_CATEGORIES = 7


# ---------------------------------------------------------------------------
# Path resolution: project root is two directories up from this script.
# Works correctly on Windows because we use os.path.* throughout.
# ---------------------------------------------------------------------------
def get_project_root():
    script_dir = os.path.dirname(os.path.abspath(__file__))
    # src/classification/ -> src/ -> project root
    return os.path.dirname(os.path.dirname(script_dir))


PROJECT_ROOT = get_project_root()
CSV_PATH = os.path.join(PROJECT_ROOT, "data", "synthetic_tickets.csv")
CACHE_PATH = os.path.join(PROJECT_ROOT, "data", "ticket_embeddings.npy")


# ---------------------------------------------------------------------------
# Dependency guard: catch a missing sentence-transformers install and give an
# actionable message instead of a raw traceback.
# ---------------------------------------------------------------------------
def import_sentence_transformers():
    try:
        from sentence_transformers import SentenceTransformer  # noqa: F401
        return SentenceTransformer
    except ImportError:
        print(
            "\nERROR: The 'sentence-transformers' package is not installed.\n"
            "Install it with:\n\n"
            "    pip install sentence-transformers\n\n"
            "Then re-run this script."
        )
        sys.exit(1)


# ---------------------------------------------------------------------------
# Model loading: first run downloads from Hugging Face. Catch connection /
# timeout errors specifically and give an actionable message.
# ---------------------------------------------------------------------------
def load_model(SentenceTransformer):
    print(
        f"Loading model '{MODEL_NAME}'...\n"
        "(This may take a minute on first run while the model downloads "
        "(~80MB) from Hugging Face.)"
    )
    try:
        model = SentenceTransformer(MODEL_NAME)
        return model
    except Exception as exc:  # noqa: BLE001
        msg = str(exc).lower()
        network_markers = (
            "connection",
            "timeout",
            "timed out",
            "network",
            "unreachable",
            "temporary failure",
            "max retries",
            "failed to establish",
            "name resolution",
            "getaddrinfo",
            "ssl",
            "proxy",
            "huggingface.co",
            "connectionerror",
            "readtimeout",
        )
        if any(marker in msg for marker in network_markers):
            print(
                "\nERROR: Could not download / load the model from Hugging Face.\n"
                "This looks like a network problem. Please check:\n"
                "  1. Your internet connection is working.\n"
                "  2. A corporate proxy or firewall is not blocking huggingface.co.\n"
                "     (If behind a proxy, set HTTP_PROXY / HTTPS_PROXY env vars.)\n"
                "  3. Then retry - the download resumes / caches locally after "
                "the first success.\n\n"
                f"Original error: {exc}"
            )
            sys.exit(1)
        # Not obviously a network error - re-raise so real bugs are visible.
        raise


# ---------------------------------------------------------------------------
# Validation gates (mirrors train_baseline_tfidf.py for consistency).
# ---------------------------------------------------------------------------
def load_and_validate_data():
    if not os.path.exists(CSV_PATH):
        print(
            f"\nERROR: Could not find the dataset CSV at:\n    {CSV_PATH}\n\n"
            "Make sure data/synthetic_tickets.csv exists and that you are "
            "running this script from the project root."
        )
        sys.exit(1)

    df = pd.read_csv(CSV_PATH)

    missing_cols = [c for c in REQUIRED_COLUMNS if c not in df.columns]
    if missing_cols:
        print(
            "\nERROR: The dataset is missing required columns: "
            f"{missing_cols}\n"
            f"Expected columns: {REQUIRED_COLUMNS}\n"
            f"Found columns:    {list(df.columns)}"
        )
        sys.exit(1)

    for col in ["title", "description", "category"]:
        if df[col].isna().any():
            n_bad = int(df[col].isna().sum())
            print(
                f"\nERROR: Column '{col}' contains {n_bad} NaN/empty value(s). "
                "Please clean the dataset before training."
            )
            sys.exit(1)

    n_categories = df["category"].nunique()
    if n_categories < MIN_CATEGORIES:
        print(
            f"\nERROR: Expected at least {MIN_CATEGORIES} categories, but found "
            f"only {n_categories}: {sorted(df['category'].unique())}"
        )
        sys.exit(1)

    # Enough rows per class to stratify a 20% test split (>= 2 per class,
    # so at least one row lands in each of train and test).
    class_counts = df["category"].value_counts()
    too_small = class_counts[class_counts < 2]
    if len(too_small) > 0:
        print(
            "\nERROR: Every category needs at least 2 rows to stratify the "
            "train/test split. These categories have too few:\n"
            f"{too_small.to_string()}"
        )
        sys.exit(1)

    return df


# ---------------------------------------------------------------------------
# Embedding cache: reuse data/ticket_embeddings.npy only if it exists AND has
# the same number of rows as the current CSV. Otherwise recompute + overwrite.
# ---------------------------------------------------------------------------
def get_full_embeddings(model, texts):
    n_rows = len(texts)

    if os.path.exists(CACHE_PATH):
        try:
            cached = np.load(CACHE_PATH)
        except Exception as exc:  # noqa: BLE001
            print(f"[cache] Failed to read cache ({exc}); recomputing.")
            cached = None

        if cached is not None and cached.shape[0] == n_rows:
            print(
                f"[cache HIT] Reusing cached embeddings from:\n    {CACHE_PATH}\n"
                f"    ({cached.shape[0]} rows, dim={cached.shape[1]})"
            )
            return cached
        elif cached is not None:
            print(
                f"[cache MISS] Cache has {cached.shape[0]} rows but the CSV has "
                f"{n_rows} rows (dataset changed). Recomputing and overwriting."
            )
    else:
        print("[cache MISS] No cache found. Computing embeddings from scratch.")

    print(
        "Encoding tickets... (this may take a minute on CPU; a progress bar "
        "will appear below)"
    )
    embeddings = model.encode(
        texts,
        show_progress_bar=True,
        convert_to_numpy=True,
    )

    try:
        np.save(CACHE_PATH, embeddings)
        print(f"[cache] Saved embeddings to:\n    {CACHE_PATH}")
    except Exception as exc:  # noqa: BLE001
        print(f"[cache] WARNING: could not save cache ({exc}); continuing.")

    return embeddings


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    print("=" * 70)
    print("EMBEDDINGS BASELINE  -  all-MiniLM-L6-v2 + LogisticRegression")
    print("=" * 70)

    SentenceTransformer = import_sentence_transformers()

    df = load_and_validate_data()
    print(f"Loaded {len(df)} tickets across {df['category'].nunique()} categories.")

    # Input text construction MUST match the TF-IDF scripts exactly.
    texts = (df["title"].astype(str) + " " + df["description"].astype(str)).tolist()
    y = df["category"].astype(str).to_numpy()

    model = load_model(SentenceTransformer)

    # Compute embeddings for the FULL dataset (with caching), then split the
    # embedding rows with the IDENTICAL split call as the TF-IDF baseline so
    # the same tickets land in train/test.
    X = get_full_embeddings(model, texts)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    print(f"\nTrain rows: {len(X_train)}   Test rows: {len(X_test)}")

    clf = LogisticRegression(max_iter=1000)
    clf.fit(X_train, y_train)

    y_pred = clf.predict(X_test)

    # -----------------------------------------------------------------------
    # Metrics (same structure as train_baseline_tfidf.py).
    # -----------------------------------------------------------------------
    acc = accuracy_score(y_test, y_pred)
    macro_f1 = f1_score(y_test, y_pred, average="macro")
    weighted_f1 = f1_score(y_test, y_pred, average="weighted")

    print("\n" + "=" * 70)
    print("RESULTS (in-distribution test split)")
    print("=" * 70)
    print(f"Accuracy    : {acc:.4f}")
    print(f"Macro F1    : {macro_f1:.4f}")
    print(f"Weighted F1 : {weighted_f1:.4f}")

    print("\nClassification report:")
    print(classification_report(y_test, y_pred, zero_division=0))

    labels = sorted(np.unique(y))
    cm = confusion_matrix(y_test, y_pred, labels=labels)

    print("Confusion matrix (rows = true, cols = predicted):")
    col_width = max(12, max(len(str(l)) for l in labels) + 2)
    header = " " * col_width + "".join(f"{str(l)[:col_width-1]:>{col_width}}" for l in labels)
    print(header)
    for i, true_label in enumerate(labels):
        row = f"{str(true_label)[:col_width-1]:<{col_width}}"
        row += "".join(f"{cm[i, j]:>{col_width}}" for j in range(len(labels)))
        print(row)

    print("\nDone.")


if __name__ == "__main__":
    main()
