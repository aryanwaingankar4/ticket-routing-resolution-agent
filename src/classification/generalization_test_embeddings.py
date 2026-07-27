"""
generalization_test_embeddings.py

Embeddings-based counterpart to generalization_test.py.

Retrains the all-MiniLM-L6-v2 + LogisticRegression pipeline on the FULL
dataset, then evaluates on the SAME 14 hand-written novel tickets used by the
TF-IDF generalization test. This gives an apples-to-apples comparison against
the known 50% (7/14) TF-IDF baseline.

Run from the project root:
    python src/classification/generalization_test_embeddings.py
"""

import os
import sys
import random

import numpy as np
import pandas as pd

from sklearn.linear_model import LogisticRegression

# ---------------------------------------------------------------------------
# Reproducibility.
# ---------------------------------------------------------------------------
random.seed(42)
np.random.seed(42)

# ---------------------------------------------------------------------------
# Shared model constant - MUST be identical to train_embeddings.py.
# ---------------------------------------------------------------------------
MODEL_NAME = "all-MiniLM-L6-v2"

REQUIRED_COLUMNS = ["id", "title", "description", "category", "resolution", "priority"]
MIN_CATEGORIES = 7

# The TF-IDF baseline score to compare against (7/14 = 50.0%).
TFIDF_BASELINE_CORRECT = 7
TFIDF_BASELINE_TOTAL = 14

# ---------------------------------------------------------------------------
# 14 novel tickets, 2 per category, phrased the way a non-technical employee
# would describe the problem (no template jargon).
#
# NOTE: This list is copied VERBATIM from generalization_test.py so the 50%
# baseline and the new embeddings score are measured on identical inputs.
# If you edit the list in one file, edit it in the other or the comparison
# stops being fair.
# ---------------------------------------------------------------------------
NOVEL_TICKETS = [
    {"text": "I got a new laptop and now the system keeps saying my username "
             "or password is wrong even though I'm sure it's right. Can someone "
             "reset me so I can get back in?",
     "expected": "Access Management"},
    {"text": "My manager said I should be able to see the finance shared folder "
             "but it just says I'm not allowed. Can you give me permission to open it?",
     "expected": "Access Management"},
    {"text": "The wifi in the third floor meeting room keeps dropping every few "
             "minutes so we can't run our video calls. Everyone else in the "
             "building seems fine.",
     "expected": "Network"},
    {"text": "Pages take forever to load today and sometimes just time out. My "
             "colleague next to me has the same slowness on her machine too.",
     "expected": "Network"},
    {"text": "I keep getting a message that there's no room left to save my files "
             "and I can't download the report I need. It says the drive is full.",
     "expected": "Storage"},
    {"text": "I tried to save my presentation but it won't let me because it says "
             "I've run out of space in my folder. How do I free some up?",
     "expected": "Storage"},
    {"text": "When I try to pull up last month's customer records the whole thing "
             "just spins and then shows an error about not being able to reach the "
             "records system.",
     "expected": "Database"},
    {"text": "The report tool says it can't find the numbers it needs and mentions "
             "something about a broken connection to where the data is kept.",
     "expected": "Database"},
    {"text": "I got a weird email pretending to be from HR asking me to type in my "
             "password on a link, and I think I might have clicked it by mistake. "
             "What should I do?",
     "expected": "Security"},
    {"text": "My antivirus popped up a warning that something suspicious was blocked "
             "and now I'm worried my computer has a virus on it.",
     "expected": "Security"},
    {"text": "Every time I open the expense program it freezes on the loading screen "
             "and then closes itself. I've tried restarting but it keeps crashing.",
     "expected": "Application"},
    {"text": "The invoicing software gives me an error and shuts down whenever I "
             "click the print button. Nothing prints at all.",
     "expected": "Application"},
    {"text": "None of the company websites are working for anyone in the office this "
             "morning - it looks like one of the main servers might be down.",
     "expected": "Infrastructure"},
    {"text": "The whole team can't reach any of our internal tools and someone said "
             "a machine in the server room overheated and shut off overnight.",
     "expected": "Infrastructure"},
]

# ---------------------------------------------------------------------------
# Path resolution: project root is two directories up from this script.
# ---------------------------------------------------------------------------
def get_project_root():
    script_dir = os.path.dirname(os.path.abspath(__file__))
    return os.path.dirname(os.path.dirname(script_dir))


PROJECT_ROOT = get_project_root()
CSV_PATH = os.path.join(PROJECT_ROOT, "data", "synthetic_tickets.csv")


# ---------------------------------------------------------------------------
# Dependency guard.
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
# Model loading with network-error handling.
# ---------------------------------------------------------------------------
def load_model(SentenceTransformer):
    print(
        f"Loading model '{MODEL_NAME}'...\n"
        "(This may take a minute on first run while the model downloads "
        "(~80MB) from Hugging Face.)"
    )
    try:
        return SentenceTransformer(MODEL_NAME)
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
                "  3. Then retry - the model caches locally after the first success.\n\n"
                f"Original error: {exc}"
            )
            sys.exit(1)
        raise


# ---------------------------------------------------------------------------
# Validation gates (mirrors the existing scripts).
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

    return df


# ---------------------------------------------------------------------------
# Validate that every "expected" label matches a real training category BEFORE
# predicting anything (exactly like generalization_test.py).
# ---------------------------------------------------------------------------
def validate_expected_labels(df):
    training_categories = set(df["category"].unique())
    expected_labels = {t["expected"] for t in NOVEL_TICKETS}

    unknown = sorted(expected_labels - training_categories)
    if unknown:
        print(
            "\nERROR: Some NOVEL_TICKETS 'expected' labels do not match any "
            "training category. Fix the labels before running.\n"
        )
        print(f"  Training categories : {sorted(training_categories)}")
        print(f"  Unknown expected    : {unknown}")
        # Helpful diff of what each unknown label might have meant.
        print("\n  Mismatches:")
        for lbl in unknown:
            print(f"    - '{lbl}' is not in the training categories.")
        sys.exit(1)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    print("=" * 70)
    print("EMBEDDINGS GENERALIZATION TEST  -  all-MiniLM-L6-v2 + LogisticRegression")
    print("=" * 70)

    SentenceTransformer = import_sentence_transformers()

    df = load_and_validate_data()
    print(f"Loaded {len(df)} tickets across {df['category'].nunique()} categories.")

    validate_expected_labels(df)

    # Input text construction MUST match the TF-IDF scripts exactly.
    train_texts = (df["title"].astype(str) + " " + df["description"].astype(str)).tolist()
    y_train = df["category"].values

    model = load_model(SentenceTransformer)

    print(
        "\nEncoding the FULL training set... (this may take a minute on CPU on "
        "first run; a progress bar will appear below)"
    )
    X_train = model.encode(train_texts, show_progress_bar=True, convert_to_numpy=True)

    clf = LogisticRegression(max_iter=1000)
    clf.fit(X_train, y_train)

    # Encode the 14 novel tickets.
    novel_texts = [t["text"] for t in NOVEL_TICKETS]
    print("\nEncoding the 14 novel tickets...")
    X_novel = model.encode(novel_texts, show_progress_bar=True, convert_to_numpy=True)

    preds = clf.predict(X_novel)

    # -----------------------------------------------------------------------
    # Per-ticket breakdown (same structure as generalization_test.py).
    # -----------------------------------------------------------------------
    print("\n" + "=" * 70)
    print("PER-TICKET RESULTS")
    print("=" * 70)

    correct = 0
    for i, ticket in enumerate(NOVEL_TICKETS):
        expected = ticket["expected"]
        predicted = preds[i]
        is_correct = (predicted == expected)
        if is_correct:
            correct += 1
        status = "CORRECT" if is_correct else "WRONG  "
        print(f"[{status}] expected={expected:<10} predicted={predicted:<10}")
        print(f"          text: {ticket['text']}")

    total = len(NOVEL_TICKETS)
    pct = 100.0 * correct / total

    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print(f"Embeddings model score: {correct}/{total} ({pct:.1f}%)")

    # -----------------------------------------------------------------------
    # Explicit one-line comparison against the TF-IDF baseline.
    # -----------------------------------------------------------------------
    tfidf_pct = 100.0 * TFIDF_BASELINE_CORRECT / TFIDF_BASELINE_TOTAL
    print(
        f"\nEmbeddings model: {correct}/{total} ({pct:.1f}%) "
        f"vs TF-IDF baseline: {TFIDF_BASELINE_CORRECT}/{TFIDF_BASELINE_TOTAL} "
        f"({tfidf_pct:.1f}%)"
    )

    delta = pct - tfidf_pct
    direction = "improvement" if delta > 0 else ("regression" if delta < 0 else "no change")
    print(f"Delta: {delta:+.1f} percentage points ({direction}).")

    print("\nDone.")


if __name__ == "__main__":
    main()
