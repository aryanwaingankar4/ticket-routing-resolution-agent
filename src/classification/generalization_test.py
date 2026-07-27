"""
generalization_test.py
======================

TRUE generalization check for the baseline TF-IDF + Logistic Regression
classifier.

WHY THIS FILE IS THE NUMBER THAT ACTUALLY MATTERS
-------------------------------------------------
train_baseline_tfidf.py reports IN-DISTRIBUTION accuracy: it tests on a held-out
slice of the SAME synthetic, template-generated data it trained on. Because
template vocabulary tends to repeat, a purely lexical model (TF-IDF) can score
near-perfectly there just by memorizing which words belong to which category.
That is not evidence of generalization.

Real support tickets are written by non-technical employees who describe
problems in their own everyday words - "my computer won't let me in this
morning", not "authentication failure on primary domain controller". TF-IDF
matches on surface tokens, so if the novel wording shares few tokens with the
training templates, it has nothing to go on.

This script therefore:
  * retrains the SAME pipeline on the FULL synthetic dataset, and
  * evaluates it on a small set of HAND-WRITTEN novel tickets that deliberately
    avoid template jargon.

The gap between the near-perfect in-distribution score (File 1) and the score
here is the concrete, evidence-based justification for upgrading to a semantic-
embeddings classifier next.

Run from the project root:
    python src/classification/generalization_test.py
"""

import os
import sys

import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
EXPECTED_COLUMNS = ["id", "title", "description", "category", "resolution", "priority"]

# ---------------------------------------------------------------------------
# Hand-written novel test tickets.
#
# Each is phrased the way an ordinary, non-technical employee would actually
# describe the problem - NOT jargon copied from training templates. At least
# 1-2 per category. The "expected" label MUST match a training category string
# exactly (validated at runtime before any prediction is made).
# ---------------------------------------------------------------------------
NOVEL_TICKETS = [
    # --- Access Management ------------------------------------------------
    {
        "text": "I got a new laptop and now the system keeps saying my username "
                "or password is wrong even though I'm sure it's right. Can someone "
                "reset me so I can get back in?",
        "expected": "Access Management",
    },
    {
        "text": "My manager said I should be able to see the finance shared folder "
                "but it just says I'm not allowed. Can you give me permission to open it?",
        "expected": "Access Management",
    },
    # --- Network ----------------------------------------------------------
    {
        "text": "The wifi in the third floor meeting room keeps dropping every few "
                "minutes so we can't run our video calls. Everyone else in the "
                "building seems fine.",
        "expected": "Network",
    },
    {
        "text": "Pages take forever to load today and sometimes just time out. My "
                "colleague next to me has the same slowness on her machine too.",
        "expected": "Network",
    },
    # --- Storage ----------------------------------------------------------
    {
        "text": "I keep getting a message that there's no room left to save my files "
                "and I can't download the report I need. It says the drive is full.",
        "expected": "Storage",
    },
    {
        "text": "I tried to save my presentation but it won't let me because it says "
                "I've run out of space in my folder. How do I free some up?",
        "expected": "Storage",
    },
    # --- Database ---------------------------------------------------------
    {
        "text": "When I try to pull up last month's customer records the whole thing "
                "just spins and then shows an error about not being able to reach the "
                "records system.",
        "expected": "Database",
    },
    {
        "text": "The report tool says it can't find the numbers it needs and mentions "
                "something about a broken connection to where the data is kept.",
        "expected": "Database",
    },
    # --- Security ---------------------------------------------------------
    {
        "text": "I got a weird email pretending to be from HR asking me to type in my "
                "password on a link, and I think I might have clicked it by mistake. "
                "What should I do?",
        "expected": "Security",
    },
    {
        "text": "My antivirus popped up a warning that something suspicious was blocked "
                "and now I'm worried my computer has a virus on it.",
        "expected": "Security",
    },
    # --- Application ------------------------------------------------------
    {
        "text": "Every time I open the expense program it freezes on the loading screen "
                "and then closes itself. I've tried restarting but it keeps crashing.",
        "expected": "Application",
    },
    {
        "text": "The invoicing software gives me an error and shuts down whenever I "
                "click the print button. Nothing prints at all.",
        "expected": "Application",
    },
    # --- Infrastructure ---------------------------------------------------
    {
        "text": "None of the company websites are working for anyone in the office this "
                "morning - it looks like one of the main servers might be down.",
        "expected": "Infrastructure",
    },
    {
        "text": "The whole team can't reach any of our internal tools and someone said "
                "a machine in the server room overheated and shut off overnight.",
        "expected": "Infrastructure",
    },
]


# ---------------------------------------------------------------------------
# Path resolution (identical rule to train_baseline_tfidf.py)
# ---------------------------------------------------------------------------
def resolve_csv_path():
    """
    Resolve data/synthetic_tickets.csv relative to THIS script's own location,
    not the invocation directory. Project root is two levels up from this file.
    """
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.abspath(os.path.join(script_dir, "..", ".."))
    return os.path.join(project_root, "data", "synthetic_tickets.csv")


# ---------------------------------------------------------------------------
# Loading + validation (same guarantees as File 1)
# ---------------------------------------------------------------------------
def load_and_validate(csv_path):
    if not os.path.exists(csv_path):
        raise FileNotFoundError(
            f"\nDataset not found at:\n    {csv_path}\n\n"
            "This script expects data/synthetic_tickets.csv to already exist.\n"
            "Run the dataset generator first, then re-run this file."
        )

    try:
        df = pd.read_csv(csv_path)
    except Exception as exc:  # noqa: BLE001
        raise RuntimeError(
            f"Failed to read CSV at {csv_path}. Underlying error: {exc}"
        ) from exc

    missing_columns = [c for c in EXPECTED_COLUMNS if c not in df.columns]
    if missing_columns:
        print("ERROR: The dataset is missing required column(s):")
        for col in missing_columns:
            print(f"    - {col}")
        print(f"\nExpected columns: {EXPECTED_COLUMNS}")
        print(f"Found columns:    {list(df.columns)}")
        sys.exit(1)

    used_columns = ["title", "description", "category"]
    nan_report = {col: int(df[col].isna().sum()) for col in used_columns}
    if any(count > 0 for count in nan_report.values()):
        print("ERROR: NaN / missing values found in required columns:")
        for col, count in nan_report.items():
            if count > 0:
                print(f"    - {col}: {count} missing value(s)")
        print("\nClean these rows in the dataset before running.")
        sys.exit(1)

    return df


def validate_expected_labels(training_categories):
    """
    CRITICAL: every expected label in NOVEL_TICKETS must EXACTLY match (case-
    sensitive) a category that appears in the training data. A silent typo would
    make a correct prediction look "WRONG" and waste debugging time, so we fail
    loudly and list every mismatch before predicting anything.
    """
    training_set = set(training_categories)
    mismatches = []
    for i, ticket in enumerate(NOVEL_TICKETS):
        if ticket["expected"] not in training_set:
            mismatches.append((i, ticket["expected"]))

    if mismatches:
        print("ERROR: One or more expected labels do not match any training category.")
        print("       (Comparison is EXACT and case-sensitive.)\n")
        print("Valid training categories:")
        for cat in sorted(training_set):
            print(f"    - {cat!r}")
        print("\nMismatched expected labels in NOVEL_TICKETS:")
        for idx, bad in mismatches:
            print(f"    - ticket index {idx}: {bad!r}")
        sys.exit(1)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    csv_path = resolve_csv_path()
    print("=" * 75)
    print("GENERALIZATION TEST: novel, non-template phrasing")
    print("=" * 75)
    print(f"Dataset: {csv_path}\n")

    df = load_and_validate(csv_path)

    # Same input construction as File 1: title + description.
    X_text = (df["title"].astype(str) + " " + df["description"].astype(str))
    y = df["category"].astype(str)

    training_categories = sorted(y.unique().tolist())
    print(f"Trained on {len(df)} tickets across {len(training_categories)} categories:")
    for cat in training_categories:
        print(f"    - {cat}")
    print()

    # Validate the hand-written labels BEFORE training/predicting.
    validate_expected_labels(training_categories)

    # -----------------------------------------------------------------------
    # Retrain the SAME pipeline as File 1, but fit on the FULL dataset. The
    # novel hand-written tickets are the true held-out test set here.
    # -----------------------------------------------------------------------
    vectorizer = TfidfVectorizer(
        max_features=5000,
        ngram_range=(1, 2),
        stop_words="english",
    )
    X_vec = vectorizer.fit_transform(X_text)

    clf = LogisticRegression(max_iter=1000)
    clf.fit(X_vec, y)

    # -----------------------------------------------------------------------
    # Predict on the novel tickets.
    # -----------------------------------------------------------------------
    novel_texts = [t["text"] for t in NOVEL_TICKETS]
    novel_expected = [t["expected"] for t in NOVEL_TICKETS]

    novel_vec = vectorizer.transform(novel_texts)
    novel_pred = clf.predict(novel_vec)

    print("-" * 75)
    print("PER-TICKET RESULTS")
    print("-" * 75)

    correct = 0
    for i, (text, expected, predicted) in enumerate(
        zip(novel_texts, novel_expected, novel_pred), start=1
    ):
        is_correct = (predicted == expected)
        if is_correct:
            correct += 1
        marker = "CORRECT" if is_correct else "WRONG"

        print(f"\n[{i:>2}] {marker}")
        print(f"     Expected  : {expected}")
        print(f"     Predicted : {predicted}")
        print(f"     Ticket    : {text}")

    # -----------------------------------------------------------------------
    # Summary + interpretation.
    # -----------------------------------------------------------------------
    total = len(NOVEL_TICKETS)
    pct = (correct / total * 100.0) if total else 0.0

    print("\n" + "=" * 75)
    print("SUMMARY")
    print("=" * 75)
    print(f"Correct: {correct}/{total}  ({pct:.1f}%)")

    print("\nINTERPRETATION")
    print("-" * 75)
    print(
        "Compare this score against the near-perfect IN-DISTRIBUTION accuracy\n"
        "reported by train_baseline_tfidf.py.\n\n"
        "If the score here is substantially LOWER than that in-distribution\n"
        "number, that gap is exactly the point: TF-IDF is a purely lexical\n"
        "(keyword-matching) model. It scores high on held-out template data\n"
        "because that data reuses the same vocabulary it memorized, but it\n"
        "struggles on these hand-written tickets because non-technical users\n"
        "describe problems in words that never appear in the training templates.\n\n"
        "TF-IDF has no notion of MEANING - 'can't log in' and 'unable to\n"
        "authenticate' look unrelated to it because they share no tokens.\n\n"
        "This is the concrete, evidence-based justification for the next step:\n"
        "a semantic-embeddings classifier (e.g. sentence-transformers), which\n"
        "represents tickets by meaning rather than surface words and should\n"
        "close this generalization gap. The baseline's job was to give us this\n"
        "number to beat - not to be the final classifier."
    )

    print("\nDone.")


if __name__ == "__main__":
    main()
