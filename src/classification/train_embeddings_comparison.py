"""
train_embeddings_comparison.py
==============================

Back-to-back comparison of THREE frozen sentence-transformer embedding models,
each feeding an identical scikit-learn LogisticRegression classifier, to test
whether a newer frozen embedding model beats the current MiniLM ceiling of
10/14 (71.4%) on the 14-ticket generalization benchmark.

Models compared:
    1. "all-MiniLM-L6-v2"       (384-dim)  -- baseline / same-session reference
    2. "BAAI/bge-base-en-v1.5"  (768-dim)  -- candidate A
    3. "intfloat/e5-base-v2"    (768-dim)  -- candidate B

Everything downstream of the embedding step is held IDENTICAL to
train_embeddings.py so this isolates the effect of the *representation*:
    - input text     = title + " " + description
    - split          = train_test_split(test_size=0.2, random_state=42, stratify=y)
    - classifier     = LogisticRegression(max_iter=1000)
    - seeds          = random.seed(42) / np.random.seed(42)

E5 PREFIX: intfloat/e5-base-v2 REQUIRES "query: " prepended to every text
(train + benchmark). MiniLM and BGE are encoded as plain text. See the
apply_prefix() helper and the two call sites flagged with `E5 PREFIX` comments.

BENCHMARKS: each model is evaluated against TWO benchmarks using the SAME
fitted classifier (no extra training run):
    1. The original 14-ticket NOVEL_TICKETS (read-only from
       generalization_test.py).
    2. The 46-ticket expanded benchmark (14 original + 32 Gemini-generated,
       self-consistency-verified), loaded from
       data/novel_tickets_expanded.json.
Both are reported side by side so it's easy to see whether a model's ranking
holds, tightens, or shifts at the larger sample size, and whether the 46-
ticket set reveals new failure patterns beyond the original 14.

CPU-only. Fully local/offline: sentence-transformers + scikit-learn only.
No Gemini / no API calls anywhere.

Run from the project root:
    python src/classification/train_embeddings_comparison.py
"""

import os
import sys
import random
import json

import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

# ---------------------------------------------------------------------------
# Reproducibility -- identical to train_embeddings.py.
# ---------------------------------------------------------------------------
random.seed(42)
np.random.seed(42)

# ---------------------------------------------------------------------------
# Split / classifier config -- MIRRORED EXACTLY from train_embeddings.py.
# Do NOT change these between models; that is the whole point of the comparison.
# ---------------------------------------------------------------------------
TEST_SIZE = 0.2
RANDOM_STATE = 42
LOGREG_MAX_ITER = 1000

REQUIRED_COLUMNS = ["id", "title", "description", "category", "resolution", "priority"]
MIN_CATEGORIES = 7

# ---------------------------------------------------------------------------
# Model registry. Each entry declares its cache filename and whether the E5
# "query: " prefix must be applied. Only E5 gets the prefix.
# ---------------------------------------------------------------------------
MODELS = [
    {
        "key": "minilm",
        "name": "all-MiniLM-L6-v2",
        "cache": "embeddings_minilm.npy",
        "e5_prefix": False,
    },
    {
        "key": "bge",
        "name": "BAAI/bge-base-en-v1.5",
        "cache": "embeddings_bge.npy",
        "e5_prefix": False,
    },
    {
        "key": "e5",
        "name": "intfloat/e5-base-v2",
        "cache": "embeddings_e5.npy",
        "e5_prefix": True,  # <-- E5 PREFIX: this model needs "query: " prepended
    },
]

E5_PREFIX = "query: "


# ---------------------------------------------------------------------------
# Path resolution: project root is two directories up from this script.
# os.path.* throughout for Windows / cross-platform safety.
# ---------------------------------------------------------------------------
def get_project_root():
    script_dir = os.path.dirname(os.path.abspath(__file__))
    # src/classification/ -> src/ -> project root
    return os.path.dirname(os.path.dirname(script_dir))


PROJECT_ROOT = get_project_root()
CSV_PATH = os.path.join(PROJECT_ROOT, "data", "synthetic_tickets.csv")
OUTPUT_DIR = os.path.join(PROJECT_ROOT, "data", "embedding_comparison")
RESULTS_CSV = os.path.join(OUTPUT_DIR, "embedding_model_comparison.csv")

# NEW: 46-ticket expanded benchmark (14 original + 32 Gemini-generated,
# self-consistency-verified). Same shape as NOVEL_TICKETS:
# [{"text": "...", "expected": "<category>"}, ...]
EXPANDED_BENCH_PATH = os.path.join(
    PROJECT_ROOT, "data", "novel_tickets_expanded.json"
)
EXPANDED_BENCH_EXPECTED_N = 45

# Make src/classification/ importable so we can pull NOVEL_TICKETS read-only.
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
if SCRIPT_DIR not in sys.path:
    sys.path.insert(0, SCRIPT_DIR)


# ---------------------------------------------------------------------------
# Import NOVEL_TICKETS READ-ONLY from generalization_test.py. We deliberately
# do not re-declare the 14 tickets here so there is a single source of truth.
# ---------------------------------------------------------------------------
def import_novel_tickets():
    try:
        from generalization_test import NOVEL_TICKETS
    except Exception as exc:  # noqa: BLE001
        print(
            "\nERROR: Could not import NOVEL_TICKETS from generalization_test.py.\n"
            f"Looked in: {SCRIPT_DIR}\n"
            f"Underlying error: {exc}"
        )
        sys.exit(1)
    return NOVEL_TICKETS


# ---------------------------------------------------------------------------
# Load the 46-ticket expanded benchmark from disk (flat JSON list). Validates
# entry count and shape here; label-vs-training-category validation happens in
# main() alongside the existing NOVEL_TICKETS check, so both benchmarks are
# guarded the same way against the SAME training_set. Fails loudly (same style
# as the NOVEL_TICKETS validation) on any problem.
# ---------------------------------------------------------------------------
def load_expanded_benchmark():
    if not os.path.exists(EXPANDED_BENCH_PATH):
        print(
            "\nERROR: Could not find the expanded benchmark at:\n"
            f"    {EXPANDED_BENCH_PATH}\n\n"
            "Make sure data/novel_tickets_expanded.json exists and that you "
            "are running this script from the project root."
        )
        sys.exit(1)

    try:
        with open(EXPANDED_BENCH_PATH, "r", encoding="utf-8") as fh:
            data = json.load(fh)
    except Exception as exc:  # noqa: BLE001
        print(
            "\nERROR: Could not parse data/novel_tickets_expanded.json as "
            f"JSON.\nUnderlying error: {exc}"
        )
        sys.exit(1)

    if not isinstance(data, list):
        print(
            "\nERROR: data/novel_tickets_expanded.json must be a flat JSON "
            f"list, but got: {type(data).__name__}"
        )
        sys.exit(1)

    if len(data) != EXPANDED_BENCH_EXPECTED_N:
        print(
            "\nERROR: Expected exactly "
            f"{EXPANDED_BENCH_EXPECTED_N} entries in "
            "data/novel_tickets_expanded.json (14 original + 32 new), but "
            f"found {len(data)}."
        )
        sys.exit(1)

    for i, entry in enumerate(data):
        if not isinstance(entry, dict):
            print(
                f"\nERROR: Entry at index {i} in the expanded benchmark is "
                f"not an object: {entry!r}"
            )
            sys.exit(1)
        for key in ("text", "expected"):
            if key not in entry:
                print(
                    f"\nERROR: Entry at index {i} in the expanded benchmark "
                    f"is missing required key {key!r}: {entry!r}"
                )
                sys.exit(1)

    return data


# ---------------------------------------------------------------------------
# Dependency guard for sentence-transformers.
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
# Data loading + validation -- mirrors train_embeddings.py.
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
# E5 PREFIX helper. Applied consistently to BOTH training texts AND benchmark
# texts, and ONLY for models whose registry entry has e5_prefix=True.
# ---------------------------------------------------------------------------
def apply_prefix(texts, use_e5_prefix):
    """Return texts, optionally with the E5 "query: " prefix prepended.

    This single helper is the ONLY place the prefix is added, so train-time and
    eval-time encoding are guaranteed to be treated identically.
    """
    if use_e5_prefix:
        return [E5_PREFIX + t for t in texts]
    return list(texts)


# ---------------------------------------------------------------------------
# Per-model embedding cache. Each model writes to its OWN file; caches are
# never shared across models (Day-8 imbalance-sweep convention). We also guard
# on row count so a stale cache from a different-sized CSV is not silently used.
# ---------------------------------------------------------------------------
def get_or_compute_embeddings(model, texts, cache_path, n_rows_expected):
    if os.path.exists(cache_path):
        try:
            cached = np.load(cache_path)
        except Exception as exc:  # noqa: BLE001
            print(f"    [cache] Failed to read cache ({exc}); recomputing.")
            cached = None

        if cached is not None and cached.shape[0] == n_rows_expected:
            print(
                f"    [cache HIT] Reusing embeddings: {os.path.basename(cache_path)} "
                f"({cached.shape[0]} rows, dim={cached.shape[1]})"
            )
            return cached
        elif cached is not None:
            print(
                f"    [cache MISS] Cache has {cached.shape[0]} rows but CSV has "
                f"{n_rows_expected} (dataset changed). Recomputing + overwriting."
            )
    else:
        print(f"    [cache MISS] No cache at {os.path.basename(cache_path)}. "
              "Computing fresh.")

    print("    Encoding on CPU (progress bar below; 768-dim models take a few "
          "minutes)...")
    embeddings = model.encode(
        texts,
        show_progress_bar=True,   # requirement 5: visible progress, not hung
        convert_to_numpy=True,
        batch_size=32,
    )

    try:
        np.save(cache_path, embeddings)
        print(f"    [cache] Saved: {cache_path}")
    except Exception as exc:  # noqa: BLE001
        print(f"    [cache] WARNING: could not save cache ({exc}); continuing.")

    return embeddings


# ---------------------------------------------------------------------------
# Run ONE model end-to-end: load, encode (cached), split, train, evaluate
# BOTH the in-distribution test split AND the 14-ticket AND 46-ticket
# benchmarks (same fitted classifier, no extra training run).
#
# Returns a dict of results, or raises on failure (caught by the caller so one
# bad model does not block the others).
# ---------------------------------------------------------------------------
def run_one_model(spec, SentenceTransformer, df, texts, y,
                  novel_texts, novel_expected,
                  bench46_texts, bench46_expected):
    name = spec["name"]
    use_e5_prefix = spec["e5_prefix"]
    cache_path = os.path.join(OUTPUT_DIR, spec["cache"])

    print("\n" + "=" * 72)
    print(f"MODEL: {name}   (E5 prefix: {'YES' if use_e5_prefix else 'no'})")
    print("=" * 72)

    # --- load model --------------------------------------------------------
    print(f"    Loading '{name}' (first run downloads from Hugging Face)...")
    model = SentenceTransformer(name)

    # Current, non-deprecated dimension accessor (avoid get_sentence_embedding_
    # dimension() FutureWarning). Fall back gracefully if the running version
    # is older and only exposes the deprecated name.
    if hasattr(model, "get_embedding_dimension"):
        embedding_dim = model.get_embedding_dimension()
    else:  # older sentence-transformers
        embedding_dim = model.get_sentence_embedding_dimension()

    # --- encode training corpus (with cache) -------------------------------
    # E5 PREFIX (train side): prefix is applied HERE, before encoding, only when
    # use_e5_prefix is True. Same helper is reused for the benchmarks below, so
    # train and eval are always treated identically for a given model.
    encode_texts = apply_prefix(texts, use_e5_prefix)
    X = get_or_compute_embeddings(model, encode_texts, cache_path, len(texts))

    if X.shape[1] != embedding_dim:
        print(f"    [note] Reported dim={embedding_dim}, cached dim={X.shape[1]}; "
              "using cached array's actual dimension for the results table.")
        embedding_dim = X.shape[1]

    # --- identical split to train_embeddings.py ----------------------------
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=y
    )
    print(f"    Train rows: {len(X_train)}   Test rows: {len(X_test)}")

    # --- fresh classifier, identical hyperparameters -----------------------
    clf = LogisticRegression(max_iter=LOGREG_MAX_ITER)
    # NOTE (per task requirement 1): the benchmark is scored by the SAME
    # classifier trained on the 0.2 train split (NOT retrained on the full
    # dataset the way the old generalization_test.py does). If you would rather
    # benchmark a full-dataset fit, fit a second clf on (X, y) here instead.
    clf.fit(X_train, y_train)

    # --- in-distribution accuracy ------------------------------------------
    y_pred = clf.predict(X_test)
    in_dist_acc = accuracy_score(y_test, y_pred)
    print(f"    In-distribution test accuracy: {in_dist_acc:.4f}")

    # --- 14-ticket benchmark -----------------------------------------------
    # E5 PREFIX (eval side): SAME helper, SAME flag -> guaranteed consistent
    # with the training-side prefixing above.
    novel_encode_texts = apply_prefix(novel_texts, use_e5_prefix)
    novel_X = model.encode(
        novel_encode_texts,
        show_progress_bar=False,
        convert_to_numpy=True,
    )
    novel_pred = clf.predict(novel_X)

    misclassified = []
    correct = 0
    for i, (text, expected, predicted) in enumerate(
        zip(novel_texts, novel_expected, novel_pred)
    ):
        if predicted == expected:
            correct += 1
        else:
            misclassified.append(
                {"index": i, "text": text, "expected": expected,
                 "predicted": predicted}
            )

    total = len(novel_texts)
    bench_pct = (correct / total * 100.0) if total else 0.0
    print(f"    Benchmark: {correct}/{total} correct ({bench_pct:.1f}%)")

    # --- 46-ticket expanded benchmark ---------------------------------------
    # SAME fitted `clf`, SAME `model` instance -- this is an ADDITIONAL
    # evaluation pass, NOT a second training run. E5 PREFIX (eval side): reuse
    # the SAME apply_prefix helper + the SAME use_e5_prefix flag so the 46-
    # ticket set is prefixed identically to the training corpus and the
    # 14-ticket set above.
    bench46_encode_texts = apply_prefix(bench46_texts, use_e5_prefix)
    bench46_X = model.encode(
        bench46_encode_texts,
        show_progress_bar=False,
        convert_to_numpy=True,
    )
    bench46_pred = clf.predict(bench46_X)

    bench46_misclassified = []
    bench46_correct = 0
    for i, (text, expected, predicted) in enumerate(
        zip(bench46_texts, bench46_expected, bench46_pred)
    ):
        if predicted == expected:
            bench46_correct += 1
        else:
            bench46_misclassified.append(
                {"index": i, "text": text, "expected": expected,
                 "predicted": predicted}
            )

    bench46_total = len(bench46_texts)
    bench46_pct = (bench46_correct / bench46_total * 100.0) if bench46_total else 0.0
    print(f"    Benchmark (46): {bench46_correct}/{bench46_total} correct "
          f"({bench46_pct:.1f}%)")

    return {
        "model_name": name,
        "embedding_dim": int(embedding_dim),
        "in_distribution_accuracy": round(float(in_dist_acc), 4),
        "benchmark_correct_count": int(correct),
        "benchmark_total": int(total),
        "benchmark_accuracy_pct": round(float(bench_pct), 1),
        "misclassified": misclassified,
        "bench46_correct_count": int(bench46_correct),
        "bench46_total": int(bench46_total),
        "bench46_accuracy_pct": round(float(bench46_pct), 1),
        "bench46_misclassified": bench46_misclassified,
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    print("=" * 72)
    print("EMBEDDING MODEL COMPARISON  -  3 models x LogisticRegression(max_iter=1000)")
    print("CPU-only | offline | sentence-transformers + scikit-learn")
    print("=" * 72)

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    SentenceTransformer = import_sentence_transformers()
    NOVEL_TICKETS = import_novel_tickets()
    EXPANDED_BENCHMARK = load_expanded_benchmark()

    df = load_and_validate_data()
    print(f"Loaded {len(df)} tickets across {df['category'].nunique()} categories.")

    # Input text + labels -- IDENTICAL construction to train_embeddings.py.
    texts = (df["title"].astype(str) + " " + df["description"].astype(str)).tolist()
    y = df["category"].astype(str).to_numpy()

    # Benchmark tickets (read-only from generalization_test.py).
    novel_texts = [t["text"] for t in NOVEL_TICKETS]
    novel_expected = [t["expected"] for t in NOVEL_TICKETS]

    # Expanded 46-ticket benchmark (read-only from disk).
    bench46_texts = [t["text"] for t in EXPANDED_BENCHMARK]
    bench46_expected = [t["expected"] for t in EXPANDED_BENCHMARK]

    # Sanity: every benchmark label must exist as a training category, or a
    # correct prediction could be scored WRONG. Fail loudly if not.
    training_set = set(np.unique(y).tolist())
    bad_labels = sorted({e for e in novel_expected if e not in training_set})
    if bad_labels:
        print(
            "\nERROR: These NOVEL_TICKETS 'expected' labels are not training "
            f"categories (exact, case-sensitive): {bad_labels}\n"
            f"Training categories: {sorted(training_set)}"
        )
        sys.exit(1)

    # SAME check for the 46-ticket set, against the SAME training_set. Fails
    # loudly in the same style if any expected label is not a known category.
    bad_labels_46 = sorted({e for e in bench46_expected if e not in training_set})
    if bad_labels_46:
        print(
            "\nERROR: These novel_tickets_expanded.json 'expected' labels are "
            f"not training categories (exact, case-sensitive): {bad_labels_46}\n"
            f"Training categories: {sorted(training_set)}"
        )
        sys.exit(1)

    results = []
    skipped = []

    for spec in MODELS:
        try:
            res = run_one_model(
                spec, SentenceTransformer, df, texts, y,
                novel_texts, novel_expected,
                bench46_texts, bench46_expected,
            )
            results.append(res)
        except Exception as exc:  # noqa: BLE001
            # Requirement 9: never let one bad model block the others.
            print(f"\n    ERROR while processing '{spec['name']}': {exc}")
            print("    Skipping this model and continuing with the next one.")
            skipped.append({"model_name": spec["name"], "error": str(exc)})

    # -----------------------------------------------------------------------
    # Per-model misclassification detail (requirement 7), now for BOTH
    # benchmarks, clearly labeled.
    # -----------------------------------------------------------------------
    print("\n" + "=" * 72)
    print("PER-MODEL MISCLASSIFIED BENCHMARK TICKETS")
    print("=" * 72)
    for res in results:
        print(f"\n### {res['model_name']}  "
              f"({res['benchmark_correct_count']}/{res['benchmark_total']} correct)")

        print("    --- 14-ticket benchmark misses ---")
        if not res["misclassified"]:
            print("    (none -- all 14 correct)")
        else:
            for m in res["misclassified"]:
                print(f"    [idx {m['index']:>2}] expected={m['expected']!r}  "
                      f"predicted={m['predicted']!r}")
                print(f"             text: {m['text']}")

        print(f"    --- 46-ticket benchmark misses ---  "
              f"({res['bench46_correct_count']}/{res['bench46_total']} correct)")
        if not res["bench46_misclassified"]:
            print("    (none -- all 46 correct)")
        else:
            for m in res["bench46_misclassified"]:
                print(f"    [idx {m['index']:>2}] expected={m['expected']!r}  "
                      f"predicted={m['predicted']!r}")
                print(f"             text: {m['text']}")

    # -----------------------------------------------------------------------
    # Comparison table -> terminal + CSV (requirement 6), now with both
    # benchmarks side by side.
    # -----------------------------------------------------------------------
    print("\n" + "=" * 72)
    print("COMPARISON TABLE")
    print("=" * 72)

    if not results:
        print("No models completed successfully; nothing to tabulate.")
    else:
        table_cols = [
            "model_name",
            "embedding_dim",
            "in_distribution_accuracy",
            "benchmark_correct_count",
            "benchmark_accuracy_pct",
            "bench46_correct_count",
            "bench46_accuracy_pct",
        ]
        table_df = pd.DataFrame([{c: r[c] for c in table_cols} for r in results])

        # Pretty terminal print (fixed-width, no external deps).
        name_w = max(len("model_name"),
                     max(len(r["model_name"]) for r in results))
        header = (f"{'model_name':<{name_w}}  {'dim':>5}  "
                  f"{'in_dist_acc':>11}  {'b14_ok':>8}  {'b14_%':>7}  "
                  f"{'b46_ok':>8}  {'b46_%':>7}")
        print(header)
        print("-" * len(header))
        for r in results:
            print(f"{r['model_name']:<{name_w}}  "
                  f"{r['embedding_dim']:>5}  "
                  f"{r['in_distribution_accuracy']:>11.4f}  "
                  f"{r['benchmark_correct_count']:>4}/{r['benchmark_total']:<3}  "
                  f"{r['benchmark_accuracy_pct']:>6.1f}%  "
                  f"{r['bench46_correct_count']:>4}/{r['bench46_total']:<3}  "
                  f"{r['bench46_accuracy_pct']:>6.1f}%")

        try:
            table_df.to_csv(RESULTS_CSV, index=False)
            print(f"\n[write] Comparison table saved to:\n    {RESULTS_CSV}")
        except Exception as exc:  # noqa: BLE001
            print(f"\n[write] WARNING: could not write results CSV ({exc}).")

    # -----------------------------------------------------------------------
    # Final skip summary (requirement 9).
    # -----------------------------------------------------------------------
    print("\n" + "=" * 72)
    print("RUN SUMMARY")
    print("=" * 72)
    print(f"Models completed: {len(results)} / {len(MODELS)}")
    if skipped:
        print("Models SKIPPED due to failure:")
        for s in skipped:
            print(f"    - {s['model_name']}: {s['error']}")
    else:
        print("No models were skipped.")

    # Quick verdict against the MiniLM ceiling, for BOTH benchmarks
    # independently. A model can beat/tie/trail MiniLM on 14 and do something
    # different on 46.
    if results:
        baseline = next((r for r in results
                         if r["model_name"] == "all-MiniLM-L6-v2"), None)
        if baseline:
            print(f"\nMiniLM baseline this run: "
                  f"{baseline['benchmark_correct_count']}/{baseline['benchmark_total']} "
                  f"({baseline['benchmark_accuracy_pct']:.1f}%)")
            for r in results:
                if r["model_name"] == "all-MiniLM-L6-v2":
                    continue
                delta = r["benchmark_correct_count"] - baseline["benchmark_correct_count"]
                verdict = ("BEATS" if delta > 0 else
                           "ties" if delta == 0 else "trails")
                print(f"  {r['model_name']}: {r['benchmark_correct_count']}"
                      f"/{r['benchmark_total']}  -> {verdict} baseline "
                      f"({delta:+d} tickets)")

            # SAME verdict logic, independently, for the 46-ticket set.
            print(f"\nMiniLM baseline this run (46-ticket): "
                  f"{baseline['bench46_correct_count']}/{baseline['bench46_total']} "
                  f"({baseline['bench46_accuracy_pct']:.1f}%)")
            for r in results:
                if r["model_name"] == "all-MiniLM-L6-v2":
                    continue
                delta46 = (r["bench46_correct_count"]
                           - baseline["bench46_correct_count"])
                verdict46 = ("BEATS" if delta46 > 0 else
                             "ties" if delta46 == 0 else "trails")
                print(f"  {r['model_name']}: {r['bench46_correct_count']}"
                      f"/{r['bench46_total']}  -> {verdict46} baseline "
                      f"({delta46:+d} tickets)")

    print("\nDone.")


if __name__ == "__main__":
    main()