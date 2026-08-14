"""
run_imbalance_sweep.py
======================
Class-imbalance experiment — training / evaluation sweep stage.

For each skew level of the "Access Management" category, trains TWO fresh
classifiers on that level's skewed dataset:
    (a) TF-IDF + LogisticRegression
    (b) MiniLM (all-MiniLM-L6-v2) embeddings + LogisticRegression

Evaluates each on:
    - the held-out in-distribution test split (accuracy + AM-specific
      precision/recall/f1 + AM<->Security confusion cells),
    - a fixed 14-ticket generalization benchmark (used VERBATIM),
    - a fixed 45-ticket expanded generalization benchmark (loaded from
      data/novel_tickets_expanded.json),
    - cascade escalation behaviour (MiniLM only, fixed threshold 0.50).

Produces one consolidated comparison table across all 5 levels, saved to
data/skewed/imbalance_sweep_results.csv.

Conventions (project-wide):
  - Paths via os.path relative to THIS script (walk up to project root).
  - random_state=42 / np.random.seed(42) everywhere.
  - Per-level MiniLM embeddings cached SEPARATELY at
    data/skewed/embeddings_am{N}.npy — the full-dataset cache
    (data/ticket_embeddings.npy) is NEVER read here, because row counts
    differ per level and reuse would silently misalign embeddings.
  - Defensive, actionable errors (no raw tracebacks) for missing input.
  - No Gemini / API calls; runs fully offline. CPU-only friendly.

Run:
    python src/experiments/run_imbalance_sweep.py
"""

import json
import os
import sys

import numpy as np
import pandas as pd

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
)
from sklearn.model_selection import train_test_split

# --------------------------------------------------------------------------- #
# Reproducibility
# --------------------------------------------------------------------------- #
RANDOM_STATE = 42
np.random.seed(RANDOM_STATE)

# --------------------------------------------------------------------------- #
# Path resolution — relative to this script, walk up to project root
# --------------------------------------------------------------------------- #
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, os.pardir, os.pardir))

DATA_DIR = os.path.join(PROJECT_ROOT, "data")
SKEWED_DIR = os.path.join(DATA_DIR, "skewed")
RESULTS_CSV = os.path.join(SKEWED_DIR, "imbalance_sweep_results.csv")
EXPANDED_JSON_PATH = os.path.join(PROJECT_ROOT, "data", "novel_tickets_expanded.json")

# --------------------------------------------------------------------------- #
# Experiment constants — kept consistent with train_embeddings.py
# --------------------------------------------------------------------------- #
TARGET_CATEGORY = "Access Management"
CONFUSION_PARTNER = "Security"          # AM <-> Security cell of interest

ALL_CATEGORIES = [
    "Infrastructure",
    "Application",
    "Security",
    "Database",
    "Storage",
    "Network",
    "Access Management",
]

SKEW_LEVELS = [571, 500, 200, 100, 50]

# Match train_embeddings.py split methodology.
TEST_SIZE = 0.20
MINILM_MODEL_NAME = "all-MiniLM-L6-v2"

# Existing cascade threshold — intentionally reused, NOT re-derived for skew.
CASCADE_THRESHOLD = 0.50

# Column name candidates (dataset may use any of these).
TEXT_COLUMN_CANDIDATES = ["text", "ticket_text", "description", "body"]
CATEGORY_COLUMN_CANDIDATES = ["category", "label", "Category"]

# TF-IDF config (fixed per spec).
TFIDF_MAX_FEATURES = 5000
TFIDF_NGRAM_RANGE = (1, 2)
TFIDF_STOP_WORDS = "english"

# Expanded benchmark expected size (14 original + 31 new).
EXPANDED_BENCHMARK_SIZE = 45


# --------------------------------------------------------------------------- #
# 14-ticket generalization benchmark — USE VERBATIM, DO NOT MODIFY
# --------------------------------------------------------------------------- #
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


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #
def _fail(message: str) -> None:
    """Print a clear, actionable error and exit non-zero (no traceback)."""
    print("\n" + "=" * 75)
    print("ERROR — cannot continue")
    print("=" * 75)
    print(message)
    print("=" * 75 + "\n")
    sys.exit(1)


def _resolve_column(df: pd.DataFrame, candidates, purpose: str, path: str) -> str:
    for c in candidates:
        if c in df.columns:
            return c
    _fail(
        f"Could not find the {purpose} column in {path}.\n"
        f"Looked for any of: {candidates}\n"
        f"Actual columns present: {list(df.columns)}"
    )


def load_expanded_benchmark(path=EXPANDED_JSON_PATH):
    """
    Load and validate the 45-ticket expanded generalization benchmark.

    The file must be a JSON list of EXACTLY 45 entries, each a dict with
    non-empty string keys "text" and "expected" (same shape as
    NOVEL_TICKETS). Fails loudly via _fail() with an actionable message on
    any problem (missing file, invalid JSON, wrong type, wrong count, or
    malformed entries) — never raises a raw traceback.

    Returns the list of dicts (same shape as NOVEL_TICKETS).
    """
    if not os.path.isfile(path):
        _fail(
            f"Required expanded benchmark file is missing:\n    {path}\n\n"
            f"This sweep now evaluates against the 45-ticket expanded "
            f"benchmark in addition to the inline 14-ticket set.\n"
            f"Make sure data/novel_tickets_expanded.json exists (a flat JSON "
            f"list of exactly {EXPANDED_BENCHMARK_SIZE} entries, each with "
            f'non-empty "text" and "expected" string fields).'
        )

    try:
        with open(path, "r", encoding="utf-8") as fh:
            data = json.load(fh)
    except json.JSONDecodeError as exc:
        _fail(
            f"Expanded benchmark file is not valid JSON:\n    {path}\n\n"
            f"    {exc}"
        )
    except Exception as exc:  # noqa: BLE001
        _fail(f"Failed to read expanded benchmark file {path}:\n    {exc}")

    if not isinstance(data, list):
        _fail(
            f"Expanded benchmark file must contain a JSON list, but the "
            f"top-level value is of type '{type(data).__name__}':\n    {path}"
        )

    if len(data) != EXPANDED_BENCHMARK_SIZE:
        _fail(
            f"Expanded benchmark must contain EXACTLY "
            f"{EXPANDED_BENCHMARK_SIZE} entries, but found {len(data)}:\n"
            f"    {path}\n\n"
            f"Expected 14 original + 31 new = {EXPANDED_BENCHMARK_SIZE} "
            f"tickets."
        )

    malformed = []
    for i, entry in enumerate(data):
        if not isinstance(entry, dict):
            malformed.append(f"  [{i}] not a dict (type={type(entry).__name__})")
            continue
        text = entry.get("text")
        expected = entry.get("expected")
        problems = []
        if not isinstance(text, str) or not text.strip():
            problems.append('missing/empty "text"')
        if not isinstance(expected, str) or not expected.strip():
            problems.append('missing/empty "expected"')
        if problems:
            malformed.append(f"  [{i}] " + "; ".join(problems))

    if malformed:
        _fail(
            f"Expanded benchmark file has malformed entries:\n    {path}\n\n"
            f"Every entry must be a dict with non-empty string \"text\" and "
            f"\"expected\" keys.\nOffending entries:\n"
            + "\n".join(malformed)
        )

    return data


def load_minilm():
    """Import + load the MiniLM model, with a clear message if unavailable."""
    try:
        from sentence_transformers import SentenceTransformer
    except ImportError:
        _fail(
            "sentence-transformers is not installed in this environment.\n"
            "Activate the venv and install it:\n"
            "    .\\venv\\Scripts\\Activate.ps1\n"
            "    pip install sentence-transformers"
        )
    try:
        model = SentenceTransformer(MINILM_MODEL_NAME)
    except Exception as exc:  # noqa: BLE001
        _fail(
            f"Failed to load MiniLM model '{MINILM_MODEL_NAME}':\n    {exc}\n"
            f"If this is a download/network issue, connect once to let the "
            f"model cache, then re-run offline."
        )
    return model


def get_level_embeddings(model, texts, level_n):
    """
    Compute (or load) per-level MiniLM embeddings.

    Cache path: data/skewed/embeddings_am{N}.npy  — separate per level.
    Reuse only if the cached row count matches the current CSV row count;
    otherwise recompute and overwrite. Prints which branch was taken.

    IMPORTANT: never reads data/ticket_embeddings.npy (full-dataset cache).
    """
    cache_path = os.path.join(SKEWED_DIR, f"embeddings_am{level_n}.npy")
    n_rows = len(texts)

    if os.path.isfile(cache_path):
        try:
            cached = np.load(cache_path)
        except Exception as exc:  # noqa: BLE001
            print(f"    [embeddings] cache at {cache_path} unreadable "
                  f"({exc}); recomputing.")
            cached = None

        if cached is not None and cached.shape[0] == n_rows:
            print(f"    [embeddings] REUSE cache {os.path.basename(cache_path)} "
                  f"(rows match: {cached.shape[0]}).")
            return cached
        elif cached is not None:
            print(f"    [embeddings] cache row count {cached.shape[0]} != "
                  f"current {n_rows}; RECOMPUTE + overwrite.")

    print(f"    [embeddings] COMPUTE MiniLM for {n_rows} rows -> "
          f"{os.path.basename(cache_path)}")
    emb = model.encode(
        list(texts),
        batch_size=64,
        show_progress_bar=False,
        convert_to_numpy=True,
    )
    emb = np.asarray(emb, dtype=np.float32)
    np.save(cache_path, emb)
    return emb


def am_report_row(y_true, y_pred):
    """Return (precision, recall, f1) for the Access Management class."""
    rep = classification_report(
        y_true, y_pred, labels=ALL_CATEGORIES,
        output_dict=True, zero_division=0,
    )
    am = rep.get(TARGET_CATEGORY, {})
    return (
        float(am.get("precision", 0.0)),
        float(am.get("recall", 0.0)),
        float(am.get("f1-score", 0.0)),
    )


def am_security_cells(y_true, y_pred):
    """
    Return the confusion-matrix cells for AM<->Security, both directions:
        am_as_sec  = true AM predicted Security
        sec_as_am  = true Security predicted AM
    """
    cm = confusion_matrix(y_true, y_pred, labels=ALL_CATEGORIES)
    idx = {cat: i for i, cat in enumerate(ALL_CATEGORIES)}
    am_i, sec_i = idx[TARGET_CATEGORY], idx[CONFUSION_PARTNER]
    am_as_sec = int(cm[am_i, sec_i])
    sec_as_am = int(cm[sec_i, am_i])
    return am_as_sec, sec_as_am


def eval_benchmark(clf, X_bench, expecteds):
    """
    Evaluate a fitted classifier on the 14-ticket benchmark.
    Returns (score_out_of_14, list_of_per_ticket_dicts).
    """
    preds = clf.predict(X_bench)
    results = []
    correct = 0
    for i, (pred, exp) in enumerate(zip(preds, expecteds)):
        ok = (pred == exp)
        correct += int(ok)
        results.append({"idx": i, "expected": exp, "pred": pred, "correct": ok})
    return correct, results


def am_benchmark_detail(bench_results):
    """
    Summarise how the 2 Access Management benchmark tickets were classified.
    Returns (num_am_correct, detail_string).
    """
    am_items = [r for r in bench_results if r["expected"] == TARGET_CATEGORY]
    num_correct = sum(1 for r in am_items if r["correct"])
    parts = []
    for r in am_items:
        if r["correct"]:
            parts.append("correct")
        else:
            parts.append(f"WRONG->{r['pred']}")
    detail = f"{num_correct}/{len(am_items)} [" + ", ".join(parts) + "]"
    return num_correct, detail


def cascade_analysis(clf, X_test, y_test, class_labels):
    """
    MiniLM cascade escalation analysis for Access Management test rows only,
    using the fixed threshold 0.50.

    Returns dict with:
        am_total, escalated, auto_accepted, escalated_frac, accepted_frac,
        wrong_total, wrong_below (correctly escalated),
        wrong_above (safety-net FAILURES).
    """
    proba = clf.predict_proba(X_test)              # (n, n_classes)
    conf = proba.max(axis=1)                       # top-class confidence
    preds = clf.classes_[proba.argmax(axis=1)]

    y_test = np.asarray(y_test)
    am_mask = (y_test == TARGET_CATEGORY)
    am_total = int(am_mask.sum())

    out = {
        "am_total": am_total,
        "escalated": 0, "auto_accepted": 0,
        "escalated_frac": 0.0, "accepted_frac": 0.0,
        "wrong_total": 0, "wrong_below": 0, "wrong_above": 0,
    }
    if am_total == 0:
        return out

    am_conf = conf[am_mask]
    am_pred = preds[am_mask]
    am_true = y_test[am_mask]

    escalated_mask = am_conf < CASCADE_THRESHOLD
    out["escalated"] = int(escalated_mask.sum())
    out["auto_accepted"] = int((~escalated_mask).sum())
    out["escalated_frac"] = out["escalated"] / am_total
    out["accepted_frac"] = out["auto_accepted"] / am_total

    wrong_mask = (am_pred != am_true)
    out["wrong_total"] = int(wrong_mask.sum())
    if out["wrong_total"] > 0:
        wrong_conf = am_conf[wrong_mask]
        out["wrong_below"] = int((wrong_conf < CASCADE_THRESHOLD).sum())
        out["wrong_above"] = int((wrong_conf >= CASCADE_THRESHOLD).sum())
    return out


def load_level_csv(level_n):
    path = os.path.join(SKEWED_DIR, f"synthetic_tickets_am{level_n}.csv")
    if not os.path.isfile(path):
        _fail(
            f"Required skewed dataset is missing:\n    {path}\n\n"
            f"Generate all skewed datasets first with:\n"
            f"    python src/experiments/generate_skewed_datasets.py"
        )
    try:
        df = pd.read_csv(path)
    except Exception as exc:  # noqa: BLE001
        _fail(f"Failed to read {path}:\n    {exc}")
    if df.empty:
        _fail(f"Skewed dataset {path} is empty.")
    return df, path


# --------------------------------------------------------------------------- #
# Per-level processing
# --------------------------------------------------------------------------- #
def process_level(level_n, minilm_model, bench_texts, bench_expecteds,
                  expanded_bench_texts, expanded_bench_expecteds):
    print("\n" + "#" * 75)
    print(f"#  SKEW LEVEL  am{level_n}")
    print("#" * 75)

    df, path = load_level_csv(level_n)
    text_col = _resolve_column(df, TEXT_COLUMN_CANDIDATES, "ticket text", path)
    cat_col = _resolve_column(df, CATEGORY_COLUMN_CANDIDATES, "category", path)

    texts = df[text_col].astype(str).tolist()
    labels = df[cat_col].astype(str).tolist()
    am_train_available = int(pd.Series(labels).value_counts().get(TARGET_CATEGORY, 0))
    print(f"  Loaded {len(df)} rows from {os.path.relpath(path, PROJECT_ROOT)}")
    print(f"  '{TARGET_CATEGORY}' rows in this level: {am_train_available}")

    # --- Embeddings (per-level cache, aligned to df row order) ---
    print("  [1] MiniLM embeddings")
    embeddings = get_level_embeddings(minilm_model, texts, level_n)
    if embeddings.shape[0] != len(df):
        _fail(
            f"[level {level_n}] embedding row count {embeddings.shape[0]} "
            f"does not match dataset rows {len(df)} — alignment broken."
        )

    # --- Split (same methodology as train_embeddings.py) ---
    print("  [2] Train/test split "
          f"(test_size={TEST_SIZE}, random_state={RANDOM_STATE}, stratified)")
    indices = np.arange(len(df))
    # Stratify to keep class proportions stable; fall back if a class is too
    # small to stratify at this skew level.
    strat = labels
    try:
        idx_train, idx_test = train_test_split(
            indices, test_size=TEST_SIZE, random_state=RANDOM_STATE,
            stratify=strat,
        )
    except ValueError:
        print("      (stratify not possible at this level — using unstratified "
              "split with same random_state)")
        idx_train, idx_test = train_test_split(
            indices, test_size=TEST_SIZE, random_state=RANDOM_STATE,
        )

    y_train = [labels[i] for i in idx_train]
    y_test = [labels[i] for i in idx_test]
    texts_train = [texts[i] for i in idx_train]
    texts_test = [texts[i] for i in idx_test]
    emb_train = embeddings[idx_train]
    emb_test = embeddings[idx_test]
    print(f"      train={len(idx_train)}  test={len(idx_test)}")

    # ------------------------------------------------------------------ #
    # (3a) TF-IDF + LogReg
    # ------------------------------------------------------------------ #
    print("  [3a] Training TF-IDF + LogisticRegression")
    tfidf = TfidfVectorizer(
        max_features=TFIDF_MAX_FEATURES,
        ngram_range=TFIDF_NGRAM_RANGE,
        stop_words=TFIDF_STOP_WORDS,
    )
    X_train_tfidf = tfidf.fit_transform(texts_train)
    X_test_tfidf = tfidf.transform(texts_test)
    clf_tfidf = LogisticRegression(max_iter=1000, random_state=RANDOM_STATE)
    clf_tfidf.fit(X_train_tfidf, y_train)

    # ------------------------------------------------------------------ #
    # (3b) MiniLM + LogReg
    # ------------------------------------------------------------------ #
    print("  [3b] Training MiniLM + LogisticRegression")
    clf_minilm = LogisticRegression(max_iter=1000, random_state=RANDOM_STATE)
    clf_minilm.fit(emb_train, y_train)

    # ------------------------------------------------------------------ #
    # (4) Held-out in-distribution evaluation
    # ------------------------------------------------------------------ #
    print("  [4] Held-out (in-distribution) evaluation "
          "(known to be uninformative — recorded for the writeup)")

    pred_tfidf = clf_tfidf.predict(X_test_tfidf)
    pred_minilm = clf_minilm.predict(emb_test)

    acc_tfidf = accuracy_score(y_test, pred_tfidf)
    acc_minilm = accuracy_score(y_test, pred_minilm)

    tfidf_am_p, tfidf_am_r, tfidf_am_f1 = am_report_row(y_test, pred_tfidf)
    minilm_am_p, minilm_am_r, minilm_am_f1 = am_report_row(y_test, pred_minilm)

    tfidf_am_as_sec, tfidf_sec_as_am = am_security_cells(y_test, pred_tfidf)
    minilm_am_as_sec, minilm_sec_as_am = am_security_cells(y_test, pred_minilm)

    print(f"      TF-IDF  held-out accuracy : {acc_tfidf:.4f}")
    print(f"      MiniLM  held-out accuracy : {acc_minilm:.4f}")
    print(f"      TF-IDF  AM  P/R/F1        : "
          f"{tfidf_am_p:.3f} / {tfidf_am_r:.3f} / {tfidf_am_f1:.3f}")
    print(f"      MiniLM  AM  P/R/F1        : "
          f"{minilm_am_p:.3f} / {minilm_am_r:.3f} / {minilm_am_f1:.3f}")
    print(f"      TF-IDF  AM->Sec / Sec->AM : "
          f"{tfidf_am_as_sec} / {tfidf_sec_as_am}")
    print(f"      MiniLM  AM->Sec / Sec->AM : "
          f"{minilm_am_as_sec} / {minilm_sec_as_am}")

    # ------------------------------------------------------------------ #
    # (5) 14-ticket generalization benchmark
    # ------------------------------------------------------------------ #
    print("  [5] 14-ticket generalization benchmark")
    X_bench_tfidf = tfidf.transform(bench_texts)
    X_bench_minilm = minilm_model.encode(
        bench_texts, batch_size=64, show_progress_bar=False,
        convert_to_numpy=True,
    ).astype(np.float32)

    tfidf_bench_score, tfidf_bench_res = eval_benchmark(
        clf_tfidf, X_bench_tfidf, bench_expecteds)
    minilm_bench_score, minilm_bench_res = eval_benchmark(
        clf_minilm, X_bench_minilm, bench_expecteds)

    tfidf_am_correct, tfidf_am_detail = am_benchmark_detail(tfidf_bench_res)
    minilm_am_correct, minilm_am_detail = am_benchmark_detail(minilm_bench_res)

    print(f"      TF-IDF  benchmark : {tfidf_bench_score}/14   "
          f"AM tickets: {tfidf_am_detail}")
    print(f"      MiniLM  benchmark : {minilm_bench_score}/14   "
          f"AM tickets: {minilm_am_detail}")

    # ------------------------------------------------------------------ #
    # (5b) 45-ticket expanded generalization benchmark
    # ------------------------------------------------------------------ #
    print("  [5b] 45-ticket expanded benchmark")
    X_bench45_tfidf = tfidf.transform(expanded_bench_texts)
    X_bench45_minilm = minilm_model.encode(
        expanded_bench_texts, batch_size=64, show_progress_bar=False,
        convert_to_numpy=True,
    ).astype(np.float32)

    tfidf_bench45_score, tfidf_bench45_res = eval_benchmark(
        clf_tfidf, X_bench45_tfidf, expanded_bench_expecteds)
    minilm_bench45_score, minilm_bench45_res = eval_benchmark(
        clf_minilm, X_bench45_minilm, expanded_bench_expecteds)

    tfidf_am45_correct, tfidf_am45_detail = am_benchmark_detail(tfidf_bench45_res)
    minilm_am45_correct, minilm_am45_detail = am_benchmark_detail(minilm_bench45_res)

    print(f"      TF-IDF  expanded45 : {tfidf_bench45_score}/45   "
          f"AM tickets: {tfidf_am45_detail}")
    print(f"      MiniLM  expanded45 : {minilm_bench45_score}/45   "
          f"AM tickets: {minilm_am45_detail}")

    # ------------------------------------------------------------------ #
    # (6) Cascade escalation analysis (MiniLM, threshold 0.50)
    # ------------------------------------------------------------------ #
    print(f"  [6] Cascade analysis (MiniLM, fixed threshold "
          f"{CASCADE_THRESHOLD}) — Access Management test rows")
    casc = cascade_analysis(clf_minilm, emb_test, y_test, clf_minilm.classes_)
    print(f"      AM test rows              : {casc['am_total']}")
    print(f"      (a) escalated (<{CASCADE_THRESHOLD})   : "
          f"{casc['escalated']} ({casc['escalated_frac']:.1%})")
    print(f"          auto-accepted (>=)    : "
          f"{casc['auto_accepted']} ({casc['accepted_frac']:.1%})")
    print(f"      (b) wrong AM predictions  : {casc['wrong_total']}")
    print(f"          wrong & escalated     : {casc['wrong_below']} "
          f"(safety net worked)")
    if casc["wrong_above"] > 0:
        print(f"          *** SAFETY-NET FAILURE : {casc['wrong_above']} "
              f"wrong AND confidently accepted (>= {CASCADE_THRESHOLD}) ***")
    else:
        print(f"          wrong & accepted      : 0 (no safety-net failures)")

    # ------------------------------------------------------------------ #
    # Consolidated row for this level
    # ------------------------------------------------------------------ #
    return {
        "skew_level": level_n,
        "am_train_count": am_train_available,
        "tfidf_heldout_acc": round(acc_tfidf, 4),
        "minilm_heldout_acc": round(acc_minilm, 4),
        "tfidf_bench_14": f"{tfidf_bench_score}/14",
        "minilm_bench_14": f"{minilm_bench_score}/14",
        "tfidf_bench_45": f"{tfidf_bench45_score}/45",
        "minilm_bench_45": f"{minilm_bench45_score}/45",
        "tfidf_am_bench": tfidf_am_detail,
        "minilm_am_bench": minilm_am_detail,
        "tfidf_am_bench_45": tfidf_am45_detail,
        "minilm_am_bench_45": minilm_am45_detail,
        "tfidf_am_to_sec": tfidf_am_as_sec,
        "minilm_am_to_sec": minilm_am_as_sec,
        "cascade_safety_net_failures": casc["wrong_above"],
    }


# --------------------------------------------------------------------------- #
# Main
# --------------------------------------------------------------------------- #
def main() -> None:
    print("=" * 75)
    print("IMBALANCE SWEEP  —  train + evaluate per skew level")
    print("=" * 75)
    print(f"Project root : {PROJECT_ROOT}")
    print(f"Skewed dir   : {SKEWED_DIR}")
    print(f"Skew levels  : {SKEW_LEVELS}")
    print(f"MiniLM model : {MINILM_MODEL_NAME}")
    print(f"Cascade thr  : {CASCADE_THRESHOLD} (reused, not re-derived)")
    print(f"random_state : {RANDOM_STATE}")
    print("-" * 75)

    if not os.path.isdir(SKEWED_DIR):
        _fail(
            f"Skewed data directory does not exist:\n    {SKEWED_DIR}\n\n"
            f"Generate the skewed datasets first:\n"
            f"    python src/experiments/generate_skewed_datasets.py"
        )

    # Pre-flight: verify every required level CSV exists before the slow work.
    missing = [
        f"synthetic_tickets_am{n}.csv"
        for n in SKEW_LEVELS
        if not os.path.isfile(os.path.join(SKEWED_DIR, f"synthetic_tickets_am{n}.csv"))
    ]
    if missing:
        _fail(
            "The following required skewed dataset(s) are missing from "
            f"data/skewed/:\n    " + "\n    ".join(missing) + "\n\n"
            "Generate them with:\n"
            "    python src/experiments/generate_skewed_datasets.py"
        )

    # Load MiniLM once and reuse across all levels (embeddings still cached
    # per level on disk).
    print("Loading MiniLM model (once, reused across levels)...")
    minilm_model = load_minilm()

    bench_texts = [t["text"] for t in NOVEL_TICKETS]
    bench_expecteds = [t["expected"] for t in NOVEL_TICKETS]

    # Load + validate the 45-ticket expanded benchmark once, reused across
    # all levels (mirrors how bench_texts/bench_expecteds are built above).
    expanded_tickets = load_expanded_benchmark()
    expanded_bench_texts = [t["text"] for t in expanded_tickets]
    expanded_bench_expecteds = [t["expected"] for t in expanded_tickets]

    consolidated = []
    for n in SKEW_LEVELS:
        row = process_level(
            n, minilm_model, bench_texts, bench_expecteds,
            expanded_bench_texts, expanded_bench_expecteds,
        )
        consolidated.append(row)

    # ------------------------------------------------------------------ #
    # Consolidated summary table
    # ------------------------------------------------------------------ #
    results_df = pd.DataFrame(consolidated)
    column_order = [
        "skew_level",
        "am_train_count",
        "tfidf_heldout_acc",
        "minilm_heldout_acc",
        "tfidf_bench_14",
        "minilm_bench_14",
        "tfidf_bench_45",
        "minilm_bench_45",
        "tfidf_am_bench",
        "minilm_am_bench",
        "tfidf_am_bench_45",
        "minilm_am_bench_45",
        "tfidf_am_to_sec",
        "minilm_am_to_sec",
        "cascade_safety_net_failures",
    ]
    results_df = results_df[column_order]

    print("\n" + "=" * 75)
    print("CONSOLIDATED RESULTS  (all skew levels)")
    print("=" * 75)
    # to_string keeps the whole table readable in a terminal paste.
    print(results_df.to_string(index=False))
    print("=" * 75)

    os.makedirs(SKEWED_DIR, exist_ok=True)
    results_df.to_csv(RESULTS_CSV, index=False)
    print(f"Saved consolidated table to:\n    "
          f"{os.path.relpath(RESULTS_CSV, PROJECT_ROOT)}")
    print("Sweep complete.")


if __name__ == "__main__":
    main()
