"""
src/experiments/plot_calibration_curves.py
==========================================

Reliability-diagram generator for the AI-Powered Intelligent Ticket Routing
& Resolution Agent's two-tier cascade classifier.

WHY THIS SCRIPT EXISTS
----------------------
This project's central research claim is that every escalation decision is
governed by a *calibrated* confidence signal measured against real data. So
far that calibration has only been demonstrated via accuracy-at-threshold
tables. This script produces the missing artifact: a proper reliability
diagram (predicted confidence vs. actually-observed accuracy) for BOTH tiers
of the cascade, evaluated against a real production-style 500-ticket batch
rather than the training / held-out split.

WHAT IT READS
-------------
1. data/synthetic_tickets.csv
       The full 4,000-ticket dataset. Columns include id, title, description,
       category. The "category" column here is treated as GROUND TRUTH.

2. data/category_stores/{Category}.csv
       Per-category CSVs from a prior 500-ticket production batch run.
       Columns include batch_ticket_id, original_id, category,
       resolution_status, resolution_text. "original_id" joins back to
       synthetic_tickets.csv's "id" column. NOTE: the "category" column in
       these files is the pipeline's FINAL ROUTED category (which file the
       ticket landed in), NOT a raw model prediction, and NO confidence
       score was ever persisted. This script therefore RECOMPUTES both
       tiers' predictions and confidences fresh from ticket text.
       Category filenames handled: Infrastructure, Application, Security,
       Database, Storage, Network, "Access Management" (literal space).

3. models/ticket_classifier.joblib
       The trained Tier-2 (MiniLM embeddings + LogisticRegression) artifact.

WHAT IT WRITES
--------------
- data/calibration_tier1_reliability_diagram.png
- data/calibration_tier2_reliability_diagram.png
      Standard reliability diagrams (one figure each), diagonal y=x perfect-
      calibration reference line, ECE annotated in the title.
- data/calibration_reliability_data.csv
      Bin-level evidence for citation: one row per non-empty bin per tier,
      columns tier, bin_lower, bin_upper, mean_predicted_confidence,
      observed_accuracy, n_tickets_in_bin.

THE ECE METRIC REPORTED
-----------------------
Expected Calibration Error (ECE) is the count-weighted average of the
absolute gap between mean predicted confidence and observed accuracy across
all non-empty confidence bins:

    ECE = sum_over_bins( (n_bin / N_total) * | conf_bin - acc_bin | )

Lower is better. 0.0 means predicted confidence exactly matches observed
accuracy in every bin (perfect calibration). This is the standard binned ECE
used throughout the calibration literature (Guo et al., 2017).

SCOPE NOTE ON TIER-1
--------------------
In the LIVE cascade, Tier-1's confidence only "counts" when Tier-1 resolves a
ticket directly (i.e. its confidence clears the 0.50 cascade threshold and it
is not escalated to Tier-2). For THIS diagnostic, however, we deliberately
evaluate Tier-1's calibration as if it classified every ticket, independent
of the 0.50 threshold, because the goal is to audit Tier-1's *raw confidence
honesty* — not to simulate live cascade routing. This distinction is printed
explicitly at runtime so results are not later misread as cascade behavior.

This script is standalone and offline. It imports exactly two reusable
functions (train_tier1, get_tier1_confidence) from
src.classification.train_cascade and otherwise depends on nothing else in the
project at runtime.
"""

import os
import sys
import traceback

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
# matplotlib must be set to a non-interactive backend BEFORE pyplot import,
# because this script runs headless from a terminal (no display, no show()).
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


# ---------------------------------------------------------------------------
# PATHS / CONSTANTS
# ---------------------------------------------------------------------------
# Project root is two directories up from this file (src/experiments/ -> root).
PROJECT_ROOT = os.path.abspath(
    os.path.join(os.path.dirname(__file__), os.pardir, os.pardir)
)

DATA_DIR = os.path.join(PROJECT_ROOT, "data")
MODELS_DIR = os.path.join(PROJECT_ROOT, "models")
CATEGORY_STORES_DIR = os.path.join(DATA_DIR, "category_stores")

SYNTHETIC_TICKETS_CSV = os.path.join(DATA_DIR, "synthetic_tickets.csv")
TIER2_MODEL_PATH = os.path.join(MODELS_DIR, "ticket_classifier.joblib")

TIER1_PNG = os.path.join(DATA_DIR, "calibration_tier1_reliability_diagram.png")
TIER2_PNG = os.path.join(DATA_DIR, "calibration_tier2_reliability_diagram.png")
RELIABILITY_CSV = os.path.join(DATA_DIR, "calibration_reliability_data.csv")

# Per-category store filenames (note the literal space in "Access Management").
CATEGORY_FILENAMES = [
    "Infrastructure",
    "Application",
    "Security",
    "Database",
    "Storage",
    "Network",
    "Access Management",
]

# Reliability-diagram convention: 10 equal-width bins spanning [0.0, 1.0].
N_BINS = 10

# Sentence-transformer model name and encode batch size, matching this
# project's convention used elsewhere.
MINILM_MODEL_NAME = "all-MiniLM-L6-v2"
ENCODE_BATCH_SIZE = 64

# The expected size of the assembled production batch (context / sanity check).
EXPECTED_BATCH_SIZE = 500


# ---------------------------------------------------------------------------
# SMALL PRINT HELPERS (house verbosity convention)
# ---------------------------------------------------------------------------
def _banner(text):
    """Print a clearly-delimited section banner."""
    bar = "=" * 70
    print("\n" + bar)
    print(text)
    print(bar)


def _fatal(message):
    """
    Print a clear, actionable error message (no raw traceback) and exit.
    Matches this project's 'no raw tracebacks' convention.
    """
    print("\n" + "!" * 70)
    print("FATAL: " + message)
    print("!" * 70)
    sys.exit(1)


def _combined_text(title, description):
    """
    Build the combined_text input exactly the way streamlit_app.py's
    combined_text() helper does: f"{title} {description}".strip().
    Guards against NaN / non-string cells coming out of pandas.
    """
    t = "" if title is None or (isinstance(title, float) and np.isnan(title)) else str(title)
    d = "" if description is None or (isinstance(description, float) and np.isnan(description)) else str(description)
    return f"{t} {d}".strip()


# ---------------------------------------------------------------------------
# TIER-1 FUNCTION IMPORT (with sys.path fallback, mirroring streamlit_app.py)
# ---------------------------------------------------------------------------
def _import_tier1_functions():
    """
    Import train_tier1 and get_tier1_confidence from
    src.classification.train_cascade. If the package-style import fails,
    append src/classification to sys.path and retry — the same fallback
    pattern used in streamlit_app.py's load_resources().
    """
    print("[import] Attempting package-style import of Tier-1 functions...")
    try:
        from src.classification.train_cascade import (
            train_tier1,
            get_tier1_confidence,
        )
        print("[import] OK: imported from src.classification.train_cascade")
        return train_tier1, get_tier1_confidence
    except Exception as exc_pkg:
        print(f"[import] Package-style import failed ({exc_pkg!r}); "
              f"falling back to sys.path injection...")

    # Fallback: make sure the project root is importable, then also add the
    # concrete src/classification directory for a bare module import.
    if PROJECT_ROOT not in sys.path:
        sys.path.insert(0, PROJECT_ROOT)
    classification_dir = os.path.join(PROJECT_ROOT, "src", "classification")
    if classification_dir not in sys.path:
        sys.path.insert(0, classification_dir)

    # Retry the package import first (now that PROJECT_ROOT is on the path).
    try:
        from src.classification.train_cascade import (
            train_tier1,
            get_tier1_confidence,
        )
        print("[import] OK: imported from src.classification.train_cascade "
              "after sys.path injection")
        return train_tier1, get_tier1_confidence
    except Exception:
        pass

    # Last resort: bare module import from src/classification on sys.path.
    try:
        from train_cascade import train_tier1, get_tier1_confidence  # type: ignore
        print("[import] OK: imported from bare module 'train_cascade'")
        return train_tier1, get_tier1_confidence
    except Exception as exc_bare:
        _fatal(
            "Could not import train_tier1 / get_tier1_confidence from "
            "src.classification.train_cascade.\n"
            f"  Last error: {exc_bare!r}\n"
            "  Checked sys.path entries including:\n"
            f"    - {PROJECT_ROOT}\n"
            f"    - {classification_dir}\n"
            "  Confirm the file src/classification/train_cascade.py exists and "
            "defines both functions."
        )


# ---------------------------------------------------------------------------
# STEP 1 — Assemble the 500-ticket evaluation set
# ---------------------------------------------------------------------------
def assemble_evaluation_set():
    """
    Read every per-category store CSV, join each row's original_id back to
    synthetic_tickets.csv to recover title/description/ground-truth category,
    and return a single tidy DataFrame with columns:
        original_id, title, description, true_category, combined_text
    """
    _banner("STEP 1 — Assembling the production evaluation set")

    if not os.path.isfile(SYNTHETIC_TICKETS_CSV):
        _fatal(
            "Ground-truth dataset not found at:\n"
            f"    {SYNTHETIC_TICKETS_CSV}\n"
            "  This file (synthetic_tickets.csv) is required as the source of "
            "true labels. Generate/restore it before running this script."
        )

    print(f"[step1] Loading ground-truth dataset: {SYNTHETIC_TICKETS_CSV}")
    try:
        gt = pd.read_csv(SYNTHETIC_TICKETS_CSV)
    except Exception as exc:
        _fatal(f"Failed to read synthetic_tickets.csv: {exc!r}")

    required_gt_cols = {"id", "title", "description", "category"}
    missing_gt = required_gt_cols - set(gt.columns)
    if missing_gt:
        _fatal(
            "synthetic_tickets.csv is missing required column(s): "
            f"{sorted(missing_gt)}.\n"
            f"  Found columns: {list(gt.columns)}"
        )
    print(f"[step1] Ground-truth dataset loaded: {len(gt):,} rows")

    # Index ground truth by id for fast lookup. Keep only what we need.
    gt_indexed = gt.set_index("id")

    # Directory presence check for the category stores.
    if not os.path.isdir(CATEGORY_STORES_DIR):
        _fatal(
            "Category-stores directory not found at:\n"
            f"    {CATEGORY_STORES_DIR}\n"
            "  This directory holds the per-category production batch CSVs "
            "required to assemble the evaluation set."
        )

    assembled_rows = []
    total_store_rows = 0
    files_seen = 0
    files_missing = 0
    join_failures = 0

    for cat_name in CATEGORY_FILENAMES:
        store_path = os.path.join(CATEGORY_STORES_DIR, cat_name + ".csv")
        if not os.path.isfile(store_path):
            # Skip gracefully — matches this project's existing convention.
            print(f"[step1] WARNING: category store missing, skipping: "
                  f"{store_path}")
            files_missing += 1
            continue

        files_seen += 1
        try:
            store_df = pd.read_csv(store_path)
        except Exception as exc:
            print(f"[step1] WARNING: could not read {store_path} "
                  f"({exc!r}); skipping this file.")
            continue

        if "original_id" not in store_df.columns:
            print(f"[step1] WARNING: {store_path} has no 'original_id' column "
                  f"(found {list(store_df.columns)}); skipping this file.")
            continue

        n_rows = len(store_df)
        total_store_rows += n_rows
        print(f"[step1] Reading store '{cat_name}': {n_rows} rows")

        for original_id in store_df["original_id"].tolist():
            # Join back to ground truth by id.
            try:
                lookup_id = original_id
                if lookup_id not in gt_indexed.index:
                    # Try an int-cast fallback in case of dtype mismatch.
                    try:
                        lookup_id = int(original_id)
                    except (TypeError, ValueError):
                        lookup_id = original_id

                if lookup_id not in gt_indexed.index:
                    join_failures += 1
                    continue

                gt_row = gt_indexed.loc[lookup_id]
                # If duplicate ids exist, .loc may return a DataFrame; take first.
                if isinstance(gt_row, pd.DataFrame):
                    gt_row = gt_row.iloc[0]

                title = gt_row["title"]
                description = gt_row["description"]
                true_category = gt_row["category"]

                assembled_rows.append({
                    "original_id": original_id,
                    "title": title,
                    "description": description,
                    "true_category": true_category,
                    "combined_text": _combined_text(title, description),
                })
            except Exception:
                # Never crash on a single bad row.
                join_failures += 1
                continue

    if files_missing:
        print(f"[step1] Note: {files_missing} category store file(s) were "
              f"missing and skipped.")

    if join_failures:
        print(f"[step1] WARNING: {join_failures} row(s) failed to join back to "
              f"synthetic_tickets.csv by original_id and were EXCLUDED.")

    eval_df = pd.DataFrame(assembled_rows)

    if eval_df.empty:
        _fatal(
            "Assembled evaluation set is EMPTY. No category-store rows joined "
            "successfully to synthetic_tickets.csv. Check that original_id "
            "values correspond to synthetic_tickets.csv 'id' values."
        )

    # Drop rows with empty combined text (nothing to classify).
    before = len(eval_df)
    eval_df = eval_df[eval_df["combined_text"].str.len() > 0].reset_index(drop=True)
    dropped_empty = before - len(eval_df)
    if dropped_empty:
        print(f"[step1] WARNING: dropped {dropped_empty} row(s) with empty "
              f"combined_text.")

    print(f"[step1] Category store files processed: {files_seen}")
    print(f"[step1] Total store rows scanned: {total_store_rows}")
    print(f"[step1] Final assembled evaluation set size: {len(eval_df)}")
    if len(eval_df) == EXPECTED_BATCH_SIZE:
        print(f"[step1] CONFIRMED: matches expected batch size of "
              f"{EXPECTED_BATCH_SIZE}.")
    else:
        print(f"[step1] Note: assembled size {len(eval_df)} differs from the "
              f"expected ~{EXPECTED_BATCH_SIZE}. This is fine if stores were "
              f"missing or joins were dropped — reported above for transparency.")

    return eval_df, gt


# ---------------------------------------------------------------------------
# STEP 2 — Recompute Tier-1 (TF-IDF) predictions + confidence
# ---------------------------------------------------------------------------
def recompute_tier1(eval_df, gt_full, train_tier1, get_tier1_confidence):
    """
    Train Tier-1 fresh on the FULL synthetic_tickets.csv (title + ' ' +
    description as input, category as label) exactly as streamlit_app.py does
    at startup, then score the 500-ticket evaluation set.

    Returns two parallel numpy arrays aligned to eval_df row order:
        tier1_pred (object array of predicted category strings)
        tier1_conf (float array of confidence-of-predicted-class)
    """
    _banner("STEP 2 — Recomputing TIER-1 (TF-IDF) predictions + confidence")

    # Build the training inputs from ONLY the 80% train split — must match
    # the exact split used by train_embeddings.py (test_size=0.2,
    # random_state=42, stratify=y) so Tier-1 never trains on the same
    # tickets it is later evaluated on.
    print("[step2] Building Tier-1 training inputs from the 80% train split "
          "(matching train_embeddings.py's split, NOT the full dataset)...")
    gt_train, _ = train_test_split(
        gt_full, test_size=0.2, random_state=42, stratify=gt_full["category"]
    )
    train_texts = (
        gt_train["title"].fillna("").astype(str)
        + " "
        + gt_train["description"].fillna("").astype(str)
    ).str.strip().tolist()
    train_labels = gt_train["category"].astype(str).tolist()

    print(f"[step2] Training Tier-1 on {len(train_texts):,} tickets "
          f"(fresh, no saved artifact)...")
    try:
        tier1_vectorizer, tier1_classifier = train_tier1(train_texts, train_labels)
    except Exception as exc:
        _fatal(f"Could not train Tier-1 via train_tier1(): {exc!r}")

    print("[step2] Tier-1 trained. Scoring the evaluation set (batched call)...")
    eval_texts = eval_df["combined_text"].tolist()

    try:
        raw_preds, raw_confs = get_tier1_confidence(
            tier1_vectorizer, tier1_classifier, eval_texts
        )
    except Exception as exc:
        _fatal(f"get_tier1_confidence() call failed: {exc!r}")

    tier1_pred = np.asarray([str(p) for p in raw_preds], dtype=object)
    tier1_conf = np.asarray(raw_confs, dtype=float)
    print(f"[step2]   scored {len(eval_texts)}/{len(eval_texts)} tickets")

    valid = np.isfinite(tier1_conf)
    
    if valid.sum() < len(eval_texts):
        print(f"[step2] WARNING: {len(eval_texts) - int(valid.sum())} ticket(s) produced a "
              f"non-numeric Tier-1 confidence and will be excluded from binning.")

    overall_acc = float(np.mean(
        tier1_pred[valid] == eval_df["true_category"].to_numpy(dtype=object)[valid]
    )) if valid.any() else float("nan")
    print(f"[step2] Tier-1 raw overall accuracy on evaluation set: "
          f"{overall_acc:.4f}")

    return tier1_pred, tier1_conf


# ---------------------------------------------------------------------------
# STEP 3 — Recompute Tier-2 (MiniLM) predictions + confidence
# ---------------------------------------------------------------------------
def recompute_tier2(eval_df):
    """
    Encode the 500 combined_texts with all-MiniLM-L6-v2, load the trained
    Tier-2 classifier, predict category and confidence-of-predicted-class.

    Returns two parallel numpy arrays aligned to eval_df row order:
        tier2_pred (object array of predicted category strings)
        tier2_conf (float array of confidence-of-predicted-class)
    """
    _banner("STEP 3 — Recomputing TIER-2 (MiniLM) predictions + confidence")

    if not os.path.isfile(TIER2_MODEL_PATH):
        _fatal(
            "Tier-2 classifier artifact not found at:\n"
            f"    {TIER2_MODEL_PATH}\n"
            "  This file (ticket_classifier.joblib) is required to score "
            "Tier-2. Train/restore it before running this script."
        )

    # Import heavy deps lazily so Tier-1-only debugging stays fast if needed.
    print("[step3] Importing SentenceTransformer and joblib...")
    try:
        from sentence_transformers import SentenceTransformer
        import joblib
    except Exception as exc:
        _fatal(
            "Failed to import required libraries for Tier-2 "
            f"(sentence-transformers / joblib): {exc!r}"
        )

    print(f"[step3] Loading embedding model: {MINILM_MODEL_NAME}")
    try:
        embedder = SentenceTransformer(MINILM_MODEL_NAME)
    except Exception as exc:
        _fatal(
            f"Could not load SentenceTransformer '{MINILM_MODEL_NAME}': "
            f"{exc!r}\n  (This runs offline — confirm the model is cached "
            "locally.)"
        )

    print(f"[step3] Loading Tier-2 classifier: {TIER2_MODEL_PATH}")
    try:
        clf = joblib.load(TIER2_MODEL_PATH)
    except Exception as exc:
        _fatal(f"Could not joblib.load the Tier-2 classifier: {exc!r}")

    eval_texts = eval_df["combined_text"].tolist()
    print(f"[step3] Encoding {len(eval_texts)} tickets "
          f"(batch_size={ENCODE_BATCH_SIZE})...")
    try:
        embeddings = embedder.encode(
            eval_texts,
            batch_size=ENCODE_BATCH_SIZE,
            show_progress_bar=True,
            convert_to_numpy=True,
        )
    except Exception as exc:
        _fatal(f"SentenceTransformer.encode() failed: {exc!r}")

    print("[step3] Predicting categories with Tier-2 classifier...")
    try:
        pred_labels = clf.predict(embeddings)
    except Exception as exc:
        _fatal(f"Tier-2 classifier .predict() failed: {exc!r}")

    print("[step3] Computing predicted-class probabilities...")
    try:
        proba = clf.predict_proba(embeddings)
        classes = list(getattr(clf, "classes_", []))
    except Exception as exc:
        _fatal(
            "Tier-2 classifier .predict_proba() failed "
            f"({exc!r}). A probability-capable classifier is required for "
            "calibration analysis."
        )

    tier2_pred = []
    tier2_conf = []
    for i in range(len(eval_texts)):
        pred_cat = pred_labels[i]
        row_proba = proba[i]
        # Same pattern as classify_ticket_cascade(): find predicted class index
        # in .classes_ and take that probability; fall back to max(proba).
        try:
            idx = classes.index(pred_cat)
            conf = float(row_proba[idx])
        except Exception:
            conf = float(np.max(row_proba))
        tier2_pred.append(str(pred_cat))
        tier2_conf.append(conf)

    tier2_pred = np.asarray(tier2_pred, dtype=object)
    tier2_conf = np.asarray(tier2_conf, dtype=float)

    overall_acc = float(np.mean(
        tier2_pred == eval_df["true_category"].to_numpy(dtype=object)
    ))
    print(f"[step3] Tier-2 overall accuracy on evaluation set: "
          f"{overall_acc:.4f}")

    return tier2_pred, tier2_conf


# ---------------------------------------------------------------------------
# STEP 4 — Build reliability-diagram bin data + ECE
# ---------------------------------------------------------------------------
def build_reliability_bins(pred, conf, true_labels, tier_number):
    """
    Bin predictions into N_BINS equal-width confidence bins over [0, 1].

    Returns:
        bins_data: list of dicts (one per NON-EMPTY bin) with keys
            bin_lower, bin_upper, mean_predicted_confidence,
            observed_accuracy, n_tickets_in_bin
        ece: float Expected Calibration Error over non-empty bins
    """
    _banner(f"STEP 4 — Building reliability bins for TIER-{tier_number}")

    true_arr = np.asarray(true_labels, dtype=object)
    conf = np.asarray(conf, dtype=float)
    pred = np.asarray(pred, dtype=object)

    # Exclude any non-finite confidences from binning.
    finite_mask = np.isfinite(conf)
    if finite_mask.sum() < len(conf):
        print(f"[step4] Tier-{tier_number}: excluding "
              f"{len(conf) - int(finite_mask.sum())} ticket(s) with "
              f"non-finite confidence.")
    conf_f = conf[finite_mask]
    pred_f = pred[finite_mask]
    true_f = true_arr[finite_mask]

    correct = (pred_f == true_f).astype(float)
    n_total = len(conf_f)
    if n_total == 0:
        _fatal(f"Tier-{tier_number}: no valid predictions to bin.")

    bin_edges = np.linspace(0.0, 1.0, N_BINS + 1)
    bins_data = []
    ece = 0.0

    print(f"[step4] Tier-{tier_number}: {n_total} tickets across {N_BINS} bins")
    print(f"[step4] {'bin range':>14} | {'n':>5} | {'mean_conf':>9} | "
          f"{'accuracy':>8} | {'|gap|':>6}")
    print("[step4] " + "-" * 58)

    for b in range(N_BINS):
        lo = bin_edges[b]
        hi = bin_edges[b + 1]
        # Right edge inclusive only on the final bin so conf==1.0 is captured.
        if b < N_BINS - 1:
            in_bin = (conf_f >= lo) & (conf_f < hi)
        else:
            in_bin = (conf_f >= lo) & (conf_f <= hi)

        n_in = int(in_bin.sum())
        if n_in == 0:
            print(f"[step4] {f'[{lo:.1f}, {hi:.1f})':>14} | "
                  f"{0:>5} |    (empty bin — skipped)")
            continue

        mean_conf = float(np.mean(conf_f[in_bin]))
        accuracy = float(np.mean(correct[in_bin]))
        gap = abs(mean_conf - accuracy)
        ece += (n_in / n_total) * gap

        bins_data.append({
            "bin_lower": round(float(lo), 4),
            "bin_upper": round(float(hi), 4),
            "mean_predicted_confidence": round(mean_conf, 6),
            "observed_accuracy": round(accuracy, 6),
            "n_tickets_in_bin": n_in,
        })

        print(f"[step4] {f'[{lo:.1f}, {hi:.1f})':>14} | {n_in:>5} | "
              f"{mean_conf:>9.4f} | {accuracy:>8.4f} | {gap:>6.4f}")

    print("[step4] " + "-" * 58)
    print(f"[step4] Tier-{tier_number} Expected Calibration Error (ECE): "
          f"{ece:.4f}")
    print("[step4] ECE = count-weighted mean |confidence - accuracy| across "
          "bins; lower = better calibrated, 0 = perfect calibration.")

    return bins_data, ece


# ---------------------------------------------------------------------------
# STEP 5 — Plot a reliability diagram
# ---------------------------------------------------------------------------
def plot_reliability_diagram(bins_data, ece, tier_number, tier_label,
                             out_path):
    """
    Render and save a single standard reliability diagram for one tier.
    """
    _banner(f"STEP 5 — Plotting TIER-{tier_number} reliability diagram")

    if not bins_data:
        print(f"[step5] Tier-{tier_number}: no non-empty bins to plot; "
              f"skipping figure.")
        return

    mean_conf = [d["mean_predicted_confidence"] for d in bins_data]
    accuracy = [d["observed_accuracy"] for d in bins_data]
    counts = [d["n_tickets_in_bin"] for d in bins_data]

    fig, ax = plt.subplots(figsize=(7.2, 6.4))

    # Perfect-calibration diagonal reference line (y = x).
    ax.plot([0, 1], [0, 1], linestyle="--", color="gray", linewidth=1.5,
            label="Perfect calibration (y = x)")

    # Reliability curve: observed accuracy vs. mean predicted confidence.
    ax.plot(mean_conf, accuracy, marker="o", markersize=7, linewidth=2.0,
            color="#1f77b4", label="Observed calibration")

    # Annotate each point with the bin ticket count for transparency.
    for xc, ya, cnt in zip(mean_conf, accuracy, counts):
        ax.annotate(f"n={cnt}", (xc, ya),
                    textcoords="offset points", xytext=(6, -10),
                    fontsize=8, color="#333333")

    ax.set_xlim(0.0, 1.0)
    ax.set_ylim(0.0, 1.0)
    ax.set_xlabel("Mean Predicted Confidence", fontsize=12)
    ax.set_ylabel("Observed Accuracy", fontsize=12)
    ax.set_title(
        f"{tier_label} Calibration — 500-Ticket Production Batch\n"
        f"Expected Calibration Error (ECE) = {ece:.4f}",
        fontsize=12,
    )
    ax.grid(True, linestyle=":", alpha=0.5)
    ax.legend(loc="upper left", fontsize=10)
    ax.set_aspect("equal", adjustable="box")

    fig.tight_layout()
    try:
        fig.savefig(out_path, dpi=180)
        print(f"[step5] Saved: {out_path}")
    except Exception as exc:
        print(f"[step5] WARNING: could not save figure {out_path}: {exc!r}")
    finally:
        plt.close(fig)


# ---------------------------------------------------------------------------
# STEP 6 — Write bin-level CSV evidence
# ---------------------------------------------------------------------------
def write_reliability_csv(tier1_bins, tier2_bins):
    """
    Write both tiers' non-empty bin data to a single CSV, distinguished by the
    'tier' column, for citation as versioned evidence.
    """
    _banner("STEP 6 — Writing bin-level reliability data CSV")

    rows = []
    for d in tier1_bins:
        rows.append({
            "tier": 1,
            "bin_lower": d["bin_lower"],
            "bin_upper": d["bin_upper"],
            "mean_predicted_confidence": d["mean_predicted_confidence"],
            "observed_accuracy": d["observed_accuracy"],
            "n_tickets_in_bin": d["n_tickets_in_bin"],
        })
    for d in tier2_bins:
        rows.append({
            "tier": 2,
            "bin_lower": d["bin_lower"],
            "bin_upper": d["bin_upper"],
            "mean_predicted_confidence": d["mean_predicted_confidence"],
            "observed_accuracy": d["observed_accuracy"],
            "n_tickets_in_bin": d["n_tickets_in_bin"],
        })

    out_df = pd.DataFrame(rows, columns=[
        "tier", "bin_lower", "bin_upper", "mean_predicted_confidence",
        "observed_accuracy", "n_tickets_in_bin",
    ])

    try:
        out_df.to_csv(RELIABILITY_CSV, index=False)
        print(f"[step6] Wrote {len(out_df)} bin row(s) to: {RELIABILITY_CSV}")
    except Exception as exc:
        print(f"[step6] WARNING: could not write reliability CSV "
              f"{RELIABILITY_CSV}: {exc!r}")


# ---------------------------------------------------------------------------
# STEP 7 — Final summary
# ---------------------------------------------------------------------------
def print_final_summary(tier1_ece, tier2_ece):
    """Print both tiers' ECE side by side and declare the better-calibrated one."""
    _banner("STEP 7 — Final calibration summary")

    print("[summary] Reliability diagrams were computed against the real "
          "500-ticket")
    print("[summary] production batch (NOT the training / held-out split).")
    print("[summary]")
    print("[summary] SCOPE NOTE: Tier-1 calibration here is evaluated as if "
          "Tier-1")
    print("[summary] classified EVERY ticket, independent of the 0.50 cascade "
          "threshold.")
    print("[summary] In the live cascade, Tier-1's confidence only 'counts' "
          "when it")
    print("[summary] resolves a ticket directly. This diagnostic deliberately "
          "audits")
    print("[summary] Tier-1's RAW confidence honesty, not live cascade routing "
          "behavior.")
    print("[summary]")
    print(f"[summary]   Tier-1 (TF-IDF)  ECE = {tier1_ece:.4f}")
    print(f"[summary]   Tier-2 (MiniLM)  ECE = {tier2_ece:.4f}")
    print("[summary]")

    if np.isnan(tier1_ece) and np.isnan(tier2_ece):
        print("[summary] Both ECE values are undefined — cannot compare.")
    elif np.isnan(tier1_ece):
        print("[summary] Tier-1 ECE undefined; Tier-2 is the only usable "
              "measurement.")
    elif np.isnan(tier2_ece):
        print("[summary] Tier-2 ECE undefined; Tier-1 is the only usable "
              "measurement.")
    elif tier1_ece < tier2_ece:
        print(f"[summary] BETTER CALIBRATED: Tier-1 (TF-IDF), by "
              f"{tier2_ece - tier1_ece:.4f} ECE.")
    elif tier2_ece < tier1_ece:
        print(f"[summary] BETTER CALIBRATED: Tier-2 (MiniLM), by "
              f"{tier1_ece - tier2_ece:.4f} ECE.")
    else:
        print("[summary] Both tiers are equally calibrated (identical ECE).")

    print("[summary]")
    print("[summary] Lower ECE = better calibrated. 0.0 = perfect calibration.")
    print("[summary] Artifacts written:")
    print(f"[summary]   - {TIER1_PNG}")
    print(f"[summary]   - {TIER2_PNG}")
    print(f"[summary]   - {RELIABILITY_CSV}")


# ---------------------------------------------------------------------------
# MAIN
# ---------------------------------------------------------------------------
def main():
    _banner("plot_calibration_curves.py — Cascade Reliability Diagram Builder")
    print(f"[main] Project root: {PROJECT_ROOT}")
    print(f"[main] Data dir:     {DATA_DIR}")
    print(f"[main] Models dir:   {MODELS_DIR}")

    # Resolve reusable Tier-1 functions up front so import problems fail early.
    train_tier1, get_tier1_confidence = _import_tier1_functions()

    # STEP 1
    eval_df, gt_full = assemble_evaluation_set()

    # STEP 2
    tier1_pred, tier1_conf = recompute_tier1(
        eval_df, gt_full, train_tier1, get_tier1_confidence
    )

    # STEP 3
    tier2_pred, tier2_conf = recompute_tier2(eval_df)

    true_labels = eval_df["true_category"].tolist()

    # STEP 4 (per tier)
    tier1_bins, tier1_ece = build_reliability_bins(
        tier1_pred, tier1_conf, true_labels, tier_number=1
    )
    tier2_bins, tier2_ece = build_reliability_bins(
        tier2_pred, tier2_conf, true_labels, tier_number=2
    )

    # STEP 5 (per tier — separate figures)
    plot_reliability_diagram(
        tier1_bins, tier1_ece, tier_number=1,
        tier_label="Tier-1 (TF-IDF)", out_path=TIER1_PNG,
    )
    plot_reliability_diagram(
        tier2_bins, tier2_ece, tier_number=2,
        tier_label="Tier-2 (MiniLM)", out_path=TIER2_PNG,
    )

    # STEP 6
    write_reliability_csv(tier1_bins, tier2_bins)

    # STEP 7
    print_final_summary(tier1_ece, tier2_ece)

    _banner("DONE — calibration reliability analysis complete")


if __name__ == "__main__":
    try:
        main()
    except SystemExit:
        # _fatal() already printed a clean message; propagate the exit code.
        raise
    except Exception as exc:
        # Absolute last-resort guard so the user never sees a raw traceback
        # dumped without context (house 'no raw tracebacks' convention).
        print("\n" + "!" * 70)
        print("UNEXPECTED ERROR — the script stopped before completing.")
        print(f"  {type(exc).__name__}: {exc}")
        print("  (Full trace below for debugging; see messages above for the "
              "last successful step.)")
        print("!" * 70)
        traceback.print_exc()
        sys.exit(1)
