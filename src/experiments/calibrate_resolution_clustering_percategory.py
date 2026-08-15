# src/experiments/calibrate_resolution_clustering_percategory.py
"""
Per-category calibration of resolution-text clustering thresholds.

The production automation-flagging feature uses ONE pooled threshold (0.80)
for resolution-text clustering, derived by pooling all 7 categories together
(precision cliff-edge: precision=1.0000 down to 0.80, then collapses at 0.75).

This DIAGNOSTIC script re-runs the SAME clustering + evaluation methodology
PER CATEGORY INDEPENDENTLY, to check whether every category's own cliff-edge
lands at/near 0.80, or whether some categories (e.g. Database, with more
bespoke/varied fixes) would need a different threshold.

MEASUREMENT ONLY. This does not change production. It reports findings for a
later human decision.
"""

import os
import sys
import csv
import random
import traceback
import itertools

import numpy as np
import pandas as pd


# ---------------------------------------------------------------------------
# Constants (verified source of truth)
# ---------------------------------------------------------------------------
MODEL_NAME = "all-MiniLM-L6-v2"
SIMILARITY_THRESHOLDS = [0.99, 0.95, 0.90, 0.85, 0.80, 0.75, 0.70, 0.65, 0.60]
CATEGORIES = [
    "Infrastructure",
    "Application",
    "Security",
    "Database",
    "Storage",
    "Network",
    "Access Management",  # literal space in last
]

POOLED_THRESHOLD = 0.80

# Categories flagged as lower-confidence due to small sample size.
# Set to 60 so it flags exactly Database (n=49) and Access Management (n=48)
# -- the two categories called out as bespoke/small in the project handoff --
# without also flagging Security (61) or Storage (53), which are close
# enough to the pooled 500-ticket scale not to need a caveat.
SMALL_N_BAR = 60

# Project root is TWO directories up from src/experiments/
_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(_THIS_DIR, os.pardir, os.pardir))

DATA_DIR = os.path.join(PROJECT_ROOT, "data")
INPUT_CSV = os.path.join(DATA_DIR, "category_stores_with_scenario_id.csv")
OUTPUT_CSV = os.path.join(
    DATA_DIR, "resolution_clustering_calibration_percategory.csv"
)
SUMMARY_CSV = os.path.join(
    DATA_DIR, "resolution_clustering_calibration_percategory_summary.csv"
)

REQUIRED_COLUMNS = ["batch_ticket_id", "category", "resolution_text", "scenario_id"]

RULE = "=" * 78


# ---------------------------------------------------------------------------
# Expected-error type (for clean, actionable messages -- no raw traceback)
# ---------------------------------------------------------------------------
class CalibrationError(Exception):
    """Raised for expected, user-actionable failures (missing file, missing
    column, etc.). Caught in __main__ and printed cleanly without a traceback."""
    pass


# ---------------------------------------------------------------------------
# Banner-style printing helpers (match project convention)
# ---------------------------------------------------------------------------
def banner(title):
    print()
    print(RULE)
    print(title)
    print(RULE)


def step(msg):
    print(f"[*] {msg}")


def ok(msg):
    print(f"[OK] {msg}")


def warn(msg):
    print(f"[!] {msg}")


# ---------------------------------------------------------------------------
# Clustering method (replicated identically from
# src/experiments/explore_resolution_clustering.py)
# ---------------------------------------------------------------------------
def cosine_similarity_matrix(embeddings):
    """L2-normalize EXPLICITLY here (encode() does NOT normalize), then
    cosine sim == dot product of normalized vectors."""
    norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
    norms[norms == 0.0] = 1.0
    normalized = embeddings / norms
    return normalized @ normalized.T


class _UnionFind:
    def __init__(self, n):
        self.parent = list(range(n))

    def find(self, x):
        while self.parent[x] != x:
            self.parent[x] = self.parent[self.parent[x]]
            x = self.parent[x]
        return x

    def union(self, a, b):
        ra, rb = self.find(a), self.find(b)
        if ra != rb:
            self.parent[rb] = ra


def group_by_threshold(sim_matrix, threshold):
    """Connected components via union-find: any pair with cosine sim >=
    threshold gets linked; connected components = clusters. Returns list of
    clusters (each a list of row indices), sorted largest-first.

    CRITICAL: this exact method (NOT agglomerative/linkage-based clustering)
    is the project's established convention for this calibration."""
    n = sim_matrix.shape[0]
    uf = _UnionFind(n)
    for i in range(n):
        row = sim_matrix[i]
        for j in range(i + 1, n):
            if row[j] >= threshold:
                uf.union(i, j)
    comps = {}
    for i in range(n):
        root = uf.find(i)
        comps.setdefault(root, []).append(i)
    clusters = list(comps.values())
    clusters.sort(key=lambda c: (-len(c), min(c)))
    return clusters


# ---------------------------------------------------------------------------
# Pairwise evaluation metric (replicated from
# src/experiments/calibrate_resolution_clustering.py)
# ---------------------------------------------------------------------------
def compute_pairwise_metrics(tickets, scenario_of, cluster_of):
    """
    tickets: list of batch_ticket_id strings for this category's population.
    scenario_of: dict batch_ticket_id -> scenario_id (ground truth).
    cluster_of: dict batch_ticket_id -> cluster_id.

    For every pair (a, b) via itertools.combinations(tickets, 2):
        gt_positive   = scenario_of[a] == scenario_of[b]
        pred_positive = cluster_of[a]  == cluster_of[b]
        TP: pred_positive AND gt_positive
        FP: pred_positive AND NOT gt_positive  (grouped different scenarios)
        FN: gt_positive AND NOT pred_positive  (missed a real same-scenario pair)

    precision = TP/(TP+FP) if denom>0 else None
    recall    = TP/(TP+FN) if denom>0 else None
    F1        = harmonic mean, None if either input is None or precision+recall==0
    Returns dict: tp, fp, fn, precision, recall, f1
    """
    tp = fp = fn = 0
    for a, b in itertools.combinations(tickets, 2):
        gt_positive = scenario_of[a] == scenario_of[b]
        pred_positive = cluster_of[a] == cluster_of[b]
        if pred_positive and gt_positive:
            tp += 1
        elif pred_positive and not gt_positive:
            fp += 1
        elif gt_positive and not pred_positive:
            fn += 1

    prec_denom = tp + fp
    rec_denom = tp + fn
    precision = (tp / prec_denom) if prec_denom > 0 else None
    recall = (tp / rec_denom) if rec_denom > 0 else None

    if precision is None or recall is None or (precision + recall) == 0:
        f1 = None
    else:
        f1 = 2.0 * precision * recall / (precision + recall)

    return {"tp": tp, "fp": fp, "fn": fn,
            "precision": precision, "recall": recall, "f1": f1}


# ---------------------------------------------------------------------------
# Cliff-edge finding (same definition as pooled case)
# ---------------------------------------------------------------------------
def find_cliff_edge(rows_by_threshold):
    cliff = None
    for thr in SIMILARITY_THRESHOLDS:  # tight -> loose
        prec = rows_by_threshold[thr]["precision"]
        if prec is None:
            # No predicted-positive pairs at this (very tight) threshold --
            # undefined precision is not evidence of a drop. Skip without
            # advancing the cliff, but keep scanning looser thresholds.
            continue
        if prec == 1.0:
            cliff = thr
        else:
            break  # a real, defined precision < 1.0 -- genuine drop
    return cliff


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------
def load_data():
    step(f"Loading data: {INPUT_CSV}")
    if not os.path.isfile(INPUT_CSV):
        raise CalibrationError(
            f"Input CSV not found:\n    {INPUT_CSV}\n"
            f"Expected it under the project's data/ directory. Project root was "
            f"resolved to:\n    {PROJECT_ROOT}\n"
            f"(project root convention: two directories up from src/experiments/)"
        )

    try:
        df = pd.read_csv(INPUT_CSV)
    except Exception as exc:
        raise CalibrationError(f"Failed to read CSV with pandas: {exc}")

    missing = [c for c in REQUIRED_COLUMNS if c not in df.columns]
    if missing:
        raise CalibrationError(
            f"Input CSV is missing required column(s): {missing}\n"
            f"Found columns: {list(df.columns)}\n"
            f"Required: {REQUIRED_COLUMNS}"
        )

    # Cast batch_ticket_id to str for consistent joining (exact project pattern).
    df["batch_ticket_id"] = df["batch_ticket_id"].astype(str)

    ok(f"Loaded {len(df)} rows.")
    return df


# ---------------------------------------------------------------------------
# Per-category processing
# ---------------------------------------------------------------------------
def process_category(model, df, category):
    """Run the full sweep for one category. Returns:
        (per_threshold_rows, cliff_edge, sample_size, low_confidence)
    where per_threshold_rows is a list of dicts matching the CSV shape."""
    banner(f"CATEGORY: {category}")

    cat_df = df[df["category"] == category].copy()

    # Skip rows with empty resolution_text (nothing to embed).
    cat_df["resolution_text"] = cat_df["resolution_text"].astype("string")
    mask_nonempty = cat_df["resolution_text"].notna() & (
        cat_df["resolution_text"].str.strip() != ""
    )
    dropped = int((~mask_nonempty).sum())
    cat_df = cat_df[mask_nonempty].reset_index(drop=True)

    n = len(cat_df)
    low_confidence = n < SMALL_N_BAR

    step(f"Sample size (non-empty resolution_text): n = {n}")
    if dropped > 0:
        warn(f"Skipped {dropped} row(s) with empty resolution_text.")
    if low_confidence:
        warn(
            f"LOW-CONFIDENCE CATEGORY: n = {n} (< {SMALL_N_BAR}). "
            f"Small-n cliff-edges are noisier than the pooled 500-ticket result. "
            f"Treat this category's cliff-edge as indicative, not definitive."
        )

    per_threshold_rows = []

    if n < 2:
        warn(
            f"Only {n} usable ticket(s) -- cannot form pairs. "
            f"Emitting empty/None metric rows for all thresholds."
        )
        for thr in SIMILARITY_THRESHOLDS:
            per_threshold_rows.append({
                "category": category, "threshold": thr, "n_tickets": n,
                "n_clusters": n, "tp": 0, "fp": 0, "fn": 0,
                "precision": None, "recall": None, "f1": None,
            })
        return per_threshold_rows, None, n, low_confidence

    tickets = cat_df["batch_ticket_id"].tolist()
    texts = cat_df["resolution_text"].tolist()
    scenario_of = dict(zip(tickets, cat_df["scenario_id"].tolist()))

    step("Encoding resolution_text with SentenceTransformer...")
    embeddings = model.encode(
        texts,
        batch_size=64,
        convert_to_numpy=True,
        show_progress_bar=True,
    )
    # NO normalization at encode time; done explicitly in cosine_similarity_matrix.
    ok(f"Encoded {len(texts)} texts -> embeddings shape {embeddings.shape}")

    step("Sweeping similarity thresholds (0.99 -> 0.60)...")
    rows_by_threshold = {}

    # sim_matrix does not depend on threshold -- compute once per category.
    sim_matrix = cosine_similarity_matrix(embeddings)

    for thr in SIMILARITY_THRESHOLDS:
        clusters = group_by_threshold(sim_matrix, thr)

        # Build cluster_of: each cluster's cluster_id = its index in the
        # returned clusters list; map every member's batch_ticket_id to it.
        cluster_of = {}
        for cluster_id, member_indices in enumerate(clusters):
            for row_idx in member_indices:
                cluster_of[tickets[row_idx]] = cluster_id

        metrics = compute_pairwise_metrics(tickets, scenario_of, cluster_of)

        row = {
            "category": category,
            "threshold": thr,
            "n_tickets": n,
            "n_clusters": len(clusters),
            "tp": metrics["tp"],
            "fp": metrics["fp"],
            "fn": metrics["fn"],
            "precision": metrics["precision"],
            "recall": metrics["recall"],
            "f1": metrics["f1"],
        }
        per_threshold_rows.append(row)
        rows_by_threshold[thr] = metrics

        print(
            f"    thr={thr:.2f}  clusters={len(clusters):4d}  "
            f"tp={metrics['tp']:6d} fp={metrics['fp']:6d} fn={metrics['fn']:6d}  "
            f"P={_fmt(metrics['precision'])} "
            f"R={_fmt(metrics['recall'])} "
            f"F1={_fmt(metrics['f1'])}"
        )

    cliff_edge = find_cliff_edge(rows_by_threshold)
    if cliff_edge is None:
        warn("No threshold achieved precision == 1.0000 (no cliff-edge found).")
    else:
        step(f"Cliff-edge (loosest threshold at precision=1.0000): {cliff_edge:.2f}")

    return per_threshold_rows, cliff_edge, n, low_confidence


def _fmt(v):
    """Format a metric value that may be None."""
    if v is None:
        return "  None"
    return f"{v:.4f}"


def _match_flag(cliff_edge):
    """Return the MATCHES / DIFFERS flag string vs the pooled 0.80 threshold."""
    if cliff_edge is None:
        return "NO CLIFF-EDGE (n/a vs pooled 0.80)"
    if abs(cliff_edge - POOLED_THRESHOLD) < 1e-9:
        return "MATCHES POOLED 0.80"
    return "DIFFERS FROM POOLED 0.80"


# ---------------------------------------------------------------------------
# Output: per-category table + two CSVs (per-threshold detail, summary) +
# final summary
# ---------------------------------------------------------------------------
def print_percategory_table(summaries):
    """summaries: list of dicts with keys category, n, cliff_edge,
    cliff_metrics (dict or None), low_confidence."""
    banner("PER-CATEGORY CLIFF-EDGE TABLE")
    header = (
        f"{'category':<20} {'n':>5} {'cliff':>6} "
        f"{'P@cliff':>8} {'R@cliff':>8} {'F1@cliff':>9}  {'flag'}"
    )
    print(header)
    print("-" * len(header))
    for s in summaries:
        cm = s["cliff_metrics"]
        p = _fmt(cm["precision"]) if cm else "  None"
        r = _fmt(cm["recall"]) if cm else "  None"
        f1 = _fmt(cm["f1"]) if cm else "  None"
        cliff = f"{s['cliff_edge']:.2f}" if s["cliff_edge"] is not None else "  n/a"
        lc = "  [LOW-N]" if s["low_confidence"] else ""
        print(
            f"{s['category']:<20} {s['n']:>5} {cliff:>6} "
            f"{p:>8} {r:>8} {f1:>9}  {_match_flag(s['cliff_edge'])}{lc}"
        )


def write_csv(all_rows):
    """Writes ONLY the clean per-(category,threshold) detail rows -- no
    appended summary block. Kept as pure tabular data so it loads cleanly
    with pd.read_csv() in any downstream plotting/analysis script."""
    step(f"Writing CSV: {OUTPUT_CSV}")
    os.makedirs(DATA_DIR, exist_ok=True)

    fieldnames = [
        "category", "threshold", "n_tickets", "n_clusters",
        "tp", "fp", "fn", "precision", "recall", "f1",
    ]
    try:
        with open(OUTPUT_CSV, "w", newline="", encoding="utf-8") as fh:
            writer = csv.DictWriter(fh, fieldnames=fieldnames)
            writer.writeheader()

            for row in all_rows:
                out = dict(row)
                for key in ("precision", "recall", "f1"):
                    if out[key] is None:
                        out[key] = ""
                writer.writerow(out)
    except OSError as exc:
        raise CalibrationError(f"Failed to write output CSV: {exc}")

    ok(f"Wrote {len(all_rows)} per-(category,threshold) rows.")


def write_summary_csv(summaries):
    """Writes a SEPARATE, properly-shaped summary CSV: one row per category,
    with its own cliff-edge, metrics-at-cliff, and match/differ flag vs the
    pooled 0.80 threshold. Kept apart from the per-threshold detail CSV so
    neither file mixes shapes."""
    step(f"Writing summary CSV: {SUMMARY_CSV}")
    os.makedirs(DATA_DIR, exist_ok=True)

    fieldnames = [
        "category", "n_tickets", "cliff_edge",
        "precision_at_cliff", "recall_at_cliff", "f1_at_cliff",
        "match_vs_pooled_0.80",
    ]
    try:
        with open(SUMMARY_CSV, "w", newline="", encoding="utf-8") as fh:
            writer = csv.DictWriter(fh, fieldnames=fieldnames)
            writer.writeheader()

            for s in summaries:
                cm = s["cliff_metrics"]
                flag = _match_flag(s["cliff_edge"])
                if s["low_confidence"]:
                    flag += " [LOW-N]"
                writer.writerow({
                    "category": s["category"],
                    "n_tickets": s["n"],
                    "cliff_edge": "" if s["cliff_edge"] is None else f"{s['cliff_edge']:.2f}",
                    "precision_at_cliff": _fmt(cm["precision"]) if cm else "",
                    "recall_at_cliff": _fmt(cm["recall"]) if cm else "",
                    "f1_at_cliff": _fmt(cm["f1"]) if cm else "",
                    "match_vs_pooled_0.80": flag,
                })
    except OSError as exc:
        raise CalibrationError(f"Failed to write summary CSV: {exc}")

    ok(f"Wrote {len(summaries)} category summary rows.")


def print_final_summary(summaries):
    banner("FINAL SUMMARY -- PER-CATEGORY vs POOLED 0.80")
    print(f"Pooled production threshold: {POOLED_THRESHOLD:.2f}")
    print()

    differing = []
    matching = []
    no_cliff = []
    for s in summaries:
        if s["cliff_edge"] is None:
            no_cliff.append(s)
        elif abs(s["cliff_edge"] - POOLED_THRESHOLD) < 1e-9:
            matching.append(s)
        else:
            differing.append(s)

    if matching:
        print("Categories whose own cliff-edge MATCHES pooled 0.80:")
        for s in matching:
            lc = " [LOW-N -- lower confidence]" if s["low_confidence"] else ""
            print(f"    - {s['category']} (n={s['n']}, cliff={s['cliff_edge']:.2f}){lc}")
        print()

    if differing:
        print("Categories whose own cliff-edge DIFFERS from pooled 0.80:")
        for s in differing:
            lc = " [LOW-N -- lower confidence]" if s["low_confidence"] else ""
            print(
                f"    - {s['category']} (n={s['n']}, cliff={s['cliff_edge']:.2f}, "
                f"delta={s['cliff_edge'] - POOLED_THRESHOLD:+.2f}){lc}"
            )
        print()
    else:
        print("No category differs from the pooled 0.80 cliff-edge.")
        print()

    if no_cliff:
        print("Categories with NO cliff-edge found (precision never exactly 1.0):")
        for s in no_cliff:
            lc = " [LOW-N -- lower confidence]" if s["low_confidence"] else ""
            print(f"    - {s['category']} (n={s['n']}){lc}")
        print()

    print(
        "NOTE: This is a measurement-only diagnostic. It does NOT change the\n"
        "production pooled threshold. Low-N categories (flagged above) carry\n"
        "less statistical confidence than the pooled 500-ticket calibration and\n"
        "should be weighed accordingly in any later decision."
    )


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def run():
    random.seed(42)
    np.random.seed(42)

    banner("PER-CATEGORY RESOLUTION-CLUSTERING CALIBRATION (DIAGNOSTIC)")
    print(f"Project root : {PROJECT_ROOT}")
    print(f"Input CSV    : {INPUT_CSV}")
    print(f"Output CSV   : {OUTPUT_CSV}")
    print(f"Summary CSV  : {SUMMARY_CSV}")
    print(f"Model        : {MODEL_NAME}")
    print(f"Thresholds   : {SIMILARITY_THRESHOLDS}")
    print(f"Pooled thr   : {POOLED_THRESHOLD}")
    print(f"Small-n bar  : {SMALL_N_BAR}")

    df = load_data()

    # Import here so a missing dependency produces a clean, actionable message
    # rather than failing at module import.
    step("Loading SentenceTransformer model (local/offline)...")
    try:
        from sentence_transformers import SentenceTransformer
    except ImportError as exc:
        raise CalibrationError(
            f"sentence-transformers is not installed in this venv: {exc}\n"
            f"Activate the project venv and install requirements, e.g.:\n"
            f"    pip install sentence-transformers"
        )

    try:
        model = SentenceTransformer(MODEL_NAME)
    except Exception as exc:
        raise CalibrationError(
            f"Failed to load model '{MODEL_NAME}': {exc}\n"
            f"Ensure the model is cached locally (fully offline run expected)."
        )
    ok("Model loaded.")

    # Warn about any expected categories entirely absent from the data.
    present = set(df["category"].unique().tolist())
    for cat in CATEGORIES:
        if cat not in present:
            warn(f"Category '{cat}' not present in data -- will report n=0.")

    all_rows = []
    summaries = []
    for category in CATEGORIES:
        rows, cliff_edge, n, low_conf = process_category(model, df, category)
        all_rows.extend(rows)

        cliff_metrics = None
        if cliff_edge is not None:
            for r in rows:
                if abs(r["threshold"] - cliff_edge) < 1e-9:
                    cliff_metrics = {
                        "precision": r["precision"],
                        "recall": r["recall"],
                        "f1": r["f1"],
                    }
                    break

        summaries.append({
            "category": category,
            "n": n,
            "cliff_edge": cliff_edge,
            "cliff_metrics": cliff_metrics,
            "low_confidence": low_conf,
        })

    print_percategory_table(summaries)
    write_csv(all_rows)
    write_summary_csv(summaries)
    print_final_summary(summaries)

    banner("DONE")


def main():
    try:
        run()
    except CalibrationError as exc:
        # Expected, user-actionable failure: clean message, no traceback.
        print()
        print(RULE)
        print("ERROR (expected / actionable)")
        print(RULE)
        print(str(exc))
        sys.exit(1)
    except Exception:
        # Genuinely unexpected error: dump full traceback for debugging.
        print()
        print(RULE)
        print("UNEXPECTED ERROR -- full traceback follows")
        print(RULE)
        traceback.print_exc()
        sys.exit(2)


if __name__ == "__main__":
    main()