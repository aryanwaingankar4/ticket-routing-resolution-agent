# src/experiments/flag_automation_candidates.py
"""
flag_automation_candidates.py
=============================

PRODUCTION automation-candidate flagger for the
"AI-Powered Intelligent Ticket Routing & Resolution Agent" project.

WHAT THIS SCRIPT IS
-------------------
This is the production payoff of an earlier calibration effort. That
calibration clustered ticket RESOLUTION TEXT (not the ticket symptom/title)
per category using cosine similarity on MiniLM ("all-MiniLM-L6-v2")
embeddings, sweeping a cosine-similarity threshold from 0.99 down to 0.60.
Pairwise precision/recall against real scenario_id ground truth revealed a
cliff-edge: precision held at 1.0000 all the way down to threshold=0.80, then
collapsed sharply below it. threshold=0.80 was deliberately chosen over the
recall-better 0.75 because a FALSE POSITIVE here (wrongly telling ops that two
different underlying problems share the same fix) is costlier than a FALSE
NEGATIVE (a missed automation opportunity, which just means the status quo
continues).

That finding is already validated. This script does NOT re-derive, re-test, or
sweep the threshold. It implements ONE fixed, calibrated threshold (0.80) in a
real, runnable feature: for each ticket category, it clusters resolution texts
and flags clusters of size >= 2 as "automation candidates" — repeated fixes
that a human should review as candidates for automation.

WHAT IT READS
-------------
Per-category CSV pipeline logs at:
    data/category_stores/{Category}.csv
Each CSV has at least the columns:
    batch_ticket_id, original_id, category, resolution_status, resolution_text
Rows with empty/missing resolution_text are skipped (nothing to embed).
NOTE: the production CSV format does NOT include a scenario_id column. This
script never reads, references, or requires scenario_id or any ground-truth
file — production has no ground truth available at run time.

WHAT IT WRITES
--------------
    data/automation_candidates.json
        Structured, full-detail output: _meta block plus one key per category
        that has at least one flagged cluster. Categories with zero flagged
        clusters are OMITTED from the per-category keys (documented in _meta).
    data/automation_candidates_summary.csv
        One row per flagged cluster for easy citation/review.

Both artifacts live under data/ and are intended to be VERSIONED (not
gitignored) because they are cited evidence for a research paper.

THE FIXED THRESHOLD
-------------------
    RESOLUTION_SIMILARITY_THRESHOLD = 0.80
DO NOT CHANGE without re-running calibration. See:
    data/resolution_clustering_calibration_results.csv
    src/experiments/calibrate_resolution_clustering.py
Precision was 1.0000 at this threshold against 500 real tickets' ground truth.
Looser thresholds are NOT calibrated and must not be used here. The threshold
is intentionally NOT exposed as a CLI argument.

USAGE
-----
    python src/experiments/flag_automation_candidates.py
    python src/experiments/flag_automation_candidates.py --input-dir data/category_stores
    python src/experiments/flag_automation_candidates.py --output-json data/automation_candidates.json
"""

import argparse
import csv
import json
import os
import sys

import numpy as np
from sentence_transformers import SentenceTransformer


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

# The model used at calibration time. Must match the calibration run exactly
# so this production script's clustering behavior is IDENTICAL to what was
# calibrated.
MODEL_NAME = "all-MiniLM-L6-v2"

# Batch size used at encode() time (matches the exploratory/calibration script).
ENCODE_BATCH_SIZE = 64

# =====================================================================
# FIXED, CALIBRATED THRESHOLD -- DO NOT CHANGE.
# ---------------------------------------------------------------------
# DO NOT CHANGE without re-running calibration. See:
#     data/resolution_clustering_calibration_results.csv
#     src/experiments/calibrate_resolution_clustering.py
# Precision was 1.0000 at this threshold against 500 real tickets' ground
# truth. Looser thresholds (e.g. 0.75) achieve better recall but are NOT
# calibrated and must not be used here. This value is deliberately NOT a CLI
# argument, so it cannot be trivially overridden -- it is an evidence-backed,
# calibrated constant, not a tunable knob.
# =====================================================================
RESOLUTION_SIMILARITY_THRESHOLD = 0.80

# Category list and exact filenames. Note "Access Management" contains a
# literal space in the filename.
CATEGORIES = [
    "Infrastructure",
    "Application",
    "Security",
    "Database",
    "Storage",
    "Network",
    "Access Management",
]

# Only clusters of at least this many members are "automation candidates".
# A cluster of 1 is not a repeated pattern -- nothing to flag.
MIN_CLUSTER_SIZE = 2

# Truncation length for the CSV summary's representative resolution text,
# matching this project's existing truncation convention (append "..." on cut).
SUMMARY_TRUNCATE_CHARS = 150

# Project root is two directories up from this file
# (src/experiments/ -> src/ -> project root).
PROJECT_ROOT = os.path.abspath(
    os.path.join(os.path.dirname(__file__), os.pardir, os.pardir)
)

# Default I/O locations, all built via os.path.* (never hardcoded slashes).
DEFAULT_INPUT_DIR = os.path.join(PROJECT_ROOT, "data", "category_stores")
DEFAULT_OUTPUT_JSON = os.path.join(PROJECT_ROOT, "data", "automation_candidates.json")
DEFAULT_OUTPUT_CSV = os.path.join(
    PROJECT_ROOT, "data", "automation_candidates_summary.csv"
)
CALIBRATION_SOURCE_REL = "data/resolution_clustering_calibration_results.csv"


# ---------------------------------------------------------------------------
# Core clustering logic (IDENTICAL math/grouping to the calibrated script).
# ---------------------------------------------------------------------------

def cosine_similarity_matrix(embeddings):
    """Explicit L2 normalization, then cosine == dot product.

    Encoding is done WITHOUT normalization at encode() time; normalization is
    performed here, separately, exactly as in the calibrated script.
    """
    norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
    norms[norms == 0.0] = 1.0  # guard against zero-vectors
    normalized = embeddings / norms
    return normalized @ normalized.T


class _UnionFind:
    """Union-find (disjoint set) with path halving, used for connected
    components. Identical to the calibrated script."""

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
    """Connected components via union-find.

    Any two tickets with cosine similarity >= threshold are linked; connected
    components become clusters. Clusters are sorted largest-first, ties broken
    by smallest member index. This matches the calibrated grouping exactly.
    """
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
# Helpers
# ---------------------------------------------------------------------------

def collapse_whitespace(text):
    """Collapse all runs of whitespace (including embedded newlines) into a
    single space and strip the ends."""
    if text is None:
        return ""
    return " ".join(text.split())


def truncate_for_summary(text, limit=SUMMARY_TRUNCATE_CHARS):
    """Collapse whitespace and truncate to `limit` chars, appending '...' when
    the text was actually cut (project truncation convention)."""
    collapsed = collapse_whitespace(text)
    if len(collapsed) <= limit:
        return collapsed
    return collapsed[:limit] + "..."


def category_csv_path(input_dir, category):
    """Build the CSV path for a category via os.path.join (handles the literal
    space in 'Access Management' correctly)."""
    return os.path.join(input_dir, category + ".csv")


def load_category_rows(csv_path):
    """Load all rows with non-empty resolution_text from a category CSV.

    Returns a list of dicts, each with keys: batch_ticket_id, original_id,
    resolution_text. Rows missing/empty resolution_text are skipped. The CSV
    reader handles embedded newlines within quoted fields natively.
    """
    rows = []
    skipped_empty = 0
    with open(csv_path, "r", encoding="utf-8", newline="") as fh:
        reader = csv.DictReader(fh)
        if reader.fieldnames is None:
            return rows, skipped_empty
        if "resolution_text" not in reader.fieldnames:
            # No resolution_text column at all -- treat as unusable.
            print(
                "    WARNING: no 'resolution_text' column found; "
                "columns present: {}".format(reader.fieldnames)
            )
            return rows, skipped_empty
        for record in reader:
            resolution_text = record.get("resolution_text")
            if resolution_text is None or resolution_text.strip() == "":
                skipped_empty += 1
                continue
            rows.append(
                {
                    "batch_ticket_id": (record.get("batch_ticket_id") or "").strip(),
                    "original_id": (record.get("original_id") or "").strip(),
                    "resolution_text": resolution_text,
                }
            )
    return rows, skipped_empty


def most_central_member_index(sim_matrix, member_indices):
    """Return the index (into `member_indices`) of the member whose embedding
    is most central to the cluster, i.e. highest average cosine similarity to
    the OTHER members. For a 2-member cluster both are equally central, so the
    first (smallest global index, given the sort) is returned.
    """
    if len(member_indices) == 1:
        return 0
    best_local = 0
    best_avg = None
    n_others = len(member_indices) - 1
    for local_i, global_i in enumerate(member_indices):
        total = 0.0
        for local_j, global_j in enumerate(member_indices):
            if local_i == local_j:
                continue
            total += float(sim_matrix[global_i][global_j])
        avg = total / n_others
        if best_avg is None or avg > best_avg:
            best_avg = avg
            best_local = local_i
    return best_local


# ---------------------------------------------------------------------------
# Per-category processing
# ---------------------------------------------------------------------------

def process_category(category, input_dir, model):
    """Process a single category end to end.

    Returns a dict with per-category stats and the list of flagged-cluster
    records, or None if the category was skipped (missing file / no usable
    rows). Never raises for expected "no data" conditions.
    """
    csv_path = category_csv_path(input_dir, category)
    print("-" * 70)
    print("CATEGORY: {}".format(category))
    print("  File: {}".format(csv_path))

    if not os.path.isfile(csv_path):
        print("  SKIP: CSV file does not exist. Skipping this category gracefully.")
        return None

    rows, skipped_empty = load_category_rows(csv_path)
    print(
        "  Loaded {} usable rows (skipped {} rows with empty/missing "
        "resolution_text).".format(len(rows), skipped_empty)
    )

    if len(rows) == 0:
        print("  SKIP: no usable rows with resolution_text. Skipping this category.")
        return None

    texts = [r["resolution_text"] for r in rows]

    # ---- Trivial single-ticket case: no clustering possible. ----
    if len(rows) == 1:
        print(
            "  Only 1 usable ticket -- trivially a singleton, no clustering "
            "possible. 0 flagged clusters, 1 singleton ticket."
        )
        return {
            "category": category,
            "tickets_processed": 1,
            "flagged_clusters": [],
            "num_flagged_clusters": 0,
            "num_flagged_tickets": 0,
            "num_singleton_tickets": 1,
        }

    # ---- Encode (NO normalization at encode() time). ----
    print(
        "  Encoding {} resolution texts with MiniLM ('{}'), "
        "batch_size={}...".format(len(texts), MODEL_NAME, ENCODE_BATCH_SIZE)
    )
    embeddings = model.encode(
        texts,
        batch_size=ENCODE_BATCH_SIZE,
        convert_to_numpy=True,
        show_progress_bar=True,
    )
    print("  Encoded. Embedding matrix shape: {}".format(embeddings.shape))

    # ---- Similarity (explicit L2 normalization, then dot). ----
    print("  Computing cosine similarity matrix (explicit L2 normalization)...")
    sim_matrix = cosine_similarity_matrix(embeddings)

    # ---- Cluster at the FIXED calibrated threshold. ----
    print(
        "  Clustering via connected components at FIXED threshold "
        "{:.2f} (calibrated, not swept)...".format(RESOLUTION_SIMILARITY_THRESHOLD)
    )
    clusters = group_by_threshold(sim_matrix, RESOLUTION_SIMILARITY_THRESHOLD)

    num_singleton_tickets = sum(1 for c in clusters if len(c) < MIN_CLUSTER_SIZE)
    flagged_source = [c for c in clusters if len(c) >= MIN_CLUSTER_SIZE]

    print(
        "  Found {} total connected components: {} flagged clusters "
        "(size >= {}) and {} singleton tickets.".format(
            len(clusters),
            len(flagged_source),
            MIN_CLUSTER_SIZE,
            num_singleton_tickets,
        )
    )

    # ---- Build flagged-cluster records. ----
    # `clusters` is already sorted largest-first (ties: smallest index), which
    # is exactly the ordering the ops-readable cluster_id should use. Assign
    # cluster_id in that same order, 0-indexed, over the FLAGGED clusters.
    flagged_records = []
    num_flagged_tickets = 0
    for cluster_id, member_indices in enumerate(flagged_source):
        size = len(member_indices)
        num_flagged_tickets += size

        members = [
            {
                "batch_ticket_id": rows[idx]["batch_ticket_id"],
                "original_id": rows[idx]["original_id"],
            }
            for idx in member_indices
        ]

        central_local = most_central_member_index(sim_matrix, member_indices)
        central_global = member_indices[central_local]
        representative_text = rows[central_global]["resolution_text"]

        flagged_records.append(
            {
                "cluster_id": cluster_id,
                "size": size,
                "members": members,
                "representative_resolution_text": representative_text,
            }
        )
        print(
            "    Cluster {}: size={}, tickets=[{}], representative "
            "batch_ticket_id={}".format(
                cluster_id,
                size,
                "; ".join(m["batch_ticket_id"] for m in members),
                rows[central_global]["batch_ticket_id"],
            )
        )

    return {
        "category": category,
        "tickets_processed": len(rows),
        "flagged_clusters": flagged_records,
        "num_flagged_clusters": len(flagged_records),
        "num_flagged_tickets": num_flagged_tickets,
        "num_singleton_tickets": num_singleton_tickets,
    }


# ---------------------------------------------------------------------------
# Output writers
# ---------------------------------------------------------------------------

def write_json_output(output_json, per_category_results, categories_processed):
    """Write data/automation_candidates.json.

    Per-category keys are only included when the category has at least one
    flagged cluster (categories with zero flagged clusters are OMITTED; this
    choice is documented in the _meta block).
    """
    total_tickets_processed = sum(
        r["tickets_processed"] for r in per_category_results
    )
    total_flagged_clusters = sum(
        r["num_flagged_clusters"] for r in per_category_results
    )
    total_flagged_tickets = sum(
        r["num_flagged_tickets"] for r in per_category_results
    )
    total_singleton_tickets = sum(
        r["num_singleton_tickets"] for r in per_category_results
    )

    output = {
        "_meta": {
            "note": (
                "PRODUCTION automation-flagging output. threshold=0.80 is "
                "FIXED and calibrated (see calibrate_resolution_clustering.py); "
                "this is not a sweep."
            ),
            "per_category_key_policy": (
                "Only categories with at least one flagged cluster (size >= "
                "{}) appear as top-level keys. Categories with zero flagged "
                "clusters are omitted.".format(MIN_CLUSTER_SIZE)
            ),
            "human_review_caution": (
                "The calibration's 1.0000 precision was measured against known "
                "scenario_id ground truth on the 500-ticket development set. "
                "Production tickets have no such ground truth at run time -- "
                "treat this flagging as a suggestion for human review, not an "
                "autonomous action, consistent with the calibration's "
                "precision-over-recall design choice."
            ),
            "model": MODEL_NAME,
            "threshold": RESOLUTION_SIMILARITY_THRESHOLD,
            "min_cluster_size": MIN_CLUSTER_SIZE,
            "calibration_source": CALIBRATION_SOURCE_REL,
            "generated_from": "data/category_stores/*.csv",
            "categories_processed": categories_processed,
            "total_tickets_processed": total_tickets_processed,
            "total_flagged_clusters": total_flagged_clusters,
            "total_flagged_tickets": total_flagged_tickets,
            "total_singleton_tickets": total_singleton_tickets,
        }
    }

    for result in per_category_results:
        if result["num_flagged_clusters"] > 0:
            output[result["category"]] = result["flagged_clusters"]

    out_dir = os.path.dirname(output_json)
    if out_dir and not os.path.isdir(out_dir):
        os.makedirs(out_dir, exist_ok=True)
    with open(output_json, "w", encoding="utf-8") as fh:
        json.dump(output, fh, indent=2, ensure_ascii=False)

    print("Wrote JSON output: {}".format(output_json))
    print(
        "  ({} categories with flagged clusters, {} total flagged "
        "clusters).".format(
            sum(1 for r in per_category_results if r["num_flagged_clusters"] > 0),
            total_flagged_clusters,
        )
    )


def write_csv_summary(output_csv, per_category_results):
    """Write data/automation_candidates_summary.csv -- one row per flagged
    cluster."""
    fieldnames = [
        "category",
        "cluster_id",
        "size",
        "batch_ticket_ids",
        "representative_resolution_text_truncated",
    ]

    out_dir = os.path.dirname(output_csv)
    if out_dir and not os.path.isdir(out_dir):
        os.makedirs(out_dir, exist_ok=True)

    rows_written = 0
    with open(output_csv, "w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        for result in per_category_results:
            for cluster in result["flagged_clusters"]:
                batch_ids = ";".join(
                    m["batch_ticket_id"] for m in cluster["members"]
                )
                writer.writerow(
                    {
                        "category": result["category"],
                        "cluster_id": cluster["cluster_id"],
                        "size": cluster["size"],
                        "batch_ticket_ids": batch_ids,
                        "representative_resolution_text_truncated": (
                            truncate_for_summary(
                                cluster["representative_resolution_text"]
                            )
                        ),
                    }
                )
                rows_written += 1

    print("Wrote CSV summary: {}".format(output_csv))
    print("  ({} flagged-cluster rows).".format(rows_written))


# ---------------------------------------------------------------------------
# Terminal summary table
# ---------------------------------------------------------------------------

def print_summary_table(per_category_results):
    """Print the final per-category summary table plus grand totals."""
    print("=" * 70)
    print("FINAL SUMMARY")
    print("=" * 70)

    header = "{:<20} | {:>10} | {:>9} | {:>9} | {:>10}".format(
        "category",
        "tickets",
        "clusters",
        "flagged",
        "singletons",
    )
    print(header)
    print("-" * len(header))

    total_tickets = 0
    total_clusters = 0
    total_flagged = 0
    total_singletons = 0

    for result in per_category_results:
        print(
            "{:<20} | {:>10} | {:>9} | {:>9} | {:>10}".format(
                result["category"],
                result["tickets_processed"],
                result["num_flagged_clusters"],
                result["num_flagged_tickets"],
                result["num_singleton_tickets"],
            )
        )
        total_tickets += result["tickets_processed"]
        total_clusters += result["num_flagged_clusters"]
        total_flagged += result["num_flagged_tickets"]
        total_singletons += result["num_singleton_tickets"]

    print("-" * len(header))
    print(
        "{:<20} | {:>10} | {:>9} | {:>9} | {:>10}".format(
            "GRAND TOTAL",
            total_tickets,
            total_clusters,
            total_flagged,
            total_singletons,
        )
    )
    print("")
    print(
        "NOTE: this calibration's 1.0000 precision was measured against known "
        "scenario_id"
    )
    print(
        "      ground truth on the 500-ticket development set. Production "
        "tickets have no"
    )
    print(
        "      such ground truth at run time -- this flagging should still be "
        "treated as a"
    )
    print(
        "      suggestion for human review, not an autonomous action, "
        "consistent with the"
    )
    print("      calibration's own precision-over-recall design choice.")


# ---------------------------------------------------------------------------
# CLI / main
# ---------------------------------------------------------------------------

def parse_args(argv=None):
    parser = argparse.ArgumentParser(
        description=(
            "Flag automation-candidate clusters of repeated ticket resolution "
            "texts per category, at the FIXED calibrated cosine threshold "
            "0.80. The threshold is intentionally NOT a CLI argument."
        )
    )
    parser.add_argument(
        "--input-dir",
        default=DEFAULT_INPUT_DIR,
        help=(
            "Directory containing per-category CSVs ({Category}.csv). "
            "Default: data/category_stores under project root."
        ),
    )
    parser.add_argument(
        "--output-json",
        default=DEFAULT_OUTPUT_JSON,
        help="Path to the structured JSON output. Default: data/automation_candidates.json",
    )
    parser.add_argument(
        "--output-csv",
        default=DEFAULT_OUTPUT_CSV,
        help=(
            "Path to the CSV summary output. "
            "Default: data/automation_candidates_summary.csv"
        ),
    )
    return parser.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)

    print("=" * 70)
    print("AUTOMATION-CANDIDATE FLAGGER (production)")
    print("=" * 70)
    print(
        "Fixed, CALIBRATED cosine-similarity threshold: {:.2f}".format(
            RESOLUTION_SIMILARITY_THRESHOLD
        )
    )
    print(
        "This threshold is FIXED and CALIBRATED -- it is NOT swept here and "
        "is not a CLI arg."
    )
    print(
        "Calibration evidence: {} (precision=1.0000 at 0.80 vs 500-ticket "
        "ground truth).".format(CALIBRATION_SOURCE_REL)
    )
    print("Model: {}".format(MODEL_NAME))
    print("Project root:  {}".format(PROJECT_ROOT))
    print("Input dir:     {}".format(args.input_dir))
    print("Output JSON:   {}".format(args.output_json))
    print("Output CSV:    {}".format(args.output_csv))
    print(
        "Only clusters of size >= {} are flagged as automation candidates; "
        "singletons are counted but not flagged.".format(MIN_CLUSTER_SIZE)
    )

    if not os.path.isdir(args.input_dir):
        print("")
        print(
            "ERROR: input directory does not exist: {}".format(args.input_dir)
        )
        print(
            "Nothing to process. (This directory holds gitignored pipeline "
            "logs; ensure the pipeline has been run.)"
        )
        return 1

    # Load the model once and reuse across all categories.
    print("")
    print("Loading SentenceTransformer model '{}'...".format(MODEL_NAME))
    model = SentenceTransformer(MODEL_NAME)
    print("Model loaded.")

    per_category_results = []
    categories_processed = []

    for category in CATEGORIES:
        result = process_category(category, args.input_dir, model)
        if result is None:
            # Skipped gracefully (missing file / no usable rows).
            continue
        per_category_results.append(result)
        categories_processed.append(category)

    print("-" * 70)

    if not per_category_results:
        print("")
        print(
            "No categories yielded usable tickets. No output files written."
        )
        return 0

    print("")
    write_json_output(args.output_json, per_category_results, categories_processed)
    print("")
    write_csv_summary(args.output_csv, per_category_results)
    print("")
    print_summary_table(per_category_results)

    return 0


if __name__ == "__main__":
    sys.exit(main())
