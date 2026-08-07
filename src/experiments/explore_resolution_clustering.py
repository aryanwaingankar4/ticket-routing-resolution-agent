# src/experiments/explore_resolution_clustering.py
#
# EXPLORATORY / DIAGNOSTIC SCRIPT -- not a finished feature.
#
# Purpose: embed the RESOLUTION TEXT (not the ticket symptom/title/description)
# of tickets within each category, then group them by embedding cosine
# similarity at several thresholds, so the results can be eyeballed before any
# real threshold is chosen or any ground-truth evaluation set is built.
#
# This script intentionally does NOT recommend or hardcode a "final" threshold.
# It runs a SWEEP across several thresholds and reports each one separately so
# the change in cluster count/size as the threshold relaxes is visible.
#
# Local/offline only: sentence-transformers + numpy. No API calls.
# Windows / cross-platform: all paths via os.path.*.

import os
import json
import numpy as np
from sentence_transformers import SentenceTransformer

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

MODEL_NAME = "all-MiniLM-L6-v2"

# Same categories / casing as the category_stores CSV filenames.
# NOTE: "Access Management" has a literal space in the filename.
CATEGORIES = [
    "Infrastructure",
    "Application",
    "Security",
    "Database",
    "Storage",
    "Network",
    "Access Management",
]

# Threshold sweep -- reported SEPARATELY per threshold. This is deliberately a
# sweep (relaxing from very strict to looser) so the trend is visible. These
# are NOT a recommendation; no single value here is "the" threshold.
SIMILARITY_THRESHOLDS = [0.99, 0.95, 0.90, 0.85, 0.80, 0.75, 0.70, 0.65, 0.60]

# Column names (from the confirmed Database.csv header).
COL_RESOLUTION = "resolution_text"
COL_BATCH_ID = "batch_ticket_id"
COL_ORIGINAL_ID = "original_id"

# Exploratory knob, OFF by default. If you later suspect the trailing
# "*Note: ...*" line is inflating similarity, flip this to True to strip it
# before embedding and re-run. Left visible on purpose; not a decision.
STRIP_MARKDOWN_NOTE = False

# How many example resolution texts to print per multi-member cluster.
MAX_EXAMPLES_PER_CLUSTER = 3

# Paths (os.path.* per project convention). This file lives in
# src/experiments/, so project root is two levels up.
_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(_THIS_DIR, os.pardir, os.pardir))
CATEGORY_STORES_DIR = os.path.join(PROJECT_ROOT, "data", "category_stores")
OUTPUT_JSON_PATH = os.path.join(PROJECT_ROOT, "data", "exploratory_clustering_results.json")

# Encoding batch size -- matches build_vector_index.py.
ENCODE_BATCH_SIZE = 64


# ---------------------------------------------------------------------------
# CSV loading (stdlib csv, no pandas dependency assumed)
# ---------------------------------------------------------------------------

import csv


def load_category_rows(category):
    """Load rows for one category CSV. Returns list of dicts, or None if the
    file doesn't exist. Rows with empty resolution_text are skipped (they have
    nothing to embed for this experiment)."""
    csv_path = os.path.join(CATEGORY_STORES_DIR, category + ".csv")
    if not os.path.exists(csv_path):
        return None

    rows = []
    skipped_empty = 0
    with open(csv_path, "r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        if reader.fieldnames is None or COL_RESOLUTION not in reader.fieldnames:
            print(f"  [WARN] '{category}.csv' missing '{COL_RESOLUTION}' column "
                  f"(found: {reader.fieldnames}). Skipping this category.")
            return None

        for idx, raw in enumerate(reader):
            resolution = (raw.get(COL_RESOLUTION) or "").strip()
            if not resolution:
                skipped_empty += 1
                continue

            # Primary id: batch_ticket_id, fall back to row index if blank.
            batch_id = (raw.get(COL_BATCH_ID) or "").strip()
            if not batch_id:
                batch_id = f"rowidx_{idx}"

            rows.append({
                "batch_ticket_id": batch_id,
                "original_id": (raw.get(COL_ORIGINAL_ID) or "").strip(),
                "resolution_text": resolution,
            })

    if skipped_empty:
        print(f"  [info] {category}: skipped {skipped_empty} row(s) with empty "
              f"{COL_RESOLUTION}.")
    return rows


def preprocess_resolution(text):
    """Return the text to embed. By default returns raw resolution_text.
    Only alters it if STRIP_MARKDOWN_NOTE is toggled on (exploratory knob)."""
    if not STRIP_MARKDOWN_NOTE:
        return text
    # Drop a trailing italic note line like: *Note: ...*
    lines = text.splitlines()
    kept = [ln for ln in lines
            if not ln.strip().startswith("*Note:")]
    return "\n".join(kept).strip()


# ---------------------------------------------------------------------------
# Similarity + threshold grouping (connected components via union-find)
# ---------------------------------------------------------------------------

def cosine_similarity_matrix(embeddings):
    """L2-normalize embeddings EXPLICITLY here (kept visible, per project
    convention -- encode() above does NOT normalize), then cosine sim == dot."""
    norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
    norms[norms == 0.0] = 1.0  # guard against zero-vectors
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
    """Connected-components grouping: any two tickets with cosine sim >=
    threshold are linked; connected components become clusters. Returns a list
    of clusters, each a list of row indices, sorted largest-first."""
    n = sim_matrix.shape[0]
    uf = _UnionFind(n)

    # Link all pairs above threshold (upper triangle only).
    for i in range(n):
        # vectorized scan of the row for j > i
        row = sim_matrix[i]
        for j in range(i + 1, n):
            if row[j] >= threshold:
                uf.union(i, j)

    # Collect components.
    comps = {}
    for i in range(n):
        root = uf.find(i)
        comps.setdefault(root, []).append(i)

    clusters = list(comps.values())
    clusters.sort(key=lambda c: (-len(c), min(c)))
    return clusters


# ---------------------------------------------------------------------------
# Reporting
# ---------------------------------------------------------------------------

def truncate_for_print(text, limit=280):
    text = " ".join(text.split())  # collapse whitespace/newlines for readability
    if len(text) <= limit:
        return text
    return text[:limit] + " [...]"


def report_category_threshold(category, threshold, rows, clusters):
    """Print exploratory summary for one category x threshold."""
    multi = [c for c in clusters if len(c) >= 2]
    singletons = [c for c in clusters if len(c) == 1]

    print(f"\n  --- {category} @ threshold {threshold:.2f} "
          f"[exploratory, not a chosen value] ---")
    print(f"    total tickets:        {len(rows)}")
    print(f"    clusters found:       {len(clusters)}")
    print(f"    multi-member clusters:{len(multi)}")
    print(f"    singleton clusters:   {len(singletons)}")

    if multi:
        print(f"    cluster sizes (multi-member, largest first): "
              f"{[len(c) for c in multi]}")

    for ci, cluster in enumerate(multi):
        print(f"    [cluster {ci}] size={len(cluster)} "
              f"-- example resolution texts (up to {MAX_EXAMPLES_PER_CLUSTER}, "
              f"eyeball for false grouping):")
        for member_idx in cluster[:MAX_EXAMPLES_PER_CLUSTER]:
            r = rows[member_idx]
            print(f"        - id={r['batch_ticket_id']} "
                  f"(orig={r['original_id']}): "
                  f"{truncate_for_print(r['resolution_text'])}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def build_json_clusters(rows, clusters):
    """Serializable cluster assignments for one category x threshold."""
    out = []
    for ci, cluster in enumerate(clusters):
        out.append({
            "cluster_id": ci,
            "size": len(cluster),
            "is_singleton": len(cluster) == 1,
            "members": [
                {
                    "batch_ticket_id": rows[m]["batch_ticket_id"],
                    "original_id": rows[m]["original_id"],
                }
                for m in cluster
            ],
        })
    return out


def main():
    print("=" * 72)
    print("EXPLORATORY resolution-text clustering (per category, threshold sweep)")
    print("This is a diagnostic pass. No 'final' threshold is chosen anywhere.")
    print("=" * 72)
    print(f"Category stores dir: {CATEGORY_STORES_DIR}")
    print(f"Output JSON:         {OUTPUT_JSON_PATH}")
    print(f"Thresholds swept:    {SIMILARITY_THRESHOLDS}")
    print(f"STRIP_MARKDOWN_NOTE knob: {STRIP_MARKDOWN_NOTE} "
          f"(exploratory; off by default)")

    print(f"\n[load] Loading model '{MODEL_NAME}' ...")
    model = SentenceTransformer(MODEL_NAME)
    print("[load] Model ready.")

    # results[category][str(threshold)] = list of cluster dicts
    results = {
        "_meta": {
            "note": "EXPLORATORY output. Thresholds are a sweep, not a "
                    "recommendation. No final threshold is implied.",
            "model": MODEL_NAME,
            "thresholds": SIMILARITY_THRESHOLDS,
            "resolution_column": COL_RESOLUTION,
            "strip_markdown_note": STRIP_MARKDOWN_NOTE,
        }
    }

    for category in CATEGORIES:
        print(f"\n{'=' * 72}\n[category] Processing: {category}")
        rows = load_category_rows(category)

        if rows is None:
            print(f"  [skip] No usable CSV for '{category}' -- skipping.")
            continue
        if len(rows) == 0:
            print(f"  [skip] '{category}' has no rows with resolution text.")
            continue

        print(f"  [encode] Encoding {len(rows)} resolution texts "
              f"with {MODEL_NAME} (batch_size={ENCODE_BATCH_SIZE}) ...")
        texts = [preprocess_resolution(r["resolution_text"]) for r in rows]

        # Same encode() pattern as build_vector_index.py: NO normalization here.
        embeddings = model.encode(
            texts,
            batch_size=ENCODE_BATCH_SIZE,
            convert_to_numpy=True,
            show_progress_bar=True,
        )
        print(f"  [encode] Done. Embedding matrix shape: {embeddings.shape}")

        # Special case: a single ticket can't pair with anything.
        if len(rows) == 1:
            print("  [info] Only 1 ticket -- trivially a singleton at all "
                  "thresholds.")
            results[category] = {}
            for threshold in SIMILARITY_THRESHOLDS:
                clusters = [[0]]
                report_category_threshold(category, threshold, rows, clusters)
                results[category][f"{threshold:.2f}"] = build_json_clusters(
                    rows, clusters)
            continue

        print("  [sim] Computing pairwise cosine similarity "
              "(explicit L2 normalization applied here) ...")
        sim = cosine_similarity_matrix(embeddings)

        results[category] = {}
        for threshold in SIMILARITY_THRESHOLDS:
            print(f"  [group] {category}: grouping at threshold "
                  f"{threshold:.2f} ...")
            clusters = group_by_threshold(sim, threshold)
            report_category_threshold(category, threshold, rows, clusters)
            results[category][f"{threshold:.2f}"] = build_json_clusters(
                rows, clusters)

    # Write JSON.
    print(f"\n{'=' * 72}\n[write] Writing cluster assignments to:\n  "
          f"{OUTPUT_JSON_PATH}")
    os.makedirs(os.path.dirname(OUTPUT_JSON_PATH), exist_ok=True)
    with open(OUTPUT_JSON_PATH, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    print("[write] Done.")

    print(f"\n{'=' * 72}")
    print("EXPLORATORY run complete. Inspect the printed clusters + the JSON.")
    print("Reminder: no threshold here is endorsed -- this is a sweep to look "
          "at, not a decision.")
    print("=" * 72)


if __name__ == "__main__":
    main()
