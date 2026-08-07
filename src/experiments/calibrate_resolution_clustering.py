# src/experiments/calibrate_resolution_clustering.py
"""
Calibrate resolution-text clustering against scenario_id ground truth.

Evaluates the exploratory clustering sweep (all-MiniLM-L6-v2, cosine-similarity
thresholds 0.99 -> 0.80) using PAIRWISE clustering metrics:

    Ground-truth-positive pair : two tickets in the same category share a
                                  scenario_id (same underlying resolution
                                  scenario -> should be clustered together).
    Predicted-positive pair    : the clustering placed both tickets in the
                                  same cluster_id at that threshold.

    TP = GT-positive AND predicted-positive
    FP = predicted-positive but NOT GT-positive  (grouped two DIFFERENT scenarios)
    FN = GT-positive but NOT predicted-positive  (MISSED a real same-scenario pair)

    precision = TP / (TP + FP)
    recall    = TP / (TP + FN)
    F1        = harmonic mean(precision, recall)

Inputs  : data/exploratory_clustering_results.json
          data/category_stores_with_scenario_id.csv
Output  : data/resolution_clustering_calibration_results.csv

Fully local/offline. Stdlib + pandas only.
"""

import os
import csv
import json
from itertools import combinations
from collections import defaultdict

import pandas as pd

# --------------------------------------------------------------------------- #
# Paths (os.path.* for Windows/cross-platform convention)
# --------------------------------------------------------------------------- #
PROJECT_ROOT = os.path.abspath(
    os.path.join(os.path.dirname(__file__), os.pardir, os.pardir)
)
DATA_DIR = os.path.join(PROJECT_ROOT, "data")
JSON_PATH = os.path.join(DATA_DIR, "exploratory_clustering_results.json")
CSV_PATH = os.path.join(DATA_DIR, "category_stores_with_scenario_id.csv")
OUTPUT_CSV_PATH = os.path.join(
    DATA_DIR, "resolution_clustering_calibration_results.csv"
)

# Threshold key strings EXACTLY as they appear in the JSON (zero-padded to
# 2 decimals). Do NOT derive via str(float(...)) -- that yields '0.9'/'0.8'
# which do not exist as keys. Formatting the _meta floats via f"{t:.2f}"
# reproduces these; we also hardcode for clarity/ordering (loosest last).
THRESHOLD_KEYS = ["0.99", "0.95", "0.90", "0.85", "0.80", "0.75", "0.70", "0.65", "0.60"]
LOOSEST_THRESHOLD_KEY = "0.60"
META_KEY = "_meta"

FP_EXAMPLES_TO_SHOW = 8
TEXT_TRUNCATE = 150


# --------------------------------------------------------------------------- #
# Metric helpers
# --------------------------------------------------------------------------- #
def _fmt_metric(value):
    """Format a metric that may be None ('N/A') for CSV/terminal."""
    return "N/A" if value is None else f"{value:.4f}"


def compute_pairwise_metrics(tickets, scenario_of, cluster_of):
    """
    Given the ticket population for one (category, threshold) slice, compute
    pairwise TP/FP/FN and precision/recall/F1.

    tickets      : list of batch_ticket_id strings (the JSON population).
    scenario_of  : dict batch_ticket_id(str) -> scenario_id (from CSV).
    cluster_of   : dict batch_ticket_id(str) -> cluster_id (from JSON slice).

    precision/recall/F1 are None when their denominator is zero.
    """
    tp = fp = fn = 0
    for a, b in combinations(tickets, 2):
        gt_positive = scenario_of[a] == scenario_of[b]
        pred_positive = cluster_of[a] == cluster_of[b]
        if pred_positive and gt_positive:
            tp += 1
        elif pred_positive and not gt_positive:
            fp += 1
        elif gt_positive and not pred_positive:
            fn += 1
        # else: true negative -- not needed for P/R/F1

    precision = tp / (tp + fp) if (tp + fp) > 0 else None
    recall = tp / (tp + fn) if (tp + fn) > 0 else None
    if precision is None or recall is None or (precision + recall) == 0:
        f1 = None
    else:
        f1 = 2 * precision * recall / (precision + recall)

    return {
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "precision": precision,
        "recall": recall,
        "f1": f1,
    }


def metrics_from_counts(tp, fp, fn):
    """Recompute precision/recall/F1 from pooled counts (aggregated view)."""
    precision = tp / (tp + fp) if (tp + fp) > 0 else None
    recall = tp / (tp + fn) if (tp + fn) > 0 else None
    if precision is None or recall is None or (precision + recall) == 0:
        f1 = None
    else:
        f1 = 2 * precision * recall / (precision + recall)
    return precision, recall, f1


# --------------------------------------------------------------------------- #
# Loading / cross-referencing
# --------------------------------------------------------------------------- #
def load_json(path):
    print(f"[load] reading clustering JSON: {path}")
    with open(path, "r", encoding="utf-8") as fh:
        data = json.load(fh)
    categories = [k for k in data.keys() if k != META_KEY]
    print(f"[load]   categories found: {categories}")

    meta = data.get(META_KEY, {})
    meta_thresholds = meta.get("thresholds")
    if meta_thresholds is not None:
        derived = [f"{t:.2f}" for t in meta_thresholds]
        if derived != THRESHOLD_KEYS:
            print(
                f"[warn]   _meta thresholds {derived} differ from hardcoded "
                f"{THRESHOLD_KEYS} -- using hardcoded ordering."
            )
    return data, categories


def load_csv(path):
    print(f"[load] reading tickets CSV: {path}")
    df = pd.read_csv(path)
    # Cast join key to str to match the JSON (which stores it as a string).
    df["batch_ticket_id"] = df["batch_ticket_id"].astype(str)
    print(f"[load]   rows: {len(df)}  categories: {sorted(df['category'].unique())}")
    return df


def build_cluster_map(cluster_list):
    """
    From a JSON threshold slice (list of cluster dicts) build:
        cluster_of : batch_ticket_id(str) -> cluster_id
        tickets    : list of batch_ticket_id(str) in this slice
    """
    cluster_of = {}
    tickets = []
    for cluster in cluster_list:
        cid = cluster["cluster_id"]
        for member in cluster["members"]:
            btid = str(member["batch_ticket_id"])
            cluster_of[btid] = cid
            tickets.append(btid)
    return cluster_of, tickets


# --------------------------------------------------------------------------- #
# Main
# --------------------------------------------------------------------------- #
def main():
    print("=" * 78)
    print("Resolution-clustering calibration (pairwise metrics vs scenario_id)")
    print("=" * 78)

    data, categories = load_json(JSON_PATH)
    df = load_csv(CSV_PATH)

    # Per-category CSV lookups: batch_ticket_id(str) -> scenario_id / text.
    scenario_by_cat = {}
    text_by_cat = {}
    csv_ids_by_cat = {}
    for cat, sub in df.groupby("category"):
        scenario_by_cat[cat] = dict(zip(sub["batch_ticket_id"], sub["scenario_id"]))
        text_by_cat[cat] = dict(
            zip(sub["batch_ticket_id"], sub["resolution_text"].fillna(""))
        )
        csv_ids_by_cat[cat] = set(sub["batch_ticket_id"])

    per_rows = []                     # rows for the output CSV
    aggregate = defaultdict(lambda: {"tp": 0, "fp": 0, "fn": 0})  # by threshold key
    population_reported = set()       # so per-category population check prints once

    print("\n" + "-" * 78)
    print("Processing categories / thresholds")
    print("-" * 78)

    for cat in categories:
        if cat not in scenario_by_cat:
            print(f"[warn] category '{cat}' present in JSON but absent from CSV "
                  f"-- skipping.")
            continue

        cat_block = data[cat]
        print(f"\n[cat] {cat}")

        for tkey in THRESHOLD_KEYS:
            if tkey not in cat_block:
                print(f"  [warn] threshold '{tkey}' missing for '{cat}' -- skipping.")
                continue

            cluster_of, tickets = build_cluster_map(cat_block[tkey])

            # --- Cross-reference: every JSON ticket must have a CSV row. ---
            scenario_of = {}
            missing = []
            for btid in tickets:
                if btid in scenario_by_cat[cat]:
                    scenario_of[btid] = scenario_by_cat[cat][btid]
                else:
                    missing.append(btid)
            if missing:
                print(f"  [warn] {tkey}: {len(missing)} JSON ticket(s) have NO "
                      f"matching CSV row (e.g. {missing[:5]}). Excluded from "
                      f"metrics for this slice.")
                tickets = [t for t in tickets if t in scenario_of]

            # --- Population sanity check (report once per category). ---
            if cat not in population_reported:
                json_ids = set(tickets) | set(missing)
                csv_ids = csv_ids_by_cat[cat]
                only_json = json_ids - csv_ids
                only_csv = csv_ids - json_ids
                if only_json or only_csv:
                    print(f"  [pop] {cat}: population mismatch -- "
                          f"{len(only_json)} only in JSON, "
                          f"{len(only_csv)} only in CSV "
                          f"(JSON={len(json_ids)}, CSV={len(csv_ids)}).")
                else:
                    print(f"  [pop] {cat}: JSON and CSV populations match "
                          f"({len(json_ids)} tickets).")
                population_reported.add(cat)

            # --- Metrics ---
            m = compute_pairwise_metrics(tickets, scenario_of, cluster_of)

            n_clusters = len(cat_block[tkey])
            print(f"  [thr {tkey}] tickets={len(tickets)} clusters={n_clusters}  "
                  f"P={_fmt_metric(m['precision'])} "
                  f"R={_fmt_metric(m['recall'])} "
                  f"F1={_fmt_metric(m['f1'])}  "
                  f"TP={m['tp']} FP={m['fp']} FN={m['fn']}")

            per_rows.append({
                "category": cat,
                "threshold": tkey,
                "n_tickets": len(tickets),
                "n_clusters": n_clusters,
                "tp": m["tp"],
                "fp": m["fp"],
                "fn": m["fn"],
                "precision": _fmt_metric(m["precision"]),
                "recall": _fmt_metric(m["recall"]),
                "f1": _fmt_metric(m["f1"]),
            })

            agg = aggregate[tkey]
            agg["tp"] += m["tp"]
            agg["fp"] += m["fp"]
            agg["fn"] += m["fn"]

    # ----------------------------------------------------------------------- #
    # Aggregated rows (pool all pairs across categories per threshold)
    # ----------------------------------------------------------------------- #
    agg_rows = []
    for tkey in THRESHOLD_KEYS:
        c = aggregate[tkey]
        p, r, f1 = metrics_from_counts(c["tp"], c["fp"], c["fn"])
        row = {
            "category": "ALL (aggregated)",
            "threshold": tkey,
            "n_tickets": "",
            "n_clusters": "",
            "tp": c["tp"],
            "fp": c["fp"],
            "fn": c["fn"],
            "precision": _fmt_metric(p),
            "recall": _fmt_metric(r),
            "f1": _fmt_metric(f1),
        }
        agg_rows.append(row)

    # ----------------------------------------------------------------------- #
    # Write CSV (per-category detail first, then aggregated block)
    # ----------------------------------------------------------------------- #
    fieldnames = [
        "category", "threshold", "n_tickets", "n_clusters",
        "tp", "fp", "fn", "precision", "recall", "f1",
    ]
    with open(OUTPUT_CSV_PATH, "w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(per_rows)
        writer.writerows(agg_rows)
    print(f"\n[write] per-category + aggregated results -> {OUTPUT_CSV_PATH}")
    print(f"[write]   {len(per_rows)} per-category rows, "
          f"{len(agg_rows)} aggregated rows.")

    # ----------------------------------------------------------------------- #
    # Terminal summary: aggregated precision/recall tradeoff across the sweep
    # ----------------------------------------------------------------------- #
    print("\n" + "=" * 78)
    print("AGGREGATED SWEEP (all categories pooled) -- precision/recall tradeoff")
    print("=" * 78)
    header = f"{'thresh':>7} | {'precision':>9} | {'recall':>7} | {'F1':>7} | " \
             f"{'TP':>7} | {'FP':>7} | {'FN':>7}"
    print(header)
    print("-" * len(header))
    for tkey in THRESHOLD_KEYS:  # 0.99 (tight) -> 0.80 (loose)
        c = aggregate[tkey]
        p, r, f1 = metrics_from_counts(c["tp"], c["fp"], c["fn"])
        print(f"{tkey:>7} | {_fmt_metric(p):>9} | {_fmt_metric(r):>7} | "
              f"{_fmt_metric(f1):>7} | {c['tp']:>7} | {c['fp']:>7} | {c['fn']:>7}")
    print("-" * len(header))
    print("Tighter thresholds (0.99) -> higher precision, lower recall.")
    print("Looser thresholds (0.80)  -> higher recall, more false groupings.")

    # ----------------------------------------------------------------------- #
    # False-positive examples at the loosest threshold (0.80)
    # ----------------------------------------------------------------------- #
    print("\n" + "=" * 78)
    print(f"FALSE-POSITIVE EXAMPLES @ threshold {LOOSEST_THRESHOLD_KEY} "
          f"(grouped together but DIFFERENT scenario_id)")
    print("=" * 78)

    fp_examples = collect_false_positive_examples(
        data, categories, scenario_by_cat, text_by_cat
    )

    if not fp_examples:
        print("No false-positive pairs found at the loosest threshold. "
              "(Every same-cluster pair genuinely shares a scenario_id.)")
    else:
        total_fp = sum(aggregate[LOOSEST_THRESHOLD_KEY]["fp"] for _ in [0])
        print(f"Total FP pairs at {LOOSEST_THRESHOLD_KEY}: "
              f"{aggregate[LOOSEST_THRESHOLD_KEY]['fp']}. "
              f"Showing up to {FP_EXAMPLES_TO_SHOW}:\n")
        for i, ex in enumerate(fp_examples[:FP_EXAMPLES_TO_SHOW], start=1):
            print(f"--- FP #{i}  [{ex['category']}] cluster_id={ex['cluster_id']} "
                  f"---")
            print(f"  ticket A={ex['a_id']} (scenario_id={ex['a_scenario']}) | "
                  f"ticket B={ex['b_id']} (scenario_id={ex['b_scenario']})")
            print(f"  A: {ex['a_text']}")
            print(f"  B: {ex['b_text']}")
            print()

    print("Done.")


def _clean_text(text):
    """Collapse newlines/whitespace and truncate for readable terminal output."""
    if text is None:
        text = ""
    flat = " ".join(str(text).split())
    if len(flat) > TEXT_TRUNCATE:
        flat = flat[:TEXT_TRUNCATE].rstrip() + "..."
    return flat


def collect_false_positive_examples(data, categories, scenario_by_cat, text_by_cat):
    """
    Walk the loosest-threshold clusters and collect same-cluster pairs whose
    scenario_ids differ (i.e. false positives), for eyeballing.
    """
    examples = []
    for cat in categories:
        if cat not in scenario_by_cat:
            continue
        cat_block = data[cat]
        if LOOSEST_THRESHOLD_KEY not in cat_block:
            continue
        scen = scenario_by_cat[cat]
        txt = text_by_cat[cat]
        for cluster in cat_block[LOOSEST_THRESHOLD_KEY]:
            members = [str(m["batch_ticket_id"]) for m in cluster["members"]]
            # Only tickets we can resolve a scenario_id for.
            members = [m for m in members if m in scen]
            for a, b in combinations(members, 2):
                if scen[a] != scen[b]:
                    examples.append({
                        "category": cat,
                        "cluster_id": cluster["cluster_id"],
                        "a_id": a,
                        "b_id": b,
                        "a_scenario": scen[a],
                        "b_scenario": scen[b],
                        "a_text": _clean_text(txt.get(a, "")),
                        "b_text": _clean_text(txt.get(b, "")),
                    })
    return examples


if __name__ == "__main__":
    main()
