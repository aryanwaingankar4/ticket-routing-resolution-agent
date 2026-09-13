# src/experiments/build_flag_validation_set.py
"""
build_flag_validation_set.py
============================

Phase 2, step 2A: build the AUTOMATION-FLAG VALIDATION SET.

WHAT THIS SCRIPT IS
-------------------
The production automation-flagging feature clusters resolution text at a
calibrated threshold (MiniLM @ 0.80). A later measurement-only re-run found
BGE's own cliff-edge at 0.90. The open question is whether production should
swap. It has been blocked for one specific reason, recorded in the README:
there is no ground-truth check for whether BGE clustering produces BETTER
automation flags -- only that its own precision/recall curve is internally
consistent.

This script builds that check. It is the direct analogue of the 9-ticket
adversarial escalation set's role for the RAG gate.

WHY NOT JUST USE scenario_id
----------------------------
`scenario_id` is the dataset generator's TEMPLATE identity. It validates
"same template", which is a proxy for "same underlying fix" -- and
template-level memorisation is precisely the confound this project already
proved un-removable in the conformal calibration set. Leaning on that proxy
to adjudicate a production swap uses it exactly where it is weakest.

So the labels here are HUMAN labels, produced without seeing scenario_id.
scenario_id is still recorded, in the separate key file, as a SECONDARY
measurement: how often the template proxy agrees with human judgement is
itself a citable result, but it is never the ground truth.

THE DISAGREEMENT REGION
-----------------------
Pairs that BOTH configurations co-cluster, or BOTH separate, carry zero
information about which configuration is better -- they would be labelled
identically and score identically. The entire decision lives in the pairs
where the two configurations DISAGREE.

So the sample is drawn only from the symmetric difference:

    disagreement = (co-clustered under MiniLM @ 0.80)
                 XOR
                   (co-clustered under BGE @ 0.90)

By construction every sampled pair is discordant: exactly one configuration
claims "these two tickets share a fix". A human label therefore awards the
point to exactly one of them, which makes this a paired head-to-head
comparison (McNemar) with maximum power per unit of labelling effort.

WHY THE SAMPLE IS PROPORTIONAL AND NOT BALANCED
-----------------------------------------------
It is tempting to sample equal numbers of "MiniLM says same, BGE says
different" and "BGE says same, MiniLM says different". DO NOT. Direction
determines which configuration a "same_fix" label rewards, so forcing the
directions to 50/50 would bias the head-to-head test toward whichever
configuration is over-represented relative to the real disagreement region.
Allocation is therefore proportional to the true per-cell disagreement counts.

BLIND LABELLING
---------------
Two files are written, on purpose:

    data/automation_flag_validation_set.json   <- you label this one
    data/automation_flag_validation_key.json   <- the scorer reads this one

The labelling file contains ONLY the category and the two tickets' text. It
does not contain scenario_id, and it does not say which configuration
co-clustered the pair -- otherwise the label would be biased by knowing which
answer favours which model. Entries are shuffled (seed 42) so that direction
cannot be inferred from ordering either.

WHAT IT READS
-------------
    data/exploratory_clustering_results.json                    (MiniLM)
    data/exploratory_clustering_results_bge-base-en-v1-5.json   (BGE)
    data/category_stores/{Category}.csv                         (text)
    data/category_stores_with_scenario_id.csv                   (key file only)

Both clustering JSONs already exist and already contain the full threshold
sweep, so nothing is re-embedded here and no model is loaded. This script is
offline, deterministic, and spends no Gemini quota.
"""

from __future__ import annotations

import argparse
import os
import csv
import json
import random
import sys
from itertools import combinations

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

# The two configurations under comparison. The thresholds are each
# configuration's OWN calibrated cliff-edge -- 0.80 for MiniLM (the live
# production value) and 0.90 for BGE (from the BGE clustering re-run).
# Comparing each model at its own cliff is the only fair comparison;
# comparing both at 0.80 would handicap BGE at a threshold its own precision
# curve does not endorse.
CONFIG_A_NAME = "minilm@0.80"
CONFIG_A_JSON = "exploratory_clustering_results.json"
CONFIG_A_THRESHOLD_KEY = "0.80"

CONFIG_B_NAME = "bge@0.90"
CONFIG_B_JSON = "exploratory_clustering_results_bge-base-en-v1-5.json"
CONFIG_B_THRESHOLD_KEY = "0.90"

CATEGORIES = [
    "Infrastructure",
    "Application",
    "Security",
    "Database",
    "Storage",
    "Network",
    "Access Management",
]

# Labelling budget for the full head-to-head set.
TARGET_SAMPLE_SIZE = 60

# ---------------------------------------------------------------------------
# PILOT MODE (--pilot)
# ---------------------------------------------------------------------------
# Building the full 60-pair set surfaced a structural property of this data:
# at their own cliff-edges NEITHER configuration ever merges across dataset
# templates. All 1,069 pairs both configurations merge, and all 411 pairs
# they disagree about, are within-template. There is therefore no observed
# precision difference between them anywhere in the region -- the entire
# disagreement is RECALL.
#
# That matters because production's stated design choice is precision over
# recall: a false "these two share a fix" claim is costlier than a missed
# automation opportunity. A rule that promotes whichever model merges more
# would invert that.
#
# The pilot spends 12 judgements instead of 60 to find out whether a
# false merge is observable in this region AT ALL. It draws from the most
# textually divergent pairs -- the place a genuine different-fix pair would
# show up if one exists anywhere.
#
# The pilot is a PROBE, not the head-to-head. It is deliberately balanced
# across directions (6 each), which would bias a win-rate comparison but is
# correct for "does either configuration make a false merge here?". The
# pilot must never be scored as the head-to-head; the scorer enforces this.
PILOT_PER_DIRECTION = 6
PILOT_SAMPLE_SIZE = PILOT_PER_DIRECTION * 2

# Near-duplicate guard. Standing project invariant: generated/sampled sets
# need a diversity check, not just a correctness check. Duplicated labelling
# points waste budget and skew the head-to-head. A candidate pair is rejected
# if the token-Jaccard of its combined resolution text against any
# already-selected pair exceeds this.
NEAR_DUPLICATE_JACCARD = 0.90

# Seed 42 everywhere, per project invariant.
RANDOM_SEED = 42

COL_RESOLUTION = "resolution_text"
COL_BATCH_ID = "batch_ticket_id"
COL_ORIGINAL_ID = "original_id"
COL_TITLE = "title"

# Paths (os.path.* per project convention). This file lives in
# src/experiments/, so project root is two levels up.
_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(_THIS_DIR, os.pardir, os.pardir))
DATA_DIR = os.path.join(PROJECT_ROOT, "data")
CATEGORY_STORES_DIR = os.path.join(DATA_DIR, "category_stores")
SCENARIO_CSV_PATH = os.path.join(DATA_DIR, "category_stores_with_scenario_id.csv")

OUTPUT_SET_PATH = os.path.join(DATA_DIR, "automation_flag_validation_set.json")
OUTPUT_KEY_PATH = os.path.join(DATA_DIR, "automation_flag_validation_key.json")

PILOT_SET_PATH = os.path.join(DATA_DIR, "automation_flag_validation_pilot.json")
PILOT_KEY_PATH = os.path.join(
    DATA_DIR, "automation_flag_validation_pilot_key.json")


# ---------------------------------------------------------------------------
# Failure handling -- clear actionable messages, not tracebacks.
# ---------------------------------------------------------------------------

def die(message):
    print("\n[ERROR] " + message + "\n", file=sys.stderr)
    sys.exit(1)


def load_json_or_die(path, what):
    if not os.path.exists(path):
        die(
            "Missing " + what + ":\n    " + path + "\n"
            "Regenerate it with src/experiments/explore_resolution_clustering.py "
            "(set MODEL_NAME and OUTPUT_JSON_PATH for the configuration you need)."
        )
    with open(path, "r", encoding="utf-8") as fh:
        return json.load(fh)


# ---------------------------------------------------------------------------
# Loading
# ---------------------------------------------------------------------------

def load_category_text(category):
    """batch_ticket_id -> {title, resolution_text, original_id} for one category."""
    path = os.path.join(CATEGORY_STORES_DIR, category + ".csv")
    if not os.path.exists(path):
        die(
            "Missing category store for '" + category + "':\n    " + path + "\n"
            "These CSVs are produced by src/experiments/process_ticket_batch.py "
            "and are versioned in the repo -- if one is absent, something has "
            "deleted it rather than it never having been generated."
        )
    out = {}
    with open(path, "r", encoding="utf-8", newline="") as fh:
        for row in csv.DictReader(fh):
            tid = (row.get(COL_BATCH_ID) or "").strip()
            text = (row.get(COL_RESOLUTION) or "").strip()
            if not tid or not text:
                continue
            out[tid] = {
                "batch_ticket_id": tid,
                "original_id": (row.get(COL_ORIGINAL_ID) or "").strip(),
                "title": (row.get(COL_TITLE) or "").strip(),
                "resolution_text": text,
            }
    if not out:
        die("Category store for '" + category + "' had no usable rows: " + path)
    return out


def load_scenario_map():
    """(category, batch_ticket_id) -> scenario_id. Key file only; never shown."""
    if not os.path.exists(SCENARIO_CSV_PATH):
        die(
            "Missing scenario ground-truth join:\n    " + SCENARIO_CSV_PATH + "\n"
            "Regenerate it with src/experiments/join_scenario_ground_truth.py."
        )
    out = {}
    with open(SCENARIO_CSV_PATH, "r", encoding="utf-8", newline="") as fh:
        for row in csv.DictReader(fh):
            cat = (row.get("category") or "").strip()
            tid = (row.get(COL_BATCH_ID) or "").strip()
            sid = (row.get("scenario_id") or "").strip()
            if cat and tid:
                out[(cat, tid)] = sid
    return out


# ---------------------------------------------------------------------------
# Disagreement region
# ---------------------------------------------------------------------------

def co_clustered_pairs(cluster_list):
    """Set of (id_a, id_b) tuples co-clustered by one configuration.

    Ids are ordered numerically inside each tuple so the same pair has one
    canonical representation under both configurations.
    """
    pairs = set()
    for cluster in cluster_list:
        ids = [m[COL_BATCH_ID] for m in cluster.get("members", [])]
        if len(ids) < 2:
            continue
        ids = sorted(ids, key=lambda x: int(x))
        for a, b in combinations(ids, 2):
            pairs.add((a, b))
    return pairs


def threshold_block(cfg_json, category, threshold_key, cfg_name):
    if category not in cfg_json:
        die(
            "Configuration '" + cfg_name + "' has no block for category '"
            + category + "'. The two clustering JSONs must cover the same "
            "categories."
        )
    cat_block = cfg_json[category]
    if threshold_key not in cat_block:
        die(
            "Configuration '" + cfg_name + "' has no threshold '"
            + threshold_key + "' for category '" + category + "'. Available: "
            + str(sorted(cat_block.keys()))
        )
    return cat_block[threshold_key]


# ---------------------------------------------------------------------------
# Near-duplicate guard
# ---------------------------------------------------------------------------

def _tokens(text):
    cleaned = []
    for ch in text.lower():
        cleaned.append(ch if (ch.isalnum() or ch.isspace()) else " ")
    return set("".join(cleaned).split())


def pair_signature(text_a, text_b):
    return _tokens(text_a) | _tokens(text_b)


def jaccard(sig_a, sig_b):
    union = sig_a | sig_b
    if not union:
        return 1.0
    return len(sig_a & sig_b) / len(union)


def divergence(text_a, text_b):
    """1 - token Jaccard of the two resolutions. 0 = identical wording.

    Used only to rank pilot candidates. A false merge, if one exists in this
    region, is most likely to sit at the divergent end -- two tickets whose
    resolutions actually describe different work. Near-identical paraphrases
    are the cases where the answer is obvious.
    """
    return 1.0 - jaccard(_tokens(text_a), _tokens(text_b))


# ---------------------------------------------------------------------------
# Proportional allocation
# ---------------------------------------------------------------------------

def allocate(cell_sizes, total):
    """Largest-remainder allocation of `total` across cells, proportional to size.

    Every non-empty cell gets at least 1 so no category/direction vanishes
    entirely, but the bulk of the allocation stays proportional -- balancing
    the directions would bias the head-to-head test (see module docstring).
    """
    non_empty = {k: v for k, v in cell_sizes.items() if v > 0}
    if not non_empty:
        return {}
    capacity = sum(non_empty.values())
    total = min(total, capacity)

    if total < len(non_empty):
        # More non-empty cells than budget: keep the largest cells.
        ordered = sorted(non_empty.items(), key=lambda kv: (-kv[1], str(kv[0])))
        return {k: 1 for k, _ in ordered[:total]}

    alloc = {k: 1 for k in non_empty}
    remaining = total - len(alloc)
    pool = sum(non_empty.values())

    shares = []
    for k, size in non_empty.items():
        exact = remaining * (size / pool)
        base = min(int(exact), non_empty[k] - alloc[k])
        alloc[k] += base
        shares.append((exact - int(exact), k))

    leftover = total - sum(alloc.values())
    for _, k in sorted(shares, key=lambda t: (-t[0], str(t[1]))):
        if leftover <= 0:
            break
        if alloc[k] < non_empty[k]:
            alloc[k] += 1
            leftover -= 1
    return alloc


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Build the Phase 2 / 2A automation-flag validation set. Default "
            "is the full proportional head-to-head set; --pilot builds the "
            "12-pair divergent-end false-merge probe instead."
        )
    )
    parser.add_argument(
        "--pilot",
        action="store_true",
        help=(
            "Build the 12-pair pilot probe (6 per direction) drawn from the "
            "most textually divergent pairs, to establish whether a false "
            "merge is observable in this region at all before spending the "
            "full 60-judgement budget."
        ),
    )
    return parser.parse_args()


def main():
    args = parse_args()
    pilot = args.pilot

    print("=" * 70)
    if pilot:
        print("Phase 2 / 2A -- build the FALSE-MERGE PILOT PROBE (12 pairs)")
    else:
        print("Phase 2 / 2A -- build the automation-flag validation set")
    print("=" * 70)
    print("Configuration A: " + CONFIG_A_NAME + "  (" + CONFIG_A_JSON + ")")
    print("Configuration B: " + CONFIG_B_NAME + "  (" + CONFIG_B_JSON + ")")
    if pilot:
        print("Target sample:   " + str(PILOT_SAMPLE_SIZE)
              + " human judgements (" + str(PILOT_PER_DIRECTION)
              + " per direction, most-divergent end)")
    else:
        print("Target sample:   " + str(TARGET_SAMPLE_SIZE)
              + " human judgements")
    print("Seed:            " + str(RANDOM_SEED))

    cfg_a = load_json_or_die(os.path.join(DATA_DIR, CONFIG_A_JSON),
                             "clustering results for " + CONFIG_A_NAME)
    cfg_b = load_json_or_die(os.path.join(DATA_DIR, CONFIG_B_JSON),
                             "clustering results for " + CONFIG_B_NAME)

    meta_a = cfg_a.get("_meta", {})
    meta_b = cfg_b.get("_meta", {})
    print("\n[load] " + CONFIG_A_NAME + " model: " + str(meta_a.get("model")))
    print("[load] " + CONFIG_B_NAME + " model: " + str(meta_b.get("model")))
    if meta_a.get("model") == meta_b.get("model"):
        die(
            "Both clustering JSONs report the same embedding model ('"
            + str(meta_a.get("model")) + "'). This comparison would be "
            "vacuous. One of the two files is stale -- regenerate it before "
            "continuing. (This is the project's recurring stale-artifact bug "
            "class, which has already surfaced four times.)"
        )

    scenario_map = load_scenario_map()

    # ---- Step 1: disagreement region, per category ------------------------
    print("\n" + "=" * 70)
    print("STEP 1 -- disagreement region")
    print("=" * 70)

    text_by_cat = {}
    cells = {}
    region_summary = []

    for category in CATEGORIES:
        text_by_cat[category] = load_category_text(category)

        pairs_a = co_clustered_pairs(
            threshold_block(cfg_a, category, CONFIG_A_THRESHOLD_KEY,
                            CONFIG_A_NAME))
        pairs_b = co_clustered_pairs(
            threshold_block(cfg_b, category, CONFIG_B_THRESHOLD_KEY,
                            CONFIG_B_NAME))

        only_a = sorted(pairs_a - pairs_b)
        only_b = sorted(pairs_b - pairs_a)

        cells[(category, "a_only")] = only_a
        cells[(category, "b_only")] = only_b

        region_summary.append({
            "category": category,
            "pairs_a": len(pairs_a),
            "pairs_b": len(pairs_b),
            "agree_both_same": len(pairs_a & pairs_b),
            "a_only": len(only_a),
            "b_only": len(only_b),
            "disagreement": len(only_a) + len(only_b),
        })

    hdr = ("{:<20}{:>14}{:>12}{:>8}{:>9}{:>9}{:>10}".format(
        "Category", CONFIG_A_NAME, CONFIG_B_NAME, "both", "A-only", "B-only",
        "disagree"))
    print("\n" + hdr)
    print("-" * len(hdr))
    for r in region_summary:
        print("{:<20}{:>14}{:>12}{:>8}{:>9}{:>9}{:>10}".format(
            r["category"], r["pairs_a"], r["pairs_b"], r["agree_both_same"],
            r["a_only"], r["b_only"], r["disagreement"]))
    total_disagree = sum(r["disagreement"] for r in region_summary)
    total_a_only = sum(r["a_only"] for r in region_summary)
    total_b_only = sum(r["b_only"] for r in region_summary)
    print("-" * len(hdr))
    print("{:<20}{:>14}{:>12}{:>8}{:>9}{:>9}{:>10}".format(
        "TOTAL", "", "", sum(r["agree_both_same"] for r in region_summary),
        total_a_only, total_b_only, total_disagree))

    if total_disagree == 0:
        die(
            "The two configurations agree on every pair. There is nothing to "
            "adjudicate and no validation set to build -- which would itself "
            "be the answer to the promotion question."
        )

    rng = random.Random(RANDOM_SEED)
    selected = []
    signatures = []
    rejected_dupes = 0
    short_cells = []

    if pilot:
        # ---- Step 2 (pilot): rank the whole region by divergence -----------
        print("\n" + "=" * 70)
        print("STEP 2 -- ranking the disagreement region by divergence")
        print("=" * 70)
        print("The pilot asks one question: is a false merge observable in")
        print("this region at all? So it takes the most textually divergent")
        print("pairs, where a genuine different-fix pair would show up if one")
        print("exists. Directions are balanced " + str(PILOT_PER_DIRECTION)
              + "/" + str(PILOT_PER_DIRECTION) + " on purpose -- that would")
        print("bias a win-rate comparison, but this is a probe, not the")
        print("head-to-head, and the scorer refuses to treat it as one.")

        ranked = {"a_only": [], "b_only": []}
        for (category, direction), pair_list in cells.items():
            for (id_a, id_b) in pair_list:
                rec_a = text_by_cat[category].get(id_a)
                rec_b = text_by_cat[category].get(id_b)
                if rec_a is None or rec_b is None:
                    continue
                div = divergence(rec_a["resolution_text"],
                                 rec_b["resolution_text"])
                ranked[direction].append(
                    (div, category, id_a, id_b, rec_a, rec_b))

        for direction in ranked:
            # Sort by divergence desc; ties broken deterministically.
            ranked[direction].sort(
                key=lambda t: (-t[0], t[1], int(t[2]), int(t[3])))

        print("\n{:<12}{:>12}{:>14}{:>14}".format(
            "Direction", "available", "max div", "median div"))
        print("-" * 52)
        for direction in ("a_only", "b_only"):
            rows = ranked[direction]
            if not rows:
                print("{:<12}{:>12}{:>14}{:>14}".format(direction, 0, "-", "-"))
                continue
            med = rows[len(rows) // 2][0]
            print("{:<12}{:>12}{:>14}{:>14}".format(
                direction, len(rows), "{:.2f}".format(rows[0][0]),
                "{:.2f}".format(med)))

        # ---- Step 3 (pilot): take the top N per direction ------------------
        print("\n" + "=" * 70)
        print("STEP 3 -- selecting the divergent end, with near-duplicate guard")
        print("=" * 70)
        print("Rejecting any candidate whose combined-resolution token Jaccard")
        print("against an already-selected pair exceeds "
              + str(NEAR_DUPLICATE_JACCARD) + ".")

        for direction in ("a_only", "b_only"):
            got = 0
            for (div, category, id_a, id_b, rec_a, rec_b) in ranked[direction]:
                if got >= PILOT_PER_DIRECTION:
                    break
                sig = pair_signature(rec_a["resolution_text"],
                                     rec_b["resolution_text"])
                if any(jaccard(sig, prev) > NEAR_DUPLICATE_JACCARD
                       for prev in signatures):
                    rejected_dupes += 1
                    continue
                signatures.append(sig)
                selected.append({
                    "category": category,
                    "direction": direction,
                    "divergence": round(div, 4),
                    "ticket_a": rec_a,
                    "ticket_b": rec_b,
                })
                got += 1
            if got < PILOT_PER_DIRECTION:
                short_cells.append(((direction, ""), PILOT_PER_DIRECTION, got))

        print("\n[sample] selected            : " + str(len(selected)))
        print("[sample] rejected as near-dup: " + str(rejected_dupes))
        if selected:
            divs = [s["divergence"] for s in selected]
            print("[sample] divergence range    : {:.2f} .. {:.2f}".format(
                min(divs), max(divs)))
    else:
        # ---- Step 2: proportional allocation -------------------------------
        print("\n" + "=" * 70)
        print("STEP 2 -- proportional allocation across (category, direction)")
        print("=" * 70)
        print("Allocation is proportional to the real disagreement counts and is")
        print("deliberately NOT balanced across directions -- see module docstring.")

        cell_sizes = {k: len(v) for k, v in cells.items()}
        alloc = allocate(cell_sizes, TARGET_SAMPLE_SIZE)

        print("\n{:<32}{:>11}{:>11}".format("Cell", "available", "allocated"))
        print("-" * 54)
        for key in sorted(alloc, key=lambda k: (k[0], k[1])):
            name = key[0] + " / " + ("A-only" if key[1] == "a_only" else "B-only")
            print("{:<32}{:>11}{:>11}".format(name, cell_sizes[key], alloc[key]))
        print("-" * 54)
        print("{:<32}{:>11}{:>11}".format(
            "TOTAL", sum(cell_sizes.values()), sum(alloc.values())))

        # ---- Step 3: sample with the near-duplicate guard ------------------
        print("\n" + "=" * 70)
        print("STEP 3 -- sampling with near-duplicate guard")
        print("=" * 70)
        print("Rejecting any candidate whose combined-resolution token Jaccard")
        print("against an already-selected pair exceeds "
              + str(NEAR_DUPLICATE_JACCARD) + ".")

        for key in sorted(alloc, key=lambda k: (k[0], k[1])):
            category, direction = key
            want = alloc[key]
            candidates = list(cells[key])
            rng.shuffle(candidates)
            got = 0
            for (id_a, id_b) in candidates:
                if got >= want:
                    break
                rec_a = text_by_cat[category].get(id_a)
                rec_b = text_by_cat[category].get(id_b)
                if rec_a is None or rec_b is None:
                    # Clustering JSON references a ticket the store lacks.
                    continue
                sig = pair_signature(rec_a["resolution_text"],
                                     rec_b["resolution_text"])
                if any(jaccard(sig, prev) > NEAR_DUPLICATE_JACCARD
                       for prev in signatures):
                    rejected_dupes += 1
                    continue
                signatures.append(sig)
                selected.append({
                    "category": category,
                    "direction": direction,
                    "ticket_a": rec_a,
                    "ticket_b": rec_b,
                })
                got += 1
            if got < want:
                short_cells.append((key, want, got))

        print("\n[sample] selected            : " + str(len(selected)))
        print("[sample] rejected as near-dup: " + str(rejected_dupes))

    if short_cells:
        print("[sample] cells that could not be filled after the guard:")
        for key, want, got in short_cells:
            print("         " + str(key[0]) + " / " + str(key[1]) + ": wanted "
                  + str(want) + ", got " + str(got))
        print("         (the guard is working -- these cells are internally "
              "near-identical)")

    # ---- Step 4: shuffle, split into blind set + key -----------------------
    print("\n" + "=" * 70)
    print("STEP 4 -- writing blind labelling file and answer key")
    print("=" * 70)

    rng.shuffle(selected)

    labelling = []
    key_rows = []
    for i, item in enumerate(selected, start=1):
        # Distinct id prefixes so a pilot file can never be silently scored
        # against the full set's answer key, or vice versa.
        pair_id = ("PL{:03d}" if pilot else "P{:03d}").format(i)
        cat = item["category"]
        a, b = item["ticket_a"], item["ticket_b"]

        labelling.append({
            "pair_id": pair_id,
            "category": cat,
            "ticket_a": {
                "batch_ticket_id": a["batch_ticket_id"],
                "title": a["title"],
                "resolution_text": a["resolution_text"],
            },
            "ticket_b": {
                "batch_ticket_id": b["batch_ticket_id"],
                "title": b["title"],
                "resolution_text": b["resolution_text"],
            },
            "label": None,
            "label_reason": "",
        })

        sid_a = scenario_map.get((cat, a["batch_ticket_id"]), "")
        sid_b = scenario_map.get((cat, b["batch_ticket_id"]), "")
        key_rows.append({
            "pair_id": pair_id,
            "category": cat,
            "direction": item["direction"],
            "co_clustered_by": (CONFIG_A_NAME if item["direction"] == "a_only"
                                else CONFIG_B_NAME),
            "separated_by": (CONFIG_B_NAME if item["direction"] == "a_only"
                             else CONFIG_A_NAME),
            "divergence": item.get("divergence"),
            "scenario_id_a": sid_a,
            "scenario_id_b": sid_b,
            "scenario_id_match": bool(sid_a) and sid_a == sid_b,
        })

    set_doc = {
        "_meta": {
            "set_type": "pilot" if pilot else "full",
            "purpose": (
                (
                    "Phase 2 / 2A FALSE-MERGE PILOT PROBE. Twelve pairs drawn "
                    "from the most textually divergent end of the "
                    "disagreement region, to establish whether either "
                    "configuration makes an observable false merge here at "
                    "all. If every pair is same_fix, the region contains no "
                    "precision signal and the promotion question cannot be "
                    "settled by flag correctness on this dataset."
                ) if pilot else (
                    "Phase 2 / 2A automation-flag validation set. Human "
                    "ground truth for whether two tickets genuinely share "
                    "the same underlying fix, used to adjudicate whether "
                    "production resolution clustering should swap from "
                    + CONFIG_A_NAME + " to " + CONFIG_B_NAME + "."
                )
            ),
            "how_to_label": (
                "For each pair, read both resolution texts and set \"label\" "
                "to exactly one of: \"same_fix\" (one automation would "
                "resolve both), \"different_fix\" (they need different "
                "fixes), or \"unclear\" (genuinely cannot tell). Optionally "
                "add a short \"label_reason\". Do not leave any label null."
            ),
            "blind_by_design": (
                "This file deliberately omits scenario_id and does not say "
                "which configuration co-clustered each pair. Both are in "
                "automation_flag_validation_key.json, which the scorer reads "
                "and the labeller should not."
            ),
            "precision_over_recall": (
                "The production design choice is that a false 'these share a "
                "fix' claim is costlier than a missed automation "
                "opportunity. When genuinely torn between same_fix and "
                "different_fix, that asymmetry is the tie-breaker; use "
                "\"unclear\" only when the texts do not support a judgement "
                "at all."
            ),
            "pairs": len(labelling),
            "seed": RANDOM_SEED,
            "near_duplicate_jaccard": NEAR_DUPLICATE_JACCARD,
        },
        "pairs": labelling,
    }

    key_doc = {
        "_meta": {
            "set_type": "pilot" if pilot else "full",
            "purpose": ("Answer key for automation_flag_validation_pilot.json."
                        if pilot else
                        "Answer key for automation_flag_validation_set.json."),
            "warning": (
                "Do not read this while labelling. It reveals which "
                "configuration co-clustered each pair."
            ),
            "config_a": {"name": CONFIG_A_NAME, "source": CONFIG_A_JSON,
                         "threshold": CONFIG_A_THRESHOLD_KEY,
                         "model": meta_a.get("model")},
            "config_b": {"name": CONFIG_B_NAME, "source": CONFIG_B_JSON,
                         "threshold": CONFIG_B_THRESHOLD_KEY,
                         "model": meta_b.get("model")},
            "disagreement_region": region_summary,
            "totals": {
                "a_only": total_a_only,
                "b_only": total_b_only,
                "disagreement": total_disagree,
                "sampled": len(key_rows),
                "rejected_near_duplicates": rejected_dupes,
            },
            "scenario_id_note": (
                "scenario_id is recorded as a SECONDARY measurement only -- "
                "how often the generator's template identity agrees with "
                "human judgement. It is never the ground truth here; see the "
                "module docstring of build_flag_validation_set.py."
            ),
            "seed": RANDOM_SEED,
        },
        "pairs": key_rows,
    }

    set_path = PILOT_SET_PATH if pilot else OUTPUT_SET_PATH
    key_path = PILOT_KEY_PATH if pilot else OUTPUT_KEY_PATH

    with open(set_path, "w", encoding="utf-8") as fh:
        json.dump(set_doc, fh, indent=2, ensure_ascii=False)
    with open(key_path, "w", encoding="utf-8") as fh:
        json.dump(key_doc, fh, indent=2, ensure_ascii=False)

    print("[write] labelling file -> " + set_path)
    print("[write] answer key     -> " + key_path)

    # ---- Step 5: the decision rule, fixed before any label exists ----------
    print("\n" + "=" * 70)
    print("STEP 5 -- the decision rule, fixed BEFORE any label exists")
    print("=" * 70)

    if pilot:
        print("The pilot does not decide the promotion. It decides whether the")
        print("full 60-judgement set is worth labelling at all.")
        print("")
        print("  If ANY pair is labelled 'different_fix':")
        print("     a false merge is observable in this region. The full set")
        print("     is worth labelling; run this script without --pilot.")
        print("")
        print("  If EVERY pair is labelled 'same_fix':")
        print("     the most divergent pairs in the entire disagreement region")
        print("     still share a fix. Neither configuration makes a false")
        print("     merge anywhere that they differ, so there is no precision")
        print("     signal to measure and the full 60 would only re-measure")
        print("     recall. That is the Phase 2 finding, not a failed run.")
    else:
        print("PRIMARY -- precision of each configuration's EXTRA merges.")
        print("Production's stated design choice is precision over recall: a")
        print("false 'these two share a fix' claim is costlier than a missed")
        print("automation opportunity. So the gate is the costly error rate,")
        print("not who merges more.")
        print("")
        print("  Each configuration's extra merges are the pairs it uniquely")
        print("  co-clusters. A 'different_fix' label on one of those is a")
        print("  FALSE MERGE by that configuration.")
        print("")
        print("  Promote " + CONFIG_B_NAME + " only if BOTH hold:")
        print("    (1) it makes ZERO observed false merges on its extra"
              " merges -")
        print("        the same standard by which the live 0.80 threshold was")
        print("        chosen (last point with precision exactly 1.0000); and")
        print("    (2) its false-merge rate is not worse than "
              + CONFIG_A_NAME + "'s.")
        print("")
        print("SECONDARY -- the binomial win-rate head-to-head, reported but")
        print("NOT decisive. It is a RECALL comparison: because every pair in")
        print("this region is within-template, a win there mostly restates")
        print("that one model merges more.")
        print("")
        print("  Any other outcome leaves production on " + CONFIG_A_NAME
              + " -- a null result is a result.")

    print("")
    if pilot:
        print("Next: label data/automation_flag_validation_pilot.json, then run")
        print("      python src/experiments/score_flag_validation_set.py --pilot")
    else:
        print("Next: label data/automation_flag_validation_set.json, then run")
        print("      python src/experiments/score_flag_validation_set.py")
    print("\n" + "=" * 70)
    print("DONE")
    print("=" * 70)


if __name__ == "__main__":
    main()
