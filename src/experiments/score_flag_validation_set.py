# src/experiments/score_flag_validation_set.py
"""
score_flag_validation_set.py
============================

Phase 2, step 2A: score the AUTOMATION-FLAG VALIDATION SET.

WHAT THIS SCRIPT IS
-------------------
The companion to build_flag_validation_set.py. It joins the human labels
against the withheld answer key and adjudicates the open production
question: should resolution clustering swap from MiniLM @ 0.80 to BGE @ 0.90?

    default   data/automation_flag_validation_set.json    (60-pair head-to-head)
    --pilot   data/automation_flag_validation_pilot.json  (12-pair probe)

THE DECISION RULE, AND WHY IT IS NOT A WIN-RATE
-----------------------------------------------
An earlier draft of this scorer decided the question with an exact binomial
test on which configuration won more discordant pairs. That rule was wrong
for this project, and the reason is worth recording.

Building the set surfaced a structural property of the data: at their own
cliff-edges NEITHER configuration ever merges across dataset templates. All
1,069 pairs both configurations merge, and all 411 pairs they disagree
about, are within-template. So the entire difference between them is RECALL
-- which within-template pairs each one manages to find.

A win-rate rule therefore promotes whichever model merges more. That
directly contradicts the production design choice, stated in the README and
baked into how the live 0.80 threshold was picked: a false "these two share
a fix" claim is costlier than a missed automation opportunity. Precision
over recall.

So the PRIMARY statistic here is the precision of each configuration's EXTRA
merges -- the pairs it uniquely co-clusters. A "different_fix" label on one
of those is a false merge by that configuration, and a false merge is the
costly error.

    Promote BGE only if BOTH hold:
      (1) zero observed false merges on its extra merges -- the same
          standard by which the live threshold was chosen (last point with
          precision exactly 1.0000); and
      (2) its false-merge rate is no worse than MiniLM's.

The binomial win-rate is still reported, clearly demoted, and labelled as
the recall comparison it is.

A null result is a result. This project reports calibration attempts that
failed, and "the evidence does not justify the swap" is a publishable
finding.

THE PILOT
---------
--pilot scores the 12-pair divergent-end probe instead. The pilot is
deliberately balanced across directions, which would bias a win-rate
comparison, so this script REFUSES to compute the head-to-head on it. The
pilot answers exactly one question: is a false merge observable in this
region at all?

SECONDARY MEASUREMENT
---------------------
The key carries each pair's `scenario_id_match` -- whether the dataset
generator considers the two tickets the same template. That is NOT the
ground truth (see build_flag_validation_set.py for why). Because every pair
in the region is within-template, this is expected to be degenerate; it is
reported so that degeneracy is visible rather than assumed.

WHAT IT WRITES
--------------
    data/automation_flag_validation_results.csv
    data/automation_flag_validation_pilot_results.csv   (--pilot)

This script is offline, deterministic, loads no model, and spends no Gemini
quota.
"""

from __future__ import annotations

import argparse
import os
import csv
import json
import math
import sys
from collections import defaultdict

VALID_LABELS = ("same_fix", "different_fix", "unclear")

ALPHA = 0.05

# Paths (os.path.* per project convention). Two levels up to project root.
_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(_THIS_DIR, os.pardir, os.pardir))
DATA_DIR = os.path.join(PROJECT_ROOT, "data")

SET_PATH = os.path.join(DATA_DIR, "automation_flag_validation_set.json")
KEY_PATH = os.path.join(DATA_DIR, "automation_flag_validation_key.json")
OUTPUT_CSV_PATH = os.path.join(
    DATA_DIR, "automation_flag_validation_results.csv")

PILOT_SET_PATH = os.path.join(DATA_DIR, "automation_flag_validation_pilot.json")
PILOT_KEY_PATH = os.path.join(
    DATA_DIR, "automation_flag_validation_pilot_key.json")
PILOT_CSV_PATH = os.path.join(
    DATA_DIR, "automation_flag_validation_pilot_results.csv")


def die(message):
    print("\n[ERROR] " + message + "\n", file=sys.stderr)
    sys.exit(1)


def load_or_die(path, what, how):
    if not os.path.exists(path):
        die("Missing " + what + ":\n    " + path + "\nBuild it first with:\n    "
            + how)
    with open(path, "r", encoding="utf-8") as fh:
        return json.load(fh)


def exact_two_sided_binomial_p(k, n):
    """Exact two-sided binomial p-value for k successes in n trials, p=0.5.

    Under p = 0.5 the distribution is symmetric, so this is twice the
    smaller tail, capped at 1.0.
    """
    if n == 0:
        return 1.0
    tail = min(k, n - k)
    cumulative = sum(math.comb(n, i) for i in range(0, tail + 1))
    return min(1.0, 2.0 * cumulative / (2.0 ** n))


def rule_of_three_upper_bound(n):
    """One-sided 95% upper bound on a rate after n trials with 0 events.

    The standard 3/n approximation. With no observed false merges, this is
    what can honestly be claimed about the unobserved rate -- and at the
    sample sizes in play it is not a small number, which is the point.
    """
    if n <= 0:
        return 1.0
    return min(1.0, 3.0 / n)


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Score the Phase 2 / 2A automation-flag validation set. Default "
            "scores the full 60-pair head-to-head; --pilot scores the "
            "12-pair divergent-end false-merge probe."
        )
    )
    parser.add_argument(
        "--pilot",
        action="store_true",
        help="Score the 12-pair pilot probe instead of the full set.",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    pilot = args.pilot

    set_path = PILOT_SET_PATH if pilot else SET_PATH
    key_path = PILOT_KEY_PATH if pilot else KEY_PATH
    csv_path = PILOT_CSV_PATH if pilot else OUTPUT_CSV_PATH
    build_cmd = ("python src/experiments/build_flag_validation_set.py"
                 + (" --pilot" if pilot else ""))

    print("=" * 70)
    if pilot:
        print("Phase 2 / 2A -- score the FALSE-MERGE PILOT PROBE")
    else:
        print("Phase 2 / 2A -- score the automation-flag validation set")
    print("=" * 70)

    set_doc = load_or_die(set_path, "validation set", build_cmd)
    key_doc = load_or_die(key_path, "validation answer key", build_cmd)

    set_meta = set_doc.get("_meta", {})
    key_meta = key_doc.get("_meta", {})

    # Guard: a pilot file scored as the full set (or vice versa) would be
    # read as evidence it is not. Refuse rather than quietly mis-report.
    expected_type = "pilot" if pilot else "full"
    for name, meta in (("set", set_meta), ("key", key_meta)):
        actual = meta.get("set_type")
        if actual and actual != expected_type:
            die(
                "The " + name + " file at\n    "
                + (set_path if name == "set" else key_path)
                + "\nis a '" + actual + "' file, but this run expects '"
                + expected_type + "'.\n"
                + ("Drop --pilot to score the full set."
                   if expected_type == "pilot" else
                   "Add --pilot to score the pilot probe.")
            )

    cfg_a = key_meta.get("config_a", {})
    cfg_b = key_meta.get("config_b", {})
    name_a = cfg_a.get("name", "config_a")
    name_b = cfg_b.get("name", "config_b")

    print("Configuration A: " + name_a + "  (model " + str(cfg_a.get("model")) + ")")
    print("Configuration B: " + name_b + "  (model " + str(cfg_b.get("model")) + ")")

    labels = {p["pair_id"]: p for p in set_doc.get("pairs", [])}
    keys = {p["pair_id"]: p for p in key_doc.get("pairs", [])}

    if set(labels) != set(keys):
        only_set = sorted(set(labels) - set(keys))
        only_key = sorted(set(keys) - set(labels))
        die(
            "The validation set and answer key do not describe the same "
            "pairs.\n  Only in set: " + str(only_set[:10])
            + "\n  Only in key: " + str(only_key[:10])
            + "\nRegenerate BOTH together with:\n    " + build_cmd
            + "\nA set scored against a mismatched key is exactly the "
            "stale-artifact failure this project keeps hitting."
        )

    # ---- Step 1: completeness and label validity ---------------------------
    print("\n" + "=" * 70)
    print("STEP 1 -- label completeness")
    print("=" * 70)

    unlabelled = [pid for pid, p in labels.items() if not p.get("label")]
    invalid = [(pid, p.get("label")) for pid, p in labels.items()
               if p.get("label") and p["label"] not in VALID_LABELS]

    if invalid:
        die("Invalid label value(s): " + str(invalid[:10])
            + "\nValid labels are exactly: " + str(VALID_LABELS))

    total = len(labels)
    print("[labels] total pairs     : " + str(total))
    print("[labels] labelled        : " + str(total - len(unlabelled)))
    print("[labels] still unlabelled: " + str(len(unlabelled)))

    if unlabelled:
        die(
            str(len(unlabelled)) + " pair(s) are still unlabelled, e.g. "
            + str(sorted(unlabelled)[:10]) + "\n"
            "Scoring a partially-labelled set would silently bias the result "
            "toward whichever pairs happened to be easy to judge.\n"
            "Finish labelling\n    " + set_path + "\nfirst."
        )

    # ---- Step 2: false merges, per configuration ---------------------------
    print("\n" + "=" * 70)
    print("STEP 2 -- PRIMARY: false merges on each configuration's extra merges")
    print("=" * 70)
    print("A pair sampled here is co-clustered by exactly one configuration.")
    print("A 'different_fix' label on it is a FALSE MERGE by that one -- the")
    print("costly error under production's precision-over-recall stance.")

    rows = []
    extra = {name_a: {"total": 0, "false": 0, "true": 0, "unclear": 0},
             name_b: {"total": 0, "false": 0, "true": 0, "unclear": 0}}
    wins_a = 0
    wins_b = 0
    unclear = 0
    scenario_agree = 0
    scenario_total = 0
    per_cat = defaultdict(lambda: {"a": 0, "b": 0, "unclear": 0})
    false_merge_examples = []

    for pid in sorted(labels):
        lab = labels[pid]
        key = keys[pid]
        label = lab["label"]
        co_by = key["co_clustered_by"]
        sep_by = key["separated_by"]
        category = key["category"]

        bucket = extra.setdefault(
            co_by, {"total": 0, "false": 0, "true": 0, "unclear": 0})
        bucket["total"] += 1

        if label == "unclear":
            winner = ""
            unclear += 1
            bucket["unclear"] += 1
            per_cat[category]["unclear"] += 1
        elif label == "same_fix":
            winner = co_by
            bucket["true"] += 1
        else:
            winner = sep_by
            bucket["false"] += 1
            false_merge_examples.append((pid, co_by, category))

        if winner == name_a:
            wins_a += 1
            per_cat[category]["a"] += 1
        elif winner == name_b:
            wins_b += 1
            per_cat[category]["b"] += 1

        if label != "unclear":
            scenario_total += 1
            if bool(key.get("scenario_id_match")) == (label == "same_fix"):
                scenario_agree += 1

        rows.append({
            "pair_id": pid,
            "category": category,
            "ticket_a": lab["ticket_a"]["batch_ticket_id"],
            "ticket_b": lab["ticket_b"]["batch_ticket_id"],
            "divergence": key.get("divergence"),
            "human_label": label,
            "co_clustered_by": co_by,
            "separated_by": sep_by,
            "false_merge_by": co_by if label == "different_fix" else "",
            "winner": winner,
            "scenario_id_match": key.get("scenario_id_match"),
            "label_reason": lab.get("label_reason", ""),
        })

    hdr = "{:<16}{:>9}{:>10}{:>9}{:>10}{:>16}".format(
        "Configuration", "extra", "same_fix", "FALSE", "unclear", "false rate")
    print("\n" + hdr)
    print("-" * len(hdr))
    for cname in (name_a, name_b):
        e = extra[cname]
        judged = e["true"] + e["false"]
        rate = ("{:.1%}".format(e["false"] / judged) if judged else "n/a")
        print("{:<16}{:>9}{:>10}{:>9}{:>10}{:>16}".format(
            cname, e["total"], e["true"], e["false"], e["unclear"], rate))

    for cname in (name_a, name_b):
        e = extra[cname]
        judged = e["true"] + e["false"]
        if judged and e["false"] == 0:
            ub = rule_of_three_upper_bound(judged)
            print("\n[bound] " + cname + ": 0 false merges in " + str(judged)
                  + " judged extra merges.")
            print("        One-sided 95% upper bound on its true false-merge")
            print("        rate is {:.1%} (rule of three). Zero observed is".format(ub))
            print("        not the same as zero.")

    if false_merge_examples:
        print("\n[false] observed false merges:")
        for pid, cname, category in false_merge_examples:
            print("        " + pid + "  " + cname + "  (" + category + ")")

    # ---- Step 3: pilot verdict, or the full head-to-head -------------------
    if pilot:
        print("\n" + "=" * 70)
        print("STEP 3 -- PILOT VERDICT")
        print("=" * 70)
        print("The pilot is balanced across directions by design, so a")
        print("win-rate comparison on it would be biased. It is deliberately")
        print("NOT computed here.")
        print("")
        total_false = sum(extra[c]["false"] for c in (name_a, name_b))
        judged_all = sum(extra[c]["true"] + extra[c]["false"]
                         for c in (name_a, name_b))
        print("[pilot] pairs judged (excluding 'unclear'): " + str(judged_all))
        print("[pilot] false merges observed             : " + str(total_false))
        print("")
        if total_false > 0:
            print("VERDICT: PROCEED TO THE FULL SET.")
            print("  A false merge IS observable in the disagreement region,")
            print("  so the full 60-judgement set can measure a real precision")
            print("  difference between the two configurations.")
            print("")
            print("  Next: python src/experiments/build_flag_validation_set.py")
            print("        label data/automation_flag_validation_set.json")
            print("        python src/experiments/score_flag_validation_set.py")
        else:
            print("VERDICT: STOP. The full set would not measure precision.")
            print("  The most textually divergent pairs in the entire")
            print("  disagreement region still share a fix. Neither")
            print("  configuration makes an observable false merge anywhere")
            print("  the two differ, so there is no precision signal to")
            print("  measure and the remaining 48 judgements would only")
            print("  re-measure recall -- which the cluster counts already")
            print("  report for free.")
            print("")
            print("  This is the Phase 2 finding: on this dataset the")
            print("  promotion question cannot be settled by flag")
            print("  correctness. " + name_a + " and " + name_b + " are")
            print("  indistinguishable on precision and differ only in")
            print("  recall, so promotion is a product decision about how")
            print("  many candidates to surface, not an evidence-backed one.")
            print("")
            print("  Production stays on " + name_a + " absent that decision.")
    else:
        print("\n" + "=" * 70)
        print("STEP 3 -- SECONDARY: win-rate head-to-head (recall comparison)")
        print("=" * 70)
        n = wins_a + wins_b
        print("Reported, but NOT decisive. Every pair in this region is")
        print("within-template, so a win here mostly restates which model")
        print("merges more. See the module docstring.")
        print("")
        print("[recall] pairs scored : " + str(n)
              + "   ('unclear' excluded: " + str(unclear) + ")")
        print("[recall] " + name_a.ljust(16) + ": " + str(wins_a))
        print("[recall] " + name_b.ljust(16) + ": " + str(wins_b))
        if n:
            p_value = exact_two_sided_binomial_p(min(wins_a, wins_b), n)
            print("[recall] exact two-sided binomial p = {:.5f}".format(p_value))

        # ---- per-category ---------------------------------------------------
        print("\n" + "-" * 70)
        print("Per-category breakdown (descriptive only -- the set was sized")
        print("for one comparison, not seven).")
        chdr = "{:<20}{:>14}{:>12}{:>10}".format(
            "Category", name_a, name_b, "unclear")
        print("\n" + chdr)
        print("-" * len(chdr))
        for category in sorted(per_cat):
            c = per_cat[category]
            print("{:<20}{:>14}{:>12}{:>10}".format(
                category, c["a"], c["b"], c["unclear"]))

        # ---- verdict --------------------------------------------------------
        print("\n" + "=" * 70)
        print("STEP 4 -- verdict against the pre-registered decision rule")
        print("=" * 70)
        print("Rule (fixed before labelling): promote " + name_b + " only if")
        print("  (1) it makes ZERO observed false merges on its extra merges,")
        print("      AND")
        print("  (2) " + name_a + " makes at least one -- i.e. there is an")
        print("      actual precision difference to promote on.")
        print("")
        print("If NEITHER makes a false merge the verdict is NO PRECISION")
        print("SIGNAL, not promotion: a tie on the costly error leaves only")
        print("recall, and promoting on recall through a precision gate would")
        print("invert production's stated cost asymmetry.")
        print("")

        eb = extra[name_b]
        ea = extra[name_a]
        judged_b = eb["true"] + eb["false"]
        judged_a = ea["true"] + ea["false"]
        rate_b = (eb["false"] / judged_b) if judged_b else None
        rate_a = (ea["false"] / judged_a) if judged_a else None

        if judged_b == 0 or judged_a == 0:
            print("VERDICT: INCONCLUSIVE. Production stays on " + name_a + ".")
            print("  One of the configurations has no judged extra merges, so")
            print("  there is nothing to compare.")
        elif eb["false"] > 0:
            print("VERDICT: DO NOT PROMOTE. Production stays on " + name_a + ".")
            print("  " + name_b + " made " + str(eb["false"])
                  + " observed false merge(s) in " + str(judged_b))
            print("  judged extra merges. Under precision-over-recall that is")
            print("  disqualifying on its own, whatever the recall gain.")
            print("  A null result is a result.")
        elif ea["false"] == 0:
            # Neither configuration made an observed false merge. There is no
            # precision difference to promote on -- and saying otherwise would
            # smuggle a recall argument in through a precision gate.
            print("VERDICT: NO PRECISION SIGNAL. Production stays on "
                  + name_a + " absent a separate product decision.")
            print("  Neither configuration made an observed false merge:")
            print("    " + name_a + ": 0 / " + str(judged_a)
                  + "   " + name_b + ": 0 / " + str(judged_b))
            print("  So the two are INDISTINGUISHABLE on the axis the")
            print("  production threshold was chosen to protect. The only")
            print("  measured difference between them is recall -- "
                  + name_b + " merges")
            print("  more -- and that is already visible in the cluster counts")
            print("  without any human labelling.")
            print("")
            print("  Promoting " + name_b + " would therefore be a PRODUCT")
            print("  decision about how many automation candidates to surface,")
            print("  not an evidence-backed precision improvement. It should")
            print("  be recorded and argued as such, not as a calibration")
            print("  result. Note also the rule-of-three bounds above: zero")
            print("  observed is not zero, and at these sample sizes neither")
            print("  configuration's true false-merge rate is tightly bounded.")
        elif rate_b is not None and rate_a is not None and rate_b < rate_a:
            print("VERDICT: PROMOTE " + name_b + ".")
            print("  " + name_b + " made zero false merges in " + str(judged_b)
                  + " judged extra merges,")
            print("  while " + name_a + " made " + str(ea["false"]) + " in "
                  + str(judged_a) + ". " + name_b + " is strictly better on")
            print("  the costly error, which is the axis the threshold exists")
            print("  to protect.")
            print("  Promoting is a SEPARATE change: update")
            print("  settings.clustering, add the CALIBRATION_PROVENANCE")
            print("  entry, re-run the adversarial gate and goldens.")
        else:
            print("VERDICT: DO NOT PROMOTE. Production stays on " + name_a + ".")
            print("  " + name_b + "'s false-merge rate is not better than "
                  + name_a + "'s.")
            print("  A null result is a result.")

    # ---- secondary scenario_id measurement ---------------------------------
    print("\n" + "=" * 70)
    print("SECONDARY -- template proxy vs human judgement")
    print("=" * 70)
    if scenario_total:
        rate = scenario_agree / scenario_total
        print("[secondary] scenario_id agrees with the human label on "
              + str(scenario_agree) + "/" + str(scenario_total)
              + " = {:.1%}".format(rate))
        print("[secondary] NOT the ground truth. Every pair in this region is")
        print("[secondary] within-template, so this number is expected to be")
        print("[secondary] degenerate -- it is printed so the degeneracy is")
        print("[secondary] visible rather than assumed.")
    else:
        print("[secondary] no non-unclear labels; nothing to compare.")

    # ---- CSV ----------------------------------------------------------------
    print("\n" + "=" * 70)
    print("Writing CSV report")
    print("=" * 70)
    fieldnames = ["pair_id", "category", "ticket_a", "ticket_b", "divergence",
                  "human_label", "co_clustered_by", "separated_by",
                  "false_merge_by", "winner", "scenario_id_match",
                  "label_reason"]
    with open(csv_path, "w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)
    print("[write] " + str(len(rows)) + " row(s) -> " + csv_path)

    print("\n" + "=" * 70)
    print("DONE")
    print("=" * 70)


if __name__ == "__main__":
    main()
