"""Error analysis of the production cascade on the 45-ticket benchmark.

Paper revision (Oct 2026): Fig. 6, Table IX and Section VII (E1, E3, E4, X2).
Offline and model-free: reads committed files only.

Every count is taken from TWO independent committed derivations and is FATAL
on any disagreement:

  * tests/goldens/benchmark_baseline.json -- the routing decision recorded by
    the golden-parity capture (category, tier, escalation);
  * data/ablation_baseline_results.csv -- the same pipeline's predictions as
    written by run_ablation_study.py --mode baseline (category only);
  * data/groundedness_results.csv -- the benchmark tickets that actually
    reached the resolver in Phase 2B (a draft exists), which must be exactly
    the tickets the goldens record as not escalated.

Writes only under data/paper_revision_oct/ and refuses to overwrite without
--force.

    .\\venv\\Scripts\\python.exe scripts/error_analysis_benchmark45.py
"""

from __future__ import annotations

import argparse
import csv
import json
import os
from collections import Counter

PROJECT_ROOT = os.path.abspath(
    os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
DATA_DIR = os.path.join(PROJECT_ROOT, "data")
OUT_DIR = os.path.join(DATA_DIR, "paper_revision_oct")
GOLDEN = os.path.join(PROJECT_ROOT, "tests", "goldens", "benchmark_baseline.json")
ABLATION = os.path.join(DATA_DIR, "ablation_baseline_results.csv")
GROUNDED = os.path.join(DATA_DIR, "groundedness_results.csv")
BENCHMARK = os.path.join(DATA_DIR, "novel_tickets_expanded.json")
OUT_TICKETS = os.path.join(OUT_DIR, "error_analysis_benchmark45_tickets.csv")
OUT_CONFUSION = os.path.join(OUT_DIR, "confusion_matrix_cascade_benchmark45.csv")
OUT_SUMMARY = os.path.join(OUT_DIR, "error_analysis_benchmark45_summary.csv")


def die(msg):
    print(f"\n[FATAL] {msg}")
    raise SystemExit(1)


def ticket_id(index):
    # build_groundedness_set.py's convention: "N%02d" % (i + 1).
    return "N%02d" % (index + 1)


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args()
    outs = (OUT_TICKETS, OUT_CONFUSION, OUT_SUMMARY)
    if any(os.path.exists(p) for p in outs) and not args.force:
        die("Outputs exist (use --force to overwrite).")

    with open(GOLDEN, encoding="utf-8") as fh:
        golden = sorted(json.load(fh)["rows"], key=lambda r: r["index"])
    with open(ABLATION, newline="", encoding="utf-8") as fh:
        ablation = [r for r in csv.DictReader(fh)
                    if r["section"] == "classification"]
    with open(GROUNDED, newline="", encoding="utf-8") as fh:
        grounded = [r for r in csv.DictReader(fh) if r["source"] == "novel-45"]
    with open(BENCHMARK, encoding="utf-8") as fh:
        bench = json.load(fh)

    n = len(bench)
    if not (len(golden) == len(ablation) == n):
        die(f"row counts differ: golden {len(golden)}, ablation "
            f"{len(ablation)}, benchmark {n}")

    # --- derivation 1 vs 2: category ---------------------------------------
    for g, a, b in zip(golden, ablation, bench):
        if int(a["key"]) != g["index"]:
            die(f"ablation key {a['key']} != golden index {g['index']}")
        if not (g["expected"] == a["expected"] == b["expected"]):
            die(f"{ticket_id(g['index'])}: expected label disagrees")
        if g["predicted_category"] != a["predicted"]:
            die(f"{ticket_id(g['index'])}: golden predicts "
                f"{g['predicted_category']}, ablation {a['predicted']}")
        if g["correct"] != (a["correct"] == "True"):
            die(f"{ticket_id(g['index'])}: correctness disagrees")
    print(f"  [ok] goldens and ablation CSV agree on all {n} predictions")

    # --- derivation 1 vs 3: escalation -------------------------------------
    drafted_golden = {ticket_id(g["index"]) for g in golden
                      if not g["actual_escalate"]}
    drafted_2b = {r["ticket_id"] for r in grounded}
    if drafted_golden != drafted_2b:
        die(f"drafted sets differ: only-golden "
            f"{sorted(drafted_golden - drafted_2b)}, only-2B "
            f"{sorted(drafted_2b - drafted_golden)}")
    for r in grounded:
        g = golden[int(r["ticket_id"][1:]) - 1]
        if r["predicted_category"] != g["predicted_category"]:
            die(f"{r['ticket_id']}: 2B predicted_category "
                f"{r['predicted_category']} != golden {g['predicted_category']}")
    print(f"  [ok] the {len(drafted_2b)} drafted tickets in Phase 2B are exactly "
          f"the goldens' non-escalated tickets, with the same categories")

    labels = {r["ticket_id"]: r["human_label"] for r in grounded}

    # --- write per-ticket table --------------------------------------------
    os.makedirs(OUT_DIR, exist_ok=True)
    with open(OUT_TICKETS, "w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(["ticket_id", "expected", "predicted", "correct", "tier",
                    "tier1_confidence", "top_similarity", "escalated",
                    "human_groundedness_label", "text"])
        for g, b in zip(golden, bench):
            tid = ticket_id(g["index"])
            w.writerow([tid, g["expected"], g["predicted_category"],
                        g["correct"], g["tier"], repr(g["tier1_confidence"]),
                        repr(g["rag_similarity"]), g["actual_escalate"],
                        labels.get(tid, ""), b["text"]])

    cats = sorted({g["expected"] for g in golden})
    cm = Counter((g["expected"], g["predicted_category"]) for g in golden)
    with open(OUT_CONFUSION, "w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(["true \\ predicted"] + cats)
        for t in cats:
            w.writerow([t] + [cm[(t, p)] for p in cats])

    mis = [g for g in golden if not g["correct"]]
    mis_esc = sum(g["actual_escalate"] for g in mis)
    ok = [g for g in golden if g["correct"]]
    ok_esc = sum(g["actual_escalate"] for g in ok)
    esc = sum(g["actual_escalate"] for g in golden)
    tier1 = sum(g["tier"] == 1 for g in golden)
    mis_drafted_ungrounded = sum(labels.get(ticket_id(g["index"])) ==
                                 "ungrounded" for g in mis)

    summary = [
        ("n", n),
        ("correct", len(ok)),
        ("misclassified", len(mis)),
        ("escalated", esc),
        ("drafted", n - esc),
        ("tier1_kept", tier1),
        ("misclassified_escalated", mis_esc),
        ("misclassified_drafted", len(mis) - mis_esc),
        ("misclassified_drafted_labelled_ungrounded", mis_drafted_ungrounded),
        ("correct_escalated", ok_esc),
        ("correct_drafted", len(ok) - ok_esc),
    ]
    for t in cats:
        tot = sum(g["expected"] == t for g in golden)
        summary.append((f"recall.{t}", f"{cm[(t, t)]}/{tot}"))
    for (t, p), c in sorted(cm.items(), key=lambda kv: (-kv[1], kv[0])):
        if t != p:
            summary.append((f"confusion.{t}->{p}", c))
    for p in cats:
        wrong_into = sum(c for (t, q), c in cm.items() if q == p and t != p)
        summary.append((f"wrong_predictions_into.{p}", wrong_into))

    with open(OUT_SUMMARY, "w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(["quantity", "value"])
        w.writerows(summary)
    for k, v in summary:
        print(f"  {k:<48} {v}")
    for p in outs:
        print(f"[write] {os.path.relpath(p, PROJECT_ROOT)}")


if __name__ == "__main__":
    main()
