"""Evaluate the unchanged pipeline on the author-written independent test set.

Paper revision (Oct 2026), Phase 3. See data/independent_test/GUIDELINES.md
for how the set is written and labelled: one author (AW) writes the tickets,
one second labeller labels every ticket blind, and disagreements are resolved
by discussion between the two. No LLM call, no retraining, no threshold
change.

Inputs (data/independent_test/):
  tickets_<CODE>.csv         the writer's tickets and labels (TEMPLATE.csv)
  second_labels_<CODE>.csv   blind second labels of that writer's tickets
                             (SECOND_LABELS_TEMPLATE.csv)
  adjudication.csv           the agreed label for every disagreement
                             (ADJUDICATION_TEMPLATE.csv); needed only if the
                             two labels disagree anywhere

Two annotation modes, and the mode is stamped into the summary:
  default             two annotators. FATAL unless every ticket has exactly
                      one second label from someone other than its writer.
                      Cohen's kappa (category and expected outcome) is
                      computed BEFORE adjudication, with raw agreement and a
                      seeded bootstrap 95% interval. Gold = the agreed label,
                      else the resolved one; FATAL if any is unresolved.
  --single-annotator  no second labels: kappa is skipped and gold = the
                      writer's label. FATAL if second-label files exist (the
                      mode must not silently discard them). The paper must
                      then state "single annotator"; the script says so.

--check-only validates and reports kappa / disagreements without loading any
model. Otherwise every ticket is routed through the production pipeline and
the same metrics as the real-world evaluation are written.

Outputs (data/independent_test/results/, refuses to overwrite without --force):
  independent_tickets_routed.csv, independent_summary.csv

    .\\venv\\Scripts\\python.exe scripts/eval_independent.py --check-only
    .\\venv\\Scripts\\python.exe scripts/eval_independent.py
    .\\venv\\Scripts\\python.exe scripts/eval_independent.py --single-annotator
"""

from __future__ import annotations

import argparse
import csv
import glob
import os
import re

from realworld_common import PROJECT_ROOT, die, route, summarise, wilson

DIR = os.path.join(PROJECT_ROOT, "data", "independent_test")
DATASET = os.path.join(PROJECT_ROOT, "data", "synthetic_tickets.csv")

OUT_OF_SCOPE = "Out of scope"
OUTCOMES = {"escalate", "proceed"}
CODE = re.compile(r"^[A-Z]{2,4}$")
TICKET_COLS = ["ticket_id", "author", "title", "description",
               "true_category", "expected_outcome", "notes"]
SECOND_COLS = ["ticket_id", "labeller", "second_category",
               "second_expected_outcome", "notes"]
ADJ_COLS = ["ticket_id", "final_category", "final_expected_outcome",
            "resolved_by", "reason"]
BOOTSTRAP_DRAWS, SEED = 10_000, 42
SINGLE_NOTE = ("SINGLE ANNOTATOR: these labels were assigned by the ticket "
               "writer alone and no inter-annotator agreement was measured. "
               "The paper must state that the independent set was labelled "
               "by a single annotator.")


def read(path, cols):
    with open(path, newline="", encoding="utf-8-sig") as fh:
        rdr = csv.DictReader(fh)
        if rdr.fieldnames != cols:
            die(f"{os.path.basename(path)}: columns {rdr.fieldnames}, "
                f"expected {cols}")
        return [{k: (v or "").strip() for k, v in r.items()} for r in rdr]


def categories():
    import pandas as pd
    return sorted(pd.read_csv(DATASET)["category"].unique())


def load_tickets(cats, warnings):
    tickets, seen = [], set()
    files = sorted(glob.glob(os.path.join(DIR, "tickets_*.csv")))
    if not files:
        die(f"no tickets_<CODE>.csv files in {DIR}")
    for path in files:
        code = os.path.basename(path)[len("tickets_"):-len(".csv")]
        if not CODE.match(code):
            die(f"{os.path.basename(path)}: author code {code!r} must be 2-4 "
                "capital letters")
        for r in read(path, TICKET_COLS):
            tid = r["ticket_id"]
            if not tid.startswith(f"{code}-") or r["author"] != code:
                die(f"{tid}: id/author does not match file code {code}")
            if tid in seen:
                die(f"duplicate ticket_id {tid}")
            seen.add(tid)
            if not r["description"]:
                die(f"{tid}: empty description")
            if r["true_category"] not in cats:
                die(f"{tid}: true_category {r['true_category']!r} not in "
                    f"{sorted(cats)}")
            if r["expected_outcome"] not in OUTCOMES:
                die(f"{tid}: expected_outcome {r['expected_outcome']!r}")
            if (r["true_category"] == OUT_OF_SCOPE) != \
                    (r["expected_outcome"] == "escalate"):
                warnings.append(f"{tid}: {r['true_category']} with "
                                f"expected_outcome {r['expected_outcome']} "
                                "(guidelines pair Out of scope with escalate)")
            tickets.append(r)
    return tickets


def load_second(tickets, cats):
    second = {}
    for path in sorted(glob.glob(os.path.join(DIR, "second_labels_*.csv"))):
        for r in read(path, SECOND_COLS):
            if r["ticket_id"] in second:
                die(f"{r['ticket_id']}: second-labelled twice")
            if not CODE.match(r["labeller"]):
                die(f"{r['ticket_id']}: labeller code {r['labeller']!r} must "
                    "be 2-4 capital letters")
            if r["second_category"] not in cats or \
                    r["second_expected_outcome"] not in OUTCOMES:
                die(f"{r['ticket_id']}: invalid second label")
            second[r["ticket_id"]] = r
    ids = {t["ticket_id"] for t in tickets}
    missing, extra = sorted(ids - set(second)), sorted(set(second) - ids)
    if missing or extra:
        die(f"second labels: {len(missing)} tickets missing "
            f"{missing[:10]}{'...' if len(missing) > 10 else ''}, "
            f"{len(extra)} unknown ids {extra[:10]}")
    for t in tickets:
        if second[t["ticket_id"]]["labeller"] == t["author"]:
            die(f"{t['ticket_id']}: second labeller is the writer")
    return second


def load_adjudication(cats):
    adj = {}
    path = os.path.join(DIR, "adjudication.csv")
    if os.path.exists(path):
        for r in read(path, ADJ_COLS):
            if r["final_category"] not in cats or \
                    r["final_expected_outcome"] not in OUTCOMES:
                die(f"{r['ticket_id']}: invalid resolved label")
            if not r["resolved_by"]:
                die(f"{r['ticket_id']}: resolved_by is empty")
            adj[r["ticket_id"]] = r
    return adj


def _kappa(ia, ib, n_labels):
    """Cohen's kappa from integer-coded labels; nan when it is undefined."""
    import numpy as np
    cm = np.zeros((n_labels, n_labels))
    np.add.at(cm, (ia, ib), 1)
    n = cm.sum()
    po = np.trace(cm) / n
    pe = float(cm.sum(1) @ cm.sum(0)) / (n * n)
    return float("nan") if pe == 1.0 else (po - pe) / (1.0 - pe)


def kappa_report(a, b, name):
    import numpy as np
    from sklearn.metrics import cohen_kappa_score
    a, b = np.asarray(a), np.asarray(b)
    k = float(cohen_kappa_score(a, b))
    labels = sorted(set(a) | set(b))
    ia = np.searchsorted(labels, a)
    ib = np.searchsorted(labels, b)
    if abs(_kappa(ia, ib, len(labels)) - k) > 1e-12:
        die(f"kappa[{name}]: bootstrap implementation disagrees with sklearn")
    rng = np.random.default_rng(SEED)
    boots = []
    for _ in range(BOOTSTRAP_DRAWS):
        idx = rng.integers(0, len(a), len(a))
        kb = _kappa(ia[idx], ib[idx], len(labels))
        if kb == kb:  # skip draws where kappa is undefined
            boots.append(kb)
    lo, hi = (float(np.percentile(boots, 2.5)),
              float(np.percentile(boots, 97.5)))
    agree = int((a == b).sum())
    print(f"  kappa[{name}] = {k:.3f}  bootstrap 95% [{lo:.3f}, {hi:.3f}]  "
          f"raw agreement {agree}/{len(a)}")
    return [(f"kappa.{name}", repr(k)), (f"kappa.{name}.boot95_lo", repr(lo)),
            (f"kappa.{name}.boot95_hi", repr(hi)),
            (f"kappa.{name}.agreement", f"{agree}/{len(a)}")]


def gold_from_two(tickets, second, adj):
    unresolved = []
    for t in tickets:
        s = second[t["ticket_id"]]
        if (t["true_category"] == s["second_category"] and
                t["expected_outcome"] == s["second_expected_outcome"]):
            t["gold_category"] = t["true_category"]
            t["gold_outcome"] = t["expected_outcome"]
            t["label_source"] = "agreed"
            continue
        a = adj.get(t["ticket_id"])
        if a is None:
            unresolved.append(f"{t['ticket_id']}: writer "
                              f"{t['true_category']}/{t['expected_outcome']}"
                              f" vs second {s['second_category']}/"
                              f"{s['second_expected_outcome']}")
            continue
        t["gold_category"] = a["final_category"]
        t["gold_outcome"] = a["final_expected_outcome"]
        t["label_source"] = "resolved"
    return unresolved


def main():
    global DIR
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--check-only", action="store_true",
                    help="validate and report kappa/disagreements; no model")
    ap.add_argument("--single-annotator", action="store_true",
                    help="no second labels: skip kappa, gold = writer labels")
    ap.add_argument("--force", action="store_true")
    ap.add_argument("--dir", default=None,
                    help="read and write another directory (testing only)")
    args = ap.parse_args()
    if args.dir:
        DIR = os.path.abspath(args.dir)
    out_dir = os.path.join(DIR, "results")
    out_tickets = os.path.join(out_dir, "independent_tickets_routed.csv")
    out_summary = os.path.join(out_dir, "independent_summary.csv")
    if not args.check_only and os.path.exists(out_summary) and not args.force:
        die(f"Output exists (use --force to overwrite):\n    {out_summary}")

    cats = set(categories()) | {OUT_OF_SCOPE}
    warnings = []
    tickets = load_tickets(cats, warnings)
    print(f"[load] {len(tickets)} tickets by "
          f"{sorted({t['author'] for t in tickets})}")
    for w in warnings:
        print(f"  [warn] {w}")
    summary = [("n_tickets", len(tickets))]

    has_second = bool(glob.glob(os.path.join(DIR, "second_labels_*.csv")))
    if args.single_annotator:
        if has_second:
            die("--single-annotator given but second_labels_*.csv exist; "
                "drop the flag so kappa is computed")
        for t in tickets:
            t["gold_category"] = t["true_category"]
            t["gold_outcome"] = t["expected_outcome"]
            t["label_source"] = "single_annotator"
        summary.append(("annotation", "single annotator"))
        print(f"\n[NOTE] {SINGLE_NOTE}\n")
    else:
        if not has_second:
            die("no second_labels_*.csv found. Add the blind second labels, "
                "or run with --single-annotator (the paper must then say "
                "'single annotator').")
        second = load_second(tickets, cats)
        labellers = sorted({second[t["ticket_id"]]["labeller"]
                            for t in tickets})
        summary += [("annotation", "two annotators"),
                    ("second_labellers", "+".join(labellers))]
        summary += kappa_report([t["true_category"] for t in tickets],
                                [second[t["ticket_id"]]["second_category"]
                                 for t in tickets], "category")
        summary += kappa_report([t["expected_outcome"] for t in tickets],
                                [second[t["ticket_id"]]
                                 ["second_expected_outcome"]
                                 for t in tickets], "outcome")
        unresolved = gold_from_two(tickets, second, load_adjudication(cats))
        if unresolved:
            print(f"\n[disagreements to resolve by discussion: "
                  f"{len(unresolved)}]")
            for u in unresolved:
                print(f"  {u}")
            if not args.check_only:
                die("record an agreed label for every disagreement in "
                    "adjudication.csv first")
        summary.append(("resolved_by_discussion",
                        sum(t.get("label_source") == "resolved"
                            for t in tickets)))

    if args.check_only:
        print("\n[check-only] nothing routed, nothing written")
        return

    rows = route([(t["ticket_id"], t["title"], t["description"])
                  for t in tickets], progress_every=50)
    for r, t in zip(rows, tickets):
        if r["ticket_id"] != t["ticket_id"]:
            die("row order lost during routing")
        r.update({"author": t["author"], "gold_category": t["gold_category"],
                  "gold_outcome": t["gold_outcome"],
                  "label_source": t["label_source"]})

    in_scope = [r for r in rows if r["gold_category"] != OUT_OF_SCOPE]
    summary.append(("n_in_scope", len(in_scope)))
    summary += [(f"in_scope.{k}", v)
                for k, v in summarise(in_scope, gold_key="gold_category",
                                      label="in-scope")]

    def prop(name, k, n):
        lo, hi = wilson(k, n)
        summary.extend([(f"{name}.k", k), (f"{name}.n", n),
                        (f"{name}.rate", repr(k / n)),
                        (f"{name}.wilson95_lo", repr(lo)),
                        (f"{name}.wilson95_hi", repr(hi))])

    prop("all.escalated", sum(r["escalated"] for r in rows), len(rows))
    esc = [r for r in rows if r["gold_outcome"] == "escalate"]
    pro = [r for r in rows if r["gold_outcome"] == "proceed"]
    if esc:
        prop("should_escalate.escalated", sum(r["escalated"] for r in esc),
             len(esc))
    if pro:
        prop("should_proceed.proceeded",
             sum(not r["escalated"] for r in pro), len(pro))

    os.makedirs(out_dir, exist_ok=True)
    with open(out_tickets, "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    with open(out_summary, "w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(["quantity", "value"])
        w.writerows(summary)
    for k, v in summary:
        print(f"  {k:<48} {v}")
    if args.single_annotator:
        print(f"\n[NOTE] {SINGLE_NOTE}")
    print(f"[write] {os.path.relpath(out_tickets, PROJECT_ROOT)}")
    print(f"[write] {os.path.relpath(out_summary, PROJECT_ROOT)}")


if __name__ == "__main__":
    main()
