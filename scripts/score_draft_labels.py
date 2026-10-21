"""Count AW's hand labels on the 30 real-ticket drafts.

Paper revision (Oct 2026), Phase 2. Offline, model-free. Reads
data/external_zenodo_7648117/draft_labels.csv (filled from
DRAFT_LABELS_README.md) and writes draft_labels_summary.csv beside it, the
only source the paper may quote for draft quality on real tickets. The labels
are a single annotator's; the summary says so.

FATAL unless every one of the 30 ids from drafts.json has a valid label in
both columns: grounded in {yes, no}; appropriate_handoff in {yes, no, na},
where yes = the retrieved tickets did not fit and the draft declined and
handed off to a human; no = they did not fit but the draft still gave a fix;
na = they did fit. A generic "a human should double-check" line alone is not
a hand-off.

    .\\venv\\Scripts\\python.exe scripts/score_draft_labels.py
"""

from __future__ import annotations

import argparse
import csv
import json
import os
from collections import Counter

from realworld_common import PROJECT_ROOT, die, wilson

BASE = os.path.join(PROJECT_ROOT, "data", "external_zenodo_7648117")
DRAFTS = os.path.join(BASE, "drafts.json")
LABELS = os.path.join(BASE, "draft_labels.csv")
OUT = os.path.join(BASE, "draft_labels_summary.csv")
COLS = ["id", "grounded", "appropriate_handoff", "notes"]


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args()
    if os.path.exists(OUT) and not args.force:
        die(f"Output exists (use --force to overwrite):\n    {OUT}")

    with open(DRAFTS, encoding="utf-8") as fh:
        drafts = {d["ticket_id"]: d for d in json.load(fh)["drafts"]}
    with open(LABELS, newline="", encoding="utf-8-sig") as fh:
        rdr = csv.DictReader(fh)
        if rdr.fieldnames != COLS:
            die(f"draft_labels.csv: columns {rdr.fieldnames}, expected {COLS}")
        rows = [{k: (v or "").strip().lower() if k != "notes" else (v or "")
                 for k, v in r.items()} for r in rdr]
    ids = [r["id"] for r in rows]
    if sorted(ids) != sorted(drafts):
        die("draft_labels.csv does not list exactly the 30 drafted ids")
    bad = [r["id"] for r in rows
           if r["grounded"] not in ("yes", "no")
           or r["appropriate_handoff"] not in ("yes", "no", "na")]
    if bad:
        die(f"{len(bad)} row(s) unlabelled or invalid: {bad}")

    n = len(rows)
    g = Counter(r["grounded"] for r in rows)
    h = Counter(r["appropriate_handoff"] for r in rows)
    routed_ok = {i for i, d in drafts.items()
                 if d["cascade_category"] == d["mapped_category"]}
    # appropriate_handoff: yes = context did not fit and the draft handed off;
    # no = context did not fit but the draft still gave a fix;
    # na = context fitted (a fix is expected).
    mismatch = h["yes"] + h["no"]
    out = [("annotation", "single annotator (AW)"), ("n", n)]

    def prop(name, k, d):
        out.append((f"{name}.k", k))
        out.append((f"{name}.n", d))
        if d:
            lo, hi = wilson(k, d)
            out.extend([(f"{name}.rate", repr(k / d)),
                        (f"{name}.wilson95_lo", repr(lo)),
                        (f"{name}.wilson95_hi", repr(hi))])

    prop("grounded.yes", g["yes"], n)
    prop("context_mismatch", mismatch, n)
    prop("handed_off_given_mismatch", h["yes"], mismatch)
    out += [("grounded.no.k", g["no"]),
            ("handoff.yes.k", h["yes"]), ("handoff.no.k", h["no"]),
            ("handoff.na.k", h["na"]),
            ("routed_correctly.k", len(routed_ok)),
            ("misrouted.k", n - len(routed_ok)),
            ("context_mismatch.among_misrouted.k",
             sum(r["appropriate_handoff"] != "na" for r in rows
                 if r["id"] not in routed_ok))]
    with open(OUT, "w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(["quantity", "value"])
        w.writerows(out)
    for k, v in out:
        print(f"  {k:<36} {v}")
    print(f"[write] {os.path.relpath(OUT, PROJECT_ROOT)}")


if __name__ == "__main__":
    main()
