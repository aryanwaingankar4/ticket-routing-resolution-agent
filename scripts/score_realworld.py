"""Score the routed Zenodo tickets against the author-confirmed mapping.

Paper revision (Oct 2026), Phase 2 steps 3-4. Offline, model-free.

Inputs (data/external_zenodo_7648117/):
  routing_results.csv    scripts/run_realworld_routing.py
  category_mapping.json  must have "status": "confirmed" -- FATAL otherwise.
                         Source categories mapped to null are excluded.

Two readings, both reported:
  all       every English ticket whose source category maps to one of ours;
            this is the traffic as it arrived, duplicates included.
  distinct  the same subset with exact-duplicate texts collapsed to the first
            occurrence (by id), recomputed WITHIN the mapped subset. The source
            contains system-generated subjects such as "File Share Access -
            [TICKET ID] - [NAME]", so the two readings can differ a lot.

Outputs (refuses to overwrite without --force):
  realworld_summary.csv      quantity,value
  realworld_by_source.csv    per source category: n, predictions, escalation
  realworld_confusion.csv    mapped category x cascade prediction (all)

    .\\venv\\Scripts\\python.exe scripts/score_realworld.py
"""

from __future__ import annotations

import argparse
import csv
import json
import os
from collections import Counter

from realworld_common import PROJECT_ROOT, die, summarise

BASE = os.path.join(PROJECT_ROOT, "data", "external_zenodo_7648117")
ROUTED = os.path.join(BASE, "routing_results.csv")
TICKETS = os.path.join(BASE, "english_tickets.csv")
MAPPING = os.path.join(BASE, "category_mapping.json")
OUT_SUMMARY = os.path.join(BASE, "realworld_summary.csv")
OUT_BY_SOURCE = os.path.join(BASE, "realworld_by_source.csv")
OUT_CONFUSION = os.path.join(BASE, "realworld_confusion.csv")


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args()
    outs = (OUT_SUMMARY, OUT_BY_SOURCE, OUT_CONFUSION)
    if any(os.path.exists(p) for p in outs) and not args.force:
        die("Outputs exist (use --force to overwrite).")

    with open(MAPPING, encoding="utf-8") as fh:
        spec = json.load(fh)
    if spec.get("status") != "confirmed":
        die(f"category_mapping.json status is {spec.get('status')!r}; the "
            "authors must confirm the mapping before anything is scored")
    mapping = spec["mapping"]

    with open(ROUTED, newline="", encoding="utf-8") as fh:
        rows = list(csv.DictReader(fh))
    with open(TICKETS, newline="", encoding="utf-8") as fh:
        texts = {r["id"]: r["text"] for r in csv.DictReader(fh)}
    if set(texts) != {r["ticket_id"] for r in rows}:
        die("routing_results.csv and english_tickets.csv cover different ids")
    if len({r["config_fingerprint"] for r in rows}) != 1:
        die("routing rows carry more than one config fingerprint")
    unknown = {r["source_category"] for r in rows} - set(mapping)
    if unknown:
        die(f"source categories missing from the mapping: {sorted(unknown)}")
    for r in rows:
        r["escalated"] = r["escalated"] == "True"
        r["true_category"] = mapping[r["source_category"]]

    summary = [("config_fingerprint", rows[0]["config_fingerprint"]),
               ("english_total", len(rows))]
    excluded = Counter(r["source_category"] for r in rows
                       if r["true_category"] is None)
    mapped = sorted((r for r in rows if r["true_category"] is not None),
                    key=lambda r: int(r["ticket_id"]))
    summary.append(("excluded_total", sum(excluded.values())))
    summary += [(f"excluded.{k}", v) for k, v in sorted(excluded.items())]
    summary.append(("mapped_total", len(mapped)))

    seen, distinct = set(), []
    for r in mapped:
        t = texts[r["ticket_id"]]
        if t not in seen:
            seen.add(t)
            distinct.append(r)

    # Same text, different source label: label noise in the source.
    by_text = {}
    for r in mapped:
        by_text.setdefault(texts[r["ticket_id"]], set()).add(r["true_category"])
    summary.append(("distinct_texts_with_conflicting_labels",
                    sum(len(v) > 1 for v in by_text.values())))

    for name, sub in (("all", mapped), ("distinct", distinct)):
        summary += [(f"{name}.{k}", v) for k, v in summarise(sub, label=name)]

    with open(OUT_SUMMARY, "w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(["quantity", "value"])
        w.writerows(summary)

    cats = sorted({r["cascade_category"] for r in rows} |
                  {c for c in mapping.values() if c})
    with open(OUT_BY_SOURCE, "w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(["source_category", "mapped_to", "n", "escalated"]
                   + [f"pred.{c}" for c in cats])
        for src in sorted(mapping):
            sub = [r for r in rows if r["source_category"] == src]
            pred = Counter(r["cascade_category"] for r in sub)
            w.writerow([src, mapping[src] or "excluded", len(sub),
                        sum(r["escalated"] for r in sub)]
                       + [pred[c] for c in cats])

    gold = sorted({r["true_category"] for r in mapped})
    cm = Counter((r["true_category"], r["cascade_category"]) for r in mapped)
    with open(OUT_CONFUSION, "w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(["true \\ predicted"] + cats)
        for g in gold:
            w.writerow([g] + [cm[(g, c)] for c in cats])

    for k, v in summary:
        print(f"  {k:<52} {v}")
    for p in outs:
        print(f"[write] {os.path.relpath(p, PROJECT_ROOT)}")


if __name__ == "__main__":
    main()
