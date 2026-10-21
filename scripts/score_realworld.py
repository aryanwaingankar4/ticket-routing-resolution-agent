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

The gate readings (escalation rate, Tier-1 share) need no label, so they are
also reported over the WHOLE English set ("english_all", "english_distinct"),
excluded categories included.

Outputs (refuses to overwrite without --force):
  realworld_summary.csv      quantity,value
  realworld_by_source.csv    per source category: n, predictions, escalation
  realworld_confusion.csv    mapped category x cascade prediction (all)

--variant NAME applies category_mapping.json's sensitivity[NAME].overrides on
top of the confirmed mapping and writes the same three files with the suffix
_sens_NAME, so a sensitivity run never touches the primary outputs.

    .\\venv\\Scripts\\python.exe scripts/score_realworld.py
    .\\venv\\Scripts\\python.exe scripts/score_realworld.py --variant fileservice_storage
"""

from __future__ import annotations

import argparse
import csv
import json
import os
from collections import Counter

from realworld_common import PROJECT_ROOT, die, summarise, wilson

BASE = os.path.join(PROJECT_ROOT, "data", "external_zenodo_7648117")
ROUTED = os.path.join(BASE, "routing_results.csv")
TICKETS = os.path.join(BASE, "english_tickets.csv")
MAPPING = os.path.join(BASE, "category_mapping.json")


def dedup(rows, texts):
    """First occurrence (by id) of each exact text."""
    seen, out = set(), []
    for r in sorted(rows, key=lambda r: int(r["ticket_id"])):
        t = texts[r["ticket_id"]]
        if t not in seen:
            seen.add(t)
            out.append(r)
    return out


def gate_rates(rows):
    """Label-free readings: escalation at the 0.67 gate and Tier-1 share."""
    n, out = len(rows), [("n", len(rows))]
    for name, k in (("escalated", sum(r["escalated"] for r in rows)),
                    ("tier1_kept", sum(int(r["tier"]) == 1 for r in rows))):
        lo, hi = wilson(k, n)
        out += [(f"{name}.k", k), (f"{name}.rate", repr(k / n)),
                (f"{name}.wilson95_lo", repr(lo)),
                (f"{name}.wilson95_hi", repr(hi))]
    return out


def proceeded(rows):
    """Of the tickets the RAG gate lets through to the drafter, how many
    were routed to the right team."""
    sub = [r for r in rows if not r["escalated"]]
    k = sum(r["cascade_category"] == r["true_category"] for r in sub)
    lo, hi = wilson(k, len(sub))
    return [("proceeded.n", len(sub)), ("proceeded_correct.k", k),
            ("proceeded_correct.rate", repr(k / len(sub))),
            ("proceeded_correct.wilson95_lo", repr(lo)),
            ("proceeded_correct.wilson95_hi", repr(hi))]


def kept_and_paired(rows):
    """Accuracy on the tickets Tier-1 answers, and cascade vs Tier-2-only by
    the committed exact McNemar (src/experiments/compare_cascade_vs_tier2.py)."""
    from scipy.stats import binomtest
    from src.experiments.compare_cascade_vs_tier2 import mcnemar_from_pairs
    kept = [r for r in rows if int(r["tier"]) == 1]
    out = [("tier1_kept_correct.k",
            sum(r["cascade_category"] == r["true_category"] for r in kept)),
           ("tier1_kept_tier2_would_be_correct.k",
            sum(r["tier2_only_category"] == r["true_category"] for r in kept))]
    pairs = [({"correct": r["cascade_category"] == r["true_category"]},
              {"correct": r["tier2_only_category"] == r["true_category"]})
             for r in rows]
    m = mcnemar_from_pairs(pairs, binomtest)
    return out + [("mcnemar_cascade_vs_tier2.b_cascade_only_right", m["b"]),
                  ("mcnemar_cascade_vs_tier2.c_tier2_only_right", m["c"]),
                  ("mcnemar_cascade_vs_tier2.p_exact", repr(m["p_value"]))]


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--force", action="store_true")
    ap.add_argument("--variant", default=None,
                    help="a sensitivity mapping named in category_mapping.json")
    args = ap.parse_args()
    suffix = f"_sens_{args.variant}" if args.variant else ""
    outs = tuple(os.path.join(BASE, f"realworld_{n}{suffix}.csv")
                 for n in ("summary", "by_source", "confusion"))
    OUT_SUMMARY, OUT_BY_SOURCE, OUT_CONFUSION = outs
    if any(os.path.exists(p) for p in outs) and not args.force:
        die("Outputs exist (use --force to overwrite).")

    with open(MAPPING, encoding="utf-8") as fh:
        spec = json.load(fh)
    if spec.get("status") != "confirmed":
        die(f"category_mapping.json status is {spec.get('status')!r}; the "
            "authors must confirm the mapping before anything is scored")
    mapping = dict(spec["mapping"])
    if args.variant:
        sens = spec.get("sensitivity", {})
        if args.variant not in sens:
            die(f"no sensitivity mapping {args.variant!r}; have {sorted(sens)}")
        over = sens[args.variant]["overrides"]
        if set(over) - set(mapping):
            die(f"overrides name unknown categories: "
                f"{sorted(set(over) - set(mapping))}")
        mapping.update(over)

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
               ("mapping", f"sensitivity:{args.variant}" if args.variant
                else "confirmed primary"),
               ("english_total", len(rows))]
    summary += [(f"english_all.{k}", v) for k, v in gate_rates(rows)]
    summary += [(f"english_distinct.{k}", v)
                for k, v in gate_rates(dedup(rows, texts))]
    excluded = Counter(r["source_category"] for r in rows
                       if r["true_category"] is None)
    mapped = sorted((r for r in rows if r["true_category"] is not None),
                    key=lambda r: int(r["ticket_id"]))
    summary.append(("excluded_total", sum(excluded.values())))
    summary += [(f"excluded.{k}", v) for k, v in sorted(excluded.items())]
    summary.append(("mapped_total", len(mapped)))

    distinct = dedup(mapped, texts)

    # Same text, different source label: label noise in the source.
    by_text = {}
    for r in mapped:
        by_text.setdefault(texts[r["ticket_id"]], set()).add(r["true_category"])
    summary.append(("distinct_texts_with_conflicting_labels",
                    sum(len(v) > 1 for v in by_text.values())))

    for name, sub in (("all", mapped), ("distinct", distinct)):
        summary += [(f"{name}.{k}", v) for k, v in summarise(sub, label=name)]
        summary += [(f"{name}.{k}", v) for k, v in proceeded(sub)]
        summary += [(f"{name}.{k}", v) for k, v in kept_and_paired(sub)]

    with open(OUT_SUMMARY, "w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(["quantity", "value"])
        w.writerows(summary)

    cats = sorted({r["cascade_category"] for r in rows} |
                  {c for c in mapping.values() if c})
    with open(OUT_BY_SOURCE, "w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(["reading", "source_category", "mapped_to", "n",
                    "escalated"] + [f"pred.{c}" for c in cats])
        # "distinct" dedups within each source category, so an excluded
        # category's count matches what a mapped one would have.
        for reading in ("all", "distinct"):
            for src in sorted(mapping):
                sub = [r for r in rows if r["source_category"] == src]
                if reading == "distinct":
                    sub = dedup(sub, texts)
                pred = Counter(r["cascade_category"] for r in sub)
                w.writerow([reading, src, mapping[src] or "excluded", len(sub),
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
