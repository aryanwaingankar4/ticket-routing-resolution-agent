"""Route every English Zenodo ticket through the unchanged pipeline.

Paper revision (Oct 2026), Phase 2 step 3. Offline: loads BGE once, makes no
LLM call (generate_resolution=False). Routing does not depend on the category
mapping, so it runs over ALL English tickets of all seven source categories;
scoring against our seven teams is a separate step that reads the
author-confirmed data/external_zenodo_7648117/category_mapping.json.

Input : data/external_zenodo_7648117/english_tickets.csv
        (scripts/prepare_zenodo_tickets.py)
Output: data/external_zenodo_7648117/routing_results.csv
        refuses to overwrite without --force

    .\\venv\\Scripts\\python.exe scripts/run_realworld_routing.py
"""

from __future__ import annotations

import argparse
import csv
import os

from realworld_common import PROJECT_ROOT, die, route

BASE = os.path.join(PROJECT_ROOT, "data", "external_zenodo_7648117")
IN_CSV = os.path.join(BASE, "english_tickets.csv")
OUT_CSV = os.path.join(BASE, "routing_results.csv")


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--force", action="store_true")
    ap.add_argument("--limit", type=int, default=None,
                    help="route only the first N tickets (smoke test; "
                         "writes nothing)")
    args = ap.parse_args()
    if os.path.exists(OUT_CSV) and not args.force and args.limit is None:
        die(f"Output exists (use --force to overwrite):\n    {OUT_CSV}")

    with open(IN_CSV, newline="", encoding="utf-8") as fh:
        tickets = list(csv.DictReader(fh))
    if args.limit is not None:
        tickets = tickets[:args.limit]
    rows = route([(t["id"], t["text"], "") for t in tickets])
    for r, t in zip(rows, tickets):
        if str(r["ticket_id"]) != str(t["id"]):
            die("row order lost during routing")
        r["source_category"] = t["source_category"]
        r["duplicate_text"] = t["duplicate_text"]

    if args.limit is not None:
        for r in rows:
            print(r)
        print("[smoke] --limit given; nothing written")
        return
    with open(OUT_CSV, "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    print(f"[write] {os.path.relpath(OUT_CSV, PROJECT_ROOT)}")


if __name__ == "__main__":
    main()
