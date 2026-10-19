"""Per-ticket top-1 similarities behind the 0.67 retrieval sweep.

Paper revision (Oct 2026), Table IX / Section VII-B (E2). Offline: no Gemini.
Loads BGE once through artifacts.load_artifacts() -- the production loader
with all three guards -- and scores through src.agent.retriever.retrieve(),
the production retrieval path. Nothing is reimplemented.

data/rag_similarity_calibration_combined.csv records only counts per
threshold. This script recomputes the top-1 similarity of every
out-of-domain (45) and in-domain (175) calibration ticket and is FATAL unless:

  * exactly 6 of the 45 out-of-domain tickets reach >= 0.67 (the paper's
    13.3%), read from the committed CSV, not typed; and
  * at EVERY threshold in the committed CSV, the reproduced out-of-domain
    pass count equals fp_ood and the in-domain pass count equals
    n_in_domain_proceed.

Writes only under data/paper_revision_oct/ and refuses to overwrite without
--force.

    .\\venv\\Scripts\\python.exe scripts/reproduce_ood_similarities.py
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import sys

PROJECT_ROOT = os.path.abspath(
    os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

DATA_DIR = os.path.join(PROJECT_ROOT, "data")
OUT_DIR = os.path.join(DATA_DIR, "paper_revision_oct")
COMBINED_CSV = os.path.join(DATA_DIR, "rag_similarity_calibration_combined.csv")
OOD_JSON = os.path.join(DATA_DIR, "ood_calibration_tickets.json")
CALIB_JSON = os.path.join(DATA_DIR, "calibration_tickets_paraphrased.json")
OUT_CSV = os.path.join(OUT_DIR, "rag_calibration_top1_similarity.csv")


def die(msg):
    print(f"\n[FATAL] {msg}")
    raise SystemExit(1)


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--force", action="store_true",
                    help="overwrite existing output")
    args = ap.parse_args()
    if os.path.exists(OUT_CSV) and not args.force:
        die(f"Output exists (use --force to overwrite):\n    {OUT_CSV}")

    from src.agent.artifacts import load_artifacts
    from src.agent.config import settings
    from src.agent.retriever import retrieve

    gate = settings.rag.similarity_threshold
    with open(COMBINED_CSV, newline="", encoding="utf-8") as fh:
        sweep = list(csv.DictReader(fh))
    with open(OOD_JSON, encoding="utf-8") as fh:
        ood = json.load(fh)
    with open(CALIB_JSON, encoding="utf-8") as fh:
        calib = json.load(fh)

    artifacts = load_artifacts(require_gemini=False)

    rows = []
    for arm, tickets in (("ood", ood), ("in_domain", calib)):
        for t in tickets:
            # Both sets carry a single 'text' field, which is exactly what
            # calibrate_rag_similarity_threshold.py embedded.
            text = str(t.get("text", "") or "").strip()
            top = retrieve(text, artifacts).retrieved[0]
            rows.append({
                "arm": arm, "id": t["id"], "text": text,
                "top1_similarity": float(top.similarity),
                "top1_id": top.id, "top1_category": top.category,
                "top1_title": top.title,
            })
        print(f"[score] {arm}: {len(tickets)} tickets")

    ood_sims = [r["top1_similarity"] for r in rows if r["arm"] == "ood"]
    ind_sims = [r["top1_similarity"] for r in rows if r["arm"] == "in_domain"]

    for s in sweep:
        thr = float(s["threshold"])
        n_ood = sum(x >= thr for x in ood_sims)
        n_ind = sum(x >= thr for x in ind_sims)
        if n_ood != int(s["fp_ood"]) or n_ind != int(s["n_in_domain_proceed"]):
            die(f"threshold {thr}: reproduced ood={n_ood} in={n_ind}, "
                f"committed fp_ood={s['fp_ood']} "
                f"n_in_domain_proceed={s['n_in_domain_proceed']}")
    print(f"  [ok] all {len(sweep)} thresholds reproduce fp_ood and "
          f"n_in_domain_proceed exactly")

    at_gate = [s for s in sweep if float(s["threshold"]) == gate]
    if len(at_gate) != 1:
        die(f"live gate {gate} is not a row of {COMBINED_CSV}")
    leaked = sum(x >= gate for x in ood_sims)
    if leaked != int(at_gate[0]["fp_ood"]) or leaked != 6:
        die(f"{leaked} of {len(ood_sims)} out-of-domain tickets >= {gate}; "
            f"the paper's 13.3% requires 6")
    print(f"  [ok] {leaked}/{len(ood_sims)} out-of-domain tickets >= {gate}")

    os.makedirs(OUT_DIR, exist_ok=True)
    rows.sort(key=lambda r: (r["arm"], -r["top1_similarity"]))
    with open(OUT_CSV, "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0]) + ["passes_gate"])
        w.writeheader()
        for r in rows:
            w.writerow({**r, "top1_similarity": repr(r["top1_similarity"]),
                        "passes_gate": r["top1_similarity"] >= gate})
    print(f"\n[write] {os.path.relpath(OUT_CSV, PROJECT_ROOT)}")


if __name__ == "__main__":
    main()
