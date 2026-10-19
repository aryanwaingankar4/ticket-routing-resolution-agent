"""Filter the Zenodo ticket set to English and draw the mapping-review samples.

Paper revision (Oct 2026), Phase 2. Offline, no model, no API call. Reads
only data/external_zenodo_7648117/raw/ (scripts/fetch_zenodo_tickets.py).

Language identification: langid.py 1.1.6 (Lui & Baldwin, ACL 2012),
UNRESTRICTED over its 97 languages, deterministic. A ticket is English when
langid's top language is 'en'. The dataset's anonymisation tags ([NAME],
[TICKET ID], [COMPANY], ...) are English words, so they are removed from the
string used for DETECTION only; the text the pipeline later sees is untouched.

The 7 source categories are kept here, all of them. The mapping onto our 7
teams is a separate, author-confirmed file (category_mapping.json); this
script writes it only as status "proposed".

Outputs (data/external_zenodo_7648117/, refuses to overwrite without --force):
  english_tickets.csv        id, text, source_category, lang, duplicate_text
  mapping_review_samples.csv 5 random English tickets (seed 42) per category
                             under review, for the authors to read
  category_mapping.json      PROPOSED mapping, status "proposed"

    .\\venv\\Scripts\\python.exe scripts/prepare_zenodo_tickets.py
"""

from __future__ import annotations

import argparse
import json
import os
import re

PROJECT_ROOT = os.path.abspath(
    os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
BASE = os.path.join(PROJECT_ROOT, "data", "external_zenodo_7648117")
RAW = os.path.join(BASE, "raw")
OUT_TICKETS = os.path.join(BASE, "english_tickets.csv")
OUT_SAMPLES = os.path.join(BASE, "mapping_review_samples.csv")
OUT_MAPPING = os.path.join(BASE, "category_mapping.json")

SEED = 42
N_SAMPLES = 5
REVIEW = ["Computer-Services", "Fileservice", "Support general", "EOL"]
TAG = re.compile(r"\[[A-Z ]+\]")

# Proposed only. "Computer-Services" is under review; "Support general" and
# "EOL" have no counterpart among our seven teams.
PROPOSED = {
    "Fileservice": "Storage",
    "Active Directory": "Access Management",
    "Software": "Application",
    "O365": "Application",
    "Computer-Services": "Infrastructure",
    "Support general": None,
    "EOL": None,
}


def die(msg):
    print(f"\n[FATAL] {msg}")
    raise SystemExit(1)


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args()
    outs = (OUT_TICKETS, OUT_SAMPLES, OUT_MAPPING)
    if any(os.path.exists(p) for p in outs) and not args.force:
        die("Outputs exist (use --force to overwrite).")

    import langid
    import pandas as pd

    x = pd.concat([pd.read_csv(os.path.join(RAW, f"X_{s}.csv"))
                   for s in ("train", "test")])
    y = pd.concat([pd.read_csv(os.path.join(RAW, f"y_{s}.csv"))
                   for s in ("train", "test")])
    d = x.merge(y, on="id", how="inner", validate="one_to_one")
    if not (len(d) == len(x) == len(y)):
        die(f"text/label join lost rows: {len(x)} texts, {len(y)} labels, "
            f"{len(d)} joined")
    d = d.rename(columns={"category_truth": "source_category"})
    d["text"] = d["text"].astype(str)
    if set(d["source_category"]) != set(PROPOSED):
        die(f"unexpected categories: {sorted(set(d['source_category']))}")

    langid.set_languages(None)  # unrestricted, all 97 languages
    d["lang"] = [langid.classify(TAG.sub(" ", t))[0] for t in d["text"]]
    en = d[d["lang"] == "en"].copy().sort_values("id")
    en["duplicate_text"] = en["text"].duplicated(keep="first")

    os.makedirs(BASE, exist_ok=True)
    en[["id", "text", "source_category", "lang", "duplicate_text"]].to_csv(
        OUT_TICKETS, index=False, encoding="utf-8")

    samples = []
    for cat in REVIEW:
        pool = en[en["source_category"] == cat]
        pick = pool.sample(n=min(N_SAMPLES, len(pool)), random_state=SEED)
        samples.append(pick[["id", "source_category", "text"]])
    pd.concat(samples).to_csv(OUT_SAMPLES, index=False, encoding="utf-8")

    with open(OUT_MAPPING, "w", encoding="utf-8") as fh:
        json.dump({"status": "proposed",
                   "note": "Awaiting author confirmation after reading "
                           "mapping_review_samples.csv. null = excluded.",
                   "mapping": PROPOSED}, fh, indent=2)

    print(f"rows {len(d)}; English (langid top = 'en') {len(en)} "
          f"({len(en) / len(d):.1%})")
    tab = (en.groupby("source_category")
             .agg(n=("id", "size"), duplicate_texts=("duplicate_text", "sum")))
    tab["all_languages"] = d.groupby("source_category").size()
    print(tab.to_string())
    for p in outs:
        print(f"[write] {os.path.relpath(p, PROJECT_ROOT)}")


if __name__ == "__main__":
    main()
