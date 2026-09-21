"""
Phase 7A -- acquire the external dataset for the external-validity check.

FEASIBILITY ONLY. This script downloads and filters; it measures nothing and
decides nothing. It never touches our dataset, artifacts, benchmarks or
thresholds, and everything it writes lands under data/external_tobibueck/.

THE DATASET, AND WHAT IT IS NOT
-------------------------------
`Tobi-Bueck/customer-support-tickets`, ~61.8k rows across three CSVs, licensed
**CC-BY-NC-4.0** (non-commercial -- fine for an academic project, but the term
and the attribution are recorded in PROVENANCE.json).

**It is INDEPENDENTLY GENERATED DATA, NOT REAL PRODUCTION DATA.** The dataset
card advertises a synthetic ticket generator from the same author. So this
tests whether our findings survive a DIFFERENT GENERATOR -- not whether they
survive reality. Any write-up must say so in those words. Treating it as real
production data would be a straightforward overclaim.

WHY NOT A REAL DATASET
----------------------
The Endava/Microsoft `all_tickets.csv` is genuinely real, and was rejected:
its text is anonymized/encrypted, so a pretrained encoder like BGE cannot read
it. Real but unreadable is worse here than synthetic but readable.

WHY NOT THE `datasets` LIBRARY
------------------------------
requirements.txt is pinned to "the versions this project's published results
were produced with", and adding a heavy dependency would change the environment
those results came from. huggingface_hub is already installed and is enough:
the repo holds plain CSVs, not a split layout. huggingface_hub is therefore
pinned as a DIRECT dependency, for the same reason requirements.txt already
gives for scipy -- a transitive pin is one nobody tested.

THE REVISION IS PINNED AND VERIFIED
-----------------------------------
An upstream edit must not silently change what was profiled, so the resolved
revision is checked against DATASET_REVISION and a mismatch is fatal. This is
the external-data form of this project's recurring bug class: data that is
wrong for its context, internally consistent, and therefore silently wrong.

Run from the project root (offline after the first fetch, no Gemini calls):
    python src/experiments/fetch_external_dataset.py
    python src/experiments/fetch_external_dataset.py --force
"""

from __future__ import annotations

import argparse
import datetime as _dt
import hashlib
import json
import os
import sys

import pandas as pd

THIS_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(THIS_DIR, os.pardir, os.pardir))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from src.agent.config import config_fingerprint                   # noqa: E402
from src.agent.logging_setup import ensure_utf8_console           # noqa: E402
from src.experiments.calibrate_conformal import _banner, _fatal   # noqa: E402

ensure_utf8_console()

DATASET_REPO = "Tobi-Bueck/customer-support-tickets"
# Pinned 2026-09-21. Verified against the Hub before Phase 7A was planned.
DATASET_REVISION = "ddf1c81a5475992c4fa6752bf1e8b4e31f07bbeb"
DATASET_LICENSE = "cc-by-nc-4.0"

DATASET_FILES = (
    "aa_dataset-tickets-multi-lang-5-2-50-version.csv",
    "dataset-tickets-multi-lang-4-20k.csv",
    "dataset-tickets-german_normalized_50_5_2.csv",
)

OUT_DIR = os.path.join(PROJECT_ROOT, "data", "external_tobibueck")
RAW_DIR = os.path.join(OUT_DIR, "raw")
ENGLISH_CSV = os.path.join(OUT_DIR, "english_subset.csv")
PROVENANCE_JSON = os.path.join(OUT_DIR, "PROVENANCE.json")

# The dataset is EN/DE. Values are normalised before matching because the
# column is free text across three files produced at different times.
ENGLISH_TOKENS = {"en", "eng", "english"}


def _norm_lang(value):
    return str(value).strip().lower() if value is not None else ""


def _sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _raw_line_count(path):
    """Second, independent derivation of the row count (rule 6).

    Counts newlines in binary rather than asking pandas, so a parser quirk
    cannot agree with itself. Embedded newlines inside quoted fields mean this
    is an UPPER BOUND on rows, not an equality check -- it is reported as such
    and only a count BELOW the parsed rows is treated as impossible.
    """
    n = 0
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            n += chunk.count(b"\n")
    return n


def download(force=False):
    from huggingface_hub import hf_hub_download

    os.makedirs(RAW_DIR, exist_ok=True)
    local = {}
    for name in DATASET_FILES:
        target = os.path.join(RAW_DIR, name)
        if os.path.isfile(target) and not force:
            print("  [cached] {n}".format(n=name))
            local[name] = target
            continue
        print("  [fetch ] {n}".format(n=name))
        path = hf_hub_download(
            repo_id=DATASET_REPO,
            filename=name,
            revision=DATASET_REVISION,
            repo_type="dataset",
        )
        # Copied out of the HF cache so the profile reads from a path this
        # project owns, and so PROVENANCE's hashes describe files that cannot
        # be evicted by a cache cleanup.
        with open(path, "rb") as src, open(target, "wb") as dst:
            for chunk in iter(lambda: src.read(1 << 20), b""):
                dst.write(chunk)
        local[name] = target
    return local


def verify_revision():
    """Fatal unless the Hub still serves the revision we pinned."""
    from huggingface_hub import HfApi

    try:
        info = HfApi().dataset_info(DATASET_REPO, revision=DATASET_REVISION)
    except Exception as exc:                      # noqa: BLE001
        _fatal(
            "Could not reach the Hugging Face Hub to verify the pinned "
            "revision.\n  {e}\n"
            "  7A pins {r} deliberately; running against an unverified "
            "revision is not acceptable.".format(e=repr(exc),
                                                 r=DATASET_REVISION))

    if info.sha != DATASET_REVISION:
        _fatal(
            "Revision mismatch.\n  pinned   {p}\n  resolved {r}\n"
            "  The upstream dataset moved. Do not profile this -- decide "
            "deliberately whether to re-pin.".format(
                p=DATASET_REVISION, r=info.sha))

    print("  [ok] revision {r} verified".format(r=DATASET_REVISION[:12]))
    return info


def load_and_filter(local):
    frames, per_file = [], {}

    for name, path in local.items():
        df = pd.read_csv(path, low_memory=False)
        df.columns = [c.strip() for c in df.columns]

        lang_col = next(
            (c for c in df.columns if c.strip().lower() == "language"), None)
        if lang_col is None:
            _fatal("No `language` column in {n}; columns are {c}".format(
                n=name, c=list(df.columns)))

        norm = df[lang_col].map(_norm_lang)
        english = df.loc[norm.isin(ENGLISH_TOKENS)].copy()

        # Rule 6: the filtered count must equal the value_counts total for the
        # English tokens, derived without the boolean mask.
        vc = norm.value_counts()
        expected = int(sum(int(vc.get(t, 0)) for t in ENGLISH_TOKENS))
        if len(english) != expected:
            _fatal(
                "English subset disagrees between two derivations for {n}: "
                "mask {a} vs value_counts {b}".format(
                    n=name, a=len(english), b=expected))

        lines = _raw_line_count(path)
        if lines < len(df):
            _fatal(
                "{n}: raw newline count {l} is BELOW the parsed row count {r}."
                " The file is not being read as written.".format(
                    n=name, l=lines, r=len(df)))

        english["_source_file"] = name
        frames.append(english)
        per_file[name] = {
            "rows_parsed": int(len(df)),
            "rows_english": int(len(english)),
            "raw_newline_count_upper_bound": int(lines),
            "language_values": {str(k): int(v) for k, v in vc.items()},
            "columns": [str(c) for c in df.columns],
            "sha256": _sha256(path),
            "bytes": int(os.path.getsize(path)),
        }
        print("  {n:52} {r:>6} rows, {e:>6} English".format(
            n=name[:52], r=len(df), e=len(english)))

    combined = pd.concat(frames, ignore_index=True, sort=False)
    return combined, per_file


def run(force=False):
    _banner("PHASE 7A - EXTERNAL DATASET ACQUISITION  (feasibility only)")
    print("Independently generated data, NOT real production data.")
    print("License: {l} (non-commercial).".format(l=DATASET_LICENSE))

    os.makedirs(OUT_DIR, exist_ok=True)

    _banner("STEP 1 - Verify the pinned revision")
    verify_revision()

    _banner("STEP 2 - Download")
    local = download(force=force)

    _banner("STEP 3 - Parse and filter to English")
    combined, per_file = load_and_filter(local)

    total_parsed = sum(v["rows_parsed"] for v in per_file.values())
    total_english = sum(v["rows_english"] for v in per_file.values())
    if len(combined) != total_english:
        _fatal("Concatenated English rows {a} != sum of per-file {b}".format(
            a=len(combined), b=total_english))

    combined.to_csv(ENGLISH_CSV, index=False)

    provenance = {
        "phase": "7A",
        "repo_id": DATASET_REPO,
        "revision": DATASET_REVISION,
        "license": DATASET_LICENSE,
        "attribution": "Tobi-Bueck/customer-support-tickets, CC-BY-NC-4.0",
        "is_real_production_data": False,
        "provenance_note": (
            "Independently generated data, not real production data. The "
            "dataset card advertises a synthetic ticket generator from the "
            "same author, so this tests whether findings survive a different "
            "generator, not whether they survive reality."
        ),
        "rejected_alternative": (
            "Endava/Microsoft all_tickets.csv -- real, but its text is "
            "anonymized/encrypted, so a pretrained encoder such as BGE cannot "
            "read it."
        ),
        "downloaded_utc": _dt.datetime.now(_dt.timezone.utc).isoformat(),
        "config_fingerprint": config_fingerprint(),
        "rows_parsed_total": int(total_parsed),
        "rows_english_total": int(total_english),
        "english_subset_csv": os.path.relpath(ENGLISH_CSV, PROJECT_ROOT),
        "per_file": per_file,
    }
    with open(PROVENANCE_JSON, "w", encoding="utf-8") as fh:
        json.dump(provenance, fh, indent=2, ensure_ascii=False)

    _banner("SUMMARY")
    print("  rows parsed (all languages) : {n}".format(n=total_parsed))
    print("  rows English                : {n}".format(n=total_english))
    print("  columns in combined subset  : {n}".format(n=combined.shape[1]))
    print("\n[write] {p}".format(p=ENGLISH_CSV))
    print("[write] {p}".format(p=PROVENANCE_JSON))
    return combined, provenance


def main():
    parser = argparse.ArgumentParser(
        description="Phase 7A -- fetch the external dataset (feasibility "
                    "only, no Gemini calls).")
    parser.add_argument("--force", action="store_true",
                        help="re-download even if the raw files are cached")
    args = parser.parse_args()
    run(force=args.force)


if __name__ == "__main__":
    main()
