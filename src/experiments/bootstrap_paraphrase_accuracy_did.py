# src/experiments/bootstrap_paraphrase_accuracy_did.py
"""
Phase 7C, POST-HOC: a paired bootstrap on the ACCURACY difference-in-differences.

THIS IS POST-HOC AND SAYS SO IN EVERY OUTPUT IT WRITES. 7C's pre-registered
primary was a COVERAGE-gap difference, and that arm is BLOCKED by the
pre-registered >= 0.95 degeneracy rule. Nothing here unblocks it, replaces it,
or supplies a verdict in its place.

WHAT IT MEASURES, AND WHY IT IS WORTH MEASURING. Scoring 7C's paired set turned
up an effect the coverage arm never showed: the paraphrase shift cost Tier-1
about 10.5 accuracy points against Tier-2's 2.5 -- roughly four times harder on
the lexical model, which is Finding 1's SURFACE-VOCABULARY mechanism appearing
directly. 7B's version shift produced no such contrast, which is what makes it
interesting. But "10.5 versus 2.5" is two point estimates on 286 tickets, and
this project does not write down a difference without an interval around it.

THE STATISTIC. accuracy is measured on the SAME 286 tickets in four
combinations, so everything is paired and one resample index is shared by all
four arrays:

    DiD = (acc_para_t1 - acc_para_t2) - (acc_orig_t1 - acc_orig_t2)

A NEGATIVE DiD means the paraphrase shift cost Tier-1 more than Tier-2, over
and above whatever difference the two tiers already had on the originals.
10,000 draws, seed 42 -- the same bootstrap configuration as 6A and the 7C
coverage arm, so the intervals are comparable.

THE RESULT IS REPORTED WHATEVER THE INTERVAL SHOWS. An interval spanning zero
is written down as plainly as one that excludes it: a post-hoc observation that
does not survive its own interval is exactly the kind of thing that should not
reach a paper unqualified.

Reuses 7C's own fitted models and paired set rather than refitting, and the
refit is proved identical to 7B's before anything is scored.

Run from the project root (offline, no Gemini calls, ~1 min):
    python src/experiments/bootstrap_paraphrase_accuracy_did.py
    python src/experiments/bootstrap_paraphrase_accuracy_did.py --force
"""

from __future__ import annotations

import argparse
import io
import json
import os
import sys

import numpy as np
import pandas as pd

THIS_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(THIS_DIR, os.pardir, os.pardir))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from src.agent.config import config_fingerprint                   # noqa: E402
from src.agent.logging_setup import ensure_utf8_console            # noqa: E402
from src.experiments.calibrate_conformal import (                  # noqa: E402
    _banner,
    _fatal,
    tier1_probabilities,
)
from src.experiments.fetch_external_dataset import (               # noqa: E402
    ENGLISH_CSV,
    OUT_DIR,
)
from src.experiments.profile_external_dataset import EMB_NPY       # noqa: E402
from src.experiments.run_external_conformal_shift import (         # noqa: E402
    PROBS_NPZ,
    SPLITS_JSON,
    build_texts,
)
from src.experiments.paraphrase_external_tickets import (          # noqa: E402
    PARAPHRASE_JSON,
)
from src.experiments.run_paraphrase_shift_conformal import (       # noqa: E402
    BOOTSTRAP_N,
    SEED,
    paired_bootstrap_did,
    paired_bootstrap_difference,
    refit_and_verify,
)

ensure_utf8_console()

OUT_JSON = os.path.join(OUT_DIR, "external_paraphrase_accuracy_did.json")


def run(force=False):
    _banner("PHASE 7C POST-HOC - accuracy difference-in-differences")
    print("POST-HOC. The pre-registered primary was a COVERAGE-gap difference")
    print("and that arm stays BLOCKED. Nothing here supplies a verdict.")

    if os.path.isfile(OUT_JSON) and not force:
        _fatal("Refusing to overwrite an existing result:\n  {p}\n"
               "Pass --force if you mean to replace it.".format(p=OUT_JSON))

    df = pd.read_csv(ENGLISH_CSV, low_memory=False)
    emb = np.load(EMB_NPY)
    manifest = json.load(io.open(SPLITS_JSON, encoding="utf-8"))
    pset = json.load(io.open(PARAPHRASE_JSON, encoding="utf-8"))
    blob = np.load(PROBS_NPZ, allow_pickle=False)

    _banner("STEP 1 - 7C's models, proved identical to 7B's")
    t1_vec, t1_clf, t2_clf, classes1, classes2 = refit_and_verify(
        df, emb, manifest, blob)

    pairs = pset["pairs"]
    row_idx = np.asarray([p["row_index"] for p in pairs], dtype=np.int64)
    labels = np.asarray([p["queue"] for p in pairs])
    source_texts = [p["source"] for p in pairs]
    para_texts = [p["paraphrase"] for p in pairs]

    corpus_texts = build_texts(df)
    if any(corpus_texts[i] != s for i, s in zip(row_idx, source_texts)):
        _fatal("Source text disagrees with the corpus.")

    from src.agent.artifacts import load_artifacts
    artifacts = load_artifacts(require_gemini=False)
    para_emb = np.asarray(artifacts.embedder.encode(
        para_texts, convert_to_numpy=True, batch_size=32,
        show_progress_bar=False), dtype=np.float32)

    _banner("STEP 2 - Per-ticket correctness, four arms")
    c1 = np.asarray(classes1)
    c2 = np.asarray(classes2)

    o_t1 = c1[np.argmax(tier1_probabilities(
        t1_vec, t1_clf, source_texts)[0], axis=1)] == labels
    p_t1 = c1[np.argmax(tier1_probabilities(
        t1_vec, t1_clf, para_texts)[0], axis=1)] == labels
    o_t2 = c2[np.argmax(t2_clf.predict_proba(emb[row_idx]), axis=1)] == labels
    p_t2 = c2[np.argmax(t2_clf.predict_proba(para_emb), axis=1)] == labels

    acc = {
        "original_tier1": float(o_t1.mean()),
        "original_tier2": float(o_t2.mean()),
        "paraphrased_tier1": float(p_t1.mean()),
        "paraphrased_tier2": float(p_t2.mean()),
    }
    for name, value in acc.items():
        print("  {n:20} {v:.4f}".format(n=name, v=value))

    # Rule 6: the headline drops, recomputed from the counts rather than the
    # rates, so a formatting error cannot become the reported effect.
    n = len(labels)
    drop_t1 = (int(o_t1.sum()) - int(p_t1.sum())) / n
    drop_t2 = (int(o_t2.sum()) - int(p_t2.sum())) / n
    print("  Tier-1 drop {a:+.4f} ({x} -> {y} of {n})".format(
        a=-drop_t1, x=int(o_t1.sum()), y=int(p_t1.sum()), n=n))
    print("  Tier-2 drop {a:+.4f} ({x} -> {y} of {n})".format(
        a=-drop_t2, x=int(o_t2.sum()), y=int(p_t2.sum()), n=n))

    _banner("STEP 3 - Paired bootstrap ({b} draws, seed {s})".format(
        b=BOOTSTRAP_N, s=SEED))
    did_mean, did_lo, did_hi = paired_bootstrap_did(p_t1, p_t2, o_t1, o_t2)
    para_mean, para_lo, para_hi = paired_bootstrap_difference(p_t1, p_t2)
    orig_mean, orig_lo, orig_hi = paired_bootstrap_difference(o_t1, o_t2)

    did_point = float((p_t1.mean() - p_t2.mean())
                      - (o_t1.mean() - o_t2.mean()))
    signal = (did_lo > 0 or did_hi < 0)

    print("  paraphrased  T1 - T2 accuracy = {p:+.4f}  [{lo:+.4f}, {hi:+.4f}]"
          .format(p=float(p_t1.mean() - p_t2.mean()), lo=para_lo, hi=para_hi))
    print("  original     T1 - T2 accuracy = {p:+.4f}  [{lo:+.4f}, {hi:+.4f}]"
          .format(p=float(o_t1.mean() - o_t2.mean()), lo=orig_lo, hi=orig_hi))
    print("  DiD = {p:+.4f}  bootstrap {m:+.4f}  [{lo:+.4f}, {hi:+.4f}]"
          .format(p=did_point, m=did_mean, lo=did_lo, hi=did_hi))
    print("  interval {e} zero -> {v}".format(
        e=("EXCLUDES" if signal else "includes"),
        v=("the shift cost Tier-1 more than Tier-2"
           if signal and did_point < 0 else
           "no signal on the difference-in-differences" if not signal else
           "signal in the OPPOSITE direction")))

    print("\n  POST-HOC. This does not replace the blocked primary, and the")
    print("  external labels remain generator-assigned and unaudited.")

    blob_out = {
        "phase": "7C",
        "post_hoc": True,
        "supersedes_nothing": True,
        "pre_registered_primary_status": "blocked_auc_ge_0.95",
        "statistic": "accuracy difference-in-differences, (para T1 - para T2) "
                     "- (orig T1 - orig T2)",
        "config_fingerprint": config_fingerprint(),
        "seed": SEED,
        "bootstrap_draws": BOOTSTRAP_N,
        "n_pairs": int(n),
        "accuracy": acc,
        "tier1_accuracy_change": -drop_t1,
        "tier2_accuracy_change": -drop_t2,
        "did_point": did_point,
        "did_bootstrap_mean": did_mean,
        "did_ci_lo": did_lo,
        "did_ci_hi": did_hi,
        "did_signal": bool(signal),
        "paraphrased_t1_minus_t2": float(p_t1.mean() - p_t2.mean()),
        "paraphrased_ci": [para_lo, para_hi],
        "original_t1_minus_t2": float(o_t1.mean() - o_t2.mean()),
        "original_ci": [orig_lo, orig_hi],
        "limitation": ("External queue labels are generator-assigned and "
                       "unaudited; this is a post-hoc exploratory statistic "
                       "and does not substitute for the blocked primary."),
    }
    with io.open(OUT_JSON, "w", encoding="utf-8") as fh:
        json.dump(blob_out, fh, indent=2)
    print("\n[write] {p}".format(p=os.path.relpath(OUT_JSON, PROJECT_ROOT)))
    return blob_out


def parse_args(argv):
    parser = argparse.ArgumentParser(
        description="Phase 7C post-hoc accuracy DiD bootstrap.")
    parser.add_argument("--force", action="store_true",
                        help="overwrite the existing result")
    return parser.parse_args(argv)


def main(argv=None):
    run(force=parse_args(argv).force)
    return 0


if __name__ == "__main__":
    sys.exit(main())
