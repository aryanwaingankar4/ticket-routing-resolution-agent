"""
Build the drift-detection reference distribution (Phase 4A).

Offline. No Gemini quota. Loads BGE + FAISS through artifacts.load_artifacts.

WHY THIS EXISTS
---------------
The Phase 4 plan named "the in-domain calibration scores in
data/conformal_calibration_bge-base-en-v1-5.json" as drift's reference. That
file stores classification fits only; calibrate_conformal.py computes the 175
retrieval similarities in its STEP 5 and discards them. Re-running that script
to add a field would rewrite published Phase 1 results as a side effect, so
the reference is its own artifact, written here.

WHAT IT RECORDS
---------------
Signal A: top-1 NON-SELF similarity per calibration ticket. The calibration
    tickets are paraphrases of indexed rows and 5.7% retrieve their own
    source; a live ticket never can, so non-self is the exchangeable analogue.
    The self-inclusive score is stored beside it for audit.
Signal B: the production pipeline's own escalation, tier and category
    decisions on the same 175 tickets (generate_resolution=False, no logging).

TWO INDEPENDENT-DERIVATION CHECKS, BOTH FATAL
---------------------------------------------
1. The OOD and adversarial detection results recomputed from these scores
   must reproduce data/conformal_novelty_results.csv exactly -- variant-level
   and seed-level OOD detection rates and the adversarial flag count -- for
   both the self-inclusive and non-self scores at every alpha. A mismatch
   means this is not the reference the RAG gate's conformal result was
   measured against.

   Deliberately NOT the published in-domain false-escalation rate. Scoring a
   set against itself yields a rate fixed by n and ties alone, so ANY 175
   distinct numbers reproduce it; a check that cannot fail is not a check.
   The first draft of this script used it, and it passed identically for
   both score sets -- which is how that was noticed. Detection rates depend
   on where the reference scores sit relative to other sets' scores, so they
   can fail.
2. For every ticket, the pipeline's top_similarity must equal the
   self-inclusive score measured directly. A mismatch means the pipeline and
   this script embed different text, and live similarities would not be
   comparable to the reference.

SIGNAL B's REFERENCE (Phase 4B): --source deployment
----------------------------------------------------
The in-domain reference escalates 0/175, so an escalation-rate test against
it is degenerate. `--source deployment` builds a second reference from the
deployment-distribution calibration set, which escalates. Its fatal check is
different, because the Phase 1 novelty reproduction is specific to the
in-domain set:

3. The pipeline's escalation count (decision.escalated) must equal a count
   derived WITHOUT the pipeline's decision: directly measured top-1
   similarity below settings.rag.similarity_threshold. Escalation counted
   two ways -- the lesson of the fifth recurring-bug instance, where an
   eligibility count read the wrong status enum and returned a plausible
   wrong number.

Run from the project root:
    python src/experiments/build_drift_reference.py
    python src/experiments/build_drift_reference.py --force   # overwrite
    python src/experiments/build_drift_reference.py --source deployment
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import sys
from collections import Counter

_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(_THIS_DIR, os.pardir, os.pardir))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

import numpy as np                                            # noqa: E402

from src.agent import conformal as cp                         # noqa: E402
from src.agent.config import config_fingerprint, settings     # noqa: E402
from src.agent.errors import AgentError                       # noqa: E402
from src.agent.logging_setup import ensure_utf8_console        # noqa: E402

ensure_utf8_console()

DATA_DIR = os.path.join(PROJECT_ROOT, "data")
CALIBRATION_JSON = os.path.join(DATA_DIR,
                                "calibration_tickets_paraphrased.json")
DEPLOYMENT_JSON = os.path.join(DATA_DIR,
                               "deployment_calibration_tickets.json")
OOD_JSON = os.path.join(DATA_DIR, "ood_calibration_tickets.json")
ADVERSARIAL_JSON = os.path.join(DATA_DIR,
                                "adversarial_escalation_tickets.json")
NOVELTY_CSV = os.path.join(DATA_DIR, "conformal_novelty_results.csv")

# Exact equality is expected -- same inputs, same float64 arithmetic -- so
# the tolerances only absorb CSV round-tripping.
RATE_TOLERANCE = 1e-12
SIMILARITY_TOLERANCE = 1e-6

# Published columns whose value depends on where the reference scores sit
# relative to other sets' scores, and which can therefore fail.
DISCRIMINATING_COLUMNS = (
    "ood_detection_rate_variants",
    "ood_detection_rate_seeds",
    "adversarial_flagged",
)


def _banner(text):
    bar = "=" * 74
    print("\n" + bar)
    print(text)
    print(bar)


def _fatal(message):
    print("\n" + "!" * 74)
    print("FATAL: " + message)
    print("!" * 74)
    sys.exit(1)


def _read_json(path, what):
    if not os.path.isfile(path):
        _fatal(f"{what} not found:\n    {path}")
    with open(path, encoding="utf-8") as fh:
        return json.load(fh)


def _published_novelty_rows():
    """{(contamination, alpha): row} from the published Phase 1 results."""
    if not os.path.isfile(NOVELTY_CSV):
        _fatal(f"Published novelty results not found:\n    {NOVELTY_CSV}\n"
               "  They are the independent check this reference must pass.\n"
               "  Produce them with:  python -m "
               "src.experiments.calibrate_conformal")
    with open(NOVELTY_CSV, newline="", encoding="utf-8") as fh:
        return {(row["contamination"], float(row["alpha"])): row
                for row in csv.DictReader(fh)}


def _check_against_published(with_self, non_self, ood_sims, ood_seeds,
                             adv_sims):
    published = _published_novelty_rows()
    n_seeds = len(set(ood_seeds))
    checked, failures = {}, []

    for contamination, scores in (("contaminated", with_self),
                                  ("clean", non_self)):
        cal = [-s for s in scores]
        ood_p = cp.conformal_p_values(cal, [-s for s in ood_sims])
        adv_p = cp.conformal_p_values(cal, [-s for s in adv_sims])
        rows = sorted((alpha, row) for (label, alpha), row in published.items()
                      if label == contamination)
        if not rows:
            _fatal(f"No published '{contamination}' rows in {NOVELTY_CSV}.")

        for alpha, row in rows:
            recomputed = {
                "ood_detection_rate_variants": float(np.mean(ood_p <= alpha)),
                "ood_detection_rate_seeds": len(
                    {s for s, p in zip(ood_seeds, ood_p) if p <= alpha}
                ) / n_seeds,
                "adversarial_flagged": float(np.sum(adv_p <= alpha)),
            }
            for column in DISCRIMINATING_COLUMNS:
                got, want = recomputed[column], float(row[column])
                ok = math.isclose(got, want, rel_tol=0,
                                  abs_tol=RATE_TOLERANCE)
                checked[f"{contamination}@{alpha}:{column}"] = got
                print(f"  {contamination:<13} a={alpha:<5} {column:<28} "
                      f"recomputed {got:<9.6g} published {want:<9.6g} "
                      f"{'ok' if ok else 'MISMATCH'}")
                if not ok:
                    failures.append(f"{contamination} a={alpha} {column}: "
                                    f"{got} != {want}")

    if failures:
        _fatal("Recomputed detection results do not reproduce the published "
               "Phase 1\n  novelty results:\n    "
               + "\n    ".join(failures)
               + "\n  This would not be the reference the RAG gate was "
               "measured against.\n  Do not write it; find out what moved.")
    return checked


def main():
    parser = argparse.ArgumentParser(
        description="Build the drift-detection reference distribution.")
    parser.add_argument("--force", action="store_true",
                        help="overwrite an existing reference file")
    parser.add_argument("--source", choices=("in-domain", "deployment"),
                        default="in-domain",
                        help="in-domain: Signal A reference (default); "
                             "deployment: Signal B reference")
    args = parser.parse_args()
    in_domain = args.source == "in-domain"
    source_json = CALIBRATION_JSON if in_domain else DEPLOYMENT_JSON

    out_path = (settings.drift.reference_path if in_domain
                else settings.drift.rate_reference_path)
    if out_path.exists() and not args.force:
        _fatal(f"Drift reference already exists:\n    {out_path}\n"
               "  Refusing to overwrite it silently. Re-run with --force if "
               "the index\n  or encoder was deliberately rebuilt.")

    _banner("STEP 1 - Loading artifacts and the fixed sets")
    try:
        from src.agent import pipeline
        from src.agent.artifacts import load_artifacts
        from src.agent.schemas import TicketIn
        from src.classification.train_tier1 import dataset_sha256
        from src.experiments.calibrate_conformal import (seed_key,
                                                         top_similarity)

        artifacts = load_artifacts(require_gemini=False)
    except AgentError as exc:
        _fatal(str(exc))

    calibration = _read_json(source_json, f"{args.source} source set")
    ood = _read_json(OOD_JSON, "OOD calibration set")
    adversarial = _read_json(ADVERSARIAL_JSON, "Adversarial set")
    print(f"  {len(calibration)} calibration tickets, {len(ood)} OOD, "
          f"{len(adversarial)} adversarial")
    print(f"  index ntotal {artifacts.index.ntotal}, "
          f"model {settings.models.embedding_model}")

    _banner("STEP 2 - Similarities (Signal A) and pipeline decisions "
            "(Signal B)")
    with_self, non_self = [], []
    n_self = n_escalated = n_tier1 = n_below_threshold = 0
    threshold = settings.rag.similarity_threshold
    categories = Counter()
    for i, rec in enumerate(calibration, 1):
        # Deployment tickets are not index rows, so there is no self to
        # exclude and the two scores coincide.
        top, nonself, is_self = top_similarity(
            rec["text"], artifacts, own_id=rec["id"] if in_domain else None)
        n_below_threshold += int(top < threshold)
        with_self.append(top)
        non_self.append(nonself)
        n_self += int(is_self)

        result = pipeline.run(
            TicketIn(title=rec["text"], ticket_id=str(rec["id"])),
            artifacts=artifacts, generate_resolution=False, emit_log=False,
        )
        if not math.isclose(result.top_similarity, top, rel_tol=0,
                            abs_tol=SIMILARITY_TOLERANCE):
            _fatal(f"Ticket {rec['id']}: pipeline top_similarity "
                   f"{result.top_similarity:.6f} != directly measured "
                   f"{top:.6f}.\n  The pipeline and this script are not "
                   "embedding the same text, so live\n  similarities would "
                   "not be comparable to this reference.")
        n_escalated += int(result.decision.escalated)
        n_tier1 += int(int(result.classification.tier) == 1)
        categories[result.classification.category] += 1
        if i % 25 == 0:
            print(f"  {i}/{len(calibration)}")

    n = len(calibration)
    print(f"\n  pipeline top_similarity == direct measurement: {n}/{n}")
    print(f"  self-retrieval: {n_self}/{n} = {n_self / n:.1%}")
    print(f"  escalated:      {n_escalated}/{n} = {n_escalated / n:.1%}")
    print(f"  Tier-1 share:   {n_tier1}/{n} = {n_tier1 / n:.1%}")
    print(f"  predicted mix:  {dict(sorted(categories.items()))}")

    if in_domain:
        _banner("STEP 3 - Independent check against published Phase 1 "
                "detection results")
        # Same retrieval call, same orientation as calibrate_conformal.py
        # STEP 5.
        ood_sims = [top_similarity(r["text"], artifacts)[0] for r in ood]
        ood_seeds = [seed_key(r["id"]) for r in ood]
        adv_sims = [top_similarity(r["text"], artifacts)[0]
                    for r in adversarial]
        checked = _check_against_published(with_self, non_self, ood_sims,
                                           ood_seeds, adv_sims)
    else:
        _banner("STEP 3 - Escalation counted two ways")
        print(f"  pipeline decision.escalated:            {n_escalated}")
        print(f"  direct top-1 similarity < {threshold}:     "
              f"{n_below_threshold}")
        if n_escalated != n_below_threshold:
            _fatal(f"Escalation count disagrees: pipeline {n_escalated}, "
                   f"direct measurement {n_below_threshold}.\n"
                   "  One of the two derivations is not measuring what it "
                   "claims. Do not write\n  a Signal B reference until the "
                   "difference is explained.")
        if n_escalated in (0, n):
            _fatal(f"The {args.source} set escalates {n_escalated}/{n}, so "
                   "an escalation-rate\n  test against it would be "
                   "degenerate -- the problem this reference exists to fix.")
        checked = {"escalated_pipeline": n_escalated,
                   "escalated_direct_below_threshold": n_below_threshold,
                   "threshold": threshold}

    _banner("STEP 4 - Writing the reference")
    from src.agent.drift import REFERENCE_SCHEMA_VERSION, DriftReference

    reference = DriftReference(
        schema_version=REFERENCE_SCHEMA_VERSION,
        embedding_model=settings.models.embedding_model,
        embedding_dim=settings.models.embedding_dim,
        index_ntotal=int(artifacts.index.ntotal),
        index_sha256=dataset_sha256(settings.models.faiss_index_path),
        source_sha256=dataset_sha256(source_json),
        config_fingerprint=config_fingerprint(),
        n=n,
        similarity_scores=non_self,
        similarity_scores_with_self=with_self,
        n_escalated=n_escalated,
        n_tier1=n_tier1,
        category_counts=dict(sorted(categories.items())),
        provenance={
            "source": ("data/" + os.path.basename(source_json)),
            "builder": "src/experiments/build_drift_reference.py",
            "self_retrieval_count": n_self,
            ("reproduced_published_detection_results" if in_domain
             else "escalation_counted_two_ways"): checked,
            "note": ("similarity_scores are non-self top-1; "
                     "similarity_scores_with_self are audit only. "
                     "category_counts are PREDICTED categories, the same "
                     "field a live decision record carries."),
        },
    )
    out_path.write_text(
        json.dumps(reference.model_dump(mode="json"), indent=2,
                   sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(f"  {out_path}")

    # Round-trip through THE loader, so the file just written is proven to
    # pass the same guard every consumer will apply.
    from src.agent.artifacts import load_drift_reference

    try:
        load_drift_reference(artifacts, source=args.source)
    except AgentError as exc:
        _fatal("The reference just written fails its own load guard:\n"
               + str(exc))
    print("  loads and passes the live-compatibility guard")

    _banner("DONE")


if __name__ == "__main__":
    main()
