# src/experiments/build_sufficiency_context.py
"""
build_sufficiency_context.py
============================

Phase 6C, step 1: persist the retrieval context the sufficiency autorater will
judge. OFFLINE -- loads BGE, spends ZERO Gemini quota.

WHAT THIS SCRIPT IS
-------------------
Runs the production pipeline over all 54 fixed benchmark tickets with
`generate_resolution=False` and captures, for each one, the ticket text and its
top-5 retrieved neighbours (title, category, description, resolution). The
output is a self-contained bundle: steps 2, 3 and 4 read it and load NO
embedding model at all.

That is deliberate and structural, not tidiness. Phase 7C's generation run was
killed for low memory because Ollama holds its weights for five minutes after
the last call, so loading BGE straight afterwards put both models resident at
once. Persisting the context here makes that overlap IMPOSSIBLE in 6C rather
than merely avoided by procedure -- the only process that touches BGE is this
one, and it exits before any rater runs.

WHY THE RATER NEVER SEES THE SIMILARITY SCORE
---------------------------------------------
6C asks whether a sufficiency check catches what the 0.67 top-1 similarity gate
misses. Putting the similarity in the rater's prompt would leak the very gate
the rater is being tested against. The bundle therefore separates the two: the
ITEM file carries only what the rater sees, and the KEY file carries the
similarity, the arm, the 2B item id and the escalation reason.

THE POPULATION, AND WHY 33 + 21
-------------------------------
Of the 54 fixed benchmark tickets, 21 escalate at the RAG gate and never reach
the resolver; 33 do reach it and are the ones Phase 2B drafted for and
human-labelled. 6C rates both groups:

    33  eligible  -- have a 2B human groundedness label. The PRIMARY population.
    21  escalated -- no draft and no label. The pre-registered consistency
                     secondary: the live gate already calls these insufficient.

ELIGIBILITY IS `decision.escalated`, NEVER THE STATUS ENUM. A RAG-gate
escalation carries status NEEDS_HUMAN_RESOLUTION, while
ResolutionStatus.ESCALATED belongs to the *filing* gate. Testing the status
enum here is occurrence 5 of this project's recurring bug class: it ran clean,
returned a plausible number, and silently classed all 21 escalating tickets as
eligible. `decision.escalated` is true in every escalation branch.

GUARDS (all fatal)
------------------
1. 33 eligible / 21 escalated, cross-checked against groundedness_key.json's
   _meta AND by set equality on the 21 escalated ids -- not just the count.
2. Retrieval identity: the recomputed top-5 ids and similarities must match the
   ones 2B recorded, for all 33. The config fingerprint has changed since 2B
   (05f391baf27c -> 9c9a5cbcb53f, from adding settings.drift.rate_reference_name),
   so this check is what makes that drift PROVABLY inert for 6C instead of
   assumed inert.
3. Every one of the 21 is a RAG-gate escalation (LOW_RETRIEVAL_SIMILARITY or
   NO_RETRIEVAL_RESULTS, top_similarity < 0.67) and the filing gate is off. A
   filing-gate escalation skips retrieval entirely, so there would be no
   context to rate and the secondary would silently shrink.
4. Non-empty retrieval for all 54.

BENCHMARKS ARE READ-ONLY. This script reads novel_tickets_expanded.json and
adversarial_escalation_tickets.json via build_groundedness_set.load_benchmarks()
and never writes to either. It also never writes to any Phase 2B artifact.

MEASUREMENT ONLY. Production stays frozen: cascade 0.50, RAG 0.67, clustering
0.80, settings.conformal.enabled and settings.drift.enabled both False.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import random
import sys

_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(_THIS_DIR, os.pardir, os.pardir))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from src.agent.artifacts import load_artifacts                    # noqa: E402
from src.agent.config import settings, config_fingerprint          # noqa: E402
from src.agent.logging_setup import ensure_utf8_console            # noqa: E402
from src.agent.pipeline import run as run_pipeline                 # noqa: E402
from src.agent.schemas import EscalationReason                     # noqa: E402
from src.experiments.build_groundedness_set import load_benchmarks  # noqa: E402

ensure_utf8_console()

DATA_DIR = os.path.join(PROJECT_ROOT, "data")
GROUNDEDNESS_KEY_PATH = os.path.join(DATA_DIR, "groundedness_key.json")
BUNDLE_PATH = os.path.join(DATA_DIR, "sufficiency_context_bundle.json")
KEY_PATH = os.path.join(DATA_DIR, "sufficiency_context_key.json")

RANDOM_SEED = 42

# The pre-registered population split. Both numbers are asserted against Phase
# 2B's own record rather than trusted from this docstring.
EXPECTED_ELIGIBLE = 33
EXPECTED_ESCALATED = 21

# Escalation reasons that still produce a retrieval to rate. A filing-gate
# escalation (LOW_CLASSIFICATION_CONFIDENCE) skips the retriever entirely.
RAG_GATE_REASONS = {
    EscalationReason.LOW_RETRIEVAL_SIMILARITY,
    EscalationReason.NO_RETRIEVAL_RESULTS,
}

SIM_TOLERANCE = 1e-6


def _banner(text):
    rule = "=" * 70
    print("\n" + rule + "\n" + text + "\n" + rule)


def _fatal(message):
    print("\n[ERROR] " + message + "\n", file=sys.stderr)
    sys.exit(1)


def load_groundedness_key():
    """Phase 2B's provenance key -- the second derivation for every count."""
    if not os.path.isfile(GROUNDEDNESS_KEY_PATH):
        _fatal(
            "Missing Phase 2B's provenance key:\n    {p}\n"
            "6C's population and its retrieval-identity guard are both defined "
            "against it. Rebuild 2B first, or run 6C from a clone that has "
            "it.".format(p=GROUNDEDNESS_KEY_PATH)
        )
    with open(GROUNDEDNESS_KEY_PATH, "r", encoding="utf-8") as fh:
        doc = json.load(fh)
    meta = doc.get("_meta", {})
    items = doc.get("items", [])
    if not items:
        _fatal("Phase 2B's key carries no items: " + GROUNDEDNESS_KEY_PATH)
    if meta.get("dry_run"):
        _fatal(
            "Phase 2B's key is marked dry_run, so it is a partial set. 6C's "
            "primary population must come from the full 2B run."
        )
    return meta, {row["ticket_id"]: row for row in items}


def route_all(artifacts):
    """Split the 54 benchmark tickets on decision.escalated. No Gemini calls."""
    benchmarks = load_benchmarks()
    eligible, escalated = [], []
    for source, tid, ticket in benchmarks:
        res = run_pipeline(ticket, artifacts, generate_resolution=False,
                           emit_log=False)
        row = (source, tid, ticket, res)
        if res.decision.escalated:
            escalated.append(row)
        else:
            eligible.append(row)
    return benchmarks, eligible, escalated


def check_population(eligible, escalated, meta_2b):
    """Guard 1 -- counts AND set equality, both against 2B's own record."""
    n_elig, n_esc = len(eligible), len(escalated)
    print("  pipeline decision.escalated : {a} eligible / {b} escalated"
          .format(a=n_elig, b=n_esc))
    print("  groundedness_key.json _meta : {a} eligible / {b} escalated"
          .format(a=meta_2b.get("eligible_population"),
                  b=meta_2b.get("escalated_no_draft")))

    if (n_elig, n_esc) != (EXPECTED_ELIGIBLE, EXPECTED_ESCALATED):
        _fatal(
            "Population split is {a}/{b}, expected {c}/{d}.\n"
            "The pre-registered 6C population is the 33 tickets Phase 2B "
            "drafted for plus the 21 the RAG gate escalates. A different split "
            "means retrieval or an artifact has changed since 2B -- stop and "
            "find out why before spending any quota.".format(
                a=n_elig, b=n_esc, c=EXPECTED_ELIGIBLE, d=EXPECTED_ESCALATED)
        )

    if (meta_2b.get("eligible_population") != EXPECTED_ELIGIBLE
            or meta_2b.get("escalated_no_draft") != EXPECTED_ESCALATED):
        _fatal(
            "Phase 2B's own key disagrees with the expected {c}/{d} split: it "
            "records {a}/{b}. Two independent derivations must agree before "
            "6C proceeds.".format(
                a=meta_2b.get("eligible_population"),
                b=meta_2b.get("escalated_no_draft"),
                c=EXPECTED_ELIGIBLE, d=EXPECTED_ESCALATED)
        )

    live_ids = {tid for _, tid, _, _ in escalated}
    recorded_ids = set(meta_2b.get("escalated_ids") or [])
    if live_ids != recorded_ids:
        _fatal(
            "The 21 escalating tickets are not the SAME 21 Phase 2B recorded.\n"
            "  only now : {a}\n  only 2B  : {b}\n"
            "Equal counts with different members is exactly this project's "
            "recurring bug shape -- internally consistent and wrong.".format(
                a=sorted(live_ids - recorded_ids),
                b=sorted(recorded_ids - live_ids))
        )
    print("  escalated id sets           : identical (21/21 by set equality)")


def check_retrieval_identity(eligible, key_2b):
    """Guard 2 -- the recomputed top-5 must be 2B's recorded top-5, for all 33."""
    mismatches = []
    for _, tid, _, res in eligible:
        row = key_2b.get(tid)
        if row is None:
            mismatches.append((tid, "not present in 2B's key", "", ""))
            continue
        live_ids = [str(r.id) for r in res.retrieval.retrieved]
        rec_ids = [str(x) for x in row.get("retrieved_ids", [])]
        if live_ids != rec_ids:
            mismatches.append((tid, "retrieved ids differ",
                               ",".join(live_ids), ",".join(rec_ids)))
            continue
        live_sims = [float(r.similarity) for r in res.retrieval.retrieved]
        rec_sims = [float(x) for x in row.get("retrieved_similarities", [])]
        if len(live_sims) != len(rec_sims) or any(
                abs(a - b) > SIM_TOLERANCE
                for a, b in zip(live_sims, rec_sims)):
            worst = max((abs(a - b) for a, b in zip(live_sims, rec_sims)),
                        default=float("nan"))
            mismatches.append((tid, "similarities differ (max |d| = %.3e)" % worst,
                               "", ""))

    if mismatches:
        lines = "\n".join(
            "    {t}: {w} {a} {b}".format(t=t, w=w, a=a, b=b)
            for t, w, a, b in mismatches[:10])
        _fatal(
            "Retrieval no longer matches what Phase 2B recorded, for {n} of "
            "{m} eligible tickets:\n{lines}\n"
            "6C rates the SAME context the human labelled against. A mismatch "
            "means the index, the encoder or the metadata has moved since 2B, "
            "and the 2B labels no longer describe this context.".format(
                n=len(mismatches), m=len(eligible), lines=lines)
        )
    print("  retrieval vs 2B's record    : identical for 33/33 "
          "(ids exact, similarities to %.0e)" % SIM_TOLERANCE)


def check_escalation_reasons(escalated):
    """Guard 3 -- all 21 must be RAG-gate escalations, with the filing gate off."""
    if settings.cascade.filing_gate_enabled:
        _fatal(
            "settings.cascade.filing_gate_enabled is True. The filing gate "
            "escalates BEFORE retrieval runs, so those tickets carry no "
            "context to rate and 6C's 21-ticket secondary would silently "
            "shrink. 6C is specified against the default (False)."
        )

    bad_reason, bad_sim, empty = [], [], []
    threshold = settings.rag.similarity_threshold
    for _, tid, _, res in escalated:
        reason = res.decision.reason
        if reason not in RAG_GATE_REASONS:
            bad_reason.append((tid, str(reason)))
        if not res.retrieval.retrieved:
            empty.append(tid)
        elif res.retrieval.top_similarity >= threshold:
            bad_sim.append((tid, res.retrieval.top_similarity))

    if bad_reason:
        _fatal(
            "These escalations are not RAG-gate escalations, so they have no "
            "retrieval to rate: {r}\nExpected one of {ok}.".format(
                r=bad_reason, ok=sorted(x.value for x in RAG_GATE_REASONS))
        )
    if empty:
        _fatal(
            "These escalating tickets returned NO retrieved neighbours, so "
            "there is no context for the rater to judge: {e}\n"
            "NO_RETRIEVAL_RESULTS is a legitimate production outcome but 6C "
            "cannot rate it; the population would need respecifying.".format(
                e=empty)
        )
    if bad_sim:
        _fatal(
            "These escalating tickets have top_similarity >= {t}, which "
            "contradicts a low-similarity escalation: {b}".format(
                t=threshold, b=bad_sim)
        )
    print("  escalation reasons          : 21/21 RAG-gate, all top1 < %.2f"
          % threshold)


def check_context_shape(rows):
    """Guard 4 -- every one of the 54 has a non-empty top-5."""
    empty = [tid for _, tid, _, res in rows if not res.retrieval.retrieved]
    if empty:
        _fatal("Empty retrieval for: " + ", ".join(empty))
    sizes = sorted({len(res.retrieval.retrieved) for _, _, _, res in rows})
    print("  retrieved-per-ticket sizes  : " + str(sizes))
    if sizes != [5]:
        print("  [note] not every ticket returned exactly 5 neighbours; the "
              "bundle records what retrieval actually produced.")


def build_items(eligible, escalated, key_2b):
    """Blind items under shuffled S-ids, plus the separate provenance key."""
    combined = ([("eligible", s, t, tk, r) for s, t, tk, r in eligible]
                + [("escalated", s, t, tk, r) for s, t, tk, r in escalated])
    # 2B's id for each eligible ticket, so step 4 can join to the human labels.
    g_by_ticket = {}
    for gid_row in key_2b.values():
        g_by_ticket[gid_row["ticket_id"]] = gid_row["item_id"]

    order = list(range(len(combined)))
    random.Random(RANDOM_SEED).shuffle(order)

    items, key_rows = [], []
    for new_i, idx in enumerate(order, start=1):
        arm, source, tid, ticket, res = combined[idx]
        sid = "S%03d" % new_i
        # What the rater sees. No similarity, no arm, no 2B label.
        items.append({
            "item_id": sid,
            "ticket_text": ticket.combined_text,
            "retrieved": [
                {
                    "rank": i + 1,
                    "category": r.category,
                    "title": r.title,
                    "description": r.description,
                    "resolution": r.resolution,
                }
                for i, r in enumerate(res.retrieval.retrieved)
            ],
        })
        # Everything the rater must not see.
        key_rows.append({
            "item_id": sid,
            "arm": arm,
            "source": source,
            "ticket_id": tid,
            "groundedness_item_id": g_by_ticket.get(tid),
            "predicted_category": res.classification.category,
            "top_similarity": res.retrieval.top_similarity,
            "escalated": bool(res.decision.escalated),
            "escalation_reason": str(res.decision.reason.value),
            "retrieved_ids": [str(r.id) for r in res.retrieval.retrieved],
            "retrieved_similarities": [float(r.similarity)
                                       for r in res.retrieval.retrieved],
        })
    return items, key_rows


def parse_args(argv=None):
    p = argparse.ArgumentParser(
        description=("Phase 6C step 1: persist the retrieval context for the "
                     "sufficiency autorater. Offline, zero Gemini calls."))
    p.add_argument("--force", action="store_true",
                   help="Overwrite an existing bundle and key.")
    return p.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)

    _banner("Phase 6C / step 1 -- build the sufficiency context bundle")
    print("Offline. ZERO Gemini calls. Measurement only.")
    print("RAG gate            : %.2f" % settings.rag.similarity_threshold)
    print("cascade gate        : %.2f" % settings.cascade.confidence_threshold)
    print("filing gate enabled : %s" % settings.cascade.filing_gate_enabled)
    print("conformal enabled   : %s" % settings.conformal.enabled)
    print("config fingerprint  : %s" % config_fingerprint())

    for path in (BUNDLE_PATH, KEY_PATH):
        if os.path.exists(path) and not args.force:
            _fatal(
                "Refusing to overwrite:\n    {p}\n"
                "Re-run with --force if that is what you intend. The raters "
                "cache by prompt hash, so a changed bundle invalidates every "
                "cached verdict built from the old context.".format(p=path)
            )

    meta_2b, key_2b = load_groundedness_key()
    print("2B key fingerprint  : %s" % meta_2b.get("config_fingerprint"))
    if meta_2b.get("config_fingerprint") != config_fingerprint():
        print("  [note] the fingerprint has changed since Phase 2B. Guard 2 "
              "below is what decides whether that matters: it compares the "
              "recomputed retrieval against 2B's recorded retrieval.")

    artifacts = load_artifacts(require_gemini=False)
    print("[load] index ntotal = %d, metadata = %d"
          % (artifacts.index.ntotal, len(artifacts.metadata)))

    _banner("Routing pass -- generate_resolution=False, no quota spent")
    benchmarks, eligible, escalated = route_all(artifacts)
    print("  benchmark tickets           : %d" % len(benchmarks))

    _banner("Guards")
    check_population(eligible, escalated, meta_2b)
    check_retrieval_identity(eligible, key_2b)
    check_escalation_reasons(escalated)
    check_context_shape(eligible + escalated)

    items, key_rows = build_items(eligible, escalated, key_2b)

    bundle = {
        "_meta": {
            "phase": "6C",
            "purpose": ("Retrieval context for the Phase 6C sufficiency "
                        "autorater. The rater sees ONLY these fields: the "
                        "ticket and its retrieved neighbours' category, "
                        "title, description and resolution."),
            "blind_by_design": ("No similarity score, no arm, no escalation "
                                "reason and no Phase 2B label appears here -- "
                                "showing the similarity would leak the very "
                                "gate 6C is tested against. That provenance "
                                "is in sufficiency_context_key.json."),
            "items": len(items),
            "eligible": len(eligible),
            "escalated": len(escalated),
            "seed": RANDOM_SEED,
            "config_fingerprint": config_fingerprint(),
            "rag_gate": settings.rag.similarity_threshold,
            "groundedness_key_fingerprint": meta_2b.get("config_fingerprint"),
            "retrieval_verified_against_2b": True,
        },
        "items": items,
    }
    key_doc = {
        "_meta": {
            "phase": "6C",
            "purpose": "Provenance key for sufficiency_context_bundle.json.",
            "warning": ("Do not put any field from this file into a rater "
                        "prompt."),
            "items": len(key_rows),
            "seed": RANDOM_SEED,
            "config_fingerprint": config_fingerprint(),
        },
        "items": key_rows,
    }

    with open(BUNDLE_PATH, "w", encoding="utf-8") as fh:
        json.dump(bundle, fh, indent=2, ensure_ascii=False)
    with open(KEY_PATH, "w", encoding="utf-8") as fh:
        json.dump(key_doc, fh, indent=2, ensure_ascii=False)

    digest = hashlib.sha256(
        open(BUNDLE_PATH, "rb").read()).hexdigest()[:12]

    _banner("DONE")
    print("[write] " + BUNDLE_PATH)
    print("[write] " + KEY_PATH)
    print("bundle sha256[:12] : " + digest)
    print("items              : %d  (%d eligible + %d escalated)"
          % (len(items), len(eligible), len(escalated)))
    print("\nNext: the dry run, three calls only --")
    print("  python src/experiments/run_sufficiency_autorater.py "
          "--backend gemini --limit 3")


if __name__ == "__main__":
    main()
