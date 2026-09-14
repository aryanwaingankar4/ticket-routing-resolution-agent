# src/experiments/build_groundedness_set.py
"""
build_groundedness_set.py
=========================

Phase 2B, step 1: generate the drafts to be judged for GROUNDEDNESS.

WHAT THIS SCRIPT IS
-------------------
Runs the production pipeline over the fixed benchmark tickets and captures,
for each ticket that actually reaches the resolver, the drafted resolution
together with the exact retrieved context that grounded it. The output is a
blind labelling file.

Groundedness asks one question: does the drafted resolution say only things
supported by the retrieved past tickets?

WHY ONLY THE BENCHMARKS, AND WHY ONLY 33
-----------------------------------------
A pre-flight diagnostic (Phase 2B part 1) measured whether the retrieved
context leaves any room to be ungrounded at all. A judge can only find an
unsupported claim if the top-5 context permits one; if all five retrieved
resolutions prescribe the same fix, any faithful draft is grounded by
construction and the judge returns 1.0 on every ticket with no variance.

    query set              all 5 -> one fix     2+ distinct fixes
    in-distribution (200)        85.0%                15.0%
    novel-45                     28.9%                71.1%
    adversarial-9                11.1%                88.9%

The in-distribution corpus is structurally degenerate -- the same failure
mode Phase 2A hit, where zero cross-template merges meant there was no
precision signal to measure. The benchmarks are not: on the novel-45, 71% of
retrievals return genuinely competing fixes and 40% span more than one
category. So the harness targets the benchmarks and drops the 500-ticket
in-distribution batch entirely.

The 500 existing drafts are unusable here for a second, independent reason:
they were generated when retrieval ran under MiniLM (see "Known
inconsistencies" in CLAUDE.md), so the context they actually saw is not the
context we would judge them against. A groundedness audit must judge a draft
against its real context, which is why this script regenerates rather than
reusing them.

Of the 54 fixed benchmark tickets, 21 escalate at the RAG gate (0.67) and
never reach Gemini, so no draft exists to judge. That leaves 33:

    novel-45        30 reach the resolver, 15 escalate
    adversarial-9    3 reach the resolver,  6 escalate

33 is the entire eligible population, not a sample, so there is no sampling
guard here and no near-duplicate check -- there is nothing to skew.

The 14-ticket NOVEL_TICKETS set in generalization_test.py was checked and is
a strict subset of the 45 (14/14 appear verbatim). It adds nothing.

COST
----
One Gemini call per draft. --limit N generates only the first N, for a cheap
prompt check before committing the rest, per this project's standing practice
of dry-running any Gemini generator first.

BENCHMARKS ARE READ-ONLY. This script reads novel_tickets_expanded.json and
adversarial_escalation_tickets.json and never writes to either.
"""

from __future__ import annotations

import argparse
import json
import os
import random
import sys
import time

_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(_THIS_DIR, os.pardir, os.pardir))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from src.agent.artifacts import load_artifacts
from src.agent.config import settings
from src.agent.pipeline import run as run_pipeline
from src.agent.schemas import TicketIn

DATA_DIR = os.path.join(PROJECT_ROOT, "data")
NOVEL_PATH = os.path.join(DATA_DIR, "novel_tickets_expanded.json")
ADVERSARIAL_PATH = os.path.join(DATA_DIR, "adversarial_escalation_tickets.json")

OUTPUT_SET_PATH = os.path.join(DATA_DIR, "groundedness_set.json")
OUTPUT_KEY_PATH = os.path.join(DATA_DIR, "groundedness_key.json")

RANDOM_SEED = 42

# Free tier is 15 req/min, 500/day. Keep at 4s or above (project invariant).
GEMINI_CALL_DELAY_SEC = 4.0

# ---------------------------------------------------------------------------
# THE RUBRIC. One definition, shared verbatim by the human labeller and the
# LLM judge -- agreement between them is only interpretable if both answered
# the same question. run_groundedness_judge.py imports this, it does not
# restate it.
# ---------------------------------------------------------------------------

RUBRIC_LABELS = ("grounded", "partially_grounded", "ungrounded")

RUBRIC_TEXT = """\
You are judging GROUNDEDNESS of a drafted IT-support resolution.

You are given a new ticket, a drafted resolution, and the past tickets that
were retrieved to ground that draft. Judge ONLY whether the draft's content is
supported by the retrieved resolutions. Do NOT judge whether the draft is a
good fix, whether it would work, or whether you would have written it
differently. A draft can be excellent advice and still be ungrounded.

Choose exactly one label:

  "grounded"
      Every substantive step the draft prescribes is traceable to at least
      one of the retrieved resolutions. Rewording, merging steps from several
      retrieved tickets, and substituting the new ticket's own hostname,
      app name or user is all fine -- that is not new content.

  "partially_grounded"
      The core fix is supported by the retrieved resolutions, but the draft
      adds at least one substantive step, cause, or claim that no retrieved
      resolution supports.

  "ungrounded"
      The fix the draft prescribes is not supported by any retrieved
      resolution -- it answers from general knowledge, or addresses a
      different problem than the retrieved tickets solved.

DECLINING IS NOT AN UNSUPPORTED CLAIM. A draft that explicitly declines to
apply the retrieved fix, states that the retrieved examples do not match the
new ticket, and recommends that a human investigate is "grounded". Judge the
substantive fix steps a draft PRESCRIBES, not its recommendation to
investigate. Penalising a correct refusal would measure the opposite of what
this rubric is for.

Then answer one further question about the draft's closing note:

  hedge_appropriate: true | false
      The drafter is instructed to end with a note saying either that the
      retrieved examples closely match, or that a human should verify.
      Answer true if that note matches how well the retrieved resolutions
      actually agree with each other and with the ticket; answer false if it
      claims a close match when the retrieved tickets in fact disagree or
      fit poorly, or hedges when they plainly agree.
"""


def die(message):
    print("\n[ERROR] " + message + "\n", file=sys.stderr)
    sys.exit(1)


def load_benchmarks():
    """Return [(source, ticket_id, TicketIn)] for both fixed benchmarks."""
    out = []

    if not os.path.exists(NOVEL_PATH):
        die("Missing the 45-ticket benchmark:\n    " + NOVEL_PATH
            + "\nThis is a fixed, versioned reference set -- if it is absent, "
              "something deleted it. It must never be regenerated.")
    novel = json.load(open(NOVEL_PATH, encoding="utf-8"))
    for i, x in enumerate(novel):
        out.append(("novel-45", "N%02d" % (i + 1),
                    TicketIn(title="", description=x["text"])))

    if os.path.exists(ADVERSARIAL_PATH):
        adv = json.load(open(ADVERSARIAL_PATH, encoding="utf-8"))
        items = adv if isinstance(adv, list) else adv.get("tickets", [])
        for i, x in enumerate(items):
            out.append(("adversarial-9", "A%02d" % (i + 1),
                        TicketIn(title=x.get("title", "") or "",
                                 description=(x.get("text")
                                              or x.get("description") or ""))))
    else:
        print("[warn] adversarial set not found at " + ADVERSARIAL_PATH
              + "; continuing with the 45-ticket benchmark only.")

    return out


def parse_args():
    p = argparse.ArgumentParser(
        description=("Generate the Phase 2B groundedness drafts. One Gemini "
                     "call per draft."))
    p.add_argument("--limit", type=int, default=None,
                   help=("Generate only the first N drafts. Use --limit 3 to "
                         "check the prompt before spending the full budget."))
    return p.parse_args()


def main():
    args = parse_args()

    print("=" * 74)
    print("Phase 2B / step 1 -- generate groundedness drafts")
    print("=" * 74)
    print("RAG gate            : %.2f" % settings.rag.similarity_threshold)
    print("Gemini model        : " + settings.models.gemini_model)
    print("Call delay          : %.1fs" % GEMINI_CALL_DELAY_SEC)
    if args.limit:
        print("MODE                : DRY RUN, first %d draft(s) only"
              % args.limit)

    artifacts = load_artifacts(require_gemini=True)
    print("[load] index ntotal = %d, metadata = %d"
          % (artifacts.index.ntotal, len(artifacts.metadata)))

    benchmarks = load_benchmarks()
    print("[load] benchmark tickets = %d" % len(benchmarks))

    # ---- Pass 1: routing only, no Gemini quota -----------------------------
    print("\n" + "=" * 74)
    print("STEP 1 -- routing pass (no Gemini calls)")
    print("=" * 74)
    print("generate_resolution=False, so the escalation decision is made")
    print("without spending any quota. Only tickets that reach the resolver")
    print("can produce a draft to judge.")

    # Eligibility is decided by `decision.escalated`, NOT by the status enum.
    # A RAG-gate escalation carries status NEEDS_HUMAN_RESOLUTION while
    # ResolutionStatus.ESCALATED belongs to the filing gate -- checking the
    # status here silently treated all 21 escalating benchmark tickets as
    # eligible and would have spent 54 Gemini calls instead of 33, drafting
    # for tickets production never sends to the resolver. `decision.escalated`
    # is true in every escalation branch, so it is the one correct test.
    eligible = []
    escalated = []
    for source, tid, ticket in benchmarks:
        res = run_pipeline(ticket, artifacts, generate_resolution=False,
                           emit_log=False)
        if res.decision.escalated:
            escalated.append((source, tid, res))
        else:
            eligible.append((source, tid, ticket, res))

    by_source = {}
    for source, tid, ticket, res in eligible:
        by_source.setdefault(source, 0)
        by_source[source] += 1
    print("\n[route] reach the resolver : %d" % len(eligible))
    for k in sorted(by_source):
        print("         %-16s %d" % (k, by_source[k]))
    print("[route] escalate (no draft): %d" % len(escalated))

    if not eligible:
        die("No benchmark ticket reaches the resolver. Nothing to generate.")

    # ---- Pass 2: generate ---------------------------------------------------
    todo = eligible[:args.limit] if args.limit else eligible
    print("\n" + "=" * 74)
    print("STEP 2 -- generating %d draft(s)  [%d Gemini call(s)]"
          % (len(todo), len(todo)))
    print("=" * 74)

    records = []
    failures = []
    for n, (source, tid, ticket, _) in enumerate(todo, start=1):
        print("  [%2d/%2d] %-14s %s" % (n, len(todo), tid,
                                        ticket.combined_text[:46]))
        try:
            res = run_pipeline(ticket, artifacts, generate_resolution=True,
                               emit_log=False)
        except Exception as exc:  # noqa: BLE001 - reported, not swallowed
            print("          FAILED: %s: %s" % (type(exc).__name__, exc))
            failures.append((tid, type(exc).__name__, str(exc)[:200]))
            continue

        if res.suggestion is None:
            print("          no draft produced (status=%s)" % res.status)
            failures.append((tid, "NoDraft", str(res.status)))
            continue

        records.append({
            "source": source,
            "ticket_id": tid,
            "ticket_text": ticket.combined_text,
            "predicted_category": res.classification.category,
            "draft": res.suggestion.text,
            "llm_model": res.suggestion.llm_model,
            "config_fingerprint": res.config_fingerprint,
            "top_similarity": res.retrieval.top_similarity,
            "retrieved": [
                {
                    "rank": i + 1,
                    "id": r.id,
                    "title": r.title,
                    "description": r.description,
                    "category": r.category,
                    "resolution": r.resolution,
                    "similarity": r.similarity,
                }
                for i, r in enumerate(res.retrieval.retrieved)
            ],
        })
        if n < len(todo):
            time.sleep(GEMINI_CALL_DELAY_SEC)

    print("\n[generate] drafts produced : %d" % len(records))
    print("[generate] failures        : %d" % len(failures))
    for tid, kind, msg in failures:
        print("           %s  %s: %s" % (tid, kind, msg))

    if not records:
        die("No drafts were produced. Nothing to write.")

    fingerprints = sorted({r["config_fingerprint"] for r in records})
    if len(fingerprints) > 1:
        die("Drafts carry more than one config fingerprint: "
            + str(fingerprints) + "\nThe configuration changed mid-run. "
            "Discard and regenerate -- this is exactly the stale-artifact "
            "class this project keeps hitting.")
    print("[generate] config fingerprint: " + fingerprints[0])

    # ---- Write the blind labelling file and the key ------------------------
    rng = random.Random(RANDOM_SEED)
    order = list(range(len(records)))
    rng.shuffle(order)

    labelling = []
    key_rows = []
    for new_i, idx in enumerate(order, start=1):
        r = records[idx]
        pair_id = "G%03d" % new_i
        labelling.append({
            "item_id": pair_id,
            "ticket_text": r["ticket_text"],
            "draft": r["draft"],
            "retrieved_resolutions": [
                {"rank": x["rank"], "category": x["category"],
                 "title": x["title"], "resolution": x["resolution"]}
                for x in r["retrieved"]
            ],
            "label": None,
            "hedge_appropriate": None,
            "unsupported_span": "",
            "label_reason": "",
        })
        key_rows.append({
            "item_id": pair_id,
            "source": r["source"],
            "ticket_id": r["ticket_id"],
            "predicted_category": r["predicted_category"],
            "top_similarity": r["top_similarity"],
            "llm_model": r["llm_model"],
            "config_fingerprint": r["config_fingerprint"],
            "retrieved_similarities": [x["similarity"] for x in r["retrieved"]],
            "retrieved_ids": [x["id"] for x in r["retrieved"]],
        })

    set_doc = {
        "_meta": {
            "purpose": ("Phase 2B groundedness labelling set. Human ground "
                        "truth for whether each drafted resolution says only "
                        "what its retrieved context supports."),
            "dry_run": bool(args.limit),
            "how_to_label": ("For each item set \"label\" to exactly one of "
                             + str(list(RUBRIC_LABELS))
                             + ", set \"hedge_appropriate\" to true or false, "
                               "and when the label is not \"grounded\" put the "
                               "offending text in \"unsupported_span\". "
                               "\"label_reason\" is optional but useful."),
            "rubric": RUBRIC_TEXT,
            "blind_by_design": ("This file carries only the ticket, the draft "
                                "and the retrieved resolutions. Provenance is "
                                "in groundedness_key.json and the LLM judge's "
                                "verdicts are in groundedness_judge.json -- "
                                "do not open either while labelling."),
            "items": len(labelling),
            "seed": RANDOM_SEED,
        },
        "items": labelling,
    }

    key_doc = {
        "_meta": {
            "purpose": "Provenance key for groundedness_set.json.",
            "dry_run": bool(args.limit),
            "config_fingerprint": fingerprints[0],
            "gemini_model": settings.models.gemini_model,
            "rag_gate": settings.rag.similarity_threshold,
            "eligible_population": len(eligible),
            "escalated_no_draft": len(escalated),
            "generated": len(records),
            "escalated_ids": [tid for _, tid, _ in escalated],
            "note": ("33 is the entire eligible population of the two fixed "
                     "benchmarks, not a sample. The 14-ticket NOVEL_TICKETS "
                     "set is a strict subset of the 45 and adds nothing."),
            "seed": RANDOM_SEED,
        },
        "items": key_rows,
    }

    with open(OUTPUT_SET_PATH, "w", encoding="utf-8") as fh:
        json.dump(set_doc, fh, indent=2, ensure_ascii=False)
    with open(OUTPUT_KEY_PATH, "w", encoding="utf-8") as fh:
        json.dump(key_doc, fh, indent=2, ensure_ascii=False)

    print("\n[write] labelling file -> " + OUTPUT_SET_PATH)
    print("[write] provenance key -> " + OUTPUT_KEY_PATH)

    print("\n" + "=" * 74)
    if args.limit:
        print("DRY RUN COMPLETE -- check the drafts above before the full run")
        print("=" * 74)
        print("Re-run without --limit to generate all %d." % len(eligible))
        print("NOTE: the full run OVERWRITES these dry-run files.")
    else:
        print("DONE")
        print("=" * 74)
        print("Next: python src/experiments/run_groundedness_judge.py"
              "   [%d more Gemini calls]" % len(records))
        print("      then label data/groundedness_set.json")
        print("      then python src/experiments/score_groundedness_set.py")


if __name__ == "__main__":
    main()
