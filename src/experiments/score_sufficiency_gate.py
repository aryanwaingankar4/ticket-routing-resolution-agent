# src/experiments/score_sufficiency_gate.py
"""
score_sufficiency_gate.py
=========================

Phase 6C, step 4: score the sufficiency autorater against Phase 2B's human
groundedness labels. OFFLINE -- zero Gemini calls, no model loading.

WHAT IS PRE-REGISTERED AND WHAT IS NOT
--------------------------------------
PRIMARY, as COUNTS not rates: over the 33 eligible tickets, the 2x2 of the
rater's verdict against the 2B human label. Of the 2 human-labelled ungrounded
drafts, how many are flagged INSUFFICIENT; of the 31 grounded, how many are
wrongly flagged.

With 2 positives NO rate, proportion, interval, kappa or significance test is
estimable on the primary axis. This script therefore refuses to print one: the
primary is a CASE STUDY, reported as counts plus every disagreement quoted in
full. The 31-ticket false-flag axis DOES carry a proportion and gets a Wilson
interval, but it is not the primary question and is not presented as one.

THE PRIMARY VERDICT FOR EVERY ITEM IS THE FIRST CALL ONLY (rep 1). The
stability repeats are reported separately and never revise it. There is NO
majority vote. A repeat that differs from its rep 1 at temperature 0.0 is a
DETERMINISM FINDING about the model, not a reason to revise the primary.

PRE-REGISTERED SECONDARIES: agreement with the live RAG gate on the 21 tickets
it already escalates; the same rubric on local qwen2.5:3b-instruct as a
cross-family check; verdict stability on the two positives' items.

POST-HOC, and labelled as such everywhere it appears:
  * the declined-draft cross-tab (see below);
  * the context_distinct_fixes covariate, REUSED from 2B's
    groundedness_results.csv rather than recomputed -- recomputing it would
    mean loading MiniLM, and 6C loads no embedding model at all.

THE POST-HOC DECLINED-DRAFT CROSS-TAB
-------------------------------------
2B's drafter was instructed to close with a note saying either that the
retrieved examples closely match, or that a human should verify. So a draft
can already contain mismatch language. This cross-tab splits the 31 grounded
tickets by whether their 2B draft declined or hedged that way.

The reason it matters: 2B's rubric says explicitly that DECLINING IS NOT AN
UNSUPPORTED CLAIM -- a draft that says the retrieved examples do not fit and
recommends a human is labelled "grounded". 6C's rater, looking at the same
retrieval, may call that same context INSUFFICIENT. Both would be right. If the
false flags CONCENTRATE on declined drafts, the two are answering different
questions and the false-flag count is not a straightforward error rate.

The detector is a transparent phrase list, and every match is recorded with
the phrase that fired so the split is auditable rather than asserted. It is
post-hoc and is reported BESIDE the primary, never in place of it.

A REJECTED HYPOTHESIS, recorded so it is not re-proposed
--------------------------------------------------------
The rater's aggressiveness is NOT attributed to near-duplicate retrieved
neighbours leaving nothing to distinguish. That explanation contradicts 2B's
own diagnostic on these same tickets: 71.1% of benchmark-45 retrievals and
88.9% of adversarial-9 retrievals contain 2+ DISTINCT fixes. The context is
heterogeneous, so "the neighbours are all the same" does not hold here. It is
recorded as a rejected hypothesis, not as a mechanism, and 6C is NOT claimed
as an instance of the project's named finding on that basis.

MEASUREMENT ONLY. Production frozen. Refuses to overwrite without --force.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import sys

_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(_THIS_DIR, os.pardir, os.pardir))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from src.agent.config import settings, config_fingerprint          # noqa: E402
from src.agent.logging_setup import ensure_utf8_console            # noqa: E402
from src.experiments.run_zeroshot_baselines import _slug           # noqa: E402

ensure_utf8_console()

DATA_DIR = os.path.join(PROJECT_ROOT, "data")
KEY_PATH = os.path.join(DATA_DIR, "sufficiency_context_key.json")
BUNDLE_PATH = os.path.join(DATA_DIR, "sufficiency_context_bundle.json")
GROUNDEDNESS_SET = os.path.join(DATA_DIR, "groundedness_set.json")
GROUNDEDNESS_CSV = os.path.join(DATA_DIR, "groundedness_results.csv")
RAW_DIR = os.path.join(DATA_DIR, "sufficiency_raw")

OUT_CSV = os.path.join(DATA_DIR, "sufficiency_gate_results.csv")
OUT_JSON = os.path.join(DATA_DIR, "sufficiency_gate_summary.json")

GEMINI_MODEL = "gemini-flash-lite-latest"
QWEN_MODEL = "qwen2.5:3b-instruct"

# Phase 2B's own retrieval-heterogeneity diagnostic, quoted for the rejected
# hypothesis. Source: build_groundedness_set.py's docstring table.
TWOB_DISTINCT_FIX_RATES = {"novel-45": 0.711, "adversarial-9": 0.889}

# POST-HOC detector. Lower-cased substring match against the 2B draft.
DECLINE_PHRASES = (
    "do not match", "does not match", "not match",
    "do not appear to match", "does not appear to match",
    "not directly applicable", "not applicable",
    "none of the retrieved", "no retrieved",
    "do not fit", "does not fit", "poor fit",
    "differ from", "different from the retrieved",
    "unrelated to", "not related to",
    "human should verify", "a human should", "human review",
    "recommend that a human", "escalate", "manual investigation",
    "manually investigate", "further investigation",
    "insufficient", "cannot be resolved using", "limited applicability",
)


def _banner(text):
    rule = "=" * 70
    print("\n" + rule + "\n" + text + "\n" + rule)


def _fatal(message):
    print("\n[ERROR] " + message + "\n", file=sys.stderr)
    sys.exit(1)


def wilson_interval(k, n, z=1.96):
    """Wilson score interval -- honest at small n, unlike the normal approx."""
    if n == 0:
        return (0.0, 1.0)
    p = k / n
    d = 1 + z * z / n
    centre = (p + z * z / (2 * n)) / d
    half = (z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n))) / d
    return (max(0.0, centre - half), min(1.0, centre + half))


def load_json(path, what):
    if not os.path.isfile(path):
        _fatal("Missing {w}:\n    {p}".format(w=what, p=path))
    with open(path, "r", encoding="utf-8") as fh:
        return json.load(fh)


def verdicts_path(backend, model, suffix=""):
    return os.path.join(
        DATA_DIR, "sufficiency_verdicts_{b}_{m}{s}.json".format(
            b=backend, m=_slug(model), s=suffix))


def load_arm(backend, model, what, required=True):
    path = verdicts_path(backend, model)
    if not os.path.isfile(path):
        if required:
            _fatal("Missing {w}:\n    {p}".format(w=what, p=path))
        print("[warn] {w} is absent ({p}); that arm is reported as not run."
              .format(w=what, p=os.path.basename(path)))
        return None
    doc = load_json(path, what)
    if doc["_meta"].get("dry_run"):
        _fatal("{w} is marked dry_run; it is not the full pass.".format(w=what))
    primary = {r["item_id"]: r for r in doc["records"] if r["is_primary"]}
    if len(primary) != 54:
        _fatal(
            "{w} carries {n} primary records, expected 54. A partial arm "
            "cannot be scored.".format(w=what, n=len(primary)))
    bad = [r["item_id"] for r in primary.values() if r["parse_error"]]
    if bad:
        print("[warn] {w}: {n} unparseable -- {b}".format(
            w=what, n=len(bad), b=", ".join(bad)))
    return doc, primary


def load_human_labels():
    """Two independent derivations: the labelling file AND the results CSV."""
    set_doc = load_json(GROUNDEDNESS_SET, "Phase 2B labelling set")
    from_set = {it["item_id"]: it["label"] for it in set_doc["items"]}
    drafts = {it["item_id"]: it["draft"] for it in set_doc["items"]}
    hedge = {it["item_id"]: it["hedge_appropriate"] for it in set_doc["items"]}

    if not os.path.isfile(GROUNDEDNESS_CSV):
        _fatal("Missing Phase 2B's results CSV:\n    " + GROUNDEDNESS_CSV)
    with open(GROUNDEDNESS_CSV, "r", encoding="utf-8") as fh:
        rows = list(csv.DictReader(fh))
    from_csv = {r["item_id"]: r["human_label"] for r in rows}
    distinct_fixes = {r["item_id"]: int(r["context_distinct_fixes"])
                      for r in rows}

    if from_set != from_csv:
        diff = {k: (from_set.get(k), from_csv.get(k))
                for k in set(from_set) | set(from_csv)
                if from_set.get(k) != from_csv.get(k)}
        _fatal(
            "Phase 2B's human labels disagree between the labelling set and "
            "the results CSV: {d}\nTwo derivations of the ground truth must "
            "agree before 6C is scored against it.".format(d=diff))
    print("  human labels          : 33/33 identical in "
          "groundedness_set.json and groundedness_results.csv")
    return from_set, drafts, hedge, distinct_fixes


def detect_decline(draft):
    """POST-HOC. Return the phrases that fired, or []."""
    low = (draft or "").lower()
    return [p for p in DECLINE_PHRASES if p in low]


def two_by_two(rows, verdict_field):
    """Counts, derived twice: by iteration and by set intersection."""
    ins = {r["item_id"] for r in rows if r[verdict_field] == "INSUFFICIENT"}
    suf = {r["item_id"] for r in rows if r[verdict_field] == "SUFFICIENT"}
    ung = {r["item_id"] for r in rows if r["human_label"] != "grounded"}
    gnd = {r["item_id"] for r in rows if r["human_label"] == "grounded"}

    cells = {
        "insufficient_and_ungrounded": len(ins & ung),
        "insufficient_and_grounded": len(ins & gnd),
        "sufficient_and_ungrounded": len(suf & ung),
        "sufficient_and_grounded": len(suf & gnd),
    }
    # Derivation 2: iterate.
    it = {k: 0 for k in cells}
    for r in rows:
        v, h = r[verdict_field], r["human_label"]
        if not v:
            continue
        key = ("insufficient" if v == "INSUFFICIENT" else "sufficient") \
            + ("_and_grounded" if h == "grounded" else "_and_ungrounded")
        it[key] += 1
    if it != cells:
        _fatal("2x2 cells disagree between derivations: {a} vs {b}"
               .format(a=cells, b=it))
    total = sum(cells.values())
    if total != len(rows):
        _fatal("2x2 cells sum to {t}, expected {n} (unparseable verdicts?)"
               .format(t=total, n=len(rows)))
    return cells


def parse_args(argv=None):
    p = argparse.ArgumentParser(
        description="Phase 6C step 4: score the sufficiency gate. Offline.")
    p.add_argument("--force", action="store_true",
                   help="Overwrite the results CSV and summary JSON.")
    return p.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)

    _banner("Phase 6C / step 4 -- score the sufficiency gate")
    print("Offline. Zero Gemini calls. Measurement only.")
    print("PRIMARY   : rep 1 only, counts not rates, 2 positives")
    print("config fingerprint : " + config_fingerprint())

    for path in (OUT_CSV, OUT_JSON):
        if os.path.exists(path) and not args.force:
            _fatal("Refusing to overwrite:\n    {p}\nRe-run with --force."
                   .format(p=path))

    key = load_json(KEY_PATH, "the 6C context key")
    key_rows = {r["item_id"]: r for r in key["items"]}

    _banner("Ground truth")
    labels, drafts, hedge, distinct_fixes = load_human_labels()

    _banner("Arms")
    gem_doc, gem = load_arm("gemini", GEMINI_MODEL, "the Gemini arm")
    print("  gemini                : 54 primary records, model %s"
          % gem_doc["_meta"]["model"])
    qwen_loaded = load_arm("ollama", QWEN_MODEL, "the Qwen arm", required=False)
    if qwen_loaded is None:
        qwen_doc, qwen = None, {}
    else:
        qwen_doc, qwen = qwen_loaded
        print("  ollama                : 54 primary records, model %s "
              "(digest %s)" % (qwen_doc["_meta"]["model"],
                               (qwen_doc["_meta"].get("model_digest")
                                or "?")[:16]))

    # ---- prompt identity across the two arms ----------------------------- #
    hash_mismatch = []
    if qwen:
        for sid in sorted(gem):
            if gem[sid]["prompt_sha256"] != qwen[sid]["prompt_sha256"]:
                hash_mismatch.append(sid)
        if hash_mismatch:
            _fatal(
                "The two arms did not see the same prompt for: {m}\n"
                "Agreement between raters is only interpretable if both "
                "answered the same question.".format(m=hash_mismatch[:10]))
        print("  prompt identity       : 54/54 byte-identical (sha256), "
              "both arms")

    # ---- per-item rows ---------------------------------------------------- #
    rows = []
    for sid in sorted(key_rows):
        k = key_rows[sid]
        gid = k["groundedness_item_id"]
        decl = detect_decline(drafts.get(gid)) if gid else []
        rows.append({
            "item_id": sid,
            "arm": k["arm"],
            "ticket_id": k["ticket_id"],
            "groundedness_item_id": gid or "",
            "top_similarity": k["top_similarity"],
            "escalation_reason": k["escalation_reason"],
            "human_label": labels.get(gid, "") if gid else "",
            "human_hedge_appropriate": (hedge.get(gid) if gid else ""),
            "context_distinct_fixes_posthoc": (distinct_fixes.get(gid, "")
                                               if gid else ""),
            "gemini_verdict": gem[sid]["verdict"],
            "gemini_reason": gem[sid]["reason"],
            "qwen_verdict": (qwen[sid]["verdict"] if qwen else ""),
            "qwen_reason": (qwen[sid]["reason"] if qwen else ""),
            "arms_agree": (bool(qwen)
                           and gem[sid]["verdict"] == qwen[sid]["verdict"]),
            "draft_declines_posthoc": bool(decl),
            "decline_phrases_posthoc": "; ".join(decl),
            "prompt_sha256": gem[sid]["prompt_sha256"],
        })

    eligible = [r for r in rows if r["arm"] == "eligible"]
    escalated = [r for r in rows if r["arm"] == "escalated"]
    if (len(eligible), len(escalated)) != (33, 21):
        _fatal("Arm split is {a}/{b}, expected 33/21."
               .format(a=len(eligible), b=len(escalated)))

    # ---- PRIMARY ---------------------------------------------------------- #
    _banner("PRIMARY -- Gemini vs the 2B human label, 33 eligible tickets")
    cells = two_by_two(eligible, "gemini_verdict")
    n_ung = cells["insufficient_and_ungrounded"] + cells["sufficient_and_ungrounded"]
    n_gnd = cells["insufficient_and_grounded"] + cells["sufficient_and_grounded"]
    print("                        human ungrounded (n=%d)   human grounded (n=%d)"
          % (n_ung, n_gnd))
    print("  rater INSUFFICIENT              %2d                        %2d"
          % (cells["insufficient_and_ungrounded"],
             cells["insufficient_and_grounded"]))
    print("  rater SUFFICIENT                %2d                        %2d"
          % (cells["sufficient_and_ungrounded"],
             cells["sufficient_and_grounded"]))
    print("\n  CAUGHT   : %d of %d human-labelled ungrounded drafts"
          % (cells["insufficient_and_ungrounded"], n_ung))
    print("  FLAGGED  : %d of %d human-labelled grounded drafts"
          % (cells["insufficient_and_grounded"], n_gnd))
    print("\n  NO rate, interval or test is reported on the %d-positive axis "
          "by design." % n_ung)

    lo, hi = wilson_interval(cells["insufficient_and_grounded"], n_gnd)
    ff_rate = cells["insufficient_and_grounded"] / n_gnd if n_gnd else float("nan")
    print("  False-flag proportion on the %d grounded: %d/%d = %.3f, "
          "Wilson 95%% [%.3f, %.3f]"
          % (n_gnd, cells["insufficient_and_grounded"], n_gnd, ff_rate, lo, hi))

    # ---- every disagreement, quoted --------------------------------------- #
    _banner("Every disagreement, quoted in full")
    disagreements = []
    for r in eligible:
        human_says_bad = r["human_label"] != "grounded"
        rater_says_bad = r["gemini_verdict"] == "INSUFFICIENT"
        if human_says_bad == rater_says_bad:
            continue
        disagreements.append(r)
        print("\n  %s (%s / %s)  top_sim %.4f" % (
            r["item_id"], r["ticket_id"], r["groundedness_item_id"],
            r["top_similarity"]))
        print("    human : %s" % r["human_label"])
        print("    rater : %s" % r["gemini_verdict"])
        print("    reason: %s" % r["gemini_reason"])
        print("    draft declines (post-hoc): %s%s" % (
            r["draft_declines_posthoc"],
            ("  [" + r["decline_phrases_posthoc"] + "]")
            if r["decline_phrases_posthoc"] else ""))
    print("\n  total disagreements: %d of 33" % len(disagreements))

    # ---- POST-HOC declined-draft cross-tab -------------------------------- #
    _banner("POST-HOC -- false flags split by whether the 2B draft declined")
    grounded = [r for r in eligible if r["human_label"] == "grounded"]
    tab = {}
    for declined in (True, False):
        grp = [r for r in grounded if r["draft_declines_posthoc"] is declined]
        flagged = [r for r in grp if r["gemini_verdict"] == "INSUFFICIENT"]
        tab["declined" if declined else "not_declined"] = {
            "n": len(grp), "flagged_insufficient": len(flagged),
            "rate": (len(flagged) / len(grp)) if grp else None,
        }
        print("  draft %-12s n=%2d   flagged INSUFFICIENT %2d%s"
              % ("declines" if declined else "does not",
                 len(grp), len(flagged),
                 ("   (%.3f)" % (len(flagged) / len(grp))) if grp else ""))
    print("\n  POST-HOC and exploratory. If the false flags concentrate on "
          "declined drafts,\n  the rater and the human label are answering "
          "DIFFERENT questions: 2B's rubric\n  says declining is not an "
          "unsupported claim, so such a draft is 'grounded'\n  even when its "
          "context is genuinely insufficient. Reported BESIDE the primary.")

    # ---- SECONDARY: the 21 the gate already escalates --------------------- #
    _banner("SECONDARY -- agreement with the live RAG gate, 21 escalated")
    agree = [r for r in escalated if r["gemini_verdict"] == "INSUFFICIENT"]
    print("  the gate says insufficient for all 21 (top1 < %.2f)"
          % settings.rag.similarity_threshold)
    print("  rater agrees          : %d of 21" % len(agree))
    lo21, hi21 = wilson_interval(len(agree), 21)
    print("  proportion            : %.3f, Wilson 95%% [%.3f, %.3f]"
          % (len(agree) / 21, lo21, hi21))
    for r in escalated:
        if r["gemini_verdict"] != "INSUFFICIENT":
            print("  DISAGREES: %s (%s) top_sim %.4f -- %s"
                  % (r["item_id"], r["ticket_id"], r["top_similarity"],
                     r["gemini_reason"]))

    # ---- SECONDARY: cross-family ------------------------------------------ #
    qwen_block = None
    if qwen:
        _banner("SECONDARY -- cross-family agreement, Gemini vs Qwen2.5-3B")
        both = [r for r in rows if r["qwen_verdict"]]
        n_agree = sum(1 for r in both if r["arms_agree"])
        qcells = two_by_two([r for r in eligible if r["qwen_verdict"]],
                            "qwen_verdict")
        qbad = sum(1 for r in rows if not r["qwen_verdict"])
        print("  overall agreement     : %d/%d = %.3f"
              % (n_agree, len(both), n_agree / len(both) if both else float("nan")))
        print("  qwen unparseable      : %d of 54" % qbad)
        print("  qwen INSUFFICIENT     : %d of %d rated"
              % (sum(1 for r in both if r["qwen_verdict"] == "INSUFFICIENT"),
                 len(both)))
        print("  qwen primary 2x2      : caught %d of %d ungrounded, "
              "flagged %d of %d grounded"
              % (qcells["insufficient_and_ungrounded"],
                 qcells["insufficient_and_ungrounded"]
                 + qcells["sufficient_and_ungrounded"],
                 qcells["insufficient_and_grounded"],
                 qcells["insufficient_and_grounded"]
                 + qcells["sufficient_and_grounded"]))
        qwen_block = {
            "model": qwen_doc["_meta"]["model"],
            "model_digest": qwen_doc["_meta"].get("model_digest"),
            "overall_agreement_n": n_agree,
            "overall_agreement_of": len(both),
            "unparseable": qbad,
            "insufficient_count": sum(1 for r in both
                                      if r["qwen_verdict"] == "INSUFFICIENT"),
            "primary_2x2": qcells,
            "prompt_identity_54_54": not hash_mismatch,
        }

    # ---- SECONDARY: stability --------------------------------------------- #
    _banner("SECONDARY -- verdict stability on the two positives")
    rep_path = verdicts_path("gemini", GEMINI_MODEL, ".repeats")
    stability = None
    if os.path.isfile(rep_path):
        rep_doc = load_json(rep_path, "the stability repeats")
        by_item = {}
        for r in rep_doc["records"]:
            by_item.setdefault(r["item_id"], {})[r["repeat"]] = r["verdict"]
        stability = {}
        differing = []
        for sid, reps in sorted(by_item.items()):
            first = reps.get(1)
            same = all(v == first for v in reps.values())
            stability[sid] = {"reps": reps, "primary_rep1": first,
                              "all_identical": same}
            print("  %s  rep1=%s  reps=%s  %s"
                  % (sid, first, [reps[k] for k in sorted(reps)],
                     "stable" if same else "DIFFERS"))
            if not same:
                differing.append(sid)
        if differing:
            print("\n  DETERMINISM FINDING: %s differ across repeats at "
                  "temperature 0.0.\n  The primary is UNCHANGED -- it is rep 1 "
                  "by pre-registration, never a majority." % differing)
        else:
            print("\n  No determinism finding: every repeat matches its rep 1.")
        print("  The primary would be rep 1 either way; repeats never revise "
              "it.")
    else:
        print("  [warn] no repeats file; the stability secondary is not run.")

    # ---- write ------------------------------------------------------------ #
    fields = ["item_id", "arm", "ticket_id", "groundedness_item_id",
              "top_similarity", "escalation_reason", "human_label",
              "human_hedge_appropriate", "context_distinct_fixes_posthoc",
              "gemini_verdict", "gemini_reason", "qwen_verdict", "qwen_reason",
              "arms_agree", "draft_declines_posthoc",
              "decline_phrases_posthoc", "prompt_sha256"]
    with open(OUT_CSV, "w", encoding="utf-8", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=fields)
        w.writeheader()
        for r in rows:
            w.writerow(r)

    n_cache = len([f for f in os.listdir(RAW_DIR)
                   if f.endswith(".json")]) if os.path.isdir(RAW_DIR) else 0
    gem_cache = len([f for f in os.listdir(RAW_DIR)
                     if f.startswith("gemini__")]) if os.path.isdir(RAW_DIR) else 0

    summary = {
        "phase": "6C",
        "question": ("Does a rater that sees only the ticket and its retrieved "
                     "context flag the cases the 0.67 similarity gate let "
                     "through?"),
        "config_fingerprint": config_fingerprint(),
        "production_unchanged": {
            "cascade": settings.cascade.confidence_threshold,
            "rag": settings.rag.similarity_threshold,
            "clustering": settings.clustering.resolution_similarity_threshold,
            "conformal_enabled": settings.conformal.enabled,
            "drift_enabled": settings.drift.enabled,
        },
        "population": {"eligible": 33, "escalated": 21, "total": 54},
        "primary": {
            "rule": ("rep 1 only; counts not rates; no majority vote over "
                     "repeats"),
            "two_by_two": cells,
            "caught_of_ungrounded": [cells["insufficient_and_ungrounded"], n_ung],
            "flagged_of_grounded": [cells["insufficient_and_grounded"], n_gnd],
            "false_flag_proportion": ff_rate,
            "false_flag_wilson95": [lo, hi],
            "no_rate_on_positive_axis": True,
            "disagreements": [
                {"item_id": r["item_id"], "ticket_id": r["ticket_id"],
                 "groundedness_item_id": r["groundedness_item_id"],
                 "top_similarity": r["top_similarity"],
                 "human_label": r["human_label"],
                 "rater_verdict": r["gemini_verdict"],
                 "rater_reason": r["gemini_reason"],
                 "draft_declines_posthoc": r["draft_declines_posthoc"]}
                for r in disagreements
            ],
        },
        "secondary_rag_gate_agreement": {
            "agree": len(agree), "of": 21,
            "wilson95": [lo21, hi21],
        },
        "secondary_cross_family": qwen_block,
        "secondary_stability": stability,
        "post_hoc": {
            "declined_draft_crosstab": tab,
            "declined_detector_phrases": list(DECLINE_PHRASES),
            "note": ("Exploratory. Reported beside the primary, never in place "
                     "of it. 2B's rubric states that declining is not an "
                     "unsupported claim, so a declining draft is labelled "
                     "grounded even when its context is insufficient -- if the "
                     "false flags concentrate there, the two measures are "
                     "answering different questions."),
            "context_distinct_fixes_source": ("reused from "
                                              "groundedness_results.csv, not "
                                              "recomputed"),
        },
        "rejected_hypothesis": {
            "claim": ("The rater's aggressiveness is explained by "
                      "near-duplicate retrieved neighbours leaving nothing to "
                      "distinguish."),
            "verdict": "REJECTED",
            "why": ("It contradicts Phase 2B's own diagnostic on these same "
                    "tickets: 71.1% of benchmark-45 and 88.9% of "
                    "adversarial-9 retrievals contain 2+ DISTINCT fixes, so "
                    "the context is heterogeneous."),
            "twob_distinct_fix_rates": TWOB_DISTINCT_FIX_RATES,
            "consequence": ("6C is NOT claimed as an instance of the project's "
                            "named finding on this basis."),
        },
        "provenance": {
            "dry_run_included_one_positive": True,
            "detail": ("The --limit 3 dry run happened to include one of the "
                       "two human-labelled positives (S002); the other landed "
                       "at S029, outside the window. Its primary verdict was "
                       "therefore seen before the full pass was launched. "
                       "Nothing was revised on seeing it: rubric, output "
                       "schema, population and primary rule were all fixed "
                       "beforehand and the prompt hash is cached."),
            "probability_at_least_one_of_two_in_first_three_of_54": 0.109,
            "shuffle_audit": ("seed-42 permutation independently re-derived "
                              "from benchmark order + 2B's escalated_ids; "
                              "identical for all 54; arms interleaved"),
            "gemini_calls_spent": 58,
            "gemini_cache_files": gem_cache,
            "total_cache_files": n_cache,
        },
    }
    with open(OUT_JSON, "w", encoding="utf-8") as fh:
        json.dump(summary, fh, indent=2, ensure_ascii=False)

    _banner("DONE")
    print("[write] " + OUT_CSV)
    print("[write] " + OUT_JSON)


if __name__ == "__main__":
    main()
