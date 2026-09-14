# src/experiments/score_groundedness_set.py
"""
score_groundedness_set.py
=========================

Phase 2B, step 3: score the groundedness set. Offline, no Gemini calls.

THREE OUTCOMES
--------------
1. GROUNDEDNESS RATE (primary). The human labels are the ground truth.

2. HEDGE APPROPRIATENESS (secondary, and free). The resolver prompt instructs
   the drafter to close with a note saying either that the retrieved examples
   closely match or that a human should verify. That is a calibration
   behaviour, it is already in every draft, and the retrieved context's
   heterogeneity is computable offline -- so we can ask directly whether the
   model hedges more when its retrieved context actually disagrees with
   itself. Zero extra calls, zero extra labelling.

3. JUDGE-HUMAN AGREEMENT (methodological). With the human labelling all 33,
   the LLM judge is not saving labour; it is the object of study. Reported as
   raw agreement and Cohen's kappa.

WHAT A NULL RESULT MEANS HERE
------------------------------
If every draft is labelled "grounded", that is a legitimate result, not a
failed run -- but it only means something because the Phase 2B pre-flight
diagnostic established that the outcome is not structurally forced. On the
novel-45 benchmark, 71% of retrievals return genuinely competing fixes and
40% span more than one category, so the drafter HAD room to be ungrounded.
A uniform pass is therefore evidence about the resolver, not an artifact of
the corpus -- which is exactly the distinction Phase 2A could not draw, and
the reason the diagnostic ran before any quota was spent.

HONEST LIMITS, STATED BEFORE THE NUMBERS
-----------------------------------------
n = 33 gives a proportion a +/- 2 s.d. band of roughly +/- 17 points at
p = 0.5, tightening to about +/- 10 near p = 0.9. Enough to detect gross
ungroundedness; not enough to resolve fine differences. Kappa on 33 items is
similarly wide. The adversarial-9 contributes only 3 drafts -- structurally
the most valuable cases, but nothing can be claimed about them alone.
"""

from __future__ import annotations

import csv
import json
import math
import os
import sys
from collections import Counter, defaultdict

_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(_THIS_DIR, os.pardir, os.pardir))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from src.experiments.build_groundedness_set import RUBRIC_LABELS

DATA_DIR = os.path.join(PROJECT_ROOT, "data")
SET_PATH = os.path.join(DATA_DIR, "groundedness_set.json")
KEY_PATH = os.path.join(DATA_DIR, "groundedness_key.json")
JUDGE_PATH = os.path.join(DATA_DIR, "groundedness_judge.json")
OUTPUT_CSV = os.path.join(DATA_DIR, "groundedness_results.csv")

# Production resolution-clustering config, reused to measure how much the
# retrieved resolutions disagree with each other. Same instrument, same
# calibrated threshold, same purpose as in the pre-flight diagnostic.
RESOLUTION_MODEL = "all-MiniLM-L6-v2"
RESOLUTION_THRESHOLD = 0.80


def die(message):
    print("\n[ERROR] " + message + "\n", file=sys.stderr)
    sys.exit(1)


def load_or_die(path, what, how):
    if not os.path.exists(path):
        die("Missing " + what + ":\n    " + path + "\nProduce it with:\n    "
            + how)
    return json.load(open(path, encoding="utf-8"))


def wilson_interval(k, n, z=1.96):
    """Wilson score interval -- honest at small n, unlike normal approximation."""
    if n == 0:
        return (0.0, 1.0)
    p = k / n
    d = 1 + z * z / n
    centre = (p + z * z / (2 * n)) / d
    half = (z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n))) / d
    return (max(0.0, centre - half), min(1.0, centre + half))


def cohens_kappa(a, b, labels):
    """Cohen's kappa for two raters over the same items."""
    n = len(a)
    if n == 0:
        return float("nan")
    agree = sum(1 for x, y in zip(a, b) if x == y) / n
    ca, cb = Counter(a), Counter(b)
    expected = sum((ca[l] / n) * (cb[l] / n) for l in labels)
    if expected >= 1.0:
        return float("nan")
    return (agree - expected) / (1 - expected)


def union_find_groups(sim, thr):
    n = sim.shape[0]
    parent = list(range(n))

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    for i in range(n):
        for j in range(i + 1, n):
            if sim[i, j] >= thr:
                ra, rb = find(i), find(j)
                if ra != rb:
                    parent[ra] = rb
    return len({find(i) for i in range(n)})


def main():
    print("=" * 74)
    print("Phase 2B / step 3 -- score the groundedness set")
    print("=" * 74)
    print("Gemini calls: 0")

    set_doc = load_or_die(
        SET_PATH, "labelling set",
        "python src/experiments/build_groundedness_set.py")
    key_doc = load_or_die(
        KEY_PATH, "provenance key",
        "python src/experiments/build_groundedness_set.py")

    items = {i["item_id"]: i for i in set_doc.get("items", [])}
    keys = {k["item_id"]: k for k in key_doc.get("items", [])}

    if set(items) != set(keys):
        die("Labelling set and provenance key describe different items.\n"
            "  only in set: " + str(sorted(set(items) - set(keys))[:8])
            + "\n  only in key: " + str(sorted(set(keys) - set(items))[:8])
            + "\nRegenerate both together.")

    if set_doc.get("_meta", {}).get("dry_run"):
        die("This set is marked dry_run -- it holds only the prompt-check "
            "drafts, not the full population. Re-run "
            "build_groundedness_set.py without --limit before scoring.")

    fps = {k.get("config_fingerprint") for k in keys.values()}
    if len(fps) > 1:
        die("Items carry more than one config fingerprint: " + str(sorted(fps))
            + "\nThey were not all produced by the same configuration.")
    print("config fingerprint: " + str(next(iter(fps))))

    # ---- completeness -------------------------------------------------------
    print("\n" + "=" * 74)
    print("STEP 1 -- label completeness")
    print("=" * 74)
    missing = [i for i, it in items.items() if not it.get("label")]
    bad = [(i, it.get("label")) for i, it in items.items()
           if it.get("label") and it["label"] not in RUBRIC_LABELS]
    no_hedge = [i for i, it in items.items()
                if it.get("label") and not isinstance(
                    it.get("hedge_appropriate"), bool)]

    print("[labels] items            : %d" % len(items))
    print("[labels] labelled         : %d" % (len(items) - len(missing)))
    print("[labels] missing label    : %d" % len(missing))
    print("[labels] missing hedge    : %d" % len(no_hedge))

    if bad:
        die("Invalid label value(s): " + str(bad[:8])
            + "\nValid labels are exactly: " + str(RUBRIC_LABELS))
    if missing:
        die("%d item(s) still unlabelled, e.g. %s\n"
            "Scoring a partially-labelled set would bias the result toward "
            "whichever drafts were easiest to judge.\nFinish labelling\n    %s"
            % (len(missing), sorted(missing)[:8], SET_PATH))
    if no_hedge:
        die("%d item(s) have a label but no boolean hedge_appropriate, e.g. "
            "%s\nSet it to true or false on every item."
            % (len(no_hedge), sorted(no_hedge)[:8]))

    # ---- context heterogeneity (offline, reuses the production instrument) --
    print("\n" + "=" * 74)
    print("STEP 2 -- retrieved-context heterogeneity")
    print("=" * 74)
    print("Instrument: %s @ %.2f, the production resolution-clustering config."
          % (RESOLUTION_MODEL, RESOLUTION_THRESHOLD))
    try:
        import numpy as np
        from sentence_transformers import SentenceTransformer
        model = SentenceTransformer(RESOLUTION_MODEL)
    except Exception as exc:  # noqa: BLE001
        die("Could not load %s for the heterogeneity measure (%s: %s).\n"
            "This step is offline and needs no API key -- check the venv."
            % (RESOLUTION_MODEL, type(exc).__name__, exc))

    heterogeneity = {}
    for iid, it in items.items():
        texts = [r["resolution"] for r in it["retrieved_resolutions"]
                 if (r.get("resolution") or "").strip()]
        if len(texts) < 2:
            heterogeneity[iid] = 1
            continue
        emb = model.encode(texts, normalize_embeddings=True,
                           show_progress_bar=False)
        heterogeneity[iid] = union_find_groups(emb @ emb.T,
                                               RESOLUTION_THRESHOLD)
    hdist = Counter(heterogeneity.values())
    print("  distinct fixes in retrieved context: "
          + ", ".join("%d: %d item(s)" % (k, hdist[k]) for k in sorted(hdist)))

    # ---- primary: groundedness ---------------------------------------------
    print("\n" + "=" * 74)
    print("STEP 3 -- PRIMARY: groundedness rate (human labels)")
    print("=" * 74)
    counts = Counter(it["label"] for it in items.values())
    n = len(items)
    for lab in RUBRIC_LABELS:
        k = counts.get(lab, 0)
        lo, hi = wilson_interval(k, n)
        print("  %-20s %3d / %d  = %5.1f%%   95%% CI [%.1f%%, %.1f%%]"
              % (lab, k, n, 100.0 * k / n, 100.0 * lo, 100.0 * hi))

    fully = counts.get("grounded", 0)
    lo, hi = wilson_interval(fully, n)
    print("\n  fully grounded: %d/%d = %.1f%%  (Wilson 95%% CI %.1f-%.1f%%)"
          % (fully, n, 100.0 * fully / n, 100.0 * lo, 100.0 * hi))
    if fully == n:
        print("\n  Every draft is grounded. Per the pre-flight diagnostic this")
        print("  is NOT structurally forced -- 71% of novel-45 retrievals")
        print("  return competing fixes -- so it is evidence about the")
        print("  resolver, not an artifact of the corpus.")

    # by source and by context heterogeneity
    by_source = defaultdict(Counter)
    by_het = defaultdict(Counter)
    for iid, it in items.items():
        by_source[keys[iid]["source"]][it["label"]] += 1
        by_het["1 fix" if heterogeneity[iid] == 1 else "2+ fixes"][it["label"]] += 1

    print("\n  by benchmark:")
    for src in sorted(by_source):
        c = by_source[src]
        tot = sum(c.values())
        print("    %-16s n=%2d  grounded=%d partial=%d ungrounded=%d"
              % (src, tot, c.get("grounded", 0),
                 c.get("partially_grounded", 0), c.get("ungrounded", 0)))

    print("\n  by retrieved-context heterogeneity:")
    for hk in sorted(by_het):
        c = by_het[hk]
        tot = sum(c.values())
        print("    %-16s n=%2d  grounded=%d partial=%d ungrounded=%d"
              % (hk, tot, c.get("grounded", 0),
                 c.get("partially_grounded", 0), c.get("ungrounded", 0)))
    print("  (descriptive only -- n=%d was sized for one rate, not a split)" % n)

    # ---- secondary: hedge appropriateness ----------------------------------
    print("\n" + "=" * 74)
    print("STEP 4 -- SECONDARY: hedge appropriateness vs context disagreement")
    print("=" * 74)
    hedge_ok = sum(1 for it in items.values() if it["hedge_appropriate"])
    lo, hi = wilson_interval(hedge_ok, n)
    print("  hedge appropriate: %d/%d = %.1f%%  (95%% CI %.1f-%.1f%%)"
          % (hedge_ok, n, 100.0 * hedge_ok / n, 100.0 * lo, 100.0 * hi))

    split = defaultdict(lambda: [0, 0])
    for iid, it in items.items():
        bucket = "1 fix" if heterogeneity[iid] == 1 else "2+ fixes"
        split[bucket][0] += 1 if it["hedge_appropriate"] else 0
        split[bucket][1] += 1
    print("\n  does the drafter hedge correctly when its context disagrees?")
    for bucket in sorted(split):
        ok, tot = split[bucket]
        print("    context %-9s  appropriate %2d/%2d = %5.1f%%"
              % (bucket, ok, tot, 100.0 * ok / tot if tot else 0.0))

    # ---- methodological: judge vs human ------------------------------------
    print("\n" + "=" * 74)
    print("STEP 5 -- METHODOLOGICAL: judge-human agreement")
    print("=" * 74)
    if not os.path.exists(JUDGE_PATH):
        print("  No judge file at " + JUDGE_PATH)
        print("  Run: python src/experiments/run_groundedness_judge.py")
        judge = {}
    else:
        jdoc = json.load(open(JUDGE_PATH, encoding="utf-8"))
        if jdoc.get("_meta", {}).get("dry_run"):
            print("  [warn] judge file is marked dry_run -- partial coverage.")
        judge = {v["item_id"]: v for v in jdoc.get("verdicts", [])}
        print("  judged items: %d of %d" % (len(judge), n))

    shared = sorted(set(judge) & set(items))
    if shared:
        h = [items[i]["label"] for i in shared]
        j = [judge[i]["label"] for i in shared]
        raw = sum(1 for x, y in zip(h, j) if x == y) / len(shared)
        kappa = cohens_kappa(h, j, RUBRIC_LABELS)
        print("\n  groundedness label")
        print("    raw agreement : %d/%d = %.1f%%"
              % (sum(1 for x, y in zip(h, j) if x == y), len(shared),
                 100.0 * raw))
        print("    Cohen's kappa : %.3f" % kappa)
        if all(x == h[0] for x in h) or all(y == j[0] for y in j):
            print("    NOTE: one rater used a single label throughout, so")
            print("    kappa is degenerate here -- chance agreement is ~1 and")
            print("    the statistic is uninformative. Read the raw agreement")
            print("    and say so explicitly rather than quoting kappa.")

        hh = [items[i]["hedge_appropriate"] for i in shared]
        jj = [judge[i]["hedge_appropriate"] for i in shared]
        raw_h = sum(1 for x, y in zip(hh, jj) if x == y) / len(shared)
        print("\n  hedge_appropriate")
        print("    raw agreement : %d/%d = %.1f%%"
              % (sum(1 for x, y in zip(hh, jj) if x == y), len(shared),
                 100.0 * raw_h))
        print("    Cohen's kappa : %.3f"
              % cohens_kappa([str(x) for x in hh], [str(y) for y in jj],
                             ["True", "False"]))

        disagreements = [i for i in shared
                         if items[i]["label"] != judge[i]["label"]]
        if disagreements:
            print("\n  items where judge and human disagree:")
            for i in disagreements:
                print("    %s  human=%-18s judge=%-18s"
                      % (i, items[i]["label"], judge[i]["label"]))
                print("        judge reason: %s"
                      % (judge[i].get("reason", "") or "")[:90])

        print("\n  Reminder: the human labels are the ground truth. The judge")
        print("  is the object of study -- these numbers describe the judge,")
        print("  never the resolver.")

    # ---- CSV ----------------------------------------------------------------
    print("\n" + "=" * 74)
    print("Writing CSV report")
    print("=" * 74)
    fields = ["item_id", "source", "ticket_id", "predicted_category",
              "top_similarity", "context_distinct_fixes", "human_label",
              "human_hedge_appropriate", "judge_label",
              "judge_hedge_appropriate", "agreement", "unsupported_span",
              "label_reason"]
    with open(OUTPUT_CSV, "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=fields)
        w.writeheader()
        for iid in sorted(items):
            it, k = items[iid], keys[iid]
            jv = judge.get(iid)
            w.writerow({
                "item_id": iid,
                "source": k["source"],
                "ticket_id": k["ticket_id"],
                "predicted_category": k.get("predicted_category", ""),
                "top_similarity": k.get("top_similarity", ""),
                "context_distinct_fixes": heterogeneity[iid],
                "human_label": it["label"],
                "human_hedge_appropriate": it["hedge_appropriate"],
                "judge_label": jv["label"] if jv else "",
                "judge_hedge_appropriate": (jv["hedge_appropriate"]
                                            if jv else ""),
                "agreement": ("" if not jv
                              else str(jv["label"] == it["label"])),
                "unsupported_span": it.get("unsupported_span", ""),
                "label_reason": it.get("label_reason", ""),
            })
    print("[write] %d row(s) -> %s" % (len(items), OUTPUT_CSV))

    print("\n" + "=" * 74)
    print("DONE")
    print("=" * 74)


if __name__ == "__main__":
    main()
