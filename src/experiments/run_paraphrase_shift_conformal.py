# src/experiments/run_paraphrase_shift_conformal.py
"""
Phase 7C, step 2 -- Finding 1 under the shift it was actually measured under.

THE QUESTION. Phase 7B applied a VERSION SHIFT within one generator and found
no contrast between the tiers. Finding 1 was measured under a PARAPHRASE /
REGISTER SHIFT, and TF-IDF's failure mechanism is SURFACE-VOCABULARY CHANGE, so
7B may never have exercised the mechanism it was testing. This script scores
the paired paraphrase set built by paraphrase_external_tickets.py and asks the
question again, like for like.

MEASUREMENT ONLY. Offline, zero Gemini calls. Nothing is promoted.

-----------------------------------------------------------------------------
PRE-REGISTRATION -- fixed before any result was seen
-----------------------------------------------------------------------------
PRIMARY: the TIER-1 MINUS TIER-2 COVERAGE-GAP DIFFERENCE at alpha = 0.10 on the
PARAPHRASED arm, with a 10,000-draw paired bootstrap at seed 42 over tickets.
    * REPLICATES if that difference is NEGATIVE with a bootstrap CI excluding
      zero -- Tier-1 loses more coverage than Tier-2. Reference magnitude on
      our corpus: -0.2222 (Tier-1 -0.2333, Tier-2 -0.0111).
    * DOES NOT REPLICATE if the CI includes zero.
Both gaps are measured against the same nominal 1 - alpha, so their difference
is exactly the coverage difference; the bootstrap resamples TICKETS, holding
the calibration quantile fixed, because the calibration set is not what varies
between the two arms.

PRE-REGISTERED SECONDARY (registered here, therefore NOT post-hoc): the same
quantity on the UNPARAPHRASED ORIGINALS of the same tickets, and the
DIFFERENCE-IN-DIFFERENCES between the arms. Finding 1 predicts approximately
zero on originals and negative on paraphrases. Reported always, headline never.

PRE-REGISTERED MANIPULATION CHECK. A null means nothing if the paraphrases did
not actually change register, so the manipulation is measured in BOTH spaces
before the result is read:
    * TF-IDF SPACE is where the mechanism lives. Surface similarity must DROP.
    * BGE SPACE is semantic preservation; similarity should stay HIGH.
Reported as mean cosine and as cross-fitted domain AUC, the latter directly
comparable to 7B's 0.8472 and 6B's figures.

DEGENERACY RULE, same as 6A/6B/7B -- name the condition, report "no
resolution", never substitute a metric that happens to resolve:
    * either tier's conformal calibration degenerate -> `degenerate`;
    * surviving paired n < 150            -> `insufficient_n_after_guards`;
    * TF-IDF-space domain AUC < 0.60      -> `manipulation_failed`;
    * TF-IDF-space domain AUC >= 0.95     -> BLOCKED, the 6B rule.

POST-HOC: anything added after results were seen carries post_hoc=True.

-----------------------------------------------------------------------------
TWO THINGS THIS SCRIPT DOES DIFFERENTLY, DELIBERATELY
-----------------------------------------------------------------------------
1. THE BAND IS COMBINED, NOT JUST THE CALIBRATION TERM. Finding 1 and 7B quote
   coverage_sd(alpha, n_cal) alone, which is right when the test arm is large.
   Here the test arm is ~300 and the TEST-SAMPLING term DOMINATES: at
   alpha=0.10, sqrt(.9*.1/300) = 0.0173 against the calibration term's 0.0082.
   Quoting the calibration term alone would understate uncertainty by about a
   factor of two, so both are reported and the combined band is what the
   verdict reads.

2. 7B'S MODELS ARE RE-USED UNCHANGED, AND THAT IS PROVED RATHER THAN ASSERTED.
   Tier-1 and Tier-2 are refit from the 7B manifest's train_row_indices, and
   the refit MUST reproduce 7B's cached test probabilities exactly (max |delta|
   = 0). A non-zero delta is fatal. This is preferred to persisting .joblib
   files that could later be mistaken for production artifacts -- and it is a
   stronger guarantee than loading a file, because it re-derives the model and
   checks it against a recorded output.

Run from the project root (offline, no Gemini calls, ~3 min):
    python src/experiments/run_paraphrase_shift_conformal.py
    python src/experiments/run_paraphrase_shift_conformal.py --force
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys

import numpy as np
import pandas as pd

THIS_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(THIS_DIR, os.pardir, os.pardir))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from src.agent import conformal as cp                              # noqa: E402
from src.agent.config import settings, config_fingerprint          # noqa: E402
from src.agent.logging_setup import ensure_utf8_console            # noqa: E402
from src.experiments.calibrate_conformal import (                  # noqa: E402
    _banner,
    _fatal,
    coverage_sd,
    tier1_probabilities,
)
from src.experiments.fetch_external_dataset import (               # noqa: E402
    ENGLISH_CSV,
    OUT_DIR,
)
from src.experiments.profile_external_dataset import EMB_NPY       # noqa: E402
from src.experiments.run_external_conformal_shift import (         # noqa: E402
    ALPHAS,
    PROBS_NPZ,
    SCORE_FUNCTION,
    SPLITS_JSON,
    TIERS,
    build_texts,
    domain_auc,
    isolation_paths,
    snapshot,
    verify_isolation,
    write_csv,
)
from src.experiments.paraphrase_external_tickets import (          # noqa: E402
    PARAPHRASE_JSON,
    length_ratio_stats,
)

ensure_utf8_console()

RESULTS_CSV = os.path.join(OUT_DIR, "external_paraphrase_conformal.csv")
SUMMARY_JSON = os.path.join(OUT_DIR, "external_paraphrase_summary.json")
OUTPUTS = (RESULTS_CSV, SUMMARY_JSON)

PRIMARY_ALPHA = 0.10
SEED = 42
BOOTSTRAP_N = 10000
MIN_PAIRED_N = 150
MANIPULATION_FLOOR_AUC = 0.60
DEGENERACY_AUC = 0.95

# Finding 1's magnitude on our corpus, for scale only -- never a threshold.
OUR_CORPUS_T1_MINUS_T2_AT_010 = -0.2222


def paired_bootstrap_difference(covered_a, covered_b, n=BOOTSTRAP_N,
                                seed=SEED):
    """95% interval for mean(covered_a) - mean(covered_b), resampling TICKETS.

    Paired because both arms are the SAME tickets: the two coverage estimates
    are positively correlated, and treating them as independent would inflate
    the interval and hide an effect that is really there.
    """
    a = np.asarray(covered_a, dtype=bool)
    b = np.asarray(covered_b, dtype=bool)
    if a.shape != b.shape:
        raise ValueError("paired arrays differ in length")

    rng = np.random.default_rng(seed)
    m = len(a)
    idx = rng.integers(0, m, size=(n, m))
    diffs = a[idx].mean(axis=1) - b[idx].mean(axis=1)
    lo, hi = np.percentile(diffs, [2.5, 97.5])
    return float(diffs.mean()), float(lo), float(hi)


def paired_bootstrap_did(p_t1, p_t2, o_t1, o_t2, n=BOOTSTRAP_N, seed=SEED):
    """Difference-in-differences: (p_t1 - p_t2) - (o_t1 - o_t2).

    One resample index is shared by all four arrays, because all four describe
    the same tickets. Drawing four independent resamples would destroy the
    pairing the design exists to exploit.
    """
    arrays = [np.asarray(x, dtype=bool) for x in (p_t1, p_t2, o_t1, o_t2)]
    m = len(arrays[0])
    if any(len(x) != m for x in arrays):
        raise ValueError("DiD arrays differ in length")

    rng = np.random.default_rng(seed)
    idx = rng.integers(0, m, size=(n, m))
    means = [x[idx].mean(axis=1) for x in arrays]
    did = (means[0] - means[1]) - (means[2] - means[3])
    lo, hi = np.percentile(did, [2.5, 97.5])
    return float(did.mean()), float(lo), float(hi)


def covered_mask(probs, labels, classes, calibration):
    """Per-ticket boolean: was the true label in the prediction set?

    Returned as a MASK rather than a rate, because every statistic below is a
    paired resample over tickets and needs the per-ticket outcome, not the
    aggregate.
    """
    sets = cp.predict_sets(probs, calibration)
    mask = np.array([y in s for s, y in zip(sets, labels)], dtype=bool)

    # Rule 6: the rate, derived a second way, must agree.
    independent = cp.coverage(sets, list(labels))
    if abs(independent - float(mask.mean())) > 1e-12:
        _fatal("Coverage disagrees between two derivations: {a} vs {b}".format(
            a=independent, b=float(mask.mean())))
    return mask, sets


def refit_and_verify(df, emb, manifest, blob):
    """Re-derive 7B's two models and PROVE they are the same models.

    A refit that silently differed would compare paraphrases against a
    different classifier than the one 7B measured -- this project's recurring
    bug in its exact shape. So the refit is checked against 7B's recorded
    output rather than trusted because the seed matches.
    """
    from sklearn.linear_model import LogisticRegression
    from src.classification.train_cascade import train_tier1

    texts = build_texts(df)
    labels = df["queue"].astype(str).to_numpy()
    train_idx = np.asarray(manifest["train_row_indices"], dtype=np.int64)
    test_idx = np.asarray(manifest["test_row_indices"], dtype=np.int64)

    train_texts = [texts[i] for i in train_idx]
    train_labels = labels[train_idx].tolist()

    t1_vec, t1_clf = train_tier1(train_texts, train_labels)
    t2_clf = LogisticRegression(max_iter=1000)
    t2_clf.fit(emb[train_idx], train_labels)

    p1, classes1 = tier1_probabilities(t1_vec, t1_clf,
                                       [texts[i] for i in test_idx])
    p2 = t2_clf.predict_proba(emb[test_idx])

    d1 = float(np.max(np.abs(p1 - blob["tier1_test"])))
    d2 = float(np.max(np.abs(p2 - blob["tier2_test"])))
    if d1 != 0.0 or d2 != 0.0:
        _fatal("The refit does NOT reproduce 7B's models: max |delta| tier1 "
               "{a}, tier2 {b}. Scoring paraphrases against a different "
               "classifier than 7B measured would answer a different "
               "question.".format(a=d1, b=d2))
    if (list(classes1) != list(blob["tier1_classes"])
            or list(t2_clf.classes_) != list(blob["tier2_classes"])):
        _fatal("Refit class order differs from 7B's.")

    print("  [ok] refit reproduces 7B EXACTLY: max |delta| = {a} / {b}, "
          "classes identical".format(a=d1, b=d2))
    return t1_vec, t1_clf, t2_clf, list(classes1), list(t2_clf.classes_)


def run(force=False):
    _banner("PHASE 7C step 2 - FINDING 1 UNDER A PARAPHRASE SHIFT")
    print("Measurement only: conformal.enabled={c}, drift.enabled={d}".format(
        c=settings.conformal.enabled, d=settings.drift.enabled))
    if settings.conformal.enabled or settings.drift.enabled:
        _fatal("This phase is measurement only; both flags must stay False.")

    for path in OUTPUTS:
        if os.path.isfile(path) and not force:
            _fatal("Refusing to overwrite an existing result:\n  {p}\n"
                   "Pass --force if you mean to replace it.".format(p=path))

    for path in (PARAPHRASE_JSON, SPLITS_JSON, PROBS_NPZ, ENGLISH_CSV,
                 EMB_NPY):
        if not os.path.isfile(path):
            _fatal("Missing {p}\n  Run paraphrase_external_tickets.py first."
                   .format(p=os.path.relpath(path, PROJECT_ROOT)))

    before = snapshot(isolation_paths() + [
        p for p in (SPLITS_JSON, PARAPHRASE_JSON,
                    os.path.join(OUT_DIR, "external_conformal_designA.csv"),
                    os.path.join(OUT_DIR, "external_conformal_designB.csv"),
                    os.path.join(OUT_DIR, "external_deferral_results.csv"))
        if os.path.isfile(p)])
    print("  isolation baseline: {n} files hashed".format(n=len(before)))

    # ---- Step 1: inputs -------------------------------------------------- #
    _banner("STEP 1 - The paired paraphrase set")
    with open(PARAPHRASE_JSON, encoding="utf-8") as fh:
        pset = json.load(fh)
    pairs = pset["pairs"]
    print("  sampled {s} | unparseable {u} | not rewritten {d} | surviving {n}"
          .format(s=pset["n_sampled"], u=pset["n_unparseable"],
                  d=pset["n_discarded_not_rewritten"], n=len(pairs)))
    print("  mean source-paraphrase BGE similarity {m:.4f}".format(
        m=pset["mean_bge_similarity_to_source"]))

    insufficient = len(pairs) < MIN_PAIRED_N
    if insufficient:
        print("  [!] n = {n} is below the pre-registered floor of {f}: the "
              "verdict will be".format(n=len(pairs), f=MIN_PAIRED_N))
        print("      `insufficient_n_after_guards` and NO RESOLUTION.")

    df = pd.read_csv(ENGLISH_CSV, low_memory=False)
    emb = np.load(EMB_NPY)
    with open(SPLITS_JSON, encoding="utf-8") as fh:
        manifest = json.load(fh)
    blob = np.load(PROBS_NPZ, allow_pickle=False)

    # ---- Step 2: 7B's models, proved unchanged --------------------------- #
    _banner("STEP 2 - Re-deriving 7B's models and proving they are the same")
    t1_vec, t1_clf, t2_clf, classes1, classes2 = refit_and_verify(
        df, emb, manifest, blob)

    # ---- Step 3: score both arms ----------------------------------------- #
    _banner("STEP 3 - Scoring both arms (the SAME tickets, paired)")
    row_idx = np.asarray([p["row_index"] for p in pairs], dtype=np.int64)
    labels = np.asarray([p["queue"] for p in pairs])
    source_texts = [p["source"] for p in pairs]
    para_texts = [p["paraphrase"] for p in pairs]

    # Pairing check, before any statistic exists.
    corpus_labels = df["queue"].astype(str).to_numpy()
    if not np.array_equal(labels, corpus_labels[row_idx]):
        _fatal("The paraphrase set's labels disagree with the corpus.")
    corpus_texts = build_texts(df)
    if any(corpus_texts[i] != s for i, s in zip(row_idx, source_texts)):
        _fatal("The paraphrase set's source text disagrees with the corpus.")
    print("  [ok] {n} pairs, same tickets in both arms, labels and source "
          "text match the corpus".format(n=len(pairs)))

    from src.agent.artifacts import load_artifacts
    artifacts = load_artifacts(require_gemini=False)

    src_emb = emb[row_idx]
    para_emb = np.asarray(artifacts.embedder.encode(
        para_texts, convert_to_numpy=True, batch_size=32,
        show_progress_bar=False), dtype=np.float32)

    src_tfidf = t1_vec.transform(source_texts).toarray()
    para_tfidf = t1_vec.transform(para_texts).toarray()

    probs = {
        ("original", "tier1"): tier1_probabilities(
            t1_vec, t1_clf, source_texts)[0],
        ("paraphrased", "tier1"): tier1_probabilities(
            t1_vec, t1_clf, para_texts)[0],
        ("original", "tier2"): t2_clf.predict_proba(src_emb),
        ("paraphrased", "tier2"): t2_clf.predict_proba(para_emb),
    }
    classes = {"tier1": classes1, "tier2": classes2}

    for arm in ("original", "paraphrased"):
        for tier in TIERS:
            acc = float(np.mean(
                np.asarray(classes[tier])[
                    np.argmax(probs[(arm, tier)], axis=1)] == labels))
            print("  accuracy {a:12} {t}: {x:.4f}".format(
                a=arm, t=tier, x=acc))

    # ---- Step 4: manipulation check -------------------------------------- #
    _banner("STEP 4 - MANIPULATION CHECK (did the register actually move?)")
    print("  TF-IDF space is where TF-IDF's failure mechanism lives, so that")
    print("  is the space the check has to pass. BGE space is the control for")
    print("  meaning being preserved.")

    def mean_cosine(a, b):
        a = np.asarray(a, dtype=np.float64)
        b = np.asarray(b, dtype=np.float64)
        na = np.linalg.norm(a, axis=1)
        nb = np.linalg.norm(b, axis=1)
        ok = (na > 0) & (nb > 0)
        sims = np.zeros(len(a))
        sims[ok] = np.sum(a[ok] * b[ok], axis=1) / (na[ok] * nb[ok])
        return float(sims.mean()), float(np.median(sims))

    bge_mean, bge_median = mean_cosine(src_emb, para_emb)
    tfidf_mean, tfidf_median = mean_cosine(src_tfidf, para_tfidf)
    auc_bge = domain_auc(src_emb, para_emb)
    auc_tfidf = domain_auc(src_tfidf, para_tfidf)

    print("  BGE    : mean cosine {m:.4f} (median {d:.4f}) | domain AUC {a:.4f}"
          .format(m=bge_mean, d=bge_median, a=auc_bge))
    print("  TF-IDF : mean cosine {m:.4f} (median {d:.4f}) | domain AUC {a:.4f}"
          .format(m=tfidf_mean, d=tfidf_median, a=auc_tfidf))
    print("  (7B's version shift, for scale: BGE-space domain AUC 0.8472)")

    # Pre-registered secondary: a large length change could move coverage for
    # BOTH tiers independently of vocabulary -- shorter text means fewer
    # features fire and flatter probabilities -- so similarity alone does not
    # fully describe the manipulation.
    lengths = length_ratio_stats(source_texts, para_texts, artifacts.embedder)
    print("  LENGTH (pre-registered secondary), paraphrase / original:")
    for space in ("token", "word"):
        st = lengths.get(space)
        if st is None:
            print("    {s:6}: UNREADABLE".format(s=space))
            continue
        print("    {s:6}: mean {m:.4f}, median {d:.4f}, aggregate {a:.4f}  "
              "({x:.1f} -> {y:.1f})".format(
                  s=space, m=st["mean_ratio"], d=st["median_ratio"],
                  a=st["aggregate_ratio"], x=st["mean_source_length"],
                  y=st["mean_paraphrase_length"]))

    manipulation_failed = auc_tfidf < MANIPULATION_FLOOR_AUC
    manipulation_blocked = auc_tfidf >= DEGENERACY_AUC
    if manipulation_failed:
        print("  [!] TF-IDF AUC below {f}: the rewrite did not move surface "
              "vocabulary.".format(f=MANIPULATION_FLOOR_AUC))
        print("      Pre-registered outcome: MANIPULATION FAILED, no "
              "resolution.")
    elif manipulation_blocked:
        print("  [!] TF-IDF AUC >= {d}: the arms are near-separable and the "
              "shift is ill-posed.".format(d=DEGENERACY_AUC))
        print("      Pre-registered outcome: BLOCKED (the 6B rule).")
    else:
        print("  [ok] the manipulation moved surface vocabulary without "
              "destroying meaning.")

    # ---- Step 5: the Finding 1 table ------------------------------------- #
    _banner("STEP 5 - The Finding 1 table, both arms")
    cal_labels = list(blob["cal_labels"])
    n_cal = len(cal_labels)
    n_test = len(pairs)

    rows = []
    covered = {}
    for alpha in ALPHAS:
        for tier in TIERS:
            calib = cp.calibrate(blob["{t}_cal".format(t=tier)], cal_labels,
                                 classes[tier], alpha=alpha,
                                 score_function=SCORE_FUNCTION,
                                 mondrian=False)
            for arm in ("original", "paraphrased"):
                mask, sets = covered_mask(probs[(arm, tier)], labels,
                                          classes[tier], calib)
                covered[(arm, tier, alpha)] = mask

                coverage = float(mask.mean())
                gap = coverage - (1.0 - alpha)
                cal_sd = coverage_sd(alpha, n_cal)
                test_sd = math.sqrt(alpha * (1 - alpha) / n_test)
                combined = math.sqrt(cal_sd ** 2 + test_sd ** 2)
                sizes = np.array([len(s) for s in sets])

                rows.append({
                    "arm": arm,
                    "tier": tier,
                    "alpha": alpha,
                    "n_calibration": n_cal,
                    "n_test": n_test,
                    "degenerate": calib.is_degenerate,
                    "coverage": coverage,
                    "coverage_gap": gap,
                    "calibration_sd": cal_sd,
                    "test_sampling_sd": test_sd,
                    "combined_sd": combined,
                    "combined_band_2sd": 2.0 * combined,
                    "gap_outside_combined_band": abs(gap) > 2.0 * combined,
                    "calibration_only_band_2sd": 2.0 * cal_sd,
                    "mean_set_size": float(sizes.mean()),
                    "singleton_rate": float(np.mean(sizes == 1)),
                    "empty_rate": float(np.mean(sizes == 0)),
                    "score_function": SCORE_FUNCTION,
                    "post_hoc": False,
                })

    for arm in ("original", "paraphrased"):
        print("\n  arm: {a}".format(a=arm))
        print("    {t:6} {al:>6} {c:>10} {g:>10} {b:>10} {o:>5} {s:>8}".format(
            t="tier", al="alpha", c="coverage", g="gap", b="comb.band",
            o="out", s="set"))
        for r in rows:
            if r["arm"] != arm:
                continue
            print("    {t:6} {al:>6.2f} {c:>10.4f} {g:>+10.4f} {b:>10.4f} "
                  "{o:>5} {s:>8.3f}".format(
                      t=r["tier"], al=r["alpha"], c=r["coverage"],
                      g=r["coverage_gap"], b=r["combined_band_2sd"],
                      o=("OUT" if r["gap_outside_combined_band"] else "in"),
                      s=r["mean_set_size"]))

    # ---- Step 6: the primary statistic ----------------------------------- #
    _banner("STEP 6 - PRIMARY: Tier-1 minus Tier-2 gap difference, "
            "alpha = {a}".format(a=PRIMARY_ALPHA))
    p_t1 = covered[("paraphrased", "tier1", PRIMARY_ALPHA)]
    p_t2 = covered[("paraphrased", "tier2", PRIMARY_ALPHA)]
    o_t1 = covered[("original", "tier1", PRIMARY_ALPHA)]
    o_t2 = covered[("original", "tier2", PRIMARY_ALPHA)]

    prim_mean, prim_lo, prim_hi = paired_bootstrap_difference(p_t1, p_t2)
    sec_mean, sec_lo, sec_hi = paired_bootstrap_difference(o_t1, o_t2)
    did_mean, did_lo, did_hi = paired_bootstrap_did(p_t1, p_t2, o_t1, o_t2)

    prim_point = float(p_t1.mean() - p_t2.mean())
    sec_point = float(o_t1.mean() - o_t2.mean())

    print("  PARAPHRASED  T1 - T2 = {p:+.4f}  bootstrap {m:+.4f} "
          "[{lo:+.4f}, {hi:+.4f}]".format(
              p=prim_point, m=prim_mean, lo=prim_lo, hi=prim_hi))
    print("  original     T1 - T2 = {p:+.4f}  bootstrap {m:+.4f} "
          "[{lo:+.4f}, {hi:+.4f}]".format(
              p=sec_point, m=sec_mean, lo=sec_lo, hi=sec_hi))
    print("  DiD (paraphrased - original) = {m:+.4f} [{lo:+.4f}, {hi:+.4f}]"
          .format(m=did_mean, lo=did_lo, hi=did_hi))
    print("  our corpus, for scale: {v:+.4f}".format(
        v=OUR_CORPUS_T1_MINUS_T2_AT_010))

    degenerate = any(r["degenerate"] for r in rows)
    signal = (prim_lo > 0 or prim_hi < 0)

    if insufficient:
        verdict = "no_resolution_insufficient_n"
    elif degenerate:
        verdict = "no_resolution_degenerate_calibration"
    elif manipulation_failed:
        verdict = "no_resolution_manipulation_failed"
    elif manipulation_blocked:
        verdict = "blocked_auc_ge_0.95"
    elif signal and prim_point < 0:
        verdict = "replicates"
    elif signal and prim_point > 0:
        verdict = "signal_in_the_opposite_direction"
    else:
        verdict = "does_not_replicate"

    _banner("VERDICT: {v}".format(v=verdict.upper()))
    if verdict == "replicates":
        print("  Tier-1 loses more coverage than Tier-2 under a paraphrase")
        print("  shift on the external corpus. Finding 1's mechanism survives")
        print("  a change of generator when the SHIFT TYPE is held fixed.")
    elif verdict == "does_not_replicate":
        print("  No evidence that Tier-1 loses more coverage than Tier-2 under")
        print("  a paraphrase shift here. Note the wording: NO EVIDENCE OF A")
        print("  DIFFERENCE, never evidence of no difference.")
    elif verdict.startswith("no_resolution") or verdict.startswith("blocked"):
        print("  The pre-registered condition above fired. No verdict is")
        print("  drawn on the primary, and no secondary is promoted to fill")
        print("  the gap.")

    print("\n  LIMITATION, stated beside the result and not after it: the")
    print("  external labels are GENERATOR-ASSIGNED AND UNAUDITED and base")
    print("  accuracy is ~35%, so a null here is WEAKER evidence than a null")
    print("  on a well-learned task.")

    # ---- Step 7: write --------------------------------------------------- #
    _banner("STEP 7 - Writing results")
    summary = {
        "phase": "7C",
        "measurement_only": True,
        "is_real_production_data": False,
        "verdict": verdict,
        "config_fingerprint": config_fingerprint(),
        "seed": SEED,
        "bootstrap_draws": BOOTSTRAP_N,
        "primary_alpha": PRIMARY_ALPHA,
        "n_pairs": n_test,
        "n_calibration": n_cal,
        "min_paired_n": MIN_PAIRED_N,
        "primary_t1_minus_t2_paraphrased": prim_point,
        "primary_bootstrap_mean": prim_mean,
        "primary_ci_lo": prim_lo,
        "primary_ci_hi": prim_hi,
        "primary_signal": bool(signal),
        "secondary_t1_minus_t2_original": sec_point,
        "secondary_ci_lo": sec_lo,
        "secondary_ci_hi": sec_hi,
        "did_mean": did_mean,
        "did_ci_lo": did_lo,
        "did_ci_hi": did_hi,
        "our_corpus_reference": OUR_CORPUS_T1_MINUS_T2_AT_010,
        "manipulation": {
            "bge_mean_cosine": bge_mean,
            "bge_median_cosine": bge_median,
            "bge_domain_auc": auc_bge,
            "tfidf_mean_cosine": tfidf_mean,
            "tfidf_median_cosine": tfidf_median,
            "tfidf_domain_auc": auc_tfidf,
            "floor_auc": MANIPULATION_FLOOR_AUC,
            "failed": bool(manipulation_failed),
            "blocked": bool(manipulation_blocked),
            "reference_7b_version_shift_bge_auc": 0.8471736293799147,
            "length_ratios": lengths,
        },
        "paraphrase_set": {
            "model": pset["model"],
            "model_digest": pset.get("model_digest"),
            "n_sampled": pset["n_sampled"],
            "n_unparseable": pset["n_unparseable"],
            "n_discarded_not_rewritten": pset["n_discarded_not_rewritten"],
            "mean_bge_similarity_to_source":
                pset["mean_bge_similarity_to_source"],
        },
        "limitation": ("External queue labels are generator-assigned and "
                       "unaudited; base accuracy is ~35%, so a null here is "
                       "weaker evidence than a null on a well-learned task."),
    }
    write_csv(RESULTS_CSV, rows)
    with open(SUMMARY_JSON, "w", encoding="utf-8") as fh:
        json.dump(summary, fh, indent=2)
    print("[write] {p}".format(p=os.path.relpath(SUMMARY_JSON, PROJECT_ROOT)))

    _banner("STEP 8 - Isolation check")
    verify_isolation(before)
    return rows, summary


def parse_args(argv):
    parser = argparse.ArgumentParser(
        description="Phase 7C step 2 -- Finding 1 under a paraphrase shift.")
    parser.add_argument("--force", action="store_true",
                        help="overwrite existing 7C result files")
    return parser.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    run(force=args.force)
    return 0


if __name__ == "__main__":
    sys.exit(main())
