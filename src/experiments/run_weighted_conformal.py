"""
Phase 6B -- weighted conformal under shift, with a domain-classifier density
ratio.

THE QUESTION
------------
Does weighted conformal repair Tier-1's coverage collapse (Finding 1)?

Finding 1 measured that a conformal predictor calibrated on the in-domain 175
loses 23.3 coverage points on the 45-ticket benchmark for Tier-1, against a
+-2 s.d. noise band of ~0.045. Phase 1 Finding 4 then showed that DISTRIBUTION
MATCHING -- rebuilding the calibration set at deployment register, matched on
size and class balance -- recovers only ~38% of that shortfall (-0.233 ->
-0.144) and leaves it six times outside the band.

6B asks the successor question: can COVARIATE REWEIGHTING do what distribution
matching could not? Weighted split conformal (Tibshirani et al. 2019; Barber et
al. 2022, "Conformal prediction beyond exchangeability") reweights calibration
scores by the covariate likelihood ratio between the test and calibration
distributions. If the shift is a covariate shift, reweighting is the principled
repair; if it is not, the shortfall lives somewhere reweighting cannot reach.

MEASUREMENT ONLY. Offline, no Gemini calls, no quota. This script never changes
a production threshold, and it refuses to run if settings.conformal.enabled has
been flipped to True.

THE PRE-REGISTRATION -- FIXED BEFORE ANY RESULT WAS SEEN
--------------------------------------------------------
Written into this docstring, and into the CSV, before the first run. Phase 6A's
primary metric turned out to have no resolution in 3 of 4 configurations and
that was discovered AFTER the run; 6B states its terms up front.

ANSWERABILITY. 6B is NOT on 6A's wall, and the reason is specific. 6A's gated
axis was selective risk at LOW COVERAGE: at the live gate's operating point the
benchmark admitted 4 tickets. 6B's primary metric is MARGINAL COVERAGE OVER ALL
45 BENCHMARK TICKETS -- no coverage restriction, every ticket counts -- and
Finding 1's gap already resolved on exactly that axis.

  * Tier-1: ANSWERABLE. A repair must move the gap by more than the band.
  * Tier-2: NOT ANSWERABLE, declared here rather than discovered afterwards.
    Its unweighted gap is -0.011, already inside the band, so there is no
    headroom for an improvement to show. Tier-2 is a SANITY CHECK -- weighting
    must not break it -- and is never reported as a finding.

PRIMARY METRIC. Tier-1 benchmark-45 marginal coverage gap at alpha = 0.10,
weighted vs unweighted, judged against the +-2 s.d. band, which is recomputed
from coverage_sd(alpha, n_cal) rather than hardcoded.

AN AMBIGUITY IN THE PRE-REGISTERED WORDING, RESOLVED BY REPORTING BOTH READINGS
-------------------------------------------------------------------------------
The pre-registered interpretation clause read "weighting closes Tier-1's gap
beyond the noise band". That phrase admits two readings, and they do not agree
on this data:

  (a) the CHANGE exceeds the band            -> delta_outside_band
  (b) the RESIDUAL gap falls inside the band -> not gap_outside_band

Reading (a) alone would let a large improvement that still leaves coverage far
below nominal be written up as "correctable", which would overclaim. Both are
therefore reported side by side, both columns are in the CSV, and the verdict
states each separately. This is disclosed rather than silently resolved in
whichever direction flatters the result -- picking one after seeing the numbers
is the thing the pre-registration exists to prevent.

`recovery_fraction` is a POST-HOC descriptive statistic added after the results
were seen, flagged as such wherever it appears. It describes; it does not
decide. The verdict rests on the pre-registered band comparisons only.

SECONDARY. The same at alpha in {0.05, 0.20}; mean set size; singleton rate;
empty-set rate. Reported always, promoted to a headline never.

DEGENERACY RULE. Report "no resolution on the gated axis", naming the
triggering condition, if ANY of:
  1. n_eff < 50 for the weights used, where n_eff = (sum w)^2 / sum(w^2);
  2. cross-fitted domain-classifier AUC >= 0.95 -- the domains are then
     near-separable and the ratio is effectively unbounded;
  3. weighted and unweighted prediction sets are IDENTICAL on the benchmark.
When the primary has no resolution, NO averaged or secondary statistic is
promoted in its place. That is 6A's AURC lesson written into the rule.

Anything added after results were seen is labelled post-hoc, in the CSV and in
the write-up.

THE LIMITATION THAT CONDITIONS HOW A NULL READS -- stated up front
------------------------------------------------------------------
The target proxy is the deployment calibration set (175, used UNLABELED), but
the TEST set is the 45-ticket benchmark. Weighted conformal assumes the test
points are drawn from the target distribution the weights describe. If the
deployment set is an imperfect proxy for benchmark register, a null result
means "reweighting toward THIS target did not repair it", not "no reweighting
could". That distinction goes in the write-up whichever way the result lands.

Run from the project root:
    python src/experiments/run_weighted_conformal.py
    python src/experiments/run_weighted_conformal.py --force
"""

from __future__ import annotations

import argparse
import csv
import os
import sys

import numpy as np

THIS_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(THIS_DIR, os.pardir, os.pardir))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from src.agent import conformal as cp                              # noqa: E402
from src.agent.config import settings                              # noqa: E402
from src.agent.logging_setup import ensure_utf8_console            # noqa: E402
from src.experiments.calibrate_conformal import (                  # noqa: E402
    DEPLOYMENT_JSON,
    OUTPUT_CSV as PUBLISHED_CONFORMAL_CSV,
    _banner,
    _fatal,
    _read_json,
    coverage_sd,
    embed,
    evaluate,
    fit_scoring_models,
    load_everything,
    tier1_probabilities,
)

ensure_utf8_console()

DATA_DIR = os.path.join(PROJECT_ROOT, "data")
RESULTS_CSV = os.path.join(DATA_DIR, "weighted_conformal_results.csv")

# Pre-registered grid.
ALPHAS = (0.05, 0.10, 0.20)
PRIMARY_ALPHA = 0.10
SPACES = ("bge", "tfidf")
TIERS = ("tier1", "tier2")
CLIP_VARIANTS = (("none", None), ("p95", 95.0), ("p99", 99.0))
SCORE_FUNCTION = "lac"

# Pre-registered degeneracy thresholds.
MIN_EFFECTIVE_N = 50.0
MAX_DOMAIN_AUC = 0.95

# Cross-fitting.
N_FOLDS = 5
SEED = 42

# The published Finding 1 gaps the unweighted rows MUST reproduce exactly
# (in_domain / contaminated / label_filter=all / lac / marginal / alpha=0.10).
# A mismatch means the probabilities are not the ones the published results
# came from -- this project's recurring bug class -- so it is fatal.
PUBLISHED_FINDING1_GAP = {
    "tier1": -0.2333333333333334,
    "tier2": -0.011111111111111072,
}
GAP_TOLERANCE = 1e-9


# --------------------------------------------------------------------------- #
# Density ratio
# --------------------------------------------------------------------------- #
def cross_fitted_domain_probabilities(x_cal, x_target):
    """Out-of-fold p(target | x) for the calibration and target points, plus a
    full-data model for scoring the benchmark.

    Cross-fitting matters here: a domain classifier evaluated on its own
    training points is overconfident, which would inflate the tails of the
    density ratio and collapse n_eff for a reason that is an artifact of the
    fit rather than a property of the data. Every calibration point therefore
    gets a probability from a model that never saw it.

    The benchmark is not part of the domain-classifier training set at all, so
    it is scored with a model fit on all of it -- the standard split.
    """
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import StratifiedKFold

    X = np.vstack([np.asarray(x_cal, dtype=np.float64),
                   np.asarray(x_target, dtype=np.float64)])
    y = np.concatenate([np.zeros(len(x_cal), dtype=int),
                        np.ones(len(x_target), dtype=int)])

    oof = np.full(y.shape, np.nan, dtype=np.float64)
    splitter = StratifiedKFold(n_splits=N_FOLDS, shuffle=True,
                               random_state=SEED)
    for train_idx, test_idx in splitter.split(X, y):
        model = LogisticRegression(max_iter=1000)
        model.fit(X[train_idx], y[train_idx])
        oof[test_idx] = model.predict_proba(X[test_idx])[:, 1]

    if np.any(np.isnan(oof)):
        _fatal("Cross-fitting left points unscored; this is a bug in the "
               "fold construction, not a data property.")

    full = LogisticRegression(max_iter=1000)
    full.fit(X, y)

    return oof[:len(x_cal)], oof[len(x_cal):], full, y


def density_ratio(p_target, n_cal, n_target):
    """w(x) = p(target|x)/p(cal|x) * n_cal/n_target.

    The prior-correction factor is kept explicitly rather than folded away.
    Here the domain classifier is fit on a balanced 175/175 sample, so the
    factor is exactly 1.0 -- verified and reported, not assumed silently.
    """
    p = np.clip(np.asarray(p_target, dtype=np.float64), 1e-12, 1 - 1e-12)
    prior_correction = float(n_cal) / float(n_target)
    return (p / (1.0 - p)) * prior_correction, prior_correction


def apply_clip(weights, percentile):
    if percentile is None:
        return np.asarray(weights, dtype=np.float64), None
    cap = float(np.percentile(weights, percentile))
    return np.minimum(np.asarray(weights, dtype=np.float64), cap), cap


def max_weight_share(weights):
    w = np.asarray(weights, dtype=np.float64)
    total = float(w.sum())
    return float(w.max() / total) if total > 0 else float("nan")


# --------------------------------------------------------------------------- #
# Coverage, computed twice
# --------------------------------------------------------------------------- #
def coverage_from_sets(sets, labels):
    """Second, independent derivation of coverage (rule 6).

    Deliberately not cp.coverage(): a plain Python loop with an explicit
    counter, so a bug in the shared helper cannot hide behind itself.
    """
    hits = 0
    for s, y in zip(sets, labels):
        if y in s:
            hits += 1
    return hits / len(labels)


def set_stats(sets, n_classes):
    sizes = np.array([len(s) for s in sets], dtype=int)
    return {
        "mean_set_size": float(sizes.mean()),
        "singleton_rate": float(np.mean(sizes == 1)),
        "empty_rate": float(np.mean(sizes == 0)),
        "full_rate": float(np.mean(sizes == n_classes)),
    }


# --------------------------------------------------------------------------- #
# Finding 1 reproduction -- fatal on mismatch
# --------------------------------------------------------------------------- #
def check_published_gap(tier, gap):
    expected = PUBLISHED_FINDING1_GAP[tier]
    if abs(gap - expected) > GAP_TOLERANCE:
        _fatal(
            "{t} unweighted coverage gap at alpha=0.10 is {g:.10f}, but the "
            "published Finding 1 figure is {e:.10f}.\n"
            "  The probabilities are not the ones the published results came "
            "from.\n  This is this project's recurring bug class -- do not use "
            "this run.".format(t=tier, g=gap, e=expected))
    print("  [ok] {t} unweighted gap {g:+.6f} reproduces Finding 1 "
          "exactly".format(t=tier, g=gap))


def check_against_published_csv():
    """Independent derivation of the anchor, read from the published CSV.

    PUBLISHED_FINDING1_GAP is a hardcoded constant in this file; this confirms
    it against the artifact the paper actually cites, so a typo in the constant
    cannot quietly become the thing the run validates against.
    """
    if not os.path.isfile(PUBLISHED_CONFORMAL_CSV):
        _fatal("Published conformal results not found:\n    {p}\n"
               "  Run: python -m src.experiments.calibrate_conformal".format(
                   p=PUBLISHED_CONFORMAL_CSV))

    found = {}
    with open(PUBLISHED_CONFORMAL_CSV, newline="", encoding="utf-8") as fh:
        for row in csv.DictReader(fh):
            if (row["calibration_source"] == "in_domain"
                    and row["contamination"] == "contaminated"
                    and row["label_filter"] == "all"
                    and row["score_function"] == SCORE_FUNCTION
                    and row["mondrian"] == "False"
                    and abs(float(row["alpha"]) - PRIMARY_ALPHA) < 1e-12):
                found[row["tier"]] = float(row["coverage_gap"])

    for tier, expected in PUBLISHED_FINDING1_GAP.items():
        if tier not in found:
            _fatal("Could not find the published {t} row in {p}".format(
                t=tier, p=PUBLISHED_CONFORMAL_CSV))
        if abs(found[tier] - expected) > 1e-6:
            _fatal(
                "The constant in this file disagrees with the published CSV "
                "for {t}:\n  constant {c:.10f}\n  CSV      {v:.10f}".format(
                    t=tier, c=expected, v=found[tier]))
    print("  [ok] anchors agree with {p}".format(
        p=os.path.basename(PUBLISHED_CONFORMAL_CSV)))


# --------------------------------------------------------------------------- #
# The run
# --------------------------------------------------------------------------- #
def run(force=False):
    _banner("PHASE 6B - WEIGHTED CONFORMAL UNDER SHIFT  (measurement only)")
    print("Production thresholds are NOT modified by this script.")

    if settings.conformal.enabled:
        _fatal("settings.conformal.enabled is True. 6B is measurement only; "
               "production must stay frozen.")

    if os.path.isfile(RESULTS_CSV) and not force:
        _fatal("Refusing to overwrite an existing result:\n  {p}\n"
               "Pass --force if you mean to replace it.".format(p=RESULTS_CSV))

    # ---- Load ------------------------------------------------------------ #
    df, embeddings, calibration, benchmark, _ood, _adv = load_everything()
    if not os.path.isfile(DEPLOYMENT_JSON):
        _fatal("The deployment calibration set is required as the target "
               "proxy:\n    {p}".format(p=DEPLOYMENT_JSON))
    deployment = _read_json(DEPLOYMENT_JSON, "Deployment calibration set")

    print("\n[load] dataset {n} rows | calibration {c} | benchmark {b} | "
          "deployment {d}".format(n=len(df), c=len(calibration),
                                  b=len(benchmark), d=len(deployment)))

    cal_texts = [r["text"] for r in calibration]
    cal_labels = [r["expected"] for r in calibration]
    bench_texts = [r["text"] for r in benchmark]
    bench_labels = [r["expected"] for r in benchmark]
    dep_texts = [r["text"] for r in deployment]

    # ---- Fit the published (contaminated) scoring models ----------------- #
    # Rule 7 exception, stated: 6B refits rather than loading the production
    # artifacts because the in-domain 175 IS the contaminated calibration set,
    # and reproducing Finding 1's -0.233 exactly requires the exact fit that
    # produced it. 6A could use the production artifacts; 6B cannot.
    _banner("STEP 1 - Fitting the published scoring models (contaminated)")
    t1_vec, t1_clf, t2_clf, n_fit = fit_scoring_models(
        df, embeddings, exclude_ids=set())
    print("  fit on {n} rows (all of them -- the published configuration)"
          .format(n=n_fit))

    _banner("STEP 2 - Encoding")
    from src.agent.artifacts import load_artifacts
    artifacts = load_artifacts(require_gemini=False)

    cal_emb = embed(cal_texts, artifacts.embedder)
    bench_emb = embed(bench_texts, artifacts.embedder)
    dep_emb = embed(dep_texts, artifacts.embedder)
    print("  calibration {c} | benchmark {b} | deployment {d}".format(
        c=cal_emb.shape, b=bench_emb.shape, d=dep_emb.shape))

    cal_tfidf = t1_vec.transform(cal_texts).toarray()
    bench_tfidf = t1_vec.transform(bench_texts).toarray()
    dep_tfidf = t1_vec.transform(dep_texts).toarray()

    # ---- Probabilities per tier ------------------------------------------ #
    probs = {}
    cal_p1, classes1 = tier1_probabilities(t1_vec, t1_clf, cal_texts)
    bench_p1, _ = tier1_probabilities(t1_vec, t1_clf, bench_texts)
    probs["tier1"] = (cal_p1, bench_p1, classes1)

    classes2 = list(t2_clf.classes_)
    probs["tier2"] = (t2_clf.predict_proba(cal_emb),
                      t2_clf.predict_proba(bench_emb), classes2)

    # ---- Reproduce Finding 1 BEFORE anything weighted -------------------- #
    _banner("STEP 3 - Reproducing Finding 1 (fatal on mismatch)")
    check_against_published_csv()

    unweighted = {}
    for tier in TIERS:
        cal_probs, bench_probs, classes = probs[tier]
        for alpha in ALPHAS:
            stats, calib = evaluate(
                cal_probs, cal_labels, bench_probs, bench_labels, classes,
                alpha, SCORE_FUNCTION, mondrian=False)
            sets = cp.predict_sets(bench_probs, calib)
            gap = stats["coverage_benchmark"] - (1.0 - alpha)

            # Rule 6: coverage recomputed independently of evaluate().
            independent = coverage_from_sets(sets, bench_labels)
            if abs(independent - stats["coverage_benchmark"]) > 1e-12:
                _fatal("Coverage disagrees between two derivations for "
                       "{t} alpha={a}: {x} vs {y}".format(
                           t=tier, a=alpha, x=independent,
                           y=stats["coverage_benchmark"]))

            unweighted[(tier, alpha)] = {
                "coverage": stats["coverage_benchmark"],
                "gap": gap,
                "sets": sets,
                **set_stats(sets, len(classes)),
            }
        check_published_gap(tier, unweighted[(tier, PRIMARY_ALPHA)]["gap"])

    # ---- Domain classifiers ---------------------------------------------- #
    _banner("STEP 4 - Domain classifiers (cross-fitted, seed {s})".format(
        s=SEED))
    from sklearn.metrics import roc_auc_score

    spaces = {
        "bge": (cal_emb, dep_emb, bench_emb),
        "tfidf": (cal_tfidf, dep_tfidf, bench_tfidf),
    }

    weights_by_space = {}
    for space, (x_cal, x_dep, x_bench) in spaces.items():
        oof_cal, oof_dep, full_model, y = cross_fitted_domain_probabilities(
            x_cal, x_dep)
        auc = float(roc_auc_score(y, np.concatenate([oof_cal, oof_dep])))

        w_cal, prior = density_ratio(oof_cal, len(cal_texts), len(dep_texts))
        p_bench = full_model.predict_proba(x_bench)[:, 1]
        w_bench, _ = density_ratio(p_bench, len(cal_texts), len(dep_texts))

        if abs(prior - 1.0) > 1e-12:
            print("  [note] prior correction is {p:.6f}, not 1.0".format(
                p=prior))
        else:
            print("  [ok] prior correction n_cal/n_target = 1.000000 "
                  "(verified no-op)")

        print("  {s:>5}: cross-fitted AUC {a:.4f} | raw n_eff {n:.1f}".format(
            s=space, a=auc, n=cp.effective_sample_size(w_cal)))
        weights_by_space[space] = {
            "auc": auc, "w_cal": w_cal, "w_bench": w_bench,
        }

    # ---- Weighted sweep -------------------------------------------------- #
    _banner("STEP 5 - Weighted conformal sweep")
    rows = []

    for tier in TIERS:
        cal_probs, bench_probs, classes = probs[tier]
        cal_scores = cp.true_label_scores(
            cal_probs, cal_labels, classes, SCORE_FUNCTION)

        for alpha in ALPHAS:
            base = unweighted[(tier, alpha)]
            sd = coverage_sd(alpha, len(cal_labels))
            band = 2.0 * sd

            rows.append({
                "space": "n/a", "tier": tier, "alpha": alpha,
                "clip": "n/a", "weighting": "unweighted",
                "domain_auc": "", "n_eff": float(len(cal_labels)),
                "max_weight_share": 1.0 / len(cal_labels),
                "clip_value": "",
                "coverage_benchmark": base["coverage"],
                "coverage_gap": base["gap"],
                "coverage_sd": sd, "noise_band_2sd": band,
                "gap_outside_band": abs(base["gap"]) > band,
                "delta_vs_unweighted": 0.0,
                "delta_outside_band": False,
                "recovery_fraction_post_hoc": 0.0,
                "sets_identical_to_unweighted": True,
                "mean_set_size": base["mean_set_size"],
                "singleton_rate": base["singleton_rate"],
                "empty_rate": base["empty_rate"],
                "full_rate": base["full_rate"],
                "degenerate_reason": "",
                "is_primary": (tier == "tier1" and alpha == PRIMARY_ALPHA),
                "post_hoc": False,
            })

            for space in SPACES:
                info = weights_by_space[space]
                for clip_name, pct in CLIP_VARIANTS:
                    w_cal, cap = apply_clip(info["w_cal"], pct)
                    w_bench, _ = apply_clip(info["w_bench"], pct)

                    n_eff = cp.effective_sample_size(w_cal)
                    # Rule 6: n_eff from a second expression, n/(1+CV^2).
                    cv2 = float(np.var(w_cal) / (np.mean(w_cal) ** 2))
                    n_eff_alt = len(w_cal) / (1.0 + cv2)
                    if abs(n_eff - n_eff_alt) > 1e-6 * max(1.0, n_eff):
                        _fatal("n_eff disagrees between two derivations: "
                               "{a} vs {b}".format(a=n_eff, b=n_eff_alt))

                    sets = cp.weighted_predict_sets(
                        bench_probs, cal_scores, w_cal, w_bench, alpha,
                        classes, score_function=SCORE_FUNCTION)

                    cov = coverage_from_sets(sets, bench_labels)
                    gap = cov - (1.0 - alpha)
                    delta = gap - base["gap"]
                    identical = (sets == base["sets"])

                    reasons = []
                    if n_eff < MIN_EFFECTIVE_N:
                        reasons.append(
                            "n_eff {n:.1f} < {m:.0f}".format(
                                n=n_eff, m=MIN_EFFECTIVE_N))
                    if info["auc"] >= MAX_DOMAIN_AUC:
                        reasons.append(
                            "domain AUC {a:.4f} >= {m}".format(
                                a=info["auc"], m=MAX_DOMAIN_AUC))
                    if identical:
                        reasons.append(
                            "weighted and unweighted sets identical")

                    rows.append({
                        "space": space, "tier": tier, "alpha": alpha,
                        "clip": clip_name, "weighting": "weighted",
                        "domain_auc": info["auc"], "n_eff": n_eff,
                        "max_weight_share": max_weight_share(w_cal),
                        "clip_value": ("" if cap is None else cap),
                        "coverage_benchmark": cov,
                        "coverage_gap": gap,
                        "coverage_sd": sd, "noise_band_2sd": band,
                        "gap_outside_band": abs(gap) > band,
                        "delta_vs_unweighted": delta,
                        "delta_outside_band": abs(delta) > band,
                        "recovery_fraction_post_hoc": (
                            delta / abs(base["gap"]) if base["gap"] else 0.0),
                        "sets_identical_to_unweighted": identical,
                        **set_stats(sets, len(classes)),
                        "degenerate_reason": "; ".join(reasons),
                        "is_primary": (tier == "tier1"
                                       and alpha == PRIMARY_ALPHA),
                        "post_hoc": False,
                    })

    write_csv(rows)
    report(rows)
    return rows


def write_csv(rows):
    if not rows:
        _fatal("No rows to write.")
    fields = list(rows[0].keys())
    try:
        with open(RESULTS_CSV, "w", newline="", encoding="utf-8") as fh:
            writer = csv.DictWriter(fh, fieldnames=fields)
            writer.writeheader()
            writer.writerows(rows)
    except OSError as exc:
        _fatal("Failed to write {p}: {e}".format(p=RESULTS_CSV, e=repr(exc)))
    print("\n[write] {p}  ({n} rows)".format(p=RESULTS_CSV, n=len(rows)))


def report(rows):
    _banner("STEP 6 - The pre-registered verdict")

    primary = [r for r in rows
               if r["is_primary"] and r["weighting"] == "weighted"]
    base = [r for r in rows
            if r["is_primary"] and r["weighting"] == "unweighted"]
    if not base:
        _fatal("No unweighted primary row; cannot read the verdict.")
    base = base[0]

    band = base["noise_band_2sd"]
    sd = base["coverage_sd"]
    print("PRIMARY: Tier-1, benchmark45, alpha={a}, marginal coverage".format(
        a=PRIMARY_ALPHA))
    print("  unweighted gap {g:+.6f}  =  {n:.1f} s.d. below nominal "
          "(+-2 s.d. band {b:.4f})".format(
              g=base["coverage_gap"], n=abs(base["coverage_gap"]) / sd, b=band))
    print()

    header = ("  {:>5} {:>5} {:>7} {:>7} {:>10} {:>10} {:>6} {:>7}  {}".format(
        "space", "clip", "n_eff", "AUC", "gap", "delta", "moved", "residual",
        "degeneracy"))
    print(header)
    print("  " + "-" * (len(header) - 2))
    for r in primary:
        print("  {s:>5} {c:>5} {n:>7.1f} {a:>7.4f} {g:>+10.6f} {d:>+10.6f} "
              "{m:>6} {res:>7}  {x}".format(
                  s=r["space"], c=r["clip"], n=r["n_eff"], a=r["domain_auc"],
                  g=r["coverage_gap"], d=r["delta_vs_unweighted"],
                  m=("yes" if r["delta_outside_band"] else "no"),
                  res=("OUT" if r["gap_outside_band"] else "in"),
                  x=r["degenerate_reason"] or "-"))
    print("    moved    = the CHANGE exceeds the +-2 s.d. band")
    print("    residual = whether the REMAINING gap is still outside it")

    resolving = [r for r in primary if not r["degenerate_reason"]]
    blocked = [r for r in primary if r["degenerate_reason"]]

    print()
    if blocked:
        print("BLOCKED BY THE PRE-REGISTRATION: {n} of {m} configurations."
              .format(n=len(blocked), m=len(primary)))
        for c in sorted({r["degenerate_reason"] for r in blocked}):
            print("    - {c}".format(c=c))
        print("  Their numbers are in the CSV and are NOT read as a result,")
        print("  however favourable they look. That is what the rule is for.")
        print()

    if not resolving:
        print("VERDICT: NO RESOLUTION ON THE GATED AXIS.")
        print("  Every weighted configuration triggered a pre-registered")
        print("  degeneracy condition. Per the pre-registration, no secondary")
        print("  or averaged statistic is promoted in its place.")
        return

    moved = [r for r in resolving if r["delta_vs_unweighted"] > band]
    restored = [r for r in resolving if not r["gap_outside_band"]]

    print("VERDICT, on the {n} resolving configuration(s):".format(
        n=len(resolving)))
    print("  (a) gap MOVED by more than the band:      {n}/{m}".format(
        n=len(moved), m=len(resolving)))
    print("  (b) gap RESTORED to within the band:      {n}/{m}".format(
        n=len(restored), m=len(resolving)))
    print()

    if restored:
        print("  Reading: coverage is restored -- the shift is correctable by")
        print("  covariate reweighting.")
    elif moved:
        worst = min(resolving, key=lambda r: abs(r["coverage_gap"]))
        print("  Reading: reweighting IMPROVES coverage measurably but does "
              "NOT repair it.")
        print("  The best resolving configuration still sits {n:.1f} s.d. "
              "below nominal".format(n=abs(worst["coverage_gap"]) / sd))
        print("  (gap {g:+.6f}). A partial recovery, not a correction -- the "
              "same".format(g=worst["coverage_gap"]))
        print("  shape as Phase 1 Finding 4's distribution matching, which "
              "recovered ~38%")
        print("  and left the gap six times outside the band. Two independent")
        print("  repair strategies, both partial, both leaving coverage 5-6 "
              "s.d. out:")
        print("  this SHARPENS the named finding rather than overturning it.")
        print("  [post-hoc, descriptive] recovery fraction {f:.1%}".format(
            f=worst["recovery_fraction_post_hoc"]))
    else:
        print("  Reading: no resolving configuration moved the gap by more "
              "than the band.")
        print("  The shift lives in the representation the score is computed "
              "in, and")
        print("  reweighting cannot reach it -- which SHARPENS Finding 1.")

    # ---- Tier-2 sanity check ------------------------------------------- #
    _banner("Tier-2 sanity check (never a finding)")
    t2_base = [r for r in rows if r["tier"] == "tier2"
               and r["weighting"] == "unweighted"]
    t2_w = [r for r in rows if r["tier"] == "tier2"
            and r["weighting"] == "weighted"]

    base_by_alpha = {r["alpha"]: r for r in t2_base}
    harmed = [r for r in t2_w
              if abs(r["coverage_gap"])
              > abs(base_by_alpha[r["alpha"]]["coverage_gap"])]
    pushed_out = [r for r in t2_w if r["gap_outside_band"]]
    moved_beyond = [r for r in t2_w if r["delta_outside_band"]]

    print("  The check asks only whether weighting BREAKS Tier-2, whose "
          "unweighted")
    print("  gap is already inside the band. It cannot show a repair.")
    print("  configurations where |gap| got worse:            {n}/{m}".format(
        n=len(harmed), m=len(t2_w)))
    print("  configurations pushed OUTSIDE the band:          {n}/{m}".format(
        n=len(pushed_out), m=len(t2_w)))
    print("  configurations whose CHANGE exceeded the band:   {n}/{m}".format(
        n=len(moved_beyond), m=len(t2_w)))

    if pushed_out or moved_beyond:
        print()
        print("  NOT CLEAN. Weighting degrades Tier-2 in places -- it does "
              "not merely")
        print("  leave it alone. Worst cases:")
        for r in sorted(pushed_out + moved_beyond,
                        key=lambda r: -abs(r["coverage_gap"]))[:4]:
            print("    {s:>5}/{c:<5} alpha={a:<5} gap {g:+.6f} "
                  "(unweighted {u:+.6f})".format(
                      s=r["space"], c=r["clip"], a=r["alpha"],
                      g=r["coverage_gap"],
                      u=base_by_alpha[r["alpha"]]["coverage_gap"]))
        print("  This is a COST of reweighting and must be reported beside "
              "any Tier-1 gain.")
    else:
        print()
        print("  Clean: weighting did not break Tier-2.")


def main():
    parser = argparse.ArgumentParser(
        description="Phase 6B -- weighted conformal under shift "
                    "(measurement only, offline, no Gemini calls).")
    parser.add_argument("--force", action="store_true",
                        help="overwrite an existing results CSV")
    args = parser.parse_args()
    run(force=args.force)


if __name__ == "__main__":
    main()
