# src/experiments/compare_deferral_rules.py
"""
Phase 6A -- conformal deferral vs a confidence threshold.

WHY THIS EXISTS. Phase 1 measured whether conformal's COVERAGE guarantee holds.
That is not the operational question. The live cascade defers on a raw
confidence threshold (0.50), and the open question in PROJECT_STATUS.md is
whether conformal should replace it. Coverage cannot answer that: it says
whether the guarantee holds, not whether the rule defers on the RIGHT tickets.
This script asks the operational question -- given identical probabilities,
does deferring on conformal set size pick better tickets to escalate?

MEASUREMENT ONLY. settings.conformal.enabled stays False whatever this shows.

THE RULES ARE RANKINGS. A deferral rule is a scalar score plus a threshold, and
a risk-coverage curve depends only on the ORDERING that score induces. Worked
out against src/agent/conformal.py:

    rule         prediction set               singleton while        ranks by
    ----------   --------------------------   --------------------   ---------
    confidence   (not a conformal rule)       p1 >= tau              p1
    lac          {y : p(y)   >= 1 - q_hat}    p2 < 1-q_hat <= p1     p2
    aps          {y : cum(y) <= q_hat}        p1 <= q_hat < p1+p2    p1+p2

So LAC-conformal deferral ranks by the SECOND-largest probability where
confidence ranks by the largest. Different orderings, so the comparison is real
rather than a re-parameterisation. Two consequences:

1. The ranking comparison is CALIBRATION-FREE -- q_hat cancels out of the
   ordering. The calibration set only maps alpha onto the coverage axis, so
   this result is not exposed to the contamination that complicates Phase 1.
2. "Defer unless singleton" is NOT monotone in alpha. Both scores admit an
   EMPTY set (LAC when p1 < 1-q_hat, APS when p1 > q_hat), which is also a
   deferral, so the singleton region is an interval in q_hat rather than a
   half-line. Measured here, not assumed away.

THE CONTROL THAT WOULD OTHERWISE BE MISSING. Confidence alone is a weak
opponent, and conformal beating only that would repeat the Phase 5B mistake of
a comparison against the wrong baseline. So MARGIN (p1 - p2) is included: the
standard selective-prediction baseline, and the arm most likely to match
conformal since both use p2.

AURC IS NOT THE HEADLINE. This project's standing position is precision over
recall -- a wrong auto-route costs more than an escalation. AURC averages over
the whole coverage range including regions this system would never operate in,
so a rule could win on AURC by being better exactly where the project does not
care. That is the failure the resolution-clustering promotion rule nearly made.
The pre-registered primary metric is therefore RISK AT THE OPERATING COVERAGE,
with AURC secondary, and the verdict rule is fixed in advance: conformal is
"better" only if it lowers risk at the operating coverage with a bootstrap
interval excluding zero. Overlapping intervals mean "no signal on the axis we
care about", never a win claimed from AURC.

Offline: production artifacts and existing CSV/JSON only. No Gemini, no quota.

Usage (from the project root):
    python src/experiments/compare_deferral_rules.py
    python src/experiments/compare_deferral_rules.py --force
"""

import os
import sys
import csv
import json
import argparse
import traceback

import numpy as np

THIS_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(THIS_DIR, os.pardir, os.pardir))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from src.agent import conformal  # noqa: E402
from src.experiments.compare_cascade_vs_tier2 import (  # noqa: E402
    _banner, _fatal,
)

DATA_DIR = os.path.join(PROJECT_ROOT, "data")
CONFORMAL_JSON = os.path.join(
    DATA_DIR, "conformal_calibration_corrected_bge-base-en-v1-5.json")

RULES = ("confidence", "margin", "lac", "aps")
EVAL_SETS = ("benchmark45", "deployment175")
TIERS = ("tier1", "tier2")

# Secondary illustrative grid. EXTENDED DOWNWARD after the first run measured
# the live 0.50 cascade gate answering only ~19% of deployment175 at Tier-1 --
# the original (0.50, 0.70, 0.80, 0.90) grid did not span the operating point
# at all, so every figure on it described coverages this system never runs at.
# The PRIMARY metric was pre-registered and is unchanged: risk at the MEASURED
# operating coverage of the live gate, which is computed independently of this
# grid and is what the verdict rule reads.
RISK_AT_COVERAGES = (0.20, 0.30, 0.50, 0.70, 0.80, 0.90)

BOOTSTRAP_N = 10000
BOOTSTRAP_SEED = 42

# The published accuracies each curve's coverage-1.0 endpoint must reproduce.
# A mismatch means the probabilities are not the ones the published results
# came from -- this project's recurring bug -- so it is fatal, not a warning.
PUBLISHED_FULL_COVERAGE = {
    ("tier2", "benchmark45"): 33,
    ("tier2", "deployment175"): 132,
    ("tier1", "benchmark45"): 16,
    ("tier1", "deployment175"): 91,
}


# --------------------------------------------------------------------------
# Deferral scores -- higher means "more confident, accept sooner"
# --------------------------------------------------------------------------
def deferral_score(probabilities, rule):
    """Scalar accept-score per row, for one rule.

    Derived from the prediction-set algebra in the module docstring. The sign
    is chosen so that LARGER always means "accept earlier", for every rule, so
    one sweep routine serves all four.
    """
    p = np.asarray(probabilities, dtype=np.float64)
    if p.ndim != 2:
        raise ValueError("probabilities must be 2-D (n_samples, n_classes)")

    # Descending sort once; every rule is a function of the top two.
    ordered = np.sort(p, axis=1)[:, ::-1]
    p1 = ordered[:, 0]
    p2 = ordered[:, 1] if ordered.shape[1] > 1 else np.zeros_like(p1)

    if rule == "confidence":
        return p1
    if rule == "margin":
        return p1 - p2
    if rule == "lac":
        # Singleton while q_hat < 1 - p2, so a SMALLER p2 survives longer.
        return -p2
    if rule == "aps":
        # Singleton while q_hat < p1 + p2, so a LARGER top-2 mass survives
        # longer. Note this is the opposite direction to intuition and is
        # exactly why it gets measured rather than assumed.
        return p1 + p2
    raise ValueError("Unknown rule {r!r}".format(r=rule))


# --------------------------------------------------------------------------
# Risk-coverage
# --------------------------------------------------------------------------
def risk_coverage_curve(scores, correct):
    """Return (coverages, risks, thresholds) for one rule.

    Accept the highest-scoring tickets first. Risk is the error rate AMONG
    ACCEPTED tickets. Ties are resolved by accepting a whole tie-group at once,
    so no rule gains an ordering from arbitrary tie-breaking -- a rule that
    cannot separate two tickets must not be credited as if it could.
    """
    scores = np.asarray(scores, dtype=np.float64)
    correct = np.asarray(correct, dtype=bool)
    if scores.shape[0] != correct.shape[0]:
        raise ValueError("scores and correct differ in length")
    n = scores.shape[0]
    if n == 0:
        raise ValueError("empty evaluation set")

    order = np.argsort(-scores, kind="stable")
    s_sorted = scores[order]
    c_sorted = correct[order]
    errors = np.cumsum(~c_sorted)

    # Keep only the last index of each tie-group, so a cut never falls inside
    # one. Vectorised deliberately: the paired bootstrap calls this ~240,000
    # times, and a Python loop over n here made the run take tens of minutes.
    keep = np.ones(n, dtype=bool)
    keep[:-1] = s_sorted[1:] != s_sorted[:-1]

    k = np.flatnonzero(keep) + 1
    return k / n, errors[keep] / k, s_sorted[keep]


def aurc(coverages, risks):
    """Trapezoidal area under the risk-coverage curve, over [min_cov, 1]."""
    if len(coverages) == 1:
        return float(risks[0])
    return float(np.trapezoid(risks, coverages) / (coverages[-1] - coverages[0]))


def oracle_aurc(correct):
    """AURC of the optimal ordering: every correct ticket accepted first.

    Built directly rather than via risk_coverage_curve, because that function
    accepts whole tie-groups at once and a perfect score vector would tie all
    correct tickets together -- which would understate the oracle.
    """
    correct = np.asarray(correct, dtype=bool)
    n = len(correct)
    n_right = int(correct.sum())
    ordered = np.concatenate([np.ones(n_right), np.zeros(n - n_right)])
    errors = np.cumsum(1.0 - ordered)
    cov = np.arange(1, n + 1) / n
    risk = errors / np.arange(1, n + 1)
    return aurc(cov, risk)


def risk_at_coverage(coverages, risks, target):
    """Risk at the smallest achievable coverage >= target.

    Reported with the coverage actually used, because at n=45 the grid is
    coarse and quoting a requested coverage that was never achievable would be
    a number describing no operating point that exists.
    """
    idx = np.searchsorted(coverages, target, side="left")
    if idx >= len(coverages):
        return float(risks[-1]), float(coverages[-1])
    return float(risks[idx]), float(coverages[idx])


# --------------------------------------------------------------------------
# Paired bootstrap
# --------------------------------------------------------------------------
def paired_bootstrap(scores_a, scores_b, correct, statistic, n=BOOTSTRAP_N,
                     seed=BOOTSTRAP_SEED):
    """95% interval for statistic(a) - statistic(b), resampling TICKETS.

    Paired because both rules score the same tickets -- the same reason Phase
    5B used McNemar rather than two independent accuracies.
    """
    rng = np.random.default_rng(seed)
    m = len(correct)
    diffs = np.empty(n, dtype=np.float64)
    for i in range(n):
        idx = rng.integers(0, m, size=m)
        c = correct[idx]
        diffs[i] = statistic(scores_a[idx], c) - statistic(scores_b[idx], c)
    lo, hi = np.percentile(diffs, [2.5, 97.5])
    return float(np.mean(diffs)), float(lo), float(hi)


def _aurc_stat(scores, correct):
    cov, risk, _ = risk_coverage_curve(scores, correct)
    return aurc(cov, risk)


def _risk_at_stat(target):
    def stat(scores, correct):
        cov, risk, _ = risk_coverage_curve(scores, correct)
        return risk_at_coverage(cov, risk, target)[0]
    return stat


# --------------------------------------------------------------------------
# Data
# --------------------------------------------------------------------------
def load_eval_set(name):
    """Reuse each set's own validated loader -- never re-read the JSON here."""
    from src.experiments import run_ablation_study as abl
    if name == "benchmark45":
        from src.classification.train_cascade import load_expanded_set
        return load_expanded_set(abl.EXPANDED_JSON_PATH)
    if name == "deployment175":
        return abl.load_deployment175_set()
    _fatal("Unknown eval set {n!r}".format(n=name))


def tier_probabilities(texts, tier, artifacts):
    """Full probability matrix plus class order, from the PRODUCTION artifacts.

    Both evaluation sets are disjoint from training -- benchmark45 is the fixed
    read-only benchmark, deployment175 is Gemini-generated deployment-register
    text rather than paraphrases of training rows -- so no leave-out fit is
    needed and none is used. This is deliberately NOT calibrate_conformal.py's
    path, which fits leave-out models because its calibration set IS
    contaminated.
    """
    if tier == "tier1":
        X = artifacts.tier1_vectorizer.transform(texts)
        clf = artifacts.tier1_classifier
        return np.asarray(clf.predict_proba(X)), list(clf.classes_)
    if tier == "tier2":
        emb = np.asarray(artifacts.embedder.encode(list(texts)),
                         dtype=np.float32)
        clf = artifacts.tier2_classifier
        return np.asarray(clf.predict_proba(emb)), list(clf.classes_)
    _fatal("Unknown tier {t!r}".format(t=tier))


def correctness(probabilities, classes, expected):
    """Boolean correct-vector from argmax, recomputed rather than trusted."""
    preds = [classes[i] for i in np.argmax(probabilities, axis=1)]
    return np.array([p == e for p, e in zip(preds, expected)], dtype=bool)


# --------------------------------------------------------------------------
# The run
# --------------------------------------------------------------------------
def run_one(tier, eval_set, artifacts, rows_out):
    records = load_eval_set(eval_set)
    texts = [r["text"] for r in records]
    expected = [r["expected"] for r in records]

    probs, classes = tier_probabilities(texts, tier, artifacts)
    correct = correctness(probs, classes, expected)
    n = len(correct)
    n_correct = int(correct.sum())

    # Rule 6 / the recurring bug class: the coverage-1.0 endpoint must
    # reproduce the published accuracy for this tier and set.
    expected_correct = PUBLISHED_FULL_COVERAGE.get((tier, eval_set))
    if expected_correct is not None and n_correct != expected_correct:
        _fatal(
            "{t} on {s}: argmax accuracy is {a}/{n}, but the published figure "
            "is {e}/{n}.\n"
            "  The probabilities are not the ones the published results came "
            "from. This is this project's recurring bug class -- do not use "
            "this run.".format(t=tier, s=eval_set, a=n_correct, n=n,
                               e=expected_correct))

    _banner("{t} / {s}   n={n}, argmax {a}/{n} = {p:.2%}".format(
        t=tier, s=eval_set, n=n, a=n_correct, p=n_correct / n))

    curves, summary = {}, {}
    for rule in RULES:
        s = deferral_score(probs, rule)
        cov, risk, thr = risk_coverage_curve(s, correct)
        curves[rule] = (s, cov, risk)
        summary[rule] = {
            "aurc": aurc(cov, risk),
            "n_points": len(cov),
            "min_coverage": float(cov[0]),
        }
        for c, r, t in zip(cov, risk, thr):
            rows_out.append({
                "tier": tier, "eval_set": eval_set, "rule": rule,
                "threshold": "{:.6f}".format(t),
                "coverage": "{:.6f}".format(c),
                "risk": "{:.6f}".format(r),
            })

    oracle = oracle_aurc(correct)
    for rule in RULES:
        summary[rule]["excess_aurc"] = summary[rule]["aurc"] - oracle
        for target in RISK_AT_COVERAGES:
            r, actual = risk_at_coverage(*curves[rule][1:], target)
            summary[rule]["risk_at_{:.0f}".format(target * 100)] = r
            summary[rule]["coverage_used_{:.0f}".format(target * 100)] = actual

    print("  oracle AURC {o:.4f}".format(o=oracle))
    header = "  {:<12}{:>9}{:>10}".format("rule", "AURC", "excess")
    header += "".join("{:>9}".format("r@%d" % (c * 100))
                      for c in RISK_AT_COVERAGES)
    print(header)
    for rule in RULES:
        s = summary[rule]
        line = "  {:<12}{:>9.4f}{:>10.4f}".format(
            rule, s["aurc"], s["excess_aurc"])
        line += "".join("{:>9.3f}".format(s["risk_at_%d" % (c * 100)])
                        for c in RISK_AT_COVERAGES)
        print(line)

    return curves, summary, correct, n, probs, classes


def operating_coverage_of_live_gate(artifacts, eval_set):
    """The fraction Tier-1 answers under the live 0.50 gate. MEASURED."""
    from src.agent.config import settings
    records = load_eval_set(eval_set)
    texts = [r["text"] for r in records]
    probs, _ = tier_probabilities(texts, "tier1", artifacts)
    conf = np.max(probs, axis=1)
    return float(np.mean(conf >= settings.cascade.confidence_threshold))


def check_conformal_fingerprint():
    """Refuse to overlay operating points from a stale calibration."""
    if not os.path.isfile(CONFORMAL_JSON):
        return None
    from src.agent.config import config_fingerprint
    try:
        with open(CONFORMAL_JSON, "r", encoding="utf-8") as fh:
            blob = json.load(fh)
    except Exception as exc:
        _fatal("Failed to read {p}: {e}".format(p=CONFORMAL_JSON, e=repr(exc)))
    stored = blob.get("config_fingerprint")
    live = config_fingerprint()
    if stored != live:
        _fatal(
            "Conformal calibration {p} carries fingerprint {s!r} but the live "
            "config is {l!r}.\n"
            "  Overlaying its operating points would plot stale "
            "calibration.".format(p=CONFORMAL_JSON, s=stored, l=live))
    return blob


def conformal_operating_points(blob, tier, eval_set, probs, classes, correct):
    """The REAL alpha-indexed operating points, from the published calibration.

    The risk-coverage curves above are calibration-free rankings. These are the
    points the actual conformal procedure would land on, so they say whether
    any achievable alpha sits anywhere near the live gate's coverage. Without
    this, 6A would compare orderings and never check that conformal can be
    *operated* at the coverage the system runs at.

    Coverage here is the SINGLETON rate (a non-singleton or empty set is a
    deferral), and risk is the error rate among singletons.
    """
    rows = []
    for key, fit in blob.get("fits", {}).items():
        parts = key.split("|")
        if len(parts) != 6:
            continue
        contamination, fit_tier, subset, score_fn, variant, alpha = parts
        if fit_tier != tier or variant != "marginal" or subset != "all":
            continue
        if score_fn not in ("lac", "aps"):
            continue
        q = fit.get("quantile")
        if q is None:
            continue

        cal = conformal.ConformalCalibration(
            alpha=float(alpha), score_function=score_fn,
            classes=list(fit["classes"]), quantile=float(q),
            n_calibration=int(fit.get("n_calibration", 0)),
        )
        sets = conformal.predict_sets(probs, cal)
        singleton = np.array([len(s) == 1 for s in sets])
        n_single = int(singleton.sum())
        rows.append({
            "tier": tier, "eval_set": eval_set, "score_function": score_fn,
            "contamination": contamination, "alpha": alpha,
            "quantile": "{:.6f}".format(float(q)),
            "singleton_rate": "{:.6f}".format(n_single / len(singleton)),
            "n_singleton": n_single,
            "risk_among_singletons": (
                "{:.6f}".format(1.0 - correct[singleton].mean())
                if n_single else ""),
            "mean_set_size": "{:.4f}".format(
                float(np.mean([len(s) for s in sets]))),
            "empty_set_rate": "{:.6f}".format(
                float(np.mean([len(s) == 0 for s in sets]))),
        })
    return rows


def plot_curves(tier, eval_set, curves, live_cov, out_path):
    """One risk-coverage figure per tier x set. Missing matplotlib is not fatal
    -- the CSVs are the result; the figure is a convenience."""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception:
        print("  (matplotlib unavailable; skipping " + os.path.basename(out_path) + ")")
        return None

    fig, ax = plt.subplots(figsize=(6.4, 4.2))
    for rule in RULES:
        _, cov, risk = curves[rule]
        ax.plot(cov, risk, marker=".", linewidth=1.2, label=rule)
    ax.axvline(live_cov, color="0.4", linestyle="--", linewidth=1.0,
               label="live 0.50 gate ({:.1%})".format(live_cov))
    ax.set_xlabel("coverage (fraction auto-routed)")
    ax.set_ylabel("risk (error rate among accepted)")
    ax.set_title("{t} / {s} -- risk vs coverage".format(t=tier, s=eval_set))
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    try:
        fig.savefig(out_path, dpi=150)
    finally:
        plt.close(fig)
    return out_path


def parse_args(argv):
    parser = argparse.ArgumentParser(
        description=("Phase 6A: risk-coverage comparison of deferral rules. "
                     "Offline, measurement only.")
    )
    parser.add_argument("--force", action="store_true",
                        help="Overwrite existing 6A output files.")
    return parser.parse_args(argv)


def _refuse_overwrite(path, force):
    if os.path.isfile(path) and not force:
        _fatal("Refusing to overwrite an existing result:\n  {p}\n"
               "Pass --force if you mean to replace it.".format(p=path))


def main(argv=None):
    args = parse_args(argv)
    from src.agent import artifacts as artifacts_mod
    from src.agent.config import settings

    _banner("Phase 6A -- conformal deferral vs a confidence threshold")
    print("MEASUREMENT ONLY. settings.conformal.enabled = {e} (unchanged)"
          .format(e=settings.conformal.enabled))
    if settings.conformal.enabled:
        _fatal("settings.conformal.enabled is True. 6A is measurement only; "
               "it must not run against a promoted gate.")

    curve_path = os.path.join(DATA_DIR, "deferral_risk_coverage.csv")
    summary_path = os.path.join(DATA_DIR, "deferral_rule_summary.csv")
    operating_path = os.path.join(
        DATA_DIR, "deferral_conformal_operating_points.csv")
    for p in (curve_path, summary_path, operating_path):
        _refuse_overwrite(p, args.force)

    conformal_blob = check_conformal_fingerprint()
    artifacts = artifacts_mod.load_artifacts(require_gemini=False)

    curve_rows = []
    summary_rows = []
    operating_rows = []
    for eval_set in EVAL_SETS:
        live_cov = operating_coverage_of_live_gate(artifacts, eval_set)
        print("")
        print("Live 0.50 cascade gate answers {c:.1%} of {s} at Tier-1 "
              "(measured).".format(c=live_cov, s=eval_set))
        for tier in TIERS:
            curves, summary, correct, n, probs, classes = run_one(
                tier, eval_set, artifacts, curve_rows)

            # Is the gated axis even measurable here? If every rule scores the
            # SAME risk at the live operating coverage -- in particular 0.0,
            # which happens when all four accept only correct tickets in that
            # region -- then "no signal" means the test had no RESOLUTION, not
            # that the rules are equivalent. Recorded rather than left for a
            # reader to notice, the same way 4B-1 reports the degenerate
            # escalation-rate test as None rather than as p = 0.
            gate_risks = [
                risk_at_coverage(curves[r][1], curves[r][2], live_cov)[0]
                for r in RULES
            ]
            degenerate = "yes" if len(set(gate_risks)) == 1 else "no"
            if degenerate == "yes":
                print("  NOTE: every rule scores risk {r:.4f} at the live gate "
                      "coverage ({c:.1%}) -- the gated axis has NO RESOLUTION "
                      "here.".format(r=gate_risks[0], c=live_cov))

            # Paired bootstrap of every rule against the incumbent.
            base = curves["confidence"][0]
            for rule in RULES:
                row = {
                    "tier": tier, "eval_set": eval_set, "rule": rule,
                    "n": n, "live_gate_coverage": "{:.6f}".format(live_cov),
                }
                row.update({k: "{:.6f}".format(v) if isinstance(v, float)
                            else v for k, v in summary[rule].items()})
                row["degenerate_at_gate"] = degenerate
                if rule != "confidence":
                    m, lo, hi = paired_bootstrap(
                        curves[rule][0], base, correct, _aurc_stat)
                    row["d_aurc_vs_confidence"] = "{:.6f}".format(m)
                    row["d_aurc_ci_low"] = "{:.6f}".format(lo)
                    row["d_aurc_ci_high"] = "{:.6f}".format(hi)
                    lm, llo, lhi = paired_bootstrap(
                        curves[rule][0], base, correct,
                        _risk_at_stat(live_cov))
                    row["d_risk_at_live_gate"] = "{:.6f}".format(lm)
                    row["d_risk_at_live_ci_low"] = "{:.6f}".format(llo)
                    row["d_risk_at_live_ci_high"] = "{:.6f}".format(lhi)
                    row["verdict_at_live_gate"] = (
                        "better" if lhi < 0 else
                        "worse" if llo > 0 else
                        "no signal on the gated axis")
                summary_rows.append(row)

            if conformal_blob is not None:
                operating_rows.extend(conformal_operating_points(
                    conformal_blob, tier, eval_set, probs, classes, correct))

            fig_path = os.path.join(
                DATA_DIR, "deferral_risk_coverage_{t}_{s}.png".format(
                    t=tier, s=eval_set))
            plot_curves(tier, eval_set, curves, live_cov, fig_path)

    _write_csv(curve_path, curve_rows)
    _write_csv(summary_path, summary_rows)
    print("")
    print("Curves written  : " + curve_path)
    print("Summary written : " + summary_path)
    if operating_rows:
        _write_csv(operating_path, operating_rows)
        print("Operating pts   : " + operating_path)
    print("")
    print("READ THIS WITH THE NUMBERS: AURC averages over coverages this")
    print("system would never run at. The gated axis is risk at the operating")
    print("coverage; overlapping bootstrap intervals mean NO SIGNAL there,")
    print("never a win claimed from AURC.")


def _write_csv(path, rows):
    if not rows:
        _fatal("No rows to write for {p}".format(p=path))
    fieldnames = []
    for r in rows:
        for k in r:
            if k not in fieldnames:
                fieldnames.append(k)
    try:
        with open(path, "w", newline="", encoding="utf-8") as fh:
            w = csv.DictWriter(fh, fieldnames=fieldnames)
            w.writeheader()
            for r in rows:
                w.writerow(r)
    except Exception as exc:
        _fatal("Failed to write {p}: {e}".format(p=path, e=repr(exc)))


if __name__ == "__main__":
    try:
        main()
    except SystemExit:
        raise
    except Exception:
        print("")
        print("=" * 70)
        print("UNEXPECTED ERROR -- full traceback follows:")
        print("=" * 70)
        traceback.print_exc()
        sys.exit(1)
