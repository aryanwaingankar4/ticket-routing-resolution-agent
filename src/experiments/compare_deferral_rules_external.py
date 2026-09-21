# src/experiments/compare_deferral_rules_external.py
"""
Phase 7B secondary -- re-asking Phase 6A's question where it can resolve.

WHY THIS EXISTS. 6A asked whether deferring on conformal set size beats
deferring on a confidence threshold, and returned NO SIGNAL on the gated axis
in all 12 comparisons -- with 3 of 4 configurations DEGENERATE at the live
gate, because the live 0.50 gate's measured operating coverage contained only
FOUR tickets on the 45-ticket benchmark and 34 on the deployment set. The
question was not answered; it was unmeasurable.

This corpus is the first place that comparison can resolve. A 10% operating
coverage on the version-400 test arm is ~1,044 tickets rather than four.

SAME RULES AS 6A, UNCHANGED:
  * four rules -- confidence, margin, lac, aps -- scored as ORDERINGS, with
    the scoring functions imported from compare_deferral_rules.py rather than
    reimplemented;
  * PRIMARY metric is RISK AT THE MEASURED OPERATING COVERAGE, with a
    10,000-draw paired bootstrap at seed 42;
  * AURC IS SECONDARY AND IS NEVER PROMOTED. This project gates on risk at the
    operating coverage; AURC averages over coverages the system never runs at.
    6A's AURC effects contradicted across sets and favoured margin, and quoting
    one as a reason to promote conformal was named as the error to avoid;
  * overlapping bootstrap intervals mean NO SIGNAL ON THE AXIS WE CARE ABOUT,
    never a win claimed elsewhere;
  * if every rule yields the same risk at the gate, the row is degenerate and
    the verdict is "no resolution".

THE OPERATING POINT IS A TRANSPLANT, AND IS DESCRIBED AS ONE. The 0.50 cascade
threshold was calibrated on OUR corpus, over OUR seven categories. Applied
here it is our gate transplanted onto a different generator's ten queues. Its
coverage on this arm is a MEASURED property of that transplant and is reported
as such -- not as a gate calibrated for this corpus.

IT READS PHASE 7B'S OWN FITTED PROBABILITIES. The primary script caches them;
refitting here would compare a different model than the one 7B measured.

MEASUREMENT ONLY, offline, zero Gemini calls. Reads nothing from 6A's outputs
and writes nothing near them.

Run from the project root (after run_external_conformal_shift.py, ~6 min):
    python src/experiments/compare_deferral_rules_external.py
    python src/experiments/compare_deferral_rules_external.py --force
"""

from __future__ import annotations

import argparse
import json
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
from src.experiments.calibrate_conformal import _banner, _fatal    # noqa: E402
from src.experiments.compare_deferral_rules import (               # noqa: E402
    BOOTSTRAP_N,
    BOOTSTRAP_SEED,
    RISK_AT_COVERAGES,
    RULES,
    _aurc_stat,
    _risk_at_stat,
    aurc,
    deferral_score,
    oracle_aurc,
    paired_bootstrap,
    risk_at_coverage,
    risk_coverage_curve,
)
from src.experiments.fetch_external_dataset import OUT_DIR         # noqa: E402
from src.experiments.run_external_conformal_shift import (         # noqa: E402
    ALPHAS,
    LABEL_NOISE_JSON,
    PROBS_NPZ,
    SCORE_FUNCTION,
    SPLITS_JSON,
    TIERS,
    isolation_paths,
    snapshot,
    verify_isolation,
    write_csv,
)

ensure_utf8_console()

RESULTS_CSV = os.path.join(OUT_DIR, "external_deferral_results.csv")
INCUMBENT = "confidence"


def load_inputs():
    for path in (SPLITS_JSON, PROBS_NPZ, LABEL_NOISE_JSON):
        if not os.path.isfile(path):
            _fatal("Missing {p}\n  Run: python src/experiments/"
                   "run_external_conformal_shift.py".format(
                       p=os.path.relpath(path, PROJECT_ROOT)))

    with open(SPLITS_JSON, encoding="utf-8") as fh:
        manifest = json.load(fh)
    with open(LABEL_NOISE_JSON, encoding="utf-8") as fh:
        label_noise = json.load(fh)
    blob = np.load(PROBS_NPZ, allow_pickle=False)
    return manifest, label_noise, blob


def correctness(probabilities, classes, labels):
    predicted = np.asarray(classes)[np.argmax(probabilities, axis=1)]
    return predicted == np.asarray(labels)


def conformal_operating_points(cal_probs, cal_labels, test_probs, classes,
                               correct):
    """The REAL alpha-indexed points the conformal procedure would land on.

    The risk-coverage curves are calibration-free rankings; these say whether
    any achievable alpha sits anywhere near the transplanted gate's coverage.
    Without them 7B would compare orderings and never check that conformal can
    be OPERATED where the system runs -- which is the check that made 6A's
    verdict honest.

    "Accept" is a SINGLETON prediction set. An empty set is a deferral, not an
    acceptance: it is the strongest out-of-distribution signal the procedure
    emits, and treating it as an accept would invert its meaning.
    """
    points = []
    for alpha in ALPHAS:
        calib = cp.calibrate(cal_probs, list(cal_labels), classes,
                             alpha=alpha, score_function=SCORE_FUNCTION,
                             mondrian=False)
        sets = cp.predict_sets(test_probs, calib)
        accept = np.array([len(s) == 1 for s in sets], dtype=bool)
        coverage = float(accept.mean())
        risk = (float(np.mean(~correct[accept])) if accept.any()
                else float("nan"))
        points.append({
            "alpha": alpha,
            "degenerate": calib.is_degenerate,
            "coverage": coverage,
            "risk": risk,
            "n_accepted": int(accept.sum()),
        })
    return points


def run(force=False):
    _banner("PHASE 7B SECONDARY - DEFERRAL RULES WHERE THE COMPARISON CAN "
            "RESOLVE")
    print("Pre-registered secondary. Same rules as 6A: risk at the operating")
    print("coverage is PRIMARY; AURC is secondary and is NEVER promoted.")
    print("Measurement only: conformal.enabled={c}".format(
        c=settings.conformal.enabled))
    if settings.conformal.enabled:
        _fatal("This phase is measurement only; conformal.enabled must be "
               "False.")

    if os.path.isfile(RESULTS_CSV) and not force:
        _fatal("Refusing to overwrite an existing result:\n  {p}\n"
               "Pass --force if you mean to replace it.".format(
                   p=RESULTS_CSV))

    before = snapshot(isolation_paths())
    manifest, label_noise, blob = load_inputs()

    test_labels = blob["test_labels"]
    cal_labels = blob["cal_labels"]
    print("  test arm {n} tickets | calibration {c}".format(
        n=len(test_labels), c=len(cal_labels)))

    # ---- The transplanted operating point -------------------------------- #
    _banner("STEP 1 - The live gate's operating coverage on this arm")
    tier1_test = blob["tier1_test"]
    gate = settings.cascade.confidence_threshold
    live_cov = float(np.mean(np.max(tier1_test, axis=1) >= gate))
    n_at_gate = int(round(live_cov * len(test_labels)))
    print("  the 0.50 cascade gate is OUR threshold, calibrated on OUR seven")
    print("  categories, transplanted onto this corpus's ten queues. Its")
    print("  coverage here is a measured property of the transplant.")
    print("  operating coverage {c:.4f}  ->  {n} tickets".format(
        c=live_cov, n=n_at_gate))
    print("  6A had FOUR tickets on benchmark45 and 34 on deployment175.")

    rows = []
    for tier in TIERS:
        _banner("STEP 2 - {t}".format(t=tier.upper()))
        test_probs = blob["{t}_test".format(t=tier)]
        cal_probs = blob["{t}_cal".format(t=tier)]
        classes = list(blob["{t}_classes".format(t=tier)])

        correct = correctness(test_probs, classes, test_labels)
        accuracy = float(correct.mean())

        # Rule 6: the curve's coverage-1.0 endpoint must reproduce the
        # accuracy the primary script measured independently.
        published = label_noise[tier]["accuracy_test"]
        if abs(accuracy - published) > 1e-9:
            _fatal("Accuracy disagrees between two derivations for {t}: "
                   "{a} here vs {b} in {p}".format(
                       t=tier, a=accuracy, b=published,
                       p=os.path.basename(LABEL_NOISE_JSON)))
        print("  accuracy {a:.4f} on {n} tickets  [matches the primary run]"
              .format(a=accuracy, n=len(correct)))

        curves, scores = {}, {}
        for rule in RULES:
            s = deferral_score(test_probs, rule)
            cov, risk, _thr = risk_coverage_curve(s, correct)
            if abs(cov[-1] - 1.0) > 1e-12:
                _fatal("{r} curve does not reach full coverage".format(r=rule))
            if abs((1.0 - risk[-1]) - accuracy) > 1e-9:
                _fatal("{r} curve's coverage-1.0 endpoint is {e} but the "
                       "measured accuracy is {a}".format(
                           r=rule, e=1.0 - risk[-1], a=accuracy))
            scores[rule] = s
            curves[rule] = (cov, risk)

        oracle = oracle_aurc(correct)
        gate_risks = {r: risk_at_coverage(*curves[r], live_cov)[0]
                      for r in RULES}
        degenerate = len(set(gate_risks.values())) == 1
        if degenerate:
            print("  [degenerate] every rule accepts the same tickets at the "
                  "gate -- no resolution")
        else:
            print("  risk at the operating coverage, by rule:")
            for rule in RULES:
                print("    {r:11} {x:.4f}".format(r=rule, x=gate_risks[rule]))

        points = conformal_operating_points(cal_probs, cal_labels, test_probs,
                                            classes, correct)
        print("  achievable conformal operating points (singleton = accept):")
        for pt in points:
            print("    alpha {a:.2f}: coverage {c:.4f} ({n} tickets), risk "
                  "{r}".format(a=pt["alpha"], c=pt["coverage"],
                               n=pt["n_accepted"],
                               r=("n/a" if np.isnan(pt["risk"])
                                  else "{:.4f}".format(pt["risk"]))))

        for rule in RULES:
            cov, risk = curves[rule]
            gate_risk, gate_cov = risk_at_coverage(cov, risk, live_cov)
            row = {
                "tier": tier,
                "rule": rule,
                "n_test": len(correct),
                "accuracy": accuracy,
                "live_gate_coverage": live_cov,
                "coverage_used": gate_cov,
                "risk_at_gate": gate_risk,
                "degenerate_at_gate": "yes" if degenerate else "no",
                "aurc": aurc(cov, risk),
                "oracle_aurc": oracle,
                "is_incumbent": rule == INCUMBENT,
                "post_hoc": False,
            }
            for target in RISK_AT_COVERAGES:
                r_at, c_at = risk_at_coverage(cov, risk, target)
                row["risk_at_{t:.2f}".format(t=target)] = r_at
                row["coverage_used_{t:.2f}".format(t=target)] = c_at

            if rule != INCUMBENT:
                mean, lo, hi = paired_bootstrap(
                    scores[rule], scores[INCUMBENT], correct,
                    _risk_at_stat(live_cov), n=BOOTSTRAP_N,
                    seed=BOOTSTRAP_SEED)
                row.update({
                    "risk_delta_vs_incumbent": mean,
                    "risk_delta_lo": lo,
                    "risk_delta_hi": hi,
                    "risk_signal": "yes" if (lo > 0 or hi < 0) else "no",
                })
                amean, alo, ahi = paired_bootstrap(
                    scores[rule], scores[INCUMBENT], correct,
                    _aurc_stat, n=BOOTSTRAP_N, seed=BOOTSTRAP_SEED)
                row.update({
                    "aurc_delta_vs_incumbent": amean,
                    "aurc_delta_lo": alo,
                    "aurc_delta_hi": ahi,
                    "aurc_signal": "yes" if (alo > 0 or ahi < 0) else "no",
                })
                print("    {r:11} vs {i}: risk delta {m:+.4f} "
                      "[{lo:+.4f}, {hi:+.4f}] {s}".format(
                          r=rule, i=INCUMBENT, m=mean, lo=lo, hi=hi,
                          s=("SIGNAL" if (lo > 0 or hi < 0) else "no signal")))
            for pt in points:
                row["conformal_coverage_a{a:.2f}".format(a=pt["alpha"])] = (
                    pt["coverage"])
            rows.append(row)

    _banner("STEP 3 - Writing results")
    write_csv(RESULTS_CSV, rows)

    _banner("STEP 4 - Isolation check")
    verify_isolation(before)

    _banner("VERDICT")
    comparisons = [r for r in rows if not r["is_incumbent"]]
    signals = [r for r in comparisons if r.get("risk_signal") == "yes"]
    print("  Gated axis (PRIMARY): {n} of {m} comparisons show a signal, "
          "measured on".format(n=len(signals), m=len(comparisons)))
    print("  {n} tickets rather than the four 6A had.".format(
        n=len(test_labels)))

    if not signals:
        print("  No evidence either way on the axis this project gates on.")
    else:
        # The sign is the whole result, so it is stated rather than left in a
        # CSV column. delta = risk(rule) - risk(confidence); risk is error
        # among accepted tickets, so POSITIVE means the rule is WORSE.
        worse = [r for r in signals if r["risk_delta_vs_incumbent"] > 0]
        better = [r for r in signals if r["risk_delta_vs_incumbent"] < 0]
        print("  Direction: {w} worse than the confidence gate, {b} better."
              .format(w=len(worse), b=len(better)))
        for r in worse:
            print("    {t} {r:11} +{d:.4f} risk vs confidence "
                  "[{lo:+.4f}, {hi:+.4f}]".format(
                      t=r["tier"], r=r["rule"], d=r["risk_delta_vs_incumbent"],
                      lo=r["risk_delta_lo"], hi=r["risk_delta_hi"]))
        conformal_worse = [r for r in worse if r["rule"] in ("lac", "aps")]
        if conformal_worse and not better:
            print("  Every resolving comparison favours the INCUMBENT. On "
                  "this corpus the")
            print("  conformal deferral rules are worse at the operating "
                  "coverage, not equal.")
            print("  That is evidence AGAINST promotion, and it is a stronger")
            print("  statement than 6A's 'no evidence either way' -- but it is")
            print("  measured on a DIFFERENT GENERATOR'S corpus, with a "
                  "transplanted gate")
            print("  and ~35% label accuracy. It does not retro-license a "
                  "claim about 6A.")

    print("  Achievable conformal coverages here are 0.3-3.9%, against a "
          "{c:.1%} gate --".format(c=live_cov))
    print("  so conformal still cannot be OPERATED where this system runs.")
    print("  AURC is reported in the CSV and is NOT promoted, whatever it "
          "shows.")
    return rows


def parse_args(argv):
    parser = argparse.ArgumentParser(
        description="Phase 7B secondary -- deferral rules on the external "
                    "test arm.")
    parser.add_argument("--force", action="store_true",
                        help="overwrite the existing result file")
    return parser.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    run(force=args.force)
    return 0


if __name__ == "__main__":
    sys.exit(main())
