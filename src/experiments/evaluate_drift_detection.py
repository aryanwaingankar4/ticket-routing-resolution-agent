"""
Evaluate the drift detector (Phase 4B-1) -- MEASUREMENT ONLY.

Offline. No Gemini quota. Seed 42. Loads BGE + FAISS through
artifacts.load_artifacts, only to collect decision records for the fixed sets.

WHAT THIS MEASURES
------------------
No window size or alarm threshold may be named without a measured null
false-alarm rate beside it. This script produces both halves:

  null    the false-alarm rate of every window test, per window size and alpha
  power   the detection rate under contamination with out-of-template text

organised around the four findings from building 4A:

  F2  "alpha by construction" is marginal only. The Beta(l, n+1-l) law behind
      the 0.126 bound, and this code, are verified on synthetic exchangeable
      scores. That says NOTHING about this project's data -- where the one
      fixed reference sits in that law needs held-out in-domain tickets
      (Phase 4B-2/4B-3).
  F3  alpha = 0.01 carries most of the reference check's power. Saturated
      cells are marked, never counted, and every alpha's detection results
      are recomputed leaving each reference ticket out in turn, so fragility
      is a number.
  F4  the in-domain reference escalates 0/175. Signal B is measured against
      the deployment-distribution reference instead, with a two-sample Fisher
      test beside the one-sample binomial.

THREE DEVIATIONS FROM THE APPROVED PLAN, EACH A NULL THAT WAS WRONG
------------------------------------------------------------------
1. The planned "1,000 random splits of the 175 in-domain scores" check could
   not fail. For DISTINCT scores (these 175 have no ties) a random split's
   flag count depends only on ranks, so it is exactly Beta-binomial for ANY
   175 distinct numbers -- the same trap as 4A's first reproduction check.
   Replaced by the synthetic null at the true n = 175, labelled a property of
   the procedure.
2. The first run's Signal B null bootstrapped windows from the 58-ticket half
   of each split. A split that puts more escalations in the reference leaves
   fewer in the pool, so every window carried extra between-pool variance and
   even Fisher read 0.07-0.18. That inflation was the design's, not the
   test's. The null is now parametric: reference and window counts drawn
   independently from the reference rates, scored by the library's own tests.
3. For the same reason Signal A power no longer fills its in-domain portion
   from a split pool. Each in-domain ticket carries its leave-one-out
   conformal p-value against the other 174, which is a valid conformal
   p-value, and contaminants are scored against the same 174-ticket
   reference size.

Every power number is PROVISIONAL: its in-domain portion reuses the 175
reference tickets. 4B-3 replaces it with the held-out set.

Run from the project root:
    python src/experiments/evaluate_drift_detection.py --smoke   # seconds,
                                                     # writes to a temp dir
    python src/experiments/evaluate_drift_detection.py
    python src/experiments/evaluate_drift_detection.py --force   # overwrite
"""

from __future__ import annotations

import argparse
import csv
import functools
import io
import json
import logging
import math
import os
import sys
import tempfile
from collections import Counter
from pathlib import Path

_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(_THIS_DIR, os.pardir, os.pardir))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

import numpy as np                                            # noqa: E402

from src.agent import drift                                   # noqa: E402
from src.agent.config import config_fingerprint, settings     # noqa: E402
from src.agent.conformal import conformal_p_values            # noqa: E402
from src.agent.errors import AgentError                       # noqa: E402
from src.agent.logging_setup import ensure_utf8_console        # noqa: E402

ensure_utf8_console()

DATA_DIR = os.path.join(PROJECT_ROOT, "data")
BENCHMARK_JSON = os.path.join(DATA_DIR, "novel_tickets_expanded.json")
OOD_JSON = os.path.join(DATA_DIR, "ood_calibration_tickets.json")
ADVERSARIAL_JSON = os.path.join(DATA_DIR,
                                "adversarial_escalation_tickets.json")
DEPLOYMENT_JSON = os.path.join(DATA_DIR,
                               "deployment_calibration_tickets.json")

OUTPUT_NAMES = ("drift_evaluation_null.csv", "drift_evaluation_power.csv",
                "drift_evaluation_summary.json")

SEED = 42
ALPHAS = (0.01, 0.05, 0.10, 0.20)
WINDOW_SIZES = (25, 50, 100, 200)
CONTAMINATION = (0.0, 0.05, 0.10, 0.25, 0.50)
WINDOW_LEVEL = 0.05          # a window test "alarms" at p <= this
DETECTION_TARGET = 0.80      # "min contamination to detect" means >= this

FULL_COUNTS = dict(
    beta_draws=100_000,
    null_references=1_000, null_windows_per_reference=20,   # 20,000 / cell
    crosscheck_windows=300,
    power_windows=2_000,
    rate_null_windows=4_000, rate_power_windows=1_000,
)
SMOKE_COUNTS = dict(
    beta_draws=20_000,
    null_references=40, null_windows_per_reference=5,
    crosscheck_windows=40,
    power_windows=60,
    rate_null_windows=60, rate_power_windows=30,
)

A_TESTS = ("marginal_binomial", "conditional_binomial", "ks")
B_TESTS = ("escalation_binomial", "escalation_fisher",
           "tier1_binomial", "tier1_fisher", "category_chi_square")


# --------------------------------------------------------------------------- #
# Plumbing                                                                     #
# --------------------------------------------------------------------------- #
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


def clopper_pearson(k, n, level=0.95):
    from scipy.stats import beta

    tail = (1 - level) / 2
    lo = 0.0 if k == 0 else float(beta.ppf(tail, k, n - k + 1))
    hi = 1.0 if k == n else float(beta.ppf(1 - tail, k + 1, n - k))
    return lo, hi


def expected_tickets(window, rate):
    """Tickets until the first alarm, for independent non-overlapping windows.

    Windows here are drawn independently, so windows-to-first-alarm is
    geometric and the expectation is exact: window / rate.

    Read it per row type: on a NULL row it is the mean tickets between FALSE
    alarms, which is the operational cost of a candidate operating point; on
    a POWER row the same arithmetic is tickets until DETECTION. It is never a
    false-alarm interval on a power row.
    """
    return None if rate <= 0 else window / rate


def _synthetic_reference(scores, n_escalated=0, n_tier1=0,
                         categories=None):
    n = len(scores)
    return drift.DriftReference(
        schema_version=drift.REFERENCE_SCHEMA_VERSION,
        embedding_model=settings.models.embedding_model,
        embedding_dim=settings.models.embedding_dim,
        index_ntotal=0, index_sha256="-", source_sha256="-",
        config_fingerprint="-", n=n,
        similarity_scores=[float(s) for s in scores],
        similarity_scores_with_self=[float(s) for s in scores],
        n_escalated=n_escalated, n_tier1=n_tier1,
        category_counts=categories or {"-": n},
    )


def _rate_row(signal, test, alpha, window, k, n_windows, **extra):
    lo, hi = clopper_pearson(k, n_windows)
    rate = k / n_windows
    return {"signal": signal, "test": test, "alpha": alpha,
            "window": window, "n_windows": n_windows, "count": k,
            "rate": rate, "ci_low": lo, "ci_high": hi,
            "expected_tickets_to_alarm": expected_tickets(window, rate),
            **extra}


# --------------------------------------------------------------------------- #
# Records: collected through the REAL sink, read back through the REAL reader  #
# --------------------------------------------------------------------------- #
def collect_records(name, items, artifacts, tmpdir):
    """Run each ticket and return the DecisionRecords the sink persisted.

    Going through configure_logging(decision_log_path=...) and
    DecisionRecord.from_json_line means the detector is evaluated on exactly
    what a deployment would persist, not on a re-derivation of it.
    """
    from src.agent import pipeline
    from src.agent.logging_setup import (configure_logging,
                                         detach_decision_log)
    from src.agent.schemas import TicketIn

    path = Path(tmpdir) / f"{name}.jsonl"
    configure_logging(level=logging.INFO, stream=io.StringIO(),
                      decision_log_path=path)
    try:
        for ticket_id, text in items:
            pipeline.run(TicketIn(title=text, ticket_id=ticket_id),
                         artifacts=artifacts, generate_resolution=False,
                         emit_log=True)
    finally:
        detach_decision_log()

    lines = path.read_text(encoding="utf-8").splitlines()
    ids = [json.loads(line)["ticket_id"] for line in lines]
    if ids != [ticket_id for ticket_id, _ in items]:
        _fatal(f"{name}: the sink persisted {len(lines)} records for "
               f"{len(items)} tickets, or out of order.\n  A drift history "
               "that silently drops or reorders records is the failure "
               "this\n  phase exists to catch.")
    return [drift.DecisionRecord.from_json_line(line) for line in lines]


# --------------------------------------------------------------------------- #
# Signal A window tests                                                        #
# --------------------------------------------------------------------------- #
@functools.lru_cache(maxsize=None)
def _null_rates(n_ref, alpha, delta):
    return (drift.marginal_null_rate(n_ref, alpha),
            drift.conditional_null_rate_bound(n_ref, alpha, delta))


def critical_count(w, null_rate):
    """Smallest flag count whose one-sided binomial p is <= WINDOW_LEVEL."""
    from scipy.stats import binom

    for k in range(w + 1):
        if binom.sf(k - 1, w, null_rate) <= WINDOW_LEVEL:
            return k
    return w + 1


def window_tests_scalar(ref_scores, window_sims, alpha, delta):
    """(flagged, marginal_p, conditional_p, ks_p) for one window.

    The readable reference implementation. The vectorised path below is
    checked against this, and this against drift.novelty_signal.
    """
    from scipy.stats import binom, kstest

    p = conformal_p_values([-s for s in ref_scores],
                           [-s for s in window_sims])
    w = len(window_sims)
    k = int(np.sum(p <= alpha))
    m, b = _null_rates(len(ref_scores), alpha, delta)
    # binomtest(alternative="greater") is P(X >= k) = sf(k - 1).
    return (k, float(binom.sf(k - 1, w, m)), float(binom.sf(k - 1, w, b)),
            float(kstest(p, "uniform").pvalue))


def window_tests_matrix(p, n_ref, alpha, delta):
    """Vectorised Signal A over a (windows x W) matrix of p-values.

    Returns {test: array of window p-values}.
    """
    from scipy.stats import binom, kstwo

    w = p.shape[1]
    k = np.sum(p <= alpha, axis=1)
    m, b = _null_rates(n_ref, alpha, delta)
    ps = np.sort(p, axis=1)
    i = np.arange(1, w + 1)
    d = np.maximum(np.max(i / w - ps, axis=1),
                   np.max(ps - (i - 1) / w, axis=1))
    return {
        "marginal_binomial": binom.sf(k - 1, w, m),
        "conditional_binomial": binom.sf(k - 1, w, b),
        "ks": np.clip(kstwo.sf(d, w), 0.0, 1.0),
    }


def crosscheck_signal_a(rng, delta, n_windows):
    """Vectorised == scalar == drift.novelty_signal, or nothing is trusted."""
    worst_lib = worst_vec = 0.0
    for i in range(n_windows):
        ref = rng.random(175)
        window = rng.random(int(rng.choice(WINDOW_SIZES)))
        alpha = float(ALPHAS[i % len(ALPHAS)])
        k, pm, pc, pks = window_tests_scalar(ref, window, alpha, delta)

        lib = drift.novelty_signal(list(window), _synthetic_reference(ref),
                                   alpha, delta)
        if lib.flagged != k:
            _fatal(f"flag count {k} != library {lib.flagged}")
        for mine, theirs in ((pm, lib.marginal_binomial_p),
                             (pc, lib.conditional_binomial_p),
                             (pks, lib.ks_p)):
            worst_lib = max(worst_lib, abs(mine - theirs))

        pv = conformal_p_values([-s for s in ref], [-s for s in window])
        vec = window_tests_matrix(pv[None, :], 175, alpha, delta)
        for test, scalar in zip(A_TESTS, (pm, pc, pks)):
            worst_vec = max(worst_vec, abs(float(vec[test][0]) - scalar))
    if worst_lib > 1e-9 or worst_vec > 1e-9:
        _fatal(f"Signal A paths disagree: scalar vs library {worst_lib:.3g},"
               f" vectorised vs scalar {worst_vec:.3g}")
    return {"windows": n_windows, "max_abs_p_scalar_vs_library": worst_lib,
            "max_abs_p_vectorised_vs_scalar": worst_vec}


# --------------------------------------------------------------------------- #
# F2 -- the Beta law behind the conditional bound                              #
# --------------------------------------------------------------------------- #
def beta_law_check(rng, delta, draws, strict):
    rows = []
    n = 175
    for alpha in ALPHAS:
        l = math.floor((n + 1) * alpha)
        # Realised rate of a fixed reference on continuous exchangeable
        # scores: P(test similarity < l-th smallest reference similarity).
        realised = np.empty(draws)
        for start in range(0, draws, 10_000):
            stop = min(start + 10_000, draws)
            block = np.sort(rng.random((stop - start, n)), axis=1)
            realised[start:stop] = block[:, l - 1]
        marginal = drift.marginal_null_rate(n, alpha)
        bound = drift.conditional_null_rate_bound(n, alpha, delta)
        empirical_q = float(np.quantile(realised, 1 - delta))
        above = float(np.mean(realised > bound))
        rows.append({
            "alpha": alpha, "l": l, "draws": draws,
            "marginal_closed_form": marginal,
            "realised_mean": float(realised.mean()),
            "conditional_bound_closed_form": bound,
            "realised_quantile_1_minus_delta": empirical_q,
            "share_of_references_above_bound": above,
            "bound_over_marginal": bound / marginal,
        })
        print(f"  a={alpha:<5} l={l:<3} marginal {marginal:.4f} vs mean "
              f"{realised.mean():.4f} | bound {bound:.4f} vs q{1 - delta:.2f}"
              f" {empirical_q:.4f} | above bound {above:.3f}")
        if strict and (abs(realised.mean() - marginal) > 5e-4
                       or abs(empirical_q - bound) > 2e-3):
            _fatal(f"The Beta law does not reproduce at alpha={alpha}. The "
                   "bound or its\n  code is wrong; no real-data number below "
                   "can be trusted.")
    return rows


# --------------------------------------------------------------------------- #
# Signal A null -- synthetic exchangeable scores at the true n = 175           #
# --------------------------------------------------------------------------- #
def novelty_null(rng, delta, counts):
    from scipy.stats import betabinom, binom

    n = 175
    n_refs = counts["null_references"]
    per_ref = counts["null_windows_per_reference"]
    total = n_refs * per_ref
    rows = []
    for w in WINDOW_SIZES:
        for alpha in ALPHAS:
            l = math.floor((n + 1) * alpha)
            m, b = _null_rates(n, alpha, delta)
            per_ref_rates = {t: np.empty(n_refs) for t in A_TESTS}
            for j in range(n_refs):
                ref = rng.random(n)
                p = conformal_p_values(-ref, -rng.random(per_ref * w))
                result = window_tests_matrix(p.reshape(per_ref, w), n,
                                             alpha, delta)
                for test in A_TESTS:
                    per_ref_rates[test][j] = np.mean(
                        result[test] <= WINDOW_LEVEL)
            for test in A_TESTS:
                k = int(round(per_ref_rates[test].sum() * per_ref))
                exact = at_bound = None
                if test != "ks":
                    k_star = critical_count(
                        w, m if test == "marginal_binomial" else b)
                    # Exact: Beta-binomial is the flag count marginal over
                    # references; at_bound is a reference sitting exactly on
                    # the conditional bound.
                    exact = float(betabinom.sf(k_star - 1, w, l, n + 1 - l))
                    at_bound = float(binom.sf(k_star - 1, w, b))
                row = _rate_row(
                    "A", test, alpha, w, k, total,
                    null_source="synthetic_exchangeable_n175",
                    exact_marginal=exact,
                    at_conditional_bound=at_bound,
                    q90_over_references=float(
                        np.quantile(per_ref_rates[test], 0.90)),
                    undefined_p=0,
                )
                row["eligible"] = row["ci_high"] <= WINDOW_LEVEL
                rows.append(row)
            print(f"  W={w:<4} a={alpha:<5} "
                  + "  ".join(f"{r['test'][:4]} {r['rate']:.4f}"
                              f"{'*' if r['eligible'] else ' '}"
                              for r in rows[-3:]))
    return rows


# --------------------------------------------------------------------------- #
# Signal A power -- leave-one-out null p-values, reference size 174            #
# --------------------------------------------------------------------------- #
def loo_null_p_values(reference_scores):
    """Each reference ticket's conformal p-value against the other 174."""
    s = np.asarray(reference_scores, dtype=np.float64)
    out = np.empty(len(s))
    for i in range(len(s)):
        out[i] = conformal_p_values(-np.delete(s, i), [-s[i]])[0]
    return out


def contaminant_p_table(reference_scores, sims):
    """(175 x len(sims)): p-values against the reference minus ticket j."""
    s = np.asarray(reference_scores, dtype=np.float64)
    return np.stack([conformal_p_values(-np.delete(s, j),
                                        -np.asarray(sims))
                     for j in range(len(s))])


def _contaminant_indexer(n_items, seeds):
    if seeds is None:
        return lambda rng: int(rng.integers(n_items))
    # OOD variants of one seed are not independent: pick the seed, then a
    # variant, so a window cannot count three near-copies as three draws.
    groups = {}
    for idx, seed in enumerate(seeds):
        groups.setdefault(seed, []).append(idx)
    groups = list(groups.values())

    def pick(rng):
        group = groups[int(rng.integers(len(groups)))]
        return group[int(rng.integers(len(group)))]
    return pick


def novelty_power(rng, reference_scores, contaminants, delta, n_windows):
    n_ref = len(reference_scores) - 1
    null_p = loo_null_p_values(reference_scores)
    tables = {name: (contaminant_p_table(reference_scores, sims),
                     _contaminant_indexer(len(sims), seeds))
              for name, (sims, seeds) in contaminants.items()}
    rows = []
    for w in WINDOW_SIZES:
        for c in CONTAMINATION:
            n_bad = int(round(c * w))
            n_good = w - n_bad
            sources = ("none",) if n_bad == 0 else tuple(tables)
            for source in sources:
                p = np.empty((n_windows, w))
                for r in range(n_windows):
                    p[r, :n_good] = rng.choice(
                        null_p, size=n_good,
                        replace=n_good > len(null_p))
                    if n_bad:
                        table, pick = tables[source]
                        j = int(rng.integers(table.shape[0]))
                        p[r, n_good:] = [table[j, pick(rng)]
                                         for _ in range(n_bad)]
                for alpha in ALPHAS:
                    result = window_tests_matrix(p, n_ref, alpha, delta)
                    for test in A_TESTS:
                        k = int(np.sum(result[test] <= WINDOW_LEVEL))
                        row = _rate_row(
                            "A", test, alpha, w, k, n_windows,
                            contaminant=source, contamination=c,
                            n_contaminants=n_bad,
                            reference="leave_one_out_174")
                        row["saturated"] = k == n_windows
                        rows.append(row)
        print(f"  W={w:<4} done")
    return rows


def min_contamination_to_detect(power_rows):
    """Smallest contamination with detection >= target, per operating point.

    Unlike a detection rate at a fixed contamination, this cannot saturate:
    it moves whenever power moves.
    """
    keyed = {}
    for r in power_rows:
        if r["contaminant"] == "none":
            continue
        key = "|".join(str(x) for x in (r["signal"], r["test"], r["alpha"],
                                        r["window"], r["contaminant"]))
        keyed.setdefault(key, []).append(r)
    out = {}
    for key, rows in sorted(keyed.items()):
        hit = [r["contamination"]
               for r in sorted(rows, key=lambda r: r["contamination"])
               if r["rate"] >= DETECTION_TARGET]
        out[key] = hit[0] if hit else None
    return out


# --------------------------------------------------------------------------- #
# F3 -- leave-one-out fragility of the detection results                       #
# --------------------------------------------------------------------------- #
def detection_metrics(ref, alpha, sets):
    cal = [-s for s in ref]
    out = {}
    for name, (sims, seeds) in sets.items():
        flagged = conformal_p_values(cal, [-s for s in sims]) <= alpha
        out[f"{name}_flag_rate"] = float(np.mean(flagged))
        if seeds is not None:
            out[f"{name}_seed_rate"] = len(
                {s for s, f in zip(seeds, flagged) if f}) / len(set(seeds))
    return out


def leave_one_out(reference_scores, sets):
    rows = {}
    for alpha in ALPHAS:
        full = detection_metrics(reference_scores, alpha, sets)
        drops = [detection_metrics(reference_scores[:i]
                                   + reference_scores[i + 1:], alpha, sets)
                 for i in range(len(reference_scores))]
        per_metric = {}
        for metric, value in full.items():
            values = [d[metric] for d in drops]
            per_metric[metric] = {
                "full_reference": value, "min": min(values),
                "max": max(values), "range": max(values) - min(values),
                "drops_that_change_it": sum(v != value for v in values),
            }
        rows[str(alpha)] = per_metric
        for metric, v in per_metric.items():
            if v["range"] > 0:
                print(f"  a={alpha:<5} {metric:<26} full "
                      f"{v['full_reference']:.3f}  range {v['min']:.3f}.."
                      f"{v['max']:.3f}  ({v['drops_that_change_it']} of "
                      f"{len(reference_scores)} drops move it)")
    return rows


# --------------------------------------------------------------------------- #
# The realistic-traffic arm                                                    #
# --------------------------------------------------------------------------- #
def realistic_traffic(rng, reference_scores, deployment_sims, delta,
                      n_windows):
    n = len(reference_scores)
    p_all = conformal_p_values([-s for s in reference_scores],
                               [-s for s in deployment_sims])
    per_ticket = []
    for alpha in ALPHAS:
        k = int(np.sum(p_all <= alpha))
        m, b = _null_rates(n, alpha, delta)
        per_ticket.append({"alpha": alpha, "flagged": k,
                           "n": len(deployment_sims),
                           "flag_rate": k / len(deployment_sims),
                           "marginal_null_rate": m,
                           "conditional_bound": b})
        print(f"  a={alpha:<5} deployment tickets flagged {k}/"
              f"{len(deployment_sims)} = {k / len(deployment_sims):.3f} "
              f"(null {m:.3f}, bound {b:.3f})")
    rows = []
    for w in WINDOW_SIZES:
        p = p_all[rng.integers(len(p_all), size=(n_windows, w))]
        for alpha in ALPHAS:
            result = window_tests_matrix(p, n, alpha, delta)
            for test in A_TESTS:
                k = int(np.sum(result[test] <= WINDOW_LEVEL))
                rows.append(_rate_row("A", test, alpha, w, k, n_windows,
                                      contaminant="deployment_register",
                                      contamination=1.0,
                                      reference="in_domain_175"))
    return per_ticket, rows


# --------------------------------------------------------------------------- #
# Signal B -- parametric null and power against the deployment reference       #
# --------------------------------------------------------------------------- #
def _category_vector(counts, names):
    total = sum(counts.values())
    return np.array([counts.get(c, 0) / total for c in names])


def rate_windows(rng, ref_n, rates, contaminant_rates, w, c, n_windows):
    """Alarm and undefined counts per Signal B test.

    Reference counts and window counts are drawn independently from the
    reference rates, so the reference's own estimation noise is in the null
    -- the thing the binomial ignores and Fisher does not. Scored by the
    library's own test functions.
    """
    names = rates["category_names"]
    n_bad = int(round(c * w))
    alarms, undefined = Counter(), Counter()
    for _ in range(n_windows):
        ref_esc = int(rng.binomial(ref_n, rates["escalation"]))
        ref_t1 = int(rng.binomial(ref_n, rates["tier1"]))
        ref_cat = rng.multinomial(ref_n, rates["categories"])
        win_esc = int(rng.binomial(w - n_bad, rates["escalation"]))
        win_t1 = int(rng.binomial(w - n_bad, rates["tier1"]))
        win_cat = rng.multinomial(w - n_bad, rates["categories"])
        if n_bad:
            win_esc += int(rng.binomial(n_bad,
                                        contaminant_rates["escalation"]))
            win_t1 += int(rng.binomial(n_bad, contaminant_rates["tier1"]))
            win_cat = win_cat + rng.multinomial(
                n_bad, contaminant_rates["categories"])

        esc = drift._rate_test(win_esc, w, ref_esc, ref_n)
        t1 = drift._rate_test(win_t1, w, ref_t1, ref_n)
        reference = _synthetic_reference(
            [0.0] * ref_n,
            categories={nm: int(x) for nm, x in zip(names, ref_cat) if x})
        window_categories = [nm for nm, x in zip(names, win_cat)
                             for _ in range(int(x))]
        mix = drift.category_mix_test(window_categories, reference)

        for test, pv in (("escalation_binomial", esc.binomial_p_two_sided),
                         ("escalation_fisher", esc.fisher_p_two_sided),
                         ("tier1_binomial", t1.binomial_p_two_sided),
                         ("tier1_fisher", t1.fisher_p_two_sided),
                         ("category_chi_square", mix.p_value)):
            if pv is None:
                undefined[test] += 1
            else:
                alarms[test] += int(pv <= WINDOW_LEVEL)
    return alarms, undefined


def _rates_from_records(records, names):
    n = len(records)
    return {
        "escalation": sum(r.escalated for r in records) / n,
        "tier1": sum(r.tier == 1 for r in records) / n,
        "categories": _category_vector(
            Counter(r.category for r in records), names),
        "category_names": names,
    }


def rate_null_and_power(rng, rate_ref, contaminant_records, counts):
    names = sorted(set(rate_ref.category_counts)
                   | {r.category for recs in contaminant_records.values()
                      for r in recs})
    rates = {
        "escalation": rate_ref.escalation_rate,
        "tier1": rate_ref.tier1_share,
        "categories": _category_vector(rate_ref.category_counts, names),
        "category_names": names,
    }
    null_rows, power_rows = [], []
    for w in WINDOW_SIZES:
        n_win = counts["rate_null_windows"]
        alarms, undefined = rate_windows(rng, rate_ref.n, rates, None, w,
                                         0.0, n_win)
        for test in B_TESTS:
            # A window whose p-value is undefined cannot raise a false alarm,
            # so the rate is over the windows where the test was defined.
            defined = n_win - undefined[test]
            row = _rate_row("B", test, None, w, alarms[test],
                            max(defined, 1),
                            null_source="parametric_deployment_reference",
                            exact_marginal=None, at_conditional_bound=None,
                            q90_over_references=None,
                            undefined_p=undefined[test])
            row["eligible"] = (row["ci_high"] <= WINDOW_LEVEL
                               and undefined[test] == 0)
            null_rows.append(row)
        print(f"  null W={w:<4} "
              + "  ".join(f"{r['test']} {r['rate']:.3f}"
                          f"{'*' if r['eligible'] else ''}"
                          for r in null_rows[-len(B_TESTS):]))

        for source, records in contaminant_records.items():
            c_rates = _rates_from_records(records, names)
            for c in CONTAMINATION[1:]:
                n_win = counts["rate_power_windows"]
                alarms, undefined = rate_windows(rng, rate_ref.n, rates,
                                                 c_rates, w, c, n_win)
                for test in B_TESTS:
                    row = _rate_row("B", test, None, w, alarms[test], n_win,
                                    contaminant=source, contamination=c,
                                    n_contaminants=int(round(c * w)),
                                    reference="parametric_deployment_175",
                                    undefined_p=undefined[test])
                    row["saturated"] = alarms[test] == n_win
                    power_rows.append(row)
    return null_rows, power_rows


# --------------------------------------------------------------------------- #
# Main                                                                         #
# --------------------------------------------------------------------------- #
def _write_csv(path, rows):
    fields = []
    for row in rows:
        fields += [k for k in row if k not in fields]
    with open(path, "w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def _json_default(value):
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return float(value)
    if isinstance(value, np.bool_):
        return bool(value)
    raise TypeError(f"not JSON serialisable: {type(value).__name__}")


def main():
    parser = argparse.ArgumentParser(
        description="Evaluate the drift detector (Phase 4B-1).")
    parser.add_argument("--force", action="store_true",
                        help="overwrite existing evaluation outputs")
    parser.add_argument("--smoke", action="store_true",
                        help="tiny counts, outputs to a temp dir; exercises "
                             "every step including the writers")
    args = parser.parse_args()

    counts = SMOKE_COUNTS if args.smoke else FULL_COUNTS
    out_dir = tempfile.mkdtemp(prefix="drift_eval_smoke_") if args.smoke \
        else DATA_DIR
    out_paths = [os.path.join(out_dir, name) for name in OUTPUT_NAMES]
    existing = [p for p in out_paths if os.path.exists(p)]
    if existing and not args.force:
        _fatal("Evaluation outputs already exist:\n    "
               + "\n    ".join(existing)
               + "\n  Refusing to overwrite published results silently. "
               "Re-run with --force.")

    delta = settings.drift.conditional_delta
    rng = np.random.default_rng(SEED)

    # Decision records go to the temporary sink only; keep the console quiet.
    from src.agent.logging_setup import configure_logging
    configure_logging(level=logging.INFO, stream=io.StringIO())

    _banner("STEP 1 - Artifacts, both references, and decision records")
    try:
        from src.agent.artifacts import load_artifacts, load_drift_reference
        from src.experiments.build_drift_reference import (
            _check_against_published)
        from src.experiments.calibrate_conformal import seed_key

        artifacts = load_artifacts(require_gemini=False)
        novelty_ref = load_drift_reference(artifacts, source="in-domain")
        rate_ref = load_drift_reference(artifacts, source="deployment")
    except AgentError as exc:
        _fatal(str(exc))

    benchmark = _read_json(BENCHMARK_JSON, "45-ticket benchmark")
    ood = _read_json(OOD_JSON, "OOD set")
    adversarial = _read_json(ADVERSARIAL_JSON, "Adversarial set")
    deployment = _read_json(DEPLOYMENT_JSON, "Deployment calibration set")

    with tempfile.TemporaryDirectory() as tmpdir:
        records = {
            "benchmark": collect_records(
                "benchmark", [(f"bench_{i}", r["text"])
                              for i, r in enumerate(benchmark)],
                artifacts, tmpdir),
            "ood": collect_records(
                "ood", [(str(r["id"]), r["text"]) for r in ood],
                artifacts, tmpdir),
            "adversarial": collect_records(
                "adversarial", [(str(r["id"]), r["text"])
                                for r in adversarial], artifacts, tmpdir),
            "deployment": collect_records(
                "deployment", [(str(r["id"]), r["text"])
                               for r in deployment], artifacts, tmpdir),
        }
    for name, recs in records.items():
        print(f"  {name:<12} {len(recs):>4} records, escalated "
              f"{sum(r.escalated for r in recs)}, tier-1 "
              f"{sum(r.tier == 1 for r in recs)}")

    # Two independent derivations of the same facts, both fatal.
    dep = records["deployment"]
    if (sum(r.escalated for r in dep), sum(r.tier == 1 for r in dep),
            dict(Counter(r.category for r in dep))) != (
            rate_ref.n_escalated, rate_ref.n_tier1, rate_ref.category_counts):
        _fatal("Deployment records read back from the sink do not reproduce "
               "the committed\n  Signal B reference. The sink, the reader or "
               "the reference is stale.")
    sims = {k: [r.top_similarity for r in v] for k, v in records.items()}
    ood_seeds = [seed_key(r["id"]) for r in ood]
    print("\n  Sink-recorded similarities against published Phase 1 "
          "detection results:")
    _check_against_published(novelty_ref.similarity_scores_with_self,
                             novelty_ref.similarity_scores, sims["ood"],
                             ood_seeds, sims["adversarial"])

    _banner("STEP 2 - F2: the Beta law behind the 0.126 bound (synthetic)")
    beta_rows = beta_law_check(rng, delta, counts["beta_draws"],
                               strict=not args.smoke)
    crosscheck = crosscheck_signal_a(rng, delta,
                                     counts["crosscheck_windows"])
    print(f"  scalar == drift.novelty_signal == vectorised on "
          f"{crosscheck['windows']} windows")

    _banner("STEP 3 - Signal A null: synthetic exchangeable, n = 175 "
            "(* = eligible)")
    null_rows = novelty_null(rng, delta, counts)

    _banner("STEP 4 - Signal A power (PROVISIONAL: leave-one-out, n = 174)")
    power_rows = novelty_power(rng, list(novelty_ref.similarity_scores), {
        "benchmark": (sims["benchmark"], None),
        "ood": (sims["ood"], ood_seeds),
        "adversarial": (sims["adversarial"], None),
    }, delta, counts["power_windows"])

    _banner("STEP 5 - F3: leave-one-out fragility (only metrics that move "
            "are printed)")
    loo = leave_one_out(list(novelty_ref.similarity_scores), {
        "ood": (sims["ood"], ood_seeds),
        "adversarial": (sims["adversarial"], None),
        "benchmark": (sims["benchmark"], None),
        "deployment": (sims["deployment"], None),
    })

    _banner("STEP 6 - Realistic-traffic arm: deployment-register tickets "
            "vs the in-domain reference")
    realistic_ticket, realistic_rows = realistic_traffic(
        rng, list(novelty_ref.similarity_scores), sims["deployment"], delta,
        counts["power_windows"])

    _banner("STEP 7 - F4: Signal B against the deployment reference "
            "(parametric, * = eligible)")
    b_null, b_power = rate_null_and_power(rng, rate_ref, {
        "benchmark": records["benchmark"], "ood": records["ood"],
    }, counts)
    null_rows += b_null
    power_rows += b_power + realistic_rows

    _banner("STEP 8 - Pre-registered eligibility and outputs")
    eligible = [r for r in null_rows if r["eligible"]]
    print(f"  eligible operating points: {len(eligible)}/{len(null_rows)}")
    for r in eligible:
        print(f"    {r['signal']} {r['test']:<22} a={r['alpha']} "
              f"W={r['window']:<4} FA {r['rate']:.4f} "
              f"(CI high {r['ci_high']:.4f})")

    summary = {
        "phase": "4B-1",
        "smoke": args.smoke,
        "status": ("PROVISIONAL -- Signal A power reuses the 175 reference "
                   "tickets as its in-domain portion; 4B-3 replaces it with "
                   "the held-out in-domain set"),
        "config_fingerprint": config_fingerprint(),
        "seed": SEED,
        "counts": counts,
        "window_level": WINDOW_LEVEL,
        "detection_target": DETECTION_TARGET,
        "references": {
            "signal_a": {"file": settings.drift.reference_name,
                         "source_sha256": novelty_ref.source_sha256,
                         "n": novelty_ref.n,
                         "n_escalated": novelty_ref.n_escalated},
            "signal_b": {"file": settings.drift.rate_reference_name,
                         "source_sha256": rate_ref.source_sha256,
                         "n": rate_ref.n,
                         "n_escalated": rate_ref.n_escalated},
        },
        "f2_beta_law_check": beta_rows,
        "signal_a_crosscheck": crosscheck,
        "deviations_from_plan": [
            "Split check dropped: for distinct scores a random split's flag "
            "count is exactly Beta-binomial whatever the data, so it could "
            "not fail. The in-domain scores have no ties.",
            "Signal B null made parametric: bootstrapping windows from a "
            "58-ticket split pool anti-correlates pool and reference and "
            "inflated every test's false-alarm rate, Fisher's included.",
            "Signal A power uses leave-one-out conformal p-values against "
            "174 reference tickets instead of a 117/58 split, for the same "
            "reason.",
        ],
        "f3_leave_one_out": loo,
        "f3_min_contamination_for_detection": min_contamination_to_detect(
            [r for r in power_rows if r["contaminant"]
             != "deployment_register"]),
        "realistic_traffic_per_ticket": realistic_ticket,
        "eligible_operating_points": [
            {k: r[k] for k in ("signal", "test", "alpha", "window", "rate",
                               "ci_high")}
            for r in eligible],
        "decision_rule": (
            "An operating point is eligible only if the upper end of the 95% "
            "Clopper-Pearson CI on its measured null false-alarm rate is <= "
            "0.05 (and, for Signal B, its p-value is never undefined). Power "
            "is reported only among eligible points and never used to pick "
            "among ineligible ones."),
        "limitations": [
            "The Signal A null is a property of the procedure under "
            "exchangeability, not of this project's data; where the fixed "
            "reference sits needs held-out tickets (4B-2/4B-3).",
            "Signal A power reuses the 175 reference tickets as its "
            "in-domain portion, and resamples 9-45 contaminants.",
            "Signal B nulls depend only on the reference rates, so they "
            "calibrate the test procedure, not the data.",
            "Contamination is abrupt and out-of-template; gradual shift is "
            "not measured.",
            "Selection risk accumulates on the 45-ticket benchmark.",
            "Template-generated data makes the null unusually clean.",
            "The Signal B reference rate (22.3%) is that of Gemini-generated "
            "benchmark-register tickets, not a production escalation rate.",
            "The realistic-traffic arm bootstraps its windows from the same "
            "175 deployment p-values with replacement, so those windows are "
            "not independent draws from the deployment register either.",
            "One RNG stream (seed 42) is threaded through every step, so "
            "these numbers reproduce only while the step order is unchanged.",
        ],
    }

    null_path, power_path, summary_path = out_paths
    _write_csv(null_path, null_rows)
    _write_csv(power_path, power_rows)
    with open(summary_path, "w", encoding="utf-8") as fh:
        json.dump(summary, fh, indent=2, sort_keys=True,
                  default=_json_default)
        fh.write("\n")
    for path in out_paths:
        print(f"  {path}")

    _banner("DONE")


if __name__ == "__main__":
    main()
