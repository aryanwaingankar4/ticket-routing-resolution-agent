"""
Conformal prediction calibration and coverage measurement.

MEASUREMENT ONLY. This script never changes a production threshold. It fits
split-conformal predictors, measures whether their guarantees actually hold on
this project's data, and writes the evidence to CSV. Promotion to production is
a separate, later decision -- the precedent set by refusing to promote BGE
clustering on cliff-edge math alone.

WHAT IT MEASURES, AND WHY EACH AXIS EXISTS
------------------------------------------
1. CONTAMINATION. train_cascade.py:1336-1338 scores the 175-ticket calibration
   set with models fit on all 4,000 rows -- and those 175 tickets are Gemini
   paraphrases of rows drawn from that same CSV. The model has seen the
   original of every ticket it is calibrated on. For conformal this is fatal in
   a specific direction: memorised sources depress nonconformity scores, which
   shrinks q_hat, which shrinks prediction sets, which puts coverage BELOW
   nominal. The guarantee fails in the reassuring direction. Both the
   contaminated and de-contaminated fits are computed so the size of the effect
   is measured rather than asserted.

2. EXCHANGEABILITY. Conformal is distribution-free but not assumption-free. The
   calibration set is in-domain paraphrase; the 45-ticket benchmark is plain,
   non-template English the models score ~71% on versus ~100% in-distribution.
   Coverage on the benchmark is the number that matters.

3. LABEL NOISE. 21 of the 175 are `flagged` -- Gemini's own category guess
   disagreed with the ground-truth label and they were kept anyway. Label noise
   breaks coverage directly, so everything is computed on all 175 and on the
   154 clean records. Filtering is not free: it destroys the 25-per-class
   balance (Infrastructure loses 6, Access Management loses 0), which matters
   for the Mondrian variant specifically.

4. TIER. Conformal is fitted on BOTH tiers, because they answer different
   questions. Tier-1 sets answer "can the cheap model resolve this alone?" --
   the decision the 0.50 gate currently makes. Tier-2 sets answer "can the
   system answer at all, or does a human need to see it?" -- the more valuable
   one, and the closer analogue to the RAG gate's job.

HOW TO READ A COVERAGE SHORTFALL
--------------------------------
The conformal guarantee is MARGINAL over the calibration draw, not conditional
on it. With one calibration set of n=175 a coverage estimate carries a standard
deviation of roughly 2 percentage points at alpha=0.10 (see the discussion in
tests/test_conformal.py). A shortfall of a point or two is noise; the
exchangeability claim rests on a gap materially larger than that, which is why
the expected standard deviation is printed alongside every figure.

Run from the project root (offline, no Gemini calls, no quota):
    python -m src.experiments.calibrate_conformal

Outputs are refused rather than overwritten if they already exist. Write a new
set beside the published one with --out-suffix <name>, or overwrite with
--force. The published conformal_novelty_results.csv is also read by
build_drift_reference.py as its reference check, so overwriting it in place
would undermine that check as well as rule 4.
"""

from __future__ import annotations

import csv
import json
import os
import sys

import numpy as np
import pandas as pd

_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(_THIS_DIR, os.pardir, os.pardir))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from src.agent import conformal as cp                       # noqa: E402
from src.agent.config import settings                       # noqa: E402
from src.agent.logging_setup import ensure_utf8_console      # noqa: E402

ensure_utf8_console()

DATA_DIR = os.path.join(PROJECT_ROOT, "data")
CALIBRATION_JSON = os.path.join(DATA_DIR,
                                "calibration_tickets_paraphrased.json")
BENCHMARK_JSON = os.path.join(DATA_DIR, "novel_tickets_expanded.json")
OOD_JSON = os.path.join(DATA_DIR, "ood_calibration_tickets.json")
ADVERSARIAL_JSON = os.path.join(DATA_DIR,
                                "adversarial_escalation_tickets.json")
# Optional: the deployment-distribution calibration set, if it has been
# generated. Its whole purpose is to satisfy the exchangeability
# assumption the in-domain set provably cannot -- see
# generate_deployment_calibration_set.py.
DEPLOYMENT_JSON = os.path.join(DATA_DIR,
                               "deployment_calibration_tickets.json")
EMBEDDINGS_NPY = os.path.join(DATA_DIR,
                              "ticket_embeddings_bge-base-en-v1-5.npy")

OUTPUT_CSV = os.path.join(DATA_DIR, "conformal_calibration_results.csv")
NOVELTY_CSV = os.path.join(DATA_DIR, "conformal_novelty_results.csv")
ARTIFACT_JSON = os.path.join(
    DATA_DIR, "conformal_calibration_bge-base-en-v1-5.json")


def output_paths(suffix=""):
    """The three result files, optionally suffixed.

    Project rule: a new result gets a new filename rather than overwriting a
    published one. `build_drift_reference.py` reads the UNSUFFIXED
    conformal_novelty_results.csv as its published reference check, so
    overwriting these in place would also quietly undermine that check.
    """
    tag = f"_{suffix}" if suffix else ""
    return (
        os.path.join(DATA_DIR, f"conformal_calibration_results{tag}.csv"),
        os.path.join(DATA_DIR, f"conformal_novelty_results{tag}.csv"),
        os.path.join(DATA_DIR,
                     f"conformal_calibration{tag}_bge-base-en-v1-5.json"),
    )


ALPHAS = [0.20, 0.10, 0.05, 0.01]
SCORE_FUNCTIONS = ["lac", "aps"]
RANDOM_STATE = 42


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


# --------------------------------------------------------------------------- #
# Loading
# --------------------------------------------------------------------------- #
def _read_json(path, what):
    if not os.path.isfile(path):
        _fatal(f"{what} not found:\n    {path}")
    with open(path, "r", encoding="utf-8") as fh:
        return json.load(fh)


def load_everything():
    if not os.path.isfile(settings.models.dataset_path):
        _fatal(f"Dataset not found: {settings.models.dataset_path}\n"
               "  Generate it with: python data/generate_dataset.py")
    df = pd.read_csv(settings.models.dataset_path)

    if not os.path.isfile(EMBEDDINGS_NPY):
        _fatal(
            f"Embedding cache not found:\n    {EMBEDDINGS_NPY}\n"
            "  Build it with: python src/classification/train_embeddings.py"
        )
    embeddings = np.load(EMBEDDINGS_NPY)

    if embeddings.shape[0] != len(df):
        _fatal(
            "EMBEDDING CACHE / DATASET OUT OF SYNC:\n"
            f"    embeddings rows = {embeddings.shape[0]}\n"
            f"    dataset rows    = {len(df)}\n"
            "  Regenerate the cache with train_embeddings.py."
        )
    if embeddings.shape[1] != settings.models.embedding_dim:
        _fatal(
            f"Embedding cache is {embeddings.shape[1]}-dim but config expects "
            f"{settings.models.embedding_dim}. Wrong model's cache."
        )

    calibration = _read_json(CALIBRATION_JSON, "Calibration set")
    benchmark = _read_json(BENCHMARK_JSON, "45-ticket benchmark")
    ood = _read_json(OOD_JSON, "OOD calibration set")
    adversarial = _read_json(ADVERSARIAL_JSON, "Adversarial set")
    return df, embeddings, calibration, benchmark, ood, adversarial


def assert_calibration_disjoint(df, calibration, excluded_ids):
    """Runtime guard for a discipline that was previously prose-only.

    train_cascade.py:62-64 and :245-250 assert in comments that the
    calibration set stays disjoint from the evaluation benchmarks, enforced
    only by construction. This checks it.
    """
    cal_ids = {int(r["id"]) for r in calibration}

    missing = cal_ids - set(df["id"].astype(int))
    if missing:
        _fatal(
            f"{len(missing)} calibration source id(s) are not in the dataset "
            f"(e.g. {sorted(missing)[:5]}). The calibration set and the "
            "dataset have diverged."
        )
    if not cal_ids.issubset(excluded_ids):
        leaked = sorted(cal_ids - excluded_ids)[:5]
        _fatal(
            "DE-CONTAMINATION FAILED: calibration source id(s) remain in the "
            f"training set (e.g. {leaked}). The scoring model would have seen "
            "the originals of tickets it is calibrated on, which biases "
            "coverage below nominal."
        )
    print(f"[guard] {len(cal_ids)} calibration source ids confirmed excluded "
          f"from the scoring models' training data.")


def report_contamination_structure(df, calibration):
    """Measure WHY row-level de-contamination cannot work here.

    Removing a calibration ticket's source row does not remove what the model
    memorised, because this dataset is template-generated: every scenario
    template has hundreds of near-identical siblings. Deleting one row leaves
    the rest of its template intact, so the model still recognises the
    paraphrase.

    The obvious stronger move -- excluding every row sharing a template with
    any calibration ticket -- is measured here too, and turns out to be
    impossible rather than merely expensive. Printed as part of the run so the
    conclusion is reproducible rather than an ad-hoc observation.
    """
    _banner("STEP 1b - Why de-contamination by row removal cannot work")

    # A template's identity is (category, scenario_id), NOT scenario_id alone.
    # scenario_id is an index WITHIN a category -- data/generate_dataset.py:634
    # says so in its own comment -- so grouping by it alone silently merges
    # seven categories' templates into one, and reports 12 templates of ~430
    # rows where there are 66 of ~62. Phase 5A corrected exactly that; see the
    # README correction note under Finding 2.
    TEMPLATE_KEY = ["category", "scenario_id"]

    cal_ids = {int(r["id"]) for r in calibration}
    grouped = df.groupby(TEMPLATE_KEY)
    total_templates = int(grouped.ngroups)
    cal_templates = set(
        map(tuple, df[df["id"].isin(cal_ids)][TEMPLATE_KEY]
            .drop_duplicates().to_numpy()))
    surviving = int(
        (~pd.MultiIndex.from_frame(df[TEMPLATE_KEY]).isin(cal_templates))
        .sum())
    sizes = grouped.size()
    median_rows = int(sizes.median())

    print(f"  template key                       : "
          f"{'+'.join(TEMPLATE_KEY)}")
    print(f"  scenario templates in dataset      : {total_templates}")
    print(f"  templates touched by the {len(cal_ids)} tickets : "
          f"{len(cal_templates)} "
          f"({len(cal_templates) / total_templates:.1%})")
    print(f"  rows per template (median)         : {median_rows}")
    print(f"  siblings left after removing 1 row : {median_rows - 1}")
    print(f"  rows surviving template exclusion  : {surviving}/{len(df)}")
    print()
    print(f"  => Row-level exclusion removes ~{1 / median_rows:.1%} of a "
          "template's evidence,")
    print("     so the fitted model is effectively unchanged.")
    print("  => Template-level exclusion would leave too little data to")
    print("     train on at all. The in-domain calibration set therefore")
    print("     CANNOT be made exchangeable with a model trained on this")
    print("     dataset. The remedy is not cleaning it -- it is calibrating")
    print("     on deployment-distribution data instead.")

    return {
        # Records WHICH grouping produced these counts, so an artifact can
        # never again be read without knowing what a "template" meant in it.
        "template_key": "+".join(TEMPLATE_KEY),
        "total_templates": total_templates,
        "calibration_templates": len(cal_templates),
        "median_rows_per_template": median_rows,
        "rows_surviving_template_exclusion": surviving,
        "dataset_rows": int(len(df)),
    }


# --------------------------------------------------------------------------- #
# Model fitting
# --------------------------------------------------------------------------- #
def build_texts(df):
    return (df["title"].fillna("").astype(str) + " "
            + df["description"].fillna("").astype(str)).tolist()


def fit_scoring_models(df, embeddings, exclude_ids):
    """Fit Tier-1 and Tier-2 on the dataset minus `exclude_ids`.

    Reuses train_cascade.train_tier1 so the Tier-1 vectorizer configuration is
    not duplicated a second time. Tier-2 is refit here rather than loaded from
    models/, because the persisted classifier was fit on all 4,000 rows and is
    exactly the contamination being controlled for.

    Embeddings are sliced from the row-aligned cache rather than recomputed --
    the cache is verified against the dataset length and dimension by the
    caller.
    """
    from sklearn.linear_model import LogisticRegression

    from src.classification.train_cascade import train_tier1

    ids = df["id"].astype(int).to_numpy()
    keep = ~np.isin(ids, list(exclude_ids)) if exclude_ids else np.ones(
        len(df), dtype=bool)

    sub = df.loc[keep]
    texts = build_texts(sub)
    labels = sub["category"].astype(str).tolist()

    tier1_vec, tier1_clf = train_tier1(texts, labels)

    tier2 = LogisticRegression(max_iter=1000)
    tier2.fit(embeddings[keep], labels)

    return tier1_vec, tier1_clf, tier2, int(keep.sum())


def embed(texts, embedder):
    arr = embedder.encode(list(texts), convert_to_numpy=True,
                          batch_size=64, show_progress_bar=False)
    return np.asarray(arr, dtype=np.float32)


def tier1_probabilities(vectorizer, clf, texts):
    """Full per-class probability matrix.

    train_cascade.get_tier1_confidence() collapses predict_proba to argmax and
    max, discarding the vector both LAC and APS need. That function is left
    untouched -- the cascade depends on it.
    """
    return clf.predict_proba(vectorizer.transform(list(texts))), list(
        clf.classes_)


# --------------------------------------------------------------------------- #
# Conformal sweep
# --------------------------------------------------------------------------- #
def evaluate(cal_probs, cal_labels, test_probs, test_labels, classes,
             alpha, score_function, mondrian):
    calibration = cp.calibrate(cal_probs, cal_labels, classes, alpha=alpha,
                               score_function=score_function,
                               mondrian=mondrian)
    cal_sets = cp.predict_sets(cal_probs, calibration)
    test_sets = cp.predict_sets(test_probs, calibration)

    sizes = np.array([len(s) for s in test_sets])
    return {
        "quantile": (calibration.quantile if not mondrian else None),
        "degenerate": calibration.is_degenerate,
        "coverage_calibration": cp.coverage(cal_sets, cal_labels),
        "coverage_benchmark": cp.coverage(test_sets, test_labels),
        "mean_set_size": float(sizes.mean()),
        "singleton_rate": float(np.mean(sizes == 1)),
        "empty_rate": float(np.mean(sizes == 0)),
        "full_rate": float(np.mean(sizes == len(classes))),
    }, calibration


def coverage_sd(alpha, n_cal):
    """Expected sd of a coverage estimate from ONE calibration draw."""
    return float(np.sqrt(alpha * (1 - alpha) / max(n_cal, 1)))


# --------------------------------------------------------------------------- #
# Novelty
# --------------------------------------------------------------------------- #
def seed_key(ood_id):
    """'ood_A1_v2' -> 'A1'. Variants of a seed are not independent."""
    parts = str(ood_id).split("_")
    return parts[1] if len(parts) > 1 else str(ood_id)


def top_similarity(text, artifacts, own_id=None):
    """Top-1 similarity, and the best non-self similarity.

    The calibration tickets are paraphrases of rows that are themselves in the
    index, so 5.7% of them retrieve their own source. Returning both lets the
    self-retrieval effect be measured instead of assumed. Following
    calibrate_rag_similarity_threshold.py:956-958, neither is silently chosen
    for the other.
    """
    from src.agent.retriever import retrieve

    result = retrieve(text, artifacts, top_k=3)
    if not result.retrieved:
        return 0.0, 0.0, False

    top = result.retrieved[0]
    is_self = own_id is not None and str(top.id) == str(own_id)

    non_self = next(
        (r.similarity for r in result.retrieved
         if own_id is None or str(r.id) != str(own_id)),
        0.0,
    )
    return float(top.similarity), float(non_self), bool(is_self)


# --------------------------------------------------------------------------- #
# Main
# --------------------------------------------------------------------------- #
def run(out_suffix="", force=False):
    _banner("CONFORMAL PREDICTION CALIBRATION  (measurement only)")
    print("Production thresholds are NOT modified by this script.")

    # Checked BEFORE the expensive work, so a refusal costs seconds rather
    # than a full re-fit.
    out_csv, novelty_csv, artifact_json = output_paths(out_suffix)
    existing = [p for p in (out_csv, novelty_csv, artifact_json)
                if os.path.exists(p)]
    if existing and not force:
        _fatal(
            "These output files already exist:\n    "
            + "\n    ".join(existing)
            + "\n  Refusing to overwrite published results. Re-run with "
            "--out-suffix <name>\n  to write new files, or --force to "
            "overwrite these."
        )
    if out_suffix:
        print(f"Writing suffixed outputs ('_{out_suffix}'); the published "
              "files are left untouched.")

    df, embeddings, calibration, benchmark, ood, adversarial = \
        load_everything()
    print(f"\n[load] dataset {len(df)} rows | embeddings {embeddings.shape}")
    print(f"[load] calibration {len(calibration)} | benchmark {len(benchmark)}"
          f" | ood {len(ood)} | adversarial {len(adversarial)}")

    cal_ids = {int(r["id"]) for r in calibration}
    n_flagged = sum(1 for r in calibration if r.get("flagged"))
    print(f"[load] flagged (label-noise) records: {n_flagged}/"
          f"{len(calibration)}")

    # ---- Fit both model pairs -------------------------------------------- #
    _banner("STEP 1 - Fitting scoring models (contaminated vs clean)")
    print("contaminated: fit on all 4,000 rows (what train_cascade.py does)")
    c_t1v, c_t1c, c_t2, n_contaminated = fit_scoring_models(
        df, embeddings, exclude_ids=set())
    print(f"  trained on {n_contaminated} rows")

    print("clean: fit with the 175 calibration source rows REMOVED")
    k_t1v, k_t1c, k_t2, n_clean = fit_scoring_models(
        df, embeddings, exclude_ids=cal_ids)
    print(f"  trained on {n_clean} rows")
    assert_calibration_disjoint(df, calibration, cal_ids)

    contamination_structure = report_contamination_structure(
        df, calibration)

    # ---- Encode calibration + benchmark once ----------------------------- #
    _banner("STEP 2 - Encoding calibration and benchmark texts")
    from src.agent.artifacts import load_artifacts
    artifacts = load_artifacts(require_gemini=False)

    cal_texts = [r["text"] for r in calibration]
    cal_labels = [r["expected"] for r in calibration]
    cal_flagged = [bool(r.get("flagged")) for r in calibration]
    bench_texts = [r["text"] for r in benchmark]
    bench_labels = [r["expected"] for r in benchmark]

    cal_emb = embed(cal_texts, artifacts.embedder)
    bench_emb = embed(bench_texts, artifacts.embedder)
    print(f"  calibration {cal_emb.shape} | benchmark {bench_emb.shape}")

    # ---- Optional deployment-distribution calibration set ----------------- #
    # The in-domain set cannot satisfy exchangeability (Step 1b). If a
    # deployment-distribution set exists, it is swept as a second calibration
    # SOURCE so the before/after coverage pair is directly comparable.
    deployment = None
    if os.path.isfile(DEPLOYMENT_JSON):
        deployment = _read_json(DEPLOYMENT_JSON, "Deployment calibration set")
        dep_texts = [r["text"] for r in deployment]
        dep_labels = [r["expected"] for r in deployment]
        dep_emb = embed(dep_texts, artifacts.embedder)
        print(f"  deployment calibration set: {len(deployment)} tickets "
              f"{dep_emb.shape}")

        overlap = set(dep_texts) & set(bench_texts)
        if overlap:
            _fatal(
                f"{len(overlap)} deployment calibration ticket(s) are "
                "verbatim benchmark tickets. Calibrating on the evaluation "
                "set would make the coverage measurement meaningless."
            )
    else:
        print("  (no deployment calibration set yet at "
              f"{DEPLOYMENT_JSON};")
        print("   in-domain only. Generate it with "
              "generate_deployment_calibration_set.py)")

    # ---- Conformal sweep -------------------------------------------------- #
    _banner("STEP 3 - Conformal sweep")
    rows = []
    fits_out = {}

    variants = [
        ("contaminated", c_t1v, c_t1c, c_t2),
        ("clean", k_t1v, k_t1c, k_t2),
    ]

    for contamination, t1v, t1c, t2 in variants:
        tier_sources = {
            "tier1": (
                tier1_probabilities(t1v, t1c, cal_texts),
                tier1_probabilities(t1v, t1c, bench_texts),
            ),
            "tier2": (
                (t2.predict_proba(cal_emb), list(t2.classes_)),
                (t2.predict_proba(bench_emb), list(t2.classes_)),
            ),
        }

        for tier, ((cal_probs, classes), (bench_probs, _)) in \
                tier_sources.items():
            for label_filter in ("all", "clean_labels"):
                if label_filter == "all":
                    keep = np.ones(len(cal_labels), dtype=bool)
                else:
                    keep = ~np.array(cal_flagged, dtype=bool)

                sub_probs = cal_probs[keep]
                sub_labels = [x for x, k in zip(cal_labels, keep) if k]

                for score_function in SCORE_FUNCTIONS:
                    for mondrian in (False, True):
                        for alpha in ALPHAS:
                            res, fitted = evaluate(
                                sub_probs, sub_labels, bench_probs,
                                bench_labels, classes, alpha,
                                score_function, mondrian,
                            )
                            row = {
                                "calibration_source": "in_domain",
                                "contamination": contamination,
                                "tier": tier,
                                "label_filter": label_filter,
                                "score_function": score_function,
                                "mondrian": mondrian,
                                "alpha": alpha,
                                "n_calibration": len(sub_labels),
                                "nominal_coverage": 1 - alpha,
                                "coverage_sd": coverage_sd(
                                    alpha, len(sub_labels)),
                            }
                            row.update(res)
                            row["coverage_gap"] = (
                                row["coverage_benchmark"] - (1 - alpha))
                            rows.append(row)

                            mode = "mondrian" if mondrian else "marginal"
                            key = (f"{contamination}|{tier}|{label_filter}"
                                   f"|{score_function}|{mode}|{alpha}")
                            fits_out[key] = fitted.to_dict()


    # ---- Same sweep, calibrated on the DEPLOYMENT-distribution set -------- #
    # This is the fix for the exchangeability failure measured above. The
    # models here are the production (full-data) ones: the deployment set
    # shares no source rows with training, so the contamination axis does not
    # apply to it and is recorded as "n/a" rather than faked.
    if deployment is not None:
        dep_tier_sources = {
            "tier1": (
                tier1_probabilities(c_t1v, c_t1c, dep_texts),
                tier1_probabilities(c_t1v, c_t1c, bench_texts),
            ),
            "tier2": (
                (c_t2.predict_proba(dep_emb), list(c_t2.classes_)),
                (c_t2.predict_proba(bench_emb), list(c_t2.classes_)),
            ),
        }

        for tier, ((cal_probs, classes), (bench_probs, _)) in \
                dep_tier_sources.items():
            for score_function in SCORE_FUNCTIONS:
                for mondrian in (False, True):
                    for alpha in ALPHAS:
                        res, fitted = evaluate(
                            cal_probs, dep_labels, bench_probs, bench_labels,
                            classes, alpha, score_function, mondrian,
                        )
                        row = {
                            "calibration_source": "deployment",
                            "contamination": "n/a",
                            "tier": tier,
                            "label_filter": "all",
                            "score_function": score_function,
                            "mondrian": mondrian,
                            "alpha": alpha,
                            "n_calibration": len(dep_labels),
                            "nominal_coverage": 1 - alpha,
                            "coverage_sd": coverage_sd(alpha,
                                                       len(dep_labels)),
                        }
                        row.update(res)
                        row["coverage_gap"] = (
                            row["coverage_benchmark"] - (1 - alpha))
                        rows.append(row)

                        mode = "mondrian" if mondrian else "marginal"
                        key = (f"deployment|{tier}|all|{score_function}"
                               f"|{mode}|{alpha}")
                        fits_out[key] = fitted.to_dict()

    print(f"  {len(rows)} configurations evaluated")

    # ---- Headline table --------------------------------------------------- #
    _banner("STEP 4 - Headline: does the guarantee hold on the benchmark?")
    print("LAC, marginal, all 175 labels. gap = benchmark coverage - nominal.")
    print("2sd = noise from ONE calibration draw; a gap inside it is not")
    print("evidence of anything.\n")
    header = (f"{'calset':<12}{'contam':<13}{'tier':<7}{'alpha':>6}"
              f"{'nominal':>9}{'cal':>8}{'bench':>8}{'gap':>8}{'2sd':>8}"
              f"{'setsz':>7}{'1-set':>8}")
    print(header)
    print("-" * len(header))
    for r in rows:
        if (r["score_function"] != "lac" or r["mondrian"]
                or r["label_filter"] != "all"):
            continue
        print(f"{r['calibration_source']:<12}{r['contamination']:<13}"
              f"{r['tier']:<7}{r['alpha']:>6.2f}"
              f"{r['nominal_coverage']:>9.3f}{r['coverage_calibration']:>8.3f}"
              f"{r['coverage_benchmark']:>8.3f}{r['coverage_gap']:>+8.3f}"
              f"{2 * r['coverage_sd']:>8.3f}{r['mean_set_size']:>7.2f}"
              f"{r['singleton_rate']:>8.3f}")

    # ---- Novelty ---------------------------------------------------------- #
    _banner("STEP 5 - Conformal novelty detection (the RAG gate, reframed)")
    print("Calibrating on in-domain inliers ONLY -- the OOD set is pure")
    print("evaluation data here, not calibration data.\n")

    cal_sims, cal_nonself, n_self = [], [], 0
    for rec in calibration:
        top, nonself, is_self = top_similarity(
            rec["text"], artifacts, own_id=rec["id"])
        cal_sims.append(top)
        cal_nonself.append(nonself)
        n_self += int(is_self)

    self_rate = n_self / len(calibration)
    print(f"  self-retrieval contamination: {n_self}/{len(calibration)} "
          f"= {self_rate:.1%}")

    ood_sims = [top_similarity(r["text"], artifacts)[0] for r in ood]
    ood_seeds = [seed_key(r["id"]) for r in ood]
    adv_sims = [top_similarity(r["text"], artifacts)[0] for r in adversarial]

    n_seeds = len(set(ood_seeds))
    print(f"  OOD: {len(ood)} variants across {n_seeds} independent seeds")
    print("  (variants of one seed are near-duplicates; seed level is the")
    print("   honest n)\n")

    novelty_rows = []
    for sim_source, contamination in (
            (cal_sims, "contaminated"), (cal_nonself, "clean")):
        cal_scores = [-s for s in sim_source]
        for alpha in ALPHAS:
            in_p = cp.conformal_p_values(cal_scores, cal_scores)
            ood_p = cp.conformal_p_values(cal_scores, [-s for s in ood_sims])
            adv_p = cp.conformal_p_values(cal_scores, [-s for s in adv_sims])

            flagged_seeds = {
                s for s, p in zip(ood_seeds, ood_p) if p <= alpha}

            novelty_rows.append({
                "contamination": contamination,
                "alpha": alpha,
                "n_calibration": len(cal_scores),
                "false_escalation_rate_in_domain": float(
                    np.mean(in_p <= alpha)),
                "ood_detection_rate_variants": float(np.mean(ood_p <= alpha)),
                "ood_detection_rate_seeds": len(flagged_seeds) / n_seeds,
                "n_ood_seeds": n_seeds,
                "adversarial_flagged": int(np.sum(adv_p <= alpha)),
                "adversarial_total": len(adv_sims),
                "min_attainable_p": 1.0 / (len(cal_scores) + 1),
            })

    hdr = (f"{'contam':<13}{'alpha':>6}{'in-dom FPR':>12}{'OOD(seed)':>11}"
           f"{'OOD(var)':>10}{'adv':>8}")
    print(hdr)
    print("-" * len(hdr))
    for r in novelty_rows:
        adv = f"{r['adversarial_flagged']}/{r['adversarial_total']}"
        print(f"{r['contamination']:<13}{r['alpha']:>6.2f}"
              f"{r['false_escalation_rate_in_domain']:>12.3f}"
              f"{r['ood_detection_rate_seeds']:>11.3f}"
              f"{r['ood_detection_rate_variants']:>10.3f}"
              f"{adv:>8}")

    # ---- Write ------------------------------------------------------------ #
    _banner("STEP 6 - Writing results")
    with open(out_csv, "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    print(f"  {out_csv}  ({len(rows)} rows)")

    with open(novelty_csv, "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=list(novelty_rows[0].keys()))
        w.writeheader()
        w.writerows(novelty_rows)
    print(f"  {novelty_csv}  ({len(novelty_rows)} rows)")

    from src.agent.config import config_fingerprint
    payload = {
        "config_fingerprint": config_fingerprint(),
        "embedding_model": settings.models.embedding_model,
        "n_dataset": len(df),
        "n_calibration": len(calibration),
        "n_flagged": n_flagged,
        "self_retrieval_rate": self_rate,
        "contamination_structure": contamination_structure,
        "alphas": ALPHAS,
        "fits": fits_out,
    }
    with open(artifact_json, "w", encoding="utf-8") as fh:
        json.dump(payload, fh, indent=2, sort_keys=True)
        fh.write("\n")
    print(f"  {artifact_json}")

    _banner("DONE - measurement only; no production threshold changed")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Conformal calibration and coverage measurement "
                    "(measurement only).")
    parser.add_argument("--out-suffix", default="",
                        help="write results to suffixed filenames, e.g. "
                             "--out-suffix corrected, leaving the published "
                             "files untouched")
    parser.add_argument("--force", action="store_true",
                        help="overwrite existing output files")
    args = parser.parse_args()

    try:
        run(out_suffix=args.out_suffix, force=args.force)
    except KeyboardInterrupt:
        print("\nInterrupted.")
        sys.exit(130)
