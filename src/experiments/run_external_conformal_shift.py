# src/experiments/run_external_conformal_shift.py
"""
Phase 7B -- does Finding 1 replicate on a corpus from a DIFFERENT GENERATOR?

WHAT FINDING 1 SAYS. Calibrated on the same 175 tickets at the same alpha,
TF-IDF loses 23.3 coverage points on the 45-ticket benchmark while BGE loses
1.1, against a +-2 s.d. band of 0.045. Conformal coverage transfers for the
dense representation and not for the lexical one. It was measured ONCE, on ONE
corpus, at n_cal = 175 -- so today it is as much a property of our
template-generated data as it might be a property of representations.

WHAT THIS SCRIPT ASKS. Whether the same contrast appears on
Tobi-Bueck/customer-support-tickets, under a shift that was MEASURED rather
than assumed, with ~2.8x tighter noise bands and ~115x the tickets at low
coverage.

THE CORPUS IS INDEPENDENTLY GENERATED DATA, NOT REAL PRODUCTION DATA. Its card
advertises a synthetic generator from the same author. This tests whether
Finding 1 survives a DIFFERENT GENERATOR, not whether it survives reality.
Every sentence of the write-up must say so.

MEASUREMENT ONLY. settings.conformal.enabled and settings.drift.enabled stay
False whatever this shows. Cascade 0.50 / RAG 0.67 / clustering 0.80 are
untouched. Offline, zero Gemini calls.

-----------------------------------------------------------------------------
PRE-REGISTRATION -- fixed before any result was seen
-----------------------------------------------------------------------------
PRIMARY -- Design A, the Finding 1 table shape.
    Calibrate on aa-file English rows with version in {51, 52}; test on
    version == 400. Both tiers fitted FRESH on this corpus only. LAC,
    alpha in {0.05, 0.10, 0.20}, seed 42. Report coverage on calibration and
    on test, the gap, the +-2 s.d. band RECOMPUTED for this n_cal, mean set
    size, singleton rate and empty rate.
    HEADLINE QUESTION: does TF-IDF lose far more coverage than BGE, as on our
    data? The verdict is read off the two tiers' gaps against the band -- not
    off set sizes and not off accuracy.

SECONDARY -- Design B, the queue-holdout magnitude sweep.
    Hold one queue out of CALIBRATION ONLY, keeping it in training so the
    label space is preserved. Report coverage gap against measured
    cross-fitted domain AUC, to ask how much shift the effect needs.
    REPORTED ALWAYS, PROMOTED TO A HEADLINE NEVER. Its confound is reportable
    and not removable: queue correlates with `type`, so a queue holdout also
    shifts the type mix.

DEGENERACY RULE (same as 6A/6B) -- name the condition, report "no resolution",
never substitute a metric that happens to resolve:
    * a calibration with no finite quantile is reported `degenerate`; no other
      alpha is substituted for it;
    * a Design B holdout whose calibration n is below
      cp.min_calibration_size(alpha) is reported `insufficient_calibration_n`
      and NOT run;
    * a Design B holdout whose domain AUC is >= 0.95 is reported BLOCKED --
      the 6B rule, which blocked that phase's BGE arm at 0.9908 -- and is
      never quoted as a result.

POST-HOC. Anything added after results were seen carries post_hoc=True.

EITHER ANSWER IS REPORTABLE. If TF-IDF's gap is not materially worse than
BGE's, the plain statement goes in: the effect may be a property of our
corpus's template structure rather than of lexical vs dense representations,
and Finding 1's scope narrows accordingly.

-----------------------------------------------------------------------------
THE THREE THINGS THE DESIGN HAD TO SPECIFY
-----------------------------------------------------------------------------
1. SPLITS. A TRAINING split disjoint from calibration, both from versions
   51+52, seed 42, stratified by queue. De-duplication happens on the POOL
   BEFORE the split, by union-find over BGE cosine >= 0.95, one representative
   per component. That does two jobs: no near-duplicate spans train and
   calibration, and no duplicated points enter the conformal quantile -- the
   defect our first deployment calibration set shipped with. The TEST arm is
   NOT de-duplicated: removing rows there would change the very coverage being
   measured.

2. BOUNDARY CONTAMINATION. For every version-400 test ticket, its nearest
   neighbour in train ∪ calibration. The whole Design A table is then reported
   TWICE -- on the full test set and on the no-neighbour subset -- because a
   memorised test arm would reproduce Finding 1's mechanism for entirely the
   wrong reason. Where the two disagree, the no-neighbour reading is the one
   that answers the question, and both are published.

3. LABEL NOISE. In-distribution accuracy for both tiers, with the limitation
   written beside the number: the external queue labels are generator-assigned
   and UNAUDITED. Label noise depresses accuracy and inflates set sizes for
   BOTH tiers, so a low ceiling bounds every downstream reading here.

-----------------------------------------------------------------------------
RULE 7 EXCEPTION, STATED
-----------------------------------------------------------------------------
This script fits Tier-1 and Tier-2 FRESH on the external training split and
never loads models/. A leave-out fit is the entire point, and the production
artifacts were fitted on our 4,000 rows over a different label space, so
loading them is not an option. The BGE encoder is still obtained through
artifacts.load_artifacts(); no SentenceTransformer is constructed locally.
Tier-1's vectorizer configuration comes from train_cascade.train_tier1, not
from a second copy of it.

Run from the project root (offline, no Gemini calls, ~5 min):
    python src/experiments/run_external_conformal_shift.py
    python src/experiments/run_external_conformal_shift.py --force
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
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
    evaluate,
    tier1_probabilities,
)
from src.experiments.run_weighted_conformal import (               # noqa: E402
    coverage_from_sets,
    cross_fitted_domain_probabilities,
    set_stats,
)
from src.experiments.flag_automation_candidates import (           # noqa: E402
    _UnionFind,
)
from src.experiments.fetch_external_dataset import (               # noqa: E402
    ENGLISH_CSV,
    OUT_DIR,
    PROVENANCE_JSON,
)
from src.experiments.profile_external_dataset import (             # noqa: E402
    EMB_NPY,
    NEAR_DUP_THRESHOLD,
)

ensure_utf8_console()

# ---- Pre-registered grid --------------------------------------------------- #
ALPHAS = (0.05, 0.10, 0.20)
TIERS = ("tier1", "tier2")
SCORE_FUNCTION = "lac"
SEED = 42
TRAIN_FRACTION = 0.60

CAL_VERSIONS = (51.0, 52.0)
TEST_VERSION = 400.0
AA_PREFIX = "aa_"

# 6B's rule: a domain classifier that separates the two arms this well makes
# the shift ill-posed rather than large. It blocked 6B's BGE arm at 0.9908.
DEGENERACY_AUC = 0.95

# The AUC recorded at the 7A gate for Design A. See the RECORD CHECK in step 6:
# it is reported against this run's fully specified recomputation, and a
# mismatch is surfaced loudly rather than quietly tolerated.
RECORDED_DESIGN_A_AUC = 0.8584

# ---- Outputs -- all new filenames, nothing existing is overwritten --------- #
SPLITS_JSON = os.path.join(OUT_DIR, "external_shift_splits.json")
DESIGN_A_CSV = os.path.join(OUT_DIR, "external_conformal_designA.csv")
DESIGN_B_CSV = os.path.join(OUT_DIR, "external_conformal_designB.csv")
CONTAMINATION_JSON = os.path.join(OUT_DIR, "external_contamination.json")
LABEL_NOISE_JSON = os.path.join(OUT_DIR, "external_label_noise.json")

OUTPUTS = (SPLITS_JSON, DESIGN_A_CSV, DESIGN_B_CSV,
           CONTAMINATION_JSON, LABEL_NOISE_JSON)

# Caches for the deferral secondary, which reads this run's fitted
# probabilities rather than refitting and risking a different model.
PROBS_NPZ = os.path.join(OUT_DIR, "external_designA_probs.npz")

ALIGNMENT_PROBE_ROWS = 64
ALIGNMENT_MIN_COSINE = 0.999
BRUTE_FORCE_PROBE_ROWS = 200


# --------------------------------------------------------------------------- #
# Isolation -- widened from 7A's three files
# --------------------------------------------------------------------------- #
def isolation_paths():
    """Every file this phase must leave byte-identical.

    7A hashed our dataset and the two benchmark files. 7B additionally touches
    conformal and deferral machinery, so the PUBLISHED RESULTS those phases
    produced are added: if this run ever wrote one of them, the comparison it
    feeds would move without anyone noticing. Hashing is cheap; the failure it
    prevents has happened six times in this project.
    """
    data = os.path.join(PROJECT_ROOT, "data")
    models = os.path.join(PROJECT_ROOT, "models")
    goldens = os.path.join(PROJECT_ROOT, "tests", "goldens")

    paths = [
        os.path.join(data, "synthetic_tickets.csv"),
        os.path.join(data, "novel_tickets_expanded.json"),
        os.path.join(data, "calibration_tickets_paraphrased.json"),
        os.path.join(data, "deployment_calibration_set.json"),
        os.path.join(data, "conformal_calibration_results.csv"),
        os.path.join(data, "weighted_conformal_results.csv"),
    ]
    for directory, pattern in ((data, "deferral"), (data, "ablation")):
        if os.path.isdir(directory):
            paths += [os.path.join(directory, f)
                      for f in sorted(os.listdir(directory))
                      if f.startswith(pattern) and f.endswith(".csv")]
    for directory in (models, goldens):
        if os.path.isdir(directory):
            paths += [os.path.join(directory, f)
                      for f in sorted(os.listdir(directory))
                      if os.path.isfile(os.path.join(directory, f))]
    return [p for p in paths if os.path.isfile(p)]


def snapshot(paths):
    return {p: hashlib.sha256(open(p, "rb").read()).hexdigest() for p in paths}


def verify_isolation(before):
    after = snapshot(list(before))
    moved = [p for p in before if before[p] != after.get(p)]
    if moved:
        _fatal("ISOLATION BREACH -- this run modified files it must not "
               "touch:\n  " + "\n  ".join(os.path.relpath(p, PROJECT_ROOT)
                                          for p in moved))
    print("  [ok] {n} project files byte-identical before and after".format(
        n=len(before)))


# --------------------------------------------------------------------------- #
# Loading and the alignment guard
# --------------------------------------------------------------------------- #
def build_texts(df):
    """The EXACT text 7A encoded: subject + ' ' + body, empties as ''.

    Any deviation here silently misaligns every cached embedding with its row,
    which is this project's recurring bug in its purest form. Tier-1 reads the
    same string, so the two tiers see identical input and the contrast between
    them is about representation rather than preprocessing.
    """
    return (df["subject"].fillna("").astype(str) + " "
            + df["body"].fillna("").astype(str)).tolist()


def load_corpus():
    if not os.path.isfile(ENGLISH_CSV):
        _fatal("English subset not found:\n    {p}\n"
               "  Run: python src/experiments/fetch_external_dataset.py"
               .format(p=ENGLISH_CSV))
    if not os.path.isfile(EMB_NPY):
        _fatal("BGE embedding cache not found:\n    {p}\n"
               "  Run: python src/experiments/profile_external_dataset.py"
               .format(p=EMB_NPY))

    df = pd.read_csv(ENGLISH_CSV, low_memory=False)
    emb = np.load(EMB_NPY)

    if emb.shape[0] != len(df):
        _fatal("Embedding cache has {a} rows for {b} CSV rows.".format(
            a=emb.shape[0], b=len(df)))
    if emb.shape[1] != settings.models.embedding_dim:
        _fatal("Embedding cache is {d}-dim but config expects {e}.".format(
            d=emb.shape[1], e=settings.models.embedding_dim))

    with open(PROVENANCE_JSON, encoding="utf-8") as fh:
        provenance = json.load(fh)
    if len(df) != provenance["rows_english_total"]:
        _fatal("English CSV has {a} rows but PROVENANCE says {b}".format(
            a=len(df), b=provenance["rows_english_total"]))

    return df, emb, provenance


def verify_embedding_alignment(df, emb, embedder, seed=SEED):
    """Re-encode a seeded sample and require it to match the cache.

    The cache is currently guarded only by a length check, and length is
    exactly the property a misaligned cache preserves. Everything downstream
    -- de-duplication, contamination, Tier-2, the domain classifier -- assumes
    row i of the cache is row i of the CSV, so that assumption is tested
    rather than trusted.
    """
    rng = np.random.default_rng(seed)
    idx = np.sort(rng.choice(len(df), size=min(ALIGNMENT_PROBE_ROWS, len(df)),
                             replace=False))
    texts = [build_texts(df.iloc[idx])[i] for i in range(len(idx))]

    fresh = np.asarray(embedder.encode(texts, convert_to_numpy=True,
                                       batch_size=32,
                                       show_progress_bar=False),
                       dtype=np.float32)
    cached = emb[idx].astype(np.float32)

    a = fresh / np.linalg.norm(fresh, axis=1, keepdims=True)
    b = cached / np.linalg.norm(cached, axis=1, keepdims=True)
    cos = np.sum(a * b, axis=1)
    worst = float(cos.min())

    if worst < ALIGNMENT_MIN_COSINE:
        _fatal("Embedding cache is NOT row-aligned with the CSV: worst cosine "
               "{w:.6f} over {n} probed rows (need >= {t}).\n"
               "  Re-encode with: python src/experiments/"
               "profile_external_dataset.py --force --reencode".format(
                   w=worst, n=len(idx), t=ALIGNMENT_MIN_COSINE))
    print("  [ok] cache row-aligned: {n} probed rows, worst cosine {w:.6f}"
          .format(n=len(idx), w=worst))
    return worst


# --------------------------------------------------------------------------- #
# Arms
# --------------------------------------------------------------------------- #
def arm_indices(df):
    """Design A's two arms, each counted two independent ways (rule 6)."""
    aa = df["_source_file"].astype(str).str.startswith(AA_PREFIX)
    version = pd.to_numeric(df["version"], errors="coerce")

    pool_mask = aa & version.isin(list(CAL_VERSIONS))
    test_mask = aa & (version == TEST_VERSION)

    pool = np.flatnonzero(pool_mask.to_numpy())
    test = np.flatnonzero(test_mask.to_numpy())

    # Second derivation: value_counts over the aa subset.
    counts = version[aa].value_counts(dropna=False)
    pool_vc = int(sum(int(counts.get(v, 0)) for v in CAL_VERSIONS))
    test_vc = int(counts.get(TEST_VERSION, 0))

    if len(pool) != pool_vc or len(test) != test_vc:
        _fatal("Arm sizes disagree between two derivations: mask "
               "{a}/{b} vs value_counts {c}/{d}".format(
                   a=len(pool), b=len(test), c=pool_vc, d=test_vc))

    if np.intersect1d(pool, test).size:
        _fatal("Calibration pool and test arm overlap by row index.")

    print("  pool (version {v}): {n} rows   [mask and value_counts agree]"
          .format(v="+".join(str(int(v)) for v in CAL_VERSIONS), n=len(pool)))
    print("  test (version {v}): {n} rows   [mask and value_counts agree]"
          .format(v=int(TEST_VERSION), n=len(test)))
    return pool, test


# --------------------------------------------------------------------------- #
# De-duplication
# --------------------------------------------------------------------------- #
def dedup_components(emb_subset, threshold=NEAR_DUP_THRESHOLD):
    """Connected components at BGE cosine >= threshold, via FAISS range search.

    Returns (component_id_per_row, representative_row_positions). The union-
    find is the production one from flag_automation_candidates, not a second
    implementation.
    """
    import faiss

    x = np.ascontiguousarray(emb_subset.astype(np.float32))
    faiss.normalize_L2(x)
    index = faiss.IndexFlatIP(x.shape[1])
    index.add(x)
    lims, _dist, ids = index.range_search(x, float(threshold))

    uf = _UnionFind(len(x))
    for i in range(len(x)):
        for j in ids[lims[i]:lims[i + 1]]:
            if int(j) != i:
                uf.union(i, int(j))

    comp = np.array([uf.find(i) for i in range(len(x))], dtype=np.int64)

    # One representative per component: the lowest row position, so the choice
    # is deterministic and independent of any RNG stream.
    reps = {}
    for i in range(len(x)):
        reps.setdefault(int(comp[i]), i)
    return comp, np.array(sorted(reps.values()), dtype=np.int64)


# --------------------------------------------------------------------------- #
# Stratified split
# --------------------------------------------------------------------------- #
def stratified_split(labels, fraction, seed=SEED):
    """Split positions into (train, calibration), stratified by label.

    Not sklearn's train_test_split: with ten queues and a smallest class in
    the fifties, a stratified split must be explicit about how it rounds, and
    the per-class n is reported in the manifest.
    """
    rng = np.random.default_rng(seed)
    labels = np.asarray(labels)
    train, cal = [], []
    for value in sorted(set(labels.tolist())):
        pos = np.flatnonzero(labels == value)
        rng.shuffle(pos)
        cut = int(len(pos) * fraction)
        train.extend(pos[:cut].tolist())
        cal.extend(pos[cut:].tolist())
    return np.array(sorted(train)), np.array(sorted(cal))


def verify_split(train_pos, cal_pos, comp, n_pool_components):
    """Disjoint by row AND by de-duplication component, and jointly complete."""
    if np.intersect1d(train_pos, cal_pos).size:
        _fatal("Train and calibration splits share row positions.")

    train_comps = set(comp[train_pos].tolist())
    cal_comps = set(comp[cal_pos].tolist())
    shared = train_comps & cal_comps
    if shared:
        _fatal("Train and calibration share {n} de-duplication components -- "
               "a near-duplicate spans the split.".format(n=len(shared)))

    union = train_comps | cal_comps
    if len(union) != n_pool_components:
        _fatal("Splits cover {a} components but the pool has {b}.".format(
            a=len(union), b=n_pool_components))

    print("  [ok] splits disjoint by row AND by component; union covers all "
          "{n} components".format(n=n_pool_components))


# --------------------------------------------------------------------------- #
# Contamination
# --------------------------------------------------------------------------- #
def nearest_in_reference(emb_query, emb_reference):
    """Top-1 cosine of each query row against a reference set, via FAISS."""
    import faiss

    q = np.ascontiguousarray(emb_query.astype(np.float32))
    r = np.ascontiguousarray(emb_reference.astype(np.float32))
    faiss.normalize_L2(q)
    faiss.normalize_L2(r)

    index = faiss.IndexFlatIP(r.shape[1])
    index.add(r)
    sims, _idx = index.search(q, 1)
    return sims[:, 0].astype(np.float64)


def brute_force_probe(emb_query, emb_reference, faiss_sims, seed=SEED,
                      n_probe=BRUTE_FORCE_PROBE_ROWS):
    """Second derivation of the top-1 similarities (rule 6), 7A's pattern."""
    rng = np.random.default_rng(seed)
    idx = rng.choice(len(emb_query), size=min(n_probe, len(emb_query)),
                     replace=False)

    q = emb_query[idx].astype(np.float64)
    r = emb_reference.astype(np.float64)
    q = q / np.linalg.norm(q, axis=1, keepdims=True)
    r = r / np.linalg.norm(r, axis=1, keepdims=True)

    best = (q @ r.T).max(axis=1)
    mismatches = int(np.sum(np.abs(best - faiss_sims[idx]) > 1e-4))
    return len(idx), mismatches


# --------------------------------------------------------------------------- #
# Conformal evaluation
# --------------------------------------------------------------------------- #
def conformal_row(cal_probs, cal_labels, test_probs, test_labels, classes,
                  alpha):
    """One pre-registered row: coverage both sides, the gap, and the band.

    Coverage is derived twice -- cp.coverage inside evaluate(), and an
    independent loop over the prediction sets -- because a coverage number is
    the one thing in this phase that cannot be sanity-checked by eye.
    """
    stats, calib = evaluate(cal_probs, cal_labels, test_probs, test_labels,
                            classes, alpha, SCORE_FUNCTION, mondrian=False)
    sets = cp.predict_sets(test_probs, calib)

    independent = coverage_from_sets(sets, test_labels)
    if abs(independent - stats["coverage_benchmark"]) > 1e-12:
        _fatal("Coverage disagrees between two derivations at alpha={a}: "
               "{x} vs {y}".format(a=alpha, x=independent,
                                   y=stats["coverage_benchmark"]))

    sd = coverage_sd(alpha, len(cal_labels))
    gap = stats["coverage_benchmark"] - (1.0 - alpha)

    return {
        "alpha": alpha,
        "n_calibration": len(cal_labels),
        "n_test": len(test_labels),
        "quantile": stats["quantile"],
        "degenerate": calib.is_degenerate,
        "coverage_calibration": stats["coverage_calibration"],
        "coverage_test": stats["coverage_benchmark"],
        "coverage_gap": gap,
        "coverage_sd": sd,
        "noise_band_2sd": 2.0 * sd,
        "gap_outside_band": abs(gap) > 2.0 * sd,
        "gap_in_sd": (gap / sd) if sd > 0 else float("nan"),
        **set_stats(sets, len(classes)),
    }


def domain_auc(emb_a, emb_b):
    """Cross-fitted domain AUC between two arms, reusing 6B's machinery."""
    from sklearn.metrics import roc_auc_score

    oof_a, oof_b, _model, y = cross_fitted_domain_probabilities(emb_a, emb_b)
    return float(roc_auc_score(y, np.concatenate([oof_a, oof_b])))


# --------------------------------------------------------------------------- #
# Writers
# --------------------------------------------------------------------------- #
def write_csv(path, rows):
    """Write rows whose key sets may differ.

    A blocked or skipped row carries fewer fields than a completed one, and
    DictWriter raises on the first mismatch. The union is taken in first-seen
    order so the blocked rows stay in the file -- dropping them would hide
    exactly the configurations the degeneracy rule exists to surface.
    """
    if not rows:
        _fatal("Refusing to write an empty result file: {p}".format(p=path))
    fields = []
    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(key)
    with open(path, "w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=fields, restval="")
        writer.writeheader()
        for row in rows:
            writer.writerow(row)
    print("[write] {p}  ({n} rows)".format(
        p=os.path.relpath(path, PROJECT_ROOT), n=len(rows)))


def write_json(path, blob):
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(blob, fh, indent=2)
    print("[write] {p}".format(p=os.path.relpath(path, PROJECT_ROOT)))


def refuse_overwrite(force):
    if force:
        return
    for path in OUTPUTS:
        if os.path.isfile(path):
            _fatal("Refusing to overwrite an existing result:\n  {p}\n"
                   "Pass --force if you mean to replace it.".format(p=path))


# --------------------------------------------------------------------------- #
# The run
# --------------------------------------------------------------------------- #
def run(force=False):
    _banner("PHASE 7B - DOES FINDING 1 REPLICATE ON A DIFFERENT GENERATOR?")
    print("Independently generated data, NOT real production data.")
    print("Measurement only: conformal.enabled={c}, drift.enabled={d}".format(
        c=settings.conformal.enabled, d=settings.drift.enabled))
    if settings.conformal.enabled or settings.drift.enabled:
        _fatal("This phase is measurement only; both flags must stay False.")

    refuse_overwrite(force)
    before = snapshot(isolation_paths())
    print("  isolation baseline: {n} files hashed".format(n=len(before)))

    # ---- Step 1: load and prove the cache is aligned --------------------- #
    _banner("STEP 1 - Corpus, embedding cache, and the alignment guard")
    df, emb, provenance = load_corpus()
    print("  {n} English rows, embeddings {s}".format(n=len(df), s=emb.shape))

    from src.agent.artifacts import load_artifacts
    artifacts = load_artifacts(require_gemini=False)
    worst_cosine = verify_embedding_alignment(df, emb, artifacts.embedder)

    texts = build_texts(df)
    labels = df["queue"].astype(str).to_numpy()

    # ---- Step 2: arms ---------------------------------------------------- #
    _banner("STEP 2 - Design A arms (counted two ways)")
    pool_idx, test_idx = arm_indices(df)

    # ---- Step 3: de-duplicate the pool, then split ----------------------- #
    _banner("STEP 3 - De-duplicating the pool, then splitting")
    comp, reps = dedup_components(emb[pool_idx])
    n_components = len(set(comp.tolist()))
    print("  pool {n} rows -> {c} components at BGE cosine >= {t} "
          "({r:.2%} distinct)".format(n=len(pool_idx), c=n_components,
                                      t=NEAR_DUP_THRESHOLD,
                                      r=n_components / len(pool_idx)))
    if len(reps) != n_components:
        _fatal("Representative count {a} != component count {b}".format(
            a=len(reps), b=n_components))

    pool_labels = labels[pool_idx]
    train_pos, cal_pos = stratified_split(pool_labels[reps], TRAIN_FRACTION)
    train_pos = reps[train_pos]
    cal_pos = reps[cal_pos]
    verify_split(train_pos, cal_pos, comp, n_components)

    train_idx = pool_idx[train_pos]
    cal_idx = pool_idx[cal_pos]
    print("  train {a} | calibration {b} | test {c}".format(
        a=len(train_idx), b=len(cal_idx), c=len(test_idx)))

    per_queue = {}
    for queue in sorted(set(labels[pool_idx][reps].tolist())):
        per_queue[queue] = {
            "train": int(np.sum(labels[train_idx] == queue)),
            "calibration": int(np.sum(labels[cal_idx] == queue)),
            "test": int(np.sum(labels[test_idx] == queue)),
        }
        print("    {q:34} train {t:>5}  cal {c:>5}  test {s:>5}".format(
            q=queue[:34], t=per_queue[queue]["train"],
            c=per_queue[queue]["calibration"], s=per_queue[queue]["test"]))

    for alpha in ALPHAS:
        need = cp.min_calibration_size(alpha)
        if len(cal_idx) < need:
            _fatal("Calibration n={n} is below the minimum {m} for alpha={a}"
                   .format(n=len(cal_idx), m=need, a=alpha))
    print("  [ok] calibration n clears cp.min_calibration_size for every "
          "alpha in {a}".format(a=ALPHAS))

    # ---- Step 4: fresh fits ---------------------------------------------- #
    _banner("STEP 4 - Fitting both tiers FRESH on the external training split")
    from sklearn.linear_model import LogisticRegression
    from src.classification.train_cascade import train_tier1

    train_texts = [texts[i] for i in train_idx]
    train_labels = labels[train_idx].tolist()

    t1_vec, t1_clf = train_tier1(train_texts, train_labels)
    t2_clf = LogisticRegression(max_iter=1000)
    t2_clf.fit(emb[train_idx], train_labels)
    print("  Tier-1 TF-IDF+LogReg and Tier-2 BGE+LogReg fitted on {n} rows, "
          "{k} classes".format(n=len(train_idx), k=len(t1_clf.classes_)))

    cal_texts = [texts[i] for i in cal_idx]
    test_texts = [texts[i] for i in test_idx]
    cal_labels = labels[cal_idx].tolist()
    test_labels = labels[test_idx].tolist()

    probs = {}
    cal_p1, classes1 = tier1_probabilities(t1_vec, t1_clf, cal_texts)
    test_p1, _ = tier1_probabilities(t1_vec, t1_clf, test_texts)
    probs["tier1"] = (cal_p1, test_p1, classes1)

    classes2 = list(t2_clf.classes_)
    probs["tier2"] = (t2_clf.predict_proba(emb[cal_idx]),
                      t2_clf.predict_proba(emb[test_idx]), classes2)

    # ---- Step 5: label noise --------------------------------------------- #
    _banner("STEP 5 - Label-noise sanity check (in-distribution accuracy)")
    print("  LIMITATION, stated beside the number: these queue labels are")
    print("  GENERATOR-ASSIGNED AND UNAUDITED. Label noise depresses accuracy")
    print("  and inflates prediction-set sizes for BOTH tiers, so a low")
    print("  ceiling here bounds every downstream reading.")
    label_noise = {}
    for tier in TIERS:
        cal_probs, test_probs, classes = probs[tier]
        cls = np.asarray(classes)
        cal_acc = float(np.mean(cls[np.argmax(cal_probs, axis=1)]
                                == np.asarray(cal_labels)))
        test_acc = float(np.mean(cls[np.argmax(test_probs, axis=1)]
                                 == np.asarray(test_labels)))
        label_noise[tier] = {
            "accuracy_calibration": cal_acc,
            "accuracy_test": test_acc,
            "n_calibration": len(cal_labels),
            "n_test": len(test_labels),
        }
        print("  {t}: calibration {a:.4f} | version-400 test {b:.4f}".format(
            t=tier, a=cal_acc, b=test_acc))

    # ---- Step 6: shift magnitude ----------------------------------------- #
    _banner("STEP 6 - Shift magnitude (cross-fitted domain AUC, BGE space)")
    auc_raw = domain_auc(emb[pool_idx], emb[test_idx])
    auc_operating = domain_auc(emb[cal_idx], emb[test_idx])
    print("  raw arms      (pool {a} vs test {b}): AUC {x:.4f}".format(
        a=len(pool_idx), b=len(test_idx), x=auc_raw))
    print("  operating     (cal  {a} vs test {b}): AUC {x:.4f}".format(
        a=len(cal_idx), b=len(test_idx), x=auc_operating))

    record_matches = abs(auc_raw - RECORDED_DESIGN_A_AUC) <= 0.01
    if not record_matches:
        print("\n  !! RECORD CHECK FAILED, reported not buried.")
        print("     PROJECT_STATUS/README record Design A's domain AUC as "
              "{r:.4f}.".format(r=RECORDED_DESIGN_A_AUC))
        print("     That value was computed ad hoc at the 7A gate and its "
              "derivation was never committed;")
        print("     it does not reproduce. This run's fully specified "
              "recomputation gives {a:.4f}".format(a=auc_raw))
        print("     (raw arms) and {b:.4f} (the arms actually operated on)."
              .format(b=auc_operating))
        print("     THE DESIGN CONCLUSION IS UNCHANGED: every variant sits "
              "well below the {d} degeneracy line".format(d=DEGENERACY_AUC))
        print("     that blocked 6B's BGE arm at 0.9908, so the shift is "
              "real AND operable.")
        print("     The recorded figure is corrected in the write-up.")

    # The decision this number actually gates. A mismatch with an ad hoc
    # record is a documentation defect; crossing the degeneracy line would
    # invalidate the design, and only that is fatal.
    if auc_operating >= DEGENERACY_AUC:
        _fatal("Operating domain AUC {a:.4f} >= {d}: the two arms are nearly "
               "separable, so the shift is ill-posed rather than large. This "
               "is 6B's blocking condition; Design A cannot be run.".format(
                   a=auc_operating, d=DEGENERACY_AUC))
    print("  [ok] operating AUC below the {d} degeneracy line".format(
        d=DEGENERACY_AUC))

    # ---- Step 7: boundary contamination ---------------------------------- #
    _banner("STEP 7 - Boundary contamination of the version-400 test arm")
    reference = np.vstack([emb[train_idx], emb[cal_idx]])
    top1 = nearest_in_reference(emb[test_idx], reference)
    has_neighbour = top1 >= NEAR_DUP_THRESHOLD

    n_probe, mismatches = brute_force_probe(emb[test_idx], reference, top1)
    if mismatches:
        _fatal("FAISS and brute force disagree on {n} of {m} probed top-1 "
               "similarities.".format(n=mismatches, m=n_probe))

    contamination = {
        "threshold": NEAR_DUP_THRESHOLD,
        "model": settings.models.embedding_model,
        "n_test": int(len(test_idx)),
        "n_reference": int(len(reference)),
        "n_with_near_neighbour": int(has_neighbour.sum()),
        "near_neighbour_rate": float(has_neighbour.mean()),
        "n_no_neighbour": int((~has_neighbour).sum()),
        "mean_top1_similarity": float(top1.mean()),
        "median_top1_similarity": float(np.median(top1)),
        "p05_top1_similarity": float(np.percentile(top1, 5)),
        "brute_force_probe_rows": int(n_probe),
        "brute_force_mismatches": int(mismatches),
        "corpus_wide_near_duplicate_rate_7a": 0.7939209511340717,
        "note": ("Coverage is reported on the full test set AND on the "
                 "no-neighbour subset. A memorised test arm would reproduce "
                 "Finding 1's mechanism for the wrong reason."),
    }
    print("  {n} / {t} test tickets have a BGE >= {h} neighbour in train u "
          "calibration = {r:.4%}".format(
              n=contamination["n_with_near_neighbour"], t=len(test_idx),
              h=NEAR_DUP_THRESHOLD, r=contamination["near_neighbour_rate"]))
    print("  top-1 similarity: median {m:.4f}, p05 {p:.4f}".format(
        m=contamination["median_top1_similarity"],
        p=contamination["p05_top1_similarity"]))
    print("  no-neighbour subset: {n} tickets".format(
        n=contamination["n_no_neighbour"]))
    print("  [ok] FAISS agrees with brute force on {n} probed rows".format(
        n=n_probe))

    # ---- Step 8: Design A ------------------------------------------------ #
    _banner("STEP 8 - DESIGN A (PRIMARY): the Finding 1 table")
    test_arms = {
        "full": np.ones(len(test_idx), dtype=bool),
        "no_neighbour": ~has_neighbour,
    }

    design_a = []
    for arm_name, mask in test_arms.items():
        if mask.sum() == 0:
            print("  [skip] test arm {a} is empty".format(a=arm_name))
            continue
        arm_labels = [y for y, keep in zip(test_labels, mask) if keep]
        for tier in TIERS:
            cal_probs, test_probs, classes = probs[tier]
            for alpha in ALPHAS:
                row = conformal_row(cal_probs, cal_labels,
                                    test_probs[mask], arm_labels,
                                    classes, alpha)
                row.update({
                    "design": "A",
                    "test_arm": arm_name,
                    "tier": tier,
                    "score_function": SCORE_FUNCTION,
                    "domain_auc_operating": auc_operating,
                    "post_hoc": False,
                })
                design_a.append(row)

    for arm_name in test_arms:
        print("\n  test arm: {a}".format(a=arm_name))
        print("    {t:6} {al:>6} {c:>10} {g:>10} {b:>9} {o:>5} {s:>8} "
              "{e:>8}".format(t="tier", al="alpha", c="coverage", g="gap",
                              b="band", o="out", s="set", e="empty"))
        for row in design_a:
            if row["test_arm"] != arm_name:
                continue
            print("    {t:6} {al:>6.2f} {c:>10.4f} {g:>+10.4f} {b:>9.4f} "
                  "{o:>5} {s:>8.3f} {e:>8.4f}".format(
                      t=row["tier"], al=row["alpha"],
                      c=row["coverage_test"], g=row["coverage_gap"],
                      b=row["noise_band_2sd"],
                      o=("OUT" if row["gap_outside_band"] else "in"),
                      s=row["mean_set_size"], e=row["empty_rate"]))

    # ---- Step 9: Design B ------------------------------------------------ #
    _banner("STEP 9 - DESIGN B (SECONDARY): queue-holdout magnitude sweep")
    print("  Reported always, promoted to a headline never.")
    print("  Confound, reportable and not removable: queue correlates with")
    print("  `type`, so a queue holdout also shifts the type mix.")

    print("  Calibration always = the split MINUS queue Q, with Q kept in")
    print("  TRAINING so the label space stays intact. The test arm is the")
    print("  ambiguous part of the recorded design, so BOTH readings are run:")
    print("    B1 `full_test`      -- test is all of version 400")
    print("    B2 `held_out_queue` -- test is version-400 rows OF queue Q")
    print("  Neither is dropped. They fail to resolve for DIFFERENT reasons,")
    print("  and that is the reportable outcome (see the summary below).")

    design_b = []
    cal_labels_arr = np.asarray(cal_labels)
    test_labels_arr = np.asarray(test_labels)
    queues = sorted(set(cal_labels_arr.tolist()))

    for variant in ("full_test", "held_out_queue"):
        print("\n  variant: {v}".format(v=variant))
        for queue in queues:
            keep = cal_labels_arr != queue
            n_kept = int(keep.sum())
            sub_labels = cal_labels_arr[keep].tolist()

            if variant == "full_test":
                test_keep = np.ones(len(test_labels_arr), dtype=bool)
            else:
                test_keep = test_labels_arr == queue
            n_test_q = int(test_keep.sum())
            sub_test_labels = test_labels_arr[test_keep].tolist()

            if n_test_q == 0:
                print("  [skip] {q}: no version-400 rows".format(q=queue))
                continue

            auc_q = domain_auc(emb[cal_idx][keep], emb[test_idx][test_keep])
            blocked = auc_q >= DEGENERACY_AUC
            n_test_classes = len(set(sub_test_labels))

            for tier in TIERS:
                cal_probs, test_probs, classes = probs[tier]
                for alpha in ALPHAS:
                    base = {
                        "design": "B",
                        "variant": variant,
                        "held_out_queue": queue,
                        "tier": tier,
                        "alpha": alpha,
                        "n_calibration": n_kept,
                        "n_test": n_test_q,
                        "n_test_classes": n_test_classes,
                        "domain_auc": auc_q,
                        "blocked_degenerate_auc": blocked,
                        "score_function": SCORE_FUNCTION,
                        "post_hoc": False,
                    }
                    if blocked:
                        base["status"] = "BLOCKED_AUC_GE_0.95"
                        design_b.append(base)
                        continue
                    if n_kept < cp.min_calibration_size(alpha):
                        base["status"] = "insufficient_calibration_n"
                        design_b.append(base)
                        continue

                    row = conformal_row(cal_probs[keep], sub_labels,
                                        test_probs[test_keep],
                                        sub_test_labels, classes, alpha)
                    row.update(base)
                    # A single-class test arm turns "marginal coverage" into
                    # "is this one class in the set", which is a different
                    # quantity. Named on the row rather than left for a
                    # reader to infer from a suspiciously round gap.
                    row["status"] = ("ok" if n_test_classes > 1
                                     else "single_class_test_arm")
                    design_b.append(row)

            shown = [r for r in design_b
                     if r["variant"] == variant
                     and r["held_out_queue"] == queue and r["alpha"] == 0.10]
            if blocked:
                gaps = "BLOCKED (AUC >= {d})".format(d=DEGENERACY_AUC)
            else:
                gaps = ", ".join("{t} {g:+.4f}".format(
                    t=r["tier"], g=r.get("coverage_gap", float("nan")))
                    for r in shown)
            print("    {q:32} AUC {a:.4f} | n_cal {n:>5} n_test {m:>5} | "
                  "alpha=0.10: {g}".format(q=queue[:32], a=auc_q, n=n_kept,
                                           m=n_test_q, g=gaps))

    # ---- Design B verdict, against the pre-registered degeneracy rule ---- #
    print("\n  DESIGN B VERDICT")
    for variant in ("full_test", "held_out_queue"):
        aucs = sorted({r["domain_auc"] for r in design_b
                       if r["variant"] == variant})
        span = aucs[-1] - aucs[0]
        single = any(r.get("status") == "single_class_test_arm"
                     for r in design_b if r["variant"] == variant)
        print("    {v:15} AUC range {lo:.4f}-{hi:.4f} (span {s:.4f})".format(
            v=variant, lo=aucs[0], hi=aucs[-1], s=span))
        if span < 0.05:
            print("      -> NO RESOLUTION: the magnitude knob does not move. "
                  "Holding one queue")
            print("         out of calibration barely changes a shift the "
                  "version split already dominates.")
        if single:
            print("      -> NO RESOLUTION: the test arm carries ONE class, so "
                  "coverage becomes")
            print("         class-conditional for that class rather than "
                  "marginal. Different quantity.")

    # ---- Step 10: write -------------------------------------------------- #
    _banner("STEP 10 - Writing results")
    manifest = {
        "phase": "7B",
        "measurement_only": True,
        "is_real_production_data": False,
        "provenance_note": provenance["provenance_note"],
        "repo_id": provenance["repo_id"],
        "revision": provenance["revision"],
        "config_fingerprint": config_fingerprint(),
        "seed": SEED,
        "score_function": SCORE_FUNCTION,
        "alphas": list(ALPHAS),
        "train_fraction": TRAIN_FRACTION,
        "near_duplicate_threshold": NEAR_DUP_THRESHOLD,
        "source_file_prefix": AA_PREFIX,
        "calibration_versions": list(CAL_VERSIONS),
        "test_version": TEST_VERSION,
        "n_pool_raw": int(len(pool_idx)),
        "n_pool_components": int(n_components),
        "pool_distinct_rate": float(n_components / len(pool_idx)),
        "n_train": int(len(train_idx)),
        "n_calibration": int(len(cal_idx)),
        "n_test": int(len(test_idx)),
        "per_queue": per_queue,
        "domain_auc_raw_arms": auc_raw,
        "domain_auc_operating": auc_operating,
        "recorded_design_a_auc_7a_gate": RECORDED_DESIGN_A_AUC,
        "recorded_auc_reproduces": bool(record_matches),
        "embedding_alignment_worst_cosine": worst_cosine,
        "embedding_alignment_probe_rows": ALIGNMENT_PROBE_ROWS,
        "train_row_indices": train_idx.tolist(),
        "calibration_row_indices": cal_idx.tolist(),
        "test_row_indices": test_idx.tolist(),
        "test_has_near_neighbour": has_neighbour.tolist(),
    }
    write_json(SPLITS_JSON, manifest)
    write_csv(DESIGN_A_CSV, design_a)
    write_csv(DESIGN_B_CSV, design_b)
    write_json(CONTAMINATION_JSON, contamination)
    write_json(LABEL_NOISE_JSON, label_noise)

    # The deferral secondary reads THESE probabilities rather than refitting:
    # a second fit would be a second model, and a comparison run against a
    # different model than the one this phase measured is the exact shape of
    # this project's recurring bug.
    np.savez_compressed(
        PROBS_NPZ,
        tier1_cal=probs["tier1"][0], tier1_test=probs["tier1"][1],
        tier2_cal=probs["tier2"][0], tier2_test=probs["tier2"][1],
        tier1_classes=np.asarray(probs["tier1"][2]),
        tier2_classes=np.asarray(probs["tier2"][2]),
        cal_labels=np.asarray(cal_labels),
        test_labels=np.asarray(test_labels),
        has_near_neighbour=has_neighbour,
    )
    print("[write] {p}".format(p=os.path.relpath(PROBS_NPZ, PROJECT_ROOT)))

    # ---- Step 11: isolation ---------------------------------------------- #
    _banner("STEP 11 - Isolation check")
    verify_isolation(before)

    _banner("DONE - Phase 7B primary measurement complete")
    print("Next: python src/experiments/compare_deferral_rules_external.py")
    return design_a, design_b


def parse_args(argv):
    parser = argparse.ArgumentParser(
        description="Phase 7B -- replicate Finding 1 on the external corpus.")
    parser.add_argument("--force", action="store_true",
                        help="overwrite existing 7B result files")
    return parser.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    run(force=args.force)
    return 0


if __name__ == "__main__":
    sys.exit(main())
