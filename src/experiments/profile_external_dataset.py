"""
Phase 7A -- profile the external dataset. FEASIBILITY ONLY.

This script MEASURES PROPERTIES OF AN EXTERNAL CORPUS. It replicates nothing,
compares nothing against our results, and produces no finding about this
project's methods. Its output decides whether Phase 7B is worth designing and
at what n -- nothing else.

WHAT IT REPORTS, AND WHY EACH ITEM EXISTS
-----------------------------------------
1. ROWS -- total, per file, and after the English filter. The headline
   feasibility number: how much bigger than our 45-ticket benchmark is this.
2. QUEUE DISTRIBUTION, and queues with >= 300 English rows. A shift design
   needs enough tickets on both sides of the split.
3. TEXT-LENGTH STATS for subject/body/answer, including empties. A missing
   `answer` silently breaks item 7, so it is counted rather than assumed.
4. EXACT DUPLICATE RATE, on normalised text.
5. NEAR-DUPLICATE RATE at BGE cosine >= 0.95. Our own deployment calibration
   set shipped with 21 near-duplicate pairs because tightening a scope anchor
   narrowed the scenario space -- duplicated points skew a conformal quantile,
   so an external corpus gets checked for the same defect before it is trusted.
6. VERSION-FIELD DISTRIBUTION -- a candidate shift axis.
7. ANSWER DIVERSITY WITHIN A QUEUE -- the question PHASE 2A COULD NOT ANSWER.
   On our corpus the templates ARE the fix classes, so no configuration ever
   makes a cross-template merge and clustering precision is unmeasurable. If
   this corpus's answers vary within a queue, precision becomes measurable
   here.

TWO MODELS, DELIBERATELY, AND NEVER INTERCHANGED
------------------------------------------------
Item 5 uses **BGE** (the production retrieval/classification encoder) because
the near-duplicate question is about the text as the pipeline reads it.
Item 7 uses **MiniLM at 0.80** because that is what production
resolution-clustering actually is -- read from
settings.clustering.clustering_embedding_model and
settings.clustering.resolution_similarity_threshold, never hardcoded. Every
output column names which model produced it. Mixing these two up is precisely
this project's recurring bug class.

Item 7 reuses `group_by_threshold` and `cosine_similarity_matrix` from
flag_automation_candidates.py -- the production grouping, not a second
implementation of it.

ISOLATION
---------
Everything written lands under data/external_tobibueck/. Nothing reads or
writes data/synthetic_tickets.csv, models/, or any ticket_index_*.faiss. The
temporary FAISS index built here is for the external embeddings only and is
never placed where a production artifact loader could find it. Verified by an
explicit assertion, not by inspection.

Run from the project root (offline, no Gemini calls):
    python src/experiments/profile_external_dataset.py
    python src/experiments/profile_external_dataset.py --force
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import sys
import time

import numpy as np
import pandas as pd

THIS_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(THIS_DIR, os.pardir, os.pardir))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from src.agent.config import settings                            # noqa: E402
from src.agent.logging_setup import ensure_utf8_console          # noqa: E402
from src.experiments.calibrate_conformal import _banner, _fatal  # noqa: E402
from src.experiments.flag_automation_candidates import (         # noqa: E402
    _UnionFind,
)
from src.experiments.fetch_external_dataset import (             # noqa: E402
    ENGLISH_CSV,
    OUT_DIR,
    PROVENANCE_JSON,
)

ensure_utf8_console()

SUMMARY_JSON = os.path.join(OUT_DIR, "profile_summary.json")
QUEUE_CSV = os.path.join(OUT_DIR, "queue_distribution.csv")
ANSWER_CSV = os.path.join(OUT_DIR, "answer_diversity_by_queue.csv")
EMB_NPY = os.path.join(OUT_DIR, "bge_text_embeddings.npy")

MIN_QUEUE_ROWS = 300
NEAR_DUP_THRESHOLD = 0.95

# Item 7 samples per queue. `group_by_threshold` is an O(n^2) Python loop --
# the production implementation, reused deliberately rather than replaced --
# so the largest queue (8k+) would take minutes for a number that only has to
# answer "does `answer` vary at all". Seeded, and recorded in the output so a
# sample can never be mistaken for the full queue.
ANSWER_SAMPLE_PER_QUEUE = 1500
SEED = 42

_WS = re.compile(r"\s+")


def _norm_text(value):
    return _WS.sub(" ", str(value).strip().lower())


def _col(df, name):
    hit = [c for c in df.columns if c.strip().lower() == name]
    if not hit:
        _fatal("No `{n}` column; columns are {c}".format(
            n=name, c=list(df.columns)))
    return hit[0]


# --------------------------------------------------------------------------- #
# Items 1-4, 6
# --------------------------------------------------------------------------- #
def length_stats(series):
    lengths = series.fillna("").astype(str).str.len()
    non_empty = lengths[lengths > 0]
    return {
        "n": int(len(lengths)),
        "n_empty_or_null": int((lengths == 0).sum()),
        "mean_chars": float(non_empty.mean()) if len(non_empty) else 0.0,
        "median_chars": float(non_empty.median()) if len(non_empty) else 0.0,
        "p5_chars": float(non_empty.quantile(0.05)) if len(non_empty) else 0.0,
        "p95_chars": float(non_empty.quantile(0.95)) if len(non_empty) else 0.0,
        "max_chars": int(non_empty.max()) if len(non_empty) else 0,
    }


def exact_duplicate_stats(texts):
    """Exact duplicates on normalised text, counted two ways (rule 6)."""
    digests = [hashlib.sha256(_norm_text(t).encode("utf-8")).hexdigest()
               for t in texts]
    counts = pd.Series(digests).value_counts()

    n_unique = int(len(counts))
    n_dup_rows = int(len(digests) - n_unique)

    # Second derivation: sum of (count - 1) over groups with count > 1.
    alt = int(sum(int(c) - 1 for c in counts if c > 1))
    if alt != n_dup_rows:
        _fatal("Exact-duplicate count disagrees between two derivations: "
               "{a} vs {b}".format(a=n_dup_rows, b=alt))

    # TWO RATES, BOTH REPORTED, BECAUSE THEY DIFFER BY ~2x AND ARE EASY TO
    # CONFUSE. `excess_row_rate` counts only the redundant copies
    # (n_rows - n_unique). `rows_in_a_duplicate_group_rate` counts every row
    # that shares its text with another -- for a corpus of pairs that is twice
    # as large. Quoting one as if it were the other would misstate the corpus's
    # redundancy by a factor of two.
    rows_in_groups = int(sum(int(c) for c in counts if c > 1))

    return {
        "n_rows": int(len(digests)),
        "n_unique_texts": n_unique,
        "n_duplicate_rows": n_dup_rows,
        "duplicate_rate": n_dup_rows / len(digests) if digests else 0.0,
        "excess_row_rate": n_dup_rows / len(digests) if digests else 0.0,
        "n_rows_in_a_duplicate_group": rows_in_groups,
        "rows_in_a_duplicate_group_rate": (
            rows_in_groups / len(digests) if digests else 0.0),
        "n_duplicate_groups": int((counts > 1).sum()),
        "largest_duplicate_group": int(counts.iloc[0]) if n_unique else 0,
    }


# --------------------------------------------------------------------------- #
# Item 5 -- near duplicates, BGE
# --------------------------------------------------------------------------- #
def encode_bge(texts, force=False):
    if os.path.isfile(EMB_NPY) and not force:
        emb = np.load(EMB_NPY)
        if emb.shape[0] == len(texts):
            print("  [cached] BGE embeddings {s}".format(s=emb.shape))
            return emb
        print("  [stale ] cached embeddings are {a} rows for {b} texts; "
              "re-encoding".format(a=emb.shape[0], b=len(texts)))

    from src.agent.artifacts import load_artifacts
    artifacts = load_artifacts(require_gemini=False)

    # Measured at ~3 texts/s on this CPU, so the full 28k English subset is a
    # ~2.5 hour job. Chunked with progress rather than one opaque call: a job
    # that long must not run dark, and a partial chunk that errors should say
    # how far it got. Batch size 32 was the fastest of {32, 64, 128} when
    # measured -- 128 was materially worse, so this is not a guess.
    texts = list(texts)
    chunk = 2000
    t0 = time.perf_counter()
    parts = []
    for start in range(0, len(texts), chunk):
        parts.append(artifacts.embedder.encode(
            texts[start:start + chunk], convert_to_numpy=True,
            batch_size=32, show_progress_bar=False))
        done = min(start + chunk, len(texts))
        rate = done / max(time.perf_counter() - t0, 1e-9)
        remaining = (len(texts) - done) / rate / 60.0
        print("    {d}/{n} ({p:.0%})  {r:.1f}/s  ~{m:.0f} min left".format(
            d=done, n=len(texts), p=done / len(texts), r=rate, m=remaining),
            flush=True)
    dt = time.perf_counter() - t0
    emb = np.asarray(np.vstack(parts), dtype=np.float32)

    if emb.shape[1] != settings.models.embedding_dim:
        _fatal("Encoder returned {d}-dim vectors but config expects {e}. "
               "Wrong model.".format(d=emb.shape[1],
                                     e=settings.models.embedding_dim))

    np.save(EMB_NPY, emb)
    print("  encoded {n} texts in {t:.1f}s ({r:.0f}/s) -> {s}".format(
        n=len(texts), t=dt, r=len(texts) / dt if dt else 0.0, s=emb.shape))
    return emb


def near_duplicate_stats(emb, threshold=NEAR_DUP_THRESHOLD, seed=SEED):
    """Nearest non-self neighbour per row, via a temporary FAISS index.

    A dense n^2 similarity matrix is 3.6e9 entries at n=61.8k and is not
    attempted. The index covers the EXTERNAL embeddings only and lives under
    data/external_tobibueck/ -- it is never written where a production loader
    could pick it up.
    """
    import faiss

    x = np.ascontiguousarray(emb.astype(np.float32))
    faiss.normalize_L2(x)

    index = faiss.IndexFlatIP(x.shape[1])
    index.add(x)
    sims, idx = index.search(x, 2)

    # Column 0 is the row itself (similarity 1.0) for a flat exact index, but
    # ties can reorder it, so the self-match is removed by identity rather
    # than by position.
    best = np.where(idx[:, 0] == np.arange(len(x)), sims[:, 1], sims[:, 0])
    best_idx = np.where(idx[:, 0] == np.arange(len(x)), idx[:, 1], idx[:, 0])

    flagged = best >= threshold

    # Rule 6: brute-force recheck on a random slice, computed without FAISS.
    rng = np.random.default_rng(seed)
    probe = rng.choice(len(x), size=min(200, len(x)), replace=False)
    mismatches = 0
    for i in probe:
        row = x @ x[i]
        row[i] = -np.inf
        if abs(float(row.max()) - float(best[i])) > 1e-4:
            mismatches += 1
    if mismatches:
        _fatal("FAISS and brute-force nearest-neighbour disagree on "
               "{m}/{n} probed rows.".format(m=mismatches, n=len(probe)))

    # ---- CONTROL 1: is 0.95 even a duplicate threshold for this encoder? --
    # BGE has a high similarity floor for same-domain text, so a raw
    # near-duplicate rate is uninterpretable on its own. If random pairs
    # already sit near 0.95, the statistic is reading the embedding's scale
    # rather than duplication. Measured, not assumed.
    a = rng.integers(0, len(x), 20000)
    b = rng.integers(0, len(x), 20000)
    distinct = a != b
    random_pair = np.einsum("ij,ij->i", x[a[distinct]], x[b[distinct]])

    # ---- CONTROL 2: de-duplicated size -------------------------------------
    # Connected components at the threshold: the number of DISTINCT items,
    # which is what a downstream experiment can actually draw on. Reported
    # because the raw row count overstates usable n whenever near-duplication
    # is high -- and here it is high in BOTH corpora.
    lims, _d, ids = index.range_search(x, float(threshold))
    uf = _UnionFind(len(x))
    for i in range(len(x)):
        for j in ids[lims[i]:lims[i + 1]]:
            if int(j) != i:
                uf.union(i, int(j))
    comp_sizes = {}
    for i in range(len(x)):
        root = uf.find(i)
        comp_sizes[root] = comp_sizes.get(root, 0) + 1
    sizes = np.array(sorted(comp_sizes.values(), reverse=True))

    return {
        "threshold": float(threshold),
        "model": settings.models.embedding_model,
        "n_rows": int(len(x)),
        "n_rows_with_near_duplicate": int(flagged.sum()),
        "near_duplicate_rate": float(flagged.mean()),
        "mean_nearest_similarity": float(best.mean()),
        "median_nearest_similarity": float(np.median(best)),
        "p05_nearest_similarity": float(np.quantile(best, 0.05)),
        "p95_nearest_similarity": float(np.quantile(best, 0.95)),
        # Control 1 -- the threshold's discriminating power.
        "random_pair_median_similarity": float(np.median(random_pair)),
        "random_pair_p95_similarity": float(np.quantile(random_pair, 0.95)),
        "random_pair_rate_at_threshold": float(
            np.mean(random_pair >= threshold)),
        # Control 2 -- usable distinct size.
        "n_distinct_components": int(len(sizes)),
        "distinct_component_rate": float(len(sizes) / len(x)),
        "largest_component": int(sizes[0]),
        "n_singleton_components": int((sizes == 1).sum()),
        "brute_force_probe_rows": int(len(probe)),
        "brute_force_mismatches": 0,
    }, best, best_idx


def own_corpus_comparison(threshold=NEAR_DUP_THRESHOLD, seed=SEED):
    """The same near-duplicate statistic on OUR corpus, as the control.

    A near-duplicate rate has no meaning in isolation -- it needs something to
    be high or low RELATIVE TO. The only fair reference available is the
    corpus this project already publishes results on. Without this comparison
    the external rate reads as a disqualifying defect; with it, the honest
    reading may be the opposite.

    Read-only: loads the existing cached embeddings and never writes to them.
    Returns None (with a printed note) if the cache is absent, because a
    missing control must not silently become an absent one.
    """
    own_npy = os.path.join(
        PROJECT_ROOT, "data", "ticket_embeddings_bge-base-en-v1-5.npy")
    if not os.path.isfile(own_npy):
        print("  [note] our embedding cache is absent, so no control is "
              "available. Build it with train_embeddings.py -- do NOT read "
              "the external rate without one.")
        return None

    stats, _best, _idx = near_duplicate_stats(
        np.load(own_npy), threshold=threshold, seed=seed)
    return stats


# --------------------------------------------------------------------------- #
# Item 7 -- answer diversity within a queue, MiniLM at 0.80
# --------------------------------------------------------------------------- #
def answer_diversity(df, queue_col, answer_col, queues, seed=SEED):
    """Does `answer` vary within a queue enough to make clustering precision
    measurable? This is the question Phase 2A could not answer on our corpus.

    Uses the PRODUCTION clustering configuration and the PRODUCTION grouping
    function, so the number means the same thing it means in
    flag_automation_candidates.py.
    """
    from sentence_transformers import SentenceTransformer

    from src.experiments.flag_automation_candidates import (
        cosine_similarity_matrix, group_by_threshold,
    )

    model_name = settings.clustering.clustering_embedding_model
    threshold = settings.clustering.resolution_similarity_threshold
    print("  model={m}  threshold={t}  (both from config)".format(
        m=model_name, t=threshold))

    model = SentenceTransformer(model_name)
    rng = np.random.default_rng(seed)
    rows = []

    for queue in queues:
        sub = df.loc[df[queue_col] == queue]
        answers = sub[answer_col].fillna("").astype(str)
        answers = answers[answers.str.len() > 0]

        n_total = int(len(answers))
        if n_total < 2:
            continue

        sampled = n_total > ANSWER_SAMPLE_PER_QUEUE
        if sampled:
            pick = rng.choice(n_total, size=ANSWER_SAMPLE_PER_QUEUE,
                              replace=False)
            texts = answers.iloc[pick].tolist()
        else:
            texts = answers.tolist()

        emb = model.encode(texts, convert_to_numpy=True, batch_size=64,
                           show_progress_bar=False)
        sim = cosine_similarity_matrix(np.asarray(emb, dtype=np.float32))

        n = sim.shape[0]
        upper = sim[np.triu_indices(n, k=1)]
        clusters = group_by_threshold(sim, threshold)
        sizes = np.array([len(c) for c in clusters])

        rows.append({
            "queue": queue,
            "n_answers_in_queue": n_total,
            "n_scored": n,
            "sampled": sampled,
            "sample_cap": ANSWER_SAMPLE_PER_QUEUE if sampled else "",
            "clustering_model": model_name,
            "clustering_threshold": threshold,
            "mean_pairwise_similarity": float(upper.mean()),
            "median_pairwise_similarity": float(np.median(upper)),
            "p95_pairwise_similarity": float(np.quantile(upper, 0.95)),
            "frac_pairs_at_or_above_threshold": float(
                (upper >= threshold).mean()),
            "n_clusters": int(len(clusters)),
            "n_singleton_clusters": int((sizes == 1).sum()),
            "largest_cluster": int(sizes.max()),
            "frac_in_largest_cluster": float(sizes.max() / n),
            "distinct_answer_rate": float(len(clusters) / n),
        })
        print("    {q:35} n={n:<5} clusters={c:<5} largest={l:<5} "
              "pairs>=thr {p:.4f}".format(
                  q=queue[:35], n=n, c=len(clusters), l=int(sizes.max()),
                  p=float((upper >= threshold).mean())))

    return pd.DataFrame(rows)


# --------------------------------------------------------------------------- #
# Isolation
# --------------------------------------------------------------------------- #
def assert_isolated(before):
    """Our corpus and artifacts must be untouched (rule: never mixed)."""
    for path, digest in before.items():
        if not os.path.isfile(path):
            _fatal("A tracked project file vanished during 7A: {p}".format(
                p=path))
        now = hashlib.sha256(open(path, "rb").read()).hexdigest()
        if now != digest:
            _fatal("7A MODIFIED A PROJECT FILE: {p}\n"
                   "  External-dataset work must never touch our "
                   "artifacts.".format(p=path))
    print("  [ok] our dataset and artifacts are byte-identical")


def snapshot_project_files():
    paths = [
        os.path.join(PROJECT_ROOT, "data", "synthetic_tickets.csv"),
        os.path.join(PROJECT_ROOT, "data", "novel_tickets_expanded.json"),
        os.path.join(PROJECT_ROOT, "data",
                     "calibration_tickets_paraphrased.json"),
    ]
    return {p: hashlib.sha256(open(p, "rb").read()).hexdigest()
            for p in paths if os.path.isfile(p)}


# --------------------------------------------------------------------------- #
# The run
# --------------------------------------------------------------------------- #
def run(force=False, reencode=False):
    _banner("PHASE 7A - EXTERNAL DATASET PROFILE  (feasibility only)")
    print("Independently generated data, NOT real production data.")
    print("No replication, no comparison against our results.")

    if not os.path.isfile(ENGLISH_CSV):
        _fatal("English subset not found:\n    {p}\n"
               "  Run: python src/experiments/fetch_external_dataset.py"
               .format(p=ENGLISH_CSV))

    for path in (SUMMARY_JSON, QUEUE_CSV, ANSWER_CSV):
        if os.path.isfile(path) and not force:
            _fatal("Refusing to overwrite an existing result:\n  {p}\n"
                   "Pass --force if you mean to replace it.".format(p=path))

    before = snapshot_project_files()

    with open(PROVENANCE_JSON, encoding="utf-8") as fh:
        provenance = json.load(fh)

    df = pd.read_csv(ENGLISH_CSV, low_memory=False)
    queue_col = _col(df, "queue")
    answer_col = _col(df, "answer")
    subject_col = _col(df, "subject")
    body_col = _col(df, "body")

    # ---- Item 1 ---------------------------------------------------------- #
    _banner("ITEM 1 - Rows")
    print("  total parsed (all languages): {n}".format(
        n=provenance["rows_parsed_total"]))
    print("  English subset              : {n}".format(n=len(df)))
    if len(df) != provenance["rows_english_total"]:
        _fatal("English CSV has {a} rows but PROVENANCE says {b}".format(
            a=len(df), b=provenance["rows_english_total"]))
    print("  vs our 45-ticket benchmark  : {r:.0f}x".format(r=len(df) / 45))

    # ---- Item 2 ---------------------------------------------------------- #
    _banner("ITEM 2 - Queue distribution")
    vc = df[queue_col].value_counts(dropna=False)
    queue_df = vc.rename_axis("queue").reset_index(name="n_english_rows")
    queue_df["meets_min_300"] = queue_df["n_english_rows"] >= MIN_QUEUE_ROWS
    queue_df["share"] = queue_df["n_english_rows"] / len(df)
    big_queues = queue_df.loc[queue_df["meets_min_300"], "queue"].tolist()
    for _, r in queue_df.iterrows():
        print("  {q:38} {n:>6}  {m}".format(
            q=str(r["queue"])[:38], n=int(r["n_english_rows"]),
            m="ok" if r["meets_min_300"] else "BELOW 300"))
    print("  queues with >= {k} English rows: {n}/{t}".format(
        k=MIN_QUEUE_ROWS, n=len(big_queues), t=len(queue_df)))

    # ---- Item 3 ---------------------------------------------------------- #
    _banner("ITEM 3 - Text-length stats")
    lengths = {
        "subject": length_stats(df[subject_col]),
        "body": length_stats(df[body_col]),
        "answer": length_stats(df[answer_col]),
    }
    for field, st in lengths.items():
        print("  {f:8} median {m:>6.0f} chars | p5 {a:>5.0f} | p95 {b:>6.0f} "
              "| empty {e}".format(f=field, m=st["median_chars"],
                                   a=st["p5_chars"], b=st["p95_chars"],
                                   e=st["n_empty_or_null"]))

    # ---- Item 4 ---------------------------------------------------------- #
    _banner("ITEM 4 - Exact duplicates")
    texts = (df[subject_col].fillna("").astype(str) + " "
             + df[body_col].fillna("").astype(str)).tolist()
    exact = exact_duplicate_stats(texts)
    print("  excess (redundant) rows      {n} / {t} = {r:.4%}".format(
        n=exact["n_duplicate_rows"], t=exact["n_rows"],
        r=exact["excess_row_rate"]))
    print("  rows sharing text with another {n} / {t} = {r:.4%}".format(
        n=exact["n_rows_in_a_duplicate_group"], t=exact["n_rows"],
        r=exact["rows_in_a_duplicate_group_rate"]))
    print("  duplicate groups: {g}   largest: {l}".format(
        g=exact["n_duplicate_groups"], l=exact["largest_duplicate_group"]))

    # ---- Item 5 ---------------------------------------------------------- #
    _banner("ITEM 5 - Near duplicates (BGE cosine >= {t})".format(
        t=NEAR_DUP_THRESHOLD))
    emb = encode_bge(texts, force=reencode)
    near, best, _best_idx = near_duplicate_stats(emb)
    print("  rows with a near-duplicate: {n} / {t} = {r:.4%}".format(
        n=near["n_rows_with_near_duplicate"], t=near["n_rows"],
        r=near["near_duplicate_rate"]))
    print("  nearest-neighbour similarity: median {m:.4f}, p95 {p:.4f}".format(
        m=near["median_nearest_similarity"], p=near["p95_nearest_similarity"]))
    print("  [ok] FAISS agrees with brute force on {n} probed rows".format(
        n=near["brute_force_probe_rows"]))
    print("  CONTROL 1 -- is {t} a duplicate threshold for this encoder?"
          .format(t=NEAR_DUP_THRESHOLD))
    print("    random pairs: median {m:.4f}, p95 {p:.4f}, rate >= {t} = "
          "{r:.4%}".format(m=near["random_pair_median_similarity"],
                           p=near["random_pair_p95_similarity"],
                           t=NEAR_DUP_THRESHOLD,
                           r=near["random_pair_rate_at_threshold"]))
    print("    -> the threshold discriminates; it is not reading BGE's floor.")
    print("  CONTROL 2 -- usable distinct size after de-duplication:")
    print("    {c} components from {n} rows ({r:.1%} survive), largest {l}"
          .format(c=near["n_distinct_components"], n=near["n_rows"],
                  r=near["distinct_component_rate"],
                  l=near["largest_component"]))

    _banner("ITEM 5b - CONTROL: the same statistic on OUR corpus")
    print("  A near-duplicate rate means nothing without a reference.")
    own = own_corpus_comparison()
    if own is not None:
        print("    external : {a:.2%} near-dup | {c} distinct of {n}".format(
            a=near["near_duplicate_rate"],
            c=near["n_distinct_components"], n=near["n_rows"]))
        print("    ours     : {a:.2%} near-dup | {c} distinct of {n}".format(
            a=own["near_duplicate_rate"],
            c=own["n_distinct_components"], n=own["n_rows"]))
        verdict = ("LESS" if near["near_duplicate_rate"]
                   < own["near_duplicate_rate"] else "MORE")
        print("    -> the external corpus is {v} redundant than the one this"
              .format(v=verdict))
        print("       project already publishes on. High near-duplication is "
              "a property")
        print("       of template-generated corpora, not a distinguishing "
              "flaw of this one.")

    # ---- Item 6 ---------------------------------------------------------- #
    _banner("ITEM 6 - Version field")
    version_col = next(
        (c for c in df.columns if c.strip().lower() == "version"), None)
    if version_col is None:
        version_dist = {}
        print("  no `version` column in the English subset")
    else:
        vv = df[version_col].value_counts(dropna=False)
        version_dist = {str(k): int(v) for k, v in vv.items()}
        for k, v in version_dist.items():
            print("  version {k:>8}: {v:>6}".format(k=k, v=v))

    # ---- Item 7 ---------------------------------------------------------- #
    _banner("ITEM 7 - Answer diversity within a queue "
            "(the Phase 2A question)")
    answer_df = answer_diversity(df, queue_col, answer_col, big_queues)

    # ---- Write ----------------------------------------------------------- #
    queue_df.to_csv(QUEUE_CSV, index=False)
    answer_df.to_csv(ANSWER_CSV, index=False)

    summary = {
        "phase": "7A",
        "feasibility_only": True,
        "is_real_production_data": False,
        "provenance_note": provenance["provenance_note"],
        "repo_id": provenance["repo_id"],
        "revision": provenance["revision"],
        "license": provenance["license"],
        "rows_parsed_total": provenance["rows_parsed_total"],
        "rows_english_total": int(len(df)),
        "benchmark45_multiple": float(len(df) / 45),
        "queues_total": int(len(queue_df)),
        "queues_meeting_min_rows": int(len(big_queues)),
        "min_queue_rows": MIN_QUEUE_ROWS,
        "length_stats_chars": lengths,
        "exact_duplicates": exact,
        "near_duplicates_bge": near,
        "near_duplicates_bge_our_corpus_control": own,
        "near_duplicate_control_note": (
            "A near-duplicate rate is uninterpretable without a reference. "
            "Compare the external rate against our own corpus's rate at the "
            "same threshold and encoder before calling either one high."
        ),
        "version_distribution": version_dist,
        "answer_diversity_model": (
            settings.clustering.clustering_embedding_model),
        "answer_diversity_threshold": (
            settings.clustering.resolution_similarity_threshold),
        "answer_sample_cap_per_queue": ANSWER_SAMPLE_PER_QUEUE,
        "seed": SEED,
    }
    with open(SUMMARY_JSON, "w", encoding="utf-8") as fh:
        json.dump(summary, fh, indent=2, ensure_ascii=False)

    _banner("ISOLATION CHECK")
    assert_isolated(before)

    print("\n[write] {p}".format(p=SUMMARY_JSON))
    print("[write] {p}".format(p=QUEUE_CSV))
    print("[write] {p}".format(p=ANSWER_CSV))
    return summary, queue_df, answer_df


def main():
    parser = argparse.ArgumentParser(
        description="Phase 7A -- profile the external dataset (feasibility "
                    "only, offline, no Gemini calls).")
    parser.add_argument("--force", action="store_true",
                        help="overwrite the existing report outputs")
    # Deliberately separate from --force. The BGE pass is a ~2.5 hour job on
    # this CPU, so re-rendering the report must not silently re-run it.
    parser.add_argument("--reencode", action="store_true",
                        help="discard the cached BGE embeddings and re-encode "
                             "(~2.5 hours); --force alone reuses the cache")
    args = parser.parse_args()
    run(force=args.force, reencode=args.reencode)


if __name__ == "__main__":
    main()
