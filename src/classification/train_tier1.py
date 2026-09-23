# src/classification/train_tier1.py
"""
Fit and PERSIST the cascade's Tier-1 model (TF-IDF + LogisticRegression).
================================================================================

WHY THIS EXISTS
---------------
Tier-1 used to be refitted from data/synthetic_tickets.csv on every process
start (artifacts._train_tier1). That is acceptable for a script and wrong for
a service: the orchestrator phase serves tickets per request, and a
per-request service must not derive a model from raw training data at boot.

The measured saving is real but modest -- fitting costs ~1.6s, loading the
persisted bundle ~0.014s, against BGE's ~60s. Speed is not the point. The
point is that the model becomes a pinned, fingerprinted artifact instead of
something re-derived at each startup from whatever the CSV happens to contain.

CRITICAL -- THE FULL DATASET, NOT AN 80/20 SPLIT
------------------------------------------------
Every other training script in this directory uses
train_test_split(test_size=0.2, random_state=42). Tier-1 does NOT. It is
fitted on all 4,000 rows, because that is what the live demo did and what the
Phase 0 goldens were captured under. Fitting on 3,200 rows instead would shift
every Tier-1 confidence, and with it every cascade routing decision -- a wrong
value that stays internally consistent and produces wrong results with no
error. That is precisely this project's recurring bug class.

So the number of rows actually fitted is recorded in the artifact's manifest
and verified at load time, and tests/test_artifacts.py pins it.

THE MANIFEST IS THE GUARD
-------------------------
A persisted model is a new opportunity for a stale artifact to survive a change
to the thing it was derived from. Tier-1's identity is not an embedding model,
so it cannot be encoded in the filename the way the BGE artifacts are. It is
the DATASET plus the vectorizer/sklearn configuration -- so those travel inside
the artifact as a manifest, and artifacts._load_tier1() refuses to load a
bundle whose manifest no longer matches reality.

Since Phase 8B.3 the identity also includes the VOCABULARY. Tier-1 is no longer
fitted with max_features=5000, because that let an unstable sort choose 760 of
the 5,000 features and a different CPU chose differently -- see VOCABULARY_NAME
below. The term -> column mapping is a committed artifact, and its hash is part
of the manifest.

verify_manifest() lives here, next to the code that writes the manifest,
specifically so the writer and the checker cannot drift apart.

Run from the project root:
    python src/classification/train_tier1.py
"""

from __future__ import annotations

import hashlib
import os
import sys
from datetime import datetime, timezone

_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(_THIS_DIR, os.pardir, os.pardir))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from src.agent.config import settings                      # noqa: E402
from src.agent.logging_setup import ensure_utf8_console     # noqa: E402

# Bumped to 2 in Phase 8B.3: the bundle now records the vocabulary it was
# fitted against, so a v1 bundle (whose vocabulary was chosen by an unstable
# sort) is rejected rather than loaded.
MANIFEST_VERSION = 2

# Keys artifacts._load_tier1() relies on. A bundle missing any of them is
# rejected rather than loaded on partial evidence.
REQUIRED_MANIFEST_KEYS = (
    "manifest_version",
    "dataset_name",
    "dataset_sha256",
    "dataset_rows",
    "fitted_rows",
    "classes",
    "sklearn_version",
    "vectorizer_params",
    "vocabulary_sha256",
    "vocabulary_size",
)

# The committed term -> column mapping. Line i of the file IS column i.
#
# WHY THIS FILE EXISTS (Phase 8B.3). TfidfVectorizer(max_features=5000) keeps
# the 5,000 most frequent terms using `(-tfs).argsort()`, an UNSTABLE
# quicksort. On this corpus 4,240 terms are strictly above the cut and 11,834
# tie at count 1 for the remaining 760 slots, so **760 of the 5,000 features
# (15.2%) are chosen by the sort's tie-break, not by the data**. numpy 2.x
# dispatches SIMD sorts by CPU, so another machine keeps a different
# vocabulary: a GitHub runner produced adv_08 tier1_conf 0.31818032412549274
# against this machine's 0.3182984770932253, a difference of 1.18e-4 -- 1e11
# times float64 noise, from a model that had not changed.
#
# Fixing the vocabulary makes the feature space a committed artifact instead of
# a property of the CPU that happened to fit it.
VOCABULARY_NAME = "tier1_vocabulary.txt"

REQUIRED_COLUMNS = ("title", "description", "category")


# --------------------------------------------------------------------------- #
# Manifest: build, and verify. Kept side by side on purpose.                   #
# --------------------------------------------------------------------------- #
def dataset_sha256(path) -> str:
    """Content hash of the dataset Tier-1 was fitted on.

    Read in chunks so this stays cheap at load time (~5ms for 1.5MB) -- the
    guard has to run on every startup, so it must not cost more than the fit
    it replaces.
    """
    digest = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def vocabulary_path():
    """data/tier1_vocabulary.txt, beside the dataset Tier-1 is fitted on."""
    return settings.models.dataset_path.parent / VOCABULARY_NAME


def vocabulary_sha256(path) -> str:
    """Content hash of the vocabulary file, CRLF normalised to LF.

    Normalised for the reason Phase 8B.1 established: raw working-tree bytes
    are a property of the machine, not of the content, so hashing them makes a
    file look changed after a checkout on another operating system.
    """
    with open(path, "rb") as fh:
        data = fh.read()
    return hashlib.sha256(data.replace(b"\r\n", b"\n")).hexdigest()


def load_vocabulary(path=None) -> dict:
    """Return {term: column index} from the committed vocabulary file.

    Fails loudly and specifically. There is deliberately NO fallback to
    `max_features=5000`: that fallback is precisely the nondeterminism this
    file exists to remove, and it would reintroduce it silently on exactly the
    machines where it matters.
    """
    path = vocabulary_path() if path is None else path
    if not os.path.isfile(path):
        raise SystemExit(
            f"[ERROR] Tier-1 vocabulary not found:\n    - {path}\n"
            "  Tier-1 is fitted against a COMMITTED vocabulary so that the\n"
            "  same 5,000 features are used on every machine. Restore the\n"
            "  file from git; do not fall back to max_features."
        )

    with open(path, "r", encoding="utf-8") as fh:
        terms = [line.rstrip("\n").rstrip("\r") for line in fh]
    while terms and terms[-1] == "":
        terms.pop()

    if not terms:
        raise SystemExit(f"[ERROR] {path} is empty.")
    if len(set(terms)) != len(terms):
        raise SystemExit(
            f"[ERROR] {path} contains duplicate terms; the term -> column "
            "mapping would be ambiguous.")

    return {term: index for index, term in enumerate(terms)}


def _vectorizer_params(vectorizer) -> dict:
    """The vectorizer settings that change Tier-1's output if they change.

    Phase 8B.3: `max_features` is None on the production vectorizer now -- the
    feature set comes from the committed vocabulary instead -- so the manifest
    records `fixed_vocabulary` as well. Both are kept: a bundle fitted the old
    way (max_features set, fixed_vocabulary False) is then visibly different
    from one fitted the new way, rather than merely differing in its numbers.
    """
    return {
        "max_features": vectorizer.max_features,
        "ngram_range": list(vectorizer.ngram_range),
        "stop_words": vectorizer.stop_words,
        "fixed_vocabulary": bool(getattr(vectorizer, "fixed_vocabulary_",
                                         vectorizer.vocabulary is not None)),
    }


def build_manifest(vectorizer, classifier, *, dataset_path, dataset_rows,
                   fitted_rows) -> dict:
    """Record everything needed to detect that this artifact went stale."""
    import sklearn

    return {
        "manifest_version": MANIFEST_VERSION,
        "dataset_name": os.path.basename(str(dataset_path)),
        "dataset_sha256": dataset_sha256(dataset_path),
        "dataset_rows": int(dataset_rows),
        "fitted_rows": int(fitted_rows),
        "classes": sorted(str(c) for c in classifier.classes_),
        "sklearn_version": sklearn.__version__,
        "vectorizer_params": _vectorizer_params(vectorizer),
        "vocabulary_sha256": vocabulary_sha256(vocabulary_path()),
        "vocabulary_size": int(len(vectorizer.vocabulary_)),
        "seed": int(settings.seed),
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "fitted_on": "full_dataset",
    }


def verify_manifest(manifest, vectorizer, classifier, *, dataset_path) -> list:
    """Return a list of human-readable problems; empty means the bundle is good.

    Returns rather than raises so the caller decides how to render the failure
    -- the library/entry-point split this project already uses for artifacts.

    Note there is deliberately no re-parse of the CSV here. The dataset hash
    pins the file byte-for-byte, and `dataset_rows` was counted from that same
    verified file at training time, so comparing `fitted_rows` against it is
    sufficient to prove a full-dataset fit without paying for a pandas read on
    every startup.
    """
    if not isinstance(manifest, dict):
        return ["the artifact has no manifest (it predates the load-time "
                "guard, or was written by something else)"]

    missing = [k for k in REQUIRED_MANIFEST_KEYS if k not in manifest]
    if missing:
        return [f"manifest is missing required key(s): {sorted(missing)}"]

    problems = []

    if manifest["manifest_version"] != MANIFEST_VERSION:
        problems.append(
            f"manifest_version is {manifest['manifest_version']}, "
            f"this code expects {MANIFEST_VERSION}"
        )

    live_sha = dataset_sha256(dataset_path)
    if manifest["dataset_sha256"] != live_sha:
        problems.append(
            "the dataset has changed since Tier-1 was fitted:\n"
            f"        fitted against sha256 "
            f"{manifest['dataset_sha256'][:16]}...\n"
            f"        {os.path.basename(str(dataset_path))} is now "
            f"{live_sha[:16]}..."
        )

    if manifest["fitted_rows"] != manifest["dataset_rows"]:
        problems.append(
            f"Tier-1 was fitted on {manifest['fitted_rows']} of "
            f"{manifest['dataset_rows']} rows. It MUST be fitted on the full "
            "dataset -- an 80/20 split shifts every Tier-1 confidence and "
            "every cascade routing decision with it."
        )

    import sklearn

    if manifest["sklearn_version"] != sklearn.__version__:
        problems.append(
            f"scikit-learn is {sklearn.__version__}, but this artifact was "
            f"fitted with {manifest['sklearn_version']}. Refit rather than "
            "assume the two agree."
        )

    vocab_file = vocabulary_path()
    if not os.path.isfile(vocab_file):
        problems.append(
            f"the committed Tier-1 vocabulary is missing: {vocab_file}")
    else:
        live_vocab_sha = vocabulary_sha256(vocab_file)
        if manifest["vocabulary_sha256"] != live_vocab_sha:
            problems.append(
                "the Tier-1 vocabulary has changed since the model was "
                "fitted:\n"
                f"        fitted against "
                f"{manifest['vocabulary_sha256'][:16]}...\n"
                f"        {VOCABULARY_NAME} is now {live_vocab_sha[:16]}...\n"
                "        The feature space is committed on purpose -- refit "
                "rather than assume the two agree."
            )

    if manifest["vocabulary_size"] != len(vectorizer.vocabulary_):
        problems.append(
            f"the bundled vectorizer has {len(vectorizer.vocabulary_)} "
            f"features, the manifest says {manifest['vocabulary_size']}")

    live_params = _vectorizer_params(vectorizer)
    if manifest["vectorizer_params"] != live_params:
        problems.append(
            "the bundled vectorizer does not match its own manifest:\n"
            f"        manifest: {manifest['vectorizer_params']}\n"
            f"        bundle:   {live_params}"
        )

    bundle_classes = sorted(str(c) for c in classifier.classes_)
    if manifest["classes"] != bundle_classes:
        problems.append(
            "the bundled classifier's classes do not match its own "
            f"manifest:\n        manifest: {manifest['classes']}\n"
            f"        bundle:   {bundle_classes}"
        )

    return problems


# --------------------------------------------------------------------------- #
# Training data. Assembled EXACTLY as artifacts._train_tier1() did, because     #
# the goldens were captured under that assembly.                               #
# --------------------------------------------------------------------------- #
def load_training_frame(dataset_path):
    """Return (texts, labels, n_rows) from the full dataset."""
    import pandas as pd

    df = pd.read_csv(dataset_path)
    missing = set(REQUIRED_COLUMNS) - set(df.columns)
    if missing:
        raise SystemExit(
            f"[ERROR] {os.path.basename(str(dataset_path))} is missing "
            f"required column(s): {sorted(missing)}\n"
            f"        Found: {list(df.columns)}"
        )

    texts = (
        df["title"].fillna("").astype(str)
        + " "
        + df["description"].fillna("").astype(str)
    ).tolist()
    labels = df["category"].astype(str).tolist()
    return texts, labels, len(df)


def main() -> None:
    ensure_utf8_console()

    try:
        import joblib
    except ImportError:
        print("[ERROR] Missing required package: joblib")
        print("        Install it with:  pip install joblib")
        raise SystemExit(1)

    try:
        import numpy as np
    except ImportError:
        print("[ERROR] Missing required package: numpy")
        print("        Install it with:  pip install numpy")
        raise SystemExit(1)

    from src.classification.train_cascade import (
        get_tier1_confidence,
        train_tier1,
    )

    dataset_path = settings.models.dataset_path
    out_path = settings.models.tier1_classifier_path

    print("=" * 70)
    print("TRAIN TIER-1  --  TF-IDF + LogisticRegression, persisted")
    print("=" * 70)

    if not dataset_path.is_file():
        print(f"\n[ERROR] Dataset not found:\n    - {dataset_path}\n")
        print("        Generate it from the project root with:")
        print("            python data/generate_dataset.py")
        raise SystemExit(1)

    texts, labels, n_rows = load_training_frame(dataset_path)
    print(f"\n[step1] Dataset       : {dataset_path.name}")
    print(f"[step1] Rows          : {n_rows}")
    print(f"[step1] Categories    : {len(set(labels))}")
    print("[step1] Fitting on the FULL dataset (no 80/20 split) -- this "
          "matches\n        the live demo and the behaviour the goldens "
          "were captured under.")

    vocabulary = load_vocabulary()
    print(f"[step1] Using the COMMITTED vocabulary: {len(vocabulary)} terms "
          f"from {VOCABULARY_NAME}")
    print("        max_features would choose 760 of these 5,000 by an "
          "unstable sort\n        tie-break among 11,834 terms tied at count "
          "1 (Phase 8B.3).")

    vectorizer, classifier = train_tier1(texts, labels, vocabulary=vocabulary)
    print(f"\n[step2] Fitted. Vocabulary: "
          f"{len(vectorizer.vocabulary_)} features")

    manifest = build_manifest(
        vectorizer, classifier,
        dataset_path=dataset_path, dataset_rows=n_rows,
        fitted_rows=len(texts),
    )
    print(f"[step3] Dataset sha256: {manifest['dataset_sha256'][:16]}...")
    print(f"[step3] scikit-learn  : {manifest['sklearn_version']}")

    out_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(
        {"vectorizer": vectorizer, "classifier": classifier,
         "manifest": manifest},
        out_path,
    )
    size_kb = out_path.stat().st_size / 1024
    print(f"\n[step4] Wrote {out_path}  ({size_kb:.0f} KB)")

    # ---- Post-write verification ------------------------------------------
    # Never trust that a persisted model reproduces the one in memory; check
    # it. If this ever fails, the artifact is not a faithful copy and the
    # cascade would route differently after a restart.
    bundle = joblib.load(out_path)
    problems = verify_manifest(
        bundle["manifest"], bundle["vectorizer"], bundle["classifier"],
        dataset_path=dataset_path,
    )
    if problems:
        print("\n[ERROR] The artifact just written does not pass its own "
              "load-time guard:")
        for problem in problems:
            print(f"    - {problem}")
        raise SystemExit(1)

    sample = texts[:200]
    live_preds, live_conf = get_tier1_confidence(vectorizer, classifier,
                                                 sample)
    disk_preds, disk_conf = get_tier1_confidence(
        bundle["vectorizer"], bundle["classifier"], sample
    )

    same_preds = list(live_preds) == list(disk_preds)
    max_diff = float(np.max(np.abs(np.asarray(live_conf)
                                   - np.asarray(disk_conf))))
    if not same_preds or max_diff != 0.0:
        print("\n[ERROR] The persisted model does not reproduce the fitted "
              "model exactly:")
        print(f"    - predictions identical : {same_preds}")
        print(f"    - max confidence delta  : {max_diff!r}")
        raise SystemExit(1)

    print(f"[step5] Round-trip verified on {len(sample)} rows: predictions "
          "identical,\n        max confidence delta 0.0")
    print("\n" + "=" * 70)
    print("DONE -- Tier-1 is persisted. The pipeline now loads it instead of")
    print("        refitting at startup.")
    print("=" * 70)


if __name__ == "__main__":
    main()
