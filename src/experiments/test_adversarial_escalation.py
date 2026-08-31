"""
src/experiments/test_adversarial_escalation.py
==============================================

Adversarial escalation regression test for the AI-Powered Intelligent Ticket
Routing & Resolution Agent.

WHAT THIS SCRIPT DOES
---------------------
Loads a fixed 9-ticket adversarial test set from
data/adversarial_escalation_tickets.json, runs each ticket through the SAME
live classification-cascade + RAG-retrieval pipeline the Streamlit demo uses,
and checks whether the system's ACTUAL human-escalation decision matches the
expected_escalate field in the JSON. Produces a pass/fail report to the
terminal and a CSV.

WHY THE ESCALATION DECISION IS WHAT IT IS (source of truth: streamlit_app.py)
-----------------------------------------------------------------------------
In src/app/streamlit_app.py's run_pipeline(), the flow is:

    1. classify_ticket_cascade(...)   -> picks a CATEGORY (Tier-1 if its
                                         confidence >= 0.50, else Tier-2).
                                         This decides WHICH MODEL LABELS the
                                         ticket. It does NOT, by itself, decide
                                         human escalation.
    2. SR.retrieve_similar_tickets(...) -> always runs; yields top_similarity.
    3. HUMAN ESCALATION DECISION:
           escalated = (top_similarity < SIMILARITY_THRESHOLD)   # 0.35
       Gemini is only called when top_similarity >= threshold.

So in the live pipeline the ACTUAL human-escalation decision depends SOLELY on
the RAG top-similarity vs SIMILARITY_THRESHOLD (0.35). The cascade tier only
affects the predicted category. This script therefore computes actual_escalate
as (top_similarity < SIMILARITY_THRESHOLD), byte-for-byte matching the live
code path. tier1_confidence and the resolving tier are recorded as DIAGNOSTIC
columns only; they never drive actual_escalate.

That is also why the JSON's expected_trigger field can be either
"rag_similarity" or "cascade_confidence_or_rag_similarity": both ultimately
escalate through the same 0.35 similarity gate; the field only documents WHY a
ticket is adversarial, and is informational here.

GEMINI IS NEVER CALLED
----------------------
This diagnostic only needs the escalate / don't-escalate decision, which is
fully determined BEFORE the Gemini call in the live pipeline. To avoid wasting
API quota on adversarial tickets (per the test's purpose), this script does
not construct a Gemini client or call the LLM at all. It reuses the exact same
threshold logic the live pipeline uses to reach the escalation decision.

REUSE, DON'T REIMPLEMENT
------------------------
This script imports and calls the project's real functions:
  - src.classification.train_cascade.train_tier1
  - src.classification.train_cascade.get_tier1_confidence
  - src.rag.suggest_resolution.retrieve_similar_tickets  (+ SIMILARITY_THRESHOLD)
  - src.app.streamlit_app.classify_ticket_cascade
        (imported when possible so cascade behavior is guaranteed identical to
         the live demo; a local, signature-identical fallback is used only if
         streamlit_app cannot be imported headlessly, e.g. because importing it
         triggers Streamlit page calls.)

Run from the project root:
    python src/experiments/test_adversarial_escalation.py

Windows/PowerShell friendly: all paths built with os.path.*; no Unix-only
assumptions. Fixed seed 42 for consistency with the rest of the project even
though this is pure inference over 9 fixed tickets.
"""

import os
import sys
import json
import csv
import random
import traceback

import numpy as np


# ---------------------------------------------------------------------------
# Reproducibility (project convention). Pure inference over 9 fixed tickets
# should not touch RNG, but we seed defensively to match the house style.
# ---------------------------------------------------------------------------
random.seed(42)
np.random.seed(42)


# ---------------------------------------------------------------------------
# PATHS / CONSTANTS
# ---------------------------------------------------------------------------
# Project-wide convention: project root is TWO directories up from this file.
#   this file: <root>/src/experiments/test_adversarial_escalation.py
#   ->         <root>/src/experiments
#   ->         <root>/src
#   ->         <root>
PROJECT_ROOT = os.path.abspath(
    os.path.join(os.path.dirname(__file__), os.pardir, os.pardir)
)

# Make the project root importable so `from src.<pkg> import ...` works no
# matter which directory the script is launched from (mirrors the fallback
# pattern used by streamlit_app.py and plot_calibration_curves.py).
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

DATA_DIR = os.path.join(PROJECT_ROOT, "data")
MODELS_DIR = os.path.join(PROJECT_ROOT, "models")

ADVERSARIAL_JSON_PATH = os.path.join(DATA_DIR, "adversarial_escalation_tickets.json")
RESULTS_CSV_PATH = os.path.join(DATA_DIR, "adversarial_escalation_results.csv")

SYNTHETIC_TICKETS_CSV = os.path.join(DATA_DIR, "synthetic_tickets.csv")
FAISS_INDEX_PATH = os.path.join(DATA_DIR, "ticket_index_bge-base-en-v1-5.faiss")
METADATA_PATH = os.path.join(DATA_DIR, "ticket_metadata_bge-base-en-v1-5.json")
CLASSIFIER_PATH = os.path.join(MODELS_DIR, "ticket_classifier_bge-base-en-v1-5.joblib")

EMBEDDING_MODEL_NAME = "BAAI/bge-base-en-v1.5"

# The APPLIED live thresholds. Mirrors streamlit_app.py exactly:
#   - CASCADE_CONFIDENCE_THRESHOLD = 0.50 (the value streamlit_app.py applies;
#     it is intentionally the applied constant, NOT train_cascade's default of
#     0.80, which is only a pre-calibration fallback there).
#   - SIMILARITY_THRESHOLD is imported from suggest_resolution.py (0.35), with a
#     0.35 fallback identical to streamlit_app.py's getattr(...) default.
CASCADE_CONFIDENCE_THRESHOLD = 0.50

EXPECTED_TICKET_COUNT = 9
TOP_K = 5

# Fields every JSON ticket record must contain.
REQUIRED_TICKET_FIELDS = [
    "id",
    "category_type",
    "text",
    "expected_escalate",
    "expected_trigger",
    "note",
]

VALID_EXPECTED_TRIGGERS = {
    "rag_similarity",
    "cascade_confidence_or_rag_similarity",
    "none",
}


# ---------------------------------------------------------------------------
# PRINT HELPERS (house verbosity convention — banners + actionable errors,
# matching plot_calibration_curves.py's _banner / _fatal style).
# ---------------------------------------------------------------------------
def _banner(text):
    """Print a clearly-delimited section banner."""
    bar = "=" * 70
    print("\n" + bar)
    print(text)
    print(bar)


def _fatal(message):
    """
    Print a clear, actionable error message (NO raw traceback) and exit.
    Matches this project's 'no raw tracebacks for expected failures'
    convention. Genuinely-unexpected errors are caught by __main__'s
    last-resort guard instead.
    """
    print("\n" + "!" * 70)
    print("FATAL: " + message)
    print("!" * 70)
    sys.exit(1)


def _combined_text(title, description):
    """
    Build combined input text exactly like streamlit_app.py's combined_text():
        f"{title} {description}".strip()
    Guards against NaN / non-string cells coming out of pandas.
    """
    def _clean(v):
        if v is None:
            return ""
        try:
            if isinstance(v, float) and np.isnan(v):
                return ""
        except (TypeError, ValueError):
            pass
        return str(v)

    return f"{_clean(title)} {_clean(description)}".strip()


# ---------------------------------------------------------------------------
# IMPORT REUSED PROJECT FUNCTIONS
# ---------------------------------------------------------------------------
def _import_cascade_functions():
    """
    Import train_tier1 + get_tier1_confidence from src.classification.
    train_cascade, mirroring the sys.path fallback pattern used elsewhere.
    """
    print("[import] Importing Tier-1 functions from "
          "src.classification.train_cascade ...")
    try:
        from src.classification.train_cascade import (
            train_tier1,
            get_tier1_confidence,
        )
        print("[import] OK: train_tier1 + get_tier1_confidence")
        return train_tier1, get_tier1_confidence
    except Exception as exc_pkg:
        print(f"[import] Package-style import failed ({exc_pkg!r}); "
              f"trying sys.path injection...")

    classification_dir = os.path.join(PROJECT_ROOT, "src", "classification")
    if classification_dir not in sys.path:
        sys.path.insert(0, classification_dir)

    try:
        from train_cascade import train_tier1, get_tier1_confidence  # type: ignore
        print("[import] OK: imported bare module 'train_cascade'")
        return train_tier1, get_tier1_confidence
    except Exception as exc_bare:
        _fatal(
            "Could not import train_tier1 / get_tier1_confidence from "
            "src.classification.train_cascade.\n"
            f"  Last error: {exc_bare!r}\n"
            "  Checked sys.path entries including:\n"
            f"    - {PROJECT_ROOT}\n"
            f"    - {classification_dir}\n"
            "  Confirm src/classification/train_cascade.py exists and defines "
            "both functions."
        )


def _import_rag_layer():
    """
    Import the RAG layer (suggest_resolution) so we can reuse
    retrieve_similar_tickets and SIMILARITY_THRESHOLD, mirroring the
    package/bare fallback used by streamlit_app.py's load_resources().
    """
    print("[import] Importing RAG layer from src.rag.suggest_resolution ...")
    try:
        from src.rag import suggest_resolution as sr
        print("[import] OK: src.rag.suggest_resolution")
        return sr
    except Exception as exc_pkg:
        print(f"[import] Package-style import failed ({exc_pkg!r}); "
              f"trying sys.path injection...")

    rag_dir = os.path.join(PROJECT_ROOT, "src", "rag")
    if rag_dir not in sys.path:
        sys.path.insert(0, rag_dir)

    try:
        import suggest_resolution as sr  # type: ignore
        print("[import] OK: imported bare module 'suggest_resolution'")
        return sr
    except Exception as exc_bare:
        _fatal(
            "Could not import src/rag/suggest_resolution.py.\n"
            f"  Last error: {exc_bare!r}\n"
            f"  Checked sys.path entries including:\n"
            f"    - {PROJECT_ROOT}\n"
            f"    - {rag_dir}\n"
            "  Confirm the file exists and that you are running from the "
            "project root."
        )


def _resolve_classify_cascade(train_tier1, get_tier1_confidence):
    """
    Return the cascade-classification function to use.

    PREFERRED: import streamlit_app.classify_ticket_cascade so cascade behavior
    is guaranteed byte-for-byte identical to the live demo.

    FALLBACK: importing streamlit_app at module load executes Streamlit page
    setup (st.set_page_config, load_resources with st.cache_resource, etc.),
    which is unsafe/likely to fail in a headless terminal. If that import
    fails, use a LOCAL function that is a signature- and logic-identical copy
    of streamlit_app.classify_ticket_cascade (same threshold, same tier-2
    predict_proba/classes_ handling). The fallback is clearly reported.
    """
    print("[import] Attempting to reuse streamlit_app.classify_ticket_cascade "
          "(exact live cascade) ...")
    try:
        from src.app.streamlit_app import classify_ticket_cascade  # type: ignore
        print("[import] OK: reusing live streamlit_app.classify_ticket_cascade")
        return classify_ticket_cascade, "streamlit_app.classify_ticket_cascade"
    except Exception as exc:
        print(f"[import] Could not import streamlit_app headlessly "
              f"({type(exc).__name__}: {exc}).")
        print("[import] Falling back to a LOCAL, logic-identical copy of "
              "classify_ticket_cascade")
        print("[import]   (same 0.50 threshold, same Tier-2 "
              "predict_proba/classes_ handling as the live function).")

    def classify_ticket_cascade(
        text,
        tier1_vectorizer,
        tier1_classifier,
        tier2_classifier,
        embedding_model,
        get_tier1_confidence_fn,
    ):
        """Local mirror of streamlit_app.py's classify_ticket_cascade()."""
        tier1_preds, tier1_confs = get_tier1_confidence_fn(
            tier1_vectorizer, tier1_classifier, [text]
        )
        tier1_pred = str(np.asarray(tier1_preds).ravel()[0])
        tier1_conf = float(np.asarray(tier1_confs).ravel()[0])

        if tier1_conf >= CASCADE_CONFIDENCE_THRESHOLD:
            return {
                "category": tier1_pred,
                "confidence": tier1_conf,
                "tier": 1,
                "tier1_pred": tier1_pred,
                "tier1_conf": tier1_conf,
            }

        # Escalate to Tier-2 (category-labeling escalation, NOT human escalation).
        embedding = embedding_model.encode([text])
        embedding = np.asarray(embedding, dtype=np.float32)
        tier2_pred = tier2_classifier.predict(embedding)[0]

        tier2_conf = None
        if hasattr(tier2_classifier, "predict_proba"):
            proba = tier2_classifier.predict_proba(embedding)[0]
            classes = list(tier2_classifier.classes_)
            try:
                idx = classes.index(tier2_pred)
                tier2_conf = float(proba[idx])
            except ValueError:
                tier2_conf = float(np.max(proba))

        return {
            "category": str(tier2_pred),
            "confidence": tier2_conf,
            "tier": 2,
            "tier1_pred": tier1_pred,
            "tier1_conf": tier1_conf,
        }

    return classify_ticket_cascade, "local copy (streamlit_app not importable)"


# ---------------------------------------------------------------------------
# STEP 1 — Load + validate the 9-ticket adversarial set
# ---------------------------------------------------------------------------
def load_adversarial_tickets(path=ADVERSARIAL_JSON_PATH):
    """
    Load and validate the adversarial ticket set. Fails loudly (clean message,
    no traceback) on any structural problem, mirroring train_cascade.py's
    loader validation style.
    """
    _banner("STEP 1 — Loading the adversarial escalation test set")

    if not os.path.isfile(path):
        _fatal(
            "Adversarial ticket set not found at:\n"
            f"    {path}\n"
            "  Expected data/adversarial_escalation_tickets.json relative to "
            "the project root.\n"
            "  Create this file (a JSON array of 9 ticket objects) before "
            "running this test."
        )

    try:
        with open(path, "r", encoding="utf-8") as fh:
            data = json.load(fh)
    except Exception as exc:
        _fatal(f"Failed to read/parse adversarial JSON at {path}: {exc!r}")

    if not isinstance(data, list) or len(data) == 0:
        _fatal(
            f"Adversarial JSON at {path} must be a non-empty JSON array, got: "
            f"{type(data).__name__}."
        )

    if len(data) != EXPECTED_TICKET_COUNT:
        _fatal(
            f"Adversarial JSON at {path} must contain EXACTLY "
            f"{EXPECTED_TICKET_COUNT} tickets, but found {len(data)}."
        )

    # Per-record structural validation. Collect ALL offenders, don't bail on
    # the first (matches the project's loader convention).
    bad = []
    seen_ids = {}
    for i, rec in enumerate(data):
        if not isinstance(rec, dict):
            bad.append((i, f"record is {type(rec).__name__}, not an object"))
            continue

        missing = [k for k in REQUIRED_TICKET_FIELDS if k not in rec]
        if missing:
            bad.append((i, f"missing field(s): {missing}"))
            continue

        if not isinstance(rec["id"], str) or not rec["id"].strip():
            bad.append((i, "'id' must be a non-empty string"))
        else:
            seen_ids.setdefault(rec["id"], []).append(i)

        if not isinstance(rec["text"], str) or not rec["text"].strip():
            bad.append((i, "'text' must be a non-empty string"))

        if not isinstance(rec["expected_escalate"], bool):
            bad.append((
                i,
                "'expected_escalate' must be a JSON boolean (true/false), got "
                f"{type(rec['expected_escalate']).__name__}",
            ))

        trig = rec.get("expected_trigger")
        if not isinstance(trig, str) or trig not in VALID_EXPECTED_TRIGGERS:
            bad.append((
                i,
                "'expected_trigger' must be one of "
                f"{sorted(VALID_EXPECTED_TRIGGERS)}, got {trig!r}",
            ))

    # Duplicate-id detection.
    dupes = {tid: idxs for tid, idxs in seen_ids.items() if len(idxs) > 1}
    for tid, idxs in dupes.items():
        bad.append((idxs[0], f"duplicate id '{tid}' also at record(s) {idxs[1:]}"))

    if bad:
        lines = "\n".join(f"        record #{i}: {why}" for i, why in bad)
        _fatal(
            f"Adversarial JSON at {path} has {len(bad)} problem(s):\n{lines}\n"
            "  Fix the file and re-run."
        )

    n_expected_escalate = sum(1 for r in data if r["expected_escalate"])
    print(f"[step1] Loaded {len(data)} adversarial tickets from "
          f"{os.path.relpath(path, PROJECT_ROOT)}")
    print(f"[step1] Expected to ESCALATE      : {n_expected_escalate}")
    print(f"[step1] Expected to NOT escalate  : {len(data) - n_expected_escalate}")
    return list(data)


# ---------------------------------------------------------------------------
# STEP 2 — Load live pipeline resources (Tier-1 trained fresh, Tier-2 +
#          FAISS index + metadata loaded from disk), mirroring
#          streamlit_app.py's load_resources() prerequisite checks.
# ---------------------------------------------------------------------------
def load_pipeline_resources(train_tier1, sr):
    """
    Build the same resources the live demo uses. Each missing prerequisite
    produces a clean, actionable message (no traceback), exactly like
    streamlit_app.py's load_resources() returns an "error" dict.

    NOTE: unlike streamlit_app.py, this diagnostic deliberately does NOT
    require GEMINI_API_KEY and does NOT build a Gemini client, because it never
    calls the LLM (the escalation decision is fully determined before any LLM
    call in the live pipeline).
    """
    _banner("STEP 2 — Loading live pipeline resources")

    # ---- Third-party deps ------------------------------------------------
    try:
        import faiss  # noqa: F401
        import joblib
        import pandas as pd
        from sentence_transformers import SentenceTransformer
    except Exception as exc:
        _fatal(
            "A required Python package failed to import "
            f"({type(exc).__name__}: {exc}).\n"
            "  Install the project dependencies first, e.g.:\n"
            "      pip install -r requirements.txt"
        )

    # ---- Prerequisite files (same checks/messages as load_resources) -----
    if not os.path.isfile(FAISS_INDEX_PATH):
        _fatal(
            "FAISS vector index not found at:\n"
            f"    {FAISS_INDEX_PATH}\n"
            "  Build it first by running:\n"
            "      python src/rag/build_vector_index.py"
        )

    if not os.path.isfile(METADATA_PATH):
        _fatal(
            "Ticket metadata not found at:\n"
            f"    {METADATA_PATH}\n"
            "  Build it first by running:\n"
            "      python src/rag/build_vector_index.py"
        )

    if not os.path.isfile(CLASSIFIER_PATH):
        _fatal(
            "Trained Tier-2 classifier not found at:\n"
            f"    {CLASSIFIER_PATH}\n"
            "  Train it first by running:\n"
            "      python src/classification/train_embeddings.py"
        )

    if not os.path.isfile(SYNTHETIC_TICKETS_CSV):
        _fatal(
            "Ticket dataset not found at:\n"
            f"    {SYNTHETIC_TICKETS_CSV}\n"
            "  This is needed to train the Tier-1 (TF-IDF) fast-path model at "
            "startup, exactly as the live demo does. Generate it first."
        )

    # ---- Load heavy artifacts -------------------------------------------
    try:
        print(f"[step2] Loading embedding model: {EMBEDDING_MODEL_NAME}")
        model = SentenceTransformer(EMBEDDING_MODEL_NAME)
    except Exception as exc:
        _fatal(
            f"Failed to load embedding model '{EMBEDDING_MODEL_NAME}' "
            f"({type(exc).__name__}: {exc}).\n"
            "  This normally runs offline once cached. If this is the first "
            "run, ensure internet access so the model can download."
        )

    try:
        print(f"[step2] Loading Tier-2 classifier: {CLASSIFIER_PATH}")
        tier2_classifier = joblib.load(CLASSIFIER_PATH)
    except Exception as exc:
        _fatal(
            "Failed to load the Tier-2 classifier "
            f"({type(exc).__name__}: {exc}).\n"
            "  Re-train it with: python src/classification/train_embeddings.py"
        )

    try:
        print(f"[step2] Loading metadata: {METADATA_PATH}")
        with open(METADATA_PATH, "r", encoding="utf-8") as fh:
            metadata = json.load(fh)
    except Exception as exc:
        _fatal(f"Failed to read metadata JSON ({type(exc).__name__}: {exc}).")

    if not isinstance(metadata, list):
        _fatal(
            f"{METADATA_PATH} is not a JSON list as expected. Rebuild it with:\n"
            "      python src/rag/build_vector_index.py"
        )

    try:
        print(f"[step2] Loading FAISS index: {FAISS_INDEX_PATH}")
        index = faiss.read_index(FAISS_INDEX_PATH)
    except Exception as exc:
        _fatal(
            f"Failed to read the FAISS index ({type(exc).__name__}: {exc}).\n"
            "  Rebuild it with: python src/rag/build_vector_index.py"
        )

    # Critical sync check, same spirit as suggest_resolution.load_index_and_metadata.
    if index.ntotal != len(metadata):
        _fatal(
            "INDEX / METADATA OUT OF SYNC:\n"
            f"    FAISS index.ntotal = {index.ntotal}\n"
            f"    len(metadata)      = {len(metadata)}\n"
            "  These MUST match, or retrieved similarities would be "
            "misaligned. Rebuild BOTH together with:\n"
            "      python src/rag/build_vector_index.py"
        )
    print(f"[step2] Index/metadata in sync: {index.ntotal} entries.")

    # ---- Train Tier-1 fresh at startup (same as the live demo) -----------
    try:
        print(f"[step2] Reading dataset for Tier-1 training: "
              f"{SYNTHETIC_TICKETS_CSV}")
        df = pd.read_csv(SYNTHETIC_TICKETS_CSV)
    except Exception as exc:
        _fatal(f"Failed to read {SYNTHETIC_TICKETS_CSV} "
               f"({type(exc).__name__}: {exc}).")

    required_cols = {"title", "description", "category"}
    missing_cols = required_cols - set(df.columns)
    if missing_cols:
        _fatal(
            f"{os.path.basename(SYNTHETIC_TICKETS_CSV)} is missing required "
            f"column(s): {sorted(missing_cols)}.\n"
            f"  Found columns: {list(df.columns)}"
        )

    try:
        tier1_texts = (
            df["title"].fillna("").astype(str)
            + " "
            + df["description"].fillna("").astype(str)
        ).tolist()
        tier1_labels = df["category"].astype(str).tolist()
        print(f"[step2] Training Tier-1 (TF-IDF + LogReg) on "
              f"{len(tier1_texts):,} tickets (fresh, as the live demo does)...")
        tier1_vectorizer, tier1_classifier = train_tier1(tier1_texts, tier1_labels)
    except Exception as exc:
        _fatal(
            "Failed to train the Tier-1 fast-path classifier "
            f"({type(exc).__name__}: {exc})."
        )

    # ---- Resolve the applied SIMILARITY_THRESHOLD (import; 0.35 fallback) -
    similarity_threshold = float(getattr(sr, "SIMILARITY_THRESHOLD", 0.35))
    print(f"[step2] Applied thresholds -> cascade confidence "
          f"{CASCADE_CONFIDENCE_THRESHOLD:.2f}, RAG similarity "
          f"{similarity_threshold:.2f}")

    return {
        "model": model,
        "tier1_vectorizer": tier1_vectorizer,
        "tier1_classifier": tier1_classifier,
        "tier2_classifier": tier2_classifier,
        "metadata": metadata,
        "index": index,
        "faiss": faiss,
        "sr": sr,
        "similarity_threshold": similarity_threshold,
    }


# ---------------------------------------------------------------------------
# STEP 3 — Run one ticket through the live pipeline logic
# ---------------------------------------------------------------------------
def run_ticket_through_pipeline(ticket, resources, classify_cascade_fn,
                                get_tier1_confidence):
    """
    Run a single ticket's text through the SAME pipeline logic as
    streamlit_app.py's run_pipeline():

        (a) cascade classification (category + tier + tier1 confidence)
        (b) RAG retrieval -> top_similarity
        (c) actual_escalate = (top_similarity < SIMILARITY_THRESHOLD)

    Gemini is intentionally NOT called (see module docstring).

    Returns a dict of recorded fields. On a per-ticket failure it returns a
    dict with "error" set so the caller can render it cleanly and count it as
    a FAIL rather than crashing the whole run.
    """
    text = ticket["text"]
    sr = resources["sr"]

    # (a) Cascade classification (reuse the live function).
    classification = classify_cascade_fn(
        text,
        resources["tier1_vectorizer"],
        resources["tier1_classifier"],
        resources["tier2_classifier"],
        resources["model"],
        get_tier1_confidence,
    )
    predicted_category = classification["category"]
    tier = classification["tier"]
    tier1_conf = classification["tier1_conf"]

    # (b) RAG retrieval (reuse the live function, exact signature/arg order).
    retrieved = sr.retrieve_similar_tickets(
        text,
        resources["model"],
        resources["index"],
        resources["metadata"],
        resources["faiss"],
        top_k=TOP_K,
    ) or []
    # Defensive sort (matches streamlit_app.run_pipeline).
    retrieved = sorted(
        retrieved, key=lambda r: r.get("similarity", 0.0), reverse=True
    )
    top_similarity = float(retrieved[0]["similarity"]) if retrieved else 0.0

    # (c) Human-escalation decision — SOLELY the RAG similarity gate, exactly
    #     as run_pipeline() decides result["escalated"].
    actual_escalate = top_similarity < resources["similarity_threshold"]

    return {
        "predicted_category": str(predicted_category),
        "tier": int(tier),
        "tier1_confidence": float(tier1_conf),
        "rag_similarity": float(top_similarity),
        "actual_escalate": bool(actual_escalate),
        "n_retrieved": len(retrieved),
    }


# ---------------------------------------------------------------------------
# STEP 4 — Per-ticket report + result assembly
# ---------------------------------------------------------------------------
def _fmt_conf(v):
    """Format a confidence/similarity value, tolerating None/NaN."""
    if v is None:
        return "n/a"
    try:
        if isinstance(v, float) and np.isnan(v):
            return "n/a"
    except (TypeError, ValueError):
        pass
    return f"{v:.4f}"


def run_all_tickets(tickets, resources, classify_cascade_fn,
                    get_tier1_confidence):
    """
    Run every ticket, print a banner-style per-ticket section, and collect a
    result row per ticket. A per-ticket exception is caught, reported cleanly,
    and recorded as a FAIL (never aborts the whole run).
    """
    _banner("STEP 3 — Running each adversarial ticket through the LIVE pipeline")
    print("[step3] Human escalation depends SOLELY on RAG top-similarity vs "
          f"the {resources['similarity_threshold']:.2f} threshold, exactly as "
          "streamlit_app.py decides it.")
    print("[step3] The cascade tier only labels the category; it does not, by "
          "itself, escalate to a human.")
    print("[step3] Gemini is NOT called for any ticket (quota preserved).")

    results = []
    for i, ticket in enumerate(tickets, start=1):
        tid = ticket["id"]
        expected_escalate = bool(ticket["expected_escalate"])

        print("\n" + "-" * 70)
        print(f"TICKET {i}/{len(tickets)}  id={tid}  "
              f"category_type={ticket['category_type']}")
        print("-" * 70)
        preview = ticket["text"].replace("\n", " ").strip()
        if len(preview) > 200:
            preview = preview[:199].rstrip() + "…"
        print(f"  text            : {preview}")
        print(f"  expected_escalate : {expected_escalate}")
        print(f"  expected_trigger  : {ticket['expected_trigger']}")

        row = {
            "id": tid,
            "category_type": ticket["category_type"],
            "text": ticket["text"],
            "expected_escalate": expected_escalate,
            "actual_escalate": None,
            "pass_fail": None,
            "tier1_confidence": None,
            "rag_similarity": None,
            "predicted_category": None,
            "note": ticket["note"],
        }

        try:
            outcome = run_ticket_through_pipeline(
                ticket, resources, classify_cascade_fn, get_tier1_confidence
            )
        except Exception as exc:
            # Per-ticket failure: clean message, no raw traceback here, mark FAIL.
            print(f"  [PIPELINE ERROR] {type(exc).__name__}: {exc}")
            print("  Recorded as FAIL for this ticket; continuing with the rest.")
            row["actual_escalate"] = "error"
            row["pass_fail"] = "FAIL"
            row["_error"] = f"{type(exc).__name__}: {exc}"
            results.append(row)
            continue

        actual_escalate = outcome["actual_escalate"]
        passed = (actual_escalate == expected_escalate)

        row["actual_escalate"] = actual_escalate
        row["pass_fail"] = "PASS" if passed else "FAIL"
        row["tier1_confidence"] = outcome["tier1_confidence"]
        row["rag_similarity"] = outcome["rag_similarity"]
        row["predicted_category"] = outcome["predicted_category"]

        tier_label = ("Tier-1 (fast)" if outcome["tier"] == 1
                      else "Tier-2 (cascade-escalated label)")
        print(f"  predicted_category: {outcome['predicted_category']}")
        print(f"  category resolved by: {tier_label}")
        print(f"  tier1_confidence  : {_fmt_conf(outcome['tier1_confidence'])} "
              f"(cascade threshold {CASCADE_CONFIDENCE_THRESHOLD:.2f})")
        print(f"  rag_similarity    : {_fmt_conf(outcome['rag_similarity'])} "
              f"(human-escalation threshold "
              f"{resources['similarity_threshold']:.2f}, "
              f"{outcome['n_retrieved']} retrieved)")
        print(f"  actual_escalate   : {actual_escalate}")
        verdict = "PASS ✔" if passed else "FAIL �’ MISMATCH"
        print(f"  RESULT            : {row['pass_fail']}  ->  {verdict}")

        results.append(row)

    return results


# ---------------------------------------------------------------------------
# STEP 5 — Final summary
# ---------------------------------------------------------------------------
def print_summary(results):
    """Print X/9 passed and explicitly list every FAIL with its mismatch."""
    _banner("STEP 4 — Summary")

    total = len(results)
    passes = [r for r in results if r["pass_fail"] == "PASS"]
    fails = [r for r in results if r["pass_fail"] != "PASS"]

    print(f"[summary] {len(passes)}/{total} passed.")

    if not fails:
        print("[summary] All adversarial escalation expectations were met. ✔")
        return

    print(f"[summary] {len(fails)} FAIL(s):")
    for r in fails:
        # Errored tickets get a distinct explanation.
        if r.get("_error") is not None:
            print(f"[summary]   - {r['id']}: pipeline ERROR "
                  f"({r['_error']}); could not evaluate escalation.")
            continue

        exp = r["expected_escalate"]
        act = r["actual_escalate"]
        if exp and not act:
            phrase = "expected to escalate but the system did NOT escalate it"
        elif act and not exp:
            phrase = "expected NOT to escalate but the system escalated it"
        else:
            # Shouldn't happen for a bool mismatch, but stay defensive.
            phrase = (f"expected_escalate={exp} but actual_escalate={act}")
        sim = _fmt_conf(r["rag_similarity"])
        print(f"[summary]   - {r['id']} {phrase} "
              f"(rag_similarity={sim}, predicted_category="
              f"{r['predicted_category']}).")


# ---------------------------------------------------------------------------
# STEP 6 — Write CSV report
# ---------------------------------------------------------------------------
def write_results_csv(results, path=RESULTS_CSV_PATH):
    """
    Write one row per ticket to the CSV, columns in the exact order requested:
        id, category_type, text, expected_escalate, actual_escalate,
        pass_fail, tier1_confidence, rag_similarity, predicted_category, note
    """
    _banner("STEP 5 — Writing CSV report")

    fieldnames = [
        "id",
        "category_type",
        "text",
        "expected_escalate",
        "actual_escalate",
        "pass_fail",
        "tier1_confidence",
        "rag_similarity",
        "predicted_category",
        "note",
    ]

    def _cell(v):
        """Serialize a value for CSV: numbers rounded, None -> '', bools kept."""
        if v is None:
            return ""
        if isinstance(v, bool):
            return str(v)  # "True"/"False"
        if isinstance(v, float):
            try:
                if np.isnan(v):
                    return ""
            except (TypeError, ValueError):
                pass
            return f"{v:.6f}"
        return v

    try:
        # newline="" is the correct, cross-platform (incl. Windows) way to
        # avoid blank lines in csv output.
        with open(path, "w", encoding="utf-8", newline="") as fh:
            writer = csv.DictWriter(fh, fieldnames=fieldnames)
            writer.writeheader()
            for r in results:
                writer.writerow({k: _cell(r.get(k)) for k in fieldnames})
        print(f"[step5] Wrote {len(results)} row(s) to: "
              f"{os.path.relpath(path, PROJECT_ROOT)}")
    except Exception as exc:
        # Non-fatal: the terminal report already stands on its own.
        print(f"[step5] WARNING: could not write CSV report {path}: "
              f"{type(exc).__name__}: {exc}")


# ---------------------------------------------------------------------------
# MAIN
# ---------------------------------------------------------------------------
def main():
    _banner("test_adversarial_escalation.py — Adversarial Escalation Regression")
    print(f"[main] Project root: {PROJECT_ROOT}")
    print(f"[main] Data dir:     {DATA_DIR}")
    print(f"[main] Models dir:   {MODELS_DIR}")

    # Resolve reused functions first so import problems fail early and clearly.
    train_tier1, get_tier1_confidence = _import_cascade_functions()
    sr = _import_rag_layer()
    classify_cascade_fn, cascade_source = _resolve_classify_cascade(
        train_tier1, get_tier1_confidence
    )
    print(f"[main] Cascade classification source: {cascade_source}")

    # STEP 1 — load the fixed adversarial set.
    tickets = load_adversarial_tickets()

    # STEP 2 — build the live pipeline resources.
    resources = load_pipeline_resources(train_tier1, sr)

    # STEP 3 — run every ticket.
    results = run_all_tickets(
        tickets, resources, classify_cascade_fn, get_tier1_confidence
    )

    # STEP 4 — summary.
    print_summary(results)

    # STEP 5 — CSV.
    write_results_csv(results)

    _banner("DONE — adversarial escalation test complete")

    # Exit code reflects pass/fail so this is CI/script friendly:
    #   0 = all passed, 1 = at least one FAIL.
    n_fail = sum(1 for r in results if r["pass_fail"] != "PASS")
    if n_fail:
        print(f"[main] Exit status: FAIL ({n_fail} ticket(s) did not match).")
        sys.exit(1)
    print("[main] Exit status: PASS (all tickets matched expectations).")
    sys.exit(0)


if __name__ == "__main__":
    try:
        main()
    except SystemExit:
        # _fatal() / main()'s explicit sys.exit already printed context;
        # propagate the intended exit code.
        raise
    except Exception as exc:
        # Absolute last-resort guard so the user never sees a raw traceback
        # without context — matching plot_calibration_curves.py's __main__.
        print("\n" + "!" * 70)
        print("UNEXPECTED ERROR — the script stopped before completing.")
        print(f"  {type(exc).__name__}: {exc}")
        print("  (Full trace below for debugging; see messages above for the "
              "last successful step.)")
        print("!" * 70)
        traceback.print_exc()
        sys.exit(1)
