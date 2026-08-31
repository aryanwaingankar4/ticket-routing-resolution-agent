# src/experiments/calibrate_rag_similarity_threshold.py
#
# Diagnostic / measurement-only script.
#
# PURPOSE
# -------
# Re-derive the RAG SIMILARITY_THRESHOLD for the BGE embedding model
# (BAAI/bge-base-en-v1.5, 768-dim) using BOTH the project's 175-ticket
# in-domain Gemini-paraphrased calibration set AND a ~45-ticket out-of-domain
# (OOD) calibration set. The threshold gates a single ticket's single top-1
# retrieval: proceed to LLM-grounded resolution when the top-1 cosine
# similarity is high enough, otherwise escalate to a human.
#
# The original 0.35 threshold was tuned for all-MiniLM-L6-v2. BGE produces
# systematically higher cosine similarities, so 0.35 is broken. A provisional
# 0.65 is currently live. The in-domain set alone is ENTIRELY in-distribution,
# so every ticket "proceeds" across the useful range and the sweep can't see
# escalation behavior; the OOD set supplies the missing negative-class signal
# (tickets that SHOULD always escalate).
#
# This script measures per-threshold confusion matrices and recommends a
# value, WITHOUT touching production
# (src/rag/suggest_resolution.py::SIMILARITY_THRESHOLD is never modified).
#
# This mirrors the conventions of
# src/experiments/calibrate_resolution_clustering_percategory.py:
#   banner()/step()/ok()/warn() helpers, a CalibrationError class caught
#   cleanly in main(), os.path project-root resolution, random.seed(42) +
#   np.random.seed(42), a clean CSV written with csv.DictWriter, and a final
#   human-readable summary. The metric here is PER-TICKET (not pairwise like
#   the clustering script) because this gate concerns one ticket's one
#   top-1 retrieval.
#
# EXTENSIONS (this revision):
#   - PART A: load the OOD set and fold it into a COMBINED confusion matrix
#     (OOD proceed = false positive, OOD escalate = true negative), plus an
#     OOD-only "leakage rate" reported alongside (not fed into the cliff-edge).
#   - PART B: harden the in-domain relevance signal -- a self-retrieval
#     contamination check, a strictly-stronger exact-source-match diagnostic
#     metric reported side-by-side, field-detection hardening, and a
#     non-monotonic cliff-edge diagnostic.
#   - PART C: write a separate _combined.csv (the original CSV is untouched).

import os
import sys
import csv
import json
import random
import statistics
import traceback

import numpy as np

# --------------------------------------------------------------------------
# Path / constants convention.
# This file lives in src/experiments/, so PROJECT_ROOT is TWO pardir hops up
# (src/experiments/<file> -> src/experiments -> src -> project root).
# --------------------------------------------------------------------------
_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(_THIS_DIR, os.pardir, os.pardir))
DATA_DIR = os.path.join(PROJECT_ROOT, "data")

INDEX_PATH = os.path.join(DATA_DIR, "ticket_index_bge-base-en-v1-5.faiss")
METADATA_PATH = os.path.join(DATA_DIR, "ticket_metadata_bge-base-en-v1-5.json")
CALIBRATION_PATH = os.path.join(DATA_DIR, "calibration_tickets_paraphrased.json")
OOD_CALIBRATION_PATH = os.path.join(DATA_DIR, "ood_calibration_tickets.json")
OUTPUT_CSV_PATH = os.path.join(DATA_DIR, "rag_similarity_calibration.csv")
OUTPUT_COMBINED_CSV_PATH = os.path.join(
    DATA_DIR, "rag_similarity_calibration_combined.csv"
)

EMBED_MODEL_NAME = "BAAI/bge-base-en-v1.5"

# Provenance markers for the printed comparison at the end.
OLD_MINILM_THRESHOLD = 0.35        # original, tuned for all-MiniLM-L6-v2
PROVISIONAL_BGE_THRESHOLD = 0.65   # current live provisional value

RULE = "=" * 78

# Threshold sweep: high -> low. Brackets both the old 0.35 and the new
# provisional 0.

SIMILARITY_THRESHOLDS = [
    0.85, 0.80, 0.75,
    0.70, 0.69, 0.68, 0.67, 0.66, 0.65, 0.64, 0.63, 0.62, 0.61, 0.60,
    0.55, 0.50, 0.45, 0.40, 0.35,
]

class CalibrationError(Exception):
    """Raised for expected, user-actionable failures. Caught in __main__ and
    printed cleanly without a traceback."""
    pass


# --------------------------------------------------------------------------
# Small output helpers (match project style).
# --------------------------------------------------------------------------
def banner(title):
    print()
    print(RULE)
    print(title)
    print(RULE)


def step(msg):
    print(f"[*] {msg}")


def ok(msg):
    print(f"[OK] {msg}")


def warn(msg):
    print(f"[!] {msg}")


# --------------------------------------------------------------------------
# Cliff-edge detection.
# --------------------------------------------------------------------------
def find_cliff_edge(rows_by_threshold):
    """
    SCAN-DIRECTION REASONING (stated explicitly, as required):

    As the threshold T INCREASES, the gate gets STRICTER: fewer tickets satisfy
    (top_similarity >= T) and proceed. The tickets that still clear a very high
    T are precisely those with the strongest top-1 matches -- which are the most
    likely to be *relevant* matches. So precision trends toward 1.0 at HIGH T
    and DEGRADES as T is LOWERED (weaker top-1 matches begin slipping into the
    "proceed" set, adding FPs -- irrelevant matches we wrongly proceeded on).

    Therefore the cliff-edge is the LOWEST T at which precision is still exactly
    1.0. We scan from HIGH T DOWN to LOW T. SIMILARITY_THRESHOLDS is already
    ordered high -> low (0.85 ... 0.35), so we iterate it in natural order:
      - while precision == 1.0, keep lowering the recorded cliff to this T
        (we want the lowest still-perfect T),
      - as soon as precision drops below 1.0 at some T, stop (loosening further
        only makes it worse).

    Undefined precision (None) occurs when (TP + FP) == 0, i.e. NO ticket
    proceeds at this T (happens at very high T where the gate rejects
    everything). Such a threshold does not demonstrate perfect precision -- it
    demonstrates nothing -- so we SKIP it and keep scanning, rather than
    treating it as a break. (This is the important None-handling fix carried
    over from the per-category script: an undefined-precision threshold must
    not be misread as "no cliff-edge found".)

    NOTE: this function is intentionally UNCHANGED from the original -- it is
    fed the COMBINED rows. A separate post-hoc diagnostic
    (report_non_monotonic_cliff) checks for a second perfect-precision band
    below the reported cliff that this first-dip-stops scan would miss.
    """
    cliff = None
    for thr in SIMILARITY_THRESHOLDS:  # high -> low
        prec = rows_by_threshold[thr]["precision"]
        if prec is None:
            # No tickets proceeded at this (very tight) T -> undefined, skip.
            continue
        if prec == 1.0:
            cliff = thr  # still perfect; record and keep lowering
        else:
            break  # precision has degraded; loosening further won't recover it
    return cliff


def report_non_monotonic_cliff(rows_by_threshold, cliff):
    """POST-HOC DIAGNOSTIC (item 7): scan the FULL sweep for any threshold
    strictly BELOW the reported cliff that ALSO has precision == 1.0 -- a
    "second perfect band" that find_cliff_edge()'s first-dip-stops scan would
    miss on a noisy 11-point sweep. Does not modify find_cliff_edge(); only
    warns.
    """
    if cliff is None:
        return
    second_band = []
    for thr in SIMILARITY_THRESHOLDS:  # high -> low
        if thr >= cliff:
            continue  # only interested in thresholds strictly below the cliff
        prec = rows_by_threshold[thr]["precision"]
        if prec is not None and prec == 1.0:
            second_band.append(thr)

    if second_band:
        warn("NON-MONOTONIC PRECISION DETECTED (combined metric):")
        warn(f"    Reported cliff-edge (highest perfect-precision run) : "
             f"T = {cliff:.2f}")
        warn(f"    Additional perfect-precision threshold(s) BELOW it  : "
             + ", ".join(f"{t:.2f}" for t in second_band))
        warn("    The reported cliff is the HIGHEST, not necessarily the "
             "ONLY, perfect-precision point. On an 11-point sweep over a "
             "few-hundred-ticket set this can be sampling noise -- inspect "
             "the full table before adopting a value.")
    else:
        ok("Cliff-edge monotonicity check: no additional perfect-precision "
           "band below the reported cliff.")


# --------------------------------------------------------------------------
# Loading helpers.
# --------------------------------------------------------------------------
def load_index_and_metadata(faiss):
    """Load the FAISS index + metadata and enforce the ntotal/len sync check.

    Adapted from suggest_resolution.load_index_and_metadata(), but raising
    CalibrationError instead of doing that module's interactive _fail()/
    sys.exit() CLI error handling -- so failures surface via this script's own
    clean try/except convention.
    """
    if not os.path.exists(INDEX_PATH):
        raise CalibrationError(
            f"FAISS index not found:\n    {INDEX_PATH}\n"
            "Build the BGE index before running calibration."
        )
    if not os.path.exists(METADATA_PATH):
        raise CalibrationError(
            f"Metadata file not found:\n    {METADATA_PATH}\n"
            "Build the BGE metadata before running calibration."
        )

    index = faiss.read_index(INDEX_PATH)
    with open(METADATA_PATH, "r", encoding="utf-8") as f:
        metadata = json.load(f)

    if index.ntotal != len(metadata):
        # FATAL -- index/metadata out of sync, would silently return wrong
        # results (position i in the index must correspond to metadata[i]).
        raise CalibrationError(
            "FAISS index and metadata are OUT OF SYNC -- refusing to continue.\n"
            f"    index.ntotal = {index.ntotal}\n"
            f"    len(metadata) = {len(metadata)}\n"
            "Rebuild the index and metadata together."
        )

    return index, metadata


# Field-detection candidate lists (module-level so both the detector and the
# both-present hardening check reference the SAME lists).
SINGLE_TEXT_CANDIDATES = [
    "text",
    "ticket_text",
    "description",
    "body",
    "content",
    "ticket",
    "query",
]

CATEGORY_FIELD_CANDIDATES = [
    "expected",
    "expected_category",
    "ground_truth_category",
    "ground_truth",
    "true_category",
    "gt_category",
    "category",
    "label",
]


def load_calibration_set():
    """Load the 175-ticket in-domain calibration set and auto-detect fields.

    The exact field names were not independently re-verified in the task prompt,
    so we detect them at load time instead of hard-coding:
      - ground-truth category field: prefer 'expected', else fall back to a
        small set of plausible names, matching the 7 known categories.
      - ticket text field: prefer a combined title+description, else a single
        text-like field.
    The detected names are printed at startup so they are visible in the log.

    FIELD-DETECTION HARDENING (item 6): if a record contains BOTH a 'text'
    field and any 'description'-like candidate, or BOTH 'expected' and
    'category', we refuse to rely on candidate-list ordering and raise a
    CalibrationError -- the caller must disambiguate explicitly rather than
    let the script silently pick one.
    """
    if not os.path.exists(CALIBRATION_PATH):
        raise CalibrationError(
            f"Calibration set not found:\n    {CALIBRATION_PATH}"
        )

    with open(CALIBRATION_PATH, "r", encoding="utf-8") as f:
        calibration = json.load(f)

    if not isinstance(calibration, list) or len(calibration) == 0:
        raise CalibrationError(
            "Calibration set is empty or not a JSON list."
        )

    sample = calibration[0]
    if not isinstance(sample, dict):
        raise CalibrationError(
            "Calibration entries are not dicts; cannot detect fields."
        )

    sample_keys = set(sample.keys())

    # --- HARDENING: ambiguous category fields. ---
    if "expected" in sample_keys and "category" in sample_keys:
        raise CalibrationError(
            "Ambiguous ground-truth field: the calibration records contain "
            "BOTH 'expected' and 'category'. Refusing to guess which is "
            "ground truth via candidate-list ordering. Remove or rename one "
            "so exactly one canonical category field remains.\n"
            f"    Available keys: {sorted(sample_keys)}"
        )

    # --- Detect the ground-truth category field. ---
    category_field = None
    for cand in CATEGORY_FIELD_CANDIDATES:
        if cand in sample:
            category_field = cand
            break
    if category_field is None:
        raise CalibrationError(
            "Could not detect the ground-truth category field in the "
            "calibration set.\n"
            f"    Available keys: {sorted(sample_keys)}\n"
            f"    Tried: {CATEGORY_FIELD_CANDIDATES}"
        )

    # --- Detect the ticket text field(s). ---
    # Assumption (stated explicitly): the query text passed to retrieval should
    # mirror how past tickets are represented. If both a title and description
    # exist we concatenate "title. description"; otherwise we use the single
    # best available text field.
    has_title = "title" in sample
    has_description = "description" in sample

    # --- HARDENING: ambiguous text fields. ---
    # If the record has a bare 'text' field AND any 'description'-like
    # candidate other than 'text' itself, the original code would silently
    # prefer whichever appears first in the candidate list. Refuse instead.
    description_like_present = [
        c for c in SINGLE_TEXT_CANDIDATES
        if c != "text" and c in sample_keys
    ]
    if "text" in sample_keys and description_like_present:
        raise CalibrationError(
            "Ambiguous ticket-text field: the calibration records contain "
            "BOTH 'text' and description-like field(s) "
            f"{description_like_present}. Refusing to guess via candidate-list "
            "ordering. Keep exactly one text source (for the paraphrased "
            "calibration set that should be 'text').\n"
            f"    Available keys: {sorted(sample_keys)}"
        )

    single_text_field = None
    for cand in SINGLE_TEXT_CANDIDATES:
        if cand in sample:
            single_text_field = cand
            break

    if not (has_title or single_text_field):
        raise CalibrationError(
            "Could not detect a ticket text field in the calibration set.\n"
            f"    Available keys: {sorted(sample_keys)}"
        )

    text_field_desc = (
        "title + description"
        if (has_title and has_description)
        else (single_text_field if single_text_field else "title")
    )

    return (calibration, category_field, has_title, has_description,
            single_text_field, text_field_desc)


def load_ood_calibration_set():
    """Load the OOD calibration set (id + text only).

    Fails cleanly via CalibrationError if missing/malformed, matching the
    existing error style. Returns the list of {"id", "text"} dicts.
    """
    if not os.path.exists(OOD_CALIBRATION_PATH):
        raise CalibrationError(
            f"OOD calibration set not found:\n    {OOD_CALIBRATION_PATH}\n"
            "Generate it first with:\n"
            "    python src\\classification\\generate_ood_calibration_set.py"
        )

    with open(OOD_CALIBRATION_PATH, "r", encoding="utf-8") as f:
        ood = json.load(f)

    if not isinstance(ood, list) or len(ood) == 0:
        raise CalibrationError(
            "OOD calibration set is empty or not a JSON list."
        )

    # Minimal schema validation: every entry needs a string id and non-empty
    # text. IDs are expected to be strings prefixed 'ood_'.
    for i, entry in enumerate(ood):
        if not isinstance(entry, dict):
            raise CalibrationError(
                f"OOD entry #{i} is not a dict."
            )
        if "id" not in entry or "text" not in entry:
            raise CalibrationError(
                f"OOD entry #{i} is missing 'id' and/or 'text'.\n"
                f"    Keys present: {sorted(entry.keys())}"
            )
        if not str(entry.get("text", "")).strip():
            raise CalibrationError(
                f"OOD entry #{i} (id={entry.get('id')!r}) has empty text."
            )

    return ood


def build_query_text(entry, has_title, has_description, single_text_field):
    """Construct the query text for a calibration ticket using the detected
    fields, mirroring how a real incoming ticket would be embedded."""
    if has_title and has_description:
        title = str(entry.get("title", "") or "").strip()
        desc = str(entry.get("description", "") or "").strip()
        if title and desc:
            return f"{title}. {desc}"
        return title or desc
    if single_text_field is not None and single_text_field in entry:
        return str(entry.get(single_text_field, "") or "").strip()
    # Last resort: title only.
    return str(entry.get("title", "") or "").strip()


# --------------------------------------------------------------------------
# Metric helpers.
# --------------------------------------------------------------------------
def compute_precision(tp, fp):
    denom = tp + fp
    return (tp / denom) if denom > 0 else None


def compute_recall(tp, fn):
    denom = tp + fn
    return (tp / denom) if denom > 0 else None


def compute_f1(precision, recall):
    if precision is None or recall is None:
        return None
    if (precision + recall) == 0:
        return None
    return 2.0 * precision * recall / (precision + recall)


def fmt(x, nd=4):
    """Format a float or None for the terminal table."""
    if x is None:
        return "  n/a "
    return f"{x:.{nd}f}"


def csv_cell(x):
    """None -> empty string, matching the clustering script's CSV convention."""
    return "" if x is None else x


# --------------------------------------------------------------------------
# Main routine.
# --------------------------------------------------------------------------
def run():
    # Match project convention even though retrieval is deterministic given a
    # fixed FAISS index and a fixed embedding model.
    random.seed(42)
    np.random.seed(42)

    banner("RAG SIMILARITY THRESHOLD CALIBRATION (BGE, per-ticket, +OOD)")
    step(f"Project root : {PROJECT_ROOT}")
    step(f"Index path   : {INDEX_PATH}")
    step(f"Metadata path: {METADATA_PATH}")
    step(f"Calib. path  : {CALIBRATION_PATH}")
    step(f"OOD path     : {OOD_CALIBRATION_PATH}")
    step(f"Output CSV   : {OUTPUT_CSV_PATH} (UNCHANGED -- not overwritten)")
    step(f"Combined CSV : {OUTPUT_COMBINED_CSV_PATH}")
    step(f"Embed model  : {EMBED_MODEL_NAME}")
    step("Scope        : MEASUREMENT ONLY -- does not modify production "
         "SIMILARITY_THRESHOLD.")

    # ---- Heavy-dep imports wrapped into clean CalibrationErrors. ----
    try:
        import faiss  # noqa: F401
    except Exception as exc:
        raise CalibrationError(f"Could not import faiss: {exc}")

    try:
        from sentence_transformers import SentenceTransformer
    except Exception as exc:
        raise CalibrationError(
            f"Could not import sentence_transformers: {exc}"
        )

    # Import the REAL retrieval function -- do NOT reimplement it.
    try:
        from src.rag.suggest_resolution import retrieve_similar_tickets
    except Exception:
        # Fallback: allow running when 'src' isn't importable as a package by
        # adding PROJECT_ROOT to sys.path, then retry.
        if PROJECT_ROOT not in sys.path:
            sys.path.insert(0, PROJECT_ROOT)
        try:
            from src.rag.suggest_resolution import retrieve_similar_tickets
        except Exception as exc:
            raise CalibrationError(
                "Could not import retrieve_similar_tickets from "
                "src.rag.suggest_resolution.\n"
                f"    Underlying error: {exc}\n"
                "Run from the project root, e.g.:\n"
                "    python -m src.experiments.calibrate_rag_similarity_threshold"
            )

    # ---- Load index + metadata (with sync check). ----
    banner("LOADING FAISS INDEX + METADATA")
    index, metadata = load_index_and_metadata(faiss)
    ok(f"Loaded index with ntotal = {index.ntotal}")
    ok(f"Loaded metadata entries = {len(metadata)} (in sync)")

    # ---- Load in-domain calibration set + detect fields. ----
    banner("LOADING IN-DOMAIN CALIBRATION SET")
    (calibration, category_field, has_title, has_description,
     single_text_field, text_field_desc) = load_calibration_set()
    ok(f"Loaded in-domain calibration tickets = {len(calibration)}")
    ok(f"Detected ground-truth category field: '{category_field}'")
    ok(f"Detected ticket text source: {text_field_desc}")
    if len(calibration) != 175:
        warn(f"Expected 175 in-domain calibration tickets, found "
             f"{len(calibration)} -- proceeding anyway.")

    # ---- Load OOD calibration set. ----
    banner("LOADING OOD CALIBRATION SET")
    ood_tickets = load_ood_calibration_set()
    ok(f"Loaded OOD calibration tickets = {len(ood_tickets)}")
    if len(ood_tickets) != 45:
        warn(f"Expected ~45 OOD tickets, found {len(ood_tickets)} "
             "-- proceeding anyway.")

    # ---- Load embedding model. ----
    banner("LOADING EMBEDDING MODEL")
    step(f"Loading {EMBED_MODEL_NAME} (first run may download weights) ...")
    model = SentenceTransformer(EMBED_MODEL_NAME)
    ok("Embedding model ready.")

    # ---- IN-DOMAIN retrieval pass (top-1). ----
    # We collect per-ticket:
    #   top_similarity, is_relevant_match (category-agreement PROXY, the
    #   ORIGINAL ground-truth signal fed into the cliff-edge), and
    #   is_exact_source_match (retrieved.id == calibration ticket.id -- the
    #   strictly-stronger DIAGNOSTIC-ONLY signal), plus a self-retrieval flag.
    banner("IN-DOMAIN RETRIEVAL PASS (top-1 per calibration ticket)")
    per_ticket = []
    skipped_empty = 0
    self_retrieval_hits = 0
    self_retrieval_checkable = 0

    for i, entry in enumerate(calibration):
        query_text = build_query_text(
            entry, has_title, has_description, single_text_field
        )
        if not query_text:
            skipped_empty += 1
            warn(f"In-domain ticket #{i} has empty query text -- skipping.")
            continue

        gt_category = entry.get(category_field)
        own_id = entry.get("id", None)

        retrieved = retrieve_similar_tickets(
            query_text, model, index, metadata, faiss, top_k=1
        )
        if not retrieved:
            warn(f"In-domain ticket #{i} returned no retrieval results "
                 "-- skipping.")
            continue

        top = retrieved[0]
        top_similarity = float(top["similarity"])
        retrieved_category = top.get("category", "")
        retrieved_id = top.get("id", None)

        # ORIGINAL relevance proxy: retrieved category == ground-truth category.
        is_relevant_match = (retrieved_category == gt_category)

        # STRONGER diagnostic signal: retrieved id == this ticket's own id.
        # Only computable when both ids are present.
        exact_checkable = (own_id is not None and retrieved_id is not None)
        is_exact_source_match = bool(
            exact_checkable and retrieved_id == own_id
        )

        # Self-retrieval contamination: same id-equality condition; tracked
        # separately for a headline caveat.
        if exact_checkable:
            self_retrieval_checkable += 1
            if retrieved_id == own_id:
                self_retrieval_hits += 1

        per_ticket.append({
            "index": i,
            "own_id": own_id,
            "top_similarity": top_similarity,
            "retrieved_category": retrieved_category,
            "retrieved_id": retrieved_id,
            "gt_category": gt_category,
            "is_relevant_match": is_relevant_match,
            "is_exact_source_match": is_exact_source_match,
            "exact_checkable": exact_checkable,
        })

    n_used = len(per_ticket)
    if n_used == 0:
        raise CalibrationError(
            "No usable in-domain calibration tickets after retrieval "
            "-- cannot calibrate."
        )
    ok(f"Retrieved top-1 for {n_used} in-domain tickets "
       f"({skipped_empty} skipped for empty text).")
    n_relevant_total = sum(1 for r in per_ticket if r["is_relevant_match"])
    step(f"Ground-truth relevant top-1 matches (category proxy) overall: "
         f"{n_relevant_total}/{n_used}")

    # Self-retrieval rate (item 4) -- computed here, reported loudly later.
    if self_retrieval_checkable > 0:
        self_retrieval_rate = self_retrieval_hits / self_retrieval_checkable
    else:
        self_retrieval_rate = None
    n_exact_total = sum(1 for r in per_ticket if r["is_exact_source_match"])
    step(f"Exact-source top-1 matches overall (diagnostic): "
         f"{n_exact_total}/{n_used}")

    # ---- OOD retrieval pass (top-1). ----
    banner("OOD RETRIEVAL PASS (top-1 per OOD ticket)")
    ood_per_ticket = []
    ood_skipped_empty = 0
    for i, entry in enumerate(ood_tickets):
        query_text = str(entry.get("text", "") or "").strip()
        if not query_text:
            ood_skipped_empty += 1
            warn(f"OOD ticket #{i} (id={entry.get('id')!r}) has empty text "
                 "-- skipping.")
            continue

        retrieved = retrieve_similar_tickets(
            query_text, model, index, metadata, faiss, top_k=1
        )
        if not retrieved:
            warn(f"OOD ticket #{i} returned no retrieval results -- skipping.")
            continue

        top = retrieved[0]
        top_similarity = float(top["similarity"])
        ood_per_ticket.append({
            "id": entry.get("id"),
            "top_similarity": top_similarity,
            "retrieved_category": top.get("category", ""),
            "retrieved_id": top.get("id", None),
        })

    n_ood_used = len(ood_per_ticket)
    if n_ood_used == 0:
        raise CalibrationError(
            "No usable OOD calibration tickets after retrieval "
            "-- cannot compute OOD signal."
        )
    ok(f"Retrieved top-1 for {n_ood_used} OOD tickets "
       f"({ood_skipped_empty} skipped for empty text).")

    # ---- Threshold sweep -> combined + diagnostic confusion matrices. ----
    # Perspective: "proceed" is the POSITIVE action; proceeding on something we
    # shouldn't have is the costly error.
    #   In-domain (category proxy):
    #     TP: proceed  & is_relevant_match
    #     FP: proceed  & NOT is_relevant_match
    #     FN: escalate & is_relevant_match
    #     TN: escalate & NOT is_relevant_match
    #   OOD (correct behavior is ALWAYS escalate):
    #     proceed  -> false positive (leakage)
    #     escalate -> true negative
    #   COMBINED:
    #     combined_tp = in_domain_tp
    #     combined_fp = in_domain_fp + ood_proceed
    #     combined_fn = in_domain_fn
    #     combined_tn = in_domain_tn + ood_escalate
    #   EXACT-SOURCE diagnostic (in-domain only; NOT fed to cliff-edge):
    #     same shape as the in-domain category proxy but using
    #     is_exact_source_match.
    banner("THRESHOLD SWEEP (combined = in-domain category proxy + OOD)")
    rows_by_threshold = {}        # combined (drives cliff-edge)
    exact_rows_by_threshold = {}  # diagnostic-only, exact-source-match

    for thr in SIMILARITY_THRESHOLDS:
        # In-domain, category proxy.
        id_tp = id_fp = id_fn = id_tn = 0
        # In-domain, exact-source diagnostic.
        ex_tp = ex_fp = ex_fn = ex_tn = 0
        for r in per_ticket:
            proceed = r["top_similarity"] >= thr

            relevant = r["is_relevant_match"]
            if proceed and relevant:
                id_tp += 1
            elif proceed and not relevant:
                id_fp += 1
            elif (not proceed) and relevant:
                id_fn += 1
            else:
                id_tn += 1

            exact = r["is_exact_source_match"]
            if proceed and exact:
                ex_tp += 1
            elif proceed and not exact:
                ex_fp += 1
            elif (not proceed) and exact:
                ex_fn += 1
            else:
                ex_tn += 1

        # OOD split.
        ood_proceed = sum(1 for r in ood_per_ticket
                          if r["top_similarity"] >= thr)
        ood_escalate = n_ood_used - ood_proceed
        ood_leakage_rate = (ood_proceed / n_ood_used) if n_ood_used else None

        # Combined confusion matrix.
        combined_tp = id_tp
        combined_fp = id_fp + ood_proceed
        combined_fn = id_fn
        combined_tn = id_tn + ood_escalate

        precision = compute_precision(combined_tp, combined_fp)
        recall = compute_recall(combined_tp, combined_fn)
        f1 = compute_f1(precision, recall)

        rows_by_threshold[thr] = {
            "threshold": thr,
            # in-domain proceed/escalate counts (category proxy view)
            "n_in_domain_proceed": id_tp + id_fp,
            "n_in_domain_escalate": id_fn + id_tn,
            # combined confusion matrix
            "tp": combined_tp,
            "fp_combined": combined_fp,
            "fn": combined_fn,
            "tn_combined": combined_tn,
            # split-out FP/TN provenance
            "fp_in_domain": id_fp,
            "fp_ood": ood_proceed,
            "tn_in_domain": id_tn,
            "tn_ood": ood_escalate,
            # OOD leakage
            "ood_leakage_rate": ood_leakage_rate,
            # combined metrics (drive cliff-edge)
            "precision": precision,       # keyed 'precision' for find_cliff_edge
            "recall": recall,
            "f1": f1,
        }

        # Exact-source diagnostic metrics.
        ex_precision = compute_precision(ex_tp, ex_fp)
        ex_recall = compute_recall(ex_tp, ex_fn)
        ex_f1 = compute_f1(ex_precision, ex_recall)
        exact_rows_by_threshold[thr] = {
            "threshold": thr,
            "tp": ex_tp, "fp": ex_fp, "fn": ex_fn, "tn": ex_tn,
            "precision": ex_precision,
            "recall": ex_recall,
            "f1": ex_f1,
        }

    # ---- Print the COMBINED per-threshold table. ----
    banner("COMBINED METRIC TABLE (in-domain category proxy + OOD)")
    print("Ground truth for the COMBINED precision/recall/f1 below is the "
          "IN-DOMAIN category-agreement proxy\n(is_relevant_match: \"retrieved "
          "category matches ground-truth category\") for the positive class, "
          "PLUS\nOOD tickets whose correct action is always 'escalate'. This "
          "is what feeds the cliff-edge.\n")
    header = (f"{'thr':>6} | {'id_proc':>7} {'id_esc':>6} | "
              f"{'tp':>4} {'fp':>4} {'fn':>4} {'tn':>4} | "
              f"{'fp_id':>5} {'fp_ood':>6} | {'leak':>6} | "
              f"{'prec':>7} {'recall':>7} {'f1':>7}")
    print(header)
    print("-" * len(header))
    for thr in SIMILARITY_THRESHOLDS:
        row = rows_by_threshold[thr]
        print(f"{thr:>6.2f} | {row['n_in_domain_proceed']:>7} "
              f"{row['n_in_domain_escalate']:>6} | "
              f"{row['tp']:>4} {row['fp_combined']:>4} {row['fn']:>4} "
              f"{row['tn_combined']:>4} | "
              f"{row['fp_in_domain']:>5} {row['fp_ood']:>6} | "
              f"{fmt(row['ood_leakage_rate'], 3):>6} | "
              f"{fmt(row['precision']):>7} {fmt(row['recall']):>7} "
              f"{fmt(row['f1']):>7}")

    # ---- Print the DIAGNOSTIC exact-source table side-by-side. ----
    banner("DIAGNOSTIC METRIC TABLE (in-domain EXACT-SOURCE match only)")
    print("This uses is_exact_source_match (retrieved.id == calibration "
          "ticket.id) -- a strictly STRONGER,\nunambiguous relevance signal. "
          "It is reported for comparison ONLY and is NOT fed into the "
          "cliff-edge\nor the recommended threshold (switching the primary "
          "metric is a methodology change needing a human decision).\n")
    ex_header = (f"{'thr':>6} | {'tp':>4} {'fp':>4} {'fn':>4} {'tn':>4} | "
                 f"{'prec':>7} {'recall':>7} {'f1':>7}")
    print(ex_header)
    print("-" * len(ex_header))
    for thr in SIMILARITY_THRESHOLDS:
        er = exact_rows_by_threshold[thr]
        print(f"{thr:>6.2f} | {er['tp']:>4} {er['fp']:>4} {er['fn']:>4} "
              f"{er['tn']:>4} | "
              f"{fmt(er['precision']):>7} {fmt(er['recall']):>7} "
              f"{fmt(er['f1']):>7}")

    # ---- Write COMBINED CSV (original CSV left untouched). ----
    banner("WRITING COMBINED CSV")
    fieldnames = [
        "threshold",
        "n_in_domain_proceed", "n_in_domain_escalate",
        "tp", "fp_combined", "fn", "tn_combined",
        "fp_in_domain", "fp_ood", "tn_in_domain", "tn_ood",
        "ood_leakage_rate",
        "precision_combined", "recall_combined", "f1_combined",
        "precision_exact_source", "recall_exact_source", "f1_exact_source",
    ]
    with open(OUTPUT_COMBINED_CSV_PATH, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for thr in SIMILARITY_THRESHOLDS:
            row = rows_by_threshold[thr]
            er = exact_rows_by_threshold[thr]
            writer.writerow({
                "threshold": row["threshold"],
                "n_in_domain_proceed": row["n_in_domain_proceed"],
                "n_in_domain_escalate": row["n_in_domain_escalate"],
                "tp": row["tp"],
                "fp_combined": row["fp_combined"],
                "fn": row["fn"],
                "tn_combined": row["tn_combined"],
                "fp_in_domain": row["fp_in_domain"],
                "fp_ood": row["fp_ood"],
                "tn_in_domain": row["tn_in_domain"],
                "tn_ood": row["tn_ood"],
                "ood_leakage_rate": csv_cell(row["ood_leakage_rate"]),
                "precision_combined": csv_cell(row["precision"]),
                "recall_combined": csv_cell(row["recall"]),
                "f1_combined": csv_cell(row["f1"]),
                "precision_exact_source": csv_cell(er["precision"]),
                "recall_exact_source": csv_cell(er["recall"]),
                "f1_exact_source": csv_cell(er["f1"]),
            })
    ok(f"Wrote {OUTPUT_COMBINED_CSV_PATH}")
    step(f"(Original {os.path.basename(OUTPUT_CSV_PATH)} intentionally left "
         "untouched.)")

    # ---- Find cliff-edge on the COMBINED metric + recommendation. ----
    banner("RECOMMENDATION (from COMBINED metric)")
    cliff = find_cliff_edge(rows_by_threshold)

    if cliff is not None:
        recommended = cliff
        basis = "cliff-edge (lowest T with combined precision == 1.0)"
    else:
        # Fallback: best F1 (ignoring None F1s), then best precision.
        best_thr = None
        best_f1 = None
        for thr in SIMILARITY_THRESHOLDS:
            f1 = rows_by_threshold[thr]["f1"]
            if f1 is None:
                continue
            if best_f1 is None or f1 > best_f1:
                best_f1 = f1
                best_thr = thr
        if best_thr is not None:
            recommended = best_thr
            basis = (f"FALLBACK: best combined F1 ({best_f1:.4f}) -- no clean "
                     "cliff-edge found")
        else:
            best_prec = None
            for thr in SIMILARITY_THRESHOLDS:
                p = rows_by_threshold[thr]["precision"]
                if p is None:
                    continue
                if best_prec is None or p > best_prec:
                    best_prec = p
                    best_thr = thr
            if best_thr is None:
                raise CalibrationError(
                    "Could not derive any recommendation: precision and F1 "
                    "undefined at every threshold (no ticket ever proceeds?)."
                )
            recommended = best_thr
            basis = (f"FALLBACK: best combined precision ({best_prec:.4f}) -- "
                     "no cliff-edge and no defined F1")

    if cliff is not None:
        ok(f"Cliff-edge found at T = {cliff:.2f}")
    else:
        warn("No clean cliff-edge (combined precision never held at exactly "
             "1.0 down a contiguous run from the top).")

    # Non-monotonic post-hoc diagnostic (item 7).
    report_non_monotonic_cliff(rows_by_threshold, cliff)

    rec_row = rows_by_threshold[recommended]
    print()
    print(f"Recommended threshold : {recommended:.2f}")
    print(f"Basis                 : {basis}")
    print(f"  at T={recommended:.2f}: precision={fmt(rec_row['precision'])}, "
          f"recall={fmt(rec_row['recall'])}, f1={fmt(rec_row['f1'])}, "
          f"combined_fp={rec_row['fp_combined']} "
          f"(in-domain {rec_row['fp_in_domain']} + OOD {rec_row['fp_ood']}), "
          f"OOD leakage={fmt(rec_row['ood_leakage_rate'], 3)}")

    # ---- FINAL SUMMARY BLOCK (item 9). ----
    banner("FINAL SUMMARY")

    # OOD similarity distribution.
    ood_sims = [r["top_similarity"] for r in ood_per_ticket]
    ood_min = min(ood_sims)
    ood_max = max(ood_sims)
    ood_mean = statistics.mean(ood_sims)
    ood_median = statistics.median(ood_sims)
    print("OOD top-1 similarity distribution (across "
          f"{n_ood_used} OOD tickets):")
    print(f"    min={ood_min:.4f}  max={ood_max:.4f}  "
          f"mean={ood_mean:.4f}  median={ood_median:.4f}")

    # Self-retrieval caveat (item 4) -- loud, not buried.
    print()
    print("!" * 78)
    print("SELF-RETRIEVAL CONTAMINATION CAVEAT (read before trusting the "
          "in-domain side):")
    if self_retrieval_rate is None:
        print("    Self-retrieval rate: NOT COMPUTABLE -- calibration records "
              "and/or metadata\n    lacked an 'id' to compare. The in-domain "
              "metric cannot be checked for near-self-match\n    contamination; "
              "treat the in-domain numbers with corresponding caution.")
    else:
        print(f"    Self-retrieval rate: {self_retrieval_rate:.1%} "
              f"({self_retrieval_hits}/{self_retrieval_checkable} in-domain "
              "tickets whose\n    top-1 retrieval IS their own source ticket "
              "in the index).")
        if self_retrieval_rate >= 0.5:
            print("    >>> HIGH: a large fraction of in-domain 'matches' are "
                  "the ticket retrieving ITSELF.\n    The in-domain "
                  "precision/similarity numbers likely reflect near-self-match, "
                  "NOT genuine\n    generalization. The cliff-edge is driven "
                  "substantially by self-retrieval -- do NOT adopt the\n    "
                  "in-domain side uncritically. The OOD leakage rate is the "
                  "more trustworthy signal here.")
        elif self_retrieval_rate > 0.0:
            print("    >>> PARTIAL: some in-domain matches are self-retrievals; "
                  "the in-domain precision is\n    inflated to that degree. "
                  "Weight the OOD signal accordingly.")
        else:
            print("    >>> NONE: no in-domain ticket retrieved its own source "
                  "row; the in-domain metric is\n    not contaminated by "
                  "self-retrieval.")
    print("    NOTE: self-matches were NOT excluded from the metric -- that "
          "would change the sample and\n    needs a human decision, not a "
          "silent script choice. Reported as-is.")
    print("!" * 78)

    # Recommendation vs prior values.
    print()
    print(f"Recommended (combined sweep): {recommended:.2f}")
    print(f"    vs. provisional BGE 0.65 (current live)")
    print(f"    vs. flawed cliff 0.85 (low-N artifact, NOT adopted)")
    print(f"    vs. old MiniLM-era 0.35")

    # OOD leakage at the key reference thresholds.
    def leakage_at(value):
        for thr in SIMILARITY_THRESHOLDS:
            if abs(thr - value) < 1e-9:
                return rows_by_threshold[thr]["ood_leakage_rate"]
        return None

    leak_rec = rows_by_threshold[recommended]["ood_leakage_rate"]
    leak_065 = leakage_at(0.65)
    leak_035 = leakage_at(0.35)

    print()
    print("OOD leakage rate (fraction of OOD tickets that WRONGLY proceed):")
    print(f"    at recommended T={recommended:.2f}: {fmt(leak_rec, 3)}")
    if leak_065 is not None:
        print(f"    at provisional  T=0.65: {fmt(leak_065, 3)}")
    else:
        print("    at provisional  T=0.65: not in swept grid.")
    if leak_035 is not None:
        print(f"    at old MiniLM   T=0.35: {fmt(leak_035, 3)}")
    else:
        print("    at old MiniLM   T=0.35: not in swept grid.")

    # Reminder of the two in-domain metrics and which is authoritative.
    print()
    print("In-domain relevance metrics (two, side by side above):")
    print("    PRIMARY (feeds cliff-edge): category-agreement proxy "
          "(is_relevant_match).")
    print("    DIAGNOSTIC ONLY          : exact-source-match "
          "(is_exact_source_match) -- stronger, not adopted.")

    banner("DONE (measurement only -- production threshold unchanged)")


def main():
    try:
        run()
    except CalibrationError as exc:
        print()
        print(RULE)
        print("ERROR (expected / actionable)")
        print(RULE)
        print(str(exc))
        sys.exit(1)
    except Exception:
        print()
        print(RULE)
        print("UNEXPECTED ERROR -- full traceback follows")
        print(RULE)
        traceback.print_exc()
        sys.exit(2)


if __name__ == "__main__":
    main()
