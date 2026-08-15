# src/experiments/run_ablation_study.py
"""
Ablation study for the AI-Powered Intelligent Ticket Routing & Resolution Agent.

Quantifies the value of two safety-net thresholds by disabling each one
individually and measuring the downstream effect:

  1. Cascade confidence threshold (0.50) -- escalates classification from
     Tier-1 (TF-IDF + LogisticRegression) to Tier-2 (MiniLM embeddings +
     classifier) when Tier-1 confidence < 0.50.
  2. RAG similarity threshold (0.35) -- escalates a ticket to a human
     (skips Gemini) when the top retrieval similarity < 0.35.

Modes (--mode):
  baseline     -- Real thresholds. 45-ticket classification accuracy AND the
                  9-ticket adversarial real-escalation decision (< 0.35).
  no-cascade   -- run_cascade(threshold=0.0) => Tier-1 raw preds for every
                  ticket, evaluated on the 45-ticket benchmark.
  no-rag       -- Real retrieval on the 9-ticket adversarial set, but the
                  escalation gate is pretend-threshold 0.0 => a Gemini call
                  WOULD be attempted whenever retrieval returns anything.
                  Gemini is NEVER actually called.

This script performs pure inference on fixed ticket sets. It never calls
Gemini (call_gemini / build_llm_prompt are never imported or invoked).
"""

import os
import sys
import csv
import json
import random
import argparse
import traceback

import numpy as np


# --------------------------------------------------------------------------
# Paths / project layout
# --------------------------------------------------------------------------
# Project root is TWO directories up from src/experiments/ (project convention).
THIS_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(THIS_DIR, os.pardir, os.pardir))

SRC_DIR = os.path.join(PROJECT_ROOT, "src")
CLASSIFICATION_DIR = os.path.join(SRC_DIR, "classification")
RAG_DIR = os.path.join(SRC_DIR, "rag")

DATA_DIR = os.path.join(PROJECT_ROOT, "data")
MODELS_DIR = os.path.join(PROJECT_ROOT, "models")

EXPANDED_JSON_PATH = os.path.join(DATA_DIR, "novel_tickets_expanded.json")
ADVERSARIAL_JSON_PATH = os.path.join(DATA_DIR, "adversarial_escalation_tickets.json")
SYNTHETIC_CSV_PATH = os.path.join(DATA_DIR, "synthetic_tickets.csv")
FAISS_INDEX_PATH = os.path.join(DATA_DIR, "ticket_index.faiss")
METADATA_JSON_PATH = os.path.join(DATA_DIR, "ticket_metadata.json")
TIER2_MODEL_PATH = os.path.join(MODELS_DIR, "ticket_classifier.joblib")

# Applied live thresholds (streamlit_app.py values).
CASCADE_CONFIDENCE_THRESHOLD = 0.50
# RAG SIMILARITY_THRESHOLD (0.35) is imported from suggest_resolution.

EMBED_MODEL_NAME = "all-MiniLM-L6-v2"

VALID_MODES = ("baseline", "no-cascade", "no-rag")


# --------------------------------------------------------------------------
# Console helpers (project conventions)
# --------------------------------------------------------------------------
def _banner(text):
    rule = "=" * 70
    print(rule)
    print(text)
    print(rule)


def _fatal(message):
    """Clean, actionable error for expected failures. No raw traceback."""
    print("")
    print("ERROR: " + str(message))
    sys.exit(1)


# --------------------------------------------------------------------------
# Imports of the REAL project functions, with the standard sys.path fallback
# pattern used across this project (try package import, then inject the
# relevant src/* dirs onto sys.path and retry).
# --------------------------------------------------------------------------
def _import_project_functions():
    # Ensure project root is importable for the package-style import.
    if PROJECT_ROOT not in sys.path:
        sys.path.insert(0, PROJECT_ROOT)

    train_tier1 = get_tier1_confidence = run_cascade = load_expanded_set = None
    retrieve_similar_tickets = SIMILARITY_THRESHOLD = None

    # ---- classification ----
    try:
        from src.classification.train_cascade import (
            train_tier1,
            get_tier1_confidence,
            run_cascade,
            load_expanded_set,
        )
    except Exception:
        if CLASSIFICATION_DIR not in sys.path:
            sys.path.insert(0, CLASSIFICATION_DIR)
        try:
            from train_cascade import (  # type: ignore
                train_tier1,
                get_tier1_confidence,
                run_cascade,
                load_expanded_set,
            )
        except Exception as exc:
            _fatal(
                "Could not import from src/classification/train_cascade.py "
                "(train_tier1, get_tier1_confidence, run_cascade, "
                "load_expanded_set). Underlying error: " + repr(exc)
            )

    # ---- rag ----
    try:
        from src.rag.suggest_resolution import (
            retrieve_similar_tickets,
            SIMILARITY_THRESHOLD,
        )
    except Exception:
        if RAG_DIR not in sys.path:
            sys.path.insert(0, RAG_DIR)
        try:
            from suggest_resolution import (  # type: ignore
                retrieve_similar_tickets,
                SIMILARITY_THRESHOLD,
            )
        except Exception as exc:
            _fatal(
                "Could not import from src/rag/suggest_resolution.py "
                "(retrieve_similar_tickets, SIMILARITY_THRESHOLD). "
                "Underlying error: " + repr(exc)
            )

    return {
        "train_tier1": train_tier1,
        "get_tier1_confidence": get_tier1_confidence,
        "run_cascade": run_cascade,
        "load_expanded_set": load_expanded_set,
        "retrieve_similar_tickets": retrieve_similar_tickets,
        "SIMILARITY_THRESHOLD": SIMILARITY_THRESHOLD,
    }


# --------------------------------------------------------------------------
# Third-party heavy imports (isolated so failures are actionable)
# --------------------------------------------------------------------------
def _import_third_party():
    try:
        import joblib
    except Exception as exc:
        _fatal("Failed to import joblib. Is the venv activated? " + repr(exc))
    try:
        import faiss
    except Exception as exc:
        _fatal("Failed to import faiss. Is the venv activated? " + repr(exc))
    try:
        import pandas as pd
    except Exception as exc:
        _fatal("Failed to import pandas. Is the venv activated? " + repr(exc))
    try:
        from sentence_transformers import SentenceTransformer
    except Exception as exc:
        _fatal(
            "Failed to import sentence_transformers. Is the venv activated? "
            + repr(exc)
        )
    return joblib, faiss, pd, SentenceTransformer


# --------------------------------------------------------------------------
# Resource loading (mirrors streamlit_app.py's load_resources())
# --------------------------------------------------------------------------
def _require_file(path, description):
    if not os.path.isfile(path):
        _fatal(
            "Missing required file for {desc}:\n  {path}\n"
            "Run the project's data/model build steps first.".format(
                desc=description, path=path
            )
        )


def load_tier1(funcs, pd):
    """Train Tier-1 fresh from data/synthetic_tickets.csv (project convention)."""
    _require_file(SYNTHETIC_CSV_PATH, "Tier-1 training data (synthetic_tickets.csv)")
    try:
        df = pd.read_csv(SYNTHETIC_CSV_PATH)
    except Exception as exc:
        _fatal("Failed to read synthetic_tickets.csv: " + repr(exc))

    for col in ("title", "description", "category"):
        if col not in df.columns:
            _fatal(
                "synthetic_tickets.csv is missing required column '{c}'. "
                "Expected columns: title, description, category.".format(c=col)
            )

    tier1_texts = (
        df["title"].fillna("").astype(str)
        + " "
        + df["description"].fillna("").astype(str)
    ).tolist()
    tier1_labels = df["category"].astype(str).tolist()

    if not tier1_texts:
        _fatal("synthetic_tickets.csv contained no rows to train Tier-1.")

    tier1_vectorizer, tier1_classifier = funcs["train_tier1"](tier1_texts, tier1_labels)
    return tier1_vectorizer, tier1_classifier


def load_tier2(joblib):
    _require_file(TIER2_MODEL_PATH, "Tier-2 classifier (ticket_classifier.joblib)")
    try:
        return joblib.load(TIER2_MODEL_PATH)
    except Exception as exc:
        _fatal("Failed to load Tier-2 classifier joblib: " + repr(exc))


def load_embedder(SentenceTransformer):
    try:
        return SentenceTransformer(EMBED_MODEL_NAME)
    except Exception as exc:
        _fatal(
            "Failed to load SentenceTransformer('{m}'): {e}".format(
                m=EMBED_MODEL_NAME, e=repr(exc)
            )
        )


def load_faiss_and_metadata(faiss):
    _require_file(FAISS_INDEX_PATH, "FAISS index (ticket_index.faiss)")
    _require_file(METADATA_JSON_PATH, "ticket metadata (ticket_metadata.json)")

    try:
        index = faiss.read_index(FAISS_INDEX_PATH)
    except Exception as exc:
        _fatal("Failed to read FAISS index: " + repr(exc))

    try:
        with open(METADATA_JSON_PATH, "r", encoding="utf-8") as fh:
            metadata = json.load(fh)
    except Exception as exc:
        _fatal("Failed to read ticket_metadata.json: " + repr(exc))

    if not isinstance(metadata, list):
        _fatal("ticket_metadata.json must be a JSON list.")

    if index.ntotal != len(metadata):
        _fatal(
            "FAISS/metadata mismatch: index.ntotal={n} but len(metadata)={m}. "
            "Rebuild the index and metadata together.".format(
                n=index.ntotal, m=len(metadata)
            )
        )

    return index, metadata


# --------------------------------------------------------------------------
# Adversarial set loading (schema validated locally; the 45-ticket set uses
# load_expanded_set from train_cascade.py)
# --------------------------------------------------------------------------
def load_adversarial_set():
    _require_file(
        ADVERSARIAL_JSON_PATH, "adversarial set (adversarial_escalation_tickets.json)"
    )
    try:
        with open(ADVERSARIAL_JSON_PATH, "r", encoding="utf-8") as fh:
            data = json.load(fh)
    except Exception as exc:
        _fatal("Failed to read adversarial_escalation_tickets.json: " + repr(exc))

    if not isinstance(data, list):
        _fatal("adversarial_escalation_tickets.json must be a JSON list.")
    if len(data) != 9:
        _fatal(
            "adversarial_escalation_tickets.json must contain EXACTLY 9 records; "
            "found {n}.".format(n=len(data))
        )

    required_keys = (
        "id",
        "category_type",
        "text",
        "expected_escalate",
        "expected_trigger",
        "note",
    )
    for i, rec in enumerate(data):
        if not isinstance(rec, dict):
            _fatal("adversarial record #{i} is not an object.".format(i=i))
        for k in required_keys:
            if k not in rec:
                _fatal(
                    "adversarial record #{i} (id={id!r}) is missing key "
                    "'{k}'.".format(i=i, id=rec.get("id"), k=k)
                )
        if not isinstance(rec["expected_escalate"], bool):
            _fatal(
                "adversarial record #{i} (id={id!r}): 'expected_escalate' "
                "must be a bool.".format(i=i, id=rec.get("id"))
            )
        if not isinstance(rec["text"], str) or not rec["text"].strip():
            _fatal(
                "adversarial record #{i} (id={id!r}): 'text' must be a "
                "non-empty string.".format(i=i, id=rec.get("id"))
            )

    return data


# --------------------------------------------------------------------------
# CSV writer
# --------------------------------------------------------------------------
def write_csv(mode, fieldnames, rows):
    out_path = os.path.join(DATA_DIR, "ablation_{mode}_results.csv".format(mode=mode))
    try:
        os.makedirs(DATA_DIR, exist_ok=True)
        with open(out_path, "w", newline="", encoding="utf-8") as fh:
            writer = csv.DictWriter(fh, fieldnames=fieldnames)
            writer.writeheader()
            for row in rows:
                writer.writerow(row)
    except Exception as exc:
        _fatal("Failed to write CSV {p}: {e}".format(p=out_path, e=repr(exc)))
    return out_path


# --------------------------------------------------------------------------
# Classification evaluation (45-ticket benchmark) via run_cascade
# --------------------------------------------------------------------------
def evaluate_classification(funcs, threshold, mode_label,
                            tier1_vec, tier1_clf, tier2_clf, embedder):
    """
    Runs run_cascade on the 45-ticket expanded benchmark at the given
    threshold and returns (rows, accuracy, n_correct, n_total, n_tier2).

    threshold=0.50 -> baseline behaviour (real cascade).
    threshold=0.0  -> no-cascade (escalate_idx always empty => Tier-1 raw).
    """
    records = funcs["load_expanded_set"](EXPANDED_JSON_PATH)

    texts = [r["text"] for r in records]
    expected = [r["expected"] for r in records]

    try:
        result = funcs["run_cascade"](
            texts,
            tier1_vec,
            tier1_clf,
            tier2_clf,
            embedder,
            threshold,
        )
    except Exception as exc:
        _fatal(
            "run_cascade failed for mode '{m}' (threshold={t}): {e}".format(
                m=mode_label, t=threshold, e=repr(exc)
            )
        )

    preds = result["preds"]
    tiers = result["tiers"]
    tier1_conf = result["tier1_conf"]
    n_tier2 = result.get("n_tier2", 0)

    if not (len(preds) == len(tiers) == len(tier1_conf) == len(texts)):
        _fatal(
            "run_cascade returned mismatched lengths for mode '{m}'.".format(
                m=mode_label
            )
        )

    rows = []
    n_correct = 0
    for i in range(len(texts)):
        pred = str(preds[i])
        exp = str(expected[i])
        correct = (pred == exp)
        if correct:
            n_correct += 1
        rows.append(
            {
                "index": i,
                "text": texts[i],
                "expected": exp,
                "predicted": pred,
                "tier1_conf": round(float(tier1_conf[i]), 6),
                "tier_used": int(tiers[i]),
                "correct": bool(correct),
            }
        )

    n_total = len(texts)
    accuracy = (n_correct / n_total) if n_total else 0.0
    return rows, accuracy, n_correct, n_total, int(n_tier2)


# --------------------------------------------------------------------------
# RAG escalation evaluation (9-ticket adversarial set)
# --------------------------------------------------------------------------
def evaluate_rag_escalation(funcs, faiss, adversarial, model, index, metadata,
                            gate_threshold):
    """
    Runs REAL retrieval for each adversarial ticket and applies an escalation
    gate at `gate_threshold`.

      escalated        = (top_similarity < gate_threshold)
      would_call_gemini= (top_similarity >= gate_threshold)

    baseline uses gate_threshold = SIMILARITY_THRESHOLD (0.35).
    no-rag   uses gate_threshold = 0.0 (pretend), so nothing escalates as
             long as retrieval returns at least one hit.

    Gemini is NEVER called. Returns (rows, summary_dict).
    """
    retrieve = funcs["retrieve_similar_tickets"]

    rows = []
    n_escalated = 0
    n_would_call = 0
    n_correct_escalation = 0

    for rec in adversarial:
        text = rec["text"]
        expected_escalate = bool(rec["expected_escalate"])

        try:
            retrieved = retrieve(text, model, index, metadata, faiss, top_k=5)
        except Exception as exc:
            _fatal(
                "retrieve_similar_tickets failed for adversarial id={id!r}: "
                "{e}".format(id=rec.get("id"), e=repr(exc))
            )

        top_similarity = retrieved[0]["similarity"] if retrieved else 0.0
        top_similarity = float(top_similarity)

        escalated = (top_similarity < gate_threshold)
        would_call_gemini = not escalated  # (top_similarity >= gate_threshold)

        if escalated:
            n_escalated += 1
        if would_call_gemini:
            n_would_call += 1
        if escalated == expected_escalate:
            n_correct_escalation += 1

        rows.append(
            {
                "id": rec.get("id"),
                "category_type": rec.get("category_type"),
                "text": text,
                "expected_escalate": expected_escalate,
                "expected_trigger": rec.get("expected_trigger"),
                "top_similarity": round(top_similarity, 6),
                "gate_threshold": gate_threshold,
                "escalated": bool(escalated),
                "would_call_gemini": bool(would_call_gemini),
                "escalation_correct": bool(escalated == expected_escalate),
                "note": rec.get("note"),
            }
        )

    summary = {
        "n_total": len(adversarial),
        "n_escalated": n_escalated,
        "n_would_call_gemini": n_would_call,
        "n_correct_escalation": n_correct_escalation,
    }
    return rows, summary


# --------------------------------------------------------------------------
# Mode runners
# --------------------------------------------------------------------------
def run_baseline(funcs, joblib, faiss, pd, SentenceTransformer):
    _banner("MODE: baseline  (cascade=0.50, rag=0.35)")

    print("Loading resources...")
    tier1_vec, tier1_clf = load_tier1(funcs, pd)
    tier2_clf = load_tier2(joblib)
    embedder = load_embedder(SentenceTransformer)
    index, metadata = load_faiss_and_metadata(faiss)
    adversarial = load_adversarial_set()

    # --- classification on the 45-ticket benchmark (real cascade @ 0.50) ---
    print("Evaluating 45-ticket classification (cascade threshold=0.50)...")
    cls_rows, accuracy, n_correct, n_total, n_tier2 = evaluate_classification(
        funcs,
        CASCADE_CONFIDENCE_THRESHOLD,
        "baseline",
        tier1_vec,
        tier1_clf,
        tier2_clf,
        embedder,
    )

    # --- real escalation on the 9-ticket adversarial set (gate @ 0.35) ---
    print("Evaluating 9-ticket adversarial escalation (rag threshold=0.35)...")
    rag_rows, rag_summary = evaluate_rag_escalation(
        funcs,
        faiss,
        adversarial,
        embedder,
        index,
        metadata,
        gate_threshold=funcs["SIMILARITY_THRESHOLD"],
    )

    # Combine both row-sets into one CSV, tagged by section.
    combined_rows = []
    for r in cls_rows:
        combined_rows.append(
            {
                "section": "classification",
                "key": r["index"],
                "text": r["text"],
                "expected": r["expected"],
                "predicted": r["predicted"],
                "tier1_conf": r["tier1_conf"],
                "tier_used": r["tier_used"],
                "top_similarity": "",
                "escalated": "",
                "would_call_gemini": "",
                "correct": r["correct"],
            }
        )
    for r in rag_rows:
        combined_rows.append(
            {
                "section": "escalation",
                "key": r["id"],
                "text": r["text"],
                "expected": r["expected_escalate"],
                "predicted": "",
                "tier1_conf": "",
                "tier_used": "",
                "top_similarity": r["top_similarity"],
                "escalated": r["escalated"],
                "would_call_gemini": r["would_call_gemini"],
                "correct": r["escalation_correct"],
            }
        )

    fieldnames = [
        "section",
        "key",
        "text",
        "expected",
        "predicted",
        "tier1_conf",
        "tier_used",
        "top_similarity",
        "escalated",
        "would_call_gemini",
        "correct",
    ]
    out_path = write_csv("baseline", fieldnames, combined_rows)

    _banner("baseline RESULTS")
    print(
        "Classification (45-ticket): {c}/{t} correct  =>  accuracy {a:.2%}  "
        "(Tier-2 escalations: {e})".format(
            c=n_correct, t=n_total, a=accuracy, e=n_tier2
        )
    )
    print(
        "Adversarial escalation (9-ticket): {esc}/{tot} correctly escalated "
        "(gate < 0.35)".format(
            esc=rag_summary["n_correct_escalation"], tot=rag_summary["n_total"]
        )
    )
    print("CSV written: " + out_path)

    return {
        "mode": "baseline",
        "accuracy": accuracy,
        "n_correct": n_correct,
        "n_total": n_total,
        "n_tier2": n_tier2,
        "rag_summary": rag_summary,
        "csv": out_path,
    }


def run_no_cascade(funcs, joblib, faiss, pd, SentenceTransformer):
    _banner("MODE: no-cascade  (cascade DISABLED via threshold=0.0)")

    print("Loading resources...")
    tier1_vec, tier1_clf = load_tier1(funcs, pd)
    tier2_clf = load_tier2(joblib)  # loaded for signature parity; unused at 0.0
    embedder = load_embedder(SentenceTransformer)

    print(
        "Evaluating 45-ticket classification with run_cascade(threshold=0.0)\n"
        "  -> escalate_idx always empty -> Tier-1 raw prediction for every ticket."
    )
    cls_rows, accuracy, n_correct, n_total, n_tier2 = evaluate_classification(
        funcs,
        0.0,
        "no-cascade",
        tier1_vec,
        tier1_clf,
        tier2_clf,
        embedder,
    )

    if n_tier2 != 0:
        # Defensive: at threshold 0.0, nothing should escalate to Tier-2.
        print(
            "WARNING: expected 0 Tier-2 escalations at threshold=0.0 but "
            "run_cascade reported {n}. Check run_cascade's strict/loose "
            "comparison.".format(n=n_tier2)
        )

    fieldnames = [
        "index",
        "text",
        "expected",
        "predicted",
        "tier1_conf",
        "tier_used",
        "correct",
    ]
    out_path = write_csv("no-cascade", fieldnames, cls_rows)

    _banner("no-cascade RESULTS")
    print(
        "Classification (45-ticket, Tier-1 only): {c}/{t} correct  =>  "
        "accuracy {a:.2%}".format(c=n_correct, t=n_total, a=accuracy)
    )
    print("CSV written: " + out_path)

    return {
        "mode": "no-cascade",
        "accuracy": accuracy,
        "n_correct": n_correct,
        "n_total": n_total,
        "n_tier2": n_tier2,
        "csv": out_path,
    }


def run_no_rag(funcs, joblib, faiss, pd, SentenceTransformer):
    _banner("MODE: no-rag  (rag gate DISABLED via pretend threshold=0.0)")

    print("Loading resources...")
    embedder = load_embedder(SentenceTransformer)
    index, metadata = load_faiss_and_metadata(faiss)
    adversarial = load_adversarial_set()

    print(
        "Evaluating 9-ticket adversarial set with pretend gate threshold=0.0\n"
        "  -> would_call_gemini = (top_similarity >= 0.0) i.e. True whenever\n"
        "     retrieval returns >=1 hit. Gemini is NEVER actually called."
    )
    rag_rows, rag_summary = evaluate_rag_escalation(
        funcs,
        faiss,
        adversarial,
        embedder,
        index,
        metadata,
        gate_threshold=0.0,
    )

    fieldnames = [
        "id",
        "category_type",
        "text",
        "expected_escalate",
        "expected_trigger",
        "top_similarity",
        "gate_threshold",
        "escalated",
        "would_call_gemini",
        "escalation_correct",
        "note",
    ]
    out_path = write_csv("no-rag", fieldnames, rag_rows)

    _banner("no-rag RESULTS")
    print(
        "Adversarial (9-ticket): {w}/{t} would now attempt a resolution "
        "(Gemini call).".format(w=rag_summary["n_would_call_gemini"], t=rag_summary["n_total"])
    )
    print(
        "  Of those, {bad} were tickets that SHOULD have escalated to a "
        "human.".format(
            bad=sum(
                1
                for r in rag_rows
                if r["would_call_gemini"] and r["expected_escalate"]
            )
        )
    )
    print("CSV written: " + out_path)

    return {
        "mode": "no-rag",
        "rag_summary": rag_summary,
        "n_would_incorrectly_resolve": sum(
            1 for r in rag_rows if r["would_call_gemini"] and r["expected_escalate"]
        ),
        "csv": out_path,
    }


# --------------------------------------------------------------------------
# Summary comparison
# --------------------------------------------------------------------------
def print_comparison(result):
    _banner("ABLATION SUMMARY")

    mode = result["mode"]

    if mode == "baseline":
        print(
            "baseline classification accuracy (45-ticket): {a:.2%} "
            "({c}/{t})".format(
                a=result["accuracy"], c=result["n_correct"], t=result["n_total"]
            )
        )
        rs = result["rag_summary"]
        print(
            "baseline: {esc}/{t} adversarial tickets correctly escalated "
            "(gate < 0.35).".format(esc=rs["n_correct_escalation"], t=rs["n_total"])
        )
        print("")
        print(
            "Run --mode no-cascade to compare classification accuracy, and "
            "--mode no-rag to\ncompare escalation behaviour against these "
            "baseline numbers."
        )

    elif mode == "no-cascade":
        print(
            "no-cascade classification accuracy (Tier-1 only, 45-ticket): "
            "{a:.2%} ({c}/{t})".format(
                a=result["accuracy"], c=result["n_correct"], t=result["n_total"]
            )
        )
        print("")
        print(
            "Compare against the baseline run's classification accuracy. The "
            "drop (if any)\nis the measured value of the cascade confidence "
            "threshold (0.50)."
        )

    elif mode == "no-rag":
        rs = result["rag_summary"]
        print(
            "no-rag: {w}/{t} adversarial tickets would now incorrectly attempt "
            "a resolution".format(w=rs["n_would_call_gemini"], t=rs["n_total"])
        )
        print(
            "        (of which {bad} SHOULD have been escalated to a "
            "human).".format(bad=result["n_would_incorrectly_resolve"])
        )
        print("")
        print(
            "Compare against 'baseline: N/9 correctly escalated'. The "
            "difference is the\nmeasured value of the RAG similarity threshold "
            "(0.35)."
        )


# --------------------------------------------------------------------------
# CLI
# --------------------------------------------------------------------------
def parse_args(argv):
    parser = argparse.ArgumentParser(
        description=(
            "Ablation study for the ticket routing agent. Measures the value "
            "of the cascade confidence threshold (0.50) and the RAG similarity "
            "threshold (0.35) by disabling each individually."
        )
    )
    parser.add_argument(
        "--mode",
        required=True,
        choices=VALID_MODES,
        help="Which ablation to run: baseline | no-cascade | no-rag",
    )
    return parser.parse_args(argv)


def main(argv=None):
    random.seed(42)
    np.random.seed(42)

    args = parse_args(argv)
    mode = args.mode

    _banner("AI Ticket Agent -- Ablation Study")
    print("Project root : " + PROJECT_ROOT)
    print("Mode         : " + mode)

    funcs = _import_project_functions()
    joblib, faiss, pd, SentenceTransformer = _import_third_party()

    if mode == "baseline":
        result = run_baseline(funcs, joblib, faiss, pd, SentenceTransformer)
    elif mode == "no-cascade":
        result = run_no_cascade(funcs, joblib, faiss, pd, SentenceTransformer)
    elif mode == "no-rag":
        result = run_no_rag(funcs, joblib, faiss, pd, SentenceTransformer)
    else:
        _fatal("Unknown mode: " + repr(mode))  # unreachable due to choices=

    print_comparison(result)


if __name__ == "__main__":
    try:
        main()
    except SystemExit:
        # _fatal / argparse already handled the exit cleanly.
        raise
    except Exception:
        # Catch-all traceback dump for genuinely unexpected errors only.
        print("")
        print("=" * 70)
        print("UNEXPECTED ERROR -- full traceback follows:")
        print("=" * 70)
        traceback.print_exc()
        sys.exit(1)
