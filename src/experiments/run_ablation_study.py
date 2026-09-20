# src/experiments/run_ablation_study.py
"""
Ablation study for the AI-Powered Intelligent Ticket Routing & Resolution Agent.

Quantifies the value of two safety-net thresholds by disabling each one
individually and measuring the downstream effect:

  1. Cascade confidence threshold (0.50) -- escalates classification from
     Tier-1 (TF-IDF + LogisticRegression) to Tier-2 (MiniLM embeddings +
     classifier) when Tier-1 confidence < 0.50.
  2. RAG similarity threshold -- escalates a ticket to a human (skips
     Gemini) when the top retrieval similarity falls below it. The live
     value comes from src/agent/config.py.

Modes (--mode):
  baseline     -- Real thresholds. Classification accuracy AND (on
                  benchmark45 only) the 9-ticket adversarial real-escalation
                  decision.
  no-cascade   -- run_cascade(threshold=0.0) => Tier-1 raw preds for every
                  ticket.
  tier2-only   -- run_cascade(threshold=TIER2_ONLY_THRESHOLD) => every ticket
                  is answered by the production Tier-2 (BGE) classifier and
                  Tier-1 never decides anything. Added in Phase 5B.
  no-rag       -- Real retrieval on the 9-ticket adversarial set, but the
                  escalation gate is pretend-threshold 0.0 => a Gemini call
                  WOULD be attempted whenever retrieval returns anything.
                  Gemini is NEVER actually called.

Evaluation sets (--set):
  benchmark45    -- data/novel_tickets_expanded.json (45 tickets, READ-ONLY).
                    The historical default; keeps the historical CSV names so
                    the published results stay byte-identical.
  deployment175  -- data/deployment_calibration_tickets.json (175 tickets,
                    25 per category). Writes *_deployment175.csv, so nothing
                    published is ever overwritten.

WHY tier2-only exists (Phase 5B). The published "the cascade is worth +35.6
points" is baseline minus no-cascade, and no-cascade is TF-IDF answering every
ticket -- so it measures the BGE-vs-TF-IDF representation gap, not the value
of CASCADING. Cascade vs Tier-2-alone is the comparison that isolates the
cascade, and it had never been run.

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

# Needed before the src.agent.config import below, since this script is run
# as a path (python src/experiments/run_ablation_study.py) rather than -m.
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

SRC_DIR = os.path.join(PROJECT_ROOT, "src")
CLASSIFICATION_DIR = os.path.join(SRC_DIR, "classification")
RAG_DIR = os.path.join(SRC_DIR, "rag")

DATA_DIR = os.path.join(PROJECT_ROOT, "data")
MODELS_DIR = os.path.join(PROJECT_ROOT, "models")

EXPANDED_JSON_PATH = os.path.join(DATA_DIR, "novel_tickets_expanded.json")
ADVERSARIAL_JSON_PATH = os.path.join(DATA_DIR, "adversarial_escalation_tickets.json")
DEPLOYMENT175_JSON_PATH = os.path.join(
    DATA_DIR, "deployment_calibration_tickets.json"
)
# MIGRATED TO BGE (was MiniLM).
#
# This script previously pointed at ticket_index.faiss, ticket_metadata.json,
# ticket_classifier.joblib and all-MiniLM-L6-v2 -- every one of them a
# pre-BGE-swap artifact. Because those four were mutually consistent, it ran
# without error and silently measured the OLD pipeline. The results CSV was
# written 2026-08-15; the BGE swap landed 2026-08-26, eleven days later, and
# this script was never re-run. The published ablation numbers therefore
# described a pipeline that no longer existed.
#
# All four now come from src/agent/config.py, so this script can never again
# drift from production independently.
from src.agent.config import settings as _settings  # noqa: E402

FAISS_INDEX_PATH = str(_settings.models.faiss_index_path)
METADATA_JSON_PATH = str(_settings.models.metadata_path)
TIER2_MODEL_PATH = str(_settings.models.tier2_classifier_path)

# Applied live thresholds -- single source of truth.
CASCADE_CONFIDENCE_THRESHOLD = _settings.cascade.confidence_threshold
# RAG SIMILARITY_THRESHOLD is imported from suggest_resolution, which itself
# now re-exports settings.rag.similarity_threshold.

EMBED_MODEL_NAME = _settings.models.embedding_model

# For summary text only; the gate value actually applied comes from
# suggest_resolution.SIMILARITY_THRESHOLD via _import_project_functions().
SIMILARITY_THRESHOLD_DISPLAY = _settings.rag.similarity_threshold

VALID_MODES = ("baseline", "no-cascade", "no-rag", "tier2-only")

# Evaluation sets. benchmark45 is the historical default and keeps the
# historical output filenames.
DEFAULT_EVAL_SET = "benchmark45"
VALID_SETS = (DEFAULT_EVAL_SET, "benchmark14", "deployment175")
EVAL_SET_SIZES = {DEFAULT_EVAL_SET: 45, "benchmark14": 14,
                  "deployment175": 175}

# tier2-only is expressed as a cascade threshold that NO Tier-1 confidence can
# reach, so every ticket escalates to Tier-2 and the rest of the path is
# byte-for-byte the baseline path -- structurally identical, not a copy.
#
# A sentinel constant is exactly the shape of this project's recurring bug
# class (a value that is wrong for its context but internally consistent), so
# run_tier2_only() ASSERTS that all N tickets were actually answered by Tier-2
# rather than trusting the arithmetic. Tier-1 confidence is a max over a
# predict_proba row and is therefore <= 1.0 by construction.
TIER2_ONLY_THRESHOLD = 2.0


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
def _import_artifacts():
    """Import THE loader (src/agent/artifacts.py).

    Isolated so an import failure is actionable rather than a traceback.
    """
    try:
        from src.agent import artifacts as artifacts_mod
    except Exception as exc:
        _fatal(
            "Failed to import src/agent/artifacts.py. Is the venv activated "
            "and are you running from the project root? " + repr(exc)
        )
    return artifacts_mod


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


def load_all(artifacts_mod):
    """Load every production artifact through THE loader.

    This script used to carry four loaders of its own, and one of them
    REFITTED Tier-1 from synthetic_tickets.csv on every run. That is the exact
    shape of this project's recurring bug class: a locally-derived model that
    stays internally consistent while silently diverging from the artifact
    production actually serves. load_artifacts() additionally applies the
    three hard guards the local loaders never had -- index/metadata alignment,
    encoder dim == index.d == configured dim, and the Tier-1 manifest against
    the dataset it was fitted on.

    The migration was verified parity-preserving BEFORE it landed (Phase 5B):
    the persisted Tier-1 and the old local refit agreed to
    max |delta tier1_conf| = 0.0 across the 45-ticket benchmark, and both
    round to the published CSVs' 6 decimals with zero mismatches.
    """
    try:
        return artifacts_mod.load_artifacts(require_gemini=False)
    except Exception as exc:
        _fatal(
            "Failed to load production artifacts via "
            "src/agent/artifacts.load_artifacts():\n  {e}\n\n"
            "Build them first, from the project root:\n"
            "  python data/generate_dataset.py\n"
            "  python src/classification/train_tier1.py\n"
            "  python src/classification/train_embeddings.py\n"
            "  python src/rag/build_vector_index.py".format(e=repr(exc))
        )


def load_deployment175_set():
    """Load the 175-ticket deployment-distribution evaluation set.

    Validated the same way load_expanded_set() validates the 45-ticket
    benchmark: exact record count, non-empty text/expected on EVERY record,
    nothing silently dropped.

    NOTE for anyone reading a result off this set: these tickets are
    Gemini-generated deployment-register text, NOT production traffic, and the
    same 175 tickets already carry Phase 1 Finding 4's conformal calibration.
    Using them for accuracy adds another use of an already multiply-used set.
    """
    _require_file(
        DEPLOYMENT175_JSON_PATH,
        "deployment evaluation set (deployment_calibration_tickets.json)",
    )
    try:
        with open(DEPLOYMENT175_JSON_PATH, "r", encoding="utf-8") as fh:
            data = json.load(fh)
    except Exception as exc:
        _fatal("Failed to read deployment_calibration_tickets.json: " + repr(exc))

    if not isinstance(data, list):
        _fatal("deployment_calibration_tickets.json must be a JSON list.")

    expected_n = EVAL_SET_SIZES["deployment175"]
    if len(data) != expected_n:
        _fatal(
            "deployment_calibration_tickets.json must contain EXACTLY {n} "
            "records; found {f}.".format(n=expected_n, f=len(data))
        )

    bad = []
    for i, rec in enumerate(data):
        if not isinstance(rec, dict):
            bad.append((i, "record is not an object"))
            continue
        for k in ("text", "expected"):
            if k not in rec:
                bad.append((i, "missing key '{k}'".format(k=k)))
            elif not isinstance(rec[k], str) or not rec[k].strip():
                bad.append((i, "'{k}' is empty or not a string".format(k=k)))

    if bad:
        lines = "\n".join(
            "  record #{i}: {why}".format(i=i, why=why) for i, why in bad[:10]
        )
        _fatal(
            "deployment_calibration_tickets.json has {n} malformed record(s) "
            "(every record needs non-empty 'text' and 'expected'):\n"
            "{lines}".format(n=len(bad), lines=lines)
        )

    return list(data)


def load_benchmark14_set():
    """The original 14-ticket generalization benchmark (READ-ONLY).

    Its source of truth is NOVEL_TICKETS in
    src/classification/generalization_test.py -- imported, never copied, so it
    cannot drift from the set every other comparison in the project uses.
    """
    try:
        from src.classification.generalization_test import NOVEL_TICKETS
    except Exception as exc:
        _fatal(
            "Failed to import NOVEL_TICKETS from "
            "src/classification/generalization_test.py: " + repr(exc)
        )

    expected_n = EVAL_SET_SIZES["benchmark14"]
    if len(NOVEL_TICKETS) != expected_n:
        _fatal(
            "NOVEL_TICKETS must contain EXACTLY {e} tickets; found {f}. This "
            "benchmark is read-only and must not have changed.".format(
                e=expected_n, f=len(NOVEL_TICKETS)
            )
        )
    for i, rec in enumerate(NOVEL_TICKETS):
        for k in ("text", "expected"):
            if k not in rec or not str(rec[k]).strip():
                _fatal(
                    "NOVEL_TICKETS[{i}] has an empty or missing "
                    "'{k}'.".format(i=i, k=k)
                )
    return [dict(r) for r in NOVEL_TICKETS]


def load_eval_set(funcs, eval_set):
    """Dispatch to the right evaluation set, validated by its own loader."""
    if eval_set == DEFAULT_EVAL_SET:
        return funcs["load_expanded_set"](EXPANDED_JSON_PATH)
    if eval_set == "benchmark14":
        return load_benchmark14_set()
    if eval_set == "deployment175":
        return load_deployment175_set()
    _fatal("Unknown evaluation set: " + repr(eval_set))


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
# Column set for the classification-only CSVs (no-cascade, tier2-only, and
# baseline on any set other than benchmark45).
CLS_FIELDNAMES = [
    "index",
    "text",
    "expected",
    "predicted",
    "tier1_conf",
    "tier_used",
    "correct",
]


def write_csv(mode, fieldnames, rows, eval_set=DEFAULT_EVAL_SET):
    # benchmark45 keeps the historical filename, so the three already-published
    # CSVs stay byte-identical. Every other set gets its own suffixed name --
    # project rule 4: a new result gets a NEW filename, never an overwrite.
    suffix = "" if eval_set == DEFAULT_EVAL_SET else "_" + eval_set
    out_path = os.path.join(
        DATA_DIR,
        "ablation_{mode}_results{sfx}.csv".format(mode=mode, sfx=suffix),
    )
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
                            tier1_vec, tier1_clf, tier2_clf, embedder,
                            eval_set=DEFAULT_EVAL_SET):
    """
    Runs run_cascade over the chosen evaluation set at the given threshold and
    returns (rows, accuracy, n_correct, n_total, n_tier2).

    threshold=0.50                  -> baseline behaviour (real cascade).
    threshold=0.0                   -> no-cascade (escalate_idx always empty
                                       => Tier-1 raw for every ticket).
    threshold=TIER2_ONLY_THRESHOLD  -> tier2-only (escalate_idx is every index
                                       => Tier-2 answers every ticket).

    Only the threshold and the record source change between modes; the routing
    path itself is the same tested run_cascade call in all three.
    """
    records = load_eval_set(funcs, eval_set)

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

    baseline uses gate_threshold = SIMILARITY_THRESHOLD (see
    src/agent/config.py).
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
def run_baseline(funcs, art, eval_set=DEFAULT_EVAL_SET):
    _banner("MODE: baseline  (cascade={c}, rag={r}, set={s})".format(
        c=CASCADE_CONFIDENCE_THRESHOLD, r=funcs["SIMILARITY_THRESHOLD"],
        s=eval_set))

    print("Loading resources...")
    tier1_vec, tier1_clf = art.tier1_vectorizer, art.tier1_classifier
    tier2_clf = art.tier2_classifier
    embedder = art.embedder

    # --- classification on the chosen set (real cascade @ 0.50) ---
    print("Evaluating {s} classification (cascade threshold={c})...".format(
        s=eval_set, c=CASCADE_CONFIDENCE_THRESHOLD))
    cls_rows, accuracy, n_correct, n_total, n_tier2 = evaluate_classification(
        funcs,
        CASCADE_CONFIDENCE_THRESHOLD,
        "baseline",
        tier1_vec,
        tier1_clf,
        tier2_clf,
        embedder,
        eval_set=eval_set,
    )

    # The adversarial escalation section is defined on the FIXED 9-ticket set
    # and measures the RAG gate, not classification. It has no meaning for a
    # different classification set, so it is skipped rather than faked.
    if eval_set != DEFAULT_EVAL_SET:
        out_path = write_csv("baseline", CLS_FIELDNAMES, cls_rows,
                             eval_set=eval_set)
        _banner("baseline RESULTS ({s})".format(s=eval_set))
        print(
            "Classification ({s}): {c}/{t} correct  =>  accuracy {a:.2%}  "
            "(answered by Tier-2: {e})".format(
                s=eval_set, c=n_correct, t=n_total, a=accuracy, e=n_tier2
            )
        )
        print(
            "Adversarial escalation section skipped: it is defined on the "
            "fixed 9-ticket set only."
        )
        print("CSV written: " + out_path)
        return {
            "mode": "baseline",
            "eval_set": eval_set,
            "accuracy": accuracy,
            "n_correct": n_correct,
            "n_total": n_total,
            "n_tier2": n_tier2,
            "rag_summary": None,
            "csv": out_path,
        }

    index, metadata = art.index, art.metadata
    faiss = art.faiss
    adversarial = load_adversarial_set()

    # --- real escalation on the 9-ticket adversarial set ---
    print("Evaluating 9-ticket adversarial escalation (rag threshold={r})..."
          .format(r=funcs["SIMILARITY_THRESHOLD"]))
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
    out_path = write_csv("baseline", fieldnames, combined_rows,
                         eval_set=eval_set)

    _banner("baseline RESULTS")
    print(
        "Classification (45-ticket): {c}/{t} correct  =>  accuracy {a:.2%}  "
        "(Tier-2 escalations: {e})".format(
            c=n_correct, t=n_total, a=accuracy, e=n_tier2
        )
    )
    print(
        "Adversarial escalation (9-ticket): {esc}/{tot} correctly escalated "
        "(gate < {g})".format(
            g=funcs["SIMILARITY_THRESHOLD"],
            esc=rag_summary["n_correct_escalation"], tot=rag_summary["n_total"]
        )
    )
    print("CSV written: " + out_path)

    return {
        "mode": "baseline",
        "eval_set": eval_set,
        "accuracy": accuracy,
        "n_correct": n_correct,
        "n_total": n_total,
        "n_tier2": n_tier2,
        "rag_summary": rag_summary,
        "csv": out_path,
    }


def run_no_cascade(funcs, art, eval_set=DEFAULT_EVAL_SET):
    _banner("MODE: no-cascade  (cascade DISABLED via threshold=0.0, set={s})"
            .format(s=eval_set))

    print("Loading resources...")
    tier1_vec, tier1_clf = art.tier1_vectorizer, art.tier1_classifier
    tier2_clf = art.tier2_classifier  # signature parity; unused at 0.0
    embedder = art.embedder

    print(
        "Evaluating {s} classification with run_cascade(threshold=0.0)\n"
        "  -> escalate_idx always empty -> Tier-1 raw prediction for every "
        "ticket.".format(s=eval_set)
    )
    cls_rows, accuracy, n_correct, n_total, n_tier2 = evaluate_classification(
        funcs,
        0.0,
        "no-cascade",
        tier1_vec,
        tier1_clf,
        tier2_clf,
        embedder,
        eval_set=eval_set,
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
    out_path = write_csv("no-cascade", fieldnames, cls_rows,
                         eval_set=eval_set)

    _banner("no-cascade RESULTS")
    print(
        "Classification ({s}, Tier-1 only): {c}/{t} correct  =>  "
        "accuracy {a:.2%}".format(s=eval_set, c=n_correct, t=n_total,
                                  a=accuracy)
    )
    print("CSV written: " + out_path)

    return {
        "mode": "no-cascade",
        "eval_set": eval_set,
        "accuracy": accuracy,
        "n_correct": n_correct,
        "n_total": n_total,
        "n_tier2": n_tier2,
        "csv": out_path,
    }


def run_tier2_only(funcs, art, eval_set=DEFAULT_EVAL_SET):
    """Every ticket answered by the production Tier-2 (BGE) classifier.

    Phase 5B. This is the comparison that isolates the CASCADE, as opposed to
    no-cascade, which isolates the REPRESENTATION (TF-IDF vs BGE). Nothing
    differs from baseline except the threshold handed to run_cascade, so any
    accuracy difference is attributable to Tier-1 keeping tickets rather than
    escalating them.
    """
    _banner("MODE: tier2-only  (Tier-1 never decides; threshold={t}, set={s})"
            .format(t=TIER2_ONLY_THRESHOLD, s=eval_set))

    print("Loading resources...")
    tier1_vec, tier1_clf = art.tier1_vectorizer, art.tier1_classifier
    tier2_clf = art.tier2_classifier
    embedder = art.embedder

    print(
        "Evaluating {s} classification with run_cascade(threshold={t})\n"
        "  -> every Tier-1 confidence is below the sentinel -> Tier-2 answers "
        "every ticket.".format(s=eval_set, t=TIER2_ONLY_THRESHOLD)
    )
    cls_rows, accuracy, n_correct, n_total, n_tier2 = evaluate_classification(
        funcs,
        TIER2_ONLY_THRESHOLD,
        "tier2-only",
        tier1_vec,
        tier1_clf,
        tier2_clf,
        embedder,
        eval_set=eval_set,
    )

    # HARD GUARD, not a warning (rule 6: check a count a second, independent
    # way). A sentinel threshold that silently failed to route everything to
    # Tier-2 would produce a plausible, internally consistent, WRONG number --
    # this project's recurring bug class. Two independent derivations must
    # agree: run_cascade's own counter, and the per-row tier_used column.
    n_rows_tier2 = sum(1 for r in cls_rows if r["tier_used"] == 2)
    if n_tier2 != n_total or n_rows_tier2 != n_total:
        _fatal(
            "tier2-only did NOT route every ticket to Tier-2.\n"
            "  run_cascade n_tier2   = {a} (expected {t})\n"
            "  rows with tier_used=2 = {b} (expected {t})\n"
            "The sentinel threshold ({s}) must exceed every possible Tier-1 "
            "confidence. Refusing to write a result that does not measure "
            "what this mode claims to measure.".format(
                a=n_tier2, b=n_rows_tier2, t=n_total, s=TIER2_ONLY_THRESHOLD
            )
        )

    out_path = write_csv("tier2-only", CLS_FIELDNAMES, cls_rows,
                         eval_set=eval_set)

    _banner("tier2-only RESULTS")
    print(
        "Classification ({s}, Tier-2 only): {c}/{t} correct  =>  "
        "accuracy {a:.2%}".format(s=eval_set, c=n_correct, t=n_total,
                                  a=accuracy)
    )
    print(
        "Verified: {n}/{t} tickets answered by Tier-2 (two independent "
        "counts agree).".format(n=n_tier2, t=n_total)
    )
    print("CSV written: " + out_path)

    return {
        "mode": "tier2-only",
        "eval_set": eval_set,
        "accuracy": accuracy,
        "n_correct": n_correct,
        "n_total": n_total,
        "n_tier2": n_tier2,
        "csv": out_path,
    }


def run_no_rag(funcs, art, eval_set=DEFAULT_EVAL_SET):
    _banner("MODE: no-rag  (rag gate DISABLED via pretend threshold=0.0)")

    print("Loading resources...")
    embedder = art.embedder
    index, metadata = art.index, art.metadata
    faiss = art.faiss
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
            "baseline classification accuracy ({s}): {a:.2%} "
            "({c}/{t})".format(
                s=result["eval_set"], a=result["accuracy"],
                c=result["n_correct"], t=result["n_total"]
            )
        )
        # rag_summary is None whenever the adversarial section was skipped,
        # which is every set other than benchmark45.
        rs = result.get("rag_summary")
        if rs is None:
            print(
                "baseline: adversarial escalation not evaluated on this set "
                "(it is defined\n          on the fixed 9-ticket set only)."
            )
        else:
            print(
                "baseline: {esc}/{t} adversarial tickets correctly escalated "
                "(gate < {g}).".format(esc=rs["n_correct_escalation"],
                                       t=rs["n_total"],
                                       g=SIMILARITY_THRESHOLD_DISPLAY)
            )
        print("")
        print(
            "Run --mode tier2-only to isolate the CASCADE, --mode no-cascade "
            "to isolate\nthe REPRESENTATION, and --mode no-rag to compare "
            "escalation behaviour."
        )

    elif mode == "no-cascade":
        print(
            "no-cascade classification accuracy (Tier-1 only, {s}): "
            "{a:.2%} ({c}/{t})".format(
                s=result["eval_set"], a=result["accuracy"],
                c=result["n_correct"], t=result["n_total"]
            )
        )
        print("")
        print(
            "Compare against the baseline run. NOTE what this difference "
            "actually is:\nno-cascade is TF-IDF answering EVERY ticket, so "
            "baseline minus no-cascade\nmeasures the BGE-vs-TF-IDF "
            "representation gap, NOT the value of cascading.\nFor that, "
            "compare baseline against --mode tier2-only."
        )

    elif mode == "tier2-only":
        print(
            "tier2-only classification accuracy (Tier-2 only, {s}): "
            "{a:.2%} ({c}/{t})".format(
                s=result["eval_set"], a=result["accuracy"],
                c=result["n_correct"], t=result["n_total"]
            )
        )
        print("")
        print(
            "This is the comparison that ISOLATES THE CASCADE: baseline "
            "against this\nnumber differs only in whether Tier-1 was allowed "
            "to answer. Run\n  python src/experiments/compare_cascade_vs_"
            "tier2.py --set {s}\nfor the paired exact McNemar test and the "
            "discordant tickets.".format(s=result["eval_set"])
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
            "({g}).".format(g=SIMILARITY_THRESHOLD_DISPLAY)
        )


# --------------------------------------------------------------------------
# CLI
# --------------------------------------------------------------------------
def parse_args(argv):
    parser = argparse.ArgumentParser(
        description=(
            "Ablation study for the ticket routing agent. Measures the value "
            "of the cascade confidence threshold (0.50) and the RAG similarity "
            "threshold by disabling each individually."
        )
    )
    parser.add_argument(
        "--mode",
        required=True,
        choices=VALID_MODES,
        help=(
            "Which ablation to run: baseline | no-cascade | tier2-only | "
            "no-rag"
        ),
    )
    parser.add_argument(
        "--set",
        dest="eval_set",
        default=DEFAULT_EVAL_SET,
        choices=VALID_SETS,
        help=(
            "Classification evaluation set. benchmark45 (default, 45 tickets, "
            "historical CSV names) or deployment175 (175 tickets, writes "
            "*_deployment175.csv)."
        ),
    )
    return parser.parse_args(argv)


def main(argv=None):
    random.seed(42)
    np.random.seed(42)

    args = parse_args(argv)
    mode = args.mode
    eval_set = args.eval_set

    # no-rag measures the RAG gate on the fixed 9-ticket adversarial set and
    # never touches a classification set. Accepting --set there would imply a
    # choice that does not exist, so it is refused rather than ignored.
    if mode == "no-rag" and eval_set != DEFAULT_EVAL_SET:
        _fatal(
            "--mode no-rag does not take --set: it evaluates the RAG gate on "
            "the fixed\n9-ticket adversarial set only, and never runs "
            "classification."
        )

    _banner("AI Ticket Agent -- Ablation Study")
    print("Project root : " + PROJECT_ROOT)
    print("Mode         : " + mode)
    if mode != "no-rag":
        print("Eval set     : {s} ({n} tickets)".format(
            s=eval_set, n=EVAL_SET_SIZES[eval_set]))

    funcs = _import_project_functions()
    artifacts_mod = _import_artifacts()
    art = load_all(artifacts_mod)

    if mode == "baseline":
        result = run_baseline(funcs, art, eval_set)
    elif mode == "no-cascade":
        result = run_no_cascade(funcs, art, eval_set)
    elif mode == "tier2-only":
        result = run_tier2_only(funcs, art, eval_set)
    elif mode == "no-rag":
        result = run_no_rag(funcs, art, eval_set)
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
