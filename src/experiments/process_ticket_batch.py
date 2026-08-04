"""
process_ticket_batch.py

Run the full existing pipeline (cascade-consistent classification -> RAG
resolution or escalation) against the simulated skewed batch produced by
simulate_ticket_intake.py.

Outputs:
    data/category_stores/{Category}.csv          (one per category, append)
    data/category_stores/escalation_needs_review.csv
    data/batch_intake/batch_summary.csv

Run from project root:
    python -m src.experiments.process_ticket_batch
    (or) python src/experiments/process_ticket_batch.py
"""

import os
import sys
import time
import traceback

import numpy as np
import pandas as pd

# --------------------------------------------------------------------------- #
# Configuration / named constants
# --------------------------------------------------------------------------- #
RANDOM_SEED = 42

FILING_CONFIDENCE_THRESHOLD = 0.50

TOP_K = 5
GEMINI_CALL_DELAY_SEC = 1.5  # Bumped from 1.0s — you hit a 429 immediately;
                             # give the free tier more headroom per call.
MAX_GEMINI_RETRIES = 3
INITIAL_BACKOFF_SEC = 4.0    # Bumped from 2.0s for the same reason.
PROGRESS_EVERY = 25

# --------------------------------------------------------------------------- #
# Path resolution: project root = two directories up from this script's dir.
# --------------------------------------------------------------------------- #
_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(_SCRIPT_DIR, "..", ".."))

if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

DATA_DIR = os.path.join(PROJECT_ROOT, "data")
BATCH_DIR = os.path.join(DATA_DIR, "batch_intake")
BATCH_CSV = os.path.join(BATCH_DIR, "incoming_tickets_batch.csv")
SUMMARY_CSV = os.path.join(BATCH_DIR, "batch_summary.csv")

CATEGORY_STORE_DIR = os.path.join(DATA_DIR, "category_stores")
ESCALATION_CSV = os.path.join(CATEGORY_STORE_DIR, "escalation_needs_review.csv")

MODEL_PATH = os.path.join(DATA_DIR, "..", "models", "ticket_classifier.joblib")
MODEL_PATH = os.path.abspath(MODEL_PATH)

CATEGORIES = [
    "Infrastructure",
    "Application",
    "Security",
    "Database",
    "Storage",
    "Network",
    "Access Management",
]

STORE_COLUMNS = [
    "batch_ticket_id",
    "original_id",
    "title",
    "description",
    "predicted_category",
    "classification_confidence",
    "retrieval_similarity",
    "resolution_status",
    "resolution_text",
    "ground_truth_category",
]
ESCALATION_COLUMNS = STORE_COLUMNS + ["escalation_reason"]


def _fail(message: str) -> None:
    """Print a clear, actionable error and exit non-zero (no raw traceback)."""
    print(f"\n[ERROR] {message}\n", file=sys.stderr)
    sys.exit(1)


# --------------------------------------------------------------------------- #
# Lazy imports of heavy / project dependencies, each with a clear failure msg.
# --------------------------------------------------------------------------- #
def load_dependencies():
    try:
        import joblib
    except ImportError:
        _fail("joblib is not installed. Activate your venv and `pip install joblib`.")

    try:
        from sentence_transformers import SentenceTransformer
    except ImportError:
        _fail(
            "sentence-transformers is not installed. Activate your venv and "
            "`pip install sentence-transformers`."
        )

    try:
        from src.rag import suggest_resolution as sr
    except ImportError as exc:
        _fail(
            "Could not import src.rag.suggest_resolution.\n"
            f"Underlying error: {exc}\n"
            "Run this from the project root so `src` is importable, e.g.:\n"
            "    python -m src.experiments.process_ticket_batch"
        )

    return joblib, SentenceTransformer, sr


def load_batch() -> pd.DataFrame:
    if not os.path.isfile(BATCH_CSV):
        _fail(
            "Batch file not found:\n"
            f"    {BATCH_CSV}\n"
            "Run the intake simulation first:\n"
            "    python -m src.experiments.simulate_ticket_intake"
        )
    df = pd.read_csv(BATCH_CSV)
    required = {
        "batch_ticket_id",
        "original_id",
        "title",
        "description",
        "ground_truth_category",
    }
    missing = required - set(df.columns)
    if missing:
        _fail(
            f"{BATCH_CSV} is missing column(s): {sorted(missing)}.\n"
            "Regenerate it with simulate_ticket_intake.py."
        )
    return df


def load_classifier(joblib):
    if not os.path.isfile(MODEL_PATH):
        _fail(
            "Trained classifier not found:\n"
            f"    {MODEL_PATH}\n"
            "Train it first:\n"
            "    python -m src.embeddings.train_embeddings\n"
            "(adjust to your actual train_embeddings entrypoint)."
        )
    try:
        clf = joblib.load(MODEL_PATH)
    except Exception as exc:  # noqa: BLE001
        _fail(f"Failed to load {MODEL_PATH}: {exc}")
    return clf


def get_embedding_model(SentenceTransformer):
    model_name = "sentence-transformers/all-MiniLM-L6-v2"
    try:
        return SentenceTransformer(model_name)
    except Exception as exc:  # noqa: BLE001
        _fail(
            f"Failed to load embedding model '{model_name}': {exc}\n"
            "Check your internet connection / model cache."
        )


# --------------------------------------------------------------------------- #
# Gemini call wrapper with retry + backoff.
#
# IMPORTANT: this does NOT call suggest_resolution.call_gemini() directly.
# That function is written for the standalone demo script: on a rate-limit
# or auth failure it calls _fail(), which does sys.exit(1). sys.exit()
# raises SystemExit, which is a BaseException, NOT an Exception - so it
# silently escapes an `except Exception` retry block and kills the whole
# batch run on the very first 429 instead of backing off. This local
# variant mirrors the same error classification but RAISES a normal
# Exception instead, so the retry loop below can actually catch it.
# --------------------------------------------------------------------------- #
def _is_rate_limit_error(exc: Exception) -> bool:
    """Best-effort detection of 429 / rate-limit style errors."""
    text = f"{type(exc).__name__} {exc}".lower()
    needles = ("429", "rate limit", "rate-limit", "resource_exhausted",
               "quota", "too many requests")
    return any(n in text for n in needles)


def call_gemini_raise_on_error(sr, client, prompt):
    """
    Batch-safe variant of suggest_resolution.call_gemini(). Classifies the
    same failure modes (rate limit, auth, network, blocked/empty response)
    but raises instead of exiting the process, so callers can retry.
    """
    try:
        response = client.models.generate_content(
            model=sr.MODEL_NAME,
            contents=prompt,
        )
    except Exception as exc:  # noqa: BLE001
        # Re-raise as a plain Exception carrying the original error text, so
        # _is_rate_limit_error() upstream can still detect "429"/"quota"/etc.
        raise RuntimeError(
            f"Gemini call failed ({type(exc).__name__}): {exc}"
        ) from exc

    text = getattr(response, "text", None)
    if text is None or not str(text).strip():
        return (
            "[Gemini returned an empty or blocked response. No suggestion "
            "could be generated. A human agent should handle this ticket.]"
        )
    return str(text).strip()


def call_gemini_with_retry(sr, client, ticket_title, ticket_description, similar):
    """
    Build the grounded prompt via suggest_resolution.build_llm_prompt (real
    kwargs: query_title, query_description, retrieved), then call Gemini
    through the raise-on-error wrapper above, with retry + exponential
    backoff on rate-limit errors specifically.
    """
    backoff = INITIAL_BACKOFF_SEC
    last_exc = None

    for attempt in range(1, MAX_GEMINI_RETRIES + 1):
        try:
            prompt = sr.build_llm_prompt(
                query_title=ticket_title,
                query_description=ticket_description,
                retrieved=similar,
            )
            resolution = call_gemini_raise_on_error(sr, client, prompt)
            return resolution
        except Exception as exc:  # noqa: BLE001
            last_exc = exc
            if _is_rate_limit_error(exc) and attempt < MAX_GEMINI_RETRIES:
                print(
                    f"    [rate-limit] attempt {attempt}/{MAX_GEMINI_RETRIES} "
                    f"failed; backing off {backoff:.1f}s..."
                )
                time.sleep(backoff)
                backoff *= 2
                continue
            raise last_exc

    raise last_exc if last_exc else RuntimeError("Unknown Gemini failure")


def get_best_similarity(sr, title, description, model, index, metadata, faiss):
    """
    Call retrieve_similar_tickets() with its real required arguments and
    return (best_score, results). It returns a list of dicts each carrying
    a 'similarity' key.
    """
    results = sr.retrieve_similar_tickets(
        f"{title} {description}", model, index, metadata, faiss, top_k=TOP_K
    )

    best = 0.0
    if results is None:
        return 0.0, []

    for r in results:
        score = r.get("similarity") if isinstance(r, dict) else None
        if score is not None:
            best = max(best, float(score))

    return best, results


# --------------------------------------------------------------------------- #
# Store I/O helpers
# --------------------------------------------------------------------------- #
def append_record(path: str, record: dict, columns: list) -> None:
    row_df = pd.DataFrame([{c: record.get(c) for c in columns}])
    write_header = not os.path.isfile(path)
    row_df.to_csv(path, mode="a", header=write_header, index=False)


def safe_category_filename(category: str) -> str:
    return os.path.join(CATEGORY_STORE_DIR, f"{category}.csv")


# --------------------------------------------------------------------------- #
# Main processing loop
# --------------------------------------------------------------------------- #
def process(batch, clf, model, sr, index, metadata, faiss, client):
    os.makedirs(CATEGORY_STORE_DIR, exist_ok=True)

    total = len(batch)
    gemini_calls_made = 0

    stats = {
        cat: {
            "volume_received": 0,
            "auto_resolved_count": 0,
            "needs_human_resolution_count": 0,
            "gemini_call_failed_count": 0,
            "correct_predictions": 0,
        }
        for cat in CATEGORIES
    }
    escalation_stats = {
        "volume_received": 0,
        "auto_resolved_count": 0,
        "needs_human_resolution_count": 0,
        "gemini_call_failed_count": 0,
        "correct_predictions": 0,
    }

    if not hasattr(clf, "classes_"):
        _fail(
            "Loaded classifier has no `classes_` attribute; cannot map "
            "predict_proba columns to category names. "
            "Check that models/ticket_classifier.joblib is the LogReg model."
        )
    class_labels = list(clf.classes_)

    for i, row in enumerate(batch.itertuples(index=False), start=1):
        title = "" if pd.isna(row.title) else str(row.title)
        description = "" if pd.isna(row.description) else str(row.description)
        ground_truth = row.ground_truth_category

        text = f"{title} {description}"
        try:
            embedding = model.encode([text])
        except Exception as exc:  # noqa: BLE001
            print(
                f"    [WARNING] Embedding failed for batch_ticket_id="
                f"{row.batch_ticket_id}: {exc}. Skipping ticket."
            )
            continue

        try:
            proba = clf.predict_proba(embedding)[0]
        except Exception as exc:  # noqa: BLE001
            print(
                f"    [WARNING] predict_proba failed for batch_ticket_id="
                f"{row.batch_ticket_id}: {exc}. Skipping ticket."
            )
            continue

        best_idx = int(np.argmax(proba))
        predicted_category = class_labels[best_idx]
        confidence = float(proba[best_idx])

        base_record = {
            "batch_ticket_id": row.batch_ticket_id,
            "original_id": row.original_id,
            "title": title,
            "description": description,
            "predicted_category": predicted_category,
            "classification_confidence": round(confidence, 6),
            "retrieval_similarity": None,
            "resolution_status": None,
            "resolution_text": None,
            "ground_truth_category": ground_truth,
        }

        is_correct = (predicted_category == ground_truth)

        if confidence < FILING_CONFIDENCE_THRESHOLD:
            record = dict(base_record)
            record["resolution_status"] = "escalated"
            record["escalation_reason"] = "low_confidence_classification"
            append_record(ESCALATION_CSV, record, ESCALATION_COLUMNS)

            escalation_stats["volume_received"] += 1
            if is_correct:
                escalation_stats["correct_predictions"] += 1

        else:
            try:
                best_sim, similar = get_best_similarity(
                    sr, title, description, model, index, metadata, faiss
                )
            except Exception as exc:  # noqa: BLE001
                print(
                    f"    [WARNING] Retrieval failed for batch_ticket_id="
                    f"{row.batch_ticket_id}: {exc}. Treating as no match."
                )
                best_sim, similar = 0.0, []

            record = dict(base_record)
            record["retrieval_similarity"] = round(float(best_sim), 6)

            if best_sim >= sr.SIMILARITY_THRESHOLD:
                if gemini_calls_made > 0:
                    time.sleep(GEMINI_CALL_DELAY_SEC)

                try:
                    resolution = call_gemini_with_retry(
                        sr, client, title, description, similar
                    )
                    gemini_calls_made += 1
                    record["resolution_status"] = "auto_resolved"
                    record["resolution_text"] = resolution
                    stats[predicted_category]["auto_resolved_count"] += 1
                except Exception as exc:  # noqa: BLE001
                    gemini_calls_made += 1
                    print(
                        f"    [WARNING] Gemini call failed for batch_ticket_id="
                        f"{row.batch_ticket_id} after retries: {exc}"
                    )
                    record["resolution_status"] = "gemini_call_failed"
                    record["resolution_text"] = None
                    stats[predicted_category]["gemini_call_failed_count"] += 1
            else:
                record["resolution_status"] = "needs_human_resolution"
                stats[predicted_category]["needs_human_resolution_count"] += 1

            store_path = safe_category_filename(predicted_category)
            append_record(store_path, record, STORE_COLUMNS)

            stats[predicted_category]["volume_received"] += 1
            if is_correct:
                stats[predicted_category]["correct_predictions"] += 1

        if i % PROGRESS_EVERY == 0 or i == total:
            print(f"Processed {i}/{total} tickets...")

    return stats, escalation_stats


def write_summary(stats, escalation_stats, total_tickets):
    rows = []

    print("\n" + "=" * 96)
    print("BATCH PROCESSING SUMMARY")
    print("(classification_accuracy is EVALUATION-ONLY, not used by the pipeline)")
    print("=" * 96)
    header = (
        f"{'Category':<20}{'Recv':>6}{'AutoRes':>9}{'NeedHuman':>11}"
        f"{'GemFail':>9}{'Acc(eval)':>11}"
    )
    print(header)
    print("-" * 96)

    for cat in CATEGORIES:
        s = stats[cat]
        recv = s["volume_received"]
        acc = (s["correct_predictions"] / recv) if recv else 0.0
        print(
            f"{cat:<20}{recv:>6}{s['auto_resolved_count']:>9}"
            f"{s['needs_human_resolution_count']:>11}"
            f"{s['gemini_call_failed_count']:>9}{acc:>11.3f}"
        )
        rows.append(
            {
                "category": cat,
                "volume_received": recv,
                "auto_resolved_count": s["auto_resolved_count"],
                "needs_human_resolution_count": s["needs_human_resolution_count"],
                "gemini_call_failed_count": s["gemini_call_failed_count"],
                "classification_accuracy_on_this_batch": round(acc, 6),
            }
        )

    e = escalation_stats
    e_recv = e["volume_received"]
    e_acc = (e["correct_predictions"] / e_recv) if e_recv else 0.0
    print("-" * 96)
    print(
        f"{'ESCALATION QUEUE':<20}{e_recv:>6}{'-':>9}{'-':>11}{'-':>9}"
        f"{e_acc:>11.3f}"
    )
    print("=" * 96)
    print(f"Total tickets processed: {total_tickets}")

    rows.append(
        {
            "category": "escalation_needs_review",
            "volume_received": e_recv,
            "auto_resolved_count": e["auto_resolved_count"],
            "needs_human_resolution_count": e["needs_human_resolution_count"],
            "gemini_call_failed_count": e["gemini_call_failed_count"],
            "classification_accuracy_on_this_batch": round(e_acc, 6),
        }
    )

    os.makedirs(BATCH_DIR, exist_ok=True)
    pd.DataFrame(rows).to_csv(SUMMARY_CSV, index=False)
    print(f"\nWrote summary to:\n    {SUMMARY_CSV}\n")


def main() -> None:
    np.random.seed(RANDOM_SEED)

    joblib, SentenceTransformer, sr = load_dependencies()

    if not hasattr(sr, "SIMILARITY_THRESHOLD"):
        _fail(
            "suggest_resolution.py has no SIMILARITY_THRESHOLD constant to "
            "import. Expected SIMILARITY_THRESHOLD = 0.35."
        )

    batch = load_batch()
    clf = load_classifier(joblib)
    model = get_embedding_model(SentenceTransformer)

    faiss = sr.load_faiss()
    index, metadata = sr.load_index_and_metadata(faiss)
    gemini_client = sr.setup_gemini_client()

    print(
        f"Loaded batch of {len(batch)} tickets.\n"
        f"FILING_CONFIDENCE_THRESHOLD = {FILING_CONFIDENCE_THRESHOLD} "
        f"(reused from cascade)\n"
        f"SIMILARITY_THRESHOLD = {sr.SIMILARITY_THRESHOLD} (imported from RAG)\n"
    )

    stats, escalation_stats = process(
        batch, clf, model, sr, index, metadata, faiss, gemini_client
    )
    write_summary(stats, escalation_stats, len(batch))


if __name__ == "__main__":
    try:
        main()
    except SystemExit:
        raise
    except Exception as exc:  # noqa: BLE001 - final safety net
        print("\n[ERROR] Unexpected failure while processing the batch:", file=sys.stderr)
        print(f"    {type(exc).__name__}: {exc}", file=sys.stderr)
        print(
            "\nIf this looks like a bug in the adapter code, check the "
            "'# ADJUST:' comments - the exact function names in "
            "suggest_resolution.py / train_embeddings.py may differ from "
            "what this script assumed.",
            file=sys.stderr,
        )
        if os.environ.get("BATCH_DEBUG") == "1":
            traceback.print_exc()
        sys.exit(1)