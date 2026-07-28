# src/rag/build_vector_index.py
"""
Build a local FAISS vector index over all IT-support tickets for the RAG layer.

This script runs ONCE (or whenever data/synthetic_tickets.csv changes). It:
  1. Loads and validates the ticket CSV.
  2. Builds embedding input text as (title + " " + description), consistent
     with the rest of the project.
  3. Embeds all tickets with sentence-transformers "all-MiniLM-L6-v2"
     (reusing the project's embedding cache when the row count matches).
  4. Builds a FAISS IndexFlatIP over L2-normalized embeddings (== cosine
     similarity search).
  5. Saves an index-aligned metadata JSON so FAISS integer indices map back
     to the correct ticket resolutions.
  6. Runs a built-in self-retrieval sanity check before finishing.

Run from the project root:
    python src/rag/build_vector_index.py
"""

import os
import sys
import json
import random

import numpy as np


# ---------------------------------------------------------------------------
# Reproducibility: keep the project's seeding convention consistent even
# though embedding + FAISS indexing here are deterministic.
# ---------------------------------------------------------------------------
random.seed(42)
np.random.seed(42)


# ---------------------------------------------------------------------------
# Path resolution convention (identical to every other script in this
# project): resolve the project root as TWO directories up from THIS file's
# own location, never relative to the invocation directory. This makes the
# script work regardless of the current working directory, including on
# Windows / PowerShell inside a venv.
#
#   this file:  <root>/src/rag/build_vector_index.py
#   dirname 1:  <root>/src/rag
#   dirname 2:  <root>/src
#   dirname 3:  <root>          <- project root
# ---------------------------------------------------------------------------
_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(os.path.dirname(_THIS_DIR))

DATA_DIR = os.path.join(PROJECT_ROOT, "data")
CSV_PATH = os.path.join(DATA_DIR, "synthetic_tickets.csv")
EMBEDDINGS_CACHE_PATH = os.path.join(DATA_DIR, "ticket_embeddings.npy")
INDEX_PATH = os.path.join(DATA_DIR, "ticket_index.faiss")
METADATA_PATH = os.path.join(DATA_DIR, "ticket_metadata.json")

MODEL_NAME = "all-MiniLM-L6-v2"

REQUIRED_COLUMNS = ["id", "title", "description", "category", "resolution", "priority"]
NO_NAN_COLUMNS = ["title", "description", "category", "resolution"]
MIN_CATEGORIES = 7


# ---------------------------------------------------------------------------
# Network / offline error detection. Same "network_markers" pattern used in
# train_embeddings.py: the first run downloads the model from huggingface.co,
# so we translate connection/timeout/proxy failures into an actionable
# message instead of a raw stack trace.
# ---------------------------------------------------------------------------
NETWORK_MARKERS = [
    "connection",
    "connectionerror",
    "timed out",
    "timeout",
    "proxy",
    "proxyerror",
    "max retries",
    "failed to establish",
    "huggingface.co",
    "name resolution",
    "temporary failure in name resolution",
    "network is unreachable",
    "getaddrinfo",
    "ssl",
    "certificate",
    "offline",
    "couldn't connect",
    "could not connect",
    "unable to load",
    "no address associated",
]


def _looks_like_network_error(exc):
    """Return True if the exception text matches a known network failure marker."""
    text = f"{type(exc).__name__} {exc}".lower()
    return any(marker in text for marker in NETWORK_MARKERS)


def _fail(message, code=1):
    """Print a clean error message (no traceback) and exit."""
    print(f"\n[ERROR] {message}", file=sys.stderr)
    sys.exit(code)


# ---------------------------------------------------------------------------
# Step 1: Load + validate the dataset.
# ---------------------------------------------------------------------------
def load_and_validate_dataset():
    try:
        import pandas as pd
    except ImportError:
        _fail(
            "pandas is required but not installed.\n"
            "Install it with:\n\n    pip install pandas\n"
        )

    if not os.path.isfile(CSV_PATH):
        _fail(
            f"Could not find the ticket dataset.\n"
            f"Expected it at: {CSV_PATH}\n"
            f"Make sure data/synthetic_tickets.csv exists at the project root."
        )

    try:
        df = pd.read_csv(CSV_PATH)
    except Exception as exc:  # noqa: BLE001 - want a clean message, not a traceback
        _fail(f"Failed to read {CSV_PATH}: {type(exc).__name__}: {exc}")

    # Required columns present.
    missing = [c for c in REQUIRED_COLUMNS if c not in df.columns]
    if missing:
        _fail(
            f"{CSV_PATH} is missing required column(s): {missing}\n"
            f"Expected columns: {REQUIRED_COLUMNS}\n"
            f"Found columns:    {list(df.columns)}"
        )

    if len(df) == 0:
        _fail(f"{CSV_PATH} contains 0 rows - nothing to index.")

    # No NaNs in the columns we depend on.
    for col in NO_NAN_COLUMNS:
        n_nan = int(df[col].isna().sum())
        if n_nan > 0:
            _fail(
                f"Column '{col}' contains {n_nan} missing (NaN) value(s) in "
                f"{CSV_PATH}. Clean the dataset before building the index."
            )

    # At least 7 categories represented.
    n_categories = df["category"].nunique()
    if n_categories < MIN_CATEGORIES:
        _fail(
            f"Expected at least {MIN_CATEGORIES} categories in the dataset, "
            f"but found only {n_categories}: {sorted(df['category'].unique())}"
        )

    # Stable ordering by CSV position. reset_index guarantees a clean 0..N-1
    # integer index so that DataFrame row i, embedding row i, FAISS index i,
    # and metadata position i all refer to the SAME ticket.
    df = df.reset_index(drop=True)

    print(f"[data] Loaded {len(df)} tickets from {CSV_PATH}")
    print(f"[data] Categories ({n_categories}): {sorted(df['category'].unique())}")
    return df


# ---------------------------------------------------------------------------
# Step 2: Build the embedding input text EXACTLY as elsewhere in the project:
#         title + " " + description.
# ---------------------------------------------------------------------------
def build_embedding_texts(df):
    texts = (df["title"].astype(str) + " " + df["description"].astype(str)).tolist()
    return texts


# ---------------------------------------------------------------------------
# Step 3: Embed with all-MiniLM-L6-v2, reusing the project's embedding cache.
#
# Cache policy (same as train_embeddings.py): if data/ticket_embeddings.npy
# exists AND its row count matches the current CSV, reuse it (cache HIT).
# Otherwise recompute (cache MISS) and save. A corrupted/unreadable cache is
# treated as a MISS rather than a crash.
# ---------------------------------------------------------------------------
def _try_load_cached_embeddings(expected_rows):
    """Return cached embeddings if valid and row-count-matching, else None."""
    if not os.path.isfile(EMBEDDINGS_CACHE_PATH):
        return None
    try:
        cached = np.load(EMBEDDINGS_CACHE_PATH)
    except Exception as exc:  # noqa: BLE001 - corrupted cache should not crash us
        print(
            f"[cache WARN] {EMBEDDINGS_CACHE_PATH} exists but could not be read "
            f"({type(exc).__name__}: {exc}). Recomputing embeddings."
        )
        return None

    if cached.ndim != 2 or cached.shape[0] != expected_rows:
        print(
            f"[cache MISS] Cached embeddings shape {getattr(cached, 'shape', None)} "
            f"does not match current dataset ({expected_rows} rows). Recomputing."
        )
        return None

    return cached


def get_embeddings(texts):
    expected_rows = len(texts)

    cached = _try_load_cached_embeddings(expected_rows)
    if cached is not None:
        print(
            f"[cache HIT] Reusing embeddings from {EMBEDDINGS_CACHE_PATH} "
            f"(shape {cached.shape}) - skipping model load and encoding."
        )
        return cached.astype(np.float32, copy=False)

    print("[cache MISS] No valid cached embeddings found - computing them now.")

    # Import sentence-transformers lazily so a missing dep gives a clean message.
    try:
        from sentence_transformers import SentenceTransformer
    except ImportError:
        _fail(
            "sentence-transformers is required but not installed.\n"
            "Install it with:\n\n    pip install sentence-transformers\n"
        )

    # Load the model. First run downloads from huggingface.co; translate
    # network failures into an actionable message.
    try:
        print(f"[model] Loading sentence-transformers model '{MODEL_NAME}' ...")
        model = SentenceTransformer(MODEL_NAME)
    except Exception as exc:  # noqa: BLE001
        if _looks_like_network_error(exc):
            _fail(
                f"Failed to download/load the model '{MODEL_NAME}' due to what "
                f"looks like a network problem:\n"
                f"    {type(exc).__name__}: {exc}\n\n"
                f"This script needs internet access the FIRST time only, to "
                f"download the model from huggingface.co.\n"
                f"Please check:\n"
                f"  - your internet connection is up,\n"
                f"  - any corporate proxy/VPN is configured (HTTP_PROXY / "
                f"HTTPS_PROXY),\n"
                f"  - huggingface.co is reachable.\n"
                f"Once the model is cached locally, future runs work offline."
            )
        _fail(f"Failed to load model '{MODEL_NAME}': {type(exc).__name__}: {exc}")

    # Encode. Do NOT normalize here - we normalize explicitly in the FAISS step
    # so the saved cache stays consistent with train_embeddings.py (raw
    # embeddings), and so the normalization used for cosine search is visible
    # and auditable at the indexing site.
    try:
        print(f"[model] Encoding {expected_rows} ticket texts ...")
        embeddings = model.encode(
            texts,
            batch_size=64,
            show_progress_bar=True,
            convert_to_numpy=True,
        )
    except Exception as exc:  # noqa: BLE001
        if _looks_like_network_error(exc):
            _fail(
                f"Encoding failed with what looks like a network error:\n"
                f"    {type(exc).__name__}: {exc}\n"
                f"Check your internet/proxy settings and retry."
            )
        _fail(f"Encoding failed: {type(exc).__name__}: {exc}")

    embeddings = np.asarray(embeddings, dtype=np.float32)

    # Persist the cache for reuse by this and the classification scripts.
    try:
        os.makedirs(DATA_DIR, exist_ok=True)
        np.save(EMBEDDINGS_CACHE_PATH, embeddings)
        print(f"[cache] Saved embeddings to {EMBEDDINGS_CACHE_PATH} (shape {embeddings.shape})")
    except Exception as exc:  # noqa: BLE001 - failing to cache is non-fatal
        print(
            f"[cache WARN] Could not save embeddings cache to "
            f"{EMBEDDINGS_CACHE_PATH} ({type(exc).__name__}: {exc}). Continuing."
        )

    return embeddings


# ---------------------------------------------------------------------------
# Step 4: Build the FAISS index.
# ---------------------------------------------------------------------------
def build_faiss_index(embeddings, faiss):
    """
    Build an IndexFlatIP over L2-normalized embeddings.

    WHY IndexFlatIP on normalized vectors?
      For unit-length (L2-normalized) vectors a and b, the inner product
      a . b equals the cosine similarity cos(a, b). So searching an inner-
      product index over normalized embeddings is EXACTLY cosine-similarity
      search. IndexFlatIP is a brute-force exact index: at 1000 vectors it is
      instant, returns exact (not approximate) nearest neighbours, needs no
      training, and is trivial to explain in a viva as
      "cosine similarity via normalized inner product". More complex indices
      (IVF/HNSW) only pay off at much larger scales and add approximation
      error we do not want here.
    """
    embeddings = np.ascontiguousarray(embeddings.astype(np.float32))

    # Normalize IN PLACE so ||v|| == 1 for every row; faiss.normalize_L2
    # mutates the array it is given.
    faiss.normalize_L2(embeddings)

    dim = embeddings.shape[1]
    index = faiss.IndexFlatIP(dim)
    index.add(embeddings)

    print(f"[faiss] Built IndexFlatIP: dim={dim}, ntotal={index.ntotal}")
    # Return the normalized embeddings too - the sanity check reuses them so
    # the query vector matches exactly what was indexed.
    return index, embeddings


# ---------------------------------------------------------------------------
# Step 5: Build index-aligned metadata.
#
# FAISS returns integer positions on search, not ticket data. metadata[i]
# MUST describe the exact ticket stored at FAISS index i. Because df, the
# embeddings, the FAISS add order, and this list all follow the same 0..N-1
# row order, position i is consistent everywhere. A single off-by-one here
# would silently return WRONG resolutions with no error - hence the explicit
# assertion in main().
# ---------------------------------------------------------------------------
def build_metadata(df):
    metadata = []
    for i in range(len(df)):
        row = df.iloc[i]
        metadata.append(
            {
                "faiss_index": i,  # explicit position for debuggability
                "id": _json_safe(row["id"]),
                "title": str(row["title"]),
                "description": str(row["description"]),
                "category": str(row["category"]),
                "resolution": str(row["resolution"]),
                "priority": _json_safe(row["priority"]),
            }
        )
    return metadata


def _json_safe(value):
    """Convert numpy scalar types to native Python types for JSON serialization."""
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return float(value)
    if value is None:
        return None
    # Keep ints/strings as-is; fall back to str for anything exotic.
    if isinstance(value, (int, float, str, bool)):
        return value
    return str(value)


# ---------------------------------------------------------------------------
# Step 9: Self-retrieval sanity check.
#
# Query the index with ticket 0's OWN (normalized) embedding. The top hit must
# be index 0 with similarity ~1.0. If not, the indexing pipeline is broken and
# we must NOT let a bad index reach the RAG query layer.
# ---------------------------------------------------------------------------
def run_sanity_check(index, normalized_embeddings):
    probe_idx = 0
    query = normalized_embeddings[probe_idx : probe_idx + 1]  # shape (1, dim)
    query = np.ascontiguousarray(query.astype(np.float32))

    scores, indices = index.search(query, k=1)
    top_idx = int(indices[0][0])
    top_score = float(scores[0][0])

    passed = (top_idx == probe_idx) and (abs(top_score - 1.0) < 1e-3)

    print("\n[sanity] Self-retrieval check (query = ticket 0's own embedding):")
    print(f"[sanity]   expected top index = {probe_idx}, got {top_idx}")
    print(f"[sanity]   top similarity      = {top_score:.6f} (expected ~1.0)")
    print(f"[sanity]   RESULT             = {'PASS' if passed else 'FAIL'}")

    return passed, top_idx, top_score


# ---------------------------------------------------------------------------
# Main.
# ---------------------------------------------------------------------------
def main():
    print("=" * 70)
    print("RAG index builder - src/rag/build_vector_index.py")
    print("=" * 70)
    print(f"[paths] project root : {PROJECT_ROOT}")

    # Import faiss up front with a clean install message if missing.
    try:
        import faiss
    except ImportError:
        _fail(
            "faiss (faiss-cpu) is required but not installed.\n"
            "Install it with:\n\n    pip install faiss-cpu\n"
        )

    # 1. Load + validate.
    df = load_and_validate_dataset()

    # 2. Build embedding texts.
    texts = build_embedding_texts(df)
    assert len(texts) == len(df), "Internal error: text count != row count"

    # 3. Embed (with cache).
    embeddings = get_embeddings(texts)
    if embeddings.shape[0] != len(df):
        _fail(
            f"Embedding row count ({embeddings.shape[0]}) does not match "
            f"dataset row count ({len(df)}). The cache may be stale - delete "
            f"{EMBEDDINGS_CACHE_PATH} and re-run."
        )

    # 4. Build FAISS index (normalizes embeddings in place -> cosine search).
    index, normalized_embeddings = build_faiss_index(embeddings, faiss)

    # 5. Build index-aligned metadata.
    metadata = build_metadata(df)

    # ---- CRITICAL alignment assertion -------------------------------------
    # metadata position i MUST correspond to FAISS index i. If these ever
    # differ, every future RAG query could return the wrong resolution with
    # no error thrown - so we fail loudly here instead.
    if len(metadata) != index.ntotal:
        _fail(
            f"FATAL METADATA/INDEX MISALIGNMENT: len(metadata)={len(metadata)} "
            f"but index.ntotal={index.ntotal}. These MUST be equal, or FAISS "
            f"indices would map to the wrong tickets. Aborting - not writing "
            f"any output files."
        )
    assert len(metadata) == index.ntotal, "alignment invariant violated"
    print(f"[align] OK: len(metadata) == index.ntotal == {index.ntotal}")

    # 9. Sanity check BEFORE writing anything to disk, so a broken index is
    #    never persisted for the downstream RAG layer to consume.
    passed, top_idx, top_score = run_sanity_check(index, normalized_embeddings)
    if not passed:
        _fail(
            f"Self-retrieval sanity check FAILED (top_idx={top_idx}, "
            f"top_score={top_score:.6f}). The index does not return a ticket "
            f"as its own nearest neighbour, which means the indexing pipeline "
            f"is broken. Refusing to save a bad index. Investigate embedding/"
            f"normalization/add order before retrying."
        )

    # 6. Persist outputs (index + metadata). Embeddings cache already saved.
    try:
        os.makedirs(DATA_DIR, exist_ok=True)
        faiss.write_index(index, INDEX_PATH)
        print(f"[write] FAISS index -> {INDEX_PATH}")
    except Exception as exc:  # noqa: BLE001
        _fail(f"Failed to write FAISS index to {INDEX_PATH}: "
              f"{type(exc).__name__}: {exc}")

    try:
        with open(METADATA_PATH, "w", encoding="utf-8") as f:
            json.dump(metadata, f, ensure_ascii=False, indent=2)
        print(f"[write] Metadata    -> {METADATA_PATH}")
    except Exception as exc:  # noqa: BLE001
        _fail(f"Failed to write metadata to {METADATA_PATH}: "
              f"{type(exc).__name__}: {exc}")

    # 10. Final summary.
    dim = int(embeddings.shape[1])
    print("\n" + "=" * 70)
    print("BUILD COMPLETE")
    print("=" * 70)
    print(f"  Total tickets indexed : {index.ntotal}")
    print(f"  Embedding dimension   : {dim}")
    print(f"  Output files:")
    print(f"    - FAISS index       : {INDEX_PATH}")
    print(f"    - Metadata (aligned): {METADATA_PATH}")
    print(f"    - Embeddings cache  : {EMBEDDINGS_CACHE_PATH}")
    print(f"  Sanity check          : PASS "
          f"(top_idx={top_idx}, sim={top_score:.6f})")
    print("=" * 70)


if __name__ == "__main__":
    main()
