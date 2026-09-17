"""
THE loader. One place that knows how to load every production artifact.

WHY THIS EXISTS
---------------
There were previously three independent load paths, and they disagreed:

    streamlit_app.load_resources()          BGE encoder, BGE index, BGE clf
    process_ticket_batch.main()             MiniLM encoder, MiniLM clf,
                                            BGE index (via sr)  <-- broken
    test_adversarial.load_pipeline_resources()   BGE, but with its own
                                            re-implemented sync check

The batch path encoded queries with a 384-dim MiniLM model and searched a
768-dim BGE index. It had not been run since before the BGE swap, so the
mismatch was latent rather than visible -- the third occurrence of this
project's recurring stale-artifact bug class.

THREE HARD GUARDS
-----------------
1. index.ntotal == len(metadata)
   Ported from suggest_resolution.py. If these drift, FAISS position i no
   longer maps to the right ticket and retrieval returns confidently wrong
   resolutions with no error.

2. embedder dim == index.d == settings.models.embedding_dim
   Added in Phase 0. This is the guard that makes the batch-script mismatch
   impossible rather than merely fixed: any future encoder/index divergence
   fails at load time with an actionable message instead of surfacing as bad
   results.

3. Tier-1's manifest still matches the dataset it was fitted on
   Added in Phase 3A, when Tier-1 stopped being refitted at startup and
   became a persisted artifact. Persisting it bought a service that does not
   refit per request, at the cost of a new place for a stale artifact to
   hide; this guard is that cost being paid. See load_tier1().

All three raise ArtifactError. The library raises; entry points decide how to
render.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from functools import lru_cache
from typing import Any

from src.agent.config import settings
from src.agent.errors import ArtifactError, ConfigError


@dataclass(frozen=True)
class Artifacts:
    """Everything the pipeline needs, loaded once and verified consistent."""

    embedder: Any
    index: Any
    metadata: list[dict]
    faiss: Any
    tier1_vectorizer: Any
    tier1_classifier: Any
    tier2_classifier: Any
    gemini_client: Any | None = None


def _import_faiss():
    try:
        import faiss
    except ImportError as exc:
        raise ArtifactError(
            "faiss (faiss-cpu) is required but not installed.\n"
            "  Install it with:  pip install faiss-cpu"
        ) from exc
    return faiss


def _load_index_and_metadata(faiss):
    index_path = settings.models.faiss_index_path
    metadata_path = settings.models.metadata_path

    missing = [p for p in (index_path, metadata_path) if not p.is_file()]
    if missing:
        raise ArtifactError(
            "Required RAG artifact(s) not found:\n"
            + "\n".join(f"    - {m}" for m in missing)
            + "\n\n  The vector index has not been built yet.\n"
            "  Build it from the project root with:\n"
            "      python src/rag/build_vector_index.py"
        )

    try:
        index = faiss.read_index(str(index_path))
    except Exception as exc:
        raise ArtifactError(
            f"Failed to read the FAISS index at {index_path}\n"
            f"  ({type(exc).__name__}: {exc})\n"
            "  The file may be corrupted. Rebuild it with:\n"
            "      python src/rag/build_vector_index.py"
        ) from exc

    try:
        with open(metadata_path, "r", encoding="utf-8") as fh:
            metadata = json.load(fh)
    except Exception as exc:
        raise ArtifactError(
            f"Failed to read metadata at {metadata_path}\n"
            f"  ({type(exc).__name__}: {exc})\n"
            "  Rebuild it with:\n"
            "      python src/rag/build_vector_index.py"
        ) from exc

    if not isinstance(metadata, list):
        raise ArtifactError(
            f"{metadata_path} is not a JSON list as expected.\n"
            "  Rebuild the index with:\n"
            "      python src/rag/build_vector_index.py"
        )

    # GUARD 1 -- index/metadata alignment.
    if index.ntotal != len(metadata):
        raise ArtifactError(
            "INDEX / METADATA OUT OF SYNC:\n"
            f"    FAISS index.ntotal = {index.ntotal}\n"
            f"    len(metadata)      = {len(metadata)}\n\n"
            "  These MUST match, or retrieved resolutions would be "
            "misaligned -- wrong answers, silently.\n"
            "  This usually means one file was rebuilt and the other was "
            "not. Rebuild BOTH together with:\n"
            "      python src/rag/build_vector_index.py"
        )

    return index, metadata


def _load_embedder():
    try:
        from sentence_transformers import SentenceTransformer
    except ImportError as exc:
        raise ArtifactError(
            "sentence-transformers is required but not installed.\n"
            "  Install it with:  pip install sentence-transformers"
        ) from exc

    name = settings.models.embedding_model
    try:
        return SentenceTransformer(name)
    except Exception as exc:
        raise ArtifactError(
            f"Failed to load the embedding model '{name}'\n"
            f"  ({type(exc).__name__}: {exc})\n"
            "  Check your internet connection or the local model cache."
        ) from exc


def _assert_dimensions(embedder, index) -> None:
    """GUARD 2 -- encoder and index must agree on dimensionality."""
    expected = settings.models.embedding_dim
    # sentence-transformers renamed this; support both spellings so the guard
    # keeps working across versions rather than silently being skipped.
    if hasattr(embedder, "get_embedding_dimension"):
        encoder_dim = embedder.get_embedding_dimension()
    else:
        encoder_dim = embedder.get_sentence_embedding_dimension()
    index_dim = index.d

    if encoder_dim != expected or index_dim != expected:
        raise ArtifactError(
            "EMBEDDING DIMENSION MISMATCH:\n"
            f"    configured embedding_dim = {expected}"
            f"   ({settings.models.embedding_model})\n"
            f"    encoder produces         = {encoder_dim}\n"
            f"    FAISS index expects      = {index_dim}\n\n"
            "  These MUST all match. A mismatch means the encoder and the "
            "index were built with different models -- searching one with "
            "the other yields meaningless results.\n"
            "  Rebuild the index with the configured model:\n"
            "      python src/rag/build_vector_index.py"
        )


def _load_tier2():
    try:
        import joblib
    except ImportError as exc:
        raise ArtifactError(
            "joblib is required but not installed.\n"
            "  Install it with:  pip install joblib"
        ) from exc

    path = settings.models.tier2_classifier_path
    if not path.is_file():
        raise ArtifactError(
            f"Tier-2 classifier not found:\n    - {path}\n\n"
            "  Train it from the project root with:\n"
            "      python src/classification/train_embeddings.py"
        )
    try:
        return joblib.load(path)
    except Exception as exc:
        raise ArtifactError(
            f"Failed to load the Tier-2 classifier at {path}\n"
            f"  ({type(exc).__name__}: {exc})\n"
            "  Retrain it with:\n"
            "      python src/classification/train_embeddings.py"
        ) from exc


def load_tier1():
    """Load the persisted Tier-1 (TF-IDF + LogReg) bundle and verify it.

    Tier-1 used to be REFITTED here on every startup, over all 4,000 rows of
    synthetic_tickets.csv. That is wrong for a service -- a per-request API
    must not derive a model from raw training data at boot -- so it is now a
    persisted artifact written by src/classification/train_tier1.py.

    Persisting it creates a new place for a stale artifact to hide, which is
    this project's recurring bug class. So the bundle carries a manifest
    recording the dataset it was fitted on, how many rows it saw, and the
    sklearn/vectorizer configuration; train_tier1.verify_manifest() checks
    that against reality here, and anything that has moved raises.

    There is deliberately NO refit-on-miss fallback. A quiet in-process refit
    would be exactly the silent-fallback pattern tests/test_no_silent_fallback
    exists to ban: it would paper over a missing or stale artifact and leave
    the cascade running on a model nobody asked for. Fail loud instead.
    """
    try:
        import joblib
    except ImportError as exc:
        raise ArtifactError(
            "joblib is required but not installed.\n"
            "  Install it with:  pip install joblib"
        ) from exc

    from src.classification.train_tier1 import verify_manifest

    path = settings.models.tier1_classifier_path
    dataset_path = settings.models.dataset_path
    retrain = (
        "  Train it from the project root with:\n"
        "      python src/classification/train_tier1.py"
    )

    if not path.is_file():
        raise ArtifactError(
            f"Tier-1 classifier not found:\n    - {path}\n\n" + retrain
        )

    if not dataset_path.is_file():
        raise ArtifactError(
            f"Dataset not found:\n    - {dataset_path}\n\n"
            "  Tier-1 cannot be verified against the data it was fitted "
            "on.\n"
            "  Generate it from the project root with:\n"
            "      python data/generate_dataset.py"
        )

    try:
        bundle = joblib.load(path)
    except Exception as exc:
        raise ArtifactError(
            f"Failed to load the Tier-1 classifier at {path}\n"
            f"  ({type(exc).__name__}: {exc})\n" + retrain
        ) from exc

    if not isinstance(bundle, dict) or not {
        "vectorizer", "classifier"
    } <= set(bundle):
        raise ArtifactError(
            f"{path.name} is not a Tier-1 bundle as expected "
            "(needs 'vectorizer', 'classifier' and 'manifest').\n" + retrain
        )

    problems = verify_manifest(
        bundle.get("manifest"),
        bundle["vectorizer"],
        bundle["classifier"],
        dataset_path=dataset_path,
    )
    if problems:
        raise ArtifactError(
            "STALE TIER-1 ARTIFACT:\n"
            + "\n".join(f"    - {problem}" for problem in problems)
            + "\n\n  The persisted model no longer matches what it was "
            "derived from.\n  Using it would route tickets with a model "
            "that is wrong for its\n  context but internally consistent -- "
            "wrong answers, no error.\n" + retrain
        )

    return bundle["vectorizer"], bundle["classifier"]


def load_drift_reference(artifacts: Artifacts | None = None):
    """Load the drift reference and verify it matches the live encoder/index.

    Drift Signal A compares live top-1 similarities against similarities
    measured when the reference was built. Against a different encoder or a
    rebuilt index those numbers are not comparable, and every p-value would
    be wrong with no error -- so the reference carries the embedding model,
    dimension, index size and index file hash, and all are checked here.

    Pass `artifacts` to also check index.ntotal against the loaded index.
    Without it, the index file hash still covers a rebuild.
    """
    from pydantic import ValidationError

    from src.agent.drift import DriftReference
    # A generic chunked file hash despite the name; reused rather than
    # re-implemented so there is one definition of "this file's hash".
    from src.classification.train_tier1 import dataset_sha256 as file_sha256

    path = settings.drift.reference_path
    rebuild = (
        "  Build it from the project root with:\n"
        "      python src/experiments/build_drift_reference.py"
    )
    if not path.is_file():
        raise ArtifactError(
            f"Drift reference not found:\n    - {path}\n\n" + rebuild
        )

    try:
        with open(path, "r", encoding="utf-8") as fh:
            reference = DriftReference.model_validate(json.load(fh))
    except (ValueError, ValidationError) as exc:
        raise ArtifactError(
            f"Drift reference at {path} is unreadable or malformed\n"
            f"  ({type(exc).__name__}: {exc})\n" + rebuild
        ) from exc

    index_path = settings.models.faiss_index_path
    if not index_path.is_file():
        raise ArtifactError(
            f"FAISS index not found:\n    - {index_path}\n"
            "  The drift reference cannot be verified without it.\n"
            "  Build it with:  python src/rag/build_vector_index.py"
        )

    reference.check_compatible(
        embedding_model=settings.models.embedding_model,
        embedding_dim=settings.models.embedding_dim,
        index_ntotal=None if artifacts is None else artifacts.index.ntotal,
        index_sha256=file_sha256(index_path),
    )
    return reference


def build_gemini_client():
    from src.agent.config import require_gemini_api_key

    try:
        from google import genai
    except ImportError as exc:
        raise ArtifactError(
            "The Google Gen AI SDK is required but not installed.\n"
            "  Install it with:  pip install google-genai\n"
            "  (This is the current unified SDK; 'google-generativeai' is "
            "deprecated.)"
        ) from exc

    api_key = require_gemini_api_key()
    try:
        return genai.Client(api_key=api_key)
    except Exception as exc:
        # Never echo the key itself.
        raise ConfigError(
            f"Failed to initialise the Gemini client "
            f"({type(exc).__name__}: {exc}).\n"
            "  Check that GEMINI_API_KEY is a valid key from "
            "https://aistudio.google.com/apikey"
        ) from exc


@lru_cache(maxsize=2)
def load_artifacts(require_gemini: bool = False) -> Artifacts:
    """Load and verify every production artifact.

    Cached, so repeated calls in one process (Streamlit reruns, a pytest
    session, a batch loop) reuse the same loaded models.

    `require_gemini=False` is deliberate for evaluation paths: the escalation
    decision is fully determined before any LLM call, so routing tests and
    calibration sweeps need neither an API key nor quota.
    """
    faiss = _import_faiss()
    index, metadata = _load_index_and_metadata(faiss)
    embedder = _load_embedder()

    _assert_dimensions(embedder, index)

    tier2 = _load_tier2()
    tier1_vectorizer, tier1_classifier = load_tier1()
    client = build_gemini_client() if require_gemini else None

    return Artifacts(
        embedder=embedder,
        index=index,
        metadata=metadata,
        faiss=faiss,
        tier1_vectorizer=tier1_vectorizer,
        tier1_classifier=tier1_classifier,
        tier2_classifier=tier2,
        gemini_client=client,
    )
