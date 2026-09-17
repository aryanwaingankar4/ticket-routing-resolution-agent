"""
Typed, frozen configuration -- the single source of truth for every
calibrated constant in this project.

WHY THIS EXISTS
---------------
Before this module, the cascade confidence threshold was declared three times
under three different names (streamlit_app.CASCADE_CONFIDENCE_THRESHOLD,
test_adversarial_escalation.CASCADE_CONFIDENCE_THRESHOLD,
process_ticket_batch.FILING_CONFIDENCE_THRESHOLD), the RAG similarity
threshold had two `getattr(sr, "SIMILARITY_THRESHOLD", 0.35)` fallbacks that
would silently revert the human-escalation gate to a dead MiniLM-era value,
and three separate loaders disagreed about which model artifacts to read.

That duplication is the direct cause of this project's recurring bug class
(stale constants surviving a model swap). One frozen, typed definition makes
the class structurally impossible rather than merely fixed.

FROZEN BY DESIGN
----------------
Every model here is immutable. The project's working convention is that
calibrated thresholds are measured, evidence-backed constants -- never
re-derived casually and never exposed as trivially-overridable CLI flags,
since that would undermine the claim that they were measured rather than
guessed. Attempting to mutate one raises at runtime.

Every calibrated value carries its evidence in CALIBRATION_PROVENANCE below,
so the justification travels with the number instead of living only in the
README.
"""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path

from pydantic import BaseModel, ConfigDict

from src.agent.errors import ConfigError

# --------------------------------------------------------------------------- #
# Paths. Project root is two directories up from this file
# (src/agent/ -> root),
# matching the convention used by every other script in this project.
# --------------------------------------------------------------------------- #
PROJECT_ROOT: Path = Path(__file__).resolve().parents[2]
DATA_DIR: Path = PROJECT_ROOT / "data"
MODELS_DIR: Path = PROJECT_ROOT / "models"


class _Frozen(BaseModel):
    """Base for every config section: immutable and strict about typos."""

    model_config = ConfigDict(
        frozen=True,
        extra="forbid",
        # Allow natural field names like `embedding_model` without pydantic
        # warning about its reserved `model_` namespace.
        protected_namespaces=(),
    )


class ModelsConfig(_Frozen):
    """Production model identities and the artifacts they produce.

    `embedding_dim` is not decoration: artifacts.py asserts it against both the
    loaded SentenceTransformer and the FAISS index at load time. That check is
    what makes a MiniLM-encoder-against-BGE-index mismatch impossible.
    """

    embedding_model: str = "BAAI/bge-base-en-v1.5"
    embedding_dim: int = 768
    gemini_model: str = "gemini-flash-lite-latest"

    # Model-aware filenames. A MiniLM artifact can never be silently mistaken
    # for a BGE one, because the identity is in the filename.
    faiss_index_name: str = "ticket_index_bge-base-en-v1-5.faiss"
    metadata_name: str = "ticket_metadata_bge-base-en-v1-5.json"
    tier2_classifier_name: str = "ticket_classifier_bge-base-en-v1-5.joblib"

    # Tier-1 is TF-IDF + LogReg, so its identity is not an embedding model --
    # it is the DATASET it was fitted on plus the vectorizer/sklearn
    # configuration. Those cannot go in the filename, so they travel in the
    # artifact's own manifest, which artifacts._load_tier1() verifies against
    # the live dataset at load time.
    tier1_classifier_name: str = "tier1_tfidf_logreg.joblib"

    # Tier-1 is TF-IDF, so it is embedding-model independent, but it is trained
    # from this dataset and must stay aligned with it.
    dataset_name: str = "synthetic_tickets.csv"

    @property
    def faiss_index_path(self) -> Path:
        return DATA_DIR / self.faiss_index_name

    @property
    def metadata_path(self) -> Path:
        return DATA_DIR / self.metadata_name

    @property
    def tier2_classifier_path(self) -> Path:
        return MODELS_DIR / self.tier2_classifier_name

    @property
    def tier1_classifier_path(self) -> Path:
        return MODELS_DIR / self.tier1_classifier_name

    @property
    def dataset_path(self) -> Path:
        return DATA_DIR / self.dataset_name


class CascadeConfig(_Frozen):
    """Tier-1 -> Tier-2 routing gate.

    Depends only on Tier-1 (TF-IDF) confidence, so it is independent of which
    embedding model Tier-2 uses -- verified by code review during the BGE swap,
    which is why it did not require recalibration then.
    """

    confidence_threshold: float = 0.50

    # process_ticket_batch.py applies a second gate on FINAL classification
    # confidence before filing a ticket; streamlit_app.py and the adversarial
    # test do not. Kept configurable so each consumer preserves its documented
    # behaviour rather than one silently inheriting the other's.
    filing_gate_enabled: bool = False
    filing_confidence_threshold: float = 0.50


class RAGConfig(_Frozen):
    """Retrieval gate: below this top-1 similarity, the LLM is never called."""

    similarity_threshold: float = 0.67
    top_k: int = 5


class ClusteringConfig(_Frozen):
    """Resolution-clustering threshold for automation flagging.

    Deliberately still calibrated on MiniLM. BGE has been measured for this
    task (pooled cliff 0.90) but NOT promoted, because no ground-truth check
    exists for whether BGE clustering produces better automation flags. See
    README "Pending".
    """

    resolution_similarity_threshold: float = 0.80
    clustering_embedding_model: str = "all-MiniLM-L6-v2"


class GeminiConfig(_Frozen):
    """LLM call behaviour. Free tier is 15 req/min, 500/day."""

    call_delay_sec: float = 4.5
    max_retries: int = 3
    backoff_base_sec: float = 2.0


class ConformalConfig(_Frozen):
    """Split conformal prediction -- MEASUREMENT ONLY at present.

    `enabled` defaults to False, so the live gates remain the calibrated
    thresholds (cascade 0.50, RAG 0.67) and pipeline results stay
    byte-identical to the Phase 0 goldens. Conformal is fitted and evaluated
    by src/experiments/calibrate_conformal.py; promotion to a production gate
    is a separate decision requiring its own evidence, exactly as with the
    BGE clustering threshold that was measured but deliberately not promoted.

    `alpha` is the target error rate: the guarantee is that the true category
    lies in the prediction set with probability at least 1 - alpha, PROVIDED
    calibration and deployment data are exchangeable. That proviso is the
    whole question here and is measured rather than assumed.
    """

    enabled: bool = False
    alpha: float = 0.10
    score_function: str = "lac"
    mondrian: bool = False
    artifact_name: str = "conformal_calibration_bge-base-en-v1-5.json"

    @property
    def artifact_path(self) -> Path:
        return DATA_DIR / self.artifact_name


class DriftConfig(_Frozen):
    """Drift detection over persisted decision records -- MEASUREMENT ONLY.

    `enabled` defaults to False and gates nothing: no routing decision reads
    any value here. The detector (src/agent/drift.py) is pure functions over
    records; its evaluation is Phase 4B.

    Deliberately absent: a window size and an alarm threshold. Naming either
    before the null false-alarm rate is measured would be a hand-tuned
    threshold wearing a lab coat -- exactly what the Phase 4 gate forbids.

    `alpha` is the per-ticket novelty level for Signal A. The window false-
    alarm rate equals alpha only MARGINALLY over calibration draws; for the
    single fixed n=175 reference it is itself random around alpha. So the
    detector also reports a calibration-conditional test against an upper
    bound alpha' held with confidence 1 - `conditional_delta`.
    """

    enabled: bool = False
    alpha: float = 0.10
    conditional_delta: float = 0.10
    reference_name: str = "drift_reference_bge-base-en-v1-5.json"

    @property
    def reference_path(self) -> Path:
        return DATA_DIR / self.reference_name


class Settings(_Frozen):
    models: ModelsConfig = ModelsConfig()
    cascade: CascadeConfig = CascadeConfig()
    rag: RAGConfig = RAGConfig()
    clustering: ClusteringConfig = ClusteringConfig()
    gemini: GeminiConfig = GeminiConfig()
    conformal: ConformalConfig = ConformalConfig()
    drift: DriftConfig = DriftConfig()

    seed: int = 42


settings = Settings()


# --------------------------------------------------------------------------- #
# Provenance. Every calibrated number must be traceable to the evidence that
# produced it; test_config.py enforces that this mapping stays complete.
# --------------------------------------------------------------------------- #
CALIBRATION_PROVENANCE: dict[str, str] = {
    "rag.similarity_threshold": (
        "0.67 -- README 'RAG Similarity Threshold Recalibration'. Derived by "
        "intersecting the 9-ticket adversarial safe range (0.6196, 0.6704] "
        "with the minimum-OOD-leakage point in that range (13.3% at 0.67 vs "
        "37.8% at the prior provisional 0.65). Confirmed 9/9 on the "
        "adversarial set before adoption."
    ),
    "cascade.confidence_threshold": (
        "0.50 -- README 'Cascade Confidence Calibration'. Third calibration "
        "attempt (175 Gemini-paraphrased in-domain tickets); the first two "
        "(in-distribution split, 35 hand-written tickets) were rejected as "
        "misleading. Emerges at a realistic 70-80% target-accuracy bar; at a "
        "90% bar the cascade collapses to pure Tier-2."
    ),
    "clustering.resolution_similarity_threshold": (
        "0.80 -- README 'Resolution-clustering calibration'. Last threshold "
        "with pairwise precision exactly 1.0000 against scenario_id ground "
        "truth; precision collapses at 0.75. Chosen over the recall-better "
        "0.75 because a false 'these tickets share a fix' claim costs more "
        "than a missed automation opportunity."
    ),
    "conformal.alpha": (
        "0.10 -- nominal target error rate. MEASUREMENT ONLY; conformal does "
        "not gate production (settings.conformal.enabled is False). Measured "
        "on this project's data, the guarantee transfers for Tier-2 (BGE) "
        "but NOT for Tier-1 (TF-IDF): benchmark coverage gaps of -0.011 and "
        "-0.233 respectively at this alpha, against a 2sd noise band of "
        "0.045. See README 'Conformal Prediction'."
    ),
    "drift.alpha": (
        "0.10 -- nominal per-ticket novelty level for drift Signal A, chosen "
        "to match conformal.alpha. MEASUREMENT ONLY; drift gates nothing "
        "(settings.drift.enabled is False). The window false-alarm rate is "
        "alpha only marginally over calibration draws, not conditionally on "
        "the fixed 175-ticket reference, so no operating point is claimed "
        "until Phase 4B measures the null false-alarm rate."
    ),
    "models.embedding_dim": (
        "768 -- BAAI/bge-base-en-v1.5. Asserted at load time against both the "
        "SentenceTransformer and the FAISS index to prevent the "
        "MiniLM-encoder-against-BGE-index class of mismatch."
    ),
}


def config_fingerprint() -> str:
    """Stable short hash of the active configuration.

    Stamped onto every PipelineResult so a stored result always carries the
    exact thresholds and model identity that produced it. This is what makes a
    stale-artifact mismatch detectable after the fact rather than silent.
    """
    payload = settings.model_dump(mode="json")
    blob = json.dumps(payload, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(blob.encode("utf-8")).hexdigest()[:12]


def require_gemini_api_key() -> str:
    """Return GEMINI_API_KEY or raise. Never defaults, never silently empty.

    Loads .env if python-dotenv is available, matching how every other script
    in this project sources the key.
    """
    try:
        from dotenv import load_dotenv

        load_dotenv(PROJECT_ROOT / ".env")
    except ImportError:  # pragma: no cover - dotenv is a declared dependency
        pass

    key = os.getenv("GEMINI_API_KEY", "").strip()
    if not key:
        raise ConfigError(
            "GEMINI_API_KEY is not set.\n"
            f"  Add it to {PROJECT_ROOT / '.env'} as:\n"
            "      GEMINI_API_KEY=your_key_here\n"
            "  Get a free key from https://aistudio.google.com/apikey"
        )
    return key
