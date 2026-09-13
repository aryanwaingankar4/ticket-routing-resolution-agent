"""
Config is the single source of truth for every calibrated constant.

These tests exist because this project has repeatedly been bitten by a
threshold or model name drifting out of sync with the value its published
results were measured at. A test that pins the documented number turns that
class of drift into a red build instead of a silent wrong answer.
"""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from src.agent.config import (
    CALIBRATION_PROVENANCE,
    config_fingerprint,
    settings,
)


def test_calibrated_thresholds_match_documented_values():
    """The three headline thresholds, exactly as the README reports them."""
    assert settings.rag.similarity_threshold == 0.67
    assert settings.cascade.confidence_threshold == 0.50
    assert settings.clustering.resolution_similarity_threshold == 0.80


def test_production_embedding_model_is_bge():
    assert settings.models.embedding_model == "BAAI/bge-base-en-v1.5"
    assert settings.models.embedding_dim == 768


def test_clustering_still_on_minilm():
    """Deliberate: BGE is measured for clustering but NOT promoted.

    See README 'Pending' -- no ground-truth check exists for whether BGE
    clustering produces better automation flags, so the production threshold
    stays on the MiniLM-calibrated value.
    """
    assert settings.clustering.clustering_embedding_model == "all-MiniLM-L6-v2"


def test_filing_gate_defaults_off():
    """Only the batch adapter enables it; the live demo never did."""
    assert settings.cascade.filing_gate_enabled is False


def test_settings_are_frozen():
    """Thresholds must not be casually reassignable at runtime."""
    with pytest.raises(ValidationError):
        settings.rag.similarity_threshold = 0.5
    with pytest.raises(ValidationError):
        settings.cascade.confidence_threshold = 0.9


@pytest.mark.parametrize("key", [
    "rag.similarity_threshold",
    "cascade.confidence_threshold",
    "clustering.resolution_similarity_threshold",
])
def test_every_calibrated_value_has_provenance(key):
    """A measured constant must carry the evidence that produced it."""
    assert key in CALIBRATION_PROVENANCE
    assert len(CALIBRATION_PROVENANCE[key]) > 80


def test_fingerprint_is_stable_and_short():
    assert config_fingerprint() == config_fingerprint()
    assert len(config_fingerprint()) == 12


def test_artifact_paths_are_model_aware():
    """Filenames encode model identity, so a MiniLM artifact cannot be
    silently mistaken for a BGE one."""
    assert "bge-base-en-v1-5" in settings.models.faiss_index_name
    assert "bge-base-en-v1-5" in settings.models.metadata_name
    assert "bge-base-en-v1-5" in settings.models.tier2_classifier_name


def test_missing_api_key_raises_not_defaults(monkeypatch):
    """Fail loud. Never fall back to a placeholder or empty key."""
    from src.agent import config as config_mod
    from src.agent.errors import ConfigError

    monkeypatch.setattr(config_mod, "require_gemini_api_key",
                        config_mod.require_gemini_api_key)
    monkeypatch.setenv("GEMINI_API_KEY", "")
    monkeypatch.setattr(
        "dotenv.load_dotenv", lambda *a, **k: False, raising=False
    )
    with pytest.raises(ConfigError, match="GEMINI_API_KEY"):
        config_mod.require_gemini_api_key()
