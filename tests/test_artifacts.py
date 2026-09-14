"""
Artifact loading guards.

These cover the specific failure mode that has hit this project three times:
an artifact surviving a model swap un-migrated, producing wrong results
without any error. The guards must fail LOUDLY, and these tests prove they
actually do rather than merely being present.
"""

from __future__ import annotations

import pytest

from src.agent.artifacts import _assert_dimensions, load_artifacts
from src.agent.config import settings
from src.agent.errors import ArtifactError

pytestmark = pytest.mark.slow


def _encoder_dim(embedder):
    if hasattr(embedder, "get_embedding_dimension"):
        return embedder.get_embedding_dimension()
    return embedder.get_sentence_embedding_dimension()


def test_index_and_metadata_are_in_sync(artifacts):
    """If these drift, FAISS position i maps to the wrong ticket and
    retrieval returns confidently wrong resolutions."""
    assert artifacts.index.ntotal == len(artifacts.metadata)
    assert artifacts.index.ntotal == 4000


def test_encoder_and_index_dimensions_agree(artifacts):
    """The guard that makes the MiniLM-encoder-vs-BGE-index bug impossible."""
    expected = settings.models.embedding_dim
    assert _encoder_dim(artifacts.embedder) == expected
    assert artifacts.index.d == expected


def test_dimension_guard_actually_raises(artifacts):
    """Prove the guard fires rather than just existing.

    Simulates precisely the process_ticket_batch.py bug: a 384-dim MiniLM
    encoder searching the 768-dim BGE index.
    """
    class FakeMiniLM:
        def get_embedding_dimension(self):
            return 384

        def get_sentence_embedding_dimension(self):
            return 384

    with pytest.raises(ArtifactError, match="DIMENSION MISMATCH"):
        _assert_dimensions(FakeMiniLM(), artifacts.index)


def test_loader_is_cached(artifacts):
    """One load per session; Streamlit reruns must not reload BGE."""
    assert load_artifacts(require_gemini=False) is artifacts


def test_no_gemini_client_when_not_required(artifacts):
    """Evaluation paths need neither an API key nor quota."""
    assert artifacts.gemini_client is None


def test_metadata_entries_have_expected_shape(artifacts):
    entry = artifacts.metadata[0]
    for key in ("id", "title", "description", "category", "resolution"):
        assert key in entry, f"metadata entry missing '{key}'"


# --------------------------------------------------------------------------- #
# Tier-1 persistence (Phase 3A).                                               #
#                                                                              #
# Tier-1 is no longer refitted at startup; it is loaded from a persisted       #
# bundle. That removes a per-request refit the orchestrator could not afford,  #
# and in exchange creates a new place for a stale artifact to hide -- this     #
# project's recurring bug class. These tests cover both halves: the swap did   #
# not change behaviour, and the guard that protects it actually fires.         #
# --------------------------------------------------------------------------- #
def _settings_with_tier1_name(name):
    """A copy of settings pointing Tier-1 at a different file in models/.

    settings is frozen by design, so this builds a copy rather than mutating
    it -- the frozen-config invariant holds even inside a test.
    """
    from src.agent.config import settings as real

    models = real.models.model_copy(update={"tier1_classifier_name": name})
    return real.model_copy(update={"models": models})


def test_persisted_tier1_matches_a_fresh_fit(artifacts, benchmark_tickets):
    """The whole point: persistence must be behaviour-preserving.

    If the loaded Tier-1 differed from a fresh fit by even a float, every
    cascade routing decision would be up for grabs and the goldens would be
    measuring a different model than the published results were.
    """
    import numpy as np

    from src.agent.config import settings
    from src.classification.train_cascade import (
        get_tier1_confidence,
        train_tier1,
    )
    from src.classification.train_tier1 import load_training_frame

    texts, labels, _ = load_training_frame(settings.models.dataset_path)
    fresh_vectorizer, fresh_classifier = train_tier1(texts, labels)

    sample = [t["text"] for t in benchmark_tickets]
    fresh_preds, fresh_conf = get_tier1_confidence(
        fresh_vectorizer, fresh_classifier, sample
    )
    loaded_preds, loaded_conf = get_tier1_confidence(
        artifacts.tier1_vectorizer, artifacts.tier1_classifier, sample
    )

    assert list(fresh_preds) == list(loaded_preds)
    assert float(np.max(np.abs(np.asarray(fresh_conf)
                               - np.asarray(loaded_conf)))) == 0.0


def test_tier1_was_fitted_on_the_full_dataset():
    """Pins the trap: Tier-1 is fitted on all 4,000 rows, not an 80/20 split.

    Every other training script in src/classification/ splits. Tier-1 must
    not, because the live demo did not and the goldens were captured that
    way. The manifest's own count is cross-checked against an independent
    read of the dataset rather than trusted on its own.
    """
    import joblib
    import pandas as pd

    from src.agent.config import settings

    bundle = joblib.load(settings.models.tier1_classifier_path)
    manifest = bundle["manifest"]
    independent_rows = len(pd.read_csv(settings.models.dataset_path))

    assert manifest["dataset_rows"] == independent_rows
    assert manifest["fitted_rows"] == independent_rows
    assert manifest["fitted_rows"] == 4000


def test_stale_tier1_artifact_is_refused(monkeypatch):
    """Prove the staleness guard fires rather than merely existing.

    Simulates the artifact surviving a change to the dataset it was fitted
    on -- the shape of every bug in this project's recurring class: wrong for
    its context, internally consistent, no error.
    """
    import joblib

    from src.agent import artifacts as artifacts_mod
    from src.agent.config import settings

    real = joblib.load(settings.models.tier1_classifier_path)
    tampered = dict(real)
    manifest = dict(real["manifest"])
    manifest["dataset_sha256"] = "0" * 64
    tampered["manifest"] = manifest

    path = settings.models.tier1_classifier_path.with_name(
        "tier1_stale_guard_test.joblib"
    )
    joblib.dump(tampered, path)
    try:
        monkeypatch.setattr(
            artifacts_mod, "settings",
            _settings_with_tier1_name(path.name),
        )
        with pytest.raises(ArtifactError, match="STALE TIER-1 ARTIFACT"):
            artifacts_mod.load_tier1()
    finally:
        path.unlink(missing_ok=True)


def test_split_fitted_tier1_is_refused(monkeypatch):
    """A Tier-1 fitted on 3,200 rows must not load.

    This is the specific mistake an implementer following the project's usual
    train_test_split convention would make, and it is invisible at runtime --
    the model works, it is just a different model.
    """
    import joblib

    from src.agent import artifacts as artifacts_mod
    from src.agent.config import settings

    real = joblib.load(settings.models.tier1_classifier_path)
    tampered = dict(real)
    manifest = dict(real["manifest"])
    manifest["fitted_rows"] = 3200
    tampered["manifest"] = manifest

    path = settings.models.tier1_classifier_path.with_name(
        "tier1_split_guard_test.joblib"
    )
    joblib.dump(tampered, path)
    try:
        monkeypatch.setattr(
            artifacts_mod, "settings",
            _settings_with_tier1_name(path.name),
        )
        with pytest.raises(ArtifactError, match="full dataset"):
            artifacts_mod.load_tier1()
    finally:
        path.unlink(missing_ok=True)


def test_missing_tier1_artifact_fails_loud_and_does_not_refit(monkeypatch):
    """No silent refit-on-miss.

    Quietly refitting would paper over a missing or stale artifact and leave
    the cascade running on a model nobody asked for -- the silent-fallback
    pattern tests/test_no_silent_fallback.py exists to ban. The error must
    name the command that fixes it.
    """
    from src.agent import artifacts as artifacts_mod

    monkeypatch.setattr(
        artifacts_mod, "settings",
        _settings_with_tier1_name("tier1_does_not_exist.joblib"),
    )
    with pytest.raises(ArtifactError, match="train_tier1.py"):
        artifacts_mod.load_tier1()
