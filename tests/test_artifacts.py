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
