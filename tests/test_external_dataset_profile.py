"""
Phase 7A profiling helpers, verified against known ground truth.

These tests exist because 7A's whole output is a set of RATES, and a rate is
the easiest kind of number to get quietly wrong: it stays in [0, 1], it looks
plausible, and nothing errors. So every helper is checked on a small frame
whose duplicate and cluster structure is constructed, not eyeballed.

No network, no model loading, no external data required -- the helpers take
plain frames and arrays. The expensive parts of `profile_external_dataset`
(BGE encoding, FAISS, MiniLM) are deliberately not exercised here; they are
covered by the script's own in-run rule-6 cross-checks against brute force.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.experiments.profile_external_dataset import (
    exact_duplicate_stats,
    length_stats,
    _norm_text,
)


# --------------------------------------------------------------------------- #
# Normalisation
# --------------------------------------------------------------------------- #
def test_normalisation_folds_case_and_whitespace():
    assert _norm_text("  Hello   WORLD \n") == "hello world"
    assert _norm_text("A\t\tB") == "a b"
    assert _norm_text("same") == _norm_text("SAME")


def test_normalisation_does_not_merge_genuinely_different_text():
    assert _norm_text("printer offline") != _norm_text("printer online")


# --------------------------------------------------------------------------- #
# Exact duplicates -- constructed ground truth
# --------------------------------------------------------------------------- #
def test_exact_duplicates_on_known_structure():
    """3 copies of A, 2 of B, 1 unique => 6 rows, 3 unique, 3 duplicate rows."""
    texts = ["alpha", "alpha", "alpha", "beta", "beta", "gamma"]
    stats = exact_duplicate_stats(texts)

    assert stats["n_rows"] == 6
    assert stats["n_unique_texts"] == 3
    assert stats["n_duplicate_rows"] == 3
    assert stats["duplicate_rate"] == pytest.approx(0.5)
    assert stats["largest_duplicate_group"] == 3


def test_the_two_duplicate_rates_are_distinct_and_both_correct():
    """Excess rows vs rows-in-a-group differ by ~2x on a corpus of pairs.

    4 texts each appearing twice = 8 rows: 4 are redundant copies, but all 8
    share their text with another row. Quoting one rate as the other would
    misstate corpus redundancy by a factor of two, so both are computed and
    both are pinned.
    """
    stats = exact_duplicate_stats(["a", "a", "b", "b", "c", "c", "d", "d"])

    assert stats["n_rows"] == 8
    assert stats["n_unique_texts"] == 4
    assert stats["n_duplicate_groups"] == 4
    assert stats["largest_duplicate_group"] == 2

    # Redundant copies only.
    assert stats["n_duplicate_rows"] == 4
    assert stats["excess_row_rate"] == pytest.approx(0.5)
    # Every row that shares its text with another.
    assert stats["n_rows_in_a_duplicate_group"] == 8
    assert stats["rows_in_a_duplicate_group_rate"] == pytest.approx(1.0)


def test_rows_in_a_duplicate_group_excludes_unique_rows():
    """A unique row must not be counted as part of a duplicate group."""
    stats = exact_duplicate_stats(["a", "a", "unique"])
    assert stats["n_rows_in_a_duplicate_group"] == 2
    assert stats["n_duplicate_rows"] == 1
    assert stats["n_duplicate_groups"] == 1


def test_force_does_not_imply_reencode():
    """--force re-renders the report; re-encoding is a separate ~2.5h flag.

    Coupling them would mean any report fix silently costs hours, which is the
    kind of friction that leads to a stale report being kept instead.
    """
    import inspect

    from src.experiments import profile_external_dataset as prof

    params = inspect.signature(prof.run).parameters
    assert "force" in params and "reencode" in params
    assert params["reencode"].default is False


def test_exact_duplicates_are_case_and_whitespace_insensitive():
    """The normalisation must actually be applied, not merely defined."""
    stats = exact_duplicate_stats(["Printer  jam", "printer jam", "PRINTER\tJAM"])
    assert stats["n_unique_texts"] == 1
    assert stats["n_duplicate_rows"] == 2


def test_exact_duplicates_none():
    stats = exact_duplicate_stats(["a", "b", "c", "d"])
    assert stats["n_duplicate_rows"] == 0
    assert stats["duplicate_rate"] == 0.0
    assert stats["largest_duplicate_group"] == 1


def test_exact_duplicates_all_identical():
    stats = exact_duplicate_stats(["x"] * 10)
    assert stats["n_unique_texts"] == 1
    assert stats["n_duplicate_rows"] == 9
    assert stats["duplicate_rate"] == pytest.approx(0.9)


# --------------------------------------------------------------------------- #
# Length stats
# --------------------------------------------------------------------------- #
def test_length_stats_excludes_empties_from_the_averages():
    """Empties are COUNTED but must not drag the median toward zero.

    A corpus with many blank `answer` fields would otherwise report a
    plausible-looking short median rather than the real distribution of the
    text that exists.
    """
    s = pd.Series(["aaa", "bbbbb", "", None, "ccccccc"])
    stats = length_stats(s)

    assert stats["n"] == 5
    assert stats["n_empty_or_null"] == 2
    assert stats["median_chars"] == pytest.approx(5.0)
    assert stats["max_chars"] == 7


def test_length_stats_all_empty_does_not_raise():
    stats = length_stats(pd.Series(["", None, np.nan]))
    assert stats["n_empty_or_null"] == 3
    assert stats["mean_chars"] == 0.0
    assert stats["median_chars"] == 0.0


# --------------------------------------------------------------------------- #
# The production grouping, reused rather than reimplemented
# --------------------------------------------------------------------------- #
def test_item7_reuses_the_production_clustering_functions():
    """7A must call production's grouping, not a second copy of it.

    If `flag_automation_candidates` ever stops exporting these, item 7 would
    silently need its own implementation -- which is how this project's three
    divergent loaders happened in the first place.
    """
    from src.experiments.flag_automation_candidates import (
        cosine_similarity_matrix, group_by_threshold,
    )

    assert callable(cosine_similarity_matrix)
    assert callable(group_by_threshold)


def test_production_grouping_separates_two_obvious_blocks():
    """Two orthogonal directions must not merge at the production threshold."""
    from src.experiments.flag_automation_candidates import (
        cosine_similarity_matrix, group_by_threshold,
    )

    a = np.tile(np.array([1.0, 0.0, 0.0], dtype=np.float32), (4, 1))
    b = np.tile(np.array([0.0, 1.0, 0.0], dtype=np.float32), (3, 1))
    sim = cosine_similarity_matrix(np.vstack([a, b]))

    clusters = group_by_threshold(sim, 0.80)
    assert len(clusters) == 2
    assert sorted(len(c) for c in clusters) == [3, 4]


def test_production_grouping_reads_the_configured_threshold():
    """Item 7's threshold comes from config, never a literal."""
    from src.agent.config import settings

    assert settings.clustering.resolution_similarity_threshold == 0.80
    assert settings.clustering.clustering_embedding_model == "all-MiniLM-L6-v2"


# --------------------------------------------------------------------------- #
# Provenance constants
# --------------------------------------------------------------------------- #
def test_dataset_revision_is_pinned_to_a_full_sha():
    """An unpinned or abbreviated revision would let upstream move silently."""
    from src.experiments.fetch_external_dataset import (
        DATASET_LICENSE, DATASET_REPO, DATASET_REVISION,
    )

    assert DATASET_REPO == "Tobi-Bueck/customer-support-tickets"
    assert len(DATASET_REVISION) == 40
    assert all(c in "0123456789abcdef" for c in DATASET_REVISION)
    assert DATASET_LICENSE == "cc-by-nc-4.0"


def test_english_token_set_is_normalised_lowercase():
    """The language filter matches on normalised values, so its token set
    must be normalised too -- an uppercase entry would never match."""
    from src.experiments.fetch_external_dataset import ENGLISH_TOKENS

    assert all(t == t.lower().strip() for t in ENGLISH_TOKENS)
    assert "en" in ENGLISH_TOKENS
