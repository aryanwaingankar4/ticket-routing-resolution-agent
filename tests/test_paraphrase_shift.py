"""
Phase 7C helpers, verified against CONSTRUCTED ground truth.

7C's output is a coverage difference and a set of ratios -- numbers that stay
in plausible ranges whatever goes wrong upstream. A parser that quietly
returned None for every response, a bootstrap that broke the pairing, or a RAM
preflight that passed when it could not read anything would each produce a
result that looks entirely normal.

The RAM preflight gets particular attention here. It was written for 7C
because 5C's `require_free_ram` double-counts a model that Ollama already
holds resident, and a check that was relaxed to fix a false refusal is exactly
the kind of change that can quietly stop refusing anything. So the test that
matters most is that it still FAILS CLOSED when residency is unreadable.

No network, no model loading, no Ollama. The `/api/ps` call is stubbed.
"""

from __future__ import annotations

import json
import numpy as np
import pytest

from src.experiments import paraphrase_external_tickets as pet
from src.experiments.run_paraphrase_shift_conformal import (
    paired_bootstrap_did,
    paired_bootstrap_difference,
)


# --------------------------------------------------------------------------- #
# Response parsing -- an unparseable answer is a data point, not a crash
# --------------------------------------------------------------------------- #
def test_parse_extracts_the_paraphrase():
    raw = json.dumps({"paraphrase": "the printer stopped working"})
    assert pet.parse_paraphrase(raw) == "the printer stopped working"


def test_parse_strips_surrounding_whitespace():
    raw = json.dumps({"paraphrase": "  spaced out  "})
    assert pet.parse_paraphrase(raw) == "spaced out"


@pytest.mark.parametrize("raw", [
    None,
    "",
    "not json at all",
    json.dumps({"wrong_key": "value"}),
    json.dumps({"paraphrase": ""}),
    json.dumps({"paraphrase": "   "}),
    json.dumps({"paraphrase": 42}),
    json.dumps(["a", "list"]),
])
def test_parse_returns_none_rather_than_raising(raw):
    """Every one of these is counted as unparseable, none of them crashes."""
    assert pet.parse_paraphrase(raw) is None


# --------------------------------------------------------------------------- #
# The RAM preflight -- the deviation, and the guard that it still fails closed
# --------------------------------------------------------------------------- #
def test_ram_preflight_fails_closed_when_residency_is_unreadable(monkeypatch):
    """THE TEST THIS DEVIATION EXISTS TO EARN.

    The corrected check adds already-resident bytes back to the available
    reading. If /api/ps cannot be read, the correction must vanish and the
    check must fall back to the strict comparison -- "could not check" must
    never behave like "checked and there is plenty".
    """
    monkeypatch.setattr(pet, "resident_cpu_mb", lambda model: None)
    monkeypatch.setattr(
        "src.experiments.run_zeroshot_baselines.available_ram_mb",
        lambda: 1000)

    with pytest.raises(SystemExit):
        pet.require_ram_accounting_for_resident("qwen2.5:3b-instruct")


def test_ram_preflight_fails_closed_when_available_is_unreadable(monkeypatch):
    monkeypatch.setattr(pet, "resident_cpu_mb", lambda model: 4000.0)
    monkeypatch.setattr(
        "src.experiments.run_zeroshot_baselines.available_ram_mb",
        lambda: None)

    with pytest.raises(SystemExit):
        pet.require_ram_accounting_for_resident("qwen2.5:3b-instruct")


def test_ram_preflight_still_refuses_when_genuinely_short(monkeypatch):
    """Resident bytes are added back, and it is STILL not enough."""
    monkeypatch.setattr(pet, "resident_cpu_mb", lambda model: 200.0)
    monkeypatch.setattr(
        "src.experiments.run_zeroshot_baselines.available_ram_mb",
        lambda: 1000)

    with pytest.raises(SystemExit):
        pet.require_ram_accounting_for_resident("qwen2.5:3b-instruct")


def test_ram_preflight_passes_when_the_model_is_already_resident(monkeypatch):
    """The false refusal this correction was written for.

    2,275 MB available with 2,064 MB already resident was refused by the
    uncorrected check against a 3,277 MB floor, despite 4,339 MB being in
    play.
    """
    monkeypatch.setattr(pet, "resident_cpu_mb", lambda model: 2064.0)
    monkeypatch.setattr(
        "src.experiments.run_zeroshot_baselines.available_ram_mb",
        lambda: 2275)

    facts = pet.require_ram_accounting_for_resident("qwen2.5:3b-instruct")
    assert facts["resident_readable"] is True
    assert facts["effective_mb"] == pytest.approx(4339.0)
    assert facts["check"] == "corrected_for_resident_model"


def test_resident_reader_counts_only_the_non_vram_part(monkeypatch):
    """A model in GPU memory does not relieve system-RAM pressure."""
    payload = {"models": [{"name": "qwen2.5:3b-instruct",
                           "size": 3 * 1048576,
                           "size_vram": 1 * 1048576}]}
    _stub_api_ps(monkeypatch, payload)
    assert pet.resident_cpu_mb("qwen2.5:3b-instruct") == pytest.approx(2.0)


def test_resident_reader_returns_zero_when_nothing_is_loaded(monkeypatch):
    _stub_api_ps(monkeypatch, {"models": []})
    assert pet.resident_cpu_mb("qwen2.5:3b-instruct") == 0.0


def test_resident_reader_returns_none_when_the_call_fails(monkeypatch):
    """None, never 0.0 -- the distinction the fail-closed path depends on."""
    def boom(*_a, **_k):
        raise OSError("connection refused")
    monkeypatch.setattr("urllib.request.urlopen", boom)
    assert pet.resident_cpu_mb("qwen2.5:3b-instruct") is None


def test_resident_reader_returns_none_when_the_payload_has_no_models_key(
        monkeypatch):
    _stub_api_ps(monkeypatch, {"unexpected": "shape"})
    assert pet.resident_cpu_mb("qwen2.5:3b-instruct") is None


def _stub_api_ps(monkeypatch, payload):
    class _Response:
        def read(self):
            return json.dumps(payload).encode("utf-8")

        def __enter__(self):
            return self

        def __exit__(self, *_a):
            return False

    monkeypatch.setattr("urllib.request.urlopen",
                        lambda *_a, **_k: _Response())


# --------------------------------------------------------------------------- #
# Sampling
# --------------------------------------------------------------------------- #
def test_sample_is_proportional_and_leaves_no_queue_out():
    labels = np.array(["big"] * 900 + ["tiny"] * 10)
    idx = pet.stratified_sample(labels, 100)
    assert int(np.sum(labels[idx] == "tiny")) >= 1
    assert 85 <= int(np.sum(labels[idx] == "big")) <= 105


def test_sample_never_asks_for_more_than_a_queue_has():
    labels = np.array(["a"] * 3 + ["b"] * 3)
    idx = pet.stratified_sample(labels, 1000)
    assert len(idx) == 6
    assert len(set(idx.tolist())) == 6


def test_sample_is_reproducible_and_seed_sensitive():
    labels = np.array(["a"] * 50 + ["b"] * 50)
    assert np.array_equal(pet.stratified_sample(labels, 20, seed=42),
                          pet.stratified_sample(labels, 20, seed=42))
    assert not np.array_equal(pet.stratified_sample(labels, 20, seed=1),
                              pet.stratified_sample(labels, 20, seed=2))


# --------------------------------------------------------------------------- #
# Length ratios -- the pre-registered secondary
# --------------------------------------------------------------------------- #
class _FakeEmbedder:
    """A tokenizer that splits on whitespace, so counts are predictable."""

    class tokenizer:  # noqa: N801
        @staticmethod
        def encode(text, add_special_tokens=False):
            return str(text).split()


def test_length_ratio_is_one_when_nothing_changed():
    texts = ["a b c d", "e f"]
    stats = pet.length_ratio_stats(texts, texts, _FakeEmbedder())
    assert stats["word"]["mean_ratio"] == pytest.approx(1.0)
    assert stats["word"]["aggregate_ratio"] == pytest.approx(1.0)


def test_length_ratio_detects_halving():
    src = ["a b c d", "e f g h"]
    par = ["a b", "e f"]
    stats = pet.length_ratio_stats(src, par, _FakeEmbedder())
    assert stats["word"]["mean_ratio"] == pytest.approx(0.5)
    assert stats["word"]["median_ratio"] == pytest.approx(0.5)


def test_mean_and_aggregate_ratios_can_disagree():
    """One extreme short pair drags the mean but barely moves the aggregate.

    Both are reported precisely because they answer different questions; a
    reader seeing only the mean would over-read a single outlier.
    """
    src = ["a b c d e f g h i j", "x"]
    par = ["a b c d e f g h i j", "x y z w"]
    stats = pet.length_ratio_stats(src, par, _FakeEmbedder())
    assert stats["word"]["mean_ratio"] == pytest.approx(2.5)
    assert stats["word"]["aggregate_ratio"] == pytest.approx(14 / 11)


def test_length_ratio_reports_none_when_the_tokenizer_is_unreachable():
    class _NoTokenizer:
        pass

    stats = pet.length_ratio_stats(["a b"], ["c d"], _NoTokenizer())
    assert stats["token"] is None
    assert stats["word"] is not None       # the word derivation still works


def test_length_ratio_does_not_divide_by_a_zero_length_source():
    stats = pet.length_ratio_stats(["", "a b"], ["x y", "c d"],
                                   _FakeEmbedder())
    assert stats["word"]["n_zero_length_sources"] == 1
    assert np.isfinite(stats["word"]["mean_ratio"])


# --------------------------------------------------------------------------- #
# The paired bootstrap -- pairing is the whole design
# --------------------------------------------------------------------------- #
def test_bootstrap_difference_is_zero_for_identical_arms():
    covered = np.array([True, False, True, True, False] * 20)
    mean, lo, hi = paired_bootstrap_difference(covered, covered, n=500)
    assert mean == 0.0 and lo == 0.0 and hi == 0.0


def test_bootstrap_difference_recovers_a_known_gap():
    a = np.array([True] * 80 + [False] * 20)
    b = np.array([True] * 60 + [False] * 40)
    mean, lo, hi = paired_bootstrap_difference(a, b, n=2000)
    assert mean == pytest.approx(0.20, abs=0.02)
    assert lo > 0


def test_bootstrap_difference_rejects_mismatched_lengths():
    with pytest.raises(ValueError):
        paired_bootstrap_difference(np.array([True]), np.array([True, False]))


def test_did_is_zero_when_both_arms_move_together():
    """The control the DiD exists for: a change that hits both tiers equally.

    If paraphrasing simply makes the task harder, both tiers lose coverage and
    the difference-in-differences stays at zero. Only a tier-SPECIFIC loss
    moves it.
    """
    o_t1 = np.array([True] * 70 + [False] * 30)
    o_t2 = np.array([True] * 60 + [False] * 40)
    p_t1 = np.array([True] * 50 + [False] * 50)
    p_t2 = np.array([True] * 40 + [False] * 60)
    mean, lo, hi = paired_bootstrap_did(p_t1, p_t2, o_t1, o_t2, n=2000)
    assert mean == pytest.approx(0.0, abs=0.02)
    assert lo < 0 < hi


def test_did_detects_a_tier_specific_loss():
    o_t1 = np.array([True] * 90 + [False] * 10)
    o_t2 = np.array([True] * 90 + [False] * 10)
    p_t1 = np.array([True] * 50 + [False] * 50)   # tier1 collapses
    p_t2 = np.array([True] * 88 + [False] * 12)   # tier2 barely moves
    mean, _lo, hi = paired_bootstrap_did(p_t1, p_t2, o_t1, o_t2, n=2000)
    assert mean < -0.30
    assert hi < 0


def test_did_rejects_mismatched_lengths():
    ok = np.array([True, False])
    with pytest.raises(ValueError):
        paired_bootstrap_did(ok, ok, ok, np.array([True]))
