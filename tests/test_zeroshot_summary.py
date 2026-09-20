"""
Phase 5C tests: the zero-shot summary, and the local-model memory guard.

WHY THIS FILE EXISTS
--------------------
Part 2 adds a second zero-shot arm, and with it three new ways to publish a
number that is internally consistent and wrong:

1. **The Wilson intervals.** These are hand-implemented in
   `summarize_zeroshot_baselines.py` so the script carries no import-time
   scipy dependency. A hand-rolled interval that is subtly wrong would still
   look like a confidence interval. Rule 6 of this project says to check every
   count against a second, independent derivation, so every interval is
   checked here against
   `scipy.stats.binomtest(...).proportion_ci(method="wilson")`.

2. **The dual accounting for unparseable output.** Counting an unparseable
   answer as correct -- or quietly dropping it from the denominator of the
   headline -- would inflate a weak model's score. Both accountings are pinned,
   including the case where they must be identical.

3. **The memory preflight.** A local model that swaps produces a latency number
   that is meaningless AND plausible. The guard must abort rather than measure
   the page file, and it must abort when it cannot read memory at all --
   "could not check" and "checked and it was fine" must not look the same.

Also pinned: the prompt-identity assertion actually fires on a mismatch. It is
the empirical proof that both arms answered the same question; a check that
cannot fail would prove nothing.

Offline: no Gemini, no Ollama, no models, no network.
"""

from __future__ import annotations

import json
import math

import pytest

from src.experiments import run_zeroshot_baselines as zs
from src.experiments import summarize_zeroshot_baselines as summ


# --------------------------------------------------------------------------
# Wilson intervals -- checked against scipy, an independent implementation
# --------------------------------------------------------------------------
@pytest.mark.parametrize(
    "successes,n",
    [
        (0, 14), (7, 14), (10, 14), (14, 14),        # benchmark14 shapes
        (0, 45), (1, 45), (23, 45), (33, 45), (40, 45), (45, 45),
        (1, 1), (3, 7),
    ],
)
def test_wilson_matches_scipy(successes, n):
    """Rule 6: a second, independent derivation of every published interval."""
    from scipy.stats import binomtest

    mine_lo, mine_hi = summ.wilson_interval(successes, n)
    theirs = binomtest(successes, n).proportion_ci(method="wilson")

    assert mine_lo == pytest.approx(theirs.low, abs=1e-12)
    assert mine_hi == pytest.approx(theirs.high, abs=1e-12)


def test_wilson_stays_inside_the_unit_interval_at_the_boundary():
    """The normal approximation runs outside [0,1] here; Wilson must not."""
    lo, hi = summ.wilson_interval(45, 45)
    assert 0.0 <= lo <= 1.0
    assert hi == 1.0
    lo0, hi0 = summ.wilson_interval(0, 45)
    assert lo0 == 0.0
    assert 0.0 <= hi0 <= 1.0


def test_wilson_on_an_empty_sample_is_nan_not_a_crash():
    lo, hi = summ.wilson_interval(0, 0)
    assert math.isnan(lo) and math.isnan(hi)


def test_wilson_rejects_impossible_counts():
    with pytest.raises(SystemExit):
        summ.wilson_interval(46, 45)


# --------------------------------------------------------------------------
# Confusion table and the accountings
# --------------------------------------------------------------------------
def _row(ticket_id, expected, predicted, parsed_ok=True, source="live",
         elapsed=1.0):
    return {
        "ticket_id": ticket_id,
        "text": "t",
        "expected": expected,
        "predicted": predicted,
        "parsed_ok": parsed_ok,
        "correct": bool(parsed_ok and predicted == expected),
        "elapsed_s": elapsed,
        "source": source,
    }


def test_confusion_reconciles_with_the_headline_count():
    rows = [
        _row("a", "Network", "Network"),
        _row("b", "Network", "Database"),
        _row("c", "Storage", "Storage"),
        _row("d", "Security", "", parsed_ok=False),
    ]
    table, labels = summ.confusion(rows)

    assert sum(sum(v.values()) for v in table.values()) == len(rows)
    diagonal = sum(table[c][c] for c in zs.CATEGORIES)
    assert diagonal == sum(1 for r in rows if r["correct"]) == 2
    assert table["Security"]["UNPARSEABLE"] == 1
    assert table["Network"]["Database"] == 1
    assert "UNPARSEABLE" in labels


def test_confusion_refuses_a_label_outside_the_seven_categories():
    with pytest.raises(SystemExit):
        summ.confusion([_row("a", "Networking", "Network")])


def test_unparseable_is_never_counted_correct():
    """The specific way this could be consistent and wrong."""
    rows = [_row("a", "Network", "Network", parsed_ok=False)]
    assert rows[0]["correct"] is False
    table, _ = summ.confusion(rows)
    assert table["Network"]["UNPARSEABLE"] == 1
    assert table["Network"]["Network"] == 0


def test_per_category_precision_and_recall():
    rows = [
        _row("a", "Network", "Network"),
        _row("b", "Network", "Network"),
        _row("c", "Storage", "Network"),
    ]
    table, _ = summ.confusion(rows)
    cats = summ.per_category(table)

    assert cats["Network"]["support"] == 2
    assert cats["Network"]["recall"] == pytest.approx(1.0)
    # Three tickets were called Network; two of them really were.
    assert cats["Network"]["precision"] == pytest.approx(2.0 / 3.0)
    assert cats["Storage"]["recall"] == pytest.approx(0.0)
    assert math.isnan(cats["Storage"]["precision"])


# --------------------------------------------------------------------------
# Latency: live rows only
# --------------------------------------------------------------------------
def test_latency_ignores_cached_rows():
    """A cached row carries the ORIGINAL run's timing, not this run's."""
    rows = [
        _row("a", "Network", "Network", source="live", elapsed=2.0),
        _row("b", "Network", "Network", source="cache", elapsed=999.0),
        _row("c", "Network", "Network", source="live", elapsed=4.0),
    ]
    lat = summ.latency_stats(rows)
    assert lat["n_live"] == 2
    assert lat["max_s"] == 4.0
    assert lat["median_s"] in (2.0, 4.0)


def test_latency_on_an_all_cached_file_reports_no_live_rows():
    rows = [_row("a", "Network", "Network", source="cache", elapsed=1.0)]
    lat = summ.latency_stats(rows)
    assert lat["n_live"] == 0
    assert math.isnan(lat["median_s"])


# --------------------------------------------------------------------------
# Prompt identity -- the check must be able to fail
# --------------------------------------------------------------------------
def _write_cache(tmp_path, monkeypatch, backend, model, eval_set, ticket_id,
                 phash):
    monkeypatch.setattr(zs, "RAW_CACHE_DIR", str(tmp_path))
    path = zs.cache_path(backend, model, eval_set, ticket_id)
    with open(path, "w", encoding="utf-8") as fh:
        json.dump({"prompt_sha256": phash, "raw": "{}"}, fh)
    return path


def test_prompt_identity_passes_when_the_hashes_agree(tmp_path, monkeypatch):
    rows = [_row("t1", "Network", "Network")]
    _write_cache(tmp_path, monkeypatch, "ollama", "m1", "benchmark14", "t1",
                 "deadbeef")
    _write_cache(tmp_path, monkeypatch, "gemini", "m2", "benchmark14", "t1",
                 "deadbeef")

    checked, missing = summ.check_prompt_identity(
        "ollama", "m1", "gemini", "m2", "benchmark14", rows)
    assert (checked, missing) == (1, 0)


def test_prompt_identity_is_fatal_when_the_hashes_differ(tmp_path, monkeypatch):
    rows = [_row("t1", "Network", "Network")]
    _write_cache(tmp_path, monkeypatch, "ollama", "m1", "benchmark14", "t1",
                 "aaaa")
    _write_cache(tmp_path, monkeypatch, "gemini", "m2", "benchmark14", "t1",
                 "bbbb")

    with pytest.raises(SystemExit):
        summ.check_prompt_identity("ollama", "m1", "gemini", "m2",
                                   "benchmark14", rows)


def test_prompt_identity_counts_uncached_tickets_rather_than_passing_them(
        tmp_path, monkeypatch):
    """A missing pair must be reported, not silently counted as verified."""
    rows = [_row("t1", "Network", "Network")]
    _write_cache(tmp_path, monkeypatch, "ollama", "m1", "benchmark14", "t1",
                 "aaaa")

    checked, missing = summ.check_prompt_identity(
        "ollama", "m1", "gemini", "m2", "benchmark14", rows)
    assert (checked, missing) == (0, 1)


def test_a_check_that_verified_nothing_is_fatal(tmp_path, monkeypatch):
    """'0 verified' must not be recorded as if it were 'all verified'.

    This fired for real: summarising the Gemini arm against the Ollama arm
    looked up the cache under the DEFAULT Ollama tag (the 7B) while the run
    used the 3B, so every ticket came back 'not cached on both sides' and the
    summary would have recorded prompt_identity_verified = 0 -- a number that
    looks like a measurement and is really a wrong filename.
    """
    monkeypatch.setattr(summ, "load_rows_with_source",
                        lambda b, m, s: ([_row("t1", "Network", "Network")],
                                         "fake.csv"))
    monkeypatch.setattr(summ, "check_prompt_identity",
                        lambda *a, **k: (0, 1))

    with pytest.raises(SystemExit):
        summ.summarise_set("ollama", "m1", "benchmark14", "gemini")


# --------------------------------------------------------------------------
# The memory preflight
# --------------------------------------------------------------------------
def test_ram_floor_aborts_below_threshold(monkeypatch):
    monkeypatch.setattr(zs, "available_ram_mb", lambda: 1500)
    with pytest.raises(SystemExit):
        zs.require_free_ram("qwen2.5:3b-instruct")


def test_ram_floor_proceeds_above_threshold(monkeypatch):
    monkeypatch.setattr(zs, "available_ram_mb", lambda: 8000)
    assert zs.require_free_ram("qwen2.5:3b-instruct") == 8000


def test_ram_floor_is_fatal_when_memory_cannot_be_read(monkeypatch):
    """'Could not check' must not look like 'checked and it was fine'."""
    monkeypatch.setattr(zs, "available_ram_mb", lambda: None)
    with pytest.raises(SystemExit):
        zs.require_free_ram("qwen2.5:3b-instruct")


def test_allow_low_ram_overrides_but_only_explicitly(monkeypatch):
    monkeypatch.setattr(zs, "available_ram_mb", lambda: 1500)
    assert zs.require_free_ram("qwen2.5:3b-instruct", allow_low_ram=True) == 1500


def test_an_unknown_model_still_gets_a_floor(monkeypatch):
    """A new tag must not silently mean 'no memory requirement'."""
    assert "not-a-real-model" not in zs.MODEL_RAM_FLOOR_MB
    monkeypatch.setattr(zs, "available_ram_mb", lambda: 100)
    with pytest.raises(SystemExit):
        zs.require_free_ram("not-a-real-model")


def test_the_floors_are_ordered_by_model_size():
    assert (zs.MODEL_RAM_FLOOR_MB["qwen2.5:7b-instruct"]
            > zs.MODEL_RAM_FLOOR_MB["qwen2.5:3b-instruct"])


def test_available_ram_reads_a_plausible_value_on_this_machine():
    """Not a mock: the real reader must return something sane, or None."""
    value = zs.available_ram_mb()
    if value is not None:
        assert 0 < value < 4_000_000
