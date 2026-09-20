"""
Phase 5C tests: the zero-shot baseline harness.

WHY THIS FILE EXISTS
--------------------
Two things in this harness can fail silently and produce a plausible number,
which is the failure mode this project keeps hitting:

1. **The parser.** If a near-miss like "Networking" were coerced onto
   "Network", the baseline would be credited with answers the model never
   gave, and the error would look like accuracy. So the parser is pinned from
   both directions: every legitimate shape is accepted, and every near-miss is
   REFUSED rather than mapped.

2. **The category list.** If the prompt offered a different label set than the
   classifier being compared against, the LLM could be marked wrong for
   answering a question it was never asked. `validate_categories()` checks the
   list against the production classifier's own `classes_` and against the
   benchmarks' labels; these tests check that the check actually fires.

Also pinned: the raw-response cache must be keyed on the PROMPT, so that
editing the prompt invalidates old answers instead of silently reusing
responses to a different question -- the cheapest possible way to publish a
result for a prompt that no longer exists.

Offline: no Gemini, no Ollama, no models. The one artifact touched is the
persisted Tier-1 bundle, read by validate_categories.
"""

from __future__ import annotations

import json

import pytest

from src.experiments import run_zeroshot_baselines as zs


# --------------------------------------------------------------------------
# The benchmarks and the category list
# --------------------------------------------------------------------------
def test_both_benchmarks_load_at_their_fixed_sizes():
    b45 = zs.load_eval_set("benchmark45")
    b14 = zs.load_eval_set("benchmark14")
    assert len(b45) == 45
    assert len(b14) == 14
    # 59 is the number the Gemini budget is planned against.
    assert len(b45) + len(b14) == 59


def test_ticket_ids_are_unique_and_prefixed_per_set():
    b45 = zs.load_eval_set("benchmark45")
    b14 = zs.load_eval_set("benchmark14")
    ids = [r["id"] for r in b45 + b14]
    assert len(ids) == len(set(ids)), "ticket ids collide across sets"
    assert all(r["id"].startswith("b45_") for r in b45)
    assert all(r["id"].startswith("b14_") for r in b14)


@pytest.mark.slow
def test_category_list_matches_the_production_classifier():
    """The real check, against the real artifact."""
    records = zs.load_eval_set("benchmark45") + zs.load_eval_set("benchmark14")
    zs.validate_categories(records)  # _fatal() raises SystemExit on mismatch


@pytest.mark.slow
def test_category_validation_rejects_a_mismatched_label_set(monkeypatch):
    """And it fires -- a check that cannot fail is not a check (4A finding 3)."""
    monkeypatch.setattr(zs, "CATEGORIES", zs.CATEGORIES[:-1] + ["Telephony"])
    with pytest.raises(SystemExit):
        zs.validate_categories(zs.load_eval_set("benchmark14"))


def test_unknown_eval_set_is_refused():
    with pytest.raises(SystemExit):
        zs.load_eval_set("benchmark99")


# --------------------------------------------------------------------------
# The prompt
# --------------------------------------------------------------------------
def test_prompt_offers_every_category_in_a_fixed_order():
    prompt = zs.build_prompt("the wifi keeps dropping")
    for category in zs.CATEGORIES:
        assert category in prompt
    positions = [prompt.index(c) for c in zs.CATEGORIES]
    assert positions == sorted(positions), "category order is not stable"


def test_prompt_is_backend_independent():
    """One prompt builder, taking only the ticket -- no backend can special-case
    itself, which is what keeps the two baselines comparable."""
    import inspect

    params = list(inspect.signature(zs.build_prompt).parameters)
    assert params == ["text"]


def test_prompt_contains_the_ticket_verbatim():
    text = "VPN times out from home but the internet is fine"
    assert text in zs.build_prompt(text)


# --------------------------------------------------------------------------
# The parser -- accepts every real shape
# --------------------------------------------------------------------------
@pytest.mark.parametrize("category", zs.CATEGORIES)
def test_parser_accepts_the_requested_json_for_every_category(category):
    raw = json.dumps({"category": category})
    assert zs.parse_response(raw) == (category, True)


@pytest.mark.parametrize("category", zs.CATEGORIES)
def test_parser_accepts_a_bare_category_name(category):
    """A bare name is a different FORMAT, not a different answer."""
    assert zs.parse_response(category) == (category, True)


def test_parser_accepts_a_fenced_json_block():
    raw = '```json\n{"category": "Access Management"}\n```'
    assert zs.parse_response(raw) == ("Access Management", True)


def test_parser_is_case_and_whitespace_insensitive():
    assert zs.parse_response("  network  ") == ("Network", True)
    assert zs.parse_response('{"category":"ACCESS MANAGEMENT"}') == (
        "Access Management", True)


# --------------------------------------------------------------------------
# The parser -- refuses everything else, rather than coercing it
# --------------------------------------------------------------------------
@pytest.mark.parametrize("raw", [
    '{"category": "Networking"}',        # near-miss: NOT Network
    '{"category": "Net"}',               # prefix: NOT Network
    '{"category": "Network issues"}',    # superset: NOT Network
    "I think this is Network because the VPN is involved",
    '{"category": "Database or Storage"}',
    '{"wrong_key": "Network"}',
    '{"category": null}',
    '{"category": 3}',
    "",
    "   ",
    None,
])
def test_parser_refuses_rather_than_coercing(raw):
    predicted, parsed_ok = zs.parse_response(raw)
    assert parsed_ok is False
    assert predicted is None


def test_a_refused_response_is_never_scored_correct():
    """An unparseable answer must not be able to land on the right label."""
    predicted, parsed_ok = zs.parse_response("Networking")
    assert not (parsed_ok and predicted == "Network")


# --------------------------------------------------------------------------
# The raw-response cache
# --------------------------------------------------------------------------
def test_cache_round_trips(tmp_path, monkeypatch):
    monkeypatch.setattr(zs, "RAW_CACHE_DIR", str(tmp_path))
    prompt = zs.build_prompt("disk is full")
    phash = zs.prompt_hash(prompt)
    path = zs.cache_path("gemini", "gemini-flash-lite-latest", "benchmark45",
                         "b45_00")
    zs.write_cache(path, {"prompt_sha256": phash, "raw": '{"category":"Storage"}',
                          "elapsed_s": 1.5})
    got = zs.read_cache(path, phash)
    assert got is not None
    assert zs.parse_response(got["raw"]) == ("Storage", True)


def test_cache_is_invalidated_when_the_prompt_changes(tmp_path, monkeypatch):
    """Editing the prompt must not silently reuse answers to the old one."""
    monkeypatch.setattr(zs, "RAW_CACHE_DIR", str(tmp_path))
    path = zs.cache_path("ollama", "qwen2.5:7b-instruct", "benchmark14",
                         "b14_00")
    zs.write_cache(path, {"prompt_sha256": zs.prompt_hash("OLD PROMPT"),
                          "raw": '{"category":"Network"}'})
    assert zs.read_cache(path, zs.prompt_hash("NEW PROMPT")) is None
    assert zs.read_cache(path, zs.prompt_hash("OLD PROMPT")) is not None


def test_cache_miss_on_an_absent_file(tmp_path, monkeypatch):
    monkeypatch.setattr(zs, "RAW_CACHE_DIR", str(tmp_path))
    path = zs.cache_path("gemini", "m", "benchmark45", "b45_99")
    assert zs.read_cache(path, "deadbeef") is None


def test_cache_paths_separate_backend_model_and_set():
    """Two backends, or two models, must never share a cache entry."""
    a = zs.cache_path("gemini", "gemini-flash-lite-latest", "benchmark45", "b45_00")
    b = zs.cache_path("ollama", "qwen2.5:7b-instruct", "benchmark45", "b45_00")
    c = zs.cache_path("ollama", "qwen2.5:3b-instruct", "benchmark45", "b45_00")
    d = zs.cache_path("ollama", "qwen2.5:7b-instruct", "benchmark14", "b45_00")
    assert len({a, b, c, d}) == 4


# --------------------------------------------------------------------------
# Quota discipline
# --------------------------------------------------------------------------
def test_gemini_delay_floor_meets_the_project_rule():
    assert zs.GEMINI_CALL_DELAY_SEC >= 4.5


def test_gemini_has_a_hard_call_ceiling_above_the_real_workload():
    """A budget guard that is below 59 would block the real run; one that is
    unbounded would not guard anything."""
    assert 59 <= zs.MAX_TOTAL_GEMINI_CALLS <= 200


# --------------------------------------------------------------------------
# The shared exact-McNemar core
# --------------------------------------------------------------------------
def test_shared_mcnemar_core_matches_a_hand_computed_table():
    """5B and 5C must run the SAME test, not two implementations of it.

    b=8, c=1 -> two-sided exact binomial on n=9. P(X<=1) = (1 + 9)/512 =
    10/512, so p = 20/512 = 0.0390625. This is the benchmark45 table Phase 5C
    actually produced, pinned here so a refactor of the shared core cannot
    quietly move a published p-value.
    """
    from scipy.stats import binomtest

    from src.experiments.compare_cascade_vs_tier2 import mcnemar_from_pairs

    def side(correct):
        return {"correct": correct}

    pairs = ([(side(True), side(False))] * 8
             + [(side(False), side(True))] * 1
             + [(side(True), side(True))] * 32
             + [(side(False), side(False))] * 4)
    res = mcnemar_from_pairs(pairs, binomtest)

    assert (res["b"], res["c"]) == (8, 1)
    assert res["n"] == 45
    assert res["left_correct"] == 40
    assert res["right_correct"] == 33
    assert res["p_value"] == pytest.approx(0.0390625)


def test_shared_core_is_the_one_used_by_the_cascade_comparison():
    """analyse() must delegate, not carry a second copy of the arithmetic."""
    import inspect

    from src.experiments import compare_cascade_vs_tier2 as cc

    assert "mcnemar_from_pairs(" in inspect.getsource(cc.analyse)
