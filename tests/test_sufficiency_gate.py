"""Phase 6C -- the retrieval-sufficiency gate.

Offline. No model loading, no network, no Gemini calls. These tests pin the
things that would silently corrupt 6C's result rather than fail loudly:

  * the three output filenames cannot collide, so a stability run cannot
    overwrite the dry-run record that is the provenance for what was seen
    before the full pass;
  * the verdict parser is strict -- a third label is never coerced into one of
    the two;
  * the prompt is backend-agnostic AND carries no similarity value and no
    0.67, because leaking the gate into the rater's prompt would make 6C
    measure nothing;
  * the bundle stays blind -- no provenance field from the key appears in it.
"""

from __future__ import annotations

import json
import os

import pytest

from src.experiments.run_sufficiency_autorater import (
    ANSWER_INSTRUCTION,
    SUFFICIENCY_RUBRIC,
    VERDICTS,
    build_prompt,
    cache_path,
    output_path,
    parse_verdict,
)

DATA_DIR = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data")
BUNDLE_PATH = os.path.join(DATA_DIR, "sufficiency_context_bundle.json")
KEY_PATH = os.path.join(DATA_DIR, "sufficiency_context_key.json")

GEMINI_MODEL = "gemini-flash-lite-latest"


# --------------------------------------------------------------------------
# Output filenames -- the regression that prompted this file
# --------------------------------------------------------------------------
def test_the_three_output_names_are_all_distinct():
    full = output_path("gemini", GEMINI_MODEL)
    partial = output_path("gemini", GEMINI_MODEL, limit=3)
    repeats = output_path("gemini", GEMINI_MODEL, only="S002,S029")
    assert len({full, partial, repeats}) == 3


def test_only_run_never_overwrites_the_dry_run_record():
    """The dry-run file is provenance; the stability run must not touch it."""
    partial = output_path("gemini", GEMINI_MODEL, limit=3)
    repeats = output_path("gemini", GEMINI_MODEL, only="S002")
    assert partial != repeats
    assert partial.endswith(".partial.json")
    assert repeats.endswith(".repeats.json")


def test_only_takes_precedence_when_both_are_given():
    """--only --limit together is a stability run, not a dry run."""
    both = output_path("gemini", GEMINI_MODEL, only="S002", limit=1)
    assert both.endswith(".repeats.json")


def test_neither_dry_variant_equals_the_full_run_name():
    full = output_path("gemini", GEMINI_MODEL)
    assert not full.endswith(".partial.json")
    assert not full.endswith(".repeats.json")
    for variant in (output_path("gemini", GEMINI_MODEL, limit=3),
                    output_path("gemini", GEMINI_MODEL, only="S001")):
        assert variant != full


def test_backends_and_models_get_separate_files():
    a = output_path("gemini", GEMINI_MODEL)
    b = output_path("ollama", "qwen2.5:3b-instruct")
    assert a != b
    # The model slug must survive into the name, so two local models cannot
    # overwrite each other's verdicts.
    assert "qwen2-5-3b-instruct" in os.path.basename(b)


# --------------------------------------------------------------------------
# Cache keys
# --------------------------------------------------------------------------
def test_cache_path_separates_backend_model_item_and_repeat():
    keys = {
        cache_path("gemini", GEMINI_MODEL, "S002", 1),
        cache_path("gemini", GEMINI_MODEL, "S002", 2),
        cache_path("gemini", GEMINI_MODEL, "S029", 1),
        cache_path("ollama", "qwen2.5:3b-instruct", "S002", 1),
    }
    assert len(keys) == 4


# --------------------------------------------------------------------------
# The parser -- strict on purpose
# --------------------------------------------------------------------------
@pytest.mark.parametrize("raw,expected", [
    ('{"verdict": "SUFFICIENT", "reason": "ok"}', "SUFFICIENT"),
    ('{"verdict": "INSUFFICIENT", "reason": "no match"}', "INSUFFICIENT"),
    ('```json\n{"verdict": "SUFFICIENT", "reason": "x"}\n```', "SUFFICIENT"),
    ('```\n{"verdict": "INSUFFICIENT", "reason": "x"}\n```', "INSUFFICIENT"),
    ('chatter {"verdict": "sufficient", "reason": "x"} trailing', "SUFFICIENT"),
])
def test_parse_verdict_accepts_the_shapes_models_actually_emit(raw, expected):
    assert parse_verdict(raw)["verdict"] == expected


@pytest.mark.parametrize("raw", [
    "",
    "   ",
    None,
    "no json here at all",
    '{"reason": "missing verdict"}',
    '{"verdict": "MAYBE", "reason": "x"}',
    '{"verdict": "PARTIALLY_SUFFICIENT", "reason": "x"}',
    '{"verdict": "", "reason": "x"}',
])
def test_parse_verdict_rejects_anything_outside_the_two_labels(raw):
    """A third label must fail loudly, never be coerced to one of the two."""
    with pytest.raises(Exception):
        parse_verdict(raw)


def test_verdict_vocabulary_is_exactly_two_labels():
    assert set(VERDICTS) == {"SUFFICIENT", "INSUFFICIENT"}


def test_reason_is_carried_through():
    got = parse_verdict('{"verdict": "INSUFFICIENT", "reason": "lacks a fix"}')
    assert got["reason"] == "lacks a fix"


# --------------------------------------------------------------------------
# The prompt -- one builder, and no leak of the gate
# --------------------------------------------------------------------------
@pytest.fixture(scope="module")
def bundle():
    if not os.path.isfile(BUNDLE_PATH):
        pytest.skip("context bundle not built; run build_sufficiency_context.py")
    with open(BUNDLE_PATH, "r", encoding="utf-8") as fh:
        return json.load(fh)


@pytest.fixture(scope="module")
def context_key():
    if not os.path.isfile(KEY_PATH):
        pytest.skip("context key not built; run build_sufficiency_context.py")
    with open(KEY_PATH, "r", encoding="utf-8") as fh:
        return json.load(fh)


def test_prompt_is_byte_identical_for_the_same_item(bundle):
    """build_prompt takes no backend argument, so both arms get one prompt."""
    item = bundle["items"][0]
    assert build_prompt(item) == build_prompt(item)


def test_prompt_carries_the_rubric_verbatim(bundle):
    prompt = build_prompt(bundle["items"][0])
    assert SUFFICIENCY_RUBRIC in prompt
    assert ANSWER_INSTRUCTION in prompt


def test_prompt_includes_each_retrieved_problem_description(bundle):
    """Sufficiency needs what problem each retrieved fix solved, not just its
    title -- that was the design decision taken at the plan gate."""
    item = bundle["items"][0]
    prompt = build_prompt(item)
    for r in item["retrieved"]:
        desc = (r.get("description") or "").strip()
        if desc:
            assert desc in prompt


def test_prompt_never_leaks_a_similarity_value(bundle, context_key):
    """THE leak guard. The rater must not see the gate it is tested against."""
    by_id = {r["item_id"]: r for r in context_key["items"]}
    for item in bundle["items"]:
        prompt = build_prompt(item)
        row = by_id[item["item_id"]]
        for sim in row["retrieved_similarities"]:
            # Any recognisable rendering of the float, to 3+ decimals.
            for rendering in ("%.3f" % sim, "%.4f" % sim, repr(float(sim))):
                assert rendering not in prompt
        assert "0.67" not in prompt
        assert "similarity" not in prompt.lower()


def test_prompt_never_leaks_the_arm_or_the_2b_label(bundle):
    for item in bundle["items"]:
        prompt = build_prompt(item).lower()
        for leaked in ("escalated", "eligible", "ungrounded", "grounded",
                       "g021", "g024"):
            assert leaked not in prompt


# --------------------------------------------------------------------------
# The bundle stays blind, and matches the pre-registered population
# --------------------------------------------------------------------------
def test_bundle_population_is_the_preregistered_54(bundle):
    meta = bundle["_meta"]
    assert meta["items"] == 54
    assert meta["eligible"] == 33
    assert meta["escalated"] == 21
    assert meta["seed"] == 42
    assert len(bundle["items"]) == 54


def test_bundle_ids_are_s001_through_s054(bundle):
    ids = [it["item_id"] for it in bundle["items"]]
    assert ids == ["S%03d" % i for i in range(1, 55)]


def test_bundle_records_that_retrieval_was_verified_against_2b(bundle):
    assert bundle["_meta"]["retrieval_verified_against_2b"] is True


def test_bundle_items_carry_no_provenance_fields(bundle):
    """Blinding: nothing that identifies the arm or the gate may be in an item."""
    forbidden = {"arm", "ticket_id", "groundedness_item_id", "top_similarity",
                 "similarity", "escalated", "escalation_reason",
                 "predicted_category", "retrieved_similarities"}
    for item in bundle["items"]:
        assert not (set(item) & forbidden)
        for r in item["retrieved"]:
            assert not (set(r) & forbidden)


def test_key_and_bundle_cover_the_same_items(bundle, context_key):
    assert ([it["item_id"] for it in bundle["items"]]
            == [r["item_id"] for r in context_key["items"]])


def test_key_arms_split_33_21(context_key):
    arms = [r["arm"] for r in context_key["items"]]
    assert arms.count("eligible") == 33
    assert arms.count("escalated") == 21


def test_every_escalated_item_is_below_the_live_rag_gate(context_key):
    from src.agent.config import settings
    for r in context_key["items"]:
        if r["arm"] == "escalated":
            assert r["top_similarity"] < settings.rag.similarity_threshold
        else:
            assert r["top_similarity"] >= settings.rag.similarity_threshold


def test_eligible_items_all_carry_a_2b_item_id(context_key):
    """The primary joins on this, so a missing one would silently shrink it."""
    eligible = [r for r in context_key["items"] if r["arm"] == "eligible"]
    assert len(eligible) == 33
    gids = [r["groundedness_item_id"] for r in eligible]
    assert all(g and g.startswith("G") for g in gids)
    assert len(set(gids)) == 33
