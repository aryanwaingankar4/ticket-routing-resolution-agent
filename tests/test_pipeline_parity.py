"""
Parity against the pre-refactor goldens -- the safety net for Phase 0.

The consolidation of four duplicate pipelines into src/agent/ is required to
be BEHAVIOUR-PRESERVING. Every routing decision, tier assignment, confidence
and similarity score must reproduce exactly what the old code produced.

If one of these fails, the refactor introduced a behavioural change. That is
a bug, not an improvement -- the published results were measured on the old
behaviour, and silently shifting them would be exactly the failure mode this
project exists to guard against.

Goldens are regenerated with:
    venv\\Scripts\\python.exe tests/capture_goldens.py
"""

from __future__ import annotations

import pytest

from src.agent import pipeline
from src.agent.schemas import EscalationReason, TicketIn

pytestmark = pytest.mark.slow

# ---------------------------------------------------------------------------
# TOLERANCES: one per KIND of number, not one for the file (Phase 8B.2)
# ---------------------------------------------------------------------------
# The goldens were captured on Windows. A single TOL of 1e-9 held there and
# failed on the first Linux run of gates.yml, reporting a similarity mismatch
# for every benchmark ticket. Nothing had regressed: an embedding similarity is
# a float32 BGE forward pass plus a FAISS inner product, and those accumulate
# in an order set by the machine's BLAS kernel and SIMD width.
#
# MEASURED on a Linux container against these exact goldens, all 54 tickets:
#
#     max |delta| similarity   2.384185791015625e-07   (= 2^-22, float32 eps)
#     max |delta| tier1_conf   1.1657341758564144e-15  (float64 rounding)
#     decision fields differing                     0
#
# So the fix is not a looser number, it is the right KIND of comparison:
# every decision is compared EXACTLY, and only the two float quantities carry a
# tolerance sized to the arithmetic that produced them.
#
# NOTE, correcting Phase 8B: tier1_conf is NOT bit-identical across platforms.
# 8B checked one ticket (adv_08), where the delta happened to be exactly 0.0,
# and generalised. Over 54 tickets TF-IDF + LogReg agrees to float64 rounding
# (<= 1.2e-15), which is a much stronger statement than the similarity's 2.4e-7
# but is not bit-identity.
# --- the similarity tolerance is DERIVED, not fitted to a machine ----------
#
# Phase 8B.3. The first version of this constant was 1e-6, chosen as "4.2x the
# worst case I measured in one container". That is a tolerance fitted to one
# machine, and a GitHub runner promptly produced ~7e-7 -- inside 1e-6, but only
# by luck, and it broke the 5e-7 CSV comparison next door. A tolerance has to
# come from the arithmetic, not from a sample of one.
#
# DERIVATION. A similarity here is a float32 inner product of two L2-normalised
# 768-dimensional vectors. For a dot product of length n computed in floating
# point with unit roundoff u, the standard bound (Higham, *Accuracy and
# Stability of Numerical Algorithms*, 2nd ed., section 3.1) is
#
#     |fl(x.y) - x.y|  <=  gamma_n |x| |y|,      gamma_n = n*u / (1 - n*u)
#
# float32 has u = 2^-24 = 5.9604644775390625e-08, the vectors are normalised so
# |x||y| = 1, and n = 768, giving
#
#     gamma_768 = 768 * 2^-24 / (1 - 768 * 2^-24) = 4.577846e-05.
#
# HONESTLY: this bounds the INNER PRODUCT only. It does not model the BGE
# forward pass, which also differs between platforms and is the larger part of
# what we observe. It is used because it is derived and because it covers the
# observed cross-platform deltas with room -- 2.384e-07 in a Linux container
# and ~7e-07 on a GitHub runner, 190x and 65x inside it -- not because it is a
# bound on the whole pipeline.
TOL_FLOAT32 = 4.577846e-05

# tier1_conf is float64 TF-IDF + LogReg. Measured worst case across 54 golden
# tickets: 1.166e-15. Kept far tighter than the similarity on purpose -- these
# are different kinds of number and must not share a tolerance.
TOL_FLOAT64 = 1e-12   # 858x the measured worst case

# A tolerance is only safe while no ticket sits near the gate it feeds --
# otherwise the tolerance could swallow a FLIPPED DECISION, which is the one
# thing parity exists to catch. Both distances are asserted below, every run.
#
#     closest similarity to the 0.67 RAG gate   1.458096e-04  =  3.19x the
#                                               derived tolerance
#     closest tier1_conf to the 0.50 cascade    1.264076e-02
#
# 3.19x is THIN, and deliberately reported rather than engineered away: it is
# the honest consequence of deriving the tolerance instead of fitting it to a
# machine. If a future benchmark ticket lands closer to 0.67 than 4.58e-05, the
# gate-headroom check below fails and that ticket must be compared exactly.


def _run(text, artifacts, ticket_id=None):
    return pipeline.run(
        TicketIn(title=text, ticket_id=ticket_id),
        artifacts=artifacts, generate_resolution=False, emit_log=False,
    )


def _report(request, line):
    """Print through pytest's own terminal writer, so it survives capture.

    The max deltas have to be visible on a PASSING run: a drift from 2.4e-07 to
    2.4e-05 would still pass the tolerance while saying something has changed
    about the machine, and a number nobody ever sees cannot tell anyone that.
    """
    try:
        request.config.get_terminal_writer().line("\n" + line)
    except Exception:            # pragma: no cover - fallback for odd runners
        print(line)


def _check_gate_headroom(rows, sim_key, conf_key, diffs):
    """A tolerance must never be wide enough to hide a flipped decision.

    If a golden similarity sat within TOL_FLOAT32 of 0.67, then two platforms
    could legitimately land on opposite sides of the gate while this test still
    called them equal -- the tolerance would be concealing exactly the
    regression it is meant to catch. Same argument for the cascade at 0.50.
    """
    from src.agent.config import settings

    for row in rows:
        margin = abs(row[sim_key] - settings.rag.similarity_threshold)
        if margin <= TOL_FLOAT32:
            diffs.append(
                f"UNSAFE TOLERANCE: a golden similarity ({row[sim_key]!r}) is "
                f"{margin:.3e} from the {settings.rag.similarity_threshold} "
                f"RAG gate, within the {TOL_FLOAT32:g} tolerance. The "
                f"tolerance could hide a flipped escalation decision. Compare "
                f"this ticket exactly, or tighten the tolerance -- do NOT "
                f"widen it.")
        cmargin = abs(row[conf_key] - settings.cascade.confidence_threshold)
        if cmargin <= TOL_FLOAT64:
            diffs.append(
                f"UNSAFE TOLERANCE: a golden tier1_conf ({row[conf_key]!r}) is "
                f"{cmargin:.3e} from the "
                f"{settings.cascade.confidence_threshold} cascade gate.")


def test_goldens_were_captured_under_current_config(adversarial_golden):
    """Goldens are only meaningful if captured at the live thresholds."""
    from src.agent.config import settings

    manifest = adversarial_golden["manifest"]
    assert manifest["similarity_threshold"] == (
        settings.rag.similarity_threshold
    )
    assert manifest["cascade_confidence_threshold"] == (
        settings.cascade.confidence_threshold
    )
    assert manifest["embedding_model"] == settings.models.embedding_model


def test_adversarial_parity(adversarial_tickets, adversarial_golden,
                            artifacts, request):
    """Decisions exactly; the two float quantities within their own tolerance."""
    golden = {r["id"]: r for r in adversarial_golden["rows"]}
    diffs = []
    max_sim = max_conf = 0.0
    worst_sim = worst_conf = None

    _check_gate_headroom(adversarial_golden["rows"], "rag_similarity",
                         "tier1_confidence", diffs)

    for t in adversarial_tickets:
        want = golden[t["id"]]
        got = _run(t["text"], artifacts, t["id"])

        # --- decisions: EXACT, no tolerance anywhere -----------------------
        if got.classification.category != want["predicted_category"]:
            diffs.append(f"{t['id']} category: {got.classification.category}"
                         f" != {want['predicted_category']}")
        if int(got.classification.tier) != want["tier"]:
            diffs.append(f"{t['id']} tier: {int(got.classification.tier)}"
                         f" != {want['tier']}")
        if got.escalated != want["actual_escalate"]:
            diffs.append(f"{t['id']} escalated: {got.escalated}"
                         f" != {want['actual_escalate']}")
        if len(got.retrieval.retrieved) != want["n_retrieved"]:
            diffs.append(f"{t['id']} n_retrieved: "
                         f"{len(got.retrieval.retrieved)} "
                         f"!= {want['n_retrieved']}")
        # The goldens predate EscalationReason and do not record it, so it is
        # checked as an INVARIANT rather than against a stored value: an
        # escalation must carry a reason and a non-escalation must not.
        if got.escalated and got.decision.reason == EscalationReason.NONE:
            diffs.append(f"{t['id']} escalated with reason NONE")
        if not got.escalated and got.decision.reason != EscalationReason.NONE:
            diffs.append(f"{t['id']} not escalated but reason is "
                         f"{got.decision.reason}")

        # --- floats: each against the arithmetic that produced it ----------
        sd = abs(got.top_similarity - want["rag_similarity"])
        cd = abs(got.classification.tier1_conf - want["tier1_confidence"])
        if sd > max_sim:
            max_sim, worst_sim = sd, t["id"]
        if cd > max_conf:
            max_conf, worst_conf = cd, t["id"]
        if sd > TOL_FLOAT32:
            diffs.append(f"{t['id']} similarity: {got.top_similarity!r}"
                         f" != {want['rag_similarity']!r} (delta {sd:.3e} > "
                         f"{TOL_FLOAT32:g})")
        if cd > TOL_FLOAT64:
            diffs.append(f"{t['id']} tier1_conf: {got.classification.tier1_conf!r}"
                         f" != {want['tier1_confidence']!r} (delta {cd:.3e} > "
                         f"{TOL_FLOAT64:g})")

    _report(request,
            f"[parity adversarial9] max |delta| similarity {max_sim:.3e} "
            f"({worst_sim}), tier1_conf {max_conf:.3e} ({worst_conf})")

    assert not diffs, "Parity broken vs pre-refactor goldens:\n" + "\n".join(
        diffs
    )


def test_benchmark_parity(benchmark_tickets, benchmark_golden, artifacts,
                          request):
    """All 45 tickets: every DECISION exact, the two floats within tolerance."""
    golden = {r["index"]: r for r in benchmark_golden["rows"]}
    diffs = []
    max_sim = max_conf = 0.0
    worst_sim = worst_conf = None

    _check_gate_headroom(benchmark_golden["rows"], "rag_similarity",
                         "tier1_confidence", diffs)

    for i, t in enumerate(benchmark_tickets):
        want = golden[i]
        got = _run(t["text"], artifacts)

        # --- decisions: EXACT ----------------------------------------------
        if got.classification.category != want["predicted_category"]:
            diffs.append(f"idx {i} category: {got.classification.category}"
                         f" != {want['predicted_category']}")
        if int(got.classification.tier) != want["tier"]:
            diffs.append(f"idx {i} tier: {int(got.classification.tier)}"
                         f" != {want['tier']}")
        if got.escalated != want["actual_escalate"]:
            diffs.append(f"idx {i} escalated: {got.escalated}"
                         f" != {want['actual_escalate']}")
        if len(got.retrieval.retrieved) != want["n_retrieved"]:
            diffs.append(f"idx {i} n_retrieved: "
                         f"{len(got.retrieval.retrieved)} "
                         f"!= {want['n_retrieved']}")
        if got.escalated and got.decision.reason == EscalationReason.NONE:
            diffs.append(f"idx {i} escalated with reason NONE")
        if not got.escalated and got.decision.reason != EscalationReason.NONE:
            diffs.append(f"idx {i} not escalated but reason is "
                         f"{got.decision.reason}")

        # --- floats ---------------------------------------------------------
        sd = abs(got.top_similarity - want["rag_similarity"])
        cd = abs(got.classification.tier1_conf - want["tier1_confidence"])
        if sd > max_sim:
            max_sim, worst_sim = sd, i
        if cd > max_conf:
            max_conf, worst_conf = cd, i
        if sd > TOL_FLOAT32:
            diffs.append(f"idx {i} similarity: {got.top_similarity!r} != "
                         f"{want['rag_similarity']!r} (delta {sd:.3e} > "
                         f"{TOL_FLOAT32:g})")
        if cd > TOL_FLOAT64:
            diffs.append(f"idx {i} tier1_conf: "
                         f"{got.classification.tier1_conf!r} != "
                         f"{want['tier1_confidence']!r} (delta {cd:.3e} > "
                         f"{TOL_FLOAT64:g})")

    _report(request,
            f"[parity benchmark45] max |delta| similarity {max_sim:.3e} "
            f"(idx {worst_sim}), tier1_conf {max_conf:.3e} (idx {worst_conf})")

    assert not diffs, "Parity broken vs pre-refactor goldens:\n" + "\n".join(
        diffs
    )


def test_benchmark_accuracy_unchanged(benchmark_tickets, artifacts):
    """32/45 is the measured BGE cascade baseline.

    Note this is NOT the 33/45 in the README's comparison table -- that is
    pure Tier-2 (frozen BGE + LogReg) with no cascade in front of it. The
    end-to-end cascade scores 32/45 because Tier-1 resolves four tickets
    itself at the 0.50 gate.
    """
    correct = sum(
        1 for t in benchmark_tickets
        if _run(t["text"], artifacts).classification.category == t["expected"]
    )
    assert correct == 32, f"benchmark accuracy moved: {correct}/45"
