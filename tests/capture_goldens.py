"""
Capture golden pipeline outputs from the CURRENT (pre-refactor) code.

Phase 0, Step 0. This runs the existing, unmodified pipeline over both fixed
benchmark sets and records every routing/gate decision to JSON. Those files
become the parity baseline that the src/agent/ refactor must reproduce
bit-for-bit.

Deliberately reuses the loader and runner already in
src/experiments/test_adversarial_escalation.py rather than re-implementing
them, so the captured baseline is the real live behaviour and not a third
copy of the pipeline.

Gemini is never called -- the escalation decision is fully determined before
any LLM call, so routing goldens need no API key and no quota.

IMPORTANT CAVEAT
----------------
The goldens currently committed under tests/goldens/ were captured BEFORE the
src/agent/ consolidation, from the original four-copy implementation. They are
therefore a genuine independent baseline, and test_pipeline_parity.py checks
the new pipeline against them.

Re-running this script now no longer produces an independent baseline:
test_adversarial_escalation.run_ticket_through_pipeline() has since become a
thin adapter over src/agent/pipeline.py, so a regenerated golden would simply
be the new pipeline agreeing with itself. Regenerate only when a behaviour
change is DELIBERATE, and say so in the commit message.

Run:
    venv\\Scripts\\python.exe tests/capture_goldens.py
"""

from __future__ import annotations

import json
import os
import sys

_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(_THIS_DIR, os.pardir))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

DATA_DIR = os.path.join(PROJECT_ROOT, "data")
GOLDENS_DIR = os.path.join(_THIS_DIR, "goldens")

ADVERSARIAL_JSON = os.path.join(
    DATA_DIR, "adversarial_escalation_tickets.json")
BENCHMARK_JSON = os.path.join(DATA_DIR, "novel_tickets_expanded.json")


def _load_json(path):
    if not os.path.isfile(path):
        raise SystemExit(f"Missing required fixture: {path}")
    with open(path, "r", encoding="utf-8") as fh:
        return json.load(fh)


def main() -> None:
    from src.experiments import test_adversarial_escalation as adv

    print("=" * 70)
    print("CAPTURE GOLDENS  --  pre-refactor pipeline baseline")
    print("=" * 70)

    # Reuse the existing live-pipeline wiring verbatim.
    train_tier1, get_tier1_confidence = adv._import_cascade_functions()
    sr = adv._import_rag_layer()
    classify_cascade_fn, classify_source = adv._resolve_classify_cascade(
        train_tier1, get_tier1_confidence
    )
    resources = adv.load_pipeline_resources(train_tier1, sr)

    print(f"\n[capture] classify_ticket_cascade source: {classify_source}")
    print(f"[capture] similarity_threshold in use: "
          f"{resources['similarity_threshold']}")

    os.makedirs(GOLDENS_DIR, exist_ok=True)

    manifest = {
        "similarity_threshold": float(
            resources["similarity_threshold"]),
        "cascade_confidence_threshold": float(
            adv.CASCADE_CONFIDENCE_THRESHOLD),
        "embedding_model": adv.EMBEDDING_MODEL_NAME,
        "index_ntotal": int(resources["index"].ntotal),
        "classify_source": classify_source,
    }

    # ---- Adversarial set (9 tickets) ---------------------------------------
    adversarial = _load_json(ADVERSARIAL_JSON)
    adv_rows = []
    for t in adversarial:
        out = adv.run_ticket_through_pipeline(
            t, resources, classify_cascade_fn, get_tier1_confidence
        )
        out["id"] = t["id"]
        out["expected_escalate"] = bool(t["expected_escalate"])
        adv_rows.append(out)
        print(f"  [adv] {t['id']:<8} "
              f"cat={out['predicted_category']:<18} "
              f"tier={out['tier']} sim={out['rag_similarity']:.6f} "
              f"escalate={out['actual_escalate']}")

    # ---- 45-ticket generalization benchmark --------------------------------
    benchmark = _load_json(BENCHMARK_JSON)
    bench_rows = []
    for i, t in enumerate(benchmark):
        out = adv.run_ticket_through_pipeline(
            {"text": t["text"]}, resources, classify_cascade_fn,
            get_tier1_confidence,
        )
        out["index"] = i
        out["expected"] = t["expected"]
        out["correct"] = (out["predicted_category"] == t["expected"])
        bench_rows.append(out)

    n_correct = sum(1 for r in bench_rows if r["correct"])
    n_adv_pass = sum(
        1 for r in adv_rows if r["actual_escalate"] == r["expected_escalate"]
    )

    print(f"\n[capture] adversarial: {n_adv_pass}/{len(adv_rows)} "
          f"match expected")
    print(f"[capture] benchmark:   {n_correct}/{len(bench_rows)} correct")

    manifest["adversarial_pass"] = f"{n_adv_pass}/{len(adv_rows)}"
    manifest["benchmark_score"] = f"{n_correct}/{len(bench_rows)}"

    for name, payload in (
        ("adversarial_baseline.json",
         {"manifest": manifest, "rows": adv_rows}),
        ("benchmark_baseline.json",
         {"manifest": manifest, "rows": bench_rows}),
    ):
        path = os.path.join(GOLDENS_DIR, name)
        with open(path, "w", encoding="utf-8") as fh:
            json.dump(payload, fh, indent=2, sort_keys=True)
            fh.write("\n")
        print(f"[capture] wrote {path}")

    print("\nDone. These files are the parity baseline for the refactor.")


if __name__ == "__main__":
    main()
