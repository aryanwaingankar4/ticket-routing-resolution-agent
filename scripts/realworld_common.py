"""Shared routing and scoring for the paper revision's extended evaluations.

Used by scripts/run_realworld_routing.py (Phase 2, Zenodo tickets) and
scripts/eval_independent.py (Phase 3, the author-written set), so both report
the same metrics computed the same way.

Routing is the UNCHANGED production pipeline: artifacts.load_artifacts() and
pipeline.run(generate_resolution=False) -- no retraining, no threshold
change, no LLM call. The Tier-2-only arm is the same two calls
src/agent/classifier.py makes (encode, tier2_classifier.predict) without the
Tier-1 gate, i.e. the ablation's tier2-only control. Where the cascade itself
answered at Tier 2 the two must agree, and a disagreement is FATAL.

Wilson intervals use the paper's convention, z = 1.96, via the committed
implementation in src/experiments/score_groundedness_set.py.
"""

from __future__ import annotations

import os
import sys

PROJECT_ROOT = os.path.abspath(
    os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)


def die(msg):
    print(f"\n[FATAL] {msg}")
    raise SystemExit(1)


def wilson(k, n):
    from src.experiments.score_groundedness_set import wilson_interval
    return wilson_interval(k, n)


def route(tickets, progress_every=200):
    """Route (ticket_id, title, description) triples. One dict per ticket.

    A single-text ticket passes its text as the title and "" as the
    description -- the convention of the adversarial gate and the drift
    evaluation. Both arms see TicketIn.combined_text, the string the
    orchestrator's classify and retrieve agents receive.
    """
    import numpy as np
    from src.agent import pipeline
    from src.agent.artifacts import load_artifacts
    from src.agent.config import config_fingerprint, settings
    from src.agent.schemas import TicketIn

    art = load_artifacts(require_gemini=False)
    fp = config_fingerprint()
    rows = []
    for i, (tid, title, description) in enumerate(tickets, start=1):
        ticket = TicketIn(title=title, description=description,
                          ticket_id=str(tid))
        res = pipeline.run(ticket, artifacts=art, generate_resolution=False,
                           emit_log=False)
        emb = np.asarray(art.embedder.encode([ticket.combined_text]),
                         dtype=np.float32)
        t2 = str(art.tier2_classifier.predict(emb)[0])
        cls = res.classification
        if int(cls.tier) == 2 and str(cls.category) != t2:
            die(f"ticket {tid}: cascade answered at Tier 2 with "
                f"{cls.category}, Tier-2-only says {t2}")
        rows.append({
            "ticket_id": tid,
            "cascade_category": str(cls.category),
            "tier": int(cls.tier),
            "tier1_category": str(cls.tier1_pred),
            "tier1_conf": repr(float(cls.tier1_conf)),
            "tier2_only_category": t2,
            "top_similarity": repr(float(res.top_similarity)),
            "escalated": bool(res.decision.escalated),
            "config_fingerprint": fp,
        })
        if progress_every and i % progress_every == 0:
            print(f"  routed {i}/{len(tickets)}")
    print(f"[route] {len(rows)} tickets; cascade threshold "
          f"{settings.cascade.confidence_threshold}, RAG threshold "
          f"{settings.rag.similarity_threshold}, fingerprint {fp}")
    return rows


def summarise(rows, gold_key="true_category", label=""):
    """Counts and Wilson CIs for a list of routed rows that carry a gold label.

    Returns a list of (quantity, value) pairs, written verbatim to a CSV.
    """
    n = len(rows)
    if n == 0:
        die(f"{label}: no rows to score")
    out = [("n", n)]

    def prop(name, k):
        lo, hi = wilson(k, n)
        out.extend([(f"{name}.k", k), (f"{name}.n", n),
                    (f"{name}.rate", repr(k / n)),
                    (f"{name}.wilson95_lo", repr(lo)),
                    (f"{name}.wilson95_hi", repr(hi))])

    prop("cascade_correct",
         sum(r["cascade_category"] == r[gold_key] for r in rows))
    prop("tier2_only_correct",
         sum(r["tier2_only_category"] == r[gold_key] for r in rows))
    prop("escalated", sum(bool(r["escalated"]) for r in rows))
    prop("tier1_kept", sum(int(r["tier"]) == 1 for r in rows))

    for cat in sorted({r[gold_key] for r in rows}):
        sub = [r for r in rows if r[gold_key] == cat]
        k = sum(r["cascade_category"] == cat for r in sub)
        k2 = sum(r["tier2_only_category"] == cat for r in sub)
        out.extend([(f"recall.cascade.{cat}", f"{k}/{len(sub)}"),
                    (f"recall.tier2_only.{cat}", f"{k2}/{len(sub)}")])
    mis = [r for r in rows if r["cascade_category"] != r[gold_key]]
    out.extend([
        ("misclassified", len(mis)),
        ("misclassified_escalated", sum(bool(r["escalated"]) for r in mis)),
        ("misclassified_drafted", sum(not bool(r["escalated"]) for r in mis)),
    ])
    pred = {}
    for r in rows:
        pred[r["cascade_category"]] = pred.get(r["cascade_category"], 0) + 1
    for cat, c in sorted(pred.items()):
        out.append((f"predicted.{cat}", c))
    return out
