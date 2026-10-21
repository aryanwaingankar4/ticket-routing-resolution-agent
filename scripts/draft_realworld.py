"""Gemini drafts for at most 30 real Zenodo tickets that pass the RAG gate.

Paper revision (Oct 2026), Phase 2 step 5. SPENDS GEMINI QUOTA: one call per
draft, at most 30 in total. Always dry-run first:

    .\\venv\\Scripts\\python.exe scripts/draft_realworld.py --limit 3
    .\\venv\\Scripts\\python.exe scripts/draft_realworld.py

Population: the DISTINCT texts of the mapped subset (confirmed primary
mapping in category_mapping.json) that routing_results.csv records as NOT
escalated -- the tickets production would actually send to the drafter.

Sample: fixed BEFORE any call and independent of --limit. Stratified by
source category with equal allocation (30 // 4 = 7 each; the remaining 2 go
to the two largest strata), random.Random(42) over ids sorted numerically,
then one seeded shuffle of the 30 so the --limit 3 dry run is the first three
of the same sample, not a different selection.

Each ticket is re-routed with the unchanged pipeline (generate_resolution=
False) and must reproduce its routing_results.csv row exactly (category,
tier, not escalated) -- FATAL otherwise. The prompt is rebuilt with
src/agent/resolver.build_prompt and hashed; a cached response under
drafts_raw/<id>.json with the same prompt hash is reused, so the dry run's
three drafts are not paid for twice and a re-run costs nothing. On a cache
miss the draft comes from pipeline.run(generate_resolution=True), the
production path, whose retrieved context must match the one the prompt was
built from. The run STOPS at the first LLM failure, so retries on a rate
limit cannot multiply the call count.

Outputs (data/external_zenodo_7648117/):
  drafts_raw/<id>.json   one cached response per ticket (written as it goes)
  drafts.json            all 30, written only by the full run
"""

from __future__ import annotations

import argparse
import csv
import datetime as dt
import hashlib
import json
import os
import random
import time
from collections import defaultdict

from realworld_common import PROJECT_ROOT, die

BASE = os.path.join(PROJECT_ROOT, "data", "external_zenodo_7648117")
ROUTED = os.path.join(BASE, "routing_results.csv")
TICKETS = os.path.join(BASE, "english_tickets.csv")
MAPPING = os.path.join(BASE, "category_mapping.json")
RAW_DIR = os.path.join(BASE, "drafts_raw")
OUT = os.path.join(BASE, "drafts.json")

N_DRAFTS = 30          # the author's cap
SEED = 42
GEMINI_CALL_DELAY_SEC = 4.0   # project invariant: >= 4 s


def select_sample():
    with open(MAPPING, encoding="utf-8") as fh:
        spec = json.load(fh)
    if spec.get("status") != "confirmed":
        die("category_mapping.json is not confirmed")
    mapping = spec["mapping"]
    with open(ROUTED, newline="", encoding="utf-8") as fh:
        routed = {r["ticket_id"]: r for r in csv.DictReader(fh)}
    with open(TICKETS, newline="", encoding="utf-8") as fh:
        texts = {r["id"]: r["text"] for r in csv.DictReader(fh)}

    seen, strata = set(), defaultdict(list)
    for tid in sorted(routed, key=int):
        r = routed[tid]
        if mapping[r["source_category"]] is None:
            continue
        if texts[tid] in seen:
            continue
        seen.add(texts[tid])
        if r["escalated"] == "False":
            strata[r["source_category"]].append(tid)

    names = sorted(strata)
    alloc = {s: N_DRAFTS // len(names) for s in names}
    for s in sorted(names, key=lambda s: (-len(strata[s]), s))[
            :N_DRAFTS - sum(alloc.values())]:
        alloc[s] += 1
    rng = random.Random(SEED)
    picked = []
    for s in names:
        if len(strata[s]) < alloc[s]:
            die(f"stratum {s} has only {len(strata[s])} eligible tickets")
        picked += rng.sample(strata[s], alloc[s])
    rng.shuffle(picked)
    if len(picked) != N_DRAFTS or len(set(picked)) != N_DRAFTS:
        die("sample size is not exactly 30 distinct tickets")
    pop = {s: len(strata[s]) for s in names}
    return picked, routed, texts, mapping, alloc, pop


def retrieved_dump(retrieval):
    return [{"rank": i + 1, "id": r.id, "title": r.title,
             "description": r.description, "category": r.category,
             "resolution": r.resolution, "similarity": float(r.similarity)}
            for i, r in enumerate(retrieval.retrieved)]


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--limit", type=int, default=None,
                    help="draft only the first N of the fixed sample")
    ap.add_argument("--force", action="store_true",
                    help="overwrite drafts.json (cached responses are reused)")
    args = ap.parse_args()
    if args.limit is None and os.path.exists(OUT) and not args.force:
        die(f"Output exists (use --force to overwrite):\n    {OUT}")

    picked, routed, texts, mapping, alloc, pop = select_sample()
    todo = picked[:args.limit] if args.limit else picked
    print(f"[sample] {len(picked)} tickets; allocation {alloc}; "
          f"eligible population {pop}")

    from src.agent import pipeline
    from src.agent.artifacts import load_artifacts
    from src.agent.config import config_fingerprint, settings
    from src.agent.resolver import build_prompt
    from src.agent.schemas import TicketIn

    art = load_artifacts(require_gemini=True)
    fp = config_fingerprint()
    os.makedirs(RAW_DIR, exist_ok=True)

    planned = []
    for tid in todo:
        r = routed[tid]
        ticket = TicketIn(title=texts[tid], description="", ticket_id=tid)
        res0 = pipeline.run(ticket, artifacts=art, generate_resolution=False,
                            emit_log=False)
        cls = res0.classification
        if (res0.decision.escalated or str(cls.category) != r["cascade_category"]
                or int(cls.tier) != int(r["tier"])):
            die(f"ticket {tid} no longer reproduces its routing_results row")
        prompt = build_prompt(ticket, res0.retrieval)
        h = hashlib.sha256(prompt.encode("utf-8")).hexdigest()
        cache = os.path.join(RAW_DIR, f"{tid}.json")
        hit = False
        if os.path.exists(cache):
            with open(cache, encoding="utf-8") as fh:
                hit = json.load(fh).get("prompt_sha256") == h
        planned.append((tid, ticket, res0, prompt, h, cache, hit))

    live = sum(not p[-1] for p in planned)
    print(f"[plan] {len(planned)} drafts: {len(planned) - live} cached, "
          f"{live} live Gemini call(s) ({settings.models.gemini_model})")
    if live > N_DRAFTS:
        die("planned live calls exceed the 30-draft cap")

    calls = 0
    for i, (tid, ticket, res0, prompt, h, cache, hit) in enumerate(planned, 1):
        if hit:
            print(f"  [{i:2d}/{len(planned)}] {tid:>5} cached")
            continue
        if calls:
            time.sleep(GEMINI_CALL_DELAY_SEC)
        t0 = time.perf_counter()
        res = pipeline.run(ticket, artifacts=art, generate_resolution=True,
                           emit_log=False)
        calls += 1
        if res.suggestion is None:
            die(f"ticket {tid}: no draft (status {res.status}); stopping so "
                f"retries cannot multiply the call count. {calls} call(s) made.")
        if build_prompt(ticket, res.retrieval) != prompt:
            die(f"ticket {tid}: retrieval changed between the two passes")
        rec = {
            "ticket_id": tid,
            "source_category": routed[tid]["source_category"],
            "mapped_category": mapping[routed[tid]["source_category"]],
            "ticket_text": ticket.combined_text,
            "cascade_category": str(res.classification.category),
            "tier": int(res.classification.tier),
            "top_similarity": float(res.retrieval.top_similarity),
            "retrieved": retrieved_dump(res.retrieval),
            "draft": res.suggestion.text,
            "llm_model": res.suggestion.llm_model,
            "prompt_sha256": h,
            "config_fingerprint": fp,
            "elapsed_s": round(time.perf_counter() - t0, 3),
            "generated_at": dt.datetime.now().isoformat(timespec="seconds"),
        }
        with open(cache, "w", encoding="utf-8") as fh:
            json.dump(rec, fh, indent=2, ensure_ascii=False)
        print(f"  [{i:2d}/{len(planned)}] {tid:>5} drafted "
              f"({rec['cascade_category']}, sim {rec['top_similarity']:.4f})")
    print(f"[gemini] {calls} live call(s) this run")

    if args.limit:
        print("[dry run] --limit given; drafts.json not written")
        return
    records = []
    for tid in picked:
        with open(os.path.join(RAW_DIR, f"{tid}.json"), encoding="utf-8") as fh:
            records.append(json.load(fh))
    with open(OUT, "w", encoding="utf-8") as fh:
        json.dump({"sample": {"n": len(picked), "seed": SEED,
                              "allocation": alloc,
                              "eligible_population": pop,
                              "population": "distinct mapped texts, not "
                                            "escalated, primary mapping"},
                   "drafts": records}, fh, indent=2, ensure_ascii=False)
    print(f"[write] {os.path.relpath(OUT, PROJECT_ROOT)}")


if __name__ == "__main__":
    main()
