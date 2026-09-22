# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this is

An IT support ticket triage agent: classify a ticket into one of 7 teams, retrieve
similar past tickets and draft a grounded resolution via Gemini, escalate to a human
whenever confidence is too low, and separately cluster resolved tickets to flag
recurring issues for automation.

It is a research project (B.Tech final year, extending toward an IEEE-style paper),
not a production service. **The experimental results ARE the deliverable.** `README.md`
is the lab notebook — it records every calibration attempt including the ones that
failed, and why. Read the relevant README section before changing anything it describes.

## Session continuity

**Read `PROJECT_STATUS.md` first.** It carries the live state: current phase, what is
done, what is in progress, the immediate next step, open questions and known risks. This
file (CLAUDE.md) describes how the project works and does not change often;
PROJECT_STATUS.md describes where it currently is and changes every session.

The work is organised as numbered phases with an explicit review gate between each —
finish a phase, report, and stop rather than rolling into the next one. **Each phase
also opens with a plan, reviewed before any code is written.** Phases 0 (pipeline
consolidation), 1 (conformal prediction), 2 (automation-flag validation and resolution
groundedness) and 3 (Tier-1 persistence, agent boundaries, HTTP service, write-up) are
complete and pushed. So are Phase 4A (history sink + detector library), Phase
4B-1 (the drift evaluation — verdict **measured, not shipped**: an eligible
operating point exists at Signal A's calibration-conditional binomial, α=0.05,
W=100–200, and nothing was promoted into config), Phase 5A (the conformal
template-grouping correction), Phase 5B (the honest ablation — the cascade is a
latency optimisation, not an accuracy gain) and Phase 5C (zero-shot LLM
classification baselines, Gemini and a local Qwen2.5-3B). The work now follows the agreed **Phase 5–9
publication-readiness programme**, which is recorded in `PROJECT_STATUS.md`
along with the current state — read it first.

**Before ending a working session**, update `PROJECT_STATUS.md`: the last-updated date,
the last commit SHA, what moved, and what the next step is. A future session should be
able to resume from that file alone without re-deriving context from the git log.

## Environment and commands

Windows + PowerShell. Python venv lives at `venv/` (gitignored).

```powershell
.\venv\Scripts\Activate.ps1
```

`GEMINI_API_KEY` must be in `.env` at the project root. Any script that calls Gemini
needs it; the classification-only and clustering scripts do not.

### Core pipeline (in order, from a clean clone)

```powershell
python data/generate_dataset.py                      # 4,000 synthetic tickets, seed 42
python src/classification/train_tier1.py             # persisted Tier-1 (TF-IDF)
python src/classification/train_embeddings.py        # production BGE classifier
python src/rag/build_vector_index.py                 # FAISS index + aligned metadata
streamlit run src/app/streamlit_app.py               # live demo
uvicorn src.service.api:app --port 8000              # HTTP service (Phase 3C)
```

The service must be started **from the project root** — `src/service/` has no
`__init__.py`, so `src.service.api` resolves as a namespace package the same way
`src.experiments.*` does. `GET /health` returns `config_fingerprint()` over the
wire, which is how you tell a deployment is serving stale artifacts.

**`POST /triage` defaults to `generate_resolution=false`.** The Gemini free tier
is 500 calls/day and a looping workflow would drain it, so spending quota is
opt-in. `/agents/resolve` spends quota by definition — it is the only endpoint
that does.

### Container and CI (Phase 8B)

```powershell
docker build -t ticket-triage:8b .
docker run --rm -p 8000:8000 ticket-triage:8b        # no API key needed
python src/service/verify_deployment.py --base-url http://localhost:8000
```

The image **builds its own artifacts** in the documented order at build time.
`.dockerignore` excludes the *committed* `data/ticket_index*.faiss`,
`data/ticket_metadata*.json` and `**/*.joblib` on purpose, so a container can
never be serving a copy of someone's local files — it must build them. **Write
nested ignore patterns with `**/`**: `.dockerignore` matches with Go's
`filepath.Match`, where `*` does not cross `/`, so a bare `*.npy` excludes
nothing under `data/` — that mistake shipped occurrence #7 of the recurring bug
class. A `RUN` guard immediately after `COPY` now fails the build if any
pre-built artifact reaches the context, so the ignore rule is not the only thing
enforcing this. The base
is pinned by digest to `python:3.14.3-slim` (the interpreter the published
results were produced under), not the floating `3.14-slim`. `train_distilbert.py`
is **not** run: ~90 min on CPU and on no production path. The build's last step
calls `src.service.api.startup()`, so all three artifact guards run before the
image exists — a stale-artifact image fails the build.

`verify_deployment.py` is the committed answer to "a number quoted at a gate
comes from a committed script". It compares `/health`'s `config_fingerprint`
against this checkout's, confirms the no-key surface (`/agents/resolve` → 503,
everything else 200), and runs `adv_08` over HTTP against **two** recorded
derivations — the 6-dp CSV and the full-precision golden. Never widen its
tolerances: a moved routing number is a finding.

CI is two workflows, neither able to spend quota. `ci.yml` (push/PR) runs
`pytest -m "not slow"` **and `tests/test_paper_artifacts.py` by path**, because
the marker would otherwise skip all of paper parity. `gates.yml`
(`workflow_dispatch` only) runs the full suite, the adversarial gate, the
ablation baseline — both CSVs checked with `git diff --ignore-cr-at-eol
--exit-code`, not by eye — and a container job that builds the image and
verifies it. **`--ignore-cr-at-eol` is required, not laziness**: Python's `csv`
module writes `lineterminator='\r\n'` on every platform, so on a Linux runner an
unchanged file is rewritten CRLF against an LF blob. The flag ignores a CR at
end of line and nothing else; a changed digit still fails.

### Tests

```powershell
pytest                     # full suite, offline, ~50s
pytest -m "not slow"       # fast subset, no model loading
pytest tests/test_pipeline_parity.py -v
```

`pytest.ini` deselects `-m gemini` by default so the suite never spends API quota.
Markers: `slow` (loads BGE + fits Tier-1), `gemini` (live API call).

**`-m "not slow"` skips ALL of paper parity.** Every test in
`tests/test_paper_artifacts.py` is marked `slow`, although it loads no model and
finishes in under four seconds — it only reads committed files under `data/`.
Run it by path whenever you run the fast subset, which is what `ci.yml` does:

```powershell
pytest -m "not slow"                      # then, always:
pytest tests/test_paper_artifacts.py      # paper parity, ~4s, no model
```

The standalone regression gate still exists and produces the CSV report:

```powershell
python src/experiments/test_adversarial_escalation.py
```

9 hand-written tickets that must all escalate or proceed correctly. **Any change to a
live threshold, embedding model, or retrieval path must be re-confirmed 9/9 here before
being committed.** This has already caught two real bugs.

**Golden parity is the refactor safety net.** `tests/goldens/*.json` record the exact
routing decisions for both fixed benchmark sets. Any change that moves a number there
is a regression unless it is deliberate. Regenerate only on purpose:

```powershell
python tests/capture_goldens.py
```

**Paper parity is the same safety net for the write-up (Phase 8A).** `paper/`
is GENERATED -- never edit anything in it by hand. `tests/test_paper_artifacts.py`
rebuilds it and fails if any `NUMBERS.md` value, table CSV or figure-data CSV
moves; PDF/PNG bytes are deliberately not compared. `paper/PROVENANCE.json`
carries the sha256 of every source file read, so a failure says which bug it
is: a result file changed (the paper is stale) or the builder changed.

```powershell
python src/experiments/build_paper_artifacts.py --force
```

### Experiments

```powershell
# Ablation — quantifies what each safety net is actually worth.
# --mode no-cascade is Tier-1 answering everything, so baseline minus
# no-cascade measures the BGE-vs-TF-IDF REPRESENTATION gap, NOT the value of
# cascading. --mode tier2-only is the control that isolates the cascade.
# --set defaults to benchmark45 and keeps the historical CSV names; any other
# set writes ablation_{mode}_results_{set}.csv, so nothing published moves.
python src/experiments/run_ablation_study.py --mode baseline
python src/experiments/run_ablation_study.py --mode no-cascade
python src/experiments/run_ablation_study.py --mode tier2-only
python src/experiments/run_ablation_study.py --mode no-rag          # no --set
python src/experiments/run_ablation_study.py --mode baseline   --set deployment175
python src/experiments/run_ablation_study.py --mode tier2-only --set deployment175

# Phase 5B — the paired test and the latency measurement (offline, no quota).
# Both refuse to overwrite their outputs without --force.
python src/experiments/compare_cascade_vs_tier2.py                  # exact McNemar
python src/experiments/compare_cascade_vs_tier2.py --set deployment175
python src/experiments/measure_inference_latency.py                 # warm, batch size 1

# Phase 5C -- zero-shot LLM baselines. TWO BACKENDS, ONE PROMPT: build_prompt()
# is backend-agnostic, so both arms get byte-identical text by construction.
# Gemini SPENDS QUOTA (59 calls for both sets) -- always --limit 3 first. Every
# raw response is cached under data/zeroshot_raw/ keyed by prompt hash, so a
# re-score costs nothing and editing the prompt invalidates old answers.
python src/experiments/run_zeroshot_baselines.py --backend gemini --limit 3
python src/experiments/run_zeroshot_baselines.py --backend gemini --set both

# Ollama is local and free. --model is REQUIRED in practice: the default is
# still the 7B, and 5C's published arm is the 3B. The run aborts rather than
# swap; --allow-low-ram overrides and stamps every response low_ram_override.
python src/experiments/run_zeroshot_baselines.py --backend ollama \
    --model qwen2.5:3b-instruct --set both

# Paired exact McNemar. --against tier2 (default) is vs the trained classifier;
# --against zeroshot compares the two zero-shot arms to each other.
python src/experiments/compare_zeroshot_vs_tier2.py --backend gemini --set benchmark45
python src/experiments/compare_zeroshot_vs_tier2.py --backend ollama \
    --model qwen2.5:3b-instruct --set benchmark45
python src/experiments/compare_zeroshot_vs_tier2.py --backend ollama \
    --model qwen2.5:3b-instruct --set benchmark45 \
    --against zeroshot --other-backend gemini

# Wilson CIs, dual unparseable accounting, per-category confusion, latency.
# --prompt-check-against PROVES the two arms saw the same prompt, ticket by
# ticket, from the cached prompt hashes -- offline, no quota.
python src/experiments/summarize_zeroshot_baselines.py --backend ollama \
    --model qwen2.5:3b-instruct --set both --prompt-check-against gemini

# RAG similarity threshold calibration (measurement only; does not edit production)
python src\classification\generate_ood_calibration_set.py     # ~3.5 min, ~45 Gemini calls
python -m src.experiments.calibrate_rag_similarity_threshold  # note: -m, not a path

# Batch intake / traffic skew
python src/experiments/simulate_ticket_intake.py
python src/experiments/process_ticket_batch.py

# Training-data skew
python src/experiments/generate_skewed_datasets.py
python src/experiments/run_imbalance_sweep.py

# Conformal prediction (measurement only; offline, no Gemini quota)
python -m src.experiments.calibrate_conformal

# Phase 6A -- conformal deferral vs a confidence threshold. Risk-coverage
# curves, AURC, and the real alpha-indexed conformal operating points.
# Offline, no quota. Refuses to overwrite its outputs without --force.
# Fatal if any curve's coverage-1.0 endpoint disagrees with the published
# accuracy (16/45, 33/45, 91/175, 132/175).
python src/experiments/compare_deferral_rules.py

# Phase 7A -- external-validity feasibility (offline, no quota, no Gemini).
# Downloads Tobi-Bueck/customer-support-tickets at a PINNED revision (fatal on
# mismatch) into data/external_tobibueck/, English subset only, NEVER mixed
# with our artifacts -- an isolation check hashes our dataset and benchmarks
# before and after. The profile refuses to overwrite without --force, and
# --force REUSES the cached embeddings: re-encoding is the separate
# --reencode flag, because the BGE pass is ~92 minutes on CPU.
python src/experiments/fetch_external_dataset.py
python src/experiments/profile_external_dataset.py

# Phase 7B -- does Finding 1 replicate on a different generator? (offline, no
# quota). Reuses 7A's embedding cache and fits BOTH tiers FRESH on the external
# training split -- a declared rule-7 exception, since the production artifacts
# were fitted on our 4,000 rows over a different label space. Step 1 re-encodes
# 64 seeded rows and is FATAL unless they match the cache: a row-count check is
# exactly what a misaligned cache passes. The isolation check is widened to 22
# files, including models/, tests/goldens/ and every published conformal,
# deferral and ablation CSV. Both scripts refuse to overwrite without --force.
python src/experiments/run_external_conformal_shift.py
python src/experiments/compare_deferral_rules_external.py    # run it second

# Phase 7C -- Finding 1 under a PARAPHRASE shift (offline, no Gemini quota;
# local Ollama qwen2.5:3b-instruct, ~2h for 299 tickets at ~18s each).
# ALWAYS --limit 3 first and read the paraphrases before spending the full run.
# UNLOAD THE MODEL BEFORE THE GUARD STEP: Ollama holds its weights for 5 min
# after the last call, so loading BGE for the similarity guard puts both models
# resident at once -- that overlap is this phase's memory peak and it got the
# run killed once. `ollama stop qwen2.5:3b-instruct`, confirm /api/ps is empty,
# then re-run with --cached-only, which refuses every live call and makes a
# cache miss or prompt-hash mismatch FATAL rather than a silent regeneration.
python src/experiments/paraphrase_external_tickets.py --limit 3
python src/experiments/paraphrase_external_tickets.py
ollama stop qwen2.5:3b-instruct
python src/experiments/paraphrase_external_tickets.py --cached-only
python src/experiments/run_paraphrase_shift_conformal.py

# Phase 6B -- weighted conformal under shift (offline, no quota). Measurement
# only; refuses to run if settings.conformal.enabled is True, and refuses to
# overwrite its CSV without --force. FATAL unless the unweighted rows reproduce
# Finding 1 exactly (Tier-1 -0.233333, Tier-2 -0.011111 at alpha=0.10), checked
# both against a constant and against conformal_calibration_results.csv.
python src/experiments/run_weighted_conformal.py

# Phase 4A -- drift reference (offline, no quota). Refuses to overwrite
# without --force; fatal unless it reproduces the published Phase 1
# OOD/adversarial detection results exactly.
python src/experiments/build_drift_reference.py

# Signal B's reference (Phase 4B). The in-domain reference above escalates
# 0/175, which makes an escalation-rate test against it degenerate; the
# deployment-distribution set escalates 39/175 (22.3%). That 22.3% is the
# rate of Gemini-generated benchmark-register tickets, NOT a measured
# production escalation rate. Fatal unless the pipeline's decision.escalated
# count and a direct below-threshold count agree.
python src/experiments/build_drift_reference.py --source deployment

# Phase 4B -- drift detector evaluation: null false-alarm rate AND power
# (offline, no quota, ~108s for --smoke). --smoke writes to a temp dir and
# skips the strict Beta-law assertion, so only the full run's numbers count.
# Refuses to overwrite data/drift_evaluation_* without --force.
python src/experiments/evaluate_drift_detection.py --smoke
python src/experiments/evaluate_drift_detection.py

# Regenerate the deployment-distribution calibration set (~360 Gemini calls,
# ~30 min). ALWAYS dry-run first -- it costs ~14 calls and catches a bad
# prompt before the full budget is spent.
python -m src.classification.generate_deployment_calibration_set \
    --per-category 1
python -m src.classification.generate_deployment_calibration_set

# Phase 6C -- the retrieval-sufficiency gate. Step 1 is OFFLINE and persists
# the context precisely so steps 2-4 load NO embedding model: BGE and a local
# generator can never be co-resident (the 7C memory lesson, enforced
# structurally). Step 2 SPENDS QUOTA -- always --limit 3 first. Every raw
# response is cached under data/sufficiency_raw/ keyed by prompt hash, with
# temperature recorded, so a re-score costs nothing.
python src/experiments/build_sufficiency_context.py
python src/experiments/run_sufficiency_autorater.py --backend gemini --limit 3
python src/experiments/run_sufficiency_autorater.py --backend gemini

# The stability secondary. --only writes *.repeats.json and --limit writes
# *.partial.json, so neither can overwrite the other or the full pass -- the
# dry-run record is the provenance for what was seen before the full pass.
python src/experiments/run_sufficiency_autorater.py --backend gemini \
    --repeats 3 --only S002,S029

# Cross-family arm, zero quota. THEN `ollama stop qwen2.5:3b-instruct` and
# confirm /api/ps is empty before anything touches BGE again.
python src/experiments/run_sufficiency_autorater.py --backend ollama \
    --model qwen2.5:3b-instruct
python src/experiments/score_sufficiency_gate.py          # offline, 0 calls

# Phase 8A -- EVERY paper table, figure and number, from committed result
# files only. Offline: no Gemini, no Ollama, no model load, no training, no
# experiment re-run. Writes ONLY under paper/ and refuses to overwrite it
# without --force. The build is byte-for-byte deterministic, PDF and PNG
# included, so a rebuild differs only when a source file changed.
python src/experiments/build_paper_artifacts.py
python src/experiments/build_paper_artifacts.py --force
python src/experiments/build_paper_artifacts.py --out DIR --no-render

# Phase 8A.1 -- the writers that gave the "no committed source" numbers a
# source. Offline, no quota. Each re-runs its ORIGINAL configuration; the
# only new behaviour is that it now writes a result file.
#   generalization_test.py    -> data/baseline_tfidf_benchmark14.csv
#     Two arms. full4000 is the ORIGINAL config (fit on all 4,000 rows) and
#     the one the paper cites; split3200 is an 80/20-fit secondary arm added
#     in 8A.1 and is a NEW measurement, never the published figure.
#   train_baseline_tfidf.py   -> data/baseline_tfidf_indistribution.csv
#   train_cascade.py          -> data/cascade_threshold_sweep.csv
#                             +  data/cascade_calibration_attempts.csv
#     Loads BGE for the 14 + 45 benchmark tickets; the 4,000-row embedding
#     cache makes the rest a cache hit. Fits both tiers locally -- a declared
#     rule-7 exception, because reproducing the original calibration is the
#     point.
python src/classification/generalization_test.py
python src/classification/train_baseline_tfidf.py
python src/classification/train_cascade.py

# DistilBERT. WITHOUT --backup-existing the script RESUMES from whatever is
# in models/distilbert_ticket_classifier/ and re-reports those checkpoints;
# with it, the directory is MOVED aside (never deleted) and training starts
# from scratch. --append-metrics keeps both runs in one CSV.
python src/classification/train_distilbert.py                    # evaluate existing
python src/classification/train_distilbert.py --backup-existing \
    --run-label fresh_retrain --append-metrics                   # retrain, ~25 min CPU

# Resolution clustering -> automation flagging
python src/experiments/join_scenario_ground_truth.py
python src/experiments/explore_resolution_clustering.py
python src/experiments/calibrate_resolution_clustering.py
python src/experiments/calibrate_resolution_clustering_percategory.py
python src/experiments/flag_automation_candidates.py          # the production feature

# Phase 2A -- automation-flag validation (offline, no Gemini quota)
python src/experiments/build_flag_validation_set.py           # add --pilot for the 12-pair probe
python src/experiments/score_flag_validation_set.py           # add --pilot to score it

# Phase 2B -- resolution groundedness. SPENDS GEMINI QUOTA: one call per
# draft, then one per judge verdict (33 + 33). ALWAYS dry-run first.
python src/experiments/build_groundedness_set.py --limit 3    # prompt check, 3 calls
python src/experiments/build_groundedness_set.py              # 33 calls
python src/experiments/run_groundedness_judge.py              # 33 calls
python src/experiments/score_groundedness_set.py              # offline, 0 calls
```

`calibrate_rag_similarity_threshold.py` is the one script run as a module (`-m`).
`src/experiments/` and `src/classification/` have no `__init__.py` — `src.experiments.*`
resolves as an implicit namespace package, so it only works from the project root.
(`src/agent/` does have one, and is a regular package.)

## Architecture

Sequential pipeline with two confidence gates, plus one offline analysis path.
As of Phase 3B the stages are separated into agents with declared dependencies behind
an orchestrator; Phase 3C put each agent behind its own HTTP endpoint in
`src/service/api.py`. **`POST /triage` still orchestrates in-process** — it calls
`pipeline.run()`, not the endpoints — because golden parity must not depend on a
running server and network hops would add failure modes to the measured path. So the
agents are independently *addressable*, not independently *running*. Describe it that
way; the honest claim is a sequential pipeline with agent boundaries and an HTTP
surface, not a distributed system.

**The failure boundary is the service's, not the library's.** `/triage` maps an
`AgentError` to HTTP 200 with an escalation envelope (the ticket needs a human; a 5xx
would wrongly invite a retry), while anything that is *not* an `AgentError` returns
500. The library still raises — deliberately, because in a calibration run a
`RetrievalError` silently becoming an escalation row would corrupt the result quietly.
`PipelineResult.classification` is required, so an escalation envelope that needs no
classification is the only shape that avoids inventing a category.

**All inference lives in `src/agent/`** — one implementation, consolidated from what
were four independent copies:

```
src/agent/
├── config.py        frozen typed config + CALIBRATION_PROVENANCE + config_fingerprint()
├── schemas.py       pydantic models for every stage; Tier/EscalationReason enums
├── errors.py        typed exceptions; ONE Gemini error ladder (was three)
├── logging_setup.py structured JSON decision logs; opt-in JSONL sink (off by default)
├── artifacts.py     THE loader, with all three hard guards; load_drift_reference()
├── conformal.py     split conformal + conformal p-values (measurement only)
├── drift.py         drift detector: pure functions over records (measurement only)
├── classifier.py    cascade Tier-1 -> Tier-2          (stage implementation)
├── retriever.py     FAISS retrieval                    (stage implementation)
├── resolver.py      prompt + Gemini call with retry    (stage implementation)
├── agents.py        the three agents: name + requires + run()
├── orchestrator.py  the sequence and BOTH gates; run(TicketIn) -> PipelineResult
└── pipeline.py      the public façade over orchestrator.run()

src/service/
└── api.py           FastAPI: per-agent endpoints, /policy/rag-gate, /triage
```

`streamlit_app.run_pipeline()`, `test_adversarial_escalation.run_ticket_through_
pipeline()` and `process_ticket_batch` are now thin adapters over `pipeline.run()`.
Add new consumers the same way — never re-implement the orchestration.

**Agents vs orchestrator (Phase 3B).** `classifier.py`, `retriever.py` and
`resolver.py` hold the parity-critical stage logic and are wrapped, not absorbed,
by the agents in `agents.py`. An agent declares `requires` — the attributes of
`Artifacts` it cannot work without — and those are validated at construction, so a
missing or misspelled dependency fails before any routing happens. **An agent
decides nothing**: no agent reads a threshold or knows what runs after it. Every
gate lives in `orchestrator.py`, which also records a `StepTrace` per agent
(including the ones deliberately skipped — a `skipped` resolution step is the
positive evidence that no LLM call was made).

```
ticket text
    |
    v
[Tier 1: TF-IDF + LogReg] --confidence >= 0.50--> category
    |  below threshold
    v
[Tier 2: BGE + LogReg] ------------------------->  category
    |
    v
[FAISS top-5 retrieval] --top1 sim >= 0.67--> [Gemini drafts grounded resolution]
    |  below threshold
    v
ESCALATE TO HUMAN  (Gemini is never called)

offline: resolved tickets -> embed resolution text -> union-find clustering
         at 0.80 -> automation candidates for human review
```

Key points that are not obvious from any single file:

- **Gemini never decides the category.** Classification is entirely the trained
  models'. Gemini's only job is resolution generation, grounded in retrieved tickets.
- **The two gates are independent and calibrated separately.** The cascade threshold
  depends only on Tier-1/TF-IDF confidence, so swapping the Tier-2 embedding model
  does not require recalibrating it. The RAG threshold *is* embedding-model-specific
  and must be recalibrated on any model swap.
- **`src/agent/config.py` is the single source of truth for every calibrated
  constant.** `suggest_resolution.SIMILARITY_THRESHOLD` still exists and still works,
  but it now re-exports `settings.rag.similarity_threshold` so the older experiment
  scripts keep running unchanged. Never redeclare a threshold anywhere else.
- **Never write `getattr(obj, "SOME_THRESHOLD", <number>)`.** That pattern silently
  reverted the human-escalation gate to a dead MiniLM value. `tests/
  test_no_silent_fallback.py` fails the build if it reappears.
- **`scenario_id` ground truth** for clustering evaluation is recovered from the
  dataset generator's own internal `random.choice()` sequence by
  `join_scenario_ground_truth.py` — it is reproducible only because the seed is fixed.

## Calibrated constants — do not casually change

These are measured values, each backed by a documented calibration exercise. The point
of the project is that they are evidence-backed rather than guessed. Never re-derive one
casually, and never expose one as a trivially-overridable CLI flag.

Every one of them lives in `src/agent/config.py`, which is frozen — assigning to one
raises `ValidationError`. Each carries its evidence in `CALIBRATION_PROVENANCE`, and
`tests/test_config.py` fails the build if a value drifts or loses its provenance entry.

| Constant | Value | Gates production? | Config path |
|---|---|---|---|
| RAG similarity threshold | **0.67** | yes | `settings.rag.similarity_threshold` |
| Cascade confidence threshold | **0.50** | yes | `settings.cascade.confidence_threshold` |
| Resolution-clustering threshold | **0.80** | yes | `settings.clustering.resolution_similarity_threshold` |
| Conformal target error rate | **0.10** | **no — measurement only** | `settings.conformal.alpha` |

`settings.conformal` exists but **does not gate production** — `enabled` is
`False`, so `PipelineResult.conformal` stays `None` and golden parity holds.
Conformal is measured by `calibrate_conformal.py`; promoting it to a live gate
is a separate decision needing its own evidence.

Production embedding model is `BAAI/bge-base-en-v1.5` (768-dim) for classification, RAG
retrieval, and cascade Tier-2. Resolution-text clustering deliberately remains on
`all-MiniLM-L6-v2` — BGE has been measured there but not promoted, because no
ground-truth validation exists for automation-flag quality. See README "Pending".

Gemini model is `gemini-flash-lite-latest` via the unified `google-genai` SDK
(`from google import genai`) — not the deprecated `google-generativeai` package.

## Project-specific invariants

- **Seed 42 everywhere.** Every classification script uses the same 80/20 stratified
  split (`test_size=0.2, random_state=42`) so results stay directly comparable.
- **Benchmarks are read-only.** `NOVEL_TICKETS` in `generalization_test.py` (14 tickets)
  and `data/novel_tickets_expanded.json` (45 tickets) are fixed reference points used
  across every method comparison. Never edit or regenerate them.
- **Conformal coverage transfers for Tier-2 but not Tier-1 — ON OUR CORPUS,
  and only there.** Calibrated on the same 175 tickets at the same alpha,
  TF-IDF loses 23.3 coverage points on the 45-ticket benchmark while BGE loses
  1.1 (noise band 4.5). Do not quote a conformal guarantee for a lexical model
  on out-of-template text. **Phase 7B tried this on the external corpus and it
  DID NOT REPLICATE**: under a measured shift, Tier-1's gap is +0.0009 against
  Tier-2's −0.0072 at α=0.10, band 0.0165 — the two tiers are
  indistinguishable. Finding 1 is **not withdrawn and not softened**; its
  **scope** narrows to our corpus. **Never write "7B refutes Finding 1"** —
  that corpus has no representation gap to find (Tier-1 34.8% vs Tier-2 37.3%,
  2.6 points, against 35.6 on ours), which weakens 7B as evidence and is a
  limitation, not a rescue.
- **7C is BLOCKED, and a blocked arm is not a null.** The pre-registered ≥0.95
  degeneracy rule fired (TF-IDF-space domain AUC **0.9972**), so **no verdict
  was drawn on the primary** and the numbers sit in the CSV marked post-hoc, as
  6B's BGE arm does. **After 7B and 7C, Finding 1 has not been shown either to
  hold or to fail outside its original corpus** — write that, not "it failed
  twice".
- **A pre-registered rule is NOT revised after seeing the results, however good
  the critique.** 7C's ≥0.95 rule was copied from 6B, where it guarded a
  **density-ratio estimate**; 7C computes no density ratio and uses the AUC only
  as a manipulation check, where a near-1.0 value means the manipulation was
  *strong*. The critique is sound and the gate still **declined to unblock** —
  it is recorded as a **specification error**, not acted on. The counter-reading
  is recorded too: a rewrite a classifier identifies with near-certainty may be
  a *different corpus* rather than a shifted one. **Lesson: a degeneracy rule
  must be justified by what the specific design estimates, never copied across
  designs.**
- **Finding 1's mechanism IS confirmed in accuracy, even though 7C is
  blocked.** The paraphrase shift cost Tier-1 **10.5 accuracy points**
  (0.3776 → 0.2727, 108 → 78 of 286) against Tier-2's **2.5** (0.3776 → 0.3531,
  108 → 101) — about 4×. Bootstrapped before being written down:
  difference-in-differences **−0.0804, 95% CI [−0.1364, −0.0210]**, excluding
  zero (10,000 paired draws, seed 42). 7B's version shift produced no such
  contrast, which supports the diagnosis that it was the wrong kind of shift.
  **Still POST-HOC** — it does not replace the blocked primary.
- **Finding 1's paper framing is FIXED (7C gate), and its three clauses travel
  together:** measured on our corpus; **external replication inconclusive** (7B
  wrong shift type, 7C blocked by its own rule); **post-hoc evidence of the
  mechanism in accuracy** on the external corpus. Never quote one clause alone.
- **Phase 7 is CLOSED — no further external experiments in this programme.**
  The set-size question (coverage may be buffered by 4.9–6.9-label prediction
  sets, so Finding 1 may need both a representation contrast *and* small sets)
  is **future work, not a 7D**.
- **A test-arm coverage band must include the test-sampling term.** Finding 1
  and 7B quote `coverage_sd(alpha, n_cal)` alone, which is fine on a
  10,441-ticket arm. At 7C's n=286 the test-sampling sd (0.0173 at α=0.10)
  **dominates** the calibration term (0.0082), so 7C reports a combined band.
  Quoting the calibration term alone on a small test arm understates
  uncertainty by roughly 2×.
- **Never measure a local model while another model is resident.** Ollama holds
  its weights for five minutes after the last call, so loading BGE straight
  after a generation run puts both in RAM at once. That overlap is what got
  7C's run killed for memory. `ollama stop <model>`, confirm `/api/ps` is
  empty, then do the embedding work.
- **7B's deferral reading must never be merged with 6A's.** 6A, on our data:
  "no evidence either way on the gated axis" — that wording still stands. 7B,
  on the external test arm where the comparison finally resolves (642 tickets
  at the gate against 6A's four): **3 of 6 comparisons signal and all three
  favour the confidence incumbent**, so conformal deferral is *worse* there.
  Different corpus, transplanted gate, ~35% label accuracy — it does not
  retro-license a claim about ours. Conformal still cannot be *operated* at the
  gate in either place.
- **A number quoted at a gate must come from a committed script.** 7A's two
  design domain AUCs (0.8584; 0.6706–0.9316) were computed in an interactive
  session, never committed, and **do not reproduce** — 7B's specified
  recomputation gives 0.8727 raw / 0.8472 operating, and five variants chasing
  the recorded value span 0.8637–0.8727. The conclusions were unaffected
  because every value cleared the same threshold, but the figures were wrong in
  print for a day. This is the concrete motivation for Phase 8A.
  **Phase 8A is the structural answer and it is now in force:** every number
  the paper may use lives in `paper/NUMBERS.md` with its source file and the
  command that regenerates it, emitted through one `emit()` choke point that
  refuses a value whose source does not exist. `tests/test_paper_artifacts.py`
  runs an **AST lint that fails if any `emit()` call passes a numeric
  literal** -- so a retyped number cannot reach the paper surface. A number
  that is not in `NUMBERS.md` is not a number the paper may use.
- **Two conventions for a 95% z coexist in this repo**, and that is recorded
  rather than silently unified: `1.96` in `score_groundedness_set.py` and
  `score_sufficiency_gate.py`, the exact quantile `1.959963984540054` in
  `summarize_zeroshot_baselines.py`. The implementations are otherwise
  algebraically identical (the 8A build asserts it) and differ by ~3.5e-6, so
  no published figure moves at reported precision. The paper adopts 1.96, and
  **a CI a source file already carries is read verbatim, never recomputed** --
  otherwise a published interval could be restated under a different z without
  anyone noticing.
- **Phase 8A.1 closed four of 8A's five "no committed source" numbers, and
  one of the four did not reproduce.** Each script that printed a number now
  writes a result file, and each was re-run in its ORIGINAL configuration.
  Reproduced exactly: DistilBERT's **7/14**, the cascade
  threshold-by-target-accuracy sweep (1.00 / 0.50 / 0.50 at the 90/80/70
  bars), and the **5.7% (10/175)** self-retrieval rate. **Did NOT reproduce:
  the TF-IDF baseline's 14-ticket score, which is 6/14, not 7/14** —
  corroborated by three independent derivations (the full-4,000-row fit, an
  80/20-split fit, and the persisted production Tier-1 artifact). **Never
  quote TF-IDF 7/14**; it is on the do-not-cite list, and the likely cause —
  that it was measured on the 1,000-ticket corpus and never re-measured after
  the scale-up — is recorded as an unconfirmable hypothesis, not a cause.
  **DistilBERT on the 45-ticket benchmark is a NEW measurement**, tagged
  `new-measurement`, never written up as a reproduction.
- **One documented number still has no committed source**: cascade
  calibration attempt 2's "34 of 35 tickets in one bucket". The 35-ticket
  hand-written set was never committed and is absent from every revision in
  the repository's history, so it cannot be re-run. It is recorded as
  `status=no_artifact` in `data/cascade_calibration_attempts.csv` and listed
  under "no committed source" in `paper/NUMBERS.md`. Do not invent a source
  for it.
- **The reliability/ECE numbers are a CEILING effect, not calibration.** On
  the 500-ticket in-distribution batch the observed accuracy is 1.0 in every
  bin of both tiers, so Tier-1's 0.1122 and Tier-2's 0.0992 are entirely the
  distance from confidence to a ceiling and both tiers are **under-confident**
  there. Never write "over-confident", and never read these as evidence of
  calibration on real traffic.
- **The 175-ticket calibration set cannot be de-contaminated.** Its tickets are
  paraphrases of training rows, but memorisation is template-level: **66
  templates, ~62 rows each, and the set touches 62 of them. Removing source
  rows changes nothing; removing whole templates would leave 210/4000 rows.** Do
  not "fix" this by filtering — it needs deployment-distribution data.
  (Corrected in Phase 5A; the published figures were 12 / ~430 / 11 / 40, from
  grouping by `scenario_id` alone. Conclusion unchanged.)
- **A template is `(category, scenario_id)`, never `scenario_id` alone.**
  `scenario_id` is an index *within* a category — `data/generate_dataset.py:634`
  says so — so grouping by it alone silently merges all seven categories'
  templates (66 → 12) and inflates rows-per-template (62 → 430). The
  per-category clustering scripts are safe because they filter to one category
  first; anything pooling across categories must use the compound key.
  `tests/test_contamination_structure.py` pins this.
- **Generated sets need a near-duplicate guard, not just a label check.** The
  first deployment calibration set had 21 near-duplicate pairs because
  tightening a scope anchor narrowed the scenario space. Duplicated
  calibration points skew the conformal quantile. Scope anchors buy label
  accuracy at the cost of diversity -- all three Gemini generators in this
  project use that pattern, so check diversity whenever you tighten one.
- **The cascade is a latency optimisation, not an accuracy gain.** Measured in
  Phase 5B against the control that was missing: cascade 32/45 vs Tier-2-only
  33/45 on the benchmark and 131/175 vs 132/175 on the deployment set — one
  ticket worse on each, exact McNemar p = 1.000 both times. It saves 8–18% of
  median per-ticket latency (Tier-2 costs 151× Tier-1: 156.40 ms vs 1.04 ms
  warm, batch size 1). **Never quote the +35.6 points as the value of
  cascading** — that is baseline minus Tier-1-only, i.e. the BGE-vs-TF-IDF
  representation gap.
- **Never write 5C as "LLMs beat the pipeline."** It is **zero-shot LLM vs
  trained classifier**, which is not like-for-like, and what it measures is
  what a 3,200-row template-generated corpus is worth on out-of-template
  phrasing. The readings differ by pair and must not be merged: Gemini is
  distinguishably better than Tier-2 on the 45 (40/45 vs 33/45, p = 0.0391);
  **Qwen2.5-3B is indistinguishable from it** (34/45, p = 1.000) — which is not
  "better"; Qwen is 6 tickets below Gemini at p = 0.070, which is not
  significant. The benchmark is Gemini-generated, so **the Qwen arm is the
  partial control for authorship** and Gemini's extra margin cannot be
  attributed to capability over authorship.
- **A retrieval-sufficiency check is NOT a usable second gate on this corpus
  (Phase 6C).** A rater seeing only the ticket and its top-5 context — never a
  draft — caught **both** 2B misses (2 of 2) but flagged **26 of the 31**
  grounded drafts as well. **Quote the false-flag count, never the catch
  alone**: a gate escalating 28 of 33 eligible tickets suppresses ~85% of
  auto-resolution to recover two bad drafts, and there is nothing to tune —
  the rater has no threshold and its verdicts are stable 3/3 at temperature 0.
  The mirror-image case matters too: of the 21 the gate escalates, the rater
  agreed on 20 and found **one (N45, top-sim 0.6397) whose context was
  adequate**, so the scalar errs in both directions. Qwen2.5-3B, on
  byte-identical prompts, is less aggressive and **misses one positive** — not
  a vendor artifact, and not a usable gate either.
- **6C is NOT an instance of the named finding, and the near-duplicate
  explanation is a REJECTED hypothesis.** It contradicts 2B's own diagnostic on
  these same tickets: 71.1% of benchmark-45 and 88.9% of adversarial-9
  retrievals carry 2+ distinct fixes, so the context is heterogeneous. The
  mechanism is a **measurement-target mismatch** — sufficiency and groundedness
  are different questions — not a corpus register mismatch. The post-hoc
  "declining drafts explain it" reading is **not supported** either: 3 of 3
  declining but 23 of 28 non-declining, so the flags are everywhere and n=3
  cannot carry a comparison.
- **6C's binding limitation is the label, not the rater.** The 2B groundedness
  labels are an **outcome proxy** — they say whether the *draft* was supported,
  not whether the *context* was sufficient. The 26 flags are false only against
  that proxy. Direct sufficiency labels are human work and have not been made.
- **In-distribution accuracy is uninformative here.** Template-generated data makes
  every model score ~100% in-distribution. Only the 14- and 45-ticket benchmarks measure
  anything real. Treat a new 100% in-distribution number as a red flag, not a success.
- **Artifacts are model-aware and filename-suffixed**
  (`ticket_index_bge-base-en-v1-5.faiss`, `ticket_classifier_bge-base-en-v1-5.joblib`).
  `artifacts.load_artifacts()` enforces three guards: `index.ntotal == len(metadata)`,
  `encoder dim == index.d == settings.models.embedding_dim`, and Tier-1's manifest
  still matching the dataset it was fitted on. Preserve all three —
  silent stale-artifact mismatch has bitten this project **four** times, most recently
  in `run_ablation_study.py`, where it reached published results. (That count is for
  the *stale-artifact* form specifically; the wider bug class it belongs to now stands
  at **six** — see "The recurring bug class".)
- **Tier-1 is a persisted artifact, not a startup refit.** It is fitted on the
  **full** 4,000 rows — never an 80/20 split, unlike every other script in
  `src/classification/` — because that is what the goldens were captured under. The
  bundle carries a manifest (dataset sha256, rows fitted, sklearn and vectorizer
  config) that `artifacts.load_tier1()` verifies; a mismatch fails loud and there is
  deliberately **no refit-on-miss fallback**. Rebuild it with
  `python src/classification/train_tier1.py`.
- **Load artifacts only through `artifacts.load_artifacts()`** (or
  `artifacts.load_tier1()` for consumers that need Tier-1 alone). Constructing a
  `SentenceTransformer`, reading the index directly, or refitting Tier-1 locally is how
  the three divergent loaders drifted apart in the first place.
  `run_ablation_study.py` was the last holdout — it carried four loaders of its
  own and refitted Tier-1 from the CSV on every run — and was migrated in Phase
  5B, verified parity-preserving first (`max |Δ tier1_conf| = 0.0`, all three
  published CSVs byte-identical after). The deliberate exceptions remain
  `calibrate_conformal.py` and `plot_calibration_curves.py`, which fit on
  leave-out subsets and must not use the production artifact.
- **Never overwrite a previous model's results file.** BGE re-runs write to
  `*_bge-base-en-v1-5.*` alongside the original MiniLM outputs, so the comparison stays
  auditable.
- **Path resolution** is always two directories up from the script's own location via
  `os.path.*`.
- **Scripts fail with clear actionable messages, not tracebacks** — the Streamlit demo
  may run live in front of an audience.
- **Gemini free tier is 15 req/min, 500/day.** Keep `GEMINI_CALL_DELAY_SEC` at 4s or
  above in any batch-calling script, dry-run before spending the full budget, and
  cache every raw response to disk so a crash or a re-scoring never re-spends
  quota. **Never run two quota-spending sub-phases on the same day** — the daily
  cap would risk a partial result mid-experiment.
- **Before committing, check for stray embedding caches:**
  `git status | Select-String "\.npy"`

## Known inconsistencies

- **An embedding-derived number is NOT bit-reproducible across platforms; a
  TF-IDF one is.** Measured in Phase 8B on adv_08. Tier-1 confidence matches
  the Windows goldens **exactly** (0.3182984770932253, delta 0.000e+00) from a
  Linux container, because TF-IDF + LogReg runs in float64. The retrieval
  similarity does not: the container returns **0.6123799085617065** against the
  goldens' **0.6123800277709961**, a fixed **-1.192e-07** offset, identical on
  every repeat — float32 BGE plus a FAISS inner product accumulate in an order
  set by the BLAS kernel and SIMD width. Nothing about the decision moves: the
  distance to the 0.67 gate is +5.762e-02, **483,352x** the offset. So
  `verify_deployment.py` checks Tier-1 exactly and the similarity at a float32
  tolerance, documented at the constant. **Do not "fix" a cross-platform
  parity failure by regenerating the goldens** — they are the Windows
  reference the published numbers were produced on.

- **Decision logs can be persisted, but nothing persists them yet.** Phase 4A added
  `configure_logging(decision_log_path=...)`, a JSONL sink for `pipeline_decision`
  records only. It is **off by default** and no script, test, service or demo turns it
  on, so no decision history exists yet — do not assume one. Enabling it (e.g. in
  `/triage`) is a separate decision with its own gate. The sink refuses a logger level
  above INFO, which would otherwise record an empty "quiet period".
- **Drift's "α by construction" is marginal only.** For the single fixed 175-ticket
  reference, the per-ticket rate of p ≤ α is Beta-distributed around α; the
  calibration-conditional bound at δ = 0.10 is 0.126 against a nominal 0.10. `drift.py`
  reports both tests. **4B-1 measured what that costs:** the marginal binomial's
  window false-alarm rate reaches 0.077–0.125 at W=200 against a nominal 0.05, so it
  is usable only at W ≤ 50; the conditional binomial holds (0.009–0.023) at every
  window. Quote only a measured rate from `data/drift_evaluation_null.csv`.
- **Three of the five drift tests must never be used as an alarm.** 4B-1 measured
  their null false-alarm rates: Signal A's **KS test** (0.077 → 0.33 as the window
  grows — it reads the p-values' 1/176 discreteness, not drift), Signal B's
  **one-sample binomial** (0.047 → 0.169, because it treats a 175-ticket estimate as
  truth; the two-sample Fisher test does not and stays near nominal), and Signal B's
  **category χ²** (0.12 → 0.46, expected-count conditions violated at n=175 over 7
  categories). All three stay in the report as descriptive reads only.
- **A drift monitor on the in-domain reference would alarm continuously.** 4B-1's
  realistic-traffic arm: deployment-register tickets — legitimate, not drift — flag at
  0.217/0.429/0.514/0.646 for α = 0.01/0.05/0.10/0.20, 4–7× the null. The binding
  constraint is what the reference is made of, not the test. Any future drift work
  needs a deployment-traffic reference first.
- **The drift reference escalates 0/175 tickets**, so the escalation-rate test is
  degenerate against it and reports `None`, not p = 0. In-domain paraphrases never
  reach the RAG gate; the reference says nothing about a deployed escalation rate.
- `process_ticket_batch.py`'s MiniLM/BGE dimension mismatch is fixed in code, but the
  script has **not** been re-run. `data/category_stores/*.csv` were produced under
  MiniLM and feed the clustering calibration behind the production 0.80 threshold;
  regenerating them under BGE would silently invalidate it. Re-running is a deliberate
  decision requiring its own re-calibration — never a side effect of another change.
- Resolution clustering is still MiniLM on purpose. BGE is measured (pooled cliff 0.90)
  but not promoted. Phase 2A built the missing ground-truth check and it returned a
  negative result: neither configuration makes a cross-template merge anywhere, so the
  two are indistinguishable on precision and differ only in recall. Promotion is now a
  product decision about review-queue capacity, not a calibration one — do not reopen
  it as a calibration question.
- **Never use an LLM judge unaudited.** Phase 2B measured one against human labels on
  the same rubric, shared verbatim so both answered the same question. Raw agreement
  was 90.9% and Cohen's κ was **−0.042**: every disagreement was the judge substituting
  *appropriateness* for *support*, in both directions. A judge is an object of study
  here, never a scaling tool, unless a human-labelled subset large enough to read every
  disagreement says otherwise. An aggregate agreement score will not reveal this.

- **Conformal deferral was measured in 6A and is NOT better than the 0.50
  confidence gate** — but the honest claim is "no evidence either way on the
  gated axis", not "conformal is worse". All 12 comparisons returned no signal
  at the live gate's *measured* operating coverage (8.9% of the benchmark,
  18.9% of the deployment set), and **3 of 4 configurations are degenerate
  there** — every rule accepts the same tickets, so the test has no resolution.
  AURC showed effects in 3 of 12 that **contradict across sets** and favour
  *margin*, not conformal. **Never quote an AURC win as a reason to promote**:
  this project gates on risk at the operating coverage, and AURC averages over
  coverages the system never runs at.
- **A near-duplicate rate is meaningless without a control.** 7A measured
  79.39% of the external corpus's rows as having a BGE>=0.95 near-duplicate,
  which reads as disqualifying until two controls are applied: random pairs sit
  at median 0.5912 (so the threshold discriminates), and **our own corpus
  scores 85.20% -- worse on every measure** (p05 0.9270 vs 0.8867; 30.3% vs
  44.2% distinct after de-duplication). High near-duplication is a property of
  template-generated corpora generally, not a flaw of that dataset. Never quote
  a redundancy or similarity rate without stating what it is high *relative to*.
- **The external dataset is INDEPENDENTLY GENERATED DATA, NOT REAL PRODUCTION
  DATA** -- its card advertises a synthetic generator from the same author. It
  tests whether findings survive a *different generator*, not reality. The
  Endava/Microsoft `all_tickets.csv` is real but anonymized/encrypted, so BGE
  cannot read it; real-but-unreadable is worse here than synthetic-but-readable.
- **In the external corpus, `version` and source-file are confounded** -- all
  11,923 version-NaN rows are `dataset-tickets-multi-lang-4-20k.csv`. They are
  one split, never two independent shift axes.
- **Weighted conformal (6B) is a PARTIAL repair, never a correction.**
  Reweighting moves Tier-1's benchmark coverage gap from -0.233 to -0.122
  (TF-IDF space, alpha=0.10) -- a change larger than the +-2 s.d. band, but the
  residual is still **5.4 s.d. below nominal**. Never write this as "the shift
  is correctable by covariate reweighting": the change cleared the band, the
  residual did not. Report both readings. Post-hoc, recovery is 47.6% against
  distribution matching's ~38% -- two independent strategies, both partial,
  which **sharpens** the named finding.
- **6B's BGE arm is degenerate and must never be quoted.** The cross-fitted
  domain AUC is **0.9908**: the in-domain and deployment sets are near-perfectly
  separable in the very space Tier-2 scores in, so the density ratio is
  ill-posed. Its numbers are the flattering ones (alpha=0.05 lands at +0.0056,
  essentially nominal) and are in the CSV as blocked, not as a result. Quoting
  them repeats 6A's AURC error. The 0.9908 itself is a quantitative statement
  of the named finding.
- **Reweighting has a cost, charged to Tier-2.** 6B's sanity check did not come
  back clean: 7 of 18 Tier-2 configurations made the gap worse and 5 were pushed
  outside the band. Report that beside any Tier-1 gain.
- **A local model must never be measured while swapping.** `OllamaBackend`
  refuses to start below a per-model RAM floor (`MODEL_RAM_FLOOR_MB`), because a
  swapping model yields a latency number that is meaningless *and* plausible.
  `--allow-low-ram` overrides it and stamps `low_ram_override: true` into every
  cached response, so such a run can never later be mistaken for a clean
  timing. An unreadable memory value is fatal too — "could not check" must not
  look like "checked and it was fine".
- **Latency is computed from live rows only.** A cached response carries the
  *original* run's `elapsed_s`, so mixing cached and live rows would report a
  number describing no run that ever happened.

## The recurring bug class

Eight occurrences so far, all the same shape: a value or artifact that is wrong for its
context, stays internally consistent, and therefore produces wrong results with no
error. The first four were model swaps; the fifth and sixth show the shape is not
limited to those — one was a routing/eligibility test, the other a grouping key.

1. Silent stale embedding cache during the BGE swap.
2. `test_adversarial_escalation.py` keeping its own MiniLM constants.
3. `process_ticket_batch.py` encoding with MiniLM against a BGE index.
4. `run_ablation_study.py` measuring the entire pre-BGE pipeline — this one reached
   published results and stood for eleven days.
5. `build_groundedness_set.py` testing `result.status == ResolutionStatus.ESCALATED`
   to decide which benchmark tickets reach the resolver. A RAG-gate escalation carries
   status `NEEDS_HUMAN_RESOLUTION`; `ESCALATED` belongs to the *filing* gate. The check
   ran clean, returned a plausible number, and silently classed all 21 escalating
   tickets as eligible — it would have spent 54 Gemini calls instead of 33 and drafted
   for tickets production never sends to the resolver, contaminating the result with
   items that do not exist in production. **Escalation is `decision.escalated`, never
   the status enum** — that field is true in every escalation branch. Caught only
   because the dry run's count (54/0) was checked against an independently measured
   number (33/21).
6. `calibrate_conformal.py`'s Step 1b grouping rows by `scenario_id` alone when
   `scenario_id` is unique only *within* a category. It merged all seven categories'
   templates, reporting 12 templates of ~430 rows where there are 66 of ~62, and
   40/4000 surviving rows where there are 210. It ran clean, the number was plausible,
   and it **reached published results** (README Finding 2) — the second instance to do
   so. Finding 2's conclusion survived, and the coverage numbers were provably
   untouched because that measurement excludes by row **id**, not by template. Caught
   by auditing the docs against the generator's own comment, not by any test; now
   pinned by `tests/test_contamination_structure.py`. Corrected in Phase 5A.
7. **Phase 8B's first container image**, whose `.dockerignore` said `*.npy` and
   therefore excluded nothing: those patterns are matched with Go's
   `filepath.Match` against the whole relative path, and `*` does not cross `/`,
   so `data/ticket_embeddings_bge-base-en-v1-5.npy` was copied straight in. The
   build ran clean, `train_embeddings.py` printed `[cache HIT]`, and the image's
   Tier-2 classifier and FAISS index were derived from a 12 MB cache encoded on
   the developer's machine months earlier rather than from the image's own
   encode — the same silent stale embedding cache as occurrence #1, arriving by
   a new route. **Caught before any result, by reading the build log**, which is
   not a control. It is now a control: a `RUN` guard after `COPY` fails the build
   if any pre-built artifact reaches the context. Note that this one was not a
   model swap, a routing test or a grouping key — it was a *packaging* rule, and
   nothing on the earlier list would have predicted it.

8. **`paper/PROVENANCE.json` hashing raw working-tree bytes.** Working-tree
   bytes are a property of the *machine*, not of the result: with
   `core.autocrlf=true` — the default on Windows and what this project is
   developed under — git stores LF and checks out CRLF, so **50 of the 51**
   paper sources are CRLF here and LF on a Linux runner. Every recorded hash
   was a Windows hash, and `tests/test_paper_artifacts.py` reported all 50 as
   "these RESULT FILES changed" on the first CI run **although nothing had
   changed** — including `src/agent/config.py`, which no one had touched. The
   hashes were internally consistent on the machine that wrote them and wrong
   on every other machine. Verified before it was touched: 51/51 hashes matched
   the Windows working tree, 50 differed from the git blob, and **50 of 50 were
   explained purely by CRLF→LF**, with nothing left over — so no file was
   genuinely stale. Fixed by hashing content with CRLF normalised to LF, in
   **both** the builder and the test helper, and pinned by three new tests: a
   CRLF and an LF copy hash identically, a one-character content change is
   still caught in both conventions, and the two implementations agree.
   The same shape lives in the gate CSVs: Python's `csv` module writes
   `lineterminator='\r\n'` on **every** platform, so a plain `git diff
   --exit-code` on Linux flags an identical file; `gates.yml` uses
   `--ignore-cr-at-eol`, which ignores a CR at end of line and nothing else.
   **Never hash working-tree bytes to decide whether a result changed.**

When touching anything model-related — or any routing/eligibility test, any grouping
key, any rule about which files reach a build, or **any hash of bytes that a
checkout can rewrite** — assume a **ninth** is waiting. Run `pytest` and the adversarial gate before
believing a green result, and check any count you rely on against a second, independent
derivation of it.

