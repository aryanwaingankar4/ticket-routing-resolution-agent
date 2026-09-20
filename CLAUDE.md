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
W=100–200, and nothing was promoted into config) and Phase 5A (the conformal
template-grouping correction). The work now follows the agreed **Phase 5–9
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

### Tests

```powershell
pytest                     # full suite, offline, ~50s
pytest -m "not slow"       # fast subset, no model loading
pytest tests/test_pipeline_parity.py -v
```

`pytest.ini` deselects `-m gemini` by default so the suite never spends API quota.
Markers: `slow` (loads BGE + fits Tier-1), `gemini` (live API call).

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
- **Conformal coverage transfers for Tier-2 but not Tier-1.** Calibrated on the
  same 175 tickets at the same alpha, TF-IDF loses 23.3 coverage points on the
  45-ticket benchmark while BGE loses 1.1 (noise band 4.5). Do not quote a
  conformal guarantee for a lexical model on out-of-template text.
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

## The recurring bug class

Six occurrences so far, all the same shape: a value or artifact that is wrong for its
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

When touching anything model-related — or any routing/eligibility test, or any grouping
key — assume a **seventh** is waiting. Run `pytest` and the adversarial gate before
believing a green result, and check any count you rely on against a second, independent
derivation of it.

