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
finish a phase, report, and stop rather than rolling into the next one. Phases 0
(pipeline consolidation) and 1 (conformal prediction) are complete and pushed.

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
```

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
# Ablation — quantifies what each safety net is actually worth
python src/experiments/run_ablation_study.py --mode baseline
python src/experiments/run_ablation_study.py --mode no-cascade
python src/experiments/run_ablation_study.py --mode no-rag

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
an orchestrator, but they still run **in one process, sequentially**. Do not describe
this as a distributed multi-agent system: no agent runs independently of the others
yet, and nothing is served over a network. That is Phase 3C.

**All inference lives in `src/agent/`** — one implementation, consolidated from what
were four independent copies:

```
src/agent/
├── config.py        frozen typed config + CALIBRATION_PROVENANCE + config_fingerprint()
├── schemas.py       pydantic models for every stage; Tier/EscalationReason enums
├── errors.py        typed exceptions; ONE Gemini error ladder (was three)
├── logging_setup.py structured JSON decision logs
├── artifacts.py     THE loader, with all three hard guards
├── classifier.py    cascade Tier-1 -> Tier-2          (stage implementation)
├── retriever.py     FAISS retrieval                    (stage implementation)
├── resolver.py      prompt + Gemini call with retry    (stage implementation)
├── agents.py        the three agents: name + requires + run()
├── orchestrator.py  the sequence and BOTH gates; run(TicketIn) -> PipelineResult
└── pipeline.py      the public façade over orchestrator.run()
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
  paraphrases of training rows, but memorisation is template-level: 12
  templates, ~430 rows each, and the set touches 11 of them. Removing source
  rows changes nothing; removing whole templates would leave 40/4000 rows. Do
  not "fix" this by filtering — it needs deployment-distribution data.
- **Generated sets need a near-duplicate guard, not just a label check.** The
  first deployment calibration set had 21 near-duplicate pairs because
  tightening a scope anchor narrowed the scenario space. Duplicated
  calibration points skew the conformal quantile. Scope anchors buy label
  accuracy at the cost of diversity -- all three Gemini generators in this
  project use that pattern, so check diversity whenever you tighten one.
- **In-distribution accuracy is uninformative here.** Template-generated data makes
  every model score ~100% in-distribution. Only the 14- and 45-ticket benchmarks measure
  anything real. Treat a new 100% in-distribution number as a red flag, not a success.
- **Artifacts are model-aware and filename-suffixed**
  (`ticket_index_bge-base-en-v1-5.faiss`, `ticket_classifier_bge-base-en-v1-5.joblib`).
  `artifacts.load_artifacts()` enforces three guards: `index.ntotal == len(metadata)`,
  `encoder dim == index.d == settings.models.embedding_dim`, and Tier-1's manifest
  still matching the dataset it was fitted on. Preserve all three —
  silent stale-artifact mismatch has bitten this project **four** times, most recently
  in `run_ablation_study.py`, where it reached published results.
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
- **Never overwrite a previous model's results file.** BGE re-runs write to
  `*_bge-base-en-v1-5.*` alongside the original MiniLM outputs, so the comparison stays
  auditable.
- **Path resolution** is always two directories up from the script's own location via
  `os.path.*`.
- **Scripts fail with clear actionable messages, not tracebacks** — the Streamlit demo
  may run live in front of an audience.
- **Gemini free tier is 15 req/min, 500/day.** Keep `GEMINI_CALL_DELAY_SEC` at 4s or
  above in any batch-calling script.
- **Before committing, check for stray embedding caches:**
  `git status | Select-String "\.npy"`

## Known inconsistencies

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

Five occurrences so far, all the same shape: a value or artifact that is wrong for its
context, stays internally consistent, and therefore produces wrong results with no
error. The first four were model swaps; the fifth shows the shape is not limited to
those.

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

When touching anything model-related — or any routing/eligibility test — assume a sixth
is waiting. Run `pytest` and the adversarial gate before believing a green result, and
check any count you rely on against a second, independent derivation of it.

