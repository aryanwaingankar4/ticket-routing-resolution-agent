# Project Overview — full briefing for a fresh session

> **This is a dated, derived snapshot, not a source of truth.** It was written
> on 2026-09-20 and is not maintained per session. `PROJECT_STATUS.md`,
> `CLAUDE.md` and `README.md` win on any conflict, and `git log` / `git status`
> win over all four. Use this file to get the whole project in mind quickly,
> then verify anything load-bearing against those.

**Purpose of this file.** A single, self-contained account of the whole project:
how the environment was built, everything that was built and measured, every
result and its caveat, the novelty claim, what is pending, and exactly where the
work stands. It is written so a brand-new session (or a new person) can read this
one file and hold the entire project in mind.

**It is a derived document, not a source of truth.** The three canonical docs
keep their roles: `CLAUDE.md` = how the project works (rarely changes),
`PROJECT_STATUS.md` = where it currently is (changes every session),
`README.md` = the lab notebook of experiments and methodology (1,984 lines, the
paper's raw material). Where this file and those disagree, they win.

- **Written:** 2026-09-20
- **Repo:** `aryanwaingankar4/ticket-routing-resolution-agent`, branch `main`
- **Last commit:** `b52d4bc` (Refresh PROJECT_STATUS.md with the Phase 4A commit
  SHA). `main` is fully pushed — `git log origin/main..main` is empty.
- **Verified today:** `pytest` → **150 passed, 0 failed** (142 committed + 8 new
  drift tests in the working tree), offline, exit 0.
- **Uncommitted work in the tree:** Phase 4B-1 (drift evaluation). See §12.

---

## 1. What this project is, in one paragraph

An IT support ticket triage agent. Given a free-text ticket it (a) classifies it
into one of 7 teams, (b) retrieves similar past tickets from a FAISS index and
asks Gemini to draft a resolution **grounded in those tickets**, (c) escalates to
a human at either of two independent confidence gates rather than guessing, and
(d) separately, offline, clusters *resolved* tickets by resolution similarity to
flag recurring issues worth automating. It is a B.Tech final-year AI/ML project
(from a nasscom hackathon use-case brief) being extended toward an IEEE-style
paper. **The experimental results are the deliverable, not the software.** The
throughline: *every escalation or automation decision is governed by a calibrated
confidence signal measured against real data, never assumed* — and the project
documents the calibration attempts that failed as carefully as the ones that
worked.

**Tech stack:** Python 3.14, scikit-learn, sentence-transformers, FAISS, Gemini
(`gemini-flash-lite-latest` via the unified `google-genai` SDK), FastAPI +
uvicorn, Streamlit, pydantic, pandas/numpy/scipy, pytest. Windows 11 +
PowerShell throughout.

**Scale:** ~29,000 lines of Python across `src/`, `tests/` and the dataset
generator; 4,000-row synthetic dataset; 150-test offline suite; 60 commits from
2026-07-28 to 2026-09-17.

---

## 2. Environment, from nothing to a running system

```powershell
# 1. Virtualenv (lives at venv/, gitignored)
python -m venv venv
.\venv\Scripts\Activate.ps1
pip install -r requirements.txt      # fully pinned

# 2. Secret: Gemini key in .env at the project root
#    GEMINI_API_KEY=...   (free key from https://aistudio.google.com/apikey)
#    Classification, retrieval and BOTH escalation gates work without it.
#    Only resolution drafting needs it.

# 3. Build every artifact, in this order, from a clean clone
python data/generate_dataset.py                 # 4,000 tickets, seed 42
python src/classification/train_tier1.py        # persisted Tier-1 (TF-IDF)
python src/classification/train_embeddings.py   # production Tier-2 (BGE)
python src/rag/build_vector_index.py            # FAISS index + aligned metadata

# 4. Run it
streamlit run src/app/streamlit_app.py          # live demo
uvicorn src.service.api:app --port 8000         # HTTP service (from the ROOT)
```

The service must start **from the project root**: `src/service/` has no
`__init__.py`, so `src.service.api` resolves as an implicit namespace package,
the same way `src.experiments.*` and `src.classification.*` do. (`src/agent/`
*does* have one and is a regular package.)

`requirements.txt` is pinned to the versions the published results were produced
with: `sentence-transformers==5.6.1`, `transformers==5.14.1`, `torch==2.13.0`,
`scikit-learn==1.9.0`, `faiss-cpu==1.14.3`, `scipy==1.18.0`, `pandas==3.0.5`,
`numpy==2.5.1`, `google-genai==2.14.0`, `pydantic==2.13.4`, `streamlit==1.60.0`,
`matplotlib==3.11.1`, `pytest==9.1.1`, `httpx==0.28.1`, `fastapi==0.140.0`,
`uvicorn==0.51.0`. Two pins exist for a specific reason: `httpx` and `scipy` were
previously only transitive, so a clean clone could not run the test suite / could
silently use an untested scipy.

Gitignored and must be regenerated: `venv/`, `models/*.joblib`,
`models/distilbert_ticket_classifier/`, `*.npy` embedding caches, `.env`.

**The Gemini budget is a real constraint on this project:** free tier is 15
requests/minute and 500/day. `gemini.call_delay_sec` stays at 4.5s in every batch
script, `POST /triage` defaults to `generate_resolution=false`, `pytest.ini`
deselects the `gemini` marker by default, and every quota-spending script is
dry-run first.

---

## 3. The dataset

`data/generate_dataset.py` (807 lines) produces `data/synthetic_tickets.csv`:

- **4,000 rows**, columns `id, title, description, category, resolution,
  priority, scenario_id`
- 7 categories, near-balanced: Infrastructure 572, Application 572, Security 572,
  Access Management 571, Network 571, Database 571, Storage 571
- Priorities: High 1,684 / Medium 1,473 / Low 843
- A "scenario" is a linked `(title_template, symptom_phrase, resolution_text)`
  tuple so one ticket's fields stay semantically consistent, and one shared
  per-ticket context dict resolves every placeholder (`{app}`, `{srv}`, `{db}`…)
  exactly once so entities never disagree across fields
- 9 scenarios per category (Database has 12) → **66 distinct
  `(category, scenario_id)` templates**, median 62 rows each. `scenario_id` is an
  index *within* a category, not a global id (see the ⚠ note in §7.6)
- Seed 42, everywhere, always. Scaled 1,000 → 4,000 rows on 2026-07-29

`scenario_id` doubles as clustering ground truth: `join_scenario_ground_truth.py`
recovers it for processed tickets by replaying the generator's own
`random.choice()` sequence — reproducible only because the seed is fixed.

**The most important fact about this dataset:** because it is template-generated,
every model scores ~100% in-distribution. *In-distribution accuracy is
uninformative here, and a new 100% is a red flag, not a success.* Everything real
is measured on three hand-built, read-only reference sets:

- **14-ticket benchmark** — `NOVEL_TICKETS` in
  `src/classification/generalization_test.py`; 2 per category in plain
  non-technical English that appears in no template.
- **45-ticket benchmark** — `data/novel_tickets_expanded.json`; the primary
  accuracy metric.
- **9-ticket adversarial escalation set** —
  `data/adversarial_escalation_tickets.json`; must-escalate / must-proceed cases
  including an off-topic weather question and a pizza request.

All three are **never edited or regenerated.**

---

## 4. Chronological history

### Week 1 (2026-07-28 → 07-31): the core pipeline, "Days 1–6"

| Commit | What landed |
|---|---|
| `4ad68c8` Day 1 | Synthetic dataset generator, validated duplicate-free, 1,000 tickets |
| `b6ec6bf` Day 2 | TF-IDF + LogReg baseline. **100% in-distribution, 7/14 (50%) on novel phrasing** — the red flag that set the project's entire methodology |
| `42fbd94` Day 3 | Frozen MiniLM embeddings + LogReg: **10/14 (71.4%)** |
| `638dd79` Day 4 | 3-way comparison adding fine-tuned DistilBERT: 7/14, classic overfit signature (100% in-dist within 1–2 epochs, generalization flat/declining) |
| `95d42b9` Day 5 | RAG layer: FAISS `IndexFlatIP` over L2-normalised embeddings + Gemini-grounded resolutions + the low-similarity escalation guard |
| `ea6ddda` | Streamlit demo end-to-end |
| `7c5416d` Day 6 | Dataset scaled 1,000 → 4,000. MiniLM re-scored **exactly 10/14, same 4 tickets wrong** → the ceiling is representation-limited, not data-limited |
| `f6bcb22`, `9f90fd0` | Cascade results documented; README becomes the lab notebook |

### August: the calibration wave

- `b105495` (08-05) — **Cascade classifier** (Tier-1 TF-IDF → Tier-2 embeddings)
  with its 3-attempt calibration methodology; **batch-intake simulation** (Zipf
  traffic skew, 500 tickets); **class-imbalance experiment** (training-data skew).
- `6c3935d` (08-07) — Resolution-clustering calibration + automation flagging;
  expanded benchmark (46 tickets at this point); embedding-model comparison.
- `b191af3` (08-14) — **The Infrastructure benchmark bug.** Every model got the
  same 5 "Infrastructure" tickets wrong. Root cause was benchmark *generation*,
  not the dataset: those 5 were physical-facilities issues (flickering lights,
  burst pipe, broken elevator) while this project's `Infrastructure` category is
  scoped to compute/servers, and the generation prompt never anchored what the
  label means *here*. Fixed by stripping 5, regenerating with an explicit
  positive/negative scope anchor, and manual review → the final **45-ticket
  benchmark**. Every model improved (MiniLM 29/46 → 32/45, BGE 31/46 → 33/45,
  E5 27/46 → 27/45), which is what validated the fix. A round number (46/49) was
  deliberately not chased.
- `9099d6d` … `b806591` (08-15) — Automation-flagging production run; reliability
  diagrams; the permanent **9-ticket adversarial set** + regression script; the
  **ablation study**; per-category clustering calibration.
- `0abe785`, `b7ff033`, `68ab78a` (08-26 → 08-31) — **The BGE swap.**
  `all-MiniLM-L6-v2` (384-dim) → `BAAI/bge-base-en-v1.5` (768-dim) for
  classification, RAG retrieval and cascade Tier-2, on the strength of 33/45 vs
  32/45; verified by two independently-trained BGE classifiers matching exactly.
  Introduced **model-aware filename suffixes** for every artifact, caught two
  real bugs, and forced the full **RAG threshold recalibration to 0.67**.

### September: the phased research programme

Phases are numbered; each opens with a reviewed plan and ends at an explicit
review gate.

| Phase | Commits | Outcome |
|---|---|---|
| **0** pipeline consolidation | `a3f0b23` (09-13) | 4 duplicate pipelines → `src/agent/`; corrected a published result |
| **1** conformal prediction | `148ad8b`, `bcb21c8` (09-14) | 4 findings; measurement only |
| BGE clustering re-run | `c9d956d` (09-14) | pooled cliff 0.90; measurement only |
| **2A** automation-flag validation | `d6a5434` → `d380710` (09-14) | negative result: indistinguishable on precision |
| **2B** resolution groundedness | `639ab6a` → `b6d0d5a` (09-14) | 93.9% grounded; LLM judge κ = −0.042 |
| **3A/3B/3C/3D** agent architecture | `1c753ae`, `156f07b`, `400eaff`, `9f893c5` (09-14 → 09-15) | Tier-1 persistence, orchestrator, HTTP service, docs; n8n dropped |
| **4** drift detection | plan `4086c8a` (09-15), **4A** `5c077dd` (09-17) | history sink + detector library + 4 findings |

---

## 5. Architecture

```
ticket text
    |
    v
[Tier 1: TF-IDF + LogReg] --confidence >= 0.50--> category
    |  below threshold
    v
[Tier 2: BGE + LogReg] -------------------------> category
    |
    v
[FAISS top-5 retrieval] --top1 sim >= 0.67--> [Gemini drafts grounded resolution]
    |  below threshold
    v
ESCALATE TO HUMAN   (Gemini is never called)

offline path: resolved tickets -> embed resolution text (MiniLM)
              -> union-find clustering at 0.80 -> automation candidates
```

Facts that are load-bearing and not obvious from any single file:

- **Gemini never decides the category.** Classification is entirely the trained
  models'. Gemini only drafts resolutions, grounded in retrieved tickets.
- **The two gates are independent and calibrated separately.** The cascade gate
  reads only Tier-1/TF-IDF confidence, so swapping the Tier-2 embedding model
  does not require recalibrating it (verified by code review during the BGE
  swap). The RAG gate *is* embedding-model-specific and must be recalibrated on
  any model swap.
- **All inference lives in `src/agent/`** — one implementation, consolidated in
  Phase 0 from *four* independent copies.
- **No agent decides anything.** Agents declare the `Artifacts` fields they
  require (validated at construction, so a missing *or misspelled* dependency
  fails before any routing happens) and do one thing. Every gate lives in
  `orchestrator.py` as a pure function testable without loading a model — an
  escalation policy that needs BGE and FAISS to test is one nobody tests, and
  this project's whole claim rests on that policy.
- **Per-agent traces.** `PipelineResult.steps` records every agent including
  skipped ones — a `skipped` resolution step is *positive* evidence that the RAG
  gate held and no LLM call was made.
- **The failure boundary is the service's, not the library's.** `/triage` maps a
  known `AgentError` to HTTP 200 + an escalation envelope (the ticket needs a
  human; a 5xx would wrongly invite a retry) and anything else to 500. The
  library still raises, deliberately: in a calibration run a `RetrievalError`
  silently becoming an escalation row would corrupt a result quietly.
  `PipelineResult.classification` is required, so an escalation envelope that
  needs no classification is the only shape that avoids inventing a category.
- **Agents are independently *addressable*, not independently *running*.**
  `/triage` orchestrates in-process via `pipeline.run()` because golden parity
  must not depend on a running server. **Never call this a distributed
  multi-agent system** — the honest claim is "a sequential pipeline with real
  agent boundaries, an orchestrator, and an HTTP surface."

### The library, file by file

| File | Lines | Role |
|---|---:|---|
| `src/agent/config.py` | 334 | Frozen typed config; every calibrated constant + `CALIBRATION_PROVENANCE` + `config_fingerprint()` |
| `src/agent/schemas.py` | 228 | Pydantic models for every stage; `Tier` / `EscalationReason` enums |
| `src/agent/errors.py` | 100 | Typed exceptions; ONE Gemini error ladder (was three) |
| `src/agent/artifacts.py` | 418 | THE loader + three hard guards; `load_tier1()`, `load_drift_reference()` |
| `src/agent/logging_setup.py` | 228 | Structured JSON decision logs; opt-in JSONL sink (off by default) |
| `src/agent/classifier.py` | 80 | Cascade Tier-1 → Tier-2 (stage implementation) |
| `src/agent/retriever.py` | 69 | FAISS retrieval (stage implementation) |
| `src/agent/resolver.py` | 130 | Prompt + Gemini call with retry ladder (stage implementation) |
| `src/agent/agents.py` | 170 | The three agents: `name` + `requires` + `run()` |
| `src/agent/orchestrator.py` | 269 | The sequence and BOTH gates; `run(TicketIn) -> PipelineResult` |
| `src/agent/pipeline.py` | 68 | Public façade over `orchestrator.run()` |
| `src/agent/conformal.py` | 373 | Split conformal + conformal p-values (measurement only) |
| `src/agent/drift.py` | 532 | Drift detector: pure functions over records (measurement only) |
| `src/service/api.py` | 404 | FastAPI: per-agent endpoints, `/policy/rag-gate`, `/triage`, `/health` |

`artifacts.load_artifacts()` enforces three guards, each because silent
stale-artifact mismatch has bitten this project:

1. `index.ntotal == len(metadata)`
2. `encoder dim == index.d == settings.models.embedding_dim`
3. Tier-1's manifest still matches the dataset it was fitted on (dataset sha256,
   rows fitted, sklearn version, vectorizer config) — with **deliberately no
   refit-on-miss fallback**

Consumers (`streamlit_app.run_pipeline()`,
`test_adversarial_escalation.run_ticket_through_pipeline()`,
`process_ticket_batch`) are thin adapters over `pipeline.run()`. Add new
consumers the same way; never re-implement the orchestration, and never
construct a `SentenceTransformer` / read the index / refit Tier-1 locally.

### The HTTP surface

| Endpoint | Purpose |
|---|---|
| `GET /health` | `config_fingerprint()` over the wire + index size + Tier-1 manifest — a deployment's stale-artifact detector |
| `POST /agents/classify` | The classification agent alone |
| `POST /agents/retrieve` | The retrieval agent alone |
| `POST /agents/resolve` | The resolution agent alone — the only endpoint that spends Gemini quota |
| `POST /policy/rag-gate` | The escalation decision for a retrieval result, so a workflow tool can branch on a boolean it did not compute |
| `POST /triage` | The whole pipeline, in-process, `generate_resolution=false` by default |

Endpoints are sync and serialised by one lock: one ticket at a time, on purpose —
a research demo, not a throughput exercise. The service loads artifacts one way
only (`require_gemini=False`) and attaches a Gemini client via
`dataclasses.replace()`, because `load_artifacts` is `lru_cache(maxsize=2)` keyed
on that flag and calling it both ways in one process would load **BGE twice**.

---

## 6. The calibrated constants

These are the point of the project: measured, evidence-backed values, each with
its evidence in `CALIBRATION_PROVENANCE`, all in frozen `src/agent/config.py`
(assigning to one raises `ValidationError`). `tests/test_config.py` fails the
build if a value drifts or loses its provenance entry.

| Constant | Value | Gates production? | Config path |
|---|---|---|---|
| RAG similarity threshold | **0.67** | yes | `settings.rag.similarity_threshold` |
| Cascade confidence threshold | **0.50** | yes | `settings.cascade.confidence_threshold` |
| Resolution-clustering threshold | **0.80** (MiniLM) | yes | `settings.clustering.resolution_similarity_threshold` |
| Conformal target error rate | 0.10 | **no — measurement only** | `settings.conformal.alpha` |
| Drift α / conditional δ | 0.10 / 0.10 | **no — measurement only** | `settings.drift.*` |

Other live settings: `rag.top_k = 5`; `cascade.filing_gate_enabled = False` (only
`process_ticket_batch.py` applies a second gate on final confidence);
`gemini.call_delay_sec = 4.5`, `max_retries = 3`;
`clustering.clustering_embedding_model = "all-MiniLM-L6-v2"` (deliberate — §7.8);
production embedding model `BAAI/bge-base-en-v1.5` (768-dim); Gemini model
`gemini-flash-lite-latest`. `suggest_resolution.SIMILARITY_THRESHOLD` still
exists but re-exports `settings.rag.similarity_threshold` so older experiment
scripts keep running. **Never redeclare a threshold anywhere else.**

**Never write `getattr(obj, "SOME_THRESHOLD", <number>)`.** That exact pattern
silently reverted the human-escalation gate to a dead MiniLM value;
`tests/test_no_silent_fallback.py` fails the build if it reappears.
`settings.drift` deliberately contains **no window size and no alarm threshold**,
and a test pins their absence — naming either before the null false-alarm rate is
measured would be "a hand-tuned threshold wearing a lab coat."

---

## 7. Every result, with its caveat

### 7.1 Classification

| Method | In-distribution | 14-ticket | 45-ticket |
|---|---|---|---|
| TF-IDF + LogReg | 100.0% | 7/14 (50.0%) | — |
| Frozen MiniLM + LogReg | 100.0% | 10/14 (71.4%) | 32/45 (71.1%) |
| **Frozen BGE + LogReg** | — | — | **33/45 (73.3%) — production** |
| Frozen E5 + LogReg | — | — | 27/45 (60.0%) |
| Fine-tuned DistilBERT (best epoch) | 100.0% | 7/14 (50.0%) | — |
| Cascade (TF-IDF → embeddings) | — | 10/14, ~21% resolved by the cheap tier | — |

One genuinely hard case survives everywhere: a "status page shows green but the
service is actually down" ticket that all three embedding models get wrong — a
realistic monitoring ambiguity, not a labelling error.

### 7.2 Cascade calibration — three attempts, two rejected

1. **In-distribution held-out split — rejected.** 100% accuracy in every
   confidence bucket; produced a threshold that looked trustworthy and missed a
   confidently-wrong real prediction entirely.
2. **35 hand-written calibration tickets — rejected.** Too sparse (34/35 in one
   bucket); the threshold was noise.
3. **175 Gemini-paraphrased calibration tickets (25/category) — adopted.** Dense
   enough to expose Tier-1's real overconfidence. 21/175 flagged by the
   self-consistency label check and kept, because ground truth was preserved.
   A threshold-derivation bug was fixed here too: the scan broke on the first
   small/noisy bucket instead of continuing to better ones.

| Target accuracy | Threshold | Novel Tier-1 resolution rate | Novel accuracy |
|---:|---:|---:|---:|
| 90% | 1.00 | 0.0% | 71.4% |
| 80% | **0.50** | 21.4% | 71.4% |
| 70% | **0.50** | 21.4% | 71.4% |

At a strict 90% bar the cascade collapses to pure Tier-2; at a 70–80% bar a real
threshold emerges that routes ~21% of real-world tickets through the cheap tier
at **zero accuracy cost**. The contribution is the validated methodology, not an
accuracy jump.

**Reliability diagrams** (both tiers, against the real 500-ticket batch): Tier-1
ECE **0.1122**, Tier-2 ECE **0.0992**; both tiers top out around 0.92/0.94
confidence — the models are **underconfident**, which is the safer failure mode
for an escalation-gated system.

### 7.3 RAG similarity threshold — the recalibration story (a citable finding)

After the BGE swap the threshold was provisionally 0.65, then recalibrated:

- **Attempt 1 — reuse the 175-ticket in-domain set: failed, informatively.**
  Every threshold from 0.35 to 0.65 gave identical results (all 175 "proceed"),
  because the set contains no out-of-domain examples. Its mechanical cliff-edge
  at T = 0.85 was a low-N artifact (only 2/175 proceed there, so precision 1.0 is
  trivial) and was **not** adopted. **Finding: an in-domain-only calibration set
  can produce an unrepresentative cliff-edge when the gate's actual job is
  detecting out-of-domain input.**
- **Attempt 2 — build a real 45-ticket OOD set** (15 hand-written seeds × 3
  Gemini paraphrases, across weather/errands, food, non-IT departmental
  questions, and vague "it's broken" messages — deliberately broader than the 9
  adversarial tickets). **Second finding: adding OOD data did not fix the
  artifact.** The combined cliff still landed at 0.85, because OOD top-1
  similarities span only 0.5051–0.7249 and contribute *zero* information at
  0.80+. A negative class only helps a cliff-edge search where its own
  distribution overlaps the swept range.
- **Two hardening checks.** Measured **5.7% (10/175)** in-domain self-retrieval
  contamination (a calibration ticket retrieving its own source row), so the
  in-domain precision curve is inflated and was deliberately not relied on; plus
  an exact-source-match diagnostic reported beside the category-agreement proxy
  without silently replacing it (switching the primary metric is a methodology
  decision, not something a script does quietly).
- **Final derivation — intersect two independent constraints.** (1) The
  adversarial set fixes a hard safe range: must-escalate tickets top out at
  **0.6196**, must-proceed bottom out at **0.6704**, so any threshold in
  (0.6196, 0.6704] passes 9/9 by construction. (2) Inside that range OOD leakage
  falls monotonically as the threshold rises while in-domain recall stays flat at
  100% — no trade-off, only a minimum: **T = 0.67, at 13.3% OOD leakage vs 37.8%
  at the provisional 0.65.** Confirmed **9/9** on a live re-run before adoption.

### 7.4 Ablation — what the safety nets are actually worth

| Mode | 45-ticket accuracy | Adversarial escalation |
|---|---:|---:|
| **Baseline** (both gates active) | **71.11% (32/45)** | 9/9 correctly escalated |
| No cascade (Tier-1 only, threshold 0) | 35.56% (16/45) | — |
| No RAG gate (threshold 0) | — | 9/9 would attempt a resolution; 6/9 should have escalated |

The cascade gate is worth **+35.6 accuracy points** — it roughly doubles
real-world accuracy over Tier-1 alone. The RAG gate prevents **6 concrete
instances** of confidently fabricated output on this set alone.

> **Correction carried in the README.** These were originally 68.89% and a
> 33.3-point gain — measured under MiniLM and never re-run after the BGE swap,
> because `run_ablation_study.py` kept its own hardcoded MiniLM index, metadata,
> classifier and model name. Four mutually-consistent stale values meant it ran
> clean and silently measured the entire pre-BGE pipeline. The CSV was written
> 2026-08-15; BGE landed 2026-08-26. **This is the only instance of the recurring
> bug class that reached published results, and it stood for eleven days.**
> Caught by the golden-parity discipline introduced in Phase 0. `no-cascade`
> correctly did *not* move (16/45), which is the internal check: Tier-1 is
> TF-IDF, so a Tier-2 swap cannot affect it.

### 7.5 Traffic skew and training-data skew

- **Batch intake (traffic skew).** A 500-ticket Zipf-skewed batch: 100% accuracy,
  zero escalations — expected, since it is sampled in-distribution. It validated
  the *plumbing* (routing, per-category stores, rate-limit handling), not
  robustness. Operationally it hit the Gemini rate limit (15 of 500 calls failed
  after retries), which is why the per-call delay went 1.5s → 4.5s; the 15 were
  retried successfully after the daily cap reset.
- **Class imbalance (training-data skew).** Access Management sampled down
  571 → 500 → 200 → 100 → 50 with the other six categories held fixed. AM was
  chosen because it scored perfectly at full data (so any degradation is
  attributable to the skew) and is semantically adjacent to Security (a
  plausible, checkable failure mode):

| AM training size | 14-ticket score | AM tickets specifically |
|---:|---:|---|
| 571 | 11/14 | both correct |
| 500 | 11/14 | both correct |
| 200 | 10/14 | one → Storage |
| 100 | 9/14 | both wrong → Security, → Storage |
| 50 | 9/14 | both still wrong |

  In-distribution accuracy stayed ~99.6–100% at every level — the benchmark is
  the only thing that saw the damage. Every wrong AM prediction fell below the
  0.50 gate and was correctly escalated, which is promising but **directional
  only** (1–3 wrong predictions per level). Methodology note: this sweep trains
  on an 80% split, so its 571 baseline is not directly comparable to the
  full-dataset 10/14; the trend is the result, not the absolute.

### 7.6 Conformal prediction (Phase 1) — measurement only

Hand-rolled split conformal in `src/agent/conformal.py`, numpy only (~150 lines
of the 373). The quantile is the exact order statistic ⌈(n+1)(1−α)⌉ rather than
`np.quantile`, whose interpolation silently voids the finite-sample guarantee.
`settings.conformal.enabled = False`, so production still gates on 0.50/0.67 and
golden parity holds. Conformal upgrades the claim from *calibrated* to *provably
risk-controlled* — but only under **exchangeability** between calibration and
deployment data, and this project is an unusually clean setting in which to
measure what happens when that fails.

**Finding 1 — coverage transfer is a property of the *representation*.** Same
175-ticket calibration set, same α, same 45-ticket benchmark:

| Tier | α | Coverage on 175 | Coverage on 45 | Gap | ±2 s.d. | Mean set size | Singletons |
|---|---:|---:|---:|---:|---:|---:|---:|
| Tier-1 (TF-IDF) | 0.20 | 0.806 | 0.578 | **−0.222** | 0.060 | 2.67 | 8.9% |
| Tier-1 (TF-IDF) | 0.10 | 0.909 | 0.667 | **−0.233** | 0.045 | 3.44 | 8.9% |
| Tier-1 (TF-IDF) | 0.05 | 0.960 | 0.844 | −0.106 | 0.033 | 5.00 | 6.7% |
| Tier-2 (BGE) | 0.20 | 0.806 | 0.756 | −0.044 | 0.060 | 1.11 | 84.4% |
| Tier-2 (BGE) | 0.10 | 0.909 | **0.889** | −0.011 | 0.045 | 1.84 | 35.6% |
| Tier-2 (BGE) | 0.05 | 0.960 | 0.978 | +0.028 | 0.033 | 2.56 | 17.8% |

TF-IDF loses 23.3 coverage points at α = 0.10 — five times its noise band — where
BGE loses 1.1, comfortably inside it. Mechanism: TF-IDF's scores are built from
surface vocabulary and a paraphrase changes the vocabulary entirely, so the
calibration score distribution does not transfer; BGE encodes meaning, which
survives. **Exchangeability is not only a property of how data was sampled — it
is a property of the representation the scores are computed in.** This is the
sharpest result in the project: it converts "BGE generalizes better" into a
statement about whether a statistical guarantee survives deployment at all. It
also reframes the cascade — Tier-1 at α=0.10 emits a singleton for only 8.9% of
benchmark tickets (by its own admission unable to commit), while Tier-2 at α=0.20
emits singletons for 84.4%. The accuracy/autonomy trade-off that the hand-tuned
0.50 encodes falls out of conformal as a consequence of the chosen risk level.

**Finding 2 — the in-domain calibration set cannot be de-contaminated.** The 175
tickets are Gemini paraphrases of rows the models trained on. Predicted
consequence: memorised sources depress nonconformity scores, shrink the quantile
and push coverage below nominal. **The prediction was wrong in practice, and the
reason is more interesting:** refitting both tiers with all 175 source rows
removed moves benchmark coverage by a mean of 0.005 across all 64 paired
configurations (max 0.067, in a low-sample Mondrian cell), and by nothing at
three decimals in most. Memorisation is at **template** level, not row level:
deleting one row removes a negligible slice of its template's evidence, while
excluding every row sharing a template would destroy the training set. So **an
in-domain calibration set built by paraphrasing training rows cannot be made
exchangeable at any amount of cleaning when the training data is
template-redundant.** The remedy is not filtering; it is deployment-distribution
data — which motivated Finding 4.

> ⚠ **The published diagnostic numbers behind Finding 2 are wrong (found
> 2026-09-20, not yet fixed).** `calibrate_conformal.py` (Step 1b, lines ~202–205)
> groups templates by `scenario_id` alone, but `scenario_id` is unique only
> *within* a category — the generator says so at `data/generate_dataset.py:634`.
> Published: 12 templates, 11 touched (91.7%), median 430 rows/template, 40/4000
> rows surviving template-level exclusion. Correct grouping by
> `(category, scenario_id)`: **66 templates, 62 touched (93.9%), median 62
> rows/template, 210/4000 surviving.** The *conclusion* stands — 210/4000 is
> still destruction rather than de-contamination, and removing one row still
> leaves ~61 siblings (~1.6% of a template's evidence, vs the published ~0.2%) —
> and the coverage-movement measurement is unaffected because it excludes by row
> id. Only the diagnostic table is wrong. It is the recurring bug class again: a
> grouping key that is internally consistent but wrong for its context. Fixing it
> means the script, the README Finding 2 table, the conformal artifact JSON, and
> a sixth entry in CLAUDE.md's bug-class list.

**Finding 3 — the RAG gate reframed as conformal novelty detection.** Calibrates
on inliers only by construction, producing a p-value for "is this ticket
exchangeable with the indexed corpus?" and escalating when p ≤ α:

| α | In-domain false-escalation rate | OOD detection (15 seeds) | OOD (45 variants) | Adversarial |
|---:|---:|---:|---:|---:|
| 0.20 | 0.194 | 100% | 100% | 9/9 |
| 0.10 | 0.091 | 100% | 97.8% | 9/9 |
| 0.05 | 0.040 | 100% | 95.6% | 9/9 |
| 0.01 | 0.000 | 93.3% | 84.4% | 6/9 |

The false-escalation rate tracks α almost exactly — choosing α *is* choosing how
often a legitimate ticket may be sent to a human unnecessarily, a quantity the
hand-tuned 0.67 never had. At any α ≥ 0.05 it matches production's 9/9. The
α = 0.01 degradation is a resolution limit, not a failure: with 175 calibration
points the smallest attainable p-value is 1/176 ≈ 0.0057. OOD detection is
headlined at **seed level (15)** because the 45 variants are 3 paraphrases per
seed and are not independent draws.

**Finding 4 — distribution matching is necessary but not sufficient.** A
175-ticket **deployment-distribution** calibration set (25/category — exactly the
in-domain set's size and balance, so any difference is attributable to
distribution alone; written in the benchmark's plain register; per-category scope
anchors; disjointness from the benchmark *enforced* by rejecting any candidate
above 0.90 cosine against any benchmark ticket, observed max 0.872;
self-consistency disagreements discarded rather than kept):

| Tier | α | Gap (in-domain) | Gap (deployment) | ±2 s.d. | Set size | Singletons |
|---|---:|---:|---:|---:|---:|---:|
| Tier-1 | 0.20 | −0.222 | −0.244 | 0.060 | 2.67 → 2.58 | 8.9% → 8.9% |
| Tier-1 | 0.10 | −0.233 | **−0.144** | 0.045 | 3.44 → 4.04 | 8.9% → 6.7% |
| Tier-1 | 0.05 | −0.106 | −0.083 | 0.033 | 5.00 → 5.13 | 6.7% → 6.7% |
| Tier-2 | 0.20 | −0.044 | **0.000** | 0.060 | 1.11 → 1.20 | 84.4% → 80.0% |
| Tier-2 | 0.10 | −0.011 | −0.011 | 0.045 | 1.84 → **1.67** | 35.6% → **46.7%** |
| Tier-2 | 0.05 | +0.028 | −0.039 | 0.033 | 2.56 → **1.89** | 17.8% → **33.3%** |
| Tier-2 | 0.01 | +0.010 | −0.012 | 0.015 | 4.76 → **2.47** | 2.2% → **20.0%** |

For Tier-1 about 38% of the shortfall closes and −0.144 is still six times the
noise band: **where the representation fails to transfer, no calibration-set
realism repairs the guarantee** — which strengthens Finding 1 rather than
competing with it (the representation is the binding constraint, the calibration
set the looser one). For Tier-2 coverage was already inside the band, so the
payoff is *set size*: at identical coverage, singletons on 46.7% of benchmark
tickets vs 35.6%, i.e. **~11 points more autonomy at the same risk**, since a
singleton is exactly the case the system can route without a human.

**The near-duplicate defect, and why it is recorded.** The first generated
deployment set contained **21 near-duplicate pairs** at ≥0.95 cosine, 3
byte-identical. Cause: tightening the Infrastructure scope anchor cut
self-consistency rejections from 36% to 3% *and* narrowed the scenario space so
the generator kept rewriting the same "ran out of memory" ticket. **Scope anchors
buy label accuracy at the cost of diversity, and all three Gemini generators in
this project use that prompt pattern** — check diversity whenever one is
tightened. Fixed at generation time (not cleaned up afterwards) with a
within-category 0.95 cap, chosen because measured within-category similarity runs
p50 0.729 / p95 0.892 / p99 0.949, so legitimate same-category tickets really do
reach 0.89. Dedup kept 159 of 175 and regenerated 16; final set has 0
near-duplicate pairs, max off-diagonal 0.9494. **It was not cosmetic:** on the
defective set Tier-2 appeared to *lose* coverage at α = 0.10 (−0.056) — a false
negative that would have been published. The pre-dedup set is committed as
`deployment_calibration_tickets.predup.json` so the difference is auditable.

Results live in `conformal_calibration_results.csv` (128 configurations: 2 tiers
× 2 contamination variants × 2 label filters × 2 score functions ×
marginal/Mondrian × 4 α) and `conformal_novelty_results.csv`.

### 7.7 Resolution clustering → automation flagging

Cluster *resolved* tickets by resolution-text similarity (connected components /
union-find over the cosine matrix), evaluated with pairwise precision/recall/F1
against `scenario_id` ground truth, swept 0.99 → 0.60.

- **MiniLM: a cliff-edge, not a slope.** Precision holds at 1.0000 down to
  **0.80**, then collapses at 0.75. 0.80 adopted — the last point with zero false
  "these tickets share a fix" claims — deliberately over the recall-better 0.75,
  because a false merge misleads operators while a missed opportunity merely
  preserves the status quo.
- **Per-category (MiniLM):** Infrastructure (n=115), Application (100), Security
  (61) and Access Management (48) cliff at 0.80, matching pooled; Storage (53)
  0.75, Network (74) 0.75, Database (49) 0.70. **Network is the notable one** —
  above the low-N bar of 60 yet still needing 0.05 looser, a credible divergence
  rather than noise. Two real bugs were fixed building this: an over-strict
  small-N bar (100 → 60) and a cliff-edge scan that broke on undefined (0/0)
  precision instead of skipping it (which had made 3 categories falsely report
  "no cliff-edge"). That corrected `find_cliff_edge()` is the version later
  reused for the RAG sweep.
- **BGE re-run (measurement only):** pooled cliff **0.90** (precision 1.0000 at
  0.90, breaking to 0.9814 at 0.85); per-category, six categories cliff at 0.85
  and Access Management at 0.90. **The citable comparison is the spread:**
  MiniLM's per-category cliffs ranged across a 0.10 band with 3/7 diverging from
  pooled; under BGE every category sits within a single 0.05 step. **BGE clusters
  more consistently across categories**, and needs a systematically higher
  absolute threshold — the same higher-similarity pattern the RAG recalibration
  found. All BGE outputs are written to `*_bge-base-en-v1-5.*` files so the
  MiniLM artifacts survive for comparison.

**The production feature** (`flag_automation_candidates.py`, fixed 0.80, no
ground-truth dependency at run time), over the full 500-ticket batch:

| Category | Tickets | Flagged clusters | Flagged tickets | Singletons |
|---|---:|---:|---:|---:|
| Infrastructure | 115 | 17 | 107 | 8 |
| Application | 100 | 17 | 84 | 16 |
| Security | 61 | 9 | 56 | 5 |
| Database | 49 | 7 | 24 | 25 |
| Storage | 53 | 10 | 45 | 8 |
| Network | 74 | 14 | 51 | 23 |
| Access Management | 48 | 9 | 29 | 19 |
| **Total** | **500** | **83** | **396** | **104** |

Citable pattern: **Infrastructure's fixes are highly standardised (93% of its
tickets land in a repeatable cluster) while Database's are comparatively bespoke
(49%)** — a genuine finding about which categories are naturally automatable, not
an artifact of the method. Flagging is explicitly a suggestion for human review,
never an autonomous action: the 1.0000 precision was measured against known
ground truth on the development set, and production tickets carry no such
guarantee at run time.

### 7.8 Phase 2A — automation-flag validation (a negative result, and the good kind)

The question: should production clustering swap from MiniLM@0.80 to BGE@0.90?
Cliff-edge math alone cannot answer it — the RAG threshold was only adopted
because a benchmark and an adversarial set could re-confirm the specific value.

**The structural diagnostic, run before spending any labelling budget.** Each
configuration compared at its *own* cliff-edge (comparing both at 0.80 would
handicap BGE at a threshold its own precision curve does not endorse): MiniLM
merges 1,167 pairs, BGE 1,382, both 1,069, MiniLM-only 98, BGE-only 313. Then the
decisive check against `scenario_id`: **not one of those merges — not one of the
1,382, nor any of the 411 disagreements — is a cross-template merge.** So the
entire measured difference between the two configurations is **recall**, not
precision.

**A pre-registered decision rule that had to be thrown out.** The original rule
(exact two-sided binomial / McNemar on which configuration wins more discordant
pairs) is statistically sound but wrong for *this* project: since the region is
313 BGE-only against 98 MiniLM-only, labelling everything `same_fix` hands BGE a
42–18 win. A dry run against synthetic labels confirmed it — the rule printed
**PROMOTE at p = 0.0027 on zero evidence about flag correctness.** It was
rewarding whichever model merges more, which inverts the cost asymmetry the
feature is built on. **A precision gate that promotes on recall is not a gate.**
Replaced *before any label was collected*: primary is now the false-merge rate on
each configuration's *extra* merges, and BGE is promoted only if it makes zero
observed false merges **and** MiniLM makes at least one; if neither does, the
verdict is **"no precision signal"**, not promotion. The binomial survives as an
explicitly demoted recall comparison.

**The pilot.** 12 pairs (6 per direction) from the most textually divergent end
of the disagreement region — ranked by `1 − token Jaccard`, because a genuinely
different fix would surface where wording diverges most. Selected pairs span
divergence 0.53–0.68 against a region median of ~0.42; the near-duplicate guard
rejected 7 candidates. Labelling was blind: `scenario_id` withheld, direction
hidden, entries shuffled. The scorer refuses to compute the head-to-head on a
pilot at all, because the deliberate 6/6 balance would bias a win-rate.

| Configuration | Extra merges judged | `same_fix` | False merges | Rate |
|---|---:|---:|---:|---:|
| MiniLM @ 0.80 | 6 | 6 | **0** | 0.0% |
| BGE @ 0.90 | 6 | 6 | **0** | 0.0% |

Closest call: **PL005**, two `fstab`-blocked boot failures where one ticket also
explicitly restarts/resumes the boot — labelled `same_fix` but flagged
low-confidence ("could be read as an implied trivial follow-on… OR as a
materially different remediation SOP"). PL009/PL010 (cause named on one side only,
word-for-word equivalent remediation) and PL011 (same branch twice) resolved
cleanly. Secondary `scenario_id` agreement was 12/12 — exactly the degeneracy the
diagnostic predicted, reported so it is visible rather than assumed.

**Zero observed is not zero.** Rule-of-three upper bounds: 50% per configuration
at 0/6, **25% pooled at 0/12**. This does not establish that no false-merge case
exists — only that none was found among the 12 most divergent pairs. The
conclusion is carried by the diagnostic underneath, not the pilot.

**Verdict: the two are indistinguishable on precision and differ only in recall,
so production stays on MiniLM@0.80.** Promotion would surface 1,382 pairs vs
1,167 — a product judgement about review-queue capacity, not a calibration
result, and it should be argued and recorded as such. The remaining 48 judgements
were deliberately not spent: the pilot existed to determine whether they would
measure anything, and they would not. **Do not reopen this as a calibration
question.** The limitation is the dataset, not the method: **template-generated
data cannot produce two tickets that look alike but need different fixes, because
the templates *are* the fix classes** — the same wall the conformal work hit from
an entirely different direction, which is itself evidence the limitation is real.

### 7.9 Phase 2B — resolution groundedness, and an LLM judge that failed

**Scope was set by a pre-flight diagnostic before any quota was spent**, because a
judge can only find an unsupported claim if the retrieved context permits one:

| Query set | All 5 retrievals → one fix | 2+ distinct fixes |
|---|---:|---:|
| In-distribution (n=200) | **85.0%** | 15.0% |
| Novel-45 | 28.9% | **71.1%** |
| Adversarial-9 | 11.1% | **88.9%** |

The in-distribution corpus is degenerate (2A's wall again); the benchmarks are
not, so 2B could proceed where 2A could not. Of 54 benchmark tickets, 21 escalate
at the RAG gate, leaving **33 — the entire eligible population, not a sample.**

- **Primary — groundedness: 31/33 = 93.9%** (Wilson 95% CI 80.4–98.3%). Zero
  `partially_grounded`; two `ungrounded`.
- Both failures share one shape: G021 (laptop disk full) and G024 (ransomware)
  retrieved irrelevant context, **correctly said so**, then prescribed a
  substantive fix from general knowledge anyway. The contrast is the finding:
  **10 drafts contain explicit mismatch language, 8 declined cleanly and are
  grounded, only these 2 acknowledged the mismatch and answered regardless.** The
  model can tell when retrieval is irrelevant; that recognition just does not
  reliably stop it answering. G024 is the cleanest demonstration that
  groundedness is not quality — "disconnect from the network and escalate to
  security" is excellent ransomware advice and entirely ungrounded.
- **Secondary — hedge appropriateness: 32/33 = 97.0%** (CI 84.7–99.5%), 10/10 on
  single-fix context and 22/23 on heterogeneous; the one miss was G033.
- **Methodological — the LLM judge fails on this rubric.** Raw agreement 30/33 =
  **90.9%**, but **Cohen's κ = −0.042**. All three disagreements fall on one axis,
  in both directions: the judge substituted *appropriateness* for *support*. It
  marked G021/G024 grounded (crediting the acknowledgement of mismatch, ignoring
  the unsupported fix that followed) and G012 ungrounded (because the supported
  fix was inappropriate to the ticket, which the rubric explicitly forbids).
  **Anti-aligned errors, not scattered ones** — which is why κ sits at or below
  zero rather than merely low. Had the judge been used as the scaling tool this
  harness started as, it would have reported ~97% grounded, missed both real
  failures and invented a third — and 90.9% raw agreement would have looked
  reassuring. **Never use an LLM judge unaudited; an aggregate agreement score
  will not reveal this.**
- Read the mechanism, not the magnitude: at n = 33 with three disagreements and
  31/33 in one category, prevalence makes κ hypersensitive and the point estimate
  carries enormous uncertainty. The load-bearing evidence is the case-by-case
  reading, possible only because the full population was labelled. **The hedge
  κ = 0.000 is degenerate and must never be cited** — the judge returned `true`
  on all 33, so it had no variance; only the 97% raw agreement is readable.

### 7.10 Phase 3 — the agent architecture (parity-preserving by design)

The one phase whose deliverable is architecture, and therefore the one most at
risk of quietly moving a published number. Every sub-phase was gated on the same
criterion — **the goldens must not move** — and none did: 45/45 and 9/9 exact,
adversarial 9/9, benchmark 32/45, `adversarial_escalation_results.csv`
byte-identical at every step. Test suite 69 → 104.

- **3A — Tier-1 persisted.** Was refitted from the CSV on every process start;
  fine for a script, wrong for a per-request service. Fit 1.56s vs load 0.014s —
  real but small beside BGE's ~60s, so **the justification is determinism and
  deployability, not speed**: the model became a pinned, fingerprinted object
  instead of something re-derived at boot. The substance is the guard (manifest,
  no refit-on-miss fallback). The specific trap: **Tier-1 is fitted on the full
  4,000 rows, not the 80/20 split every other script in `src/classification/`
  uses**, because that is what the goldens were captured under — a split fit
  would shift every cascade routing decision invisibly. Persistence verified
  bit-exact against a fresh fit (identical predictions, max confidence delta
  0.0). 3A also closed a live duplication: `streamlit_app` and
  `test_adversarial_escalation` each fitted their *own* Tier-1 and bypassed the
  artifact. `calibrate_conformal.py` and `plot_calibration_curves.py` keep their
  own fits on purpose (deliberate leave-out subsets). Suite 69 → 75.
- **3B — agent boundaries + orchestrator.** Declared dependencies; the
  orchestrator owns both gates as pure functions; per-agent `StepTrace`s
  including skipped steps, and `agent_status` / `agent_latency_ms` in the decision
  log (the per-agent history drift detection reads). Stage implementations
  deliberately untouched — they hold the parity-critical logic (the L2-normalise
  before search, the calibrated prompt, the retry ladder) and are wrapped, not
  absorbed. `pipeline.run()` stayed a façade, so all seven call sites were
  untouched. One observable difference on a path unreachable in practice: a
  missing Gemini client now records `error_kind` `ArtifactError` instead of
  `LLMError`. Suite 75 → 92.
- **3C — the HTTP service.** Gated in two halves; both held. Half 1: nothing that
  existed moved. Half 2: the new surface is specified by tests, since no golden
  covers it — `/triage` reproduces the library's decision for all 9 adversarial
  tickets and 15 of the 45; a fault-injected `RetrievalError` returns 200 +
  escalation envelope; a fault-injected `ValueError` returns 500. Verified
  against a real uvicorn process, not only `TestClient`: `adv_08` over HTTP
  returns tier 2, `tier1_conf` 0.3183, similarity 0.6124, escalated — the
  adversarial gate's numbers exactly. Two pure renames so the service does not
  reach into private names (`_rag_gate`/`_filing_gate` → `rag_gate`/`filing_gate`,
  `_build_gemini_client` → `build_gemini_client`). Suite 92 → 104.
- **3D — docs, and n8n dropped deliberately.** The README had drifted three phases
  behind the code (it still said the system was "not yet a true multi-agent
  system"). **n8n was dropped, not deferred:** an `IF confidence < threshold` node
  would be a second, untested copy of the calibrated 0.67 gate, living where
  neither `pytest`, the goldens, nor the adversarial gate can reach it — the most
  expensive possible instance of this project's recurring bug class — and a
  workflow JSON cannot be regression-tested against `tests/goldens/*.json`. The
  gap with Paper 1 closes on the orchestrator existing and being tested, not on
  which tool draws it. What n8n was genuinely for survives without authority:
  `POST /policy/rag-gate` returns the decision computed by the tested Python, so a
  workflow branches on a boolean it did not compute. **An n8n figure remains
  available as pure presentation for the paper; nothing depends on it.**

### 7.11 Phase 4A — decision-history sink + drift detector (complete and gated)

Framing: the recurring bug class is a value or artifact that is wrong for its
context, internally consistent, and therefore silent. `artifacts.py`'s guards
catch that at *load* time; drift detection is the population-level detector for
the same class — a world that has moved away from what the gates were calibrated
on shows up as a shift **in the gate's own inputs** before it shows up as a
visibly wrong answer.

**A prerequisite found while planning: there was no history.**
`logging_setup.py`, `orchestrator.py` and `schemas.py` all described decision
records as "the raw input for drift detection", and nothing ever persisted them —
`configure_logging()` attached only a stderr `StreamHandler`, and every
evaluation path passed `emit_log=False`. Doc-ahead-of-code, the same gap the BGE
clustering re-run closed.

What landed in `5c077dd`:

- `settings.drift` — `enabled=False`, `alpha=0.10`, `conditional_delta=0.10`,
  **no window size and no alarm threshold**, with `test_config.py` pinning their
  absence
- An **opt-in JSONL sink** (`configure_logging(decision_log_path=...)`),
  `pipeline_decision` records only, `schema_version: 1`, off by default and
  enabled by nothing in the project. It refuses a logger level above INFO,
  *before changing anything*, because a starved sink would record an empty
  history that reads as a quiet period
- `src/agent/drift.py` — pure: no I/O, no models, so it is testable without BGE.
  **Signal A** (primary): retrieval-novelty uniformity via
  `conformal_p_values()` verbatim (per ticket
  `p = conformal_p_values(cal_scores, [-top_similarity])`; under exchangeability
  p is super-uniform and drift piles mass at low p), with an exact binomial
  *and* a calibration-conditional test, plus a one-sample KS against U[0,1].
  **Signal B**: escalation rate, Tier-1 share (a *published* number — if it moves
  in deployment that is itself a finding), category mix. **Signal C**: config
  fingerprint counts, in their own field — a window with an unexpected
  fingerprint is a **deployment fault, not a distribution shift**, and conflating
  the two would bury it. The report carries statistics only, no alarms
- `load_drift_reference()` guarding embedding model, dimension, index size and
  FAISS file hash — deliberately *not* a full-fingerprint check, which would fire
  on every unrelated config edit
- `build_drift_reference.py` → `data/drift_reference_bge-base-en-v1-5.json`;
  refuses to overwrite without `--force`, and is fatal unless it reproduces the
  published Phase 1 detection values exactly (it reproduces all 24, plus 175/175
  pipeline-vs-direct similarity match)
- Reference distribution is the same 175 in-domain tickets the RAG gate was
  calibrated against — deliberately **not** a rolling baseline from recent logs,
  which re-centres on whatever is arriving and so cannot see slow drift
- `scipy==1.18.0` pinned. Suite 104 → **142**

**Four findings from building it:**

1. **The planned reference did not exist as data.** The plan named scores in
   `conformal_calibration_bge-base-en-v1-5.json`; that file holds classification
   fits only, and `calibrate_conformal.py` discards the 175 similarities.
   Doc-ahead-of-code, again. Built as a separate artifact rather than by re-running
   the Phase 1 script.
2. **"False-alarm rate is α by construction" was overstated in the plan.** It
   holds marginally over calibration draws, not for one fixed n=175 reference,
   where the per-ticket rate is Beta(17,159)-distributed. Exact marginal rate
   17/176 = 0.0966; **calibration-conditional upper bound at δ=0.10 is 0.126**
   against a nominal 0.10. At a ~200-ticket window that gap is as large as the
   window's own sampling noise. Both tests ship. **Never quote a window
   false-alarm rate that 4B has not measured.**
3. **The first independent-derivation check could not fail.** It compared a
   recomputed in-domain false-escalation rate to the published one, but a set
   scored against itself gives a rate fixed by n and ties alone — any 175 distinct
   numbers pass. Caught because self-inclusive and non-self scores "reproduced"
   identically. Replaced with the published OOD variant rate, OOD seed rate and
   adversarial flag count, which do differ between the two score sets at α=0.01
   (8 vs 6 adversarial flagged). Caveat: at α ≥ 0.05 most of those columns
   saturate, so the α=0.01 rows carry most of the check's power.
4. **The reference escalates 0/175.** In-domain paraphrases never reach the RAG
   gate, so a binomial against rate 0 would return p = 0 on the first escalation;
   the detector reports `degenerate_reference` with a `None` p-value instead.
   Consequence for the thesis: **this reference says nothing about a deployed
   escalation rate.** Reference Tier-1 share is 23/175 = 13.1%, and the predicted
   category mix is uneven (Database 43, Infrastructure 11) although the true
   labels are 25 each.

---

## 8. Research / novelty claim

Two real papers were reviewed to identify a genuine gap:

- **Paper 1** (multi-agent CX architecture) proposed an escalation/orchestration
  design but never implemented or empirically tested it.
- **Paper 2** (IT-ticket classification) rigorously tested classification on real
  enterprise data but never touched resolution generation, RAG, or
  confidence-based escalation. It *did* handle real-world class imbalance, which
  directly inspired this project's class-imbalance experiment.

**The contribution.** An actual end-to-end pipeline covering classification,
retrieval-grounded resolution, confidence-based escalation and
automation-flagging, empirically measured at every layer — including honest
reporting of the calibration methods that *didn't* work, and of behaviour under
both traffic-volume skew and training-data skew. Confidence-based cascading is
not itself novel (Viola & Jones 2001; FrugalGPT applies it to LLM cost). The
honest novelty claim is **applying and rigorously measuring this exact pattern,
calibrated against real data independently at three separate layers,
specifically for IT ticket triage** — plus the methodological findings that fall
out of it:

1. An in-domain-only calibration set can produce a misleading cliff-edge for a
   gate whose job is detecting out-of-domain input — and adding negative-class
   data does not fix it if the negative class's own distribution does not reach
   into the swept range (§7.3).
2. Conformal coverage transfer is a property of the **representation**, not only
   of how the data was sampled (§7.6, Finding 1) — the sharpest result here.
3. A calibration set built by paraphrasing template-generated training rows
   cannot be de-contaminated at any amount of filtering (Finding 2); and matching
   the deployment distribution is **necessary but not sufficient** (Finding 4).
4. An LLM judge can show 90.9% raw agreement with a human while being
   systematically anti-aligned (κ = −0.042), and only reading every disagreement
   reveals it (§7.9).
5. A pre-registered decision rule can invert a project's own cost asymmetry, and
   a synthetic-label dry run catches it before any labels are spent (§7.8).

**Important distinction to state carefully.** What exists: independent
Classification, Retrieval and Resolution agents with declared dependencies, an
orchestrator that owns every routing decision, each agent individually
addressable over HTTP, covered by 150 tests plus two fixed benchmark sets — this
is what closes Paper 1's gap. What does *not* exist: distributed execution, a
message bus, concurrency, autonomous agents, or any agent that decides anything.
**The honest claim is a sequential pipeline with real agent boundaries, an
orchestrator, and an HTTP surface.**

---

## 9. Testing and the safety nets

```powershell
pytest                     # full suite, offline, 150 tests, ~110s
pytest -m "not slow"       # fast subset, no model loading
pytest tests/test_pipeline_parity.py -v
python src/experiments/test_adversarial_escalation.py   # 9/9 gate + CSV report
python tests/capture_goldens.py                         # regenerate ONLY on purpose
```

- `pytest.ini` deselects `-m gemini` by default so the suite never spends quota.
  Markers: `slow` (loads BGE + fits Tier-1), `gemini` (live API call).
- **Golden parity is the refactor safety net.** `tests/goldens/*.json` record the
  exact routing decisions for both fixed benchmark sets (45/45 and 9/9). Any
  change that moves a number there is a regression unless it is deliberate.
- **The adversarial gate is the threshold safety net.** 9 hand-written tickets
  that must all escalate or proceed correctly. **Any change to a live threshold,
  embedding model, or retrieval path must be re-confirmed 9/9 before being
  committed.** It has already caught two real bugs.
- Test files: `test_config.py`, `test_schemas.py`, `test_artifacts.py`,
  `test_orchestrator.py`, `test_pipeline_parity.py`, `test_adversarial.py`,
  `test_service.py`, `test_conformal.py`, `test_drift.py`, `test_logging_sink.py`,
  `test_no_silent_fallback.py`, plus `conftest.py` (read-only benchmark fixtures)
  and `capture_goldens.py`.
- Suite growth by phase: 0 → 44 (Phase 0) → 69 → 75 (3A) → 92 (3B) → 104 (3C) →
  142 (4A) → 150 (uncommitted 4B-1 tests).

---

## 10. The recurring bug class — read this before touching anything model-related

Five documented occurrences, all the same shape: **a value or artifact that is
wrong for its context, stays internally consistent, and therefore produces wrong
results with no error.** The first four were model swaps; the fifth shows the
shape is not limited to those.

1. Silent stale embedding cache during the BGE swap.
2. `test_adversarial_escalation.py` keeping its own MiniLM constants (it passed
   3/9 at the stale threshold and the failure was initially misread).
3. `process_ticket_batch.py` encoding with MiniLM against a BGE index — fixed in
   Phase 0, and it had been that way for weeks.
4. `run_ablation_study.py` measuring the entire pre-BGE pipeline. **This one
   reached published results and stood for eleven days.**
5. `build_groundedness_set.py` testing `result.status == ResolutionStatus.ESCALATED`
   to decide which benchmark tickets reach the resolver. A RAG-gate escalation
   carries status `NEEDS_HUMAN_RESOLUTION`; `ESCALATED` belongs to the *filing*
   gate. The check ran clean, returned a plausible number, and silently classed
   all 21 escalating tickets as eligible — it would have spent 54 Gemini calls
   instead of 33 and drafted for tickets production never sends to the resolver.
   **Escalation is `decision.escalated`, never the status enum.** Caught only
   because the dry run's count (54/0) was checked against an independently
   measured number (33/21).

Plus a **suspected sixth, found 2026-09-20 and not yet fixed**: the
`scenario_id`-without-category template grouping in `calibrate_conformal.py`
(see the ⚠ note in §7.6). Same shape — internally consistent, plausible output,
wrong for its context, reached published results.

**Operational rule:** when touching anything model-related — or any
routing/eligibility test — assume another is waiting. Run `pytest` and the
adversarial gate before believing a green result, and **check any count you rely
on against a second, independent derivation of it.**

---

## 11. Known limits, inconsistencies, risks and open questions

### Hard limits of the dataset (the wall three phases hit independently)

- In-distribution accuracy is uninformative; only the 14- and 45-ticket
  benchmarks measure anything real.
- The 175-ticket in-domain calibration set **cannot be de-contaminated** —
  memorisation is template-level.
- Template-generated data **cannot** produce two tickets that look alike but need
  different fixes, so Phase 2A could not measure clustering precision at all.
- 85% of in-distribution retrievals collapse to a single fix, so groundedness was
  only measurable on the 33 out-of-template benchmark drafts.
- **Selection risk is accumulating on the 45-ticket benchmark** — it now carries
  classification accuracy, conformal coverage, groundedness and (in 4B) drift. No
  single experiment can see this.

### Known inconsistencies in the repo

- **Decision logs can be persisted, but nothing persists them.** The 4A sink is
  off by default and no script, test, service or demo turns it on — **no decision
  history exists yet; do not assume one.** Enabling it (e.g. in `/triage`) is a
  separate decision with its own gate.
- **Drift's "α by construction" is marginal only** (conditional bound 0.126 at
  δ=0.10 vs nominal 0.10). Never quote a window false-alarm rate 4B has not
  measured.
- **The drift reference escalates 0/175**, so the escalation-rate test is
  degenerate against it and reports `None`, not p = 0.
- `process_ticket_batch.py`'s dimension mismatch is fixed in code but the script
  has **not** been re-run: `data/category_stores/*.csv` were produced under
  MiniLM and feed the clustering calibration behind the production 0.80
  threshold. Regenerating them under BGE would silently invalidate it —
  re-running requires re-deriving that threshold in the same change and is
  **never a side effect**.
- Resolution clustering is still MiniLM on purpose; BGE is measured but not
  promoted, and promotion is now a *product* decision (§7.8).
- **Never use an LLM judge unaudited** here (§7.9).

### Risks to carry forward

- The recurring bug class has surfaced five times (once reaching published
  results); assume a sixth — one is already suspected in §7.6.
- Tier-1 being a persisted artifact is **new surface** for that class: a clean
  clone must run `train_tier1.py`, and the artifact must be rebuilt whenever
  `synthetic_tickets.csv` changes. Do **not** add a refit-on-miss fallback to
  make the error go away.
- Scope anchors buy label accuracy at the cost of diversity — check diversity
  whenever one is tightened (three generators use the pattern).
- Gemini free tier: regenerating the deployment calibration set costs ~360 calls;
  budget a day and dry-run first.
- Conformal guarantees are **marginal over the calibration draw**, not
  conditional on it; with n = 175 a single coverage estimate carries ~2 points of
  s.d. Do not read a small gap as a finding.

### Open questions (each needs its own decision and evidence)

1. **Should conformal be promoted to a production gate?** Tier-2 at α = 0.20
   lands exactly on nominal coverage with 80% singletons, a plausible replacement
   for the cascade's 0.50. Needs a deliberate decision, not a quiet flip of
   `enabled`.
2. **Should `process_ticket_batch.py` be re-run under BGE?** Code fixed, not
   executed; see the risk above.
3. **Can an LLM judge be used unaudited anywhere here?** Phase 2B says not for a
   support-vs-quality distinction.
4. **Should resolution clustering swap to BGE@0.90?** The evidence half is
   answered (no, indistinguishable on precision); what remains is the product
   call about review-queue capacity.
5. **Is the 45-ticket benchmark large enough to carry the conformal claims?**
   Each ticket is worth 2.2 coverage points, so ±2 s.d. at α = 0.10 is ~4.5
   points. Findings 1 and 2 clear that comfortably; some Tier-2 differences do
   not.
6. *(Answered in 3A)* Should Tier-1 be persisted? **Yes, and it is.**

---

## 12. Where the work stands (2026-09-20)

### Committed and pushed

Everything through `b52d4bc`. Phases 0, 1, 2A, 2B, 3A–3D and **4A** are complete
and gated; production gates are unchanged throughout at cascade **0.50**, RAG
**0.67**, clustering **0.80**.

Health, re-run today rather than quoted:

| Check | Result |
|---|---|
| `pytest` | **150 passed**, 0 failed, offline, exit 0 |
| Golden parity | 45/45 and 9/9 exact (inside the suite) |
| Adversarial gate (last recorded run) | 9/9 PASS, CSV byte-identical |
| Ablation baseline (last recorded run) | 71.11% (32/45), CSV byte-identical |
| `/health` fingerprint | `9c9a5cbcb53f` after 4B-1 (was `830c211b1fe9` at 4A; moved by `settings.drift.rate_reference_name`), index 4000, Tier-1 4000 rows |

### Phase 4B-1 (drift evaluation) — audited, run, gated on 2026-09-20

> **Updated after this file's first draft.** 4B-1 was audited (no correctness
> bugs), run in full, and gated the same day; its verdict is **measured, not
> shipped** — an eligible operating point exists (Signal A
> calibration-conditional binomial, α=0.05, W=100–200) and nothing was promoted
> into config, because the realistic-traffic arm showed the in-domain reference
> alarms at 4–7× its null on legitimate deployment-register traffic. The README's
> "Phase 4 — drift detection" section and `PROJECT_STATUS.md` carry the full
> result; the files below are what that gate covered.

| File | State |
|---|---|
| `src/experiments/evaluate_drift_detection.py` | **new, ~940 lines** — the null + power evaluation. Offline, no quota, seed 42. Outputs `drift_evaluation_null.csv`, `drift_evaluation_power.csv`, `drift_evaluation_summary.json` — **all three produced by the full run on 2026-09-20** |
| `data/drift_rate_reference_deployment_bge-base-en-v1-5.json` | **new** — Signal B's reference, built from the deployment-distribution set: n=175, **39 escalate (22.3%)**, 33 Tier-1, fingerprint `9c9a5cbcb53f` |
| `src/agent/config.py` | adds `drift.rate_reference_name` + `rate_reference_path` + its provenance entry |
| `src/agent/artifacts.py` | `load_drift_reference(source="in-domain"\|"deployment")`, same guard both ways |
| `src/agent/drift.py` | adds a **two-sample Fisher exact test** beside the one-sample binomial (so the reference rate's own estimation noise is accounted for), and lets Signal B use a different reference from Signal A |
| `src/experiments/build_drift_reference.py` | `--source deployment`, with escalation counted two ways (pipeline `decision.escalated` vs direct below-threshold) and fatal on disagreement or a degenerate rate |
| `tests/test_drift.py`, `tests/test_config.py` | +8 tests (Fisher vs hand computation, Fisher more conservative than the binomial, rate-reference defaulting, the committed reference being live and non-degenerate, unknown source refused) |

The script's docstring records **three deviations from the approved 4B plan, each
because a planned null was wrong**:

1. The planned "1,000 random splits of the 175 in-domain scores" check **could
   not fail** — for distinct scores a random split's flag count depends only on
   ranks, so it is exactly Beta-binomial for *any* 175 distinct numbers (the same
   trap as 4A's finding 3). Replaced by a synthetic null at the true n = 175,
   labelled a property of the procedure.
2. The first Signal B null bootstrapped windows from the 58-ticket half of each
   split, so every window carried extra between-pool variance and even Fisher read
   0.07–0.18. The null is now parametric: reference and window counts drawn
   independently from the reference rates.
3. Signal A power no longer fills its in-domain portion from a split pool; each
   in-domain ticket carries its leave-one-out conformal p-value against the other
   174, and contaminants are scored against the same 174-ticket reference size.

It also states that **every power number is provisional** because its in-domain
portion reuses the 175 reference tickets — 4B-3 is meant to replace that with a
held-out set. Signal B's 22.3% is **the escalation rate of Gemini-generated
benchmark-register tickets, not a measured production rate.**

### Immediate next step

**Review the 4B-1 gate** (committed, unpushed). Then **5A**: correct the
conformal template-grouping diagnostic described in the ⚠ note in §7.6 — it
changes a published number, so it gets its own sub-phase and gate.

### Remaining roadmap

1. **Phase 4B** — 4B-1 is done and gated (above); 4B-2/4B-3 (a held-out
   in-domain set, to replace the provisional power numbers) remain. Out of scope by
   prior decision: any live `/drift` endpoint, drift *gating* production,
   retraining triggers, input-side embedding drift (MMD), Docker/CI.
2. **Docker/CI packaging** — the last item in the agreed sequence. Cheaper now
   than when scoped: the service is the deployable unit, `/health` reports the
   config fingerprint, and `requirements.txt` is fully pinned.
3. **Re-run both Phase 2 harnesses on deployment-distribution data** — both are
   built, guarded, and need no code changes; only real resolved tickets.
4. Two deliberately deferred decisions, each needing its own gate: turning the
   decision-log sink on anywhere (e.g. in `/triage`), and any `/drift` endpoint.
5. Optional, presentation-only: an n8n figure branching on `POST /policy/rag-gate`.

---

## 13. Annotated file structure

```
ticket-routing-agent/
├── CLAUDE.md                  how the project works (rarely changes)
├── PROJECT_STATUS.md          where it currently is (every session)
├── README.md                  the lab notebook, 1,984 lines
├── PROJECT_OVERVIEW.md        this briefing (derived)
├── requirements.txt           fully pinned
├── pytest.ini                 markers; deselects `gemini` by default
├── .env                       gitignored — GEMINI_API_KEY
│
├── data/
│   ├── generate_dataset.py                          the generator (seed 42)
│   ├── synthetic_tickets.csv                        4,000 rows
│   ├── ticket_index_bge-base-en-v1-5.faiss          production index
│   ├── ticket_metadata_bge-base-en-v1-5.json        aligned metadata
│   ├── ticket_index.faiss / ticket_metadata.json    MiniLM-era, kept
│   ├── novel_tickets_expanded.json                  45-ticket benchmark (READ-ONLY)
│   ├── adversarial_escalation_tickets.json          9-ticket gate (READ-ONLY)
│   ├── calibration_tickets_paraphrased.json         175 in-domain calibration
│   ├── deployment_calibration_tickets.json          175 deployment-distribution
│   ├── deployment_calibration_tickets.predup.json   the 21-near-duplicate version
│   ├── deployment_calibration_rejected.json         every rejected candidate + reason
│   ├── ood_calibration_tickets.json                 45-ticket OOD set
│   ├── conformal_calibration_*.json/.csv            Phase 1 outputs
│   ├── conformal_novelty_results.csv
│   ├── drift_reference_bge-base-en-v1-5.json        Signal A reference (4A)
│   ├── drift_rate_reference_deployment_*.json       Signal B reference (4B, untracked)
│   ├── rag_similarity_calibration.csv               Attempt 1 (in-domain only)
│   ├── rag_similarity_calibration_combined.csv      Attempt 2 (+OOD)
│   ├── resolution_clustering_calibration_*.csv      pooled + per-category, MiniLM + BGE
│   ├── automation_candidates.json / _summary.csv    the production feature's output
│   ├── automation_flag_validation_*.json/.csv       Phase 2A harness + pilot
│   ├── groundedness_set.json / _key.json / _judge.json / _results.csv   Phase 2B
│   ├── ablation_{baseline,no-cascade,no-rag}_results.csv
│   ├── adversarial_escalation_results.csv           regression report
│   ├── calibration_tier{1,2}_reliability_diagram.png, calibration_reliability_data.csv
│   ├── category_stores/            per-category resolved tickets (MiniLM-era!)
│   ├── batch_intake/               the 500-ticket skewed batch
│   ├── embedding_comparison/       3-way model comparison
│   └── skewed/                     class-imbalance datasets + sweep results
│
├── src/
│   ├── agent/            THE library — all inference (see §5)
│   ├── service/api.py    FastAPI surface (no __init__.py — run from root)
│   ├── classification/   train_tier1, train_embeddings, train_cascade,
│   │                     train_baseline_tfidf, train_distilbert,
│   │                     train_embeddings_comparison, generalization_test
│   │                     (source of truth for NOVEL_TICKETS),
│   │                     generate_{calibration,ood_calibration,deployment_calibration}_set,
│   │                     build/generate/fix/finalize_* benchmark scripts
│   ├── rag/              build_vector_index.py, suggest_resolution.py
│   ├── app/              streamlit_app.py (the live demo)
│   └── experiments/      every measurement: run_ablation_study,
│                         test_adversarial_escalation, calibrate_conformal,
│                         calibrate_rag_similarity_threshold,
│                         calibrate_resolution_clustering(_percategory),
│                         explore_resolution_clustering, flag_automation_candidates,
│                         build/score_flag_validation_set,
│                         build_groundedness_set, run_groundedness_judge,
│                         score_groundedness_set, join_scenario_ground_truth,
│                         simulate_ticket_intake, process_ticket_batch,
│                         generate_skewed_datasets, run_imbalance_sweep,
│                         plot_calibration_curves, retry_failed_resolutions,
│                         build_drift_reference, evaluate_drift_detection (4B)
│
├── tests/                150 tests + goldens/ + capture_goldens.py
├── models/               gitignored: tier1_tfidf_logreg.joblib,
│                         ticket_classifier_bge-base-en-v1-5.joblib,
│                         ticket_classifier.joblib (MiniLM-era),
│                         distilbert_ticket_classifier/
└── venv/                 gitignored
```

---

## 14. Command cookbook

```powershell
# --- Core pipeline, in order, from a clean clone -------------------------
python data/generate_dataset.py
python src/classification/train_tier1.py
python src/classification/train_embeddings.py
python src/rag/build_vector_index.py
streamlit run src/app/streamlit_app.py
uvicorn src.service.api:app --port 8000          # from the project ROOT

# --- Service (PowerShell; curl is an alias for Invoke-WebRequest) --------
Invoke-RestMethod http://localhost:8000/health
Invoke-RestMethod -Method Post http://localhost:8000/triage `
  -ContentType 'application/json' `
  -Body '{"title":"vpn keeps dropping every ten minutes"}'

# --- Tests and gates ----------------------------------------------------
pytest
pytest -m "not slow"
python src/experiments/test_adversarial_escalation.py
python tests/capture_goldens.py                  # ONLY on purpose

# --- Ablation -----------------------------------------------------------
python src/experiments/run_ablation_study.py --mode baseline
python src/experiments/run_ablation_study.py --mode no-cascade
python src/experiments/run_ablation_study.py --mode no-rag

# --- Conformal (offline, no quota) --------------------------------------
python -m src.experiments.calibrate_conformal

# --- RAG threshold calibration ------------------------------------------
python src\classification\generate_ood_calibration_set.py     # ~45 Gemini calls
python -m src.experiments.calibrate_rag_similarity_threshold   # note: -m

# --- Clustering / automation flagging -----------------------------------
python src/experiments/join_scenario_ground_truth.py
python src/experiments/explore_resolution_clustering.py
python src/experiments/calibrate_resolution_clustering.py
python src/experiments/calibrate_resolution_clustering_percategory.py
python src/experiments/flag_automation_candidates.py           # production feature

# --- Phase 2 harnesses --------------------------------------------------
python src/experiments/build_flag_validation_set.py --pilot
python src/experiments/score_flag_validation_set.py --pilot
python src/experiments/build_groundedness_set.py --limit 3     # ALWAYS dry-run
python src/experiments/build_groundedness_set.py               # 33 Gemini calls
python src/experiments/run_groundedness_judge.py               # 33 Gemini calls
python src/experiments/score_groundedness_set.py               # offline

# --- Skew experiments ---------------------------------------------------
python src/experiments/simulate_ticket_intake.py
python src/experiments/process_ticket_batch.py     # ⚠ see Known risks first
python src/experiments/generate_skewed_datasets.py
python src/experiments/run_imbalance_sweep.py

# --- Phase 4 drift ------------------------------------------------------
python src/experiments/build_drift_reference.py                    # Signal A
python src/experiments/build_drift_reference.py --source deployment # Signal B
python src/experiments/evaluate_drift_detection.py --smoke         # 4B, in progress
python src/experiments/evaluate_drift_detection.py

# --- Deployment calibration set (~360 calls, ~30 min) -------------------
python -m src.classification.generate_deployment_calibration_set --per-category 1
python -m src.classification.generate_deployment_calibration_set

# --- Before committing --------------------------------------------------
git status | Select-String "\.npy"
```

`calibrate_rag_similarity_threshold`, `calibrate_conformal` and
`generate_deployment_calibration_set` run as modules (`-m`) because
`src/experiments/` and `src/classification/` have no `__init__.py` and resolve as
implicit namespace packages — so everything runs **from the project root**.

---

## 15. Working conventions (how to work on this project)

- **Phase-gated workflow.** Large scope is *sequenced into phases*, never cut.
  Every phase opens with a plan reviewed before any code is written, and ends at
  an explicit review checkpoint: finish the phase, report, stop. Work stays
  committed but unpushed until the gate clears. Commit directly to `main`; no
  feature branches.
- **Before declaring a gate cleared, re-run the health checks** rather than
  quoting the numbers in `PROJECT_STATUS.md` — this project's bug class is a
  value that is stale but internally consistent.
- **Update `PROJECT_STATUS.md` before ending a working session**: last-updated
  date, last commit SHA, what moved, what is next.
- **Human ground truth is the author's to produce.** Never label, infer, or
  substitute a model for a human label in this project.
- **Check a decision rule against the cost asymmetry** (precision over recall)
  *before* pre-registering it — Phase 2A is why.
- Seed 42 everywhere; the same 80/20 stratified split
  (`test_size=0.2, random_state=42`) in every classification script, **except**
  `train_tier1.py`, which fits on all 4,000 rows on purpose.
- Benchmarks are read-only. Never edit or regenerate the 14-, 45- or 9-ticket
  sets.
- Path resolution is always two directories up from the script's own location
  via `os.path.*`.
- Scripts fail with clear actionable messages, not tracebacks — the Streamlit
  demo may run live in front of an audience.
- Never overwrite a previous model's results file; BGE re-runs write to
  `*_bge-base-en-v1-5.*` alongside the MiniLM originals so comparisons stay
  auditable.
- Keep `GEMINI_CALL_DELAY_SEC` ≥ 4s in any batch-calling script, and dry-run
  anything that spends quota.

### The five-line version, for briefing a fresh session

> Read `PROJECT_STATUS.md`, then `CLAUDE.md`. This is a research project where
> the results are the deliverable, organised in numbered phases with a plan at
> the start and a review gate at the end of each. All inference lives in
> `src/agent/`; every calibrated constant lives in frozen `config.py` with its
> provenance; the goldens and the 9-ticket adversarial gate must be re-confirmed
> before any threshold, model or retrieval change is committed. The project's
> recurring bug is a value that is wrong for its context but internally
> consistent, so check every count against a second, independent derivation.
> Phases 0–3 and 4A are done and pushed; Phase 4B (drift evaluation) is written
> but unrun and uncommitted.
