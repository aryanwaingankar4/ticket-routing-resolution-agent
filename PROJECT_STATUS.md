# Project Status

**Last updated:** 2026-09-15
**Last commit:** `4086c8a` — Phase 4 plan: drift detection
**Branch:** `main`, level with `origin/main` (nothing unpushed)
**Current phase:** **Phase 3 complete; Phase 4 (drift detection) is planned
and approved but NOT started.** No Phase 4 code exists — the section below is
the plan, not results.

3A (Tier-1 persistence), 3B (agent boundaries + orchestrator), 3C (HTTP
service + failure boundary) and 3D (the README write-up) are all done, gated
and pushed. Production gates unchanged throughout — the whole phase moved no
number. The one deliberate omission: **the n8n workflow was dropped, not
deferred by accident.** See "Phase 3D" below.

Read this file first. `CLAUDE.md` describes how the project works and rarely
changes; this file describes where it currently is and changes every session.

---

## Health at a glance

| Check | Command | Current |
|---|---|---|
| Test suite | `pytest` | **104 passed**, offline, ~152s |
| Service | `uvicorn src.service.api:app --port 8000` → `GET /health` | fingerprint `7538ea7cceb1`, index 4000, Tier-1 4000 rows |
| Escalation regression gate | `python src/experiments/test_adversarial_escalation.py` | **9/9 PASS** |
| Golden parity | `pytest tests/test_pipeline_parity.py` | 45/45 and 9/9 exact |
| Ablation baseline (45-ticket) | `run_ablation_study.py --mode baseline` | **71.11%** (32/45) |
| Phase 2A pilot | `score_flag_validation_set.py --pilot` | **0/12 false merges**, STOP verdict |
| Phase 2B groundedness | `score_groundedness_set.py` | **31/33 grounded** (93.9%), judge κ = −0.042 |

Production gates are unchanged and remain the calibrated values: cascade
**0.50**, RAG similarity **0.67**, resolution clustering **0.80**.

---

## What's done

### Phase 0 — pipeline consolidation (`a3f0b23`)

Four independent copies of the classify → retrieve → escalate pipeline
collapsed into one implementation in `src/agent/`, proven behaviour-preserving
against goldens captured *before* any code moved (45/45 and 9/9 exact).

- Frozen typed config, every calibrated constant in one place with provenance
- Pydantic schemas for every stage; enums replacing free-string statuses
- One loader with two hard guards: index/metadata alignment, and
  encoder dim == index dim == configured dim
- Typed exceptions replacing `sys.exit()`; one Gemini error ladder, was three
- Structured JSON decision logs
- 44 tests where `tests/` had been empty
- **Fixed** `process_ticket_batch.py` encoding with MiniLM against a BGE index
- **Corrected a published result**: the ablation study had been measuring the
  entire pre-BGE pipeline. 68.89% → 71.11%; cascade gain 33.3 → 35.6 points

### Phase 1 — conformal prediction (`148ad8b`, `bcb21c8`)

Split conformal over both cascade tiers plus conformal novelty detection for
the RAG gate. **Measurement only** — `settings.conformal.enabled` is `False`,
production gates are untouched, golden parity holds.

Four findings:

1. **Coverage transfer is a property of the representation.** Same calibration
   set, same α, same benchmark: TF-IDF loses 23.3 coverage points, BGE loses
   1.1, against a 4.5-point noise band.
2. **The in-domain calibration set cannot be de-contaminated.** Memorisation is
   template-level — 12 templates, ~430 rows each, the set touches 11. Row
   removal changes nothing; template removal would leave 40 of 4,000 rows.
3. **Conformal novelty detection matches production**: 9/9 adversarial at
   α ≥ 0.05, 100% seed-level OOD detection, with a calibrated false-escalation
   rate the hand-tuned 0.67 threshold never had.
4. **Distribution matching is necessary but not sufficient.** A 175-ticket
   deployment-distribution calibration set recovers ~38% of Tier-1's shortfall
   and leaves it six times outside the noise band. For Tier-2 the payoff is set
   size instead: identical coverage, singletons on 46.7% of benchmark tickets
   versus 35.6% — more autonomy at the same risk.

### BGE resolution-clustering re-run (`c9d956d`)

Measurement only; production automation-flagging still runs at 0.80 on MiniLM.
Closes a doc-ahead-of-code gap — the README had described these results since
the README commit while the code and data that produced them were never in
history.

- BGE's pooled cliff-edge is **0.90** (precision 1.0000 at 0.90, breaking to
  0.9814 at 0.85), against MiniLM's 0.80
- Per-category: six categories cliff at 0.85, Access Management at 0.90
- The citable comparison is the **spread**: MiniLM's per-category cliffs ranged
  across a 0.10 band with 3 of 7 diverging from pooled; under BGE every category
  sits within a single 0.05 step. BGE clusters more consistently across
  categories
- Output paths are model-aware suffixed, so the MiniLM artefacts survive for
  comparison rather than being overwritten

---

### Phase 2A — automation-flag validation (complete, negative result)

Built the ground-truth check the README named as the prerequisite for
swapping resolution clustering from MiniLM@0.80 to BGE@0.90. **The answer is
that this dataset cannot answer the question**, and that is the finding.

#### The structural diagnostic

At their own cliff-edges, **neither configuration ever merges across dataset
templates**. All 1,069 pairs both merge, and all 411 pairs they disagree
about, are within-template. MiniLM@0.80 merges 1,167 pairs total; BGE@0.90
merges 1,382; the disagreement is 98 MiniLM-only + 313 BGE-only.

So the entire measured difference between them is **recall**, not precision.

#### A pre-registered rule that had to be thrown out

The original rule — an exact binomial on which configuration wins more
discordant pairs — would have promoted whichever model merges more,
inverting the project's precision-over-recall stance. A dry run against
synthetic labels confirmed it: label everything `same_fix` and it printed
PROMOTE at p = 0.0027 on zero evidence about flag correctness.

Replaced before any label was collected. **Primary** is now the false-merge
rate on each configuration's *extra* merges; promote BGE only if it makes
zero observed false merges **and** MiniLM makes at least one. If neither
does, the verdict is **no precision signal**, not promotion. The binomial
survives as an explicitly demoted recall comparison.

#### The pilot and its result

12 pairs (6 per direction) from the most divergent end of the region,
divergence 0.53–0.68 against a region median of ~0.42, labelled blind.

| Configuration | Extra merges judged | False merges | Rate |
|---|---:|---:|---:|
| MiniLM @ 0.80 | 6 | **0** | 0.0% |
| BGE @ 0.90 | 6 | **0** | 0.0% |

**Zero false merges by either.** The closest call was PL005, labelled
`same_fix` but flagged low-confidence — two `fstab` boot failures where one
ticket adds an explicit restart step and the other does not. PL009/PL010
(cause named on one side only, identical remediation) and PL011 (same
branch twice) resolved cleanly. Secondary `scenario_id` agreement was 12/12,
the degeneracy the diagnostic predicted.

**Zero observed is not zero.** Rule-of-three upper bounds: 50% per
configuration at 0/6, 25% pooled at 0/12. This does not establish that no
false-merge case exists — only that none was found among the 12 most
divergent pairs. What carries the conclusion is the diagnostic underneath
it, not the pilot alone.

#### Conclusion

MiniLM@0.80 and BGE@0.90 are **indistinguishable on precision** and differ
only in recall. Promotion would surface more candidates (1,382 vs 1,167
pairs), which is a product judgement about review-queue capacity, not a
calibration result. **Production stays on MiniLM@0.80.** The other 48
judgements were not spent — the pilot existed to determine whether they
would measure anything, and they would not.

The limitation is the dataset, not the method: template-generated data
cannot produce two tickets that look alike but need different fixes, because
the templates *are* the fix classes. Same wall the conformal work hit from a
different direction.

---

### Phase 2B — resolution groundedness (complete)

Committed in `639ab6a` (harness + data) and the finding commit below.
Measurement-only; production untouched.

**Scope was set by a pre-flight diagnostic run before any quota was spent.** A
judge can only find an unsupported claim if the retrieved context permits one:

| Query set | All 5 → one fix | 2+ distinct fixes |
|---|---:|---:|
| In-distribution (n=200) | **85.0%** | 15.0% |
| Novel-45 | 28.9% | **71.1%** |
| Adversarial-9 | 11.1% | **88.9%** |

The in-distribution corpus is degenerate — the same wall 2A hit. The
benchmarks are not, so 2B could proceed where 2A could not. Of 54 benchmark
tickets 21 escalate at the RAG gate, leaving **33** — the entire eligible
population, not a sample.

**Primary — groundedness: 31/33 = 93.9%** (Wilson 95% CI 80.4–98.3%). Zero
`partially_grounded`; two `ungrounded`.

Both failures share one shape. G021 (laptop disk full) and G024 (ransomware)
retrieved irrelevant context, correctly *said so*, and then prescribed a
substantive fix from general knowledge anyway. The contrast is what makes it a
finding: **10 drafts contain explicit mismatch language, 8 declined cleanly
and are grounded, only these 2 acknowledged the mismatch and answered
regardless.** The model can tell when retrieval is irrelevant; that
recognition just does not reliably stop it answering. G024 is the cleanest
illustration that groundedness is not quality — "disconnect from the network
and escalate to security" is excellent ransomware advice and entirely
ungrounded.

**Secondary — hedge appropriateness: 32/33 = 97.0%** (CI 84.7–99.5%), 10/10
on single-fix context and 22/23 on heterogeneous. The one miss was G033.

**Methodological — the LLM judge fails on this rubric.** Raw agreement 30/33 =
90.9%, but **Cohen's κ = −0.042**. All three disagreements fall on one axis,
in both directions: the judge substituted *appropriateness* for *support*.
G021/G024 it marked grounded, crediting the acknowledgement of mismatch and
ignoring the unsupported fix that followed — over-applying the
declining-is-not-unsupported clause. G012 it marked ungrounded because the
supported fix was inappropriate to the ticket, which the rubric explicitly
forbids. Anti-aligned errors, not scattered ones, which is why κ sits at or
below zero rather than merely low.

Read the mechanism, not the magnitude: at n = 33 with three disagreements and
31/33 in one category, the prevalence effect makes κ hypersensitive and the
point estimate carries enormous uncertainty. The load-bearing evidence is the
case-by-case reading, possible only because the full population was labelled.
Had the judge been used as the scaling tool this harness started as, it would
have reported ~97% grounded, missed both real failures and invented a third —
and 90.9% raw agreement would have looked reassuring.

The hedge κ of 0.000 is degenerate and must never be cited: the **judge**
returned `true` on all 33, so it had no variance; only the 97% raw agreement
is readable.

---

### Phase 3A — Tier-1 persistence (`1c753ae`)

First sub-phase of the orchestrator work. Tier-1 was refitted from
`synthetic_tickets.csv` on every process start; a per-request service cannot
do that, so it is now a persisted artifact built by
`python src/classification/train_tier1.py`.

**Parity-preserving, as Phase 3 is meant to be — no number moved.** Goldens
exact (45/45 and 9/9), adversarial 9/9, and
`data/adversarial_escalation_results.csv` regenerates byte-identical.
Persistence is verified bit-exact against a fresh fit: predictions identical,
max confidence delta 0.0.

Measured: fit 1.56s vs load 0.014s. The latency win is real but small beside
BGE's ~60s, so the justification is determinism and deployability — the model
is a pinned, fingerprinted object rather than something re-derived at boot.

**The guard is the substance.** Persisting a model creates a new place for a
stale artifact to hide, so the bundle carries a manifest (dataset sha256, rows
fitted, sklearn version, vectorizer config) that `artifacts.load_tier1()`
verifies. No refit-on-miss fallback — that would be the silent-fallback
pattern `test_no_silent_fallback.py` bans. The specific trap it catches:
Tier-1 is fitted on the **full** 4,000 rows, not the 80/20 split every other
script in `src/classification/` uses, and a split fit would shift every
cascade routing decision invisibly.

**Also consolidated three Tier-1 derivations into one.** `streamlit_app.py`
and `test_adversarial_escalation.py` each still fitted their own Tier-1 and
passed it to `pipeline.run()`, so both bypassed the persisted artifact — the
same shape as the four divergent loaders Phase 0 removed. Both now load
through `artifacts.load_tier1()`. `calibrate_conformal.py` and
`plot_calibration_curves.py` keep their own fits deliberately: they fit on
leave-out subsets and must not use the production artifact.

Test suite 69 → 75; `conformal.py` was also added to the import-side-effect
guard list, where it had been missing since Phase 1.

---

### Phase 3B — agent boundaries and the orchestrator (`156f07b`)

The restructure the phase is named for. The three stages were free functions
with three different signatures, each taking the whole `Artifacts` blob and
reaching into whatever it needed; nothing declared what a stage actually
depended on, so nothing could be moved or served independently.

**Parity-preserving — no number moved.** Goldens exact, adversarial 9/9,
benchmark 32/45, regression CSV byte-identical. **No consumer changed**:
`pipeline.run()` stayed as a façade over `orchestrator.run()`, so all seven
call sites were untouched.

What is actually different, and why each matters for 3C:

- **Declared dependencies.** Each agent names the `Artifacts` fields it
  requires, validated at construction — a missing *or misspelled* dependency
  fails before routing starts. That declaration is the agent boundary written
  down, and it is what makes splitting the agents across HTTP endpoints in 3C
  mechanical rather than exploratory.
- **The orchestrator owns all routing.** No agent reads a threshold or knows
  what runs after it. Both gates are pure functions (`_filing_gate`,
  `_rag_gate`) testable without loading a model — an escalation policy that
  needs BGE and FAISS to test is one nobody tests.
- **Per-agent traces.** `PipelineResult.steps` records every agent including
  the skipped ones, and the decision log carries `agent_status` and
  `agent_latency_ms`. A resolution step marked `skipped` is the positive
  evidence that the RAG gate held and no LLM call was made. This is the
  per-agent history drift detection will read.

Stage implementations (`classifier.py`, `retriever.py`, `resolver.py`) were
deliberately **not** touched — they hold the parity-critical logic (the
L2-normalise before search, the calibrated prompt, the retry ladder) and are
wrapped, not absorbed.

The one observable difference, on a path unreachable in practice: a missing
Gemini client is now recorded with `error_kind` `ArtifactError` where it said
`LLMError`. Same status, same escalation, same gate.

Test suite 75 → 92.

---

### Phase 3C — the HTTP service and the agent failure boundary (`400eaff`)

`src/service/api.py`: per-agent endpoints, `/policy/rag-gate`, `/triage`, and
`/health`. The first phase that adds externally-visible behaviour, so it was
gated in two halves and **both held**.

*Half 1 — nothing that existed moved.* The library is untouched: goldens exact,
adversarial 9/9, benchmark 32/45, regression CSV byte-identical.

*Half 2 — the new surface is specified by tests, since no golden covers it.*
`/triage` reproduces the library's decision for all 9 adversarial tickets and
15 of the 45; a fault-injected `RetrievalError` returns 200 with an escalation
envelope; a fault-injected `ValueError` returns 500. Verified against a real
uvicorn process, not only TestClient: `adv_08` over HTTP returns tier 2,
`tier1_conf` 0.3183, similarity 0.6124, escalated — the adversarial gate's
numbers exactly.

**The failure boundary is the service's, not the library's — a correction to
what this file said last session.** The plan was a new `EscalationReason`, and
that turned out to be the wrong design: `PipelineResult.classification` is a
**required** field, so a classification failure cannot produce a
`PipelineResult` without inventing a category, and fabricating a routing
decision is the one thing this system is built not to do. Making it optional
would have pushed `None`-safety onto ~20 dereference sites and traded a caught
error for an `AttributeError` in the live demo.

So the boundary lives in the API's own response envelope, which needs no
classification. The library still raises — deliberately: in a calibration run a
`RetrievalError` silently becoming an escalation row would corrupt the result
quietly, which is the recurring bug class wearing a new hat. **Only known
`AgentError`s become escalations; anything else is a 500**, because laundering
an unknown bug into a plausible response is that same failure shape.

Other decisions worth carrying forward:

- **`/triage` orchestrates in-process**, calling `pipeline.run()` rather than
  its own endpoints. Golden parity must not depend on a running server. The
  agents are independently *addressable*, not independently *running* — do not
  describe this as a distributed system.
- **`/policy/rag-gate`** returns the escalation decision computed by the tested
  Python, so 3D's n8n workflow gets a visible IF-branch without ever holding a
  threshold.
- **`/triage` defaults to `generate_resolution=false`.** 500 calls/day, and a
  looping workflow would drain it.
- **The service loads artifacts one way only** (`require_gemini=False`) and
  attaches a Gemini client via `dataclasses.replace()`. `load_artifacts` is
  `lru_cache(maxsize=2)` keyed on that flag, so calling it both ways in one
  process would load **BGE twice**.
- Endpoints are sync and serialised by one lock: one ticket at a time, on
  purpose. This is a research demo, not a throughput exercise.

Two library renames, both pure and behaviour-free, so the service does not
reach into private names: `orchestrator._rag_gate`/`_filing_gate` →
`rag_gate`/`filing_gate`, and `artifacts._build_gemini_client` →
`build_gemini_client`.

Test suite 92 → 104. `httpx` pinned in `requirements.txt` — needed by
`fastapi.testclient`, previously only a transitive dependency, so a clean clone
could not have run the suite.

---

### Phase 3D — the README write-up, and n8n dropped (`9f893c5`)

Docs only; no code touched, so the gates are unchanged from `400eaff`.

`README.md` is the lab notebook and had drifted **three phases behind the
code**: it still said the system was *"not yet a true multi-agent system"* and
scoped the restructure as future work via n8n, and its structure tree predated
the Phase 0 consolidation entirely — `src/agent/` was not in it.

Now documented: a full "Phase 3 — the agent architecture" section covering all
three sub-phases, including the two places the plan changed and why (the
failure boundary moving to the service; n8n being superseded). The "Important
distinction" in Research / Novelty now separates **what exists** (agents,
orchestrator, HTTP surface — closing Paper 1's gap) from **what does not** (no
distributed execution, no concurrency, no autonomous agents). The agents are
independently *addressable*, not independently *running*.

**The n8n workflow was dropped deliberately, not deferred.** Orchestration in
n8n would put an `IF confidence < threshold` node — a second, untested copy of
the calibrated 0.67 gate — where neither `pytest`, the goldens, nor the
adversarial gate can reach it. That is the most expensive possible instance of
this project's recurring bug class, and a workflow JSON cannot be
regression-tested against `tests/goldens/*.json` either. What n8n was actually
for survives without authority: `POST /policy/rag-gate` returns the decision
computed by the tested Python, so a workflow can branch on a boolean it did not
compute. **An n8n figure remains available as pure presentation if it is ever
wanted for the paper; nothing depends on it.**

Also fixed: the service examples in "How to Run" use `Invoke-RestMethod`, not
`curl` — on Windows PowerShell `curl` is an alias for `Invoke-WebRequest` and
accepts neither `-X` nor `-d`. Both commands were run against a live server
before being documented.

---

## Phase 4 — drift detection (PLANNED, NOT STARTED)

**No code exists for this yet. Everything below is the approved plan.**

### Why, and the framing

The recurring bug class — five documented instances — is a value or artifact
that is **wrong for its context, internally consistent, and therefore silent**.
`artifacts.py`'s three guards catch that at *load* time. Drift detection is the
population-level detector for the same class: an artifact that is wrong but
still loadable, or a world that has moved away from what the gates were
calibrated on, shows up as a distribution shift **in the gate's own inputs**
before it shows up as a visibly wrong answer.

### The prerequisite found while planning: there is no history

`logging_setup.py`, `orchestrator.py` and `schemas.py` all describe decision
records as "the raw input for drift detection". **They are never persisted.**
`configure_logging()` attaches only a `StreamHandler(sys.stderr)` — no
`FileHandler`, no `.jsonl`, no `logs/` — and every evaluation path passes
`emit_log=False` besides. Doc-ahead-of-code, the same gap the BGE clustering
re-run closed. Also logged under CLAUDE.md "Known inconsistencies". Making the
history real is 4A's first task.

### The mechanism

- **Signal A (primary) — retrieval novelty uniformity.** Reuses
  `conformal.conformal_p_values()` verbatim. Per ticket,
  `p = conformal_p_values(cal_scores, [-top_similarity])`; under
  exchangeability `p` is (super)uniform on [0,1], and drift piles mass at low
  `p`. Window test: exact binomial on `#{p ≤ α}` against α, with a one-sample
  KS against U[0,1] as a secondary read. **The false-alarm rate is α by
  construction, not by tuning** — the same property that made conformal
  novelty detection worth reporting in Phase 1, applied to a window rather
  than a ticket.
- **Signal B — the rates the thesis rests on.** Escalation rate, Tier-1 share
  (a *published* number — if it moves in deployment that is itself a finding),
  and category mix against the reference mix.
- **Signal C — config fingerprint, reported separately.** More than one
  fingerprint in a window, or one differing from the reference, is a
  **deployment fault, not a distribution shift**; conflating the two would
  bury it. The recurring bug class, detected after the fact.

**Reference distribution:** the in-domain calibration scores in
`data/conformal_calibration_bge-base-en-v1-5.json` — the same reference the RAG
gate was calibrated against. Deliberately **not** a rolling baseline from
recent logs, which re-centres on whatever is arriving and so cannot see slow
drift; that would define away the failure mode that matters most.

### Sub-phases, gate between each

**4A — persist the history, and the detector library.** `settings.drift`
(`enabled` default `False`); an *optional* JSONL sink in `logging_setup.py`,
opt-in and **default off**, following the `filing_gate_enabled` precedent so no
script inherits a new side effect; `src/agent/drift.py` as pure functions over
records returning a `DriftReport` — **no I/O, no models**, so the detector is
testable without BGE; `tests/test_drift.py`.

**4B — the evaluation (this is the finding).**
`src/experiments/evaluate_drift_detection.py`, offline, no quota. Windows built
from the **existing fixed sets only**: null windows resampled in-distribution,
contaminated windows mixing the 45-ticket benchmark / 45-ticket OOD set /
adversarial 9 at 0 / 5 / 10 / 25 / 50%. Reports detection rate across
(window size × contamination) **and** the null false-alarm rate. The useful
operational output: how many tickets pass before a given contamination level is
noticed.

### Gate criterion — not purely parity-preserving, so it splits

1. **Nothing that already exists moves.** Drift gates no routing decision;
   goldens exact, 9/9, benchmark 32/45, regression CSV byte-identical, and a
   test proving the log sink writes nothing by default.
2. **The detector ships with its false-alarm rate, or it does not ship.** A
   detection-rate number alone is a hand-tuned threshold wearing a lab coat,
   and this project has rejected exactly that twice (the in-distribution split;
   the 35-ticket calibration set). No operating window or threshold gets named
   without the null measurement beside it.
3. **Limitations written down with the result**, not after: selection risk is
   accumulating on the 45-ticket benchmark (accuracy, conformal coverage,
   groundedness, now drift); template-generated data makes the null unusually
   clean, so the measured false-alarm rate is optimistic; and contamination
   with benchmark/OOD text is a *known, abrupt* shift, so it measures power
   against out-of-template text rather than the gradual shift a deployed
   monitor would face. The same dataset wall Phases 1 and 2 hit, from a third
   direction.

**Out of scope:** any live `/drift` endpoint (a separate decision once
thresholds are measured), drift *gating* production, retraining triggers,
input-side embedding drift (MMD — a second detector would need its own
false-alarm measurement, and Signal A already has one), and Docker/CI.

---

## In progress

**Nothing is mid-flight.** Working tree clean, everything pushed. No Phase 4
code has been written.

---

## Immediate next step

**Begin Phase 4 implementation once the plan above is reviewed and approved —
resuming tomorrow.** Start with 4A, stop at its gate, and do not roll into 4B.

After Phase 4, one item remains from the agreed sequence: **Docker/CI
packaging** — cheaper now than when it was scoped, since the service is the
deployable unit, `/health` reports the config fingerprint, and
`requirements.txt` is fully pinned.

Phase 3 was planned with two decisions taken up front:

- **Orchestration logic stays in Python.** Agents become independent modules
  behind a typed contract with FastAPI as transport; n8n may wrap the
  endpoints later for the demo but owns no decision, because the escalation
  gate must stay inside pytest and the goldens.
- **Phase 3 is parity-preserving architecture, not a new experiment.** Success
  is that no number moves. Any new claim needs its own evidence and gate.

Remaining sub-phases, each with its own gate:

| Sub-phase | Scope |
|---|---|
| ~~3B~~ | ~~Classification / Retrieval / Resolution behind a typed contract, plus `orchestrator.py`~~ — **done** |
| ~~3C~~ | ~~FastAPI service — per-agent endpoints, `/triage`, health endpoint, plus the agent failure boundary~~ — **done** |
| ~~3D~~ | ~~architecture write-up~~ — **done**; the n8n workflow was **dropped deliberately**, and remains available as optional presentation only |

After Phase 3: drift detection → Docker/CI packaging.

---

## Open questions

- **Should conformal be promoted to a production gate?** Currently measurement
  only. Tier-2 at α = 0.20 lands exactly on nominal coverage with 80% singletons,
  which is a plausible replacement for the cascade's 0.50. Needs a deliberate
  decision with its own evidence, not a quiet flip of `enabled`.
- **Should `process_ticket_batch.py` be re-run under BGE?** The code is fixed but
  has not been executed. See Known risks.
- **Can an LLM judge be used unaudited anywhere in this project?** Phase 2B
  says not for a support-versus-quality distinction: raw agreement of 90.9%
  concealed a systematically wrong judge (κ = −0.042), and only labelling the
  full population exposed it. Any future use of an LLM judge needs a human-
  labelled subset large enough to read every disagreement, not just an
  aggregate agreement score.
- **Should resolution clustering swap to BGE@0.90?** Phase 2 answered the
  *evidence* half: no, not on precision grounds, because the two are
  indistinguishable there on this dataset. What remains is a product call —
  BGE surfaces 1,382 co-clustered pairs against MiniLM's 1,167, and whether
  that extra recall is wanted depends on review-queue capacity. Decide it as
  a product question or re-run the harness on deployment-distribution data;
  do not reopen it as a calibration question.
- ~~**Should Tier-1 be persisted rather than fitted at startup?**~~ **Answered
  in Phase 3A: yes, and it is.** Loaded from `models/tier1_tfidf_logreg.joblib`
  through `artifacts.load_tier1()`, behind a manifest guard, with no behaviour
  change.
- **Is the 45-ticket benchmark large enough to carry the conformal claims?** Each
  ticket is worth 2.2 coverage points, so a ±2 s.d. band at α = 0.10 is ~4.5
  points. Findings 1 and 2 clear that comfortably; some Tier-2 differences do not.

---

## Known risks

- **Re-running `process_ticket_batch.py` would silently invalidate a calibrated
  threshold.** `data/category_stores/*.csv` were produced under MiniLM and feed
  the resolution-clustering calibration behind the production 0.80 threshold.
  Regenerating them under BGE requires re-deriving that threshold in the same
  change. Never a side effect.
- **The recurring stale-artifact bug class has surfaced five times**, once
  reaching published results and standing for eleven days. Anything touching a
  model, index, or threshold should be assumed to have a sixth instance
  waiting. Run `pytest` and the adversarial gate before believing a green
  result.
- **Tier-1 is now a persisted artifact, which is new surface for that class.**
  A clean clone must run `python src/classification/train_tier1.py`, and the
  artifact must be rebuilt whenever `synthetic_tickets.csv` changes. The
  manifest guard turns both into a loud failure rather than a silent wrong
  answer, but the guard is only as good as the manifest — do not add a
  refit-on-miss fallback to make the error go away.
- **Scope anchors buy label accuracy at the cost of diversity.** Tightening the
  Infrastructure anchor cut self-consistency rejections from 36% to 3% and
  produced 21 near-duplicate pairs. Three Gemini generators in this project use
  that prompt pattern; check diversity whenever one is tightened.
- **Gemini free tier is 15 requests/minute, 500/day.** Regenerating the
  deployment calibration set costs ~360 calls. Budget a day for it and dry-run
  first.
- **Conformal guarantees are marginal over the calibration draw**, not
  conditional on it. With n = 175 a single coverage estimate carries ~2 points of
  standard deviation. Do not read a small gap as a finding.
