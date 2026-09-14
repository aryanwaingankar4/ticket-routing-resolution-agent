# Project Status

**Last updated:** 2026-09-14
**Last commit:** `156f07b` — Phase 3B: agent boundaries and the
orchestrator
**Branch:** `main`, **ahead of `origin/main` (unpushed)** — Phase 3B is
committed locally and waiting on the review gate before it is pushed
**Current phase:** **Phase 3 (multi-agent orchestrator), sub-phases 3A and 3B
complete.** Both parity-preserving: no number moved. Production gates
unchanged. 3A's gate cleared and was pushed (`9320c18`).

Read this file first. `CLAUDE.md` describes how the project works and rarely
changes; this file describes where it currently is and changes every session.

---

## Health at a glance

| Check | Command | Current |
|---|---|---|
| Test suite | `pytest` | **92 passed**, offline, ~73s |
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

## In progress

**Nothing is mid-flight.** Working tree clean. Phase 3B is committed locally
and **not pushed** — it is waiting on the review gate.

---

## Immediate next step

**The Phase 3B review gate is here.** Confirm the change reads correctly, then
push. Do not start 3C before it clears.

**3C is where the agent failure boundary lands.** It was deliberately deferred
from 3B: agent errors still propagate exactly as they always have, and a
FastAPI service cannot crash the process on one bad request. Expect a new
`EscalationReason` for it, and note that it is a real behaviour change on the
failure path — plan it as such.

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
| 3C | FastAPI service — per-agent endpoints, `/triage`, startup preload, health endpoint reporting `config_fingerprint()`, plus the agent failure boundary deferred from 3B |
| 3D | n8n workflow over the endpoints (presentation only) + architecture write-up |

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
