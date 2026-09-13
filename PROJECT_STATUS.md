# Project Status

**Last updated:** 2026-09-14
**Last commit:** `d6a5434` — Add Phase 2/2A automation-flag validation
harness (no finding yet)
**Branch:** `main`, level with `origin/main` (nothing unpushed)
**Current phase:** Phase 2 complete (automation-flag validation). Result is
negative and deliberately so — see below. Production gates unchanged.

Read this file first. `CLAUDE.md` describes how the project works and rarely
changes; this file describes where it currently is and changes every session.

---

## Health at a glance

| Check | Command | Current |
|---|---|---|
| Test suite | `pytest` | **69 passed**, offline, ~60s |
| Escalation regression gate | `python src/experiments/test_adversarial_escalation.py` | **9/9 PASS** |
| Golden parity | `pytest tests/test_pipeline_parity.py` | 45/45 and 9/9 exact |
| Ablation baseline (45-ticket) | `run_ablation_study.py --mode baseline` | **71.11%** (32/45) |
| Phase 2 pilot | `score_flag_validation_set.py --pilot` | **0/12 false merges**, STOP verdict |

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

### Phase 2 — automation-flag validation (complete, negative result)

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

## In progress

**Nothing is mid-flight.** Working tree clean, everything pushed.

---

## Immediate next step

**Phase 2 is closed. The review gate is here** — confirm the finding reads
correctly before the next phase starts.

Next in the agreed sequence: **multi-agent orchestrator** → drift detection
→ Docker/CI packaging.

One prerequisite to settle first, already flagged as an open question: the
orchestrator cannot refit Tier-1 per request, so Tier-1 needs persisting
rather than fitting at startup. That is the natural first task of that
phase rather than a separate one.

---

## Open questions

- **Should conformal be promoted to a production gate?** Currently measurement
  only. Tier-2 at α = 0.20 lands exactly on nominal coverage with 80% singletons,
  which is a plausible replacement for the cascade's 0.50. Needs a deliberate
  decision with its own evidence, not a quiet flip of `enabled`.
- **Should `process_ticket_batch.py` be re-run under BGE?** The code is fixed but
  has not been executed. See Known risks.
- **Should resolution clustering swap to BGE@0.90?** Phase 2 answered the
  *evidence* half: no, not on precision grounds, because the two are
  indistinguishable there on this dataset. What remains is a product call —
  BGE surfaces 1,382 co-clustered pairs against MiniLM's 1,167, and whether
  that extra recall is wanted depends on review-queue capacity. Decide it as
  a product question or re-run the harness on deployment-distribution data;
  do not reopen it as a calibration question.
- **Should Tier-1 be persisted rather than fitted at startup?** It is currently
  refit on every startup. Not needed yet; it blocks the orchestrator phase, where
  a per-request service cannot refit per call.
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
- **The recurring stale-artifact bug class has surfaced four times**, most
  recently reaching published results and standing for eleven days. Anything
  touching a model, index, or threshold should be assumed to have a fifth
  instance waiting. Run `pytest` and the adversarial gate before believing a
  green result.
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
