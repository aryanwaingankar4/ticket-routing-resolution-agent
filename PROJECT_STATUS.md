# Project Status

**Last updated:** 2026-09-14
**Last commit:** `bcb21c8` — Add deployment-distribution calibration set:
distribution matching is necessary but not sufficient for conformal coverage
**Branch:** `main`, level with `origin/main` (nothing unpushed)
**Current phase:** Phases 0 and 1 complete. Ready for Phase 2 planning.

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

---

## In progress

Nothing is mid-flight. The working tree is clean apart from the item below.

**Uncommitted and deliberately so — BGE clustering Phase 2 work.** Three
modified scripts (`calibrate_resolution_clustering.py`,
`..._percategory.py`, `explore_resolution_clustering.py`) plus four BGE result
files. This predates the current work and is unrelated to Phases 0 and 1. The
README already documents these results while the code that produced them is not
in history. It wants its own commit; it was kept out of the phase commits so
those stayed clean.

---

## Immediate next step

**Plan Phase 2 — the resolution-quality evaluation harness.** Enter plan mode,
present the plan, and wait for review before writing code.

It is the highest-value next phase for a specific reason: there is currently no
ground truth for resolution quality anywhere in the project, and that absence is
exactly what blocks the long-standing BGE automation-flagging decision. Phase 2
would unblock a decision that has been deferred for want of a validation method,
rather than adding a measurement for its own sake.

Remaining sequence after that: multi-agent orchestrator → drift detection →
Docker/CI packaging.

---

## Open questions

- **Should conformal be promoted to a production gate?** Currently measurement
  only. Tier-2 at α = 0.20 lands exactly on nominal coverage with 80% singletons,
  which is a plausible replacement for the cascade's 0.50. Needs a deliberate
  decision with its own evidence, not a quiet flip of `enabled`.
- **Should `process_ticket_batch.py` be re-run under BGE?** The code is fixed but
  has not been executed. See Known risks.
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
