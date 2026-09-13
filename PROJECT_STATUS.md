# Project Status

**Last updated:** 2026-09-14
**Last commit:** `65840c8` — Refresh PROJECT_STATUS.md after committing the
BGE clustering re-run
**Branch:** `main`, level with `origin/main` (nothing unpushed)
**Current phase:** Phase 2 (automation-flag validation harness). The 2A
harness is committed; **no finding has been produced yet** — the 12-pair
pilot is built and still unlabelled.

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

## In progress

**Phase 2 / 2A — the automation-flag validation harness. Committed and
pushed. Waiting on 12 human labels.**

**No result exists yet.** The harness is the tool, not the finding. Nothing
in this repository currently states an outcome for the MiniLM@0.80 vs
BGE@0.90 promotion question, and nothing should until the pilot is labelled
and scored.

Naming note: the README previously used "Phase 2" for the already-committed
BGE clustering measurement. That work is now retitled **"BGE clustering
re-run"** throughout, and **"Phase 2" means this harness**.

Files that landed:

| File | What it is |
|---|---|
| `src/experiments/build_flag_validation_set.py` | Builds the set. `--pilot` builds the 12-pair probe. |
| `src/experiments/score_flag_validation_set.py` | Scores it. `--pilot` scores the probe. |
| `data/automation_flag_validation_set.json` | 60-pair blind labelling file (unlabelled) |
| `data/automation_flag_validation_key.json` | Withheld answer key |
| `data/automation_flag_validation_pilot.json` | **12-pair pilot — label this one first** |
| `data/automation_flag_validation_pilot_key.json` | Pilot answer key |
| `README.md` (modified) | Phase 2 naming fix + new Pending entry |

Both scripts are offline, deterministic (seed 42, byte-identical across
re-runs), load no model and spend no Gemini quota.

### The structural finding that reshaped 2A

Building the full set surfaced this, and it is the reason the design changed
mid-phase:

**At their own cliff-edges, neither configuration ever merges across dataset
templates.** All 1,069 pairs both configurations merge, and all 411 pairs
they disagree about, are within-template. MiniLM@0.80 merges 1,167 pairs
total; BGE@0.90 merges 1,382; the disagreement is 98 MiniLM-only + 313
BGE-only.

So the entire measured difference between the two is **recall**, not
precision. The originally pre-registered rule — an exact binomial on which
configuration wins more discordant pairs — would therefore have promoted
whichever model merges more, inverting production's stated
precision-over-recall stance. A dry run confirmed it: label everything
`same_fix` and the old rule printed PROMOTE at p = 0.0027, on zero real
evidence about flag correctness.

### The amended decision rule

**Primary** is the false-merge rate on each configuration's *extra* merges
(the pairs it uniquely co-clusters). A `different_fix` label there is a false
merge — the costly error.

    Promote BGE only if it makes ZERO observed false merges AND MiniLM
    makes at least one. If NEITHER makes a false merge the verdict is
    NO PRECISION SIGNAL, not promotion.

The binomial win-rate is still reported but demoted and explicitly labelled
as the recall comparison it is. The scorer also prints rule-of-three upper
bounds, because zero observed is not zero: 0/42 bounds BGE's true rate only
at 7.1%, and 0/18 bounds MiniLM's at 16.7%.

### The pilot

Rather than spend 60 judgements, `--pilot` draws **12 pairs (6 per
direction) from the most divergent end** of the region — divergence 0.53–0.68
against a region median of ~0.42 — where a genuine different-fix pair would
appear if one exists anywhere. It is balanced across directions on purpose,
which would bias a win-rate test, so the scorer refuses to compute the
head-to-head on it. Distinct `PL###` pair ids prevent a pilot file being
scored against the full key.

**If the pilot comes back all `same_fix`**, that is the Phase 2 finding: this
dataset structurally cannot distinguish MiniLM@0.80 from BGE@0.90 on
precision, only on recall, and promotion becomes a product decision about how
many candidates to surface rather than an evidence-backed calibration
result. Inspection of the most divergent pairs suggests this is the likely
outcome — even the max-divergence pair differs only in verbosity and app
name — but that is an impression from a handful, not a measurement.

---

## Immediate next step

**Label the 12-pair pilot**, then score it:

```powershell
# label data/automation_flag_validation_pilot.json -- set each "label" to
# exactly one of: same_fix | different_fix | unclear
python src/experiments/score_flag_validation_set.py --pilot
```

The scorer refuses to run on a partially-labelled file, rejects invalid label
strings, and refuses to score a pilot against the full key — each with an
actionable message rather than a traceback.

Then, depending on the pilot:

- **Any `different_fix`** → a false merge is observable; build and label the
  full 60-pair set (`build_flag_validation_set.py` with no flag).
- **All `same_fix`** → write up the no-precision-signal result as the Phase 2
  finding in the README, and close 2A without spending the other 48
  judgements.

The harness is already committed. The finding lands as its own separate
commit once the pilot resolves, so the tool and the result stay distinct in
the history.

Remaining sequence after Phase 2: multi-agent orchestrator → drift detection
→ Docker/CI packaging.

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
