# Project Status

**Last updated:** 2026-09-23
**Last commit to move code or a result:** `3885393` — Phase 8B, Docker + CI.
**All five gates passed.**

> **COMMITTED BUT NOT PUSHED — and NOT because a gate failed.** All five gates
> passed (see below). `git push origin main` was rejected by GitHub:
>
> ```
> ! [remote rejected] main -> main (refusing to allow a Personal Access Token
>   to create or update workflow `.github/workflows/ci.yml` without `workflow`
>   scope)
> ```
>
> The stored credential cannot create files under `.github/workflows/`. Nothing
> in the repository is wrong and nothing needs re-running. Fix the credential
> and push: add the **`workflow`** scope to the Personal Access Token at
> <https://github.com/settings/tokens> (then `git push origin main`), or switch
> the remote to SSH. Two commits are waiting: `3885393` and `6f776a6`.
>
> **`ci.yml`'s measured runtime is unknown until that push lands**, since the
> workflow cannot run before it exists on GitHub.
Phase 8A.1 (`d3b8f35`) is gated and PUSHED.
Phase 8A (`b243a2f`) is gated and PUSHED.
Phase 6C (`c5497bf`) is also gated and PUSHED.
Phase 7C (`0c9ff8b`) is also gated and PUSHED.
Phase 7B (`f789b8d`) is also gated and PUSHED.
Phase 7A (`ba98843`) is also gated and PUSHED. Phase 6B (`8419db6`) is
also gated and pushed. Phase 6A
(`8f2f5e1`) and Phase 5C (`4793785`, `1200451`, `a8557df`) are gated and
pushed.
**Branch:** `main`, level with `origin/main`, working tree clean.

**Current phase: Phase 8B — COMPLETE and GATED** (`3885393`, 2026-09-23). Docker + CI, the
last item on the agreed roadmap before the paper. Offline: **zero Gemini calls,
zero Ollama calls.** Production frozen; `conformal.enabled` and `drift.enabled`
stay `False`.

**What landed.** A digest-pinned `Dockerfile`
(`python:3.14.3-slim@sha256:5e59aae3…`, the interpreter the published results
were produced under, not the floating `3.14-slim`) that builds its own
artifacts in the documented order and serves `uvicorn src.service.api:app`;
`.dockerignore`; two workflows (`ci.yml` on push/PR, `gates.yml` manual); and
`src/service/verify_deployment.py`, a committed verifier for a *running*
deployment — because a number quoted at a gate comes from a committed script.

**THE FINDING: occurrence #7 of the recurring bug class, in the packaging
rule.** The first image built clean, ran clean and answered correctly, and was
wrong. `.dockerignore` said `*.npy`; those patterns use Go's `filepath.Match`,
where `*` does not cross `/`, so nothing under `data/` was excluded. The
developer's local 12 MB embedding cache was copied in,
`train_embeddings.py` printed **`[cache HIT]`**, and the image's Tier-2
classifier and FAISS index were derived from a file encoded on another machine
in August — byte-identical hash, original mtime. **It was not a model swap, a
routing/eligibility test or a grouping key**, the three shapes the list had
generalised to, and it was caught by *reading a build log*, which is not a
control. It is a control now: a `RUN` guard after `COPY` fails the build if any
pre-built artifact reaches the context, and the patterns are written `**/`. The
rebuild reports `[cache MISS] … Computing embeddings from scratch` and spends
~508 s encoding its own 4,000 rows.

**Container verification (all checks pass).** `config_fingerprint`
**`9c9a5cbcb53f`** served by the container, **`9c9a5cbcb53f`** computed from
`src/agent/config.py` — identical. Index/metadata 4,000 = 4,000. No-key
surface: `/health`, `/agents/classify`, `/agents/retrieve`, `/policy/rag-gate`
and `/triage` all 200 with **no** `GEMINI_API_KEY`; `/agents/resolve` 503, as
documented. adv_08 over HTTP: **tier 2**, Tier-1 confidence
**0.3182984770932253** (*exactly* the golden, delta 0.000e+00), similarity
**0.6123799085617065**, escalated, Infrastructure, no draft.

**SECOND FINDING: an embedding-derived number is not bit-reproducible across
platforms; a TF-IDF one is.** The container's similarity differs from the
Windows goldens' 0.6123800277709961 by a fixed **-1.192e-07**, identical on
every repeated call — float32 BGE + FAISS accumulate in an order set by the
BLAS kernel and SIMD width. Tier-1 (float64) matched exactly. The published
6-dp figure is unchanged and the decision is untouched: the distance to the
0.67 gate is +5.762e-02, **483,352x** the offset. The verifier's first version
asserted full-precision equality and failed; **the goldens were not
regenerated** — the *check* was mis-specified, asserting a guarantee the
pipeline does not make. It now checks Tier-1 exactly and the similarity at a
documented float32 tolerance, printing the delta every run.

**Gates, re-run and not quoted:** `pytest` **422 passed**, 0 failed (68.6 s);
adversarial **9/9**, CSV byte-identical (22 s); goldens **45/45** and **9/9**
(45.9 s); ablation baseline **32/45**, CSV byte-identical (15 s); paper parity
**16/16** (3.8 s). **Isolation: 771 tracked result, golden, paper and model
files hashed before and after — none changed.**

**Not measured, and stated as such:** `gates.yml` is `workflow_dispatch` and
this machine has no `gh` CLI or token, so its CI runtime is unknown; the local
timings above are the proxy. The image is 4.2 GB, ~476 MB of it a duplicated
copy of the weights from `chown -R` in the non-root layer — known, fixable,
left alone because size gates nothing. adv_08 is one ticket, not a full
in-container golden run.

**Next: Phase 9A — the IEEE draft.**

---

**Phase 8A.1 is COMPLETE, GATED and PUSHED** (`d3b8f35`, 2026-09-22). The five numbers 8A
flagged as having no committed machine-readable source now have one, except
the one that cannot. Each script that printed a number got a **writer**, and
each was re-run in its **original configuration** (seed 42, same split, same
settings). Offline: **zero Gemini calls, zero Ollama calls.** Production
frozen; `conformal.enabled` and `drift.enabled` stay `False`.

**Reproduced exactly (4):** the cascade threshold-by-target-accuracy sweep
(1.0001 / 0.50 / 0.50 at the 90/80/70 bars; 14-ticket Tier-1 share
0.0% / 21.4% / 21.4%; 14-ticket accuracy 71.4% throughout), the
self-retrieval contamination rate (**10/175 = 5.7%**, with both existing
calibration CSVs byte-identical afterwards), fine-tuned DistilBERT's
**7/14** (on a from-scratch retrain, at every epoch of both runs), and both
**100.0%** in-distribution figures.

**DID NOT REPRODUCE (1): the TF-IDF baseline's 14-ticket score is 6/14
(42.9%), not 7/14.** Three independent derivations agree — the original
full-4,000-row fit, an 80/20-split fit added as a secondary arm, and the
persisted production Tier-1 artifact (same hyperparameters, different script,
saved months earlier). 7/14 is on the **do-not-cite** list and the documents
are corrected. Likely cause: it was measured on the earlier **1,000-ticket**
corpus and never re-measured after the scale-up — the DistilBERT row in the
same table still carries its "at 1,000 tickets" caveat and the TF-IDF row
never did. **Unconfirmable** (that corpus was never committed), so it is
recorded as a hypothesis, not a cause. The conclusion is unchanged and
slightly strengthened: the gap to frozen embeddings is four tickets, not
three.

**NEW MEASUREMENT: DistilBERT on the 45-ticket benchmark**, never run before.
**It does not carry a single-ticket reading** — the retrain's best epoch
scores **18/45 (40.0%)** and the original checkpoints score **21/45 (46.7%)**.
Two runs of an identical configuration differ by three tickets, so CPU
fine-tuning reproduces exactly on the 14-ticket axis and **not** on this one.
Both are committed; **no ordered comparison** between them — two draws do not
measure a difference.

**Still no source (1):** cascade calibration **attempt 2**, "34 of 35
hand-written tickets in one bucket". That set was never committed and is
absent from every revision in the repository's history, so it cannot be
re-run. Recorded as `status=no_artifact` in the attempts CSV. Nothing was
invented.

**Paper surface:** `paper/NUMBERS.md` goes **161 → 188 numbers** over
**45 → 51 hashed sources**, 42 table CSV frames, no-source list **5 → 1**,
**0 anchored mismatches**. Six new committed result files under `data/`.
Three cleanups landed with it: `train_cascade.py`'s three duplicate copies of
the confidence bin edges collapsed to one constant (values unchanged),
`train_distilbert.py`'s hardcoded `7/14` replaced by a read of the committed
baseline file, and both benchmarks scored by one loop so they cannot drift
apart.

**Next: Phase 8B (Docker + CI).**

**Phase 8A is COMPLETE, GATED and PUSHED** (`b243a2f`, 2026-09-22). Every paper number, table
and figure now regenerates from committed result files through one script,
`src/experiments/build_paper_artifacts.py`. Offline: **zero Gemini calls, zero
Ollama calls, no model load, no training run, no experiment re-run.** It reads
`data/` and writes only `paper/`.

**What it produced:** 16 tables (T1–T16, 39 CSV frames, each also as booktabs
LaTeX), 7 figures (F1–F7, PDF + PNG at 300 dpi + the data CSV + the caption
behind each), `paper/NUMBERS.md` with **161 numbers** (id, value, source file,
regenerating command, tags), `paper/FRAMING.md` (the agreed write-up framing,
numbers interpolated from NUMBERS.md), `paper/RECONCILIATION.md` and
`paper/PROVENANCE.json` (sha256 of all 45 sources). The build is
**byte-for-byte deterministic including PDF and PNG**.

**The rule is structural, not aspirational.** Every value passes through one
`emit()` choke point that refuses a source file that does not exist, and
`tests/test_paper_artifacts.py` runs an **AST lint that fails if any `emit()`
call passes a numeric literal**. A retyped number cannot reach the paper.
Paired comparisons use the exact McNemar test; proportions carry a Wilson 95%
interval except on pre-registered case-study axes, where `emit()` *raises* if
a CI is requested; a difference inside its own noise band is auto-tagged
`within-band`. The builder's statistics are cross-checked at build time
against the three implementations already committed in `src/experiments/`.

**Audit result: 54 anchors, 0 mismatches**, every anchor found in at least one
document — the four project documents agree with their source files on every
load-bearing figure. Two things surfaced: **two conventions for a 95% z**
coexist in the repo (1.96 vs the exact quantile; algebraically identical,
~3.5e-6 apart, no published figure affected — and a CI a source already
carries is read verbatim, never recomputed), and **five documented numbers
have no committed machine-readable source** (TF-IDF 7/14, DistilBERT 7/14,
DistilBERT on the 45, the cascade threshold-by-target-accuracy table, and the
5.7% self-retrieval rate). Nothing was invented or re-run to give them one;
they are flagged for a write-up decision. One write-up correction: the
reliability ECEs are a **ceiling effect** — observed accuracy is 1.0 in every
bin, so both tiers are **under-confident**, never "over-confident".

Nothing promoted; production frozen. **Followed by Phase 8A.1 above.**

**Phase 6C is COMPLETE, GATED and PUSHED** (`c5497bf`, 2026-09-22).
**Verdict: a retrieval-sufficiency check catches both 2B misses and is STILL
NOT USABLE as a second gate.** Primary, as counts over the 33 eligible
tickets: **caught 2 of 2** human-labelled ungrounded drafts, **flagged 26 of
the 31** grounded ones (false-flag proportion 0.839, Wilson [0.674, 0.929]; no
rate is reported on the 2-positive axis, by pre-registration). A gate that
escalates 28 of 33 tickets currently reaching the resolver suppresses ~85% of
auto-resolution to recover two bad drafts, and **nothing can be tuned** — the
rater has no threshold and its verdicts are stable 3/3 at temperature 0.0.
Secondaries: **20 of 21** agreement with the live gate, with **one ticket
(N45, top-sim 0.6397) the gate escalated although its context was adequate** —
the scalar errs in both directions; cross-family **Qwen2.5-3B agrees 39/54
(0.722)** on **54/54 byte-identical prompts** and **misses one positive**, so
this is not a vendor artifact. **6C is NOT an instance of the named finding**,
and the near-duplicate explanation is a **recorded REJECTED hypothesis** (it
contradicts 2B's 71.1% / 88.9% distinct-fix diagnostic). The post-hoc
declining-draft reading is **not supported** (3/3 vs 23/28; n=3 carries no
comparison). **The binding limitation is the label, not the rater**: 2B's
groundedness labels are an outcome proxy for draft support, not context
sufficiency. 58 Gemini calls, cap 70. Nothing promoted; production frozen.
**Next: Phase 8A.**

**Phase 7C is COMPLETE, GATED and PUSHED** (`0c9ff8b`, 2026-09-22).
**Verdict: BLOCKED** (`blocked_auc_ge_0.95`) — the pre-registered degeneracy
rule fired at a TF-IDF-space domain AUC of **0.9972**, so **no verdict is drawn
on the primary**. 7C set out to test Finding 1 under the shift it was actually
measured under; it did not get to answer. **After 7B and 7C, Finding 1 has not
been shown either to hold or to fail outside its original corpus.** Two post-hoc
notes carry forward: the blocking rule may be **mis-specified for this design**
(a question for the gate), and the mechanism **is** visible in accuracy — the
paraphrase shift cost Tier-1 **10.5 points** against Tier-2's **2.5** — but not
in coverage. Phase 7B is COMPLETE, GATED and PUSHED (`f789b8d`, 2026-09-22).
**Verdict: FINDING 1 DOES NOT REPLICATE on a different generator's corpus.**
Under a measured covariate shift (domain AUC 0.8472), TF-IDF and BGE transfer
coverage **equally well** — Tier-1's gap at α=0.10 is **+0.0009** where ours is
**−0.2333**. Eleven of twelve pre-registered readings sit inside a band 2.8×
tighter than the one that measured Finding 1. **Finding 1 is SCOPE-NARROWED,
NOT WITHDRAWN.** The claim becomes *coverage transfer depends on the
representation **under paraphrase shift**, on our corpus; untested elsewhere
until 7C* — 7B applied a **version** shift, and TF-IDF's failure mechanism is
surface-vocabulary change, which a version shift may not produce. The named
cross-phase finding is **sharpened** by this, not damaged. Two further results: the **6A
deferral question resolved for the first time** (3 of 6 comparisons, *all*
favouring the confidence incumbent — conformal deferral is **worse** there, on
that corpus only), and a **RECORD CORRECTION** — 7A's ad hoc domain AUCs do not
reproduce and are replaced. Nothing promoted; `settings.conformal.enabled` and
`settings.drift.enabled` stay `False`. **The 7C gate DECLINED to unblock** — a
pre-registered rule is not revised after seeing results, so the spec error is
recorded instead. **Next: Phase 6C on a fresh-quota day, then 8A. NO FURTHER
EXTERNAL EXPERIMENTS in this programme.**

**Phase 7A is COMPLETE, GATED and PUSHED** (2026-09-22). **Design A chosen as
7B's primary** (version split within the `aa` file, 51+52 → 400, domain AUC
recorded as 0.8584 — **corrected by 7B to 0.8727 raw / 0.8472 operating**);
Design B pre-registered as secondary; **Design C
rejected** — a coverage drop on German would measure encoder competence, not
distribution shift. Verdict: the
external dataset **can carry 7B** — 28,261 English rows (628× the benchmark;
12,500 distinct after de-duplication, 278×), 10/10 queues ≥300, and the Phase
2A clustering-precision question is **answerable** there. Methodological
finding: **our own corpus is MORE near-duplicated than the external one**
(85.20% vs 79.39%). Phase 6B is COMPLETE, GATED and PUSHED (2026-09-21). Verdict: weighted conformal
is a **partial repair, not a correction** — Tier-1's benchmark gap moves
−0.2333 → −0.1222 at α=0.10, a change beyond the ±2 s.d. band but a residual
still **5.4 s.d. below nominal**. **Never write "the shift is correctable by
covariate reweighting."** Nothing promoted; `settings.conformal.enabled` stays
`False`. **Next: Phase 6C (Gemini-spending) or Phase 7 — see "Immediate next
step".**

**Recorded 2026-09-21, after the 6A gate: the corpus's nasscom-brief origin, as
a framing note for Phase 9A.** No experiment, dataset, threshold or result
moved — see "The corpus's origin" below. It is the second 9A carry-forward item.

Phase 5B (the honest ablation) is **complete, gated and pushed**. It changed what a published claim *means*
without moving any measured number: the ablation's "+35.6 points for the
cascade" is baseline minus Tier-1-only, i.e. the BGE-vs-TF-IDF representation
gap, not the value of cascading. Against the control that was missing
(Tier-2-only) the cascade is **one ticket worse on both evaluation sets and
statistically indistinguishable** (exact McNemar p = 1.000 each). What it
actually buys is **8–18% of median per-ticket latency**. Production gates
unchanged; all three published ablation CSVs byte-identical. Phase 5A's
correction and Phase 4B-1's "measured, not shipped" both stand.

**Two doc corrections were made during the 4B-1 session**, both places where
this file contradicted git: it said `5c077dd` was unpushed (it was not) and that
4B had not started (4B-1 was already written in the tree, unrun and
uncommitted). Trust `git log` / `git status` over this file when they disagree.

3A (Tier-1 persistence), 3B (agent boundaries + orchestrator), 3C (HTTP
service + failure boundary) and 3D (the README write-up) are all done, gated
and pushed. Production gates unchanged throughout — the whole phase moved no
number. The one deliberate omission: **the n8n workflow was dropped, not
deferred by accident.** See "Phase 3D" below.

Read this file first. `CLAUDE.md` describes how the project works and rarely
changes; this file describes where it currently is and changes every session.

**Where to look:** "Health at a glance" for the current numbers, "Named finding
for the paper" for the result the write-up is built around, **"The agreed Phase
5–9 programme" for the roadmap**, and "Immediate next step" for what to do
first. Everything between is the record of what has already landed.

---

## Health at a glance

| Check | Command | Current |
|---|---|---|
| Test suite | `pytest` | **422 passed** (406 + 12 from 8A + 4 from 8A.1), 0 failed/skipped, offline |
| **Paper artifacts (8A.1)** | `build_paper_artifacts.py --force` | 16 tables, 7 figures, **188 numbers**, **51 sources** hashed; byte-for-byte deterministic incl. PDF/PNG |
| Paper parity (8A.1) | `pytest tests/test_paper_artifacts.py` | NUMBERS.md, every table CSV and every figure-data CSV byte-identical on rebuild |
| Document reconciliation (8A.1) | same script, `RECONCILIATION.md` | **59 anchors, 0 mismatches**; **1 number with no committed source** (was 5); 2 z-conventions recorded |
| **TF-IDF baseline, benchmark14 (8A.1)** | `generalization_test.py` | **6/14 (42.9%)** — the published 7/14 **does not reproduce**; 3 independent derivations agree |
| DistilBERT, benchmark14 (8A.1) | `train_distilbert.py --backup-existing` | **7/14** — reproduces exactly, every epoch of both runs |
| DistilBERT, benchmark45 (8A.1) | same | **18/45** retrain vs **21/45** original checkpoints — NEW measurement, ±3 tickets between runs |
| Cascade threshold sweep (8A.1) | `train_cascade.py` | **1.0001 / 0.50 / 0.50** at the 90/80/70 bars — reproduces exactly |
| Self-retrieval contamination (8A.1) | `calibrate_rag_similarity_threshold.py` | **10/175 (5.7%)** — reproduces exactly; existing CSVs byte-identical |
| **Sufficiency gate (6C)** | `score_sufficiency_gate.py` | **caught 2/2, flagged 26/31** — false-flag 0.839, Wilson [0.674, 0.929]; **not usable as a gate** |
| Sufficiency vs the live gate (6C) | same script | **20/21** agree; the 1 disagreement (N45, 0.6397) is context the gate escalated **although it was adequate** |
| Sufficiency cross-family (6C) | `run_sufficiency_autorater.py --backend ollama` | Qwen2.5-3B agrees **39/54 (0.722)**, prompts **54/54 byte-identical**, **misses 1 of 2** positives |
| External corpus (7A) | `profile_external_dataset.py` | 28,261 English rows (628× benchmark45); **12,500 distinct** after dedup; near-dup **79.39%** vs **our own 85.20%** |
| **Finding 1 on external data (7B)** | `run_external_conformal_shift.py` | **does NOT replicate** — Tier-1 gap **+0.0009** vs Tier-2 **−0.0072** at α=0.10, band 0.0165, n_cal 1,327 / n_test 10,441 |
| Boundary contamination (7B) | same script | **1.46%** of test tickets have a BGE≥0.95 neighbour in train∪cal; full and no-neighbour readings agree to 0.0011 |
| External label ceiling (7B) | same script | Tier-1 **0.3476**, Tier-2 **0.3732** over 10 queues — labels generator-assigned and **unaudited** |
| **Finding 1 under paraphrase shift (7C)** | `run_paraphrase_shift_conformal.py` | **BLOCKED** — TF-IDF domain AUC **0.9972** ≥ 0.95; manipulation worked (TF-IDF cosine **0.2489**, BGE **0.8462**), n=286 pairs |
| Accuracy under paraphrase shift (7C) | same script | Tier-1 **0.3776 → 0.2727** vs Tier-2 **0.3776 → 0.3531** — the lexical model hit **4× harder** |
| Deferral rules, external (7B) | `compare_deferral_rules_external.py` | gated axis **resolves**: 3 of 6 signals, **all favouring the confidence incumbent**; gate coverage 6.15% = **642 tickets** (6A had 4) |
| Service | `uvicorn src.service.api:app --port 8000` → `GET /health` | fingerprint **`9c9a5cbcb53f`** (was `830c211b1fe9`; changed by adding `settings.drift.rate_reference_name`), index 4000, Tier-1 4000 rows |
| Drift evaluation | `python src/experiments/evaluate_drift_detection.py` | 25/68 operating points eligible; verdict **measured, not shipped** |
| Drift reference | `python src/experiments/build_drift_reference.py` | reproduces all 24 published Phase 1 detection values exactly; 175/175 pipeline-vs-direct similarity match |
| Escalation regression gate | `python src/experiments/test_adversarial_escalation.py` | **9/9 PASS** |
| Golden parity | `pytest tests/test_pipeline_parity.py` | 45/45 and 9/9 exact |
| Ablation baseline (45-ticket) | `run_ablation_study.py --mode baseline` | **71.11%** (32/45) |
| Ablation tier2-only (45-ticket) | `run_ablation_study.py --mode tier2-only` | **73.33%** (33/45) — the cascade is −1 ticket |
| Cascade vs Tier-2 (both sets) | `compare_cascade_vs_tier2.py [--set deployment175]` | exact McNemar **p = 1.000** on each |
| Warm inference latency | `measure_inference_latency.py` | Tier-1 **1.04 ms** vs Tier-2 **156.40 ms** median (151×) |
| Phase 2A pilot | `score_flag_validation_set.py --pilot` | **0/12 false merges**, STOP verdict |
| Phase 2B groundedness | `score_groundedness_set.py` | **31/33 grounded** (93.9%), judge κ = −0.042 |
| Zero-shot Gemini (5C) | `run_zeroshot_baselines.py --backend gemini` | **40/45** and **14/14**, 0 unparseable, median 0.81 s |
| Zero-shot Qwen2.5-3B (5C) | `run_zeroshot_baselines.py --backend ollama --model qwen2.5:3b-instruct` | **34/45** and **12/14**, 0 unparseable, median 6.98 s (CPU) |
| Weighted conformal (6B) | `run_weighted_conformal.py` | Tier-1 gap −0.2333 → **−0.1222** (TF-IDF, α=0.10); residual **5.4 s.d. out**; BGE arm **blocked**, domain AUC **0.9908** — and BGE separates *more* easily yet transfers *better* |
| Deferral rules (6A) | `compare_deferral_rules.py` | **no signal on the gated axis in all 12 comparisons**; 3 of 4 configurations degenerate at the live gate |
| Prompt identity across the two 5C arms | `summarize_zeroshot_baselines.py --prompt-check-against` | **59/59 byte-identical** (sha256), both directions |

Production gates are unchanged and remain the calibrated values: cascade
**0.50**, RAG similarity **0.67**, resolution clustering **0.80**.

---

## Named finding for the paper (Phase 9A picks this up)

**"The calibration/reference distribution, not the test or method, is the
binding constraint."** Approved as a named finding on 2026-09-20. **No new
experiment** — it is a reframing of results already measured and published.

**SCOPE WIDENED BY PHASE 5C (2026-09-21).** The finding was first stated over
the *calibration/reference* distribution alone. 5C reached the same wall from
the *training* distribution, so the mechanism is broader than the original
wording. The heading is kept for continuity and the README carries a scope note
beside it — **Phase 9A must settle the final wording.** The two original
instances are unchanged and are not weakened by the addition.

Its evidence is three phases reaching the same conclusion independently, through
the same register-mismatch mechanism:

1. **Coverage side — Phase 1, Finding 4.** A deployment-distribution
   calibration set matched to the in-domain set's size and class balance
   recovered ~38% of Tier-1's coverage shortfall (−0.233 → −0.144 at α=0.10) and
   left it six times outside the noise band. Distribution matching is necessary
   but not sufficient.
2. **Monitoring side — Phase 4B-1, the realistic-traffic arm.** Deployment-
   register tickets flag against the in-domain reference at 0.217/0.429/0.514/
   0.646 for α = 0.01/0.05/0.10/0.20, versus nulls of 0.006/0.046/0.097/0.199 —
   4–7×. No test, α or window fixes it; the conditional binomial's null is a
   clean 0.009–0.023 at those same operating points.
3. **Accuracy side — Phase 5C, the zero-shot baselines.** A classifier trained
   on 3,200 corpus rows is **indistinguishable from a 3B open-weight model that
   never saw it** (34/45 vs 33/45, exact McNemar p = 1.000) and distinguishably
   worse than zero-shot Gemini (40/45, p = 0.0391) on out-of-template phrasing.
   The training data's advantage does not survive the crossing either.

One is about whether a finite-sample guarantee survives deployment, the second
about whether a monitor can run without false alarms, the third about whether
accuracy bought with training data transfers at all. All three fail for the same
reason and are repaired by the same thing, which is what makes the constraint a
property of the corpus rather than of any method.

**Phase 2A is still NOT an instance.** It is a **related dataset
limitation** — template redundancy, where the templates *are* the fix classes,
so clustering precision is unmeasurable on this corpus. That is training-data
redundancy limiting what can be evaluated, not a calibration/reference register
mismatch. Present it alongside as a second, independent corpus constraint; do
not fold it into the named finding. **Phase 5C, by contrast, IS an instance** —
it is a register mismatch, not a redundancy problem.

Written up in README "Named finding — the calibration/reference distribution,
not the test or method, is the binding constraint".

### Aryan's steer for 9A, recorded 2026-09-21 — DO NOT ACT ON IT YET

Given after the 5C gate cleared, as direction for the write-up phase. **Nothing
was rewritten on the strength of it; the final wording is 9A's call, made with
the full results in view.** Recorded here verbatim in substance so it survives
to the phase that owes the decision:

- Name the general finding as **"the data distribution a component is fitted or
  calibrated on is the binding constraint, not the method"**.
- Put **three scoped instances underneath it**: the *calibration/reference
  distribution* (Phase 1 Finding 4; Phase 4B-1) and the *training distribution*
  (Phase 5C).
- **Phase 2A stays a related dataset limitation, not an instance.**

**Added 2026-09-21 after the 6A gate — a SECOND named finding, methodological.**
6A's AURC result goes in the paper as a **methodological finding**, paired with
the Phase 2A decision-rule inversion. They are **two independent instances of the
same failure: an ungated or averaged metric manufacturing significance where the
pre-registered, gated comparison has no resolution.**

- **Phase 2A — the decision-rule inversion.** A valid exact binomial on which
  clustering configuration won more discordant pairs would have promoted
  whichever model *merged more*, because every pair in the comparison region was
  within-template and the two configurations differed only in recall. The rule
  would have inverted the project's precision-over-recall asymmetry through the
  gate meant to enforce it.
- **Phase 6A — AURC vs the gated axis.** AURC returned significant effects in 3
  of 12 comparisons, in **contradictory directions** across the two evaluation
  sets, and favouring the simpler baseline (margin) rather than conformal — while
  the pre-registered gated metric had **no resolution at all in 3 of 4
  configurations**.

**The verdict wording is fixed and must not drift:** *"no evidence it defers
better on the gated axis"*. **Never** *"conformal is worse"* — the data does not
support the stronger claim, and 6A's own degeneracy is the reason why.

Two things he asked to survive into the paper from 5C specifically:

1. **The 5C headline is NOT "Gemini beats the pipeline."** It is: *a 3B model on
   a laptop CPU with no training on this corpus is not beaten by a classifier
   fitted on 3,200 of its rows, and that holds across two vendors.*
2. **Keep both per-category findings.** Qwen's zero Database predictions (0/5
   and 0/2 recall, misroutes into Application), and the **shared 50%
   Infrastructure failure across both LLMs and the trained classifier** — the
   second being evidence that the hard cases are properties of the *tickets*.
   **Cross-reference that to the Infrastructure benchmark-scope bug** (the
   generation prompt that left "Infrastructure" unanchored and produced 5
   mislabeled tickets, README "Benchmark scope anchor").

### The corpus's origin — the nasscom brief (recorded 2026-09-21, for 9A)

**Framing note only. No experiment, dataset, threshold or result was touched
when this was recorded.** It exists because the paper's experimental-setup
section needs the corpus's provenance stated, and because the provenance
changes how the corpus's limitations should be read.

**The origin.** This project comes from a **nasscom hackathon use-case brief**.
The brief itself **explicitly recommended a synthetic, LLM-generated dataset**
— ~1,000 tickets carrying `title`, `description`, `category`, `resolution` and
`priority` — and specified the **7 categories**, along with classification,
routing, resolution suggestion, confidence-based escalation and automation
flagging. Our dataset follows that brief **at 4,000 rows** rather than 1,000.

Verified against the data on 2026-09-21, by two independent derivations:
`data/synthetic_tickets.csv` is **4,000 rows** (4,001 lines including the
header; `pandas` agrees at 4,000) across **7 categories** — Access Management,
Application, Database, Infrastructure, Network, Security, Storage — and its
columns are the brief's five fields plus our own `id` and `scenario_id`.

**What 9A must do with it.** The experimental-setup section should state the
brief's origin **as the reason for the synthetic corpus** — the synthetic data
is a followed specification, not a convenience substitute for real tickets —
and should then **immediately** state that we treat the corpus as an **object
of study**. Three measurements are readings of what the corpus is worth, and
they belong next to that sentence:

1. **Phase 2A — the templates *are* the fix classes.** Clustering precision is
   unmeasurable on this corpus, because no configuration makes a cross-template
   merge anywhere. A negative result about the corpus, not about the method.
2. **Finding 2 — template-level memorisation defeats de-contamination.** 66
   templates of ~62 rows each; the 175-ticket calibration set touches 62 of
   them, so removing source rows changes nothing and removing whole templates
   would leave 210/4000. (Figures as corrected in Phase 5A.)
3. **Phase 5C — a 3B open-weight model that never saw the corpus is
   indistinguishable from a classifier fitted on 3,200 of its rows** (34/45 vs
   33/45, exact McNemar p = 1.000) on out-of-template phrasing.

**The ordering is the argument, and it is load-bearing:** origin first, then
the corpus as object of study. Stated that way, the corpus's limits read as
*findings this project measured and published* rather than as a weakness a
reviewer has to discover. This is the same move as the named finding — the data
distribution a component is fitted or calibrated on is the binding constraint —
and 5C is an instance of both, so the two sections must not contradict each
other on wording. **9A settles the final wording; nothing is rewritten now.**

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
  entire pre-BGE pipeline. 68.89% → 71.11%; the baseline-minus-Tier-1-only gap
  33.3 → 35.6 points. **That gap is not "the cascade gain" and must never be
  quoted as one** — it is the BGE-vs-TF-IDF representation gap, because the
  no-cascade arm is TF-IDF answering everything. Phase 5B added the missing
  Tier-2-only control and the cascade turns out to be **−1 ticket against
  Tier-2 alone on both evaluation sets, at exact McNemar p = 1.000**; what it
  buys is 8–18% of per-ticket latency, not accuracy. The arithmetic above is
  unchanged — only the claim it was attached to.

### Phase 1 — conformal prediction (`148ad8b`, `bcb21c8`)

Split conformal over both cascade tiers plus conformal novelty detection for
the RAG gate. **Measurement only** — `settings.conformal.enabled` is `False`,
production gates are untouched, golden parity holds.

Four findings:

1. **Coverage transfer is a property of the representation.** Same calibration
   set, same α, same benchmark: TF-IDF loses 23.3 coverage points, BGE loses
   1.1, against a 4.5-point noise band.
2. **The in-domain calibration set cannot be de-contaminated.** Memorisation is
   template-level — **66 templates, ~62 rows each, the set touches 62**. Row
   removal changes nothing; template removal would leave **210 of 4,000 rows**.
   (Diagnostic corrected in Phase 5A; it previously read 12 / ~430 / 11 / 40
   because the grouping key omitted `category`. Conclusion unchanged, coverage
   numbers unaffected.)
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

## Phase 5A — the conformal template-grouping correction (`cdac8f6`, gated and pushed)

Planned and approved 2026-09-20 after the error was found while auditing the
project's own docs. It **changed a published number**, which is why it was given
its own sub-phase and gate rather than riding along inside 4B.

**The bug.** `calibrate_conformal.py`'s Step 1b grouped rows by `scenario_id`
alone, but `scenario_id` is an index *within* a category
(`data/generate_dataset.py:634` says so in its own comment), so it merged all
seven categories' templates. Corrected to `(category, scenario_id)`:

| Quantity | Published (wrong) | Corrected |
|---|---:|---:|
| Scenario templates | 12 | **66** |
| Templates touched by the 175 tickets | 11 (91.7%) | **62 (93.9%)** |
| Median rows per template | 430 | **62** |
| Rows surviving template-level exclusion | 40 / 4000 | **210 / 4000** |
| Evidence removed per deleted row | ~0.2% | **~1.6%** |

**Finding 2's conclusion is unchanged** — 210 of 4,000 is still destruction, not
de-contamination — and **no coverage number moved**, because that measurement
excludes by row **id**, never by template. Proven, not argued: re-running the
script produced result CSVs **byte-identical** to the published ones (so they
are not committed as duplicates; `--out-suffix` reproduces them), and the
corrected artifact differs in exactly two keys —
`contamination_structure` and the recorded `config_fingerprint`
(`05f391baf27c` → `9c9a5cbcb53f`, moved by Phase 4's `settings.drift`
additions, which cannot affect conformal). The 90 KB `fits` block is identical.

**Verified by independent re-derivation before any code changed** (rule 6): 66
four ways, 62 and 210 two ways each, and grouping by `scenario_id` alone
reproduces the published 12/11/40/430 exactly — which is what confirmed the
diagnosis rather than inferring it.

**Two hardening changes came with it.** The script now **refuses to overwrite**
its three outputs without `--force` and takes `--out-suffix`, so results get new
filenames (rule 4) — this mattered more than expected, because
`build_drift_reference.py` reads the *published* `conformal_novelty_results.csv`
as its reference check, and an in-place overwrite would have turned that check
into a self-comparison. And `tests/test_contamination_structure.py` pins both the
corrected counts and the fact that a `scenario_id`-only grouping disagrees with
them, so a revert fails the build.

**Logged as the sixth instance of the recurring bug class** and the second to
reach published results. The stale-artifact count in CLAUDE.md stays at four
(this was a grouping key, not a stale artifact) with a pointer that the wider
class now stands at six.

---

## Phase 5B — the honest ablation (gated and pushed in `2a744ed`, `bf94e63`, `be1f7c0`)

Planned and approved 2026-09-20. **Measurement only**: no production threshold,
artifact, benchmark or golden changed, and all three published ablation CSVs
regenerate byte-identical.

### The prerequisite check, run and reported before any code changed

Were the published **33/45** (BGE alone) and **32/45** (cascade) even the same
classifier, or was 33/45 an in-memory 80/20-split model from
`train_embeddings_comparison.py`? **The same classifier, bit-identical** —
derived four ways: the production joblib scores 33/45; a refit under the
comparison script's own recipe from its cached `embeddings_bge.npy` also scores
33/45; the two agree 45/45 ticket-by-ticket with `max abs coef difference =
0.0` and identical `classes_`; and a recomputed cascade @0.50 matches the
published `ablation_baseline_results.csv` 45/45 on `predicted`. **No seventh
bug.** The comparison was sound; the *claim attached to it* was not.

### What was actually wrong

"The cascade is worth +35.6 points" is `baseline (32/45)` minus `no-cascade
(16/45)`, and no-cascade is **TF-IDF answering every ticket**. It measures the
BGE-vs-TF-IDF representation gap. The control that isolates cascading — the
strong model answering everything — had never been run.

### Result

| Set | Cascade | Tier-2 only | No cascade | Δ | Exact McNemar |
|---|---:|---:|---:|---:|---:|
| 45-ticket benchmark | 32/45 (71.11%) | **33/45 (73.33%)** | 16/45 (35.56%) | −1 | b=0, c=1, **p = 1.000** |
| 175-ticket deployment | 131/175 (74.86%) | **132/175 (75.43%)** | 91/175 (52.00%) | −1 | b=1, c=2, **p = 1.000** |

One ticket worse on both sets, and **indistinguishable from zero** on both. The
honest reading is *no detectable accuracy difference*, not *worse*.

Where it acts: Tier-1 keeps **4/45 (8.9%)** of benchmark tickets and gets **2 of
those 4** right (Tier-2 would have got 3); on the deployment set it keeps
**33/175 (18.9%)** and gets 27 right against Tier-2's 28. Tier-1's confident
errors are not scattered — three of the four discordant tickets across both sets
are Tier-1 answering **Database** on a Network or Security ticket.

Latency, measured warm (200 timed single-ticket runs per tier, batch size 1,
after 20 discarded warmups): **Tier-1 1.04 ms median / 1.74 ms p95** against
**Tier-2 156.40 ms / 227.05 ms** — Tier-2 costs **151×**. The cascade always
runs Tier-1 *and then* Tier-2 when unconfident, so its expected cost is
`tier1 + (1 − share) × tier2`: **143.54 ms vs 156.40 ms (−8.2%)** on the
benchmark, **127.95 ms vs 156.40 ms (−18.2%)** on the deployment set.

**So the cascade is a latency optimisation costing a statistically undetectable
amount of accuracy — 8–18% less compute per ticket.** That is a real engineering
result and it is not this project's contribution. The contribution is the
**calibrated escalation gates**, and the paper's accuracy story should rest on
them. This is the fifth time a control changed the reading of a result here (the
ablation's own BGE correction, 2A's pre-registered rule, 2B's judge, 4B-1's two
unusable tests, now this) — **no uncontrolled comparison in this project has
survived being controlled.**

### Limitations, written beside the numbers

- At n=45 with **one** discordant pair the McNemar has essentially no power: it
  could not detect an effect of this size if one existed. That cuts both ways,
  and is exactly why the raw one-ticket gap was never evidence. The 175 set adds
  resolution but reaches only 3 discordant pairs.
- The 175-ticket set is **Gemini-generated deployment-register text, not
  production traffic**, and those same tickets already carry Phase 1 Finding 4's
  conformal calibration — another use of an already multiply-used set.
- Latency is one machine, one process, batch size 1, CPU only (13th Gen Intel
  Core i5-1334U, 12 logical CPUs, Python 3.14.3, Windows 11), no competing load
  controlled. It bounds per-ticket inference cost on this hardware; it is not
  throughput or served latency. The Tier-1 share is a property of these two
  sets, not a deployment rate.

### One latent bug closed

`run_ablation_study.py` was the **last holdout from rule 7**: four loaders of
its own, one of which **refitted Tier-1 from `synthetic_tickets.csv` on every
run**. Migrated to `artifacts.load_artifacts()`, which also gives it the three
hard guards it never had. Verified parity-preserving *before* landing
(`max |Δ tier1_conf| = 0.0` across the 45, both rounding identically at 6
decimals) and byte-identical *after*.

### Gate — re-run, not quoted

`pytest` **169 passed** (153 + 16 new); adversarial **9/9** with
`data/adversarial_escalation_results.csv` byte-identical; goldens **45/45 and
9/9** exact; ablation baseline **32/45 = 71.11%** and no-cascade **16/45**, both
CSVs byte-identical; no `.npy`, `.jsonl` or `logs/` anywhere afterwards.

New files: `src/experiments/compare_cascade_vs_tier2.py`,
`src/experiments/measure_inference_latency.py`, `tests/test_ablation_modes.py`,
and seven new result CSVs (three `*_deployment175`, `ablation_tier2-only_*`, two
`cascade_vs_tier2_mcnemar_*`, one `inference_latency_*`).

---

## Phase 4 — drift detection (4A DONE + PUSHED; 4B-1 DONE + PUSHED)

### Phase 4B-1 — the evaluation (gated and pushed in `8869f0c`)

Planned and approved 2026-09-20 as **audit → run → verify → gate** of the 4B-1
code that already existed uncommitted in the tree. No fresh implementation.

**Audit: no correctness bugs.** Rules 2/4/5/7 clean (no local encoder, no direct
index read, no local Tier-1 refit, zero Gemini quota, no production threshold
touched, `settings.drift` still free of any window/alarm field). **Escalation is
read from `decision.escalated` at every layer** and no status enum appears in the
drift code, so the fifth bug instance's shape is structurally absent. Every
load-bearing count has a second derivation and all agree: Signal B's 39
escalations = an independent below-0.67 count; sink-recorded similarities
reproduce **all 24** published Phase 1 detection values; benchmark 15/45 +
adversarial 6/9 = 21 of 54 reproduces the published Phase 2B split. The 175
in-domain scores are all distinct, which makes 4A's "the split check could not
fail" literally true. Two limitation lines were added (the realistic-traffic
arm's bootstrap dependence; single-RNG step-order determinism) — the only
changes made to the script.

**Null first, as pre-registered. The headline is that two of five tests are
unusable and a third is window-limited:**

- **Signal A conditional binomial: eligible in all 16 cells** (measured
  false-alarm 0.0092–0.0226).
- **Signal A marginal binomial: eligible only at W=25 (α ≤ 0.10) and W=50
  (α ≤ 0.05)**, reaching 0.077–0.125 at W=200 — 2–2.5× nominal. 4A's finding 2
  measured: a long window is what resolves the fixed reference's Beta offset.
- **Signal A KS: never eligible** (0.077 at W=25 → 0.33 at W=200); it reads the
  p-values' 1/176 discreteness, not drift. Descriptive only, never an alarm.
- **Signal B one-sample binomial: never eligible** (0.047 → 0.169). **Fisher
  rescues it** — eligible to W=25 (escalation) and W=100 (Tier-1 share) — because
  it carries the reference rate's own estimation noise.
- **Signal B category χ²: never eligible** (0.12 → 0.46); 7 categories against
  n=175 violates the expected-count conditions. Descriptive only.
- **25 of 68 operating points eligible.** F2 reproduces on 100,000 draws
  (marginal 0.0966 vs 0.0965; bound 0.1259 vs realised q90 0.1257; 0.099 above
  bound vs δ=0.10) and three Signal A implementations agree to <1e-9.

**Power (eligible points only, PROVISIONAL).** Conditional binomial at α=0.05,
W=100: 0.921 at 10% OOD, 0.980 at 10% adversarial, 0.998 at 25% benchmark. At
α=0.01, W=100: 0.898 at just 5% OOD. **Signal B is effectively powerless here** —
`escalation_fisher` needs 50% OOD to reach 0.913 and `tier1_fisher` never
reaches the 80% target.

**Two findings that decide the verdict:**

1. **The best-powered point rests on one reference ticket.** At α=0.01, l=1, so
   the threshold *is* the smallest calibration similarity; one leave-one-out drop
   moves adversarial detection 0.667 → 0.889 and OOD 0.844 → 0.911. At α ≥ 0.05
   (l ≥ 8) everything is stable. **So the candidate is α=0.05 at W=100–200**, not
   the strongest cell.
2. **Realistic traffic reads as drift.** Deployment-register tickets flag at
   0.217 / 0.429 / 0.514 / 0.646 for α = 0.01 / 0.05 / 0.10 / 0.20 against nulls
   of 0.006 / 0.046 / 0.097 / 0.199 — 4–7×. A monitor on this reference would
   alarm continuously on legitimate traffic. Same wall as Phase 1 Finding 4,
   reached from the monitoring side: the reference's register is the binding
   constraint, not the test.

**Verdict: measured, not shipped.** `settings.drift.enabled` stays `False`, no
window size or alarm threshold entered config, and the eligible point is recorded
as a measurement. Gate criterion 1 held — re-run, not quoted: pytest 150,
adversarial 9/9 with its CSV byte-identical, goldens 45/45 and 9/9, ablation
baseline 32/45 = 71.11% with its CSV byte-identical, and no `logs/`, `.jsonl` or
`.npy` after the full run. Limitations are written beside the numbers in the
README entry (provisional power; Signal B's 22.3% is a Gemini benchmark-register
rate, not production; parametric Signal B nulls; abrupt out-of-template
contamination; 45-ticket selection risk; optimistic null on template data).

### Phase 4A — history sink + detector library (`5c077dd`, gated and pushed)

**Gate criterion 1 held — nothing that existed moved.** Re-run after the
change, not quoted: pytest 142/142 (golden parity 45/45 and 9/9 exact inside
it), adversarial gate 9/9 with `data/adversarial_escalation_results.csv`
byte-identical, ablation baseline 32/45 = 71.11% with
`data/ablation_baseline_results.csv` byte-identical. No `logs/`, `.jsonl` or
`.npy` appeared anywhere after the full suite, the gate and the ablation run.

What landed:

- **`settings.drift`** — `enabled=False`, `alpha=0.10`,
  `conditional_delta=0.10`, the reference filename. **No window size and no
  alarm threshold**, and `test_config.py` pins their absence.
- **Opt-in JSONL sink** — `configure_logging(decision_log_path=...)`, receives
  `pipeline_decision` records only, carries `schema_version: 1`. Off by
  default, and **nothing in the project enables it**. It refuses a logger
  level above INFO, before changing anything, because a starved sink records
  an empty history that reads as a quiet period.
- **`src/agent/drift.py`** — pure (no I/O, no models). Signal A: conformal
  p-values via `conformal_p_values()` verbatim, with a marginal *and* a
  calibration-conditional binomial, plus KS. Signal B: escalation rate, Tier-1
  share, category mix. Signal C: fingerprint counts, in their own field. The
  report carries statistics only, no alarms.
- **`load_drift_reference()`** in `artifacts.py` — refuses a reference built
  against a different embedding model, dimension, index size or FAISS file
  hash. Deliberately not a full-fingerprint check, which would fire on every
  unrelated config edit.
- **`src/experiments/build_drift_reference.py`** →
  `data/drift_reference_bge-base-en-v1-5.json`. Refuses to overwrite without
  `--force`.
- `scipy==1.18.0` pinned (was only transitive). Test suite 104 → 142.

**Four findings from building it, all written down with the result:**

1. **The planned reference did not exist as data.** The plan named the scores
   in `conformal_calibration_bge-base-en-v1-5.json`; that file holds
   classification fits only, and `calibrate_conformal.py` discards the 175
   similarities. Doc-ahead-of-code, again. Built as a separate artifact rather
   than by re-running the Phase 1 script over its published outputs.
2. **"False-alarm rate is α by construction" was overstated in the plan.** It
   holds marginally over calibration draws, not for our one fixed n=175
   reference, where the per-ticket rate is Beta(17, 159)-distributed. The
   exact marginal rate is 17/176 = 0.0966; the conditional upper bound at
   δ=0.10 is **0.126**. At a ~200-ticket window that gap is as large as the
   window's own sampling noise. Both tests ship; 4B measures both null rates.
3. **The first independent-derivation check could not fail.** It compared the
   recomputed in-domain false-escalation rate to the published one, but a set
   scored against itself gives a rate fixed by n and ties alone, so any 175
   distinct numbers pass. It was caught because self-inclusive and non-self
   scores "reproduced" identically. Replaced with the published OOD variant
   rate, OOD seed rate and adversarial flag count, which depend on the scores'
   positions and do differ between the two score sets at α=0.01 (8 vs 6
   adversarial flagged). **Caveat:** at α ≥ 0.05 most of those columns
   saturate (1.0, 9/9), so the α=0.01 rows carry most of the check's power.
4. **The reference escalates 0/175 tickets.** In-domain paraphrases never
   reach the RAG gate, so a binomial against rate 0 returns p = 0 on the first
   escalation. The detector reports that test as `degenerate_reference` with a
   `None` p-value. Consequence for the thesis: this reference says nothing
   about a deployed escalation rate. Reference Tier-1 share is 23/175 = 13.1%;
   the predicted category mix is uneven (Database 43, Infrastructure 11)
   although the true labels are 25 each.

### The original Phase 4 plan (approved; corrections from 4A noted inline)

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
  than a ticket. **[Corrected in 4A: marginally only. For the one fixed
  reference the conditional bound is 0.126 at α=0.10; see finding 2.]**
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
drift; that would define away the failure mode that matters most. **[Corrected
in 4A: those scores were never stored. The reference is the same 175 tickets,
rebuilt into `data/drift_reference_bge-base-en-v1-5.json`; see finding 1.]**

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

## Phase 5C — zero-shot LLM baselines (COMPLETE, gated and pushed in `4793785`)

Both arms are done, the gate was re-run clean, and the phase is pushed.

### The result

| Model | benchmark45 | 95% Wilson | benchmark14 | 95% Wilson | Unparseable |
|---|---:|---|---:|---|---:|
| Zero-shot Gemini (`gemini-flash-lite-latest`) | **40/45 (88.89%)** | [76.50%, 95.16%] | **14/14 (100%)** | [78.47%, 100%] | 0/59 |
| Zero-shot Qwen2.5-3B-Instruct (local, CPU) | **34/45 (75.56%)** | [61.33%, 85.76%] | **12/14 (85.71%)** | [60.06%, 95.99%] | 0/59 |
| Trained Tier-2 (BGE + LogReg) | 33/45 (73.33%) | — | 10/14 (71.43%) | — | n/a |

Paired exact McNemar, all six comparisons:

| Comparison | Set | b / c | Δ | p |
|---|---|---|---:|---|
| Gemini vs Tier-2 | benchmark45 | 8 / 1 | +7 | **0.0391 — distinguishable** |
| Gemini vs Tier-2 | benchmark14 | 4 / 0 | +4 | 0.125 |
| Qwen-3B vs Tier-2 | benchmark45 | 7 / 6 | +1 | 1.000 |
| Qwen-3B vs Tier-2 | benchmark14 | 4 / 2 | +2 | 0.688 |
| Qwen-3B vs Gemini | benchmark45 | 1 / 7 | −6 | 0.0703 |
| Qwen-3B vs Gemini | benchmark14 | 0 / 2 | −2 | 0.500 |

**Read each pair separately.** Gemini is distinguishably better than the trained
classifier on the 45. **Qwen-3B is indistinguishable from it** — that is not
"better". Qwen is 6 below Gemini at p = 0.070, which is not significant.

**Qwen-3B never emits "Database" once in 59 tickets** (0/5 and 0/2 recall); the
misroutes land in Application, whose precision drops to 47%. That one category
is 3 of the 6 tickets separating it from Gemini. **Both LLMs fail Infrastructure
at exactly 50%** — a failure shared by both vendors and the trained classifier
is a property of the tickets, not of any model.

Latency: Tier-2 **0.156 s**, Gemini **0.81 s** (5.2×), Qwen-3B CPU **6.98 s**
(44.7×).

### THE FRAMING NOTE — agreed, and load-bearing

**5C measures what a 3,200-row template-generated corpus is worth on
out-of-template phrasing. It is NOT "LLMs beat the pipeline"** and must never
be written up that way. Zero-shot LLM vs *trained* classifier is not
like-for-like, in exactly the way 5B found baseline-vs-Tier-1-only was not.

**5C is the third independent view of the named finding**, and the first from
the *training* side rather than the calibration/reference side. This widened the
finding's scope — see "Named finding" above; **Phase 9A must settle the final
wording.**

### Limitations, recorded beside the result

- **3B is a floor for the non-Gemini family, not a fair ceiling.** It was chosen
  to fit available RAM. Its 6-ticket gap to Gemini **cannot be attributed**:
  this run cannot separate *"model too small"* from *"Gemini authored the
  evaluation data"*. Do not write the gap as evidence of either.
- **The 45-ticket benchmark is Gemini-generated** (human-label-reviewed), so the
  Gemini arm is scored on text from its own family. **The Qwen arm is the
  partial control** — the "zero-shot ≥ trained" direction survives removing the
  authorship advantage; Gemini's extra margin does not.
- Both sets are small; the Wilson intervals overlap heavily and n=14 resolves
  almost nothing.
- Latency is **not** hardware-matched: Qwen CPU-only on an i5-1334U vs Gemini on
  hosted accelerators. Both pinned temperature 0 and JSON-constrained decoding;
  Ollama additionally fixed seed 42 and capped output at 48 tokens.
- **Qwen2.5-3B is under the *Qwen Research* licence**, not Apache-2.0 (the 7B
  is). Check that before calling it "freely available" in the paper.

### What was built

- `run_zeroshot_baselines.py`: RAM preflight (`MODEL_RAM_FLOOR_MB`, aborts
  rather than swap; `--allow-low-ram` stamps every response), **exact** Ollama
  model matching (the old prefix match passed a guard that claimed to verify the
  model was installed), and `model_digest` recorded per response.
- `compare_zeroshot_vs_tier2.py`: `--against zeroshot` for model-vs-model,
  reusing `pair_by_index` and `mcnemar_from_pairs` — no second McNemar.
- `summarize_zeroshot_baselines.py` (new): Wilson CIs, dual unparseable
  accounting, per-category confusion, live-rows-only latency, and the
  prompt-identity proof.
- `tests/test_zeroshot_summary.py` (new): 32 tests, including every Wilson
  interval checked against `scipy`'s `proportion_ci(method="wilson")`.

### Verifications (rule 6 — second independent derivations)

- **Prompt identity: 59/59 byte-identical** by sha256, checked in both
  directions. `build_prompt()` being shared is the guarantee; this is the proof.
- **Call accounting:** 59 cached Ollama response files vs 42 + 14 live + 3
  dry-run = 59.
- **Both committed Gemini comparison CSVs regenerated byte-identical** after the
  compare-script edit, verified before any local-model time was spent. Part 1
  wrote `tier2` in the `direction` column but `tier2only` in the field names, so
  deriving one from the other would have silently rewritten both.
- **Confusion diagonal** is recomputed against the headline count and is fatal
  on disagreement.

### Quota

Part 1 spent **59 Gemini calls** (2026-09-21). **Part 2 spent zero** — Ollama is
local and part 1's responses were re-read from cache. 5C must not share a day
with 6C.

### 5C gate — RE-RUN, not quoted

| Check | Result |
|---|---|
| `pytest` | **246 passed**, 0 failed, ~165s |
| Adversarial escalation | **9/9 PASS**, CSV byte-identical |
| Golden parity | **45/45 and 9/9** exact (4 tests) |
| Ablation baseline | **32/45 = 71.11%**, 9/9 escalations |
| Published result CSVs | none modified (`git status` on `data/` clean) |
| Stray `.npy` | none |

---


## Phase 6A — conformal deferral vs a confidence threshold (gated and pushed in `8f2f5e1`)

**Gate cleared and pushed.** Measurement only; `settings.conformal.enabled` stays
`False` and nothing was promoted.

### The verdict

**Do not promote conformal to the live deferral gate.** The honest form is
**"no evidence it defers better on the axis this project gates on"**, NOT
"conformal is worse".

### Why the rules are comparable at all

Each deferral rule reduces to a scalar ranking: confidence ranks by `p1`,
**LAC-conformal ranks by `p2`**, APS by `p1+p2`. So LAC-conformal deferral ranks
by the *second*-largest probability where confidence ranks by the largest —
genuinely different orderings. Two consequences: the comparison is
**calibration-free** (q cancels, so Phase 1's contamination does not reach it),
and **"defer unless singleton" is not monotone in alpha** because an *empty* set
is also a deferral. `tests/test_deferral_rules.py` brute-forces real sets through
`conformal.predict_sets()` over a grid of quantiles to prove the reduction,
rather than trusting the algebra.

### Result 1 — no signal on the gated axis, and 3 of 4 cases cannot resolve it

Live 0.50 gate operating coverage, **measured**: Tier-1 answers **8.9% (4/45)**
of the benchmark and **18.9% (34/175)** of the deployment set.

All 12 comparisons return "no signal on the gated axis" — no bootstrap interval
excludes zero. More importantly:

| Tier / set | Risk at live gate, ALL four rules | Measurable? |
|---|---|---|
| tier1 / benchmark45 | 0.500 (2 errors in 4 accepted) | **No — degenerate** |
| tier2 / benchmark45 | 0.000 | **No — degenerate** |
| tier2 / deployment175 | 0.000 | **No — degenerate** |
| tier1 / deployment175 | 0.147–0.206 (n=34) | Yes |

**In 3 of 4 configurations every rule accepts the same tickets at that coverage**,
so the test has no resolution. The script detects and records this, the same way
4B-1 reports its degenerate escalation-rate test as `None` rather than p = 0.

### Result 2 — AURC manufactures findings the gated axis does not support

Significant in **3 of 12**, contradicting each other: margin is *better* on
tier2/benchmark45 (−0.0207, CI [−0.0477, −0.0006]) and *worse* on
tier1/deployment175 (+0.0121, CI [0.0003, 0.0249]); LAC is *worse* on
tier1/deployment175 (+0.0556, CI [0.0242, 0.0899]). **The cost-asymmetry failure
caught in the act** — an ungated average produces three publishable-looking
effects where the gated axis shows nothing, and they favour *margin*, not
conformal.

### Result 3 — where conformal can actually be operated

Singleton rate (the fraction auto-routed), from Phase 1's published calibration:
Tier-2 LAC runs 4.0% / 18.3% / 34.3% / 78.9% at alpha = 0.01 / 0.05 / 0.10 /
0.20; APS is far more conservative at every alpha (0.0% / 4.6% / 8.6% / 16.6%).
**alpha=0.01 auto-routes essentially nothing**, so the tightest guarantee is
operationally useless here. **APS at alpha=0.20 returns 13.7% EMPTY sets** on the
deployment set — the predicted non-monotonicity, measured.

### Limitations, recorded beside the result

- **The primary metric is degenerate in 3 of 4 configurations.** At 8.9%
  coverage on n=45 the gate accepts **4 tickets**. That is a statement about the
  evaluation sets and the gate's very low operating coverage, not about the rules.
- n=45 resolves almost nothing (one ticket = 2.2 points); bootstrap intervals
  there are wide.
- **deployment175 is now multiply used** — Phase 1 Finding 4, Phase 5B, and now
  6A — which weakens it further as independent evidence.
- The operating-point overlay uses Phase 1's calibration, fitted on the
  contaminated in-domain 175. The fingerprint is checked; the contamination is not
  repaired.
- **The secondary coverage grid was extended downward after the first run**, once
  the live gate measured 8.9%/18.9% and the original (50/70/80/90%) grid turned
  out not to span the operating point. The **primary** metric — risk at the
  measured operating coverage — was pre-registered and unchanged, and is what the
  verdict rule reads.

### What would actually answer the question

A held-out set large enough that the live gate's ~10–20% operating coverage
contains more than a handful of tickets. At n=45 that region is 4 tickets. This
is the same constraint the named finding describes, arriving a fourth time.

### Gate — RE-RUN, not quoted

| Check | Result |
|---|---|
| `pytest` | **261 passed**, 0 failed, ~80s (246 + 15 from 6A) |
| Adversarial escalation | **9/9 PASS**, CSV byte-identical |
| Golden parity | **45/45 and 9/9** exact |
| Ablation baseline | **32/45 = 71.11%**, 9/9 escalations |
| Published result CSVs | none modified |
| Stray `.npy` | none |

---

## Phase 6B — weighted conformal under shift (COMPLETE, gated and pushed in `8419db6`)

**Measurement only.** `settings.conformal.enabled` and `settings.drift.enabled`
stay `False`; cascade 0.50, RAG 0.67, clustering 0.80 untouched. Offline, zero
Gemini calls. Nothing was promoted.

### The verdict, in the wording that must not drift

**Weighted conformal is a PARTIAL REPAIR, not a correction.** Reweighting moves
Tier-1's benchmark-45 coverage gap from **−0.2333 to −0.1222** at α = 0.10
(TF-IDF space), a change of **+0.1111** against a ±2 s.d. band of 0.0454 — but
the residual is still **5.4 s.d. below nominal**.

**Never write "the shift is correctable by covariate reweighting."** The change
cleared the band; the residual did not. Both readings are reported.

### Why both readings are reported — a pre-registration ambiguity, disclosed

The pre-registered interpretation clause read *"weighting closes Tier-1's gap
beyond the noise band"*. That admits two readings which **disagree on this
data**:

- **(a) the CHANGE exceeds the band** → 3/3 resolving configurations.
- **(b) the RESIDUAL falls inside the band** → 0/3.

Reading (a) alone would have licensed "correctable". Both are therefore in the
CSV and in the verdict, and the ambiguity is disclosed rather than resolved in
whichever direction flatters the result. Picking one after seeing the numbers is
exactly what the pre-registration exists to prevent.

`recovery_fraction` (47.6%) is **post-hoc and descriptive**, flagged as such in
the CSV and the README. It describes; it does not decide.

### Result 1 — the primary axis

| Space | Cross-fitted AUC | n_eff | Gap at α=0.10 | Change | Residual |
|---|---|---|---|---|---|
| (unweighted) | — | 175 | **−0.2333** | — | 10.3 s.d. out |
| BGE | 0.9908 | 145.0–153.9 | −0.1667 | +0.0667 | **blocked** |
| TF-IDF | 0.9295 | 124.4–129.1 | **−0.1222** | **+0.1111** | **5.4 s.d. out** |

All three clip variants (none/p95/p99) give the same gap per space; clipping
barely matters because `n_eff` never collapsed.

**This sharpens the named finding.** Distribution matching recovered ~38%
(Finding 4) and reweighting recovers 47.6% — two independent repair strategies,
different mechanisms, both partial, both leaving coverage 5–6 s.d. outside
nominal. If neither matching the calibration distribution nor reweighting it
closes the gap, the constraint is a property of the corpus, not the method.

> **GUARD — do NOT write that reweighting recovers more than distribution
> matching.** Both percentages are **post-hoc**, and the quantity that matters
> is the **residual gap: −0.122 vs −0.144**, differing by **0.022 — inside the
> ±2 s.d. band of 0.045**. The two are **statistically indistinguishable**. The
> permitted claim is that both are partial and both leave the gap 5–6 s.d. out.

### A second finding — separability does not imply score shift

**BGE separates calibration from deployment MORE easily than TF-IDF (domain AUC
0.9908 vs 0.9295), yet BGE is the space whose coverage transfers** (Tier-2 gap
−0.011, inside the band; Tier-1/TF-IDF −0.233). **Separability of two
distributions in a representation does not imply that scores computed in that
representation shift.** Cross-reference **Finding 1**: domain AUC measures
whether the *inputs* are distinguishable, coverage transfer measures whether the
*scores* are exchangeable, and the first does not predict the second. The
intuitive inference — "the embedding can tell them apart, so calibration will
not transfer" — is wrong here.

### Result 2 — the production representation is the degenerate one

The BGE arm triggered a **pre-registered** degeneracy condition: cross-fitted
domain **AUC 0.9908**. The in-domain and deployment sets are near-perfectly
separable **in the very space Tier-2 scores in**, so a density ratio is
ill-posed there.

**The blocked numbers are the flattering ones** — BGE at α = 0.05 moves the gap
from −0.1056 to **+0.0056**, essentially exactly nominal. They are in the CSV as
blocked and are **NOT a result**. Quoting them would repeat 6A's AURC error.
The 0.9908 is itself a quantitative statement of the named finding.

### Result 3 — the Tier-2 sanity check is NOT clean

Declared in advance as a check weighting must not break; it did not come back
clean. Of 18 Tier-2 configurations: **7 made |gap| worse, 5 were pushed outside
the band, 3 changed by more than the band.** Worst are all TF-IDF (α=0.20:
−0.0444 → −0.0667; α=0.10: −0.0111 → −0.0556).

**Reweighting has a cost and it is charged to the tier that did not need
repairing.** Report it beside any Tier-1 gain.

### Limitations, recorded beside the result

- **The target proxy is not the test set.** Weights are built toward the
  deployment 175; coverage is measured on the benchmark 45. So this measures
  "reweighting toward *this* target did not repair it", **not** "no reweighting
  could". Stated before the run, not after.
- **Weights are estimated, not known** (Barber et al. 2022 give the cost); no
  part of the residual is attributed to a particular cause.
- **n = 45**, so the ±2 s.d. band is ~4.5 points and smaller differences are not
  read.
- **Tier-2 carries no finding by construction**, as declared in advance.

### What was built

- `src/agent/conformal.py` — **additive only**: `weighted_conformal_quantile`,
  `_weighted_quantiles` (batched), `weighted_predict_sets`,
  `effective_sample_size`, `true_label_scores`. No existing function changed;
  production imports only `conformal_p_values`, untouched.
- `src/experiments/run_weighted_conformal.py` — the experiment, carrying the
  pre-registration in its docstring.
- `tests/test_weighted_conformal.py` — 22 tests (45 with parameterisation).
- `data/weighted_conformal_results.csv` — 42 rows, new filename.

### Verifications (rule 6 — second independent derivations)

- Unweighted rows **reproduce Finding 1 exactly**, checked **twice**: against a
  constant in the script *and* against the published
  `conformal_calibration_results.csv`, so a typo in the constant cannot become
  the thing the run validates against. Fatal on mismatch.
- **Uniform weights reproduce the unweighted quantile exactly** (`==`), over 4
  alphas × 6 sizes. Both reduce to rank `ceil((n+1)(1−α))` from the same float
  expression.
- **Batched quantile == scalar quantile**, exactly.
- **Coverage and `n_eff` each recomputed by a second expression** (`n/(1+CV²)`),
  fatal on disagreement.
- **Synthetic covariate-shift test** with `P(Y|X)` fixed and the true ratio
  supplied: 0.829 unweighted → 0.889 weighted at nominal 0.90. The first version
  of this test **did not bite** (a flatter test distribution over-covered at
  0.893) and would have passed a broken implementation; it was rebuilt.
- **Prior correction `n_cal/n_target` verified to be exactly 1.0**, not dropped.

### Rule-7 exception, stated

6B refits via `fit_scoring_models(df, embeddings, exclude_ids=set())` rather
than loading production artifacts. Not optional: the in-domain 175 **is** the
contaminated calibration set, and reproducing Finding 1's −0.233 exactly
requires the exact fit that produced it. 6A could use production artifacts
because both its eval sets were disjoint from training; 6B cannot.

### 6B gate — RE-RUN, not quoted

| Check | Result |
|---|---|
| `pytest` | **306 passed**, 0 failed (was 261) |
| Adversarial escalation | **9/9 PASS**, CSV sha256 `d53e40c616ace5a7` **byte-identical** |
| Golden parity | **45/45 and 9/9** exact |
| Ablation baseline (45) | **71.11% (32/45)**, CSV sha256 `8282e2db4353ca57` byte-identical |

---

## Phase 7A — external-validity feasibility (COMPLETE, gated and pushed in `ba98843`)

**FEASIBILITY ONLY — no replication, no experiment, no finding about this
project's methods.** Offline, zero Gemini calls. Production frozen; isolation
check confirms our dataset and artifacts byte-identical before and after.

### Verdict: the dataset CAN carry 7B

`Tobi-Bueck/customer-support-tickets`, revision
**`ddf1c81a5475992c4fa6752bf1e8b4e31f07bbeb`**, licence **CC-BY-NC-4.0**.

- **61,765 rows → 28,261 English** = **628×** our 45-ticket benchmark, and
  **12,500 distinct items after de-duplication = 278×**.
- **10/10 queues clear ≥300 English rows.**
- **Only 6 empty answers** — item 7 is viable.

**The framing is load-bearing and must appear in these words:
INDEPENDENTLY GENERATED DATA, NOT REAL PRODUCTION DATA.** The dataset card
advertises a synthetic ticket generator from the same author. Phase 7 tests
whether the findings survive **a different generator**, not whether they survive
reality.

**Why the real alternative was rejected:** Endava/Microsoft `all_tickets.csv` is
real, but its text is anonymized/encrypted, so BGE cannot read it. Real but
unreadable is worse than synthetic but readable.

### The methodological finding — a rate is meaningless without a control

**79.39% of external rows have a near-duplicate at BGE ≥ 0.95.** In isolation
that reads as disqualifying. Two controls, both now computed *inside* the
script, say otherwise:

1. **Is 0.95 a duplicate threshold for this encoder?** Random pairs sit at
   median **0.5912**; only **0.0100%** reach 0.95. It discriminates.
2. **High relative to what?** Against the corpus we already publish on:

| | External | **Ours** |
|---|---|---|
| Near-duplicate rate | **79.39%** | **85.20%** |
| Nearest-neighbour p05 | 0.8867 | 0.9270 |
| Distinct / rows | 12,500 / 28,261 (**44.2%**) | 1,213 / 4,000 (**30.3%**) |

**Our own corpus is MORE redundant on every measure.** High near-duplication is
a property of template-generated corpora generally — the same mechanism
Finding 2 and Phase 2A document from the inside — not a flaw distinguishing this
dataset. De-duplication is mandatory in 7B, but it is not a reason to prefer our
corpus. **Publishing 79.39% without the control would have been a plausible,
internally consistent, wrong conclusion.**

### Item 7 — the Phase 2A question IS answerable here

Measured with the production configuration (MiniLM @ 0.80 from config, reusing
`group_by_threshold` rather than reimplementing it). Distinct-answer rate
**0.26–0.55**; Technical Support gives **819 clusters from 1,500 answers, 646
singletons, largest cluster 209**.

Every queue has **both genuine singletons and substantial clusters** — merge
candidates *and* items that must not merge. That is exactly what our corpus
lacks (templates *are* the fix classes), so **clustering precision becomes
measurable here**.

### Limitations, recorded beside the numbers

- Independently generated, not real. CC-BY-NC-4.0 (non-commercial).
- **Item 7 sampled at 1,500/queue**, seed 42, because `group_by_threshold` is
  production's O(n²) loop. `sampled`/`sample_cap` columns record it per row.
- **`version` and source-file are confounded** — all 11,923 version-NaN rows are
  the `4-20k` file. One split, not two axes.
- **12.9% of rows have no subject** (3,639); text is subject+body so they are
  body-only, not empty.
- **Encode cost:** 28,261 texts in **5,541 s (~92 min, 5.1/s)**. Throughput
  varied **2.9–5.1/s** with machine load, so a single-number estimate for this
  box would be false precision.

### 7A gate — RE-RUN, not quoted

| Check | Result |
|---|---|
| `pytest` | **322 passed**, 0 failed (306 + 16 new; the sum is the second derivation) |
| Adversarial escalation | **9/9 PASS**, CSV sha256 `d53e40c616ace5a7` byte-identical |
| Golden parity | **45/45 and 9/9** exact |
| Ablation baseline (45) | **71.11% (32/45)**, CSV sha256 `8282e2db4353ca57` byte-identical |
| Isolation | our dataset + benchmarks byte-identical before/after |

---

## Phase 7B — Finding 1 does NOT replicate (COMPLETE, gated and pushed in `f789b8d`)

Offline, **zero Gemini calls**, measurement only. Production frozen; both
`enabled` flags stay `False`. The isolation check was widened from 7A's three
files to **22** — our dataset, both benchmarks, the deployment calibration set,
`models/`, `tests/goldens/`, and every published conformal, deferral and
ablation CSV — all hashed before and after, all byte-identical.

### The primary result

Design A: calibrate on the `aa` file's English rows at `version` ∈ {51, 52},
test on `version` == 400. Pool 5,897 raw → **3,305** de-duplication components
(56.05% distinct) → **1,978 train / 1,327 calibration**, seed 42, stratified by
queue. Test **10,441**, used in full. Both tiers fitted **fresh on the external
training split only**.

| tier | α | coverage | gap | ±2 s.d. band | outside? |
|---|---|---|---|---|---|
| tier1 | 0.05 | 0.9480 | −0.0020 | 0.0120 | in |
| tier1 | 0.10 | 0.9009 | **+0.0009** | 0.0165 | in |
| tier1 | 0.20 | 0.7780 | −0.0220 | 0.0220 | **OUT** (in on the no-neighbour arm) |
| tier2 | 0.05 | 0.9545 | +0.0045 | 0.0120 | in |
| tier2 | 0.10 | 0.8928 | −0.0072 | 0.0165 | in |
| tier2 | 0.20 | 0.7907 | −0.0093 | 0.0220 | in |

On our corpus Tier-1's α=0.10 gap is **−0.2333** against Tier-2's −0.0111.
Here the two tiers are **indistinguishable**, and the band is 2.8× tighter
(0.0165 vs 0.0454). **The pre-registered headline question — "does TF-IDF lose
far more coverage than BGE, as on our data?" — is answered NO.**

### The fixed wordings for 7B

- **Finding 1 is SCOPE-NARROWED, NOT WITHDRAWN.** Its numbers on our corpus are
  unchanged and were re-verified by 6B.
- **The paper's claim is fixed as:** *coverage transfer depends on the
  representation **under paraphrase shift**, on our corpus; untested elsewhere
  until 7C.* **The shift mechanism goes in the claim**, not only the corpus —
  narrowing to "a property of our corpus" alone silently asserts that 7B's
  shift and Finding 1's shift were equivalent tests.
- **Two candidate explanations for the null. State BOTH; let NEITHER rescue the
  finding.**
  1. **7B may not be like-for-like.** Design A is a **version shift from the
     same generator**; Finding 1 was measured under a **paraphrase/register
     shift**. TF-IDF's failure mechanism is **surface-vocabulary change**, and
     a version shift need not produce any. **Phase 7C tests exactly this.**
  2. **No representation contrast to find** — Tier-1 34.8% vs Tier-2 37.3%,
     2.6 points against 35.6 on ours. Finding 1's mechanism needs a better
     representation to be the one that transfers.
- **Never write "7B refutes Finding 1."** Until 7C reports, the honest position
  is that the finding holds under the shift it was measured under, on the
  corpus it was measured on, and has not been shown to hold anywhere else —
  and that 7B's null is weaker evidence against it than the bare numbers
  suggest.
- **The named cross-phase finding is SHARPENED by this, not damaged.** "The
  calibration/reference distribution, not the test or method, is the binding
  constraint" predicts exactly this outcome. 7B is the **fourth** phase to
  reach that wall and the first to reach it by *removing* an effect rather than
  by failing to repair one.
- **Never quote an external set size against ours.** Mean sets run 3.5–6.7 of
  **ten** labels here versus seven categories on ours; the corpora are not
  comparable in absolute terms.

### The deferral secondary — 6A's question resolves, for the first time

The transplanted 0.50 gate covers **6.15% = 642 tickets** here, against the
**four** that left 6A's gated axis unmeasurable.

**3 of 6 comparisons resolve, and all three favour the confidence incumbent:**
Tier-1 LAC +0.0466 [+0.0156, +0.0779], Tier-1 APS +0.0532 [+0.0236, +0.0836],
Tier-2 APS +0.0330 [+0.0126, +0.0530]. Risk is error among accepted tickets, so
positive means **worse**. No comparison favours conformal.

**Fixed wording.** This **does** say conformal deferral is worse rather than
equal *on this corpus, at this coverage*. It **does not** retro-license
"conformal is worse" as a claim about ours — **6A's wording, "no evidence
either way on the gated axis", still stands for the data 6A measured.**
Different corpus, transplanted gate, ~35% label accuracy.

Conformal still cannot be **operated** at the gate: achievable α-indexed
coverages are 0.39%/0.75%/2.43% (Tier-1) and 0.34%/1.82%/3.90% (Tier-2) against
6.15%. **AURC agrees with the gated axis this time and is still not promoted** —
a coherence observation, nothing more.

### Design B resolved nothing, twice, for two different named reasons

The recorded design never specified the test arm, so **both readings were run
rather than one chosen after the fact**:

| variant | AUC range | why no resolution |
|---|---|---|
| `full_test` | 0.8312–0.8566 (span 0.025) | the magnitude knob does not move — a queue holdout barely changes a shift the version split already dominates |
| `held_out_queue` | 0.8499–0.9693 (span 0.120) | the test arm carries **one class**, so coverage becomes class-conditional rather than marginal — a different quantity |

The degeneracy rule also fired for real: **Billing and Payments hits AUC 0.9693
and is BLOCKED**, the same condition that blocked 6B's BGE arm. Its rows are in
the CSV marked blocked, never as results.

### RECORD CORRECTION — 7A's ad hoc domain AUCs do not reproduce

**Design A was recorded as 0.8584.** Fully specified recomputation (5-fold
cross-fitted LogReg in BGE space, seed 42, reusing 6B's
`cross_fitted_domain_probabilities`) gives **0.8727 raw / 0.8472 operating**.
Five variants were tried while chasing the recorded value and they span
0.8637–0.8727; none lands on 0.8584. **Design B's recorded 0.6706–0.9316 does
not reproduce either**, though the queue *ordering* does.

The figures were computed ad hoc at the 7A gate and never committed — which is
exactly why 8A exists. **The design conclusion is unchanged**: every value sits
below the 0.95 degeneracy line, which is what the numbers were recorded to
establish. README's 7A design table now carries both the struck-through and the
corrected values.

Also corrected: the post-de-duplication estimate of "~2,600" extrapolated the
corpus-wide 44.2% distinct rate; measured within the pool it is **56.05% →
3,305**.

### Verification

Embedding-cache alignment **proved, not assumed** (64 seeded rows re-encoded,
worst cosine 0.99999994 — a row-count check is exactly what a misaligned cache
passes). Rule 6 throughout: arm sizes two ways, coverage two ways, FAISS vs
brute force on 200 probes (0 mismatches), every deferral curve's coverage-1.0
endpoint against the independently measured accuracy. Split integrity checked
by row **and by de-duplication component**. **18 new unit tests** pin two bugs
found while building 7B: a CSV writer that dropped ragged rows (it would have
deleted every BLOCKED Design B row) and a split that was row-disjoint while
sharing components.

**Gates re-run, never quoted:** `pytest` **340 passed**; adversarial **9/9**
with its CSV byte-identical; goldens **45/45** and **9/9**; ablation baseline
**32/45**.

---

## Phase 7C — Finding 1 under a paraphrase shift: BLOCKED (COMPLETE, gated and pushed in `0c9ff8b`)

Offline, **zero Gemini calls** (local Ollama). Measurement only; production
frozen; both `enabled` flags stay `False`. Isolation check: **27 files**
byte-identical, including 7B's own outputs.

### The verdict is a NON-ANSWER, and is reported as one

**`blocked_auc_ge_0.95`.** The pre-registered degeneracy rule fired: the
TF-IDF-space cross-fitted domain AUC between originals and paraphrases is
**0.9972**, above the 0.95 line borrowed from 6B. **No verdict is drawn on the
primary**, and the blocked numbers sit in the CSV marked blocked — as 6B's BGE
arm does — not as a result.

**After 7B and 7C, Finding 1's status is unchanged:** it holds under the shift
it was measured under, on the corpus it was measured on, and has **not been
shown either to hold or to fail anywhere else.**

### The manipulation worked — that is not in doubt

| space | mean cosine | domain AUC |
|---|---|---|
| **BGE** (meaning preserved) | 0.8462 | 0.9981 |
| **TF-IDF** (surface vocabulary moved) | **0.2489** | **0.9972** |

Surface vocabulary is almost entirely replaced while meaning survives. 7B's
version shift, for scale, sits at BGE-space AUC 0.8472.

**Length ratio (pre-registered secondary), paraphrase/original:** tokens mean
1.0279 / median 0.9543 / **aggregate 0.9417**; words **aggregate 0.9344**.
About 6% shorter — no length confound.

### SPEC ERROR, OWNED — and the gate DECLINED to unblock

In **6B** the ≥0.95 rule gated a **density-ratio estimate**, where
near-separability makes the ratio ill-posed — the rule measured what it gated.
**7C estimates no density ratio**; it runs plain split conformal and uses the
AUC only as a *manipulation check*, where a near-1.0 value means the
manipulation was **strong**. Carrying the threshold across may gate the wrong
quantity — the error this project has already named once, when a clustering
promotion rule measured recall while claiming to gate precision.

**The counter-reading is real and is recorded too:** a rewrite a classifier
identifies with near-certainty is arguably a *different corpus* rather than a
shifted one.

**Decided at the gate (2026-09-22): DO NOT UNBLOCK.** A pre-registered rule is
not revised after seeing the results, however good the critique — that is
precisely what this project refuses to do. The verdict stays BLOCKED, the
critique is recorded as a **specification error**, and the unblocked coverage
numbers stay in the CSV **labelled post-hoc, never quoted as the verdict**.

> **Lesson for every future pre-registration: a degeneracy rule must be
> justified by what the specific design estimates, not copied across designs.**
> Ask what quantity becomes ill-posed at the threshold, and confirm that
> quantity exists in *this* experiment.

### POST-HOC: the mechanism IS visible — in accuracy, not coverage

| arm | Tier-1 accuracy | Tier-2 accuracy |
|---|---|---|
| original | 0.3776 | 0.3776 |
| paraphrased | **0.2727** | **0.3531** |
| change | **−10.5 pts** | **−2.5 pts** |

**The paraphrase shift hit the lexical model roughly four times harder** —
Finding 1's mechanism, confirmed directly. 7B never produced this contrast,
which supports the diagnosis that its version shift was the wrong kind of shift.

**It survives its own interval.** Bootstrapped before being written up — 10,000
paired draws, seed 42, the same configuration as 6A and the blocked arm:

| statistic | point | 95% CI |
|---|---|---|
| paraphrased, T1 − T2 | −0.0804 | [−0.1259, −0.0350] |
| original, T1 − T2 | +0.0000 | [−0.0455, +0.0455] |
| **difference-in-differences** | **−0.0804** | **[−0.1364, −0.0210]** |

The interval **excludes zero**. In counts, Tier-1 fell **108 → 78** of 286 while
Tier-2 fell **108 → 101**. **Still POST-HOC and exploratory**: it does not
replace the blocked primary and the labels remain unaudited.

**FUTURE WORK, not tested and not pursued in this programme:** coverage may be
buffered by **large prediction sets** (4.9–6.9 of 10 labels here), so Finding 1
may require **both** a representation contrast **and** small sets. Recorded as
future work, not as a hypothesis this programme goes on to check.

The identical original-arm accuracies were checked, not assumed: the tiers
disagree on **87 of 286** tickets with a **23/23** discordant split and
different probability matrices. A genuine coincidence.

### Limitations, beside the result

- Labels generator-assigned and **unaudited**, base accuracy ~35%. A null is
  weak evidence here — and the low accuracy may be the *cause* of the null, not
  incidental to it.
- The paraphraser is a **3B local model** with a recognisable style, part of why
  the domain classifier separates the arms so easily.
- Independently generated data, not real production data — corpus and
  paraphrases both.

### Run provenance, and the RAM deviation

The generation run was **killed twice by the harness for low system memory** and
**resumed from cache both times**, with no regeneration and no data loss: first
at 168/299, then after all 299 had completed, during the guard step. **The
second kill's cause was the BGE/Ollama overlap** — Ollama holds its model for
five minutes after the last call, so loading BGE for the similarity guard puts
both models resident at once. That is the memory peak of the phase.

The fix, applied before the final pass: `ollama stop qwen2.5:3b-instruct`
confirmed against `/api/ps`, then **`--cached-only`**, which refuses every live
call and makes a cache miss or prompt-hash mismatch fatal rather than a silent
regeneration. It verified **299/299 prompt hashes** and made **zero** calls.

**DEVIATION, accepted at the dry-run gate:** 5C's `require_free_ram`
double-counts a model Ollama already holds, and refused a run with 4,339 MB
genuinely in play (2,275 available + 2,064 resident, floor 3,277).
`--allow-low-ram` would have proceeded while stamping every response
`low_ram_override` — permanently and **falsely** marking the paraphrases as
untrustworthy. So the **check was corrected, not bypassed**: a 7C-local
preflight adds resident CPU-side bytes back. `run_zeroshot_baselines.py` is
untouched. **It still fails closed** — an unreadable residency counts as zero
resident — and four tests pin exactly that.

### Verification

7B's models **proved** unchanged (refit reproduces cached probabilities at
max |Δ| = 0.0, classes identical); pairing re-verified against the corpus
before any statistic; coverage derived twice; guard counts two ways; **the band
is COMBINED**, since at n=286 the test-sampling sd (0.0173) dominates the
calibration term (0.0082) — quoting the calibration term alone, as Finding 1
and 7B do on far larger test arms, would understate uncertainty by ~2×.

**Gates re-run, never quoted:** `pytest` **372 passed**; adversarial **9/9**
with its CSV byte-identical; goldens **45/45** and **9/9**; ablation baseline
**32/45**.

---

## The agreed Phase 5–9 programme

Agreed 2026-09-20. **The paper is the deliverable.** This replaces the earlier
candidate list: a fresh session should resume from this table, not re-derive it.
Everything in Phases 5–9 is **measurement only**.

### Phase 5 — Correctness and honest baselines

| Sub-phase | Scope | State |
|---|---|---|
| **5A** | Conformal template-grouping correction | **DONE** (`cdac8f6`) |
| **5B** | Honest ablation | **DONE** (`2a744ed`, gated and pushed) |
| **5C** | **Zero-shot LLM classification baselines** | **DONE** (`4793785`, gated and pushed) |

**5B — honest ablation. DONE** (see "Phase 5B" above). The like-for-like check
cleared — 33/45 and 32/45 came from the same bit-identical classifier, so there
was no seventh bug — and the finding is that the cascade is a **latency
optimisation**, not an accuracy gain: −1 ticket on both sets at exact McNemar
p = 1.000, buying 8–18% of median per-ticket latency.

**5C — zero-shot LLM baselines.** Gemini plus one free non-Gemini model run
locally through Ollama, so the comparison is not single-vendor. **~60 Gemini
calls**; dry-run first and cache every raw response.

### Phase 6 — New research experiments (measurement only)

| Sub-phase | Scope |
|---|---|
| **6A** | Conformal deferral vs a confidence threshold — risk–coverage curves and AURC — **DONE** (`8f2f5e1`, gated and pushed) |
| **6B** | Weighted conformal under shift, with a domain-classifier density ratio — **DONE** (`8419db6`, gated and pushed) |
| **6C** | Retrieval-sufficiency gate, on the 33 groundedness tickets — **DONE, GATED and PUSHED** (`c5497bf`). Caught 2/2, flagged 26/31: **not usable as a gate**. 58 Gemini calls |

6A is the experiment that makes Phase 1's conformal work operational without
promoting it: it asks whether deferring on set size beats deferring on a
confidence threshold, on the same data. 6B is the direct successor to the named
finding — it tests whether reweighting can do what distribution matching could
not.

### Phase 7 — External validity

| Sub-phase | Scope |
|---|---|
| **7A** | Feasibility check on `Tobi-Bueck/customer-support-tickets` — **DONE** (`ba98843`, gated and pushed) |

**PHASE 7 IS CLOSED as of the 7C gate (2026-09-22). No further external
experiments in this programme.** External replication of Finding 1 is
**inconclusive**: 7B applied a version shift (the wrong type), 7C applied the
right type but was **blocked by its own pre-registered rule**. The set-size
question 7C raised post-hoc is **future work, not a 7D**.

| **7B** | Replicate Finding 1 on it — **DONE, GATED and PUSHED** (`f789b8d`). **It does NOT replicate** under a *version* shift. Design B resolved nothing, twice, for two named reasons |
| **7C** | Replicate Finding 1 under a **paraphrase** shift — **DONE, GATED and PUSHED** (`0c9ff8b`). **BLOCKED**, and the gate declined to unblock. Post-hoc: the mechanism shows in accuracy, DiD **−0.0804** [−0.1364, −0.0210] |

**PHASE 7 IS NOT OPTIONAL — priority RAISED 2026-09-21 after the 6A gate.**
Evaluation-set size at low coverage is now the binding constraint in **four**
places: Phase 1's coverage transfer, Phase 4B-1's provisional power numbers,
Phase 5C's corpus ceiling, and Phase 6A, where the live gate's ~9–19% operating
coverage contains only 4 tickets on the benchmark and 34 on the deployment set,
leaving the gated axis unmeasurable in 3 of 4 configurations.

`Tobi-Bueck/customer-support-tickets` carries **61.8k rows** and is **the only
planned work where a low-coverage comparison could actually resolve** — a 10%
operating coverage there is thousands of tickets rather than four. That makes
Phase 7 a prerequisite for re-asking 6A's question (and 6B's) with any power,
not an external-validity nicety.

**7A's framing is load-bearing:** that dataset is itself LLM-generated, so it is
treated as **independently generated data, not real production data**. It tests
whether the findings survive a different generator — not whether they survive
reality. Any write-up must say so in those words.

### Phase 8 — Reproducibility

| Sub-phase | Scope |
|---|---|
| **8A** | `build_paper_artifacts.py` + `paper/NUMBERS.md` + a parity test for it — **DONE and GATED**. 16 tables, 7 figures, 161 numbers; audit 54 anchors / 0 mismatches / 5 no-source |
| **8A.1** | Give the five "no committed source" numbers a source — **DONE**. 4 sourced, 1 unsourceable, and **1 non-reproduction (TF-IDF 6/14, not 7/14)**. 188 numbers / 51 sources / 59 anchors / 0 mismatches |
| **8B** | Docker + CI — **DONE and GATED** (2026-09-23). Digest-pinned image that builds its own artifacts, two workflows, a committed deployment verifier. Found **occurrence #7** of the recurring bug class in the `.dockerignore` rule |

8A is the structural answer to this project's recurring bug class: every number
in the paper regenerated from one script, with a parity test that fails when a
published figure drifts — the goldens pattern applied to the write-up.

### Phase 9 — The paper

| Sub-phase | Scope |
|---|---|
| **9A** | IEEE draft |
| **9B** | Pre-submission audit |
| **9C** | Author explainer |

9A builds on the **named finding** above rather than re-deriving it, and keeps
Phase 2A as a *related* corpus limitation rather than a third instance of the
mechanism.

### Explicitly FUTURE WORK — not in this programme

- **4B-2 / 4B-3 — a held-out in-domain set** (and a deployment-traffic drift
  reference). Until they exist, Signal A's power numbers stay provisional and no
  drift monitor can run; that limitation is already written next to the 4B-1
  result.
- **Re-running the Phase 2 harnesses on deployment-distribution data.** Both
  harnesses are built and guarded and need no code changes — only real resolved
  tickets.

### Still deferred, each needing its own gate

Turning the decision-log sink on anywhere (e.g. in `/triage`), and any `/drift`
endpoint.

### Rules for every sub-phase (unchanged)

- **Plan mode first.** Files touched, new files, outputs, anything that could
  move a published number, how it will be verified, and the Gemini cost. Stop
  for approval; no code before it.
- **Measurement only. Production stays frozen** at cascade **0.50**, RAG
  **0.67**, clustering **0.80**, with `settings.conformal.enabled` and
  `settings.drift.enabled` both `False`.
- **Gates are re-run, never quoted:** `pytest`, the adversarial gate at 9/9 with
  its CSV byte-identical, goldens 45/45 and 9/9, ablation baseline 32/45.
- **Commit and stop.** Push only on "gate cleared".
- **Never run two Gemini-spending sub-phases (5C, 6C) on the same day** — the
  500/day cap would risk a partial result mid-experiment.
- Benchmarks (14 / 45 / 9 tickets) are read-only; seed 42; a new result gets a
  new filename; check every count against a second, independent derivation.

---

## Immediate next step

**Phase 9A — the IEEE draft.** Opens in plan mode like every sub-phase. It
builds on the **named finding** recorded above rather than re-deriving it, and
keeps Phase 2A as a *related* corpus limitation rather than a third instance of
the mechanism. Phase 9A must also settle Finding 1's final wording, whose three
clauses travel together (measured on our corpus; external replication
inconclusive; post-hoc mechanism evidence in accuracy).

Two things Phase 8B hands it. **A seventh instance of the recurring bug
class**, which is now a *packaging* rule rather than a model swap, a
routing/eligibility test or a grouping key — the paper's "recurring bug class"
narrative should say the shape generalises further than the first six
suggested, and that this one was caught by reading a log rather than by a
control. And **a cross-platform reproducibility fact**: Tier-1's confidence is
bit-identical between Windows and a Linux container, the BGE/FAISS similarity
is not (fixed -1.192e-07), which belongs in the reproducibility section beside
the container.

**Phase 8B is DONE and GATED** (see the top of this file). Docker + CI landed:
`pytest`, the adversarial gate at 9/9 with its CSV byte-identical, goldens
45/45 and 9/9, the ablation baseline at 32/45, and **paper parity** —
`tests/test_paper_artifacts.py`, which fails if any published number, table CSV
or figure-data CSV drifts — now run in CI, the first two in `ci.yml` on every
push and the rest in `gates.yml` on demand.

**8A.1 is the argument for putting paper parity in CI.** The number it caught
(TF-IDF 7/14 → 6/14) had been wrong in the README for as long as the dataset
had been 4,000 rows, and nothing could have caught it, because no file
recorded it. Now one does, and the parity test compares it on every run.

### What 8A left for the write-up to decide — SETTLED BY 8A.1

**Superseded.** 8A.1 took the third option for four of the five (add the
missing writer and re-run the original configuration) and established that the
fifth cannot be sourced at all. One of the four **did not reproduce**: the
TF-IDF baseline's 14-ticket score is **6/14**, not 7/14. See the 8A.1 block at
the top of this file. The original list is kept below as the record of what
8A found.

**Five numbers have no committed machine-readable source.** They are flagged
in `paper/NUMBERS.md` and `paper/RECONCILIATION.md`, never quoted as though
they had one, and nothing was invented or re-run to manufacture one:

1. **TF-IDF + LogReg, 7/14** — `train_baseline_tfidf.py` and
   `generalization_test.py` print their results and write no file.
2. **Fine-tuned DistilBERT, 7/14** — `train_distilbert.py` writes only
   `label_mapping.json`.
3. **DistilBERT on the 45-ticket benchmark** — never measured at all.
4. **The cascade threshold-by-target-accuracy table** (and "three attempts,
   two rejected") — `train_cascade.py` prints its sweep and writes nothing.
5. **Self-retrieval contamination 5.7% (10/175)** — computed inside
   `calibrate_rag_similarity_threshold.py`, never written to a column.

Each is a decision: keep it as a figure whose derivation is a script that must
be re-executed, drop it from the paper, or add the missing writer in a later
phase. **Re-running any of them is out of scope for a measurement-only
phase**, so 8A flagged them rather than fixing them.

**PHASE 7 IS CLOSED and 8A changed nothing about that.**

**PHASE 7 IS CLOSED. NO FURTHER EXTERNAL EXPERIMENTS in this programme.** 7A
(feasibility), 7B (version shift) and 7C (paraphrase shift) are all gated and
pushed. External replication of Finding 1 is **inconclusive** and is written up
as such; the set-size question raised post-hoc in 7C is **future work**, not a
7D.

### What 8A must cover

**8A's remit grew because of 7B and 7C.** Two recorded design AUCs turned out to
be ad hoc figures whose derivation was never committed and which do not
reproduce (the 7B record correction). **Any number quoted at a gate must come
from a committed script**, not from an interactive session. 8A is the structural
answer: every number in the paper regenerated from one script, with a parity
test that fails when a published figure drifts — the goldens pattern applied to
the write-up.

6C adds two figures to that surface: the **2×2 counts (2/2 caught, 26/31
flagged)** and the **cross-family agreement (39/54)**. Both are in
`data/sufficiency_gate_summary.json` and must be read from there, never
retyped.

### Order after 8A

**8A → 8B → 9A–9C.**

**8A's remit grew because of 7B and 7C.** Two recorded design AUCs turned out to
be ad hoc figures whose derivation was never committed and which do not
reproduce (the 7B record correction). **Any number quoted at a gate must come
from a committed script**, not from an interactive session.

### Carried forward for Phase 9A

1. 5C widened the named finding from the calibration/reference distribution to
   the training distribution as well; **9A settles the final wording.**
2. **The corpus's nasscom-brief origin** opens the experimental-setup section,
   immediately followed by the corpus-as-object-of-study framing (Phase 2A,
   Finding 2, Phase 5C).
3. **6B's wordings are fixed:** "a partial repair, not a correction" — never
   "the shift is correctable by covariate reweighting"; the BGE arm is
   **blocked, not a result**; and **no ordered comparison** between reweighting
   and distribution matching (residuals differ by 0.022, inside the 0.045 band).
4. **6B's second finding:** separability does not imply score shift — BGE
   separates calibration from deployment more easily than TF-IDF (0.9908 vs
   0.9295) yet is the space whose coverage transfers. Cross-reference Finding 1.
5. **7A's control lesson:** never quote a redundancy or similarity rate without
   stating what it is high *relative to*. Our own corpus is **more**
   near-duplicated (85.20%) than the external one (79.39%). 7B applies it
   again: its 1.46% boundary-contamination rate is low because the *reference
   set* is 3,305 rows, not because the corpus changed.
6. **FINDING 1'S PAPER FRAMING IS FIXED** (7C gate, 2026-09-22), and the three
   clauses travel together — none is quoted alone:
   **measured on our corpus; external replication inconclusive (7B applied the
   wrong shift type, 7C was blocked by its own pre-registered rule); with
   post-hoc evidence of the mechanism in accuracy on the external corpus
   (difference-in-differences −0.0804, 95% CI [−0.1364, −0.0210]).**
7. **7B's deferral reading must not be merged with 6A's.** 6A: "no evidence
   either way on the gated axis", for our data. 7B: conformal deferral is
   **worse** — 3 of 6 resolving comparisons, all favouring the incumbent — on a
   different generator's corpus, with a transplanted gate and ~35% label
   accuracy. Two readings, two scopes, reported separately.
8. **The 7B record correction belongs in the reproducibility section**, as the
   concrete motivation for 8A: a number quoted at a gate and never regenerated
   from a committed script is a number that can stop being true.
9. **6C's wordings are fixed.** The headline is **"catches both, and is still
   not usable as a gate"** — never "the sufficiency check caught the two cases
   the gate missed", which is true and misleading alone. **6C is NOT an
   instance of the named finding**; the near-duplicate explanation is a
   **rejected hypothesis** (2B's own 71.1% / 88.9% distinct-fix diagnostic
   refutes it), and the mechanism is a **measurement-target mismatch** between
   sufficiency and groundedness. The **binding limitation is the label** — 2B's
   groundedness labels are an outcome proxy for draft support, not context
   sufficiency, so the 26 flags are false only against that proxy. The
   **N45 counter-case** (the gate escalated a ticket whose context was
   adequate) travels with the result: the scalar errs in both directions.
   **No ordered comparison** in the post-hoc declining-draft split — 3/3
   against 23/28 with n=3 carries none.
10. **7C's spec error belongs in the methods section**, as the concrete case for
   pre-registration discipline: the rule was kept even though the critique of it
   was sound, because revising a rule after seeing results is what the
   discipline exists to prevent. **A degeneracy rule must be justified by what
   the specific design estimates, not copied across designs.**


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
- **The recurring bug class has surfaced six times** (four of them
  stale-artifact specifically), twice reaching published results — once
  standing for eleven days. Anything touching a model, index, threshold,
  routing/eligibility test or grouping key should be assumed to have a
  **seventh** instance waiting. Run `pytest` and the adversarial gate before
  believing a green result, and check every count against a second,
  independent derivation. Phase 5B closed one latent instance:
  `run_ablation_study.py` was still refitting Tier-1 locally instead of
  loading the persisted artifact.
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
