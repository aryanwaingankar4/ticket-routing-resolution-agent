# RECONCILIATION.md -- the documents against their source files

**GENERATED. Do not edit by hand.**

Phase 8A's rule: where a document and a committed result file disagree, the DOCUMENT is wrong and is fixed. No result file was edited to match a document, and no number was invented to give a document a source.

## Convention notes

The repository contains **two conventions for a 95% z**: `1.96` in `src/experiments/score_groundedness_set.py` and `src/experiments/score_sufficiency_gate.py`, and the exact normal quantile `1.959963984540054` in `src/experiments/summarize_zeroshot_baselines.py`. The two implementations are otherwise algebraically identical -- the build asserts that -- and they differ by about 3.5e-6, so **no published figure is affected at reported precision**. The paper adopts `1.96`. Confidence intervals that a source file already carries are read verbatim rather than recomputed, so a published interval can never be silently restated under a different z.

---

## Pass A -- anchored checks

For each anchor, the document's stated value is compared against the value the committed file currently produces. This is the pass that would catch a Phase 7A-style drift.

| anchor | id | expected | status | found in |
|---|---|---|---|---|
| cascade, benchmark45 | `T1.cascade.benchmark45` | 32/45 | ok | CLAUDE.md, PROJECT_OVERVIEW.md, PROJECT_STATUS.md, README.md |
| Tier-2-only, benchmark45 | `T1.tier2only.benchmark45` | 33/45 | ok | CLAUDE.md, PROJECT_OVERVIEW.md, PROJECT_STATUS.md, README.md |
| Tier-1-only, benchmark45 | `T1.tier1only.benchmark45` | 16/45 | ok | CLAUDE.md, PROJECT_OVERVIEW.md, PROJECT_STATUS.md, README.md |
| cascade, deployment175 | `T1.cascade.deployment175` | 131/175 | ok | CLAUDE.md, PROJECT_STATUS.md, README.md |
| Tier-2-only, deployment175 | `T1.tier2only.deployment175` | 132/175 | ok | CLAUDE.md, PROJECT_STATUS.md, README.md |
| Tier-1-only, deployment175 | `T1.tier1only.deployment175` | 91/175 | ok | CLAUDE.md, PROJECT_STATUS.md, README.md |
| TF-IDF baseline, benchmark14 | `T1.tfidf_baseline.benchmark14` | 6/14 | ok | CLAUDE.md, PROJECT_STATUS.md, README.md |
| DistilBERT, benchmark14 | `T1.distilbert.benchmark14` | 7/14 | ok | CLAUDE.md, PROJECT_OVERVIEW.md, PROJECT_STATUS.md, README.md |
| self-retrieval contamination | `T5.self_retrieval_rate` | 10/175 | ok | CLAUDE.md, PROJECT_OVERVIEW.md, PROJECT_STATUS.md, README.md |
| cascade sweep, threshold at the 70% target | `T4.sweep.threshold.target70` | 0.50 | ok | CLAUDE.md, PROJECT_OVERVIEW.md, PROJECT_STATUS.md, README.md |
| zero-shot Gemini, benchmark45 | `T1.zeroshot_gemini.benchmark45` | 40/45 | ok | CLAUDE.md, PROJECT_STATUS.md, README.md |
| zero-shot Gemini, benchmark14 | `T1.zeroshot_gemini.benchmark14` | 14/14 | ok | PROJECT_STATUS.md, README.md |
| zero-shot Qwen, benchmark45 | `T1.zeroshot_qwen.benchmark45` | 34/45 | ok | CLAUDE.md, PROJECT_STATUS.md, README.md |
| zero-shot Qwen, benchmark14 | `T1.zeroshot_qwen.benchmark14` | 12/14 | ok | PROJECT_STATUS.md, README.md |
| Gemini vs Tier-2, exact McNemar | `T1.mcnemar.gemini_vs_tier2.benchmark45` | 0.0391 | ok | CLAUDE.md, PROJECT_STATUS.md, README.md |
| Qwen vs Gemini, exact McNemar | `T1.mcnemar.qwen_vs_gemini.benchmark45` | 0.070 | ok | CLAUDE.md, PROJECT_STATUS.md, README.md |
| Qwen Application precision | `T2.qwen.application.precision.benchmark45` | 47 | ok | README.md |
| cascade vs Tier-2, benchmark45 | `T3.mcnemar.benchmark45` | 1.000 | ok | CLAUDE.md, PROJECT_STATUS.md, README.md |
| Tier-1 warm latency | `T3.latency.tier1_median_ms` | 1.04 | ok | CLAUDE.md, PROJECT_STATUS.md, README.md |
| Tier-2 warm latency | `T3.latency.tier2_median_ms` | 156.40 | ok | CLAUDE.md, PROJECT_STATUS.md, README.md |
| Tier-2 / Tier-1 latency ratio | `T3.latency.tier2_over_tier1` | 151 | ok | CLAUDE.md, PROJECT_STATUS.md, README.md |
| Tier-1 ECE | `T4.tier1.ece` | 0.1122 | ok | CLAUDE.md, PROJECT_OVERVIEW.md, README.md |
| Tier-2 ECE | `T4.tier2.ece` | 0.0992 | ok | CLAUDE.md, PROJECT_OVERVIEW.md, README.md |
| OOD leakage at the gate | `T5.ood_leakage_at_gate` | 13.3 | ok | PROJECT_OVERVIEW.md, README.md |
| OOD leakage two steps below | `T5.ood_leakage_below_gate` | 37.8 | ok | PROJECT_OVERVIEW.md, README.md |
| adversarial safe range, low | `T5.safe_range_low` | 0.6196 | ok | PROJECT_OVERVIEW.md, README.md |
| Finding 1, Tier-1 gap | `T7.tier1.coverage_gap` | -0.2333 | ok | CLAUDE.md, PROJECT_STATUS.md, README.md |
| Finding 1, Tier-2 gap | `T7.tier2.coverage_gap` | -0.0111 | ok | CLAUDE.md, PROJECT_STATUS.md, README.md |
| 6B, reweighted Tier-1 gap | `T7.weighted.tier1.coverage_gap` | -0.1222 | ok | PROJECT_STATUS.md, README.md |
| 6B, BGE-arm domain AUC (BLOCKED) | `T7.weighted.bge.domain_auc` | 0.9908 | ok | CLAUDE.md, PROJECT_STATUS.md, README.md |
| templates in the dataset | `T8.templates_total` | 66 | ok | CLAUDE.md, PROJECT_OVERVIEW.md, PROJECT_STATUS.md |
| templates touched by the calibration set | `T8.templates_touched` | 62 | ok | CLAUDE.md, PROJECT_STATUS.md |
| rows surviving template exclusion | `T8.rows_surviving_template_exclusion` | 210 | ok | CLAUDE.md, PROJECT_OVERVIEW.md, PROJECT_STATUS.md, README.md |
| external English rows | `T11.7a.rows_english` | 28261 | ok | PROJECT_STATUS.md, README.md |
| external near-duplicate rate | `T11.7a.near_duplicate_rate_external` | 79.39 | ok | CLAUDE.md, PROJECT_STATUS.md, README.md |
| our corpus near-duplicate rate | `T11.7a.near_duplicate_rate_our_corpus` | 85.20 | ok | CLAUDE.md, PROJECT_STATUS.md, README.md |
| 7B, Tier-1 gap | `T11.7b.tier1.coverage_gap` | 0.0009 | ok | CLAUDE.md, PROJECT_STATUS.md, README.md |
| 7B, Tier-2 gap | `T11.7b.tier2.coverage_gap` | -0.0072 | ok | CLAUDE.md, PROJECT_STATUS.md, README.md |
| 7B, noise band | `T11.7b.noise_band_2sd` | 0.0165 | ok | CLAUDE.md, PROJECT_STATUS.md, README.md |
| 7B, operating domain AUC | `T11.7b.domain_auc_operating` | 0.8472 | ok | CLAUDE.md, PROJECT_STATUS.md, README.md |
| 7B, boundary contamination | `T11.7b.boundary_contamination_rate` | 1.46 | ok | PROJECT_STATUS.md, README.md |
| 7C, TF-IDF domain AUC (BLOCKED) | `T11.7c.tfidf_domain_auc` | 0.9972 | ok | CLAUDE.md, PROJECT_STATUS.md, README.md |
| 7C, TF-IDF cosine | `T11.7c.tfidf_mean_cosine` | 0.2489 | ok | PROJECT_STATUS.md, README.md |
| 7C, pairs | `T11.7c.n_pairs` | 286 | ok | CLAUDE.md, PROJECT_STATUS.md, README.md |
| 7C, difference-in-differences (post-hoc) | `T11.7c.did_point` | -0.0804 | ok | CLAUDE.md, PROJECT_STATUS.md, README.md |
| 6A, gate coverage on benchmark45 | `T12.6a.gate_coverage.benchmark45` | 8.9 | ok | CLAUDE.md, PROJECT_OVERVIEW.md, PROJECT_STATUS.md, README.md |
| 6A, gate coverage on deployment175 | `T12.6a.gate_coverage.deployment175` | 18.9 | ok | CLAUDE.md, PROJECT_STATUS.md, README.md |
| drift eligible operating points | `T13.eligible_operating_points` | 25/68 | ok | PROJECT_STATUS.md |
| drift realistic traffic, alpha 0.05 | `T13.realistic_traffic.flag_rate.alpha0.05` | 0.429 | ok | CLAUDE.md, PROJECT_STATUS.md, README.md |
| drift realistic traffic, alpha 0.20 | `T13.realistic_traffic.flag_rate.alpha0.2` | 0.646 | ok | CLAUDE.md, PROJECT_STATUS.md, README.md |
| 2B, grounded drafts | `T14.grounded` | 31/33 | ok | PROJECT_OVERVIEW.md, PROJECT_STATUS.md, README.md |
| 2B, judge raw agreement | `T14.judge.raw_agreement` | 30/33 | ok | CLAUDE.md, PROJECT_OVERVIEW.md, PROJECT_STATUS.md, README.md |
| 2B, judge Cohen's kappa | `T14.judge.cohens_kappa` | -0.042 | ok | CLAUDE.md, PROJECT_OVERVIEW.md, PROJECT_STATUS.md, README.md |
| 6C, caught (case study) | `T15.caught_of_ungrounded` | 2/2 | ok | CLAUDE.md, PROJECT_STATUS.md, README.md |
| 6C, false flags | `T15.flagged_of_grounded` | 26/31 | ok | PROJECT_STATUS.md, README.md |
| 6C, false-flag proportion | `T15.false_flag_proportion` | 0.839 | ok | PROJECT_STATUS.md, README.md |
| clustering production threshold | `T16.production.threshold` | 0.80 | ok | CLAUDE.md, PROJECT_OVERVIEW.md, PROJECT_STATUS.md, README.md |
| clustering BGE pooled cliff | `T16.bge.pooled_cliff` | 0.90 | ok | CLAUDE.md, PROJECT_OVERVIEW.md, PROJECT_STATUS.md, README.md |
| 2A pilot false merges | `T16.pilot.false_merges` | 0/12 | ok | PROJECT_OVERVIEW.md, PROJECT_STATUS.md, README.md |

**0 anchored mismatch(es).**

---

## Pass B -- sweep

Every statistic-shaped token in the four documents -- decimals with three or more places, `k/n` counts, and one-decimal percentages -- classified against the emitted set. Prose numbers, dates, line counts and file sizes are not statistics, so an `unmatched` token is triage output, not a defect.

| document | verdict | occurrences |
|---|---|---|
| CLAUDE.md | DO-NOT-CITE literal | 5 |
| CLAUDE.md | matched to a committed source | 45 |
| CLAUDE.md | unmatched | 50 |
| PROJECT_OVERVIEW.md | DO-NOT-CITE literal | 1 |
| PROJECT_OVERVIEW.md | matched to a committed source | 79 |
| PROJECT_OVERVIEW.md | unmatched | 153 |
| PROJECT_STATUS.md | DO-NOT-CITE literal | 8 |
| PROJECT_STATUS.md | matched to a committed source | 244 |
| PROJECT_STATUS.md | unmatched | 280 |
| README.md | DO-NOT-CITE literal | 14 |
| README.md | matched to a committed source | 298 |
| README.md | unmatched | 582 |

### DO-NOT-CITE literals still present in the documents

These are expected in the lab notebook, which records superseded results on purpose. They must not appear in the paper.

| document | token | occurrences |
|---|---|---|
| CLAUDE.md | `0.6706` | 1 |
| CLAUDE.md | `0.8584` | 1 |
| CLAUDE.md | `0.9316` | 1 |
| CLAUDE.md | `40/4000` | 1 |
| CLAUDE.md | `47.6%` | 1 |
| PROJECT_OVERVIEW.md | `40/4000` | 1 |
| PROJECT_STATUS.md | `0.6706` | 1 |
| PROJECT_STATUS.md | `0.8584` | 3 |
| PROJECT_STATUS.md | `0.9316` | 1 |
| PROJECT_STATUS.md | `40/4000` | 1 |
| PROJECT_STATUS.md | `47.6%` | 2 |
| README.md | `0.6706` | 4 |
| README.md | `0.8584` | 4 |
| README.md | `0.9316` | 3 |
| README.md | `40/4000` | 1 |
| README.md | `47.6%` | 2 |

---

## Numbers with NO committed machine-readable source

Nothing below was invented, re-derived or re-run. Each is a decision about the paper.

| number | stated in the docs | where | why there is no source |
|---|---|---|---|
| Cascade calibration attempt 2: 34 of 35 hand-written tickets collapsed into one confidence bucket | 34/35 in one bucket | README.md, cascade calibration section | The 35-ticket hand-written calibration set was never committed and is absent from every revision in this repository's history, so the attempt cannot be re-run and its bucket counts cannot be regenerated. Phase 8A.1 sourced attempts 1 and 3 and recorded this one as status=no_artifact in data/cascade_calibration_attempts.csv rather than inventing it. |
