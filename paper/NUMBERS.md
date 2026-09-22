# NUMBERS.md -- every number the paper may use

**GENERATED. Do not edit by hand.** Rebuild with:

```powershell
python src/experiments/build_paper_artifacts.py --force
```

Phase 7A recorded two domain AUCs that were computed in an interactive session, quoted at a gate, and never committed. They do not reproduce. Every value below therefore names the committed file it came from and the command that regenerates that file. A number that is not in this table is not a number the paper may use.

Config fingerprint at build time: `9c9a5cbcb53f`.

Conventions: proportions carry a Wilson 95% interval computed with z = 1.96, except on pre-registered case-study axes, which report counts only. Paired comparisons on the same tickets use the EXACT McNemar test. A difference smaller than its own noise band is tagged `within-band` automatically.

Tags: `post-hoc` (not a pre-registered result), `BLOCKED` (a pre-registered rule fired; no verdict), `no-resolution` (the comparison could not resolve), `degenerate` (the test has no resolution at this operating point), `case-study` (counts only, by pre-registration), `within-band` (inside the measurement's own noise band).

---

## Clause groups -- numbers that may never be quoted alone

### Clause group `5C`

> PHASE 5C -- the correct sentence is: a 3B model on a laptop CPU with no training on this corpus is not beaten by a classifier fitted on 3,200 of its rows, across two vendors. NEVER 'LLMs beat the pipeline'. It is zero-shot LLM vs trained classifier, which is not like-for-like, and what it measures is what a template-generated corpus is worth on out-of-template phrasing. The readings differ by pair and must not be merged: Gemini is distinguishably better than Tier-2 on the 45; Qwen2.5-3B is INDISTINGUISHABLE from it, which is not 'better'. The benchmark is Gemini-generated, so the Qwen arm is the partial control for authorship.

| id | value | source file | regenerating command | tags |
|---|---|---|---|---|
| `T1.zeroshot_gemini.benchmark45` | 40/45 [95% CI 0.765009, 0.951595] | `data/zeroshot_summary_gemini_gemini-flash-lite-latest_benchmark45.csv` | `python src/experiments/run_ablation_study.py --mode baseline; python src/experiments/summarize_zeroshot_baselines.py` |  |
| `T1.zeroshot_gemini.benchmark14` | 14/14 | `data/zeroshot_summary_gemini_gemini-flash-lite-latest_benchmark14.csv` | `python src/experiments/run_ablation_study.py --mode baseline; python src/experiments/summarize_zeroshot_baselines.py` |  |
| `T1.zeroshot_qwen.benchmark45` | 34/45 [95% CI 0.613266, 0.857645] | `data/zeroshot_summary_ollama_qwen2-5-3b-instruct_benchmark45.csv` | `python src/experiments/run_ablation_study.py --mode baseline; python src/experiments/summarize_zeroshot_baselines.py` |  |
| `T1.zeroshot_qwen.benchmark14` | 12/14 | `data/zeroshot_summary_ollama_qwen2-5-3b-instruct_benchmark14.csv` | `python src/experiments/run_ablation_study.py --mode baseline; python src/experiments/summarize_zeroshot_baselines.py` |  |
| `T1.zeroshot_gemini.unparseable.benchmark45` | 0 | `data/zeroshot_summary_gemini_gemini-flash-lite-latest_benchmark45.csv` | `python src/experiments/run_ablation_study.py --mode baseline; python src/experiments/summarize_zeroshot_baselines.py` |  |
| `T1.zeroshot_qwen.unparseable.benchmark45` | 0 | `data/zeroshot_summary_ollama_qwen2-5-3b-instruct_benchmark45.csv` | `python src/experiments/run_ablation_study.py --mode baseline; python src/experiments/summarize_zeroshot_baselines.py` |  |
| `T1.mcnemar.gemini_vs_tier2.benchmark45` | 0.039062 | `data/zeroshot_summary_gemini_gemini-flash-lite-latest_benchmark45.csv` | `python src/experiments/run_ablation_study.py --mode baseline; python src/experiments/summarize_zeroshot_baselines.py` |  |
| | *Gemini is distinguishably better than the trained Tier-2 here.* | | | |
| `T1.mcnemar.gemini_vs_tier2.benchmark14` | 0.125 | `data/zeroshot_summary_gemini_gemini-flash-lite-latest_benchmark14.csv` | `python src/experiments/run_ablation_study.py --mode baseline; python src/experiments/summarize_zeroshot_baselines.py` |  |
| `T1.mcnemar.qwen_vs_tier2.benchmark45` | 1 | `data/zeroshot_summary_ollama_qwen2-5-3b-instruct_benchmark45.csv` | `python src/experiments/run_ablation_study.py --mode baseline; python src/experiments/summarize_zeroshot_baselines.py` |  |
| | *INDISTINGUISHABLE from Tier-2, which is not 'better'.* | | | |
| `T1.mcnemar.qwen_vs_tier2.benchmark14` | 0.6875 | `data/zeroshot_summary_ollama_qwen2-5-3b-instruct_benchmark14.csv` | `python src/experiments/run_ablation_study.py --mode baseline; python src/experiments/summarize_zeroshot_baselines.py` |  |
| `T1.mcnemar.qwen_vs_gemini.benchmark45` | 0.070312 | `data/zeroshot_summary_ollama_qwen2-5-3b-instruct_benchmark45.csv` | `python src/experiments/run_ablation_study.py --mode baseline; python src/experiments/summarize_zeroshot_baselines.py` |  |
| | *6 tickets apart and NOT significant; the benchmark is Gemini-generated, so the Qwen arm is the partial control for authorship.* | | | |
| `T1.mcnemar.qwen_vs_gemini.benchmark14` | 0.5 | `data/zeroshot_summary_ollama_qwen2-5-3b-instruct_benchmark14.csv` | `python src/experiments/run_ablation_study.py --mode baseline; python src/experiments/summarize_zeroshot_baselines.py` |  |
| `T2.qwen.database.recall.benchmark45` | 0/5 | `data/zeroshot_summary_ollama_qwen2-5-3b-instruct_benchmark45.csv` | `python src/experiments/summarize_zeroshot_baselines.py --backend ollama --model qwen2.5:3b-instruct --set both --prompt-check-against gemini` |  |
| | *Qwen2.5-3B predicted Database for no ticket at all; the precision cell is undefined, not zero.* | | | |
| `T2.qwen.database.recall.benchmark14` | 0/2 | `data/zeroshot_summary_ollama_qwen2-5-3b-instruct_benchmark14.csv` | `python src/experiments/summarize_zeroshot_baselines.py --backend ollama --model qwen2.5:3b-instruct --set both --prompt-check-against gemini` |  |
| `T2.qwen.application.precision.benchmark45` | 0.466667 | `data/zeroshot_summary_ollama_qwen2-5-3b-instruct_benchmark45.csv` | `python src/experiments/summarize_zeroshot_baselines.py --backend ollama --model qwen2.5:3b-instruct --set both --prompt-check-against gemini` |  |
| | *Everything Qwen could not place went to Application.* | | | |
| `T2.qwen.infrastructure.recall.benchmark45` | 0.5 | `data/zeroshot_summary_ollama_qwen2-5-3b-instruct_benchmark45.csv` | `python src/experiments/summarize_zeroshot_baselines.py --backend ollama --model qwen2.5:3b-instruct --set both --prompt-check-against gemini` |  |
| | *Infrastructure fails at ~50% for the trained model AND both LLMs -- a property of the register, not of any one method.* | | | |
| `T2.gemini.infrastructure.recall.benchmark45` | 0.5 | `data/zeroshot_summary_gemini_gemini-flash-lite-latest_benchmark45.csv` | `python src/experiments/summarize_zeroshot_baselines.py --backend ollama --model qwen2.5:3b-instruct --set both --prompt-check-against gemini` |  |
| `T2.prompt_identity.benchmark45` | 45 | `data/zeroshot_summary_ollama_qwen2-5-3b-instruct_benchmark45.csv` | `python src/experiments/summarize_zeroshot_baselines.py --backend ollama --model qwen2.5:3b-instruct --set both --prompt-check-against gemini` |  |
| | *Byte-identical prompts, verified by sha256 in both directions.* | | | |

### Clause group `6A`

> PHASE 6A -- the honest claim is 'no evidence either way on the gated axis', NEVER 'conformal is worse' on our data. All twelve comparisons returned no signal at the live gate's measured operating coverage, and three of four configurations are degenerate there. An AURC win is never a reason to promote: this project gates on risk at the operating coverage, and AURC averages over coverages the system never runs at.

| id | value | source file | regenerating command | tags |
|---|---|---|---|---|
| `T12.6a.comparisons_total` | 12 | `data/deferral_rule_summary.csv` | `python src/experiments/compare_deferral_rules.py; python src/experiments/compare_deferral_rules_external.py` |  |
| `T12.6a.comparisons_with_signal` | 0 | `data/deferral_rule_summary.csv` | `python src/experiments/compare_deferral_rules.py; python src/experiments/compare_deferral_rules_external.py` |  |
| | *ZERO. The honest claim is 'no evidence either way on the gated axis' -- never 'conformal is worse' on our data.* | | | |
| `T12.6a.degenerate_configurations` | 3/4 | `data/deferral_rule_summary.csv` | `python src/experiments/compare_deferral_rules.py; python src/experiments/compare_deferral_rules_external.py` | `degenerate` |
| | *Every rule accepts the same tickets at the live gate, so the test has no resolution there.* | | | |
| `T12.6a.gate_coverage.benchmark45` | 0.088889 | `data/deferral_rule_summary.csv` | `python src/experiments/compare_deferral_rules.py; python src/experiments/compare_deferral_rules_external.py` |  |
| `T12.6a.gate_coverage.deployment175` | 0.188571 | `data/deferral_rule_summary.csv` | `python src/experiments/compare_deferral_rules.py; python src/experiments/compare_deferral_rules_external.py` |  |
| `T12.6a.degenerate_rows` | 12 | `data/deferral_rule_summary.csv` | `python src/experiments/compare_deferral_rules.py; python src/experiments/compare_deferral_rules_external.py` | `degenerate` |

### Clause group `6B`

> PHASE 6B -- weighted conformal is a PARTIAL REPAIR, NOT A CORRECTION. Never write 'the shift is correctable by covariate reweighting'. Both readings are reported together: the CHANGE cleared the +/-2 s.d. band, and the RESIDUAL is still 5-6 s.d. below nominal. No ordered comparison between reweighting and distribution matching -- the residuals differ by less than the band. The BGE arm is BLOCKED, not a result.

| id | value | source file | regenerating command | tags |
|---|---|---|---|---|
| `T7.weighted.tier1.coverage_gap` | -0.122222 | `data/weighted_conformal_results.csv` | `python -m src.experiments.calibrate_conformal; python src/experiments/run_weighted_conformal.py` |  |
| | *A PARTIAL REPAIR. Never 'the shift is correctable by covariate reweighting'.* | | | |
| `T7.weighted.tier1.delta_vs_unweighted` | 0.111111 | `data/weighted_conformal_results.csv` | `python -m src.experiments.calibrate_conformal; python src/experiments/run_weighted_conformal.py` |  |
| | *READING 1 of the pre-registered clause: the CHANGE cleared the +/-2 s.d. band (true).* | | | |
| `T7.weighted.tier1.residual_in_sd` | -5.3895 | `data/weighted_conformal_results.csv` | `python -m src.experiments.calibrate_conformal; python src/experiments/run_weighted_conformal.py` |  |
| | *READING 2 of the same clause: the RESIDUAL is still this many s.d. below nominal. Both readings are reported together.* | | | |
| `T7.weighted.bge.domain_auc` | 0.990792 | `data/weighted_conformal_results.csv` | `python -m src.experiments.calibrate_conformal; python src/experiments/run_weighted_conformal.py` | `BLOCKED`, `degenerate` |
| | *The density ratio is ill-posed in this space. The BGE arm's flattering numbers are in the CSV as BLOCKED, not as a result. The 0.9908 is itself a statement of the named finding.* | | | |

### Clause group `6C`

> PHASE 6C -- the headline is 'catches both, and is still not usable as a gate', never 'the sufficiency check caught the two cases the gate missed', which is true and misleading alone. 6C is NOT an instance of the named finding; the near-duplicate explanation is a REJECTED hypothesis, refuted by 2B's own distinct-fix diagnostic, and the mechanism is a MEASUREMENT-TARGET MISMATCH between sufficiency and groundedness. The binding limitation is the LABEL, not the rater. The N45 counter-case travels with the result: the scalar errs in both directions.

| id | value | source file | regenerating command | tags |
|---|---|---|---|---|
| `T15.caught_of_ungrounded` | 2/2 | `data/sufficiency_gate_summary.json` | `python src/experiments/score_sufficiency_gate.py` | `case-study` |
| | *Caught BOTH. Never quote this without the false-flag count beside it.* | | | |
| `T15.flagged_of_grounded` | 26/31 [95% CI 0.673653, 0.929075] | `data/sufficiency_gate_summary.json` | `python src/experiments/score_sufficiency_gate.py` |  |
| | *The false-flag proportion, measured against 2B's groundedness labels -- which are an OUTCOME PROXY for draft support, not a direct sufficiency label. That is 6C's binding limitation.* | | | |
| `T15.false_flag_proportion` | 0.83871 | `data/sufficiency_gate_summary.json` | `python src/experiments/score_sufficiency_gate.py` |  |
| `T15.escalated_of_eligible` | 28/33 | `data/sufficiency_gate_summary.json` | `python src/experiments/score_sufficiency_gate.py` |  |
| | *A gate escalating this many of the tickets that currently reach the resolver suppresses ~85% of auto-resolution to recover two bad drafts, and nothing can be tuned.* | | | |
| `T15.population.eligible` | 33 | `data/sufficiency_gate_summary.json` | `python src/experiments/score_sufficiency_gate.py` |  |
| `T15.population.escalated` | 21 | `data/sufficiency_gate_summary.json` | `python src/experiments/score_sufficiency_gate.py` |  |

### Clause group `7A-control`

> PHASE 7A's CONTROL LESSON -- never quote a redundancy or similarity rate without stating what it is high RELATIVE TO. Our own corpus is more near-duplicated than the external one. High near-duplication is a property of template-generated corpora generally, not a flaw of that dataset.

| id | value | source file | regenerating command | tags |
|---|---|---|---|---|
| `T11.7a.near_duplicate_rate_external` | 0.793921 | `data/external_tobibueck/profile_summary.json` | `python src/experiments/profile_external_dataset.py; python src/experiments/run_external_conformal_shift.py; python src/experiments/run_paraphrase_shift_conformal.py` |  |
| `T11.7a.near_duplicate_rate_our_corpus` | 0.852 | `data/external_tobibueck/profile_summary.json` | `python src/experiments/profile_external_dataset.py; python src/experiments/run_external_conformal_shift.py; python src/experiments/run_paraphrase_shift_conformal.py` |  |
| | *OUR corpus is MORE near-duplicated than the external one. A redundancy rate is meaningless without saying what it is high relative to.* | | | |

### Clause group `finding1`

> FINDING 1 -- the three clauses travel together, and none is quoted alone:
>   (1) MEASURED ON OUR CORPUS: split-conformal coverage transfers for the BGE representation but not the lexical one.
>   (2) EXTERNAL REPLICATION IS INCONCLUSIVE: Phase 7B applied a VERSION shift -- the wrong shift type for a mechanism that is surface-vocabulary change -- and Phase 7C applied the right type but was BLOCKED by its own pre-registered degeneracy rule. After 7B and 7C, Finding 1 has been shown neither to hold nor to fail outside its original corpus. Never write '7B refutes Finding 1'.
>   (3) POST-HOC EVIDENCE OF THE MECHANISM IN ACCURACY on the external corpus: difference-in-differences -0.0804, 95% CI [-0.1364, -0.0210]. Post-hoc, and it does not replace the blocked primary.

| id | value | source file | regenerating command | tags |
|---|---|---|---|---|
| `T7.tier1.coverage_gap` | -0.233333 | `data/conformal_calibration_results.csv` | `python -m src.experiments.calibrate_conformal; python src/experiments/run_weighted_conformal.py` |  |
| | *TF-IDF loses this much coverage moving from the calibration set to the benchmark.* | | | |
| `T7.tier2.coverage_gap` | -0.011111 | `data/conformal_calibration_results.csv` | `python -m src.experiments.calibrate_conformal; python src/experiments/run_weighted_conformal.py` |  |
| `T7.noise_band_2sd` | 0.045356 | `data/conformal_calibration_results.csv` | `python -m src.experiments.calibrate_conformal; python src/experiments/run_weighted_conformal.py` |  |
| | *+/-2 s.d. on the calibration draw at n=175.* | | | |
| `T7.tier1.gap_in_sd` | -10.289 | `data/conformal_calibration_results.csv` | `python -m src.experiments.calibrate_conformal; python src/experiments/run_weighted_conformal.py` |  |
| `T7.tier2.gap_in_sd` | -0.489954 | `data/conformal_calibration_results.csv` | `python -m src.experiments.calibrate_conformal; python src/experiments/run_weighted_conformal.py` | `within-band` |
| `T11.7b.tier1.coverage_gap` | 0.000872 | `data/external_tobibueck/external_conformal_designA.csv` | `python src/experiments/profile_external_dataset.py; python src/experiments/run_external_conformal_shift.py; python src/experiments/run_paraphrase_shift_conformal.py` |  |
| | *Under a VERSION shift, Tier-1 transfers as well as Tier-2.* | | | |
| `T11.7b.tier2.coverage_gap` | -0.007174 | `data/external_tobibueck/external_conformal_designA.csv` | `python src/experiments/profile_external_dataset.py; python src/experiments/run_external_conformal_shift.py; python src/experiments/run_paraphrase_shift_conformal.py` | `within-band` |
| `T11.7b.noise_band_2sd` | 0.016471 | `data/external_tobibueck/external_conformal_designA.csv` | `python src/experiments/profile_external_dataset.py; python src/experiments/run_external_conformal_shift.py; python src/experiments/run_paraphrase_shift_conformal.py` |  |
| `T11.7b.rows_inside_band` | 5/6 | `data/external_tobibueck/external_conformal_designA.csv` | `python src/experiments/profile_external_dataset.py; python src/experiments/run_external_conformal_shift.py; python src/experiments/run_paraphrase_shift_conformal.py` |  |
| `T11.7c.verdict` | blocked_auc_ge_0.95 | `data/external_tobibueck/external_paraphrase_summary.json` | `python src/experiments/profile_external_dataset.py; python src/experiments/run_external_conformal_shift.py; python src/experiments/run_paraphrase_shift_conformal.py` | `BLOCKED` |
| | *The pre-registered degeneracy rule fired, so NO verdict is drawn on the primary. A blocked arm is not a null.* | | | |
| `T11.7c.tfidf_domain_auc` | 0.997194 | `data/external_tobibueck/external_paraphrase_summary.json` | `python src/experiments/profile_external_dataset.py; python src/experiments/run_external_conformal_shift.py; python src/experiments/run_paraphrase_shift_conformal.py` | `BLOCKED` |
| `T11.7c.accuracy.tier1_change` | -0.104895 | `data/external_tobibueck/external_paraphrase_accuracy_did.json` | `python src/experiments/profile_external_dataset.py; python src/experiments/run_external_conformal_shift.py; python src/experiments/run_paraphrase_shift_conformal.py` | `post-hoc` |
| | *The mechanism IS visible in accuracy even though the coverage primary is blocked.* | | | |
| `T11.7c.accuracy.tier2_change` | -0.024476 | `data/external_tobibueck/external_paraphrase_accuracy_did.json` | `python src/experiments/profile_external_dataset.py; python src/experiments/run_external_conformal_shift.py; python src/experiments/run_paraphrase_shift_conformal.py` | `post-hoc` |
| `T11.7c.did_point` | -0.08042 [95% CI -0.136364, -0.020979] | `data/external_tobibueck/external_paraphrase_accuracy_did.json` | `python src/experiments/profile_external_dataset.py; python src/experiments/run_external_conformal_shift.py; python src/experiments/run_paraphrase_shift_conformal.py` | `post-hoc` |
| | *10,000 paired bootstrap draws, seed 42. Excludes zero. Still POST-HOC -- it does not replace the blocked primary.* | | | |
| `T11.7c.tier1.coverage_gap_posthoc` | -0.025874 | `data/external_tobibueck/external_paraphrase_conformal.csv` | `python src/experiments/profile_external_dataset.py; python src/experiments/run_external_conformal_shift.py; python src/experiments/run_paraphrase_shift_conformal.py` | `BLOCKED`, `post-hoc` |
| | *POST-HOC only. Never quote 7C coverage as a verdict.* | | | |

### Clause group `finding2`

> FINDING 2 -- the 175-ticket calibration set cannot be de-contaminated. Memorisation is TEMPLATE-level, and a template is (category, scenario_id), never scenario_id alone. The pre-Phase-5A diagnostic is superseded and appears only in the do-not-cite list; Finding 2's conclusion survived the correction because the coverage measurement excludes by row id, not by template.

| id | value | source file | regenerating command | tags |
|---|---|---|---|---|
| `T8.templates_total` | 66 | `data/conformal_calibration_corrected_bge-base-en-v1-5.json` | `python -m src.experiments.calibrate_conformal` |  |
| | *A template is (category, scenario_id). scenario_id alone is unique only WITHIN a category.* | | | |
| `T8.templates_touched` | 62 | `data/conformal_calibration_corrected_bge-base-en-v1-5.json` | `python -m src.experiments.calibrate_conformal` |  |
| `T8.median_rows_per_template` | 62 | `data/conformal_calibration_corrected_bge-base-en-v1-5.json` | `python -m src.experiments.calibrate_conformal` |  |
| `T8.rows_surviving_template_exclusion` | 210 | `data/conformal_calibration_corrected_bge-base-en-v1-5.json` | `python -m src.experiments.calibrate_conformal` |  |
| | *Removing whole templates would leave this many of 4000 rows -- the set cannot be de-contaminated by filtering.* | | | |
| `T8.dataset_rows` | 4000 | `data/conformal_calibration_corrected_bge-base-en-v1-5.json` | `python -m src.experiments.calibrate_conformal` |  |
| `T8.max_gap_movement_clean_vs_contaminated` | 0.022222 | `data/conformal_calibration_results.csv` | `python -m src.experiments.calibrate_conformal` |  |
| | *Excluding contaminated rows moves the coverage gap by at most this much anywhere on the grid -- Finding 2's conclusion is unchanged by the 5A correction, and that measurement excludes by row id, not by template.* | | | |

### Clause group `named-finding`

> THE NAMED CROSS-PHASE FINDING -- the calibration/reference distribution, not the test or the method, is the binding constraint. Instances: Phase 1 Finding 4 (coverage), Phase 4B's realistic-traffic arm (drift), Phase 5C (classification, which widened it to the TRAINING distribution as well). Phase 2A is a RELATED corpus limitation, not an instance. Phase 6C is NOT an instance. Phase 9A settles the final wording.

| id | value | source file | regenerating command | tags |
|---|---|---|---|---|
| `T10.tier1.gap_in_domain` | -0.233333 | `data/conformal_calibration_results.csv` | `python -m src.experiments.calibrate_conformal` |  |
| `T10.tier1.gap_deployment` | -0.144444 | `data/conformal_calibration_results.csv` | `python -m src.experiments.calibrate_conformal` |  |
| | *Same method, same alpha, same benchmark -- only the calibration DISTRIBUTION changed.* | | | |

---

## Ungrouped numbers, by table and figure

### F1

| id | value | source file | regenerating command | tags |
|---|---|---|---|---|
| `F1.gate.cascade` | 0.5 | `src/agent/config.py` | `python src/experiments/build_paper_artifacts.py` |  |
| `F1.gate.rag` | 0.67 | `src/agent/config.py` | `python src/experiments/build_paper_artifacts.py` |  |
| `F1.gate.clustering` | 0.8 | `src/agent/config.py` | `python src/experiments/build_paper_artifacts.py` |  |

### T1

| id | value | source file | regenerating command | tags |
|---|---|---|---|---|
| `T1.cascade.benchmark45` | 32/45 [95% CI 0.566312, 0.822701] | `data/ablation_baseline_results.csv` | `python src/experiments/run_ablation_study.py --mode baseline; python src/experiments/summarize_zeroshot_baselines.py` |  |
| `T1.cascade.benchmark14` | 10/14 | `data/ablation_baseline_results_benchmark14.csv` | `python src/experiments/run_ablation_study.py --mode baseline; python src/experiments/summarize_zeroshot_baselines.py` |  |
| `T1.cascade.deployment175` | 131/175 [95% CI 0.679425, 0.807039] | `data/ablation_baseline_results_deployment175.csv` | `python src/experiments/run_ablation_study.py --mode baseline; python src/experiments/summarize_zeroshot_baselines.py` |  |
| `T1.tier2only.benchmark45` | 33/45 [95% CI 0.589609, 0.840353] | `data/ablation_tier2-only_results.csv` | `python src/experiments/run_ablation_study.py --mode baseline; python src/experiments/summarize_zeroshot_baselines.py` |  |
| `T1.tier2only.benchmark14` | 10/14 | `data/ablation_tier2-only_results_benchmark14.csv` | `python src/experiments/run_ablation_study.py --mode baseline; python src/experiments/summarize_zeroshot_baselines.py` |  |
| `T1.tier2only.deployment175` | 132/175 [95% CI 0.685491, 0.812156] | `data/ablation_tier2-only_results_deployment175.csv` | `python src/experiments/run_ablation_study.py --mode baseline; python src/experiments/summarize_zeroshot_baselines.py` |  |
| `T1.tier1only.benchmark45` | 16/45 [95% CI 0.232189, 0.501645] | `data/ablation_no-cascade_results.csv` | `python src/experiments/run_ablation_study.py --mode baseline; python src/experiments/summarize_zeroshot_baselines.py` |  |
| | *Tier-1 answering everything -- the TF-IDF representation, NOT a measure of what cascading is worth.* | | | |
| `T1.tier1only.deployment175` | 91/175 [95% CI 0.446347, 0.592794] | `data/ablation_no-cascade_results_deployment175.csv` | `python src/experiments/run_ablation_study.py --mode baseline; python src/experiments/summarize_zeroshot_baselines.py` |  |
| `T1.minilm.benchmark45` | 32/45 | `data/embedding_comparison/embedding_model_comparison.csv` | `python src/experiments/run_ablation_study.py --mode baseline; python src/experiments/summarize_zeroshot_baselines.py` |  |
| `T1.e5.benchmark45` | 27/45 | `data/embedding_comparison/embedding_model_comparison.csv` | `python src/experiments/run_ablation_study.py --mode baseline; python src/experiments/summarize_zeroshot_baselines.py` |  |
| `T1.mcnemar.cascade_vs_tier2.benchmark45` | 1 | `data/cascade_vs_tier2_mcnemar_benchmark45.csv` | `python src/experiments/run_ablation_study.py --mode baseline; python src/experiments/summarize_zeroshot_baselines.py` |  |
| | *The cascade is one ticket WORSE and indistinguishable.* | | | |
| `T1.mcnemar.cascade_vs_tier2.deployment175` | 1 | `data/cascade_vs_tier2_mcnemar_deployment175.csv` | `python src/experiments/run_ablation_study.py --mode baseline; python src/experiments/summarize_zeroshot_baselines.py` |  |

### T3

| id | value | source file | regenerating command | tags |
|---|---|---|---|---|
| `T3.repr_gap_points` | 35.5556 | `data/ablation_baseline_results.csv` | `python src/experiments/run_ablation_study.py --mode {baseline,no-cascade,tier2-only,no-rag}; python src/experiments/compare_cascade_vs_tier2.py; python src/experiments/measure_inference_latency.py` |  |
| | *baseline MINUS Tier-1-only. This is the BGE-vs-TF-IDF REPRESENTATION gap. It is NOT what cascading is worth -- see the do-not-cite list.* | | | |
| `T3.cascade_delta.benchmark45` | -1 | `data/ablation_baseline_results.csv` | `python src/experiments/run_ablation_study.py --mode {baseline,no-cascade,tier2-only,no-rag}; python src/experiments/compare_cascade_vs_tier2.py; python src/experiments/measure_inference_latency.py` |  |
| | *The cascade is one ticket worse than the Tier-2-only control.* | | | |
| `T3.cascade_delta.deployment175` | -1 | `data/ablation_baseline_results_deployment175.csv` | `python src/experiments/run_ablation_study.py --mode {baseline,no-cascade,tier2-only,no-rag}; python src/experiments/compare_cascade_vs_tier2.py; python src/experiments/measure_inference_latency.py` |  |
| `T3.mcnemar.benchmark45` | 1 | `data/cascade_vs_tier2_mcnemar_benchmark45.csv` | `python src/experiments/run_ablation_study.py --mode {baseline,no-cascade,tier2-only,no-rag}; python src/experiments/compare_cascade_vs_tier2.py; python src/experiments/measure_inference_latency.py` |  |
| `T3.mcnemar.deployment175` | 1 | `data/cascade_vs_tier2_mcnemar_deployment175.csv` | `python src/experiments/run_ablation_study.py --mode {baseline,no-cascade,tier2-only,no-rag}; python src/experiments/compare_cascade_vs_tier2.py; python src/experiments/measure_inference_latency.py` |  |
| `T3.latency.tier1_median_ms` | 1.0356 | `data/inference_latency_13th-gen-intel-r-core-tm-i5-1334u.csv` | `python src/experiments/run_ablation_study.py --mode {baseline,no-cascade,tier2-only,no-rag}; python src/experiments/compare_cascade_vs_tier2.py; python src/experiments/measure_inference_latency.py` |  |
| `T3.latency.tier2_median_ms` | 156.4024 | `data/inference_latency_13th-gen-intel-r-core-tm-i5-1334u.csv` | `python src/experiments/run_ablation_study.py --mode {baseline,no-cascade,tier2-only,no-rag}; python src/experiments/compare_cascade_vs_tier2.py; python src/experiments/measure_inference_latency.py` |  |
| `T3.latency.tier1_p95_ms` | 1.7392 | `data/inference_latency_13th-gen-intel-r-core-tm-i5-1334u.csv` | `python src/experiments/run_ablation_study.py --mode {baseline,no-cascade,tier2-only,no-rag}; python src/experiments/compare_cascade_vs_tier2.py; python src/experiments/measure_inference_latency.py` |  |
| `T3.latency.tier2_p95_ms` | 227.0484 | `data/inference_latency_13th-gen-intel-r-core-tm-i5-1334u.csv` | `python src/experiments/run_ablation_study.py --mode {baseline,no-cascade,tier2-only,no-rag}; python src/experiments/compare_cascade_vs_tier2.py; python src/experiments/measure_inference_latency.py` |  |
| `T3.latency.tier2_over_tier1` | 151.0259 | `data/inference_latency_13th-gen-intel-r-core-tm-i5-1334u.csv` | `python src/experiments/run_ablation_study.py --mode {baseline,no-cascade,tier2-only,no-rag}; python src/experiments/compare_cascade_vs_tier2.py; python src/experiments/measure_inference_latency.py` |  |
| | *Warm, batch size 1, on the recorded CPU.* | | | |
| `T3.latency.saving.benchmark45` | 0.082268 | `data/inference_latency_13th-gen-intel-r-core-tm-i5-1334u.csv` | `python src/experiments/run_ablation_study.py --mode {baseline,no-cascade,tier2-only,no-rag}; python src/experiments/compare_cascade_vs_tier2.py; python src/experiments/measure_inference_latency.py` |  |
| | *This -- not accuracy -- is what the cascade buys.* | | | |
| `T3.latency.saving.deployment175` | 0.181951 | `data/inference_latency_13th-gen-intel-r-core-tm-i5-1334u.csv` | `python src/experiments/run_ablation_study.py --mode {baseline,no-cascade,tier2-only,no-rag}; python src/experiments/compare_cascade_vs_tier2.py; python src/experiments/measure_inference_latency.py` |  |
| `T3.norag.escalation_correct` | 3/9 | `data/ablation_no-rag_results.csv` | `python src/experiments/run_ablation_study.py --mode {baseline,no-cascade,tier2-only,no-rag}; python src/experiments/compare_cascade_vs_tier2.py; python src/experiments/measure_inference_latency.py` |  |
| | *Removing the RAG gate costs the adversarial set; the baseline scores 6/9 there.* | | | |
| `T3.baseline.escalation_correct` | 6/9 | `data/ablation_baseline_results.csv` | `python src/experiments/run_ablation_study.py --mode {baseline,no-cascade,tier2-only,no-rag}; python src/experiments/compare_cascade_vs_tier2.py; python src/experiments/measure_inference_latency.py` |  |

### T4

| id | value | source file | regenerating command | tags |
|---|---|---|---|---|
| `T4.tier1.ece` | 0.112214 | `data/calibration_reliability_data.csv` | `python src/experiments/plot_calibration_curves.py` |  |
| | *Count-weighted binned ECE, recomputed from the committed bins, on the 500-ticket IN-DISTRIBUTION production batch.* | | | |
| `T4.tier2.ece` | 0.099248 | `data/calibration_reliability_data.csv` | `python src/experiments/plot_calibration_curves.py` |  |
| `T4.observed_accuracy_every_bin` | 1 | `data/calibration_reliability_data.csv` | `python src/experiments/plot_calibration_curves.py` |  |
| | *The MINIMUM observed accuracy across all bins of both tiers. It is 1.0, so both tiers are systematically UNDER-confident here and the ECE is entirely the distance to a ceiling. In-distribution accuracy is uninformative on template-generated data -- treat a new 100% as a red flag.* | | | |
| `T4.cascade.threshold` | 0.5 | `src/agent/config.py` | `python src/experiments/plot_calibration_curves.py` |  |
| | *The live gate. Frozen; Phases 5-9 are measurement only.* | | | |
| `T4.reliability.n_tickets` | 500 | `data/calibration_reliability_data.csv` | `python src/experiments/plot_calibration_curves.py` |  |

### T5

| id | value | source file | regenerating command | tags |
|---|---|---|---|---|
| `T5.gate` | 0.67 | `src/agent/config.py` | `python -m src.experiments.calibrate_rag_similarity_threshold` |  |
| `T5.ood_leakage_at_gate` | 0.133333 | `data/rag_similarity_calibration_combined.csv` | `python -m src.experiments.calibrate_rag_similarity_threshold` |  |
| `T5.ood_leakage_below_gate` | 0.377778 | `data/rag_similarity_calibration_combined.csv` | `python -m src.experiments.calibrate_rag_similarity_threshold` |  |
| | *At threshold 0.65 -- dropping the gate two steps nearly triples OOD leakage.* | | | |
| `T5.in_domain_proceed_at_gate` | 175/175 | `data/rag_similarity_calibration_combined.csv` | `python -m src.experiments.calibrate_rag_similarity_threshold` |  |
| `T5.safe_range_low` | 0.619577 | `data/ablation_no-rag_results.csv` | `python -m src.experiments.calibrate_rag_similarity_threshold` |  |
| | *Highest similarity among the adversarial tickets that MUST escalate.* | | | |
| `T5.safe_range_high` | 0.670427 | `data/ablation_no-rag_results.csv` | `python -m src.experiments.calibrate_rag_similarity_threshold` |  |
| | *Lowest similarity among those that must proceed. The gate at 0.67 sits inside this band, and it errs in BOTH directions -- see FRAMING.md.* | | | |

### T6

| id | value | source file | regenerating command | tags |
|---|---|---|---|---|
| `T6.skew.min_am_rows` | 50 | `data/skewed/imbalance_sweep_results.csv` | `python src/experiments/run_imbalance_sweep.py; python src/experiments/process_ticket_batch.py` |  |
| `T6.skew.max_am_rows` | 571 | `data/skewed/imbalance_sweep_results.csv` | `python src/experiments/run_imbalance_sweep.py; python src/experiments/process_ticket_batch.py` |  |
| `T6.skew.heldout_acc_at_min_rows` | 0.9957 | `data/skewed/imbalance_sweep_results.csv` | `python src/experiments/run_imbalance_sweep.py; python src/experiments/process_ticket_batch.py` |  |
| | *In-distribution accuracy stays ~1.0 across the whole sweep -- template-generated data makes it uninformative.* | | | |
| `T6.skew.cascade_safety_net_failures_total` | 0 | `data/skewed/imbalance_sweep_results.csv` | `python src/experiments/run_imbalance_sweep.py; python src/experiments/process_ticket_batch.py` |  |
| `T6.batch.volume_total` | 500 | `data/batch_intake/batch_summary.csv` | `python src/experiments/run_imbalance_sweep.py; python src/experiments/process_ticket_batch.py` |  |
| `T6.batch.auto_resolved_total` | 485 | `data/batch_intake/batch_summary.csv` | `python src/experiments/run_imbalance_sweep.py; python src/experiments/process_ticket_batch.py` |  |
| `T6.batch.gemini_failed_total` | 15 | `data/batch_intake/batch_summary.csv` | `python src/experiments/run_imbalance_sweep.py; python src/experiments/process_ticket_batch.py` |  |
| | *Produced under MiniLM. Re-running under BGE would silently invalidate the 0.80 clustering calibration -- see PROJECT_STATUS.md, Known risks.* | | | |

### T7

| id | value | source file | regenerating command | tags |
|---|---|---|---|---|
| `T7.alpha` | 0.1 | `src/agent/config.py` | `python -m src.experiments.calibrate_conformal; python src/experiments/run_weighted_conformal.py` |  |
| | *settings.conformal.enabled is False -- measurement only, and it gates nothing in production.* | | | |
| `T7.tier1.mean_set_size` | 3.4 | `data/conformal_calibration_results.csv` | `python -m src.experiments.calibrate_conformal; python src/experiments/run_weighted_conformal.py` |  |
| `T7.tier2.mean_set_size` | 1.8222 | `data/conformal_calibration_results.csv` | `python -m src.experiments.calibrate_conformal; python src/experiments/run_weighted_conformal.py` |  |
| `T7.tier1.singleton_rate` | 0.088889 | `data/conformal_calibration_results.csv` | `python -m src.experiments.calibrate_conformal; python src/experiments/run_weighted_conformal.py` |  |
| `T7.tier2.singleton_rate` | 0.377778 | `data/conformal_calibration_results.csv` | `python -m src.experiments.calibrate_conformal; python src/experiments/run_weighted_conformal.py` |  |

### T9

| id | value | source file | regenerating command | tags |
|---|---|---|---|---|
| `T9.false_escalation_in_domain` | 0.091429 | `data/conformal_novelty_results.csv` | `python -m src.experiments.calibrate_conformal` |  |
| | *Tracks alpha by construction -- the guarantee is marginal over the calibration draw, not conditional on it.* | | | |
| `T9.ood_detection_seed_level` | 1 | `data/conformal_novelty_results.csv` | `python -m src.experiments.calibrate_conformal` |  |
| `T9.ood_detection_variant_level` | 0.977778 | `data/conformal_novelty_results.csv` | `python -m src.experiments.calibrate_conformal` |  |
| `T9.adversarial_flagged` | 9/9 | `data/conformal_novelty_results.csv` | `python -m src.experiments.calibrate_conformal` |  |
| `T9.min_attainable_p` | 0.005682 | `data/conformal_novelty_results.csv` | `python -m src.experiments.calibrate_conformal` |  |
| | *1/(n_cal+1) at n=175. No p-value can go below it, which is why the KS test reads discreteness rather than drift.* | | | |

### T10

| id | value | source file | regenerating command | tags |
|---|---|---|---|---|
| `T10.deployment.n_calibration` | 175 | `data/conformal_calibration_results.csv` | `python -m src.experiments.calibrate_conformal` |  |

### T11

| id | value | source file | regenerating command | tags |
|---|---|---|---|---|
| `T11.7a.rows_english` | 28261 | `data/external_tobibueck/profile_summary.json` | `python src/experiments/profile_external_dataset.py; python src/experiments/run_external_conformal_shift.py; python src/experiments/run_paraphrase_shift_conformal.py` |  |
| `T11.7a.distinct_texts` | 23747 | `data/external_tobibueck/profile_summary.json` | `python src/experiments/profile_external_dataset.py; python src/experiments/run_external_conformal_shift.py; python src/experiments/run_paraphrase_shift_conformal.py` |  |
| `T11.7a.is_real_production_data` | false -- independently generated data, a different generator, not reality | `data/external_tobibueck/profile_summary.json` | `python src/experiments/profile_external_dataset.py; python src/experiments/run_external_conformal_shift.py; python src/experiments/run_paraphrase_shift_conformal.py` |  |
| `T11.7b.n_test` | 10441 | `data/external_tobibueck/external_conformal_designA.csv` | `python src/experiments/profile_external_dataset.py; python src/experiments/run_external_conformal_shift.py; python src/experiments/run_paraphrase_shift_conformal.py` |  |
| `T11.7b.n_calibration` | 1327 | `data/external_tobibueck/external_conformal_designA.csv` | `python src/experiments/profile_external_dataset.py; python src/experiments/run_external_conformal_shift.py; python src/experiments/run_paraphrase_shift_conformal.py` |  |
| `T11.7b.domain_auc_operating` | 0.847174 | `data/external_tobibueck/external_conformal_designA.csv` | `python src/experiments/profile_external_dataset.py; python src/experiments/run_external_conformal_shift.py; python src/experiments/run_paraphrase_shift_conformal.py` |  |
| | *Replaces the two uncommitted 7A figures -- see the do-not-cite list.* | | | |
| `T11.7b.boundary_contamination_rate` | 0.014558 | `data/external_tobibueck/external_contamination.json` | `python src/experiments/profile_external_dataset.py; python src/experiments/run_external_conformal_shift.py; python src/experiments/run_paraphrase_shift_conformal.py` |  |
| | *Low because the reference set is 3305 rows, not because the corpus changed.* | | | |
| `T11.7b.label_ceiling_tier1` | 0.347572 | `data/external_tobibueck/external_label_noise.json` | `python src/experiments/profile_external_dataset.py; python src/experiments/run_external_conformal_shift.py; python src/experiments/run_paraphrase_shift_conformal.py` |  |
| | *Generator-assigned, UNAUDITED labels. Tier-1 and Tier-2 are only ~2.6 points apart here, against 35.6 on our corpus -- there is little representation gap for 7B to find, which is a limitation of 7B as evidence, not a rescue of Finding 1.* | | | |
| `T11.7b.label_ceiling_tier2` | 0.37324 | `data/external_tobibueck/external_label_noise.json` | `python src/experiments/profile_external_dataset.py; python src/experiments/run_external_conformal_shift.py; python src/experiments/run_paraphrase_shift_conformal.py` |  |
| `T11.7b.designB.blocked_queue` | Billing and Payments | `data/external_tobibueck/external_conformal_designB.csv` | `python src/experiments/profile_external_dataset.py; python src/experiments/run_external_conformal_shift.py; python src/experiments/run_paraphrase_shift_conformal.py` | `BLOCKED`, `no-resolution` |
| | *Design B resolved nothing, twice, for two named reasons; no Design B number may be a headline.* | | | |
| `T11.7b.designB.single_class_rows` | 54 | `data/external_tobibueck/external_conformal_designB.csv` | `python src/experiments/profile_external_dataset.py; python src/experiments/run_external_conformal_shift.py; python src/experiments/run_paraphrase_shift_conformal.py` | `no-resolution` |
| `T11.7b.designB.blocked_rows` | 6 | `data/external_tobibueck/external_conformal_designB.csv` | `python src/experiments/profile_external_dataset.py; python src/experiments/run_external_conformal_shift.py; python src/experiments/run_paraphrase_shift_conformal.py` | `BLOCKED`, `no-resolution` |
| `T11.7c.tfidf_mean_cosine` | 0.248875 | `data/external_tobibueck/external_paraphrase_summary.json` | `python src/experiments/profile_external_dataset.py; python src/experiments/run_external_conformal_shift.py; python src/experiments/run_paraphrase_shift_conformal.py` |  |
| | *The manipulation WORKED -- a near-1.0 AUC here means it was strong, which is why the rule was mis-specified for this design. It was kept anyway; see FRAMING.md.* | | | |
| `T11.7c.bge_mean_cosine` | 0.846181 | `data/external_tobibueck/external_paraphrase_summary.json` | `python src/experiments/profile_external_dataset.py; python src/experiments/run_external_conformal_shift.py; python src/experiments/run_paraphrase_shift_conformal.py` |  |
| `T11.7c.n_pairs` | 286 | `data/external_tobibueck/external_paraphrase_summary.json` | `python src/experiments/profile_external_dataset.py; python src/experiments/run_external_conformal_shift.py; python src/experiments/run_paraphrase_shift_conformal.py` |  |
| `T11.7c.calibration_only_band_2sd` | 0.016471 | `data/external_tobibueck/external_paraphrase_conformal.csv` | `python src/experiments/profile_external_dataset.py; python src/experiments/run_external_conformal_shift.py; python src/experiments/run_paraphrase_shift_conformal.py` |  |
| | *Quoting this alone on a 286-ticket test arm understates uncertainty by roughly 2x.* | | | |
| `T11.7c.combined_band_2sd` | 0.039116 | `data/external_tobibueck/external_paraphrase_conformal.csv` | `python src/experiments/profile_external_dataset.py; python src/experiments/run_external_conformal_shift.py; python src/experiments/run_paraphrase_shift_conformal.py` |  |
| | *Calibration term AND test-sampling term. This is the band 7C reports.* | | | |

### T12

| id | value | source file | regenerating command | tags |
|---|---|---|---|---|
| `T12.7b.gate_coverage` | 0.061488 | `data/external_tobibueck/external_deferral_results.csv` | `python src/experiments/compare_deferral_rules.py; python src/experiments/compare_deferral_rules_external.py` |  |
| | *The gated axis RESOLVES here -- hundreds of tickets at the gate against 6A's four.* | | | |
| `T12.7b.n_at_gate` | 642 | `data/external_tobibueck/external_deferral_results.csv` | `python src/experiments/compare_deferral_rules.py; python src/experiments/compare_deferral_rules_external.py` |  |
| `T12.7b.risk_signals` | 3/6 | `data/external_tobibueck/external_deferral_results.csv` | `python src/experiments/compare_deferral_rules.py; python src/experiments/compare_deferral_rules_external.py` |  |
| | *All favouring the confidence incumbent -- on THAT corpus, with a transplanted gate and ~35% label accuracy. This must never be merged with 6A's reading.* | | | |

### T13

| id | value | source file | regenerating command | tags |
|---|---|---|---|---|
| `T13.eligible_operating_points` | 25/68 | `data/drift_evaluation_summary.json` | `python src/experiments/evaluate_drift_detection.py` |  |
| | *Verdict: MEASURED, NOT SHIPPED. Nothing was promoted into config; settings.drift.enabled stays False.* | | | |
| `T13.conditional_binomial.null_rate_min` | 0.00915 | `data/drift_evaluation_null.csv` | `python src/experiments/evaluate_drift_detection.py` |  |
| | *Holds at every window -- the only Signal A test that does.* | | | |
| `T13.conditional_binomial.null_rate_max` | 0.02255 | `data/drift_evaluation_null.csv` | `python src/experiments/evaluate_drift_detection.py` |  |
| `T13.marginal_binomial.null_rate_at_w200_min` | 0.0773 | `data/drift_evaluation_null.csv` | `python src/experiments/evaluate_drift_detection.py` |  |
| | *Against a nominal 0.05 -- usable only at W <= 50.* | | | |
| `T13.marginal_binomial.null_rate_at_w200_max` | 0.12525 | `data/drift_evaluation_null.csv` | `python src/experiments/evaluate_drift_detection.py` |  |
| `T13.ks.null_rate_at_w25` | 0.0747 | `data/drift_evaluation_null.csv` | `python src/experiments/evaluate_drift_detection.py` |  |
| | *The KS test reads the p-values' 1/176 discreteness, not drift. Descriptive read only -- never an alarm.* | | | |
| `T13.ks.null_rate_at_w200` | 0.33915 | `data/drift_evaluation_null.csv` | `python src/experiments/evaluate_drift_detection.py` |  |
| `T13.realistic_traffic.flag_rate.alpha0.01` | 0.217143 | `data/drift_evaluation_summary.json` | `python src/experiments/evaluate_drift_detection.py` |  |
| | *Deployment-register tickets are LEGITIMATE, not drift. A monitor on the in-domain reference would alarm continuously; the binding constraint is what the reference is made of.* | | | |
| `T13.realistic_traffic.flag_rate.alpha0.05` | 0.428571 | `data/drift_evaluation_summary.json` | `python src/experiments/evaluate_drift_detection.py` |  |
| | *Deployment-register tickets are LEGITIMATE, not drift. A monitor on the in-domain reference would alarm continuously; the binding constraint is what the reference is made of.* | | | |
| `T13.realistic_traffic.flag_rate.alpha0.1` | 0.514286 | `data/drift_evaluation_summary.json` | `python src/experiments/evaluate_drift_detection.py` |  |
| | *Deployment-register tickets are LEGITIMATE, not drift. A monitor on the in-domain reference would alarm continuously; the binding constraint is what the reference is made of.* | | | |
| `T13.realistic_traffic.flag_rate.alpha0.2` | 0.645714 | `data/drift_evaluation_summary.json` | `python src/experiments/evaluate_drift_detection.py` |  |
| | *Deployment-register tickets are LEGITIMATE, not drift. A monitor on the in-domain reference would alarm continuously; the binding constraint is what the reference is made of.* | | | |

### T14

| id | value | source file | regenerating command | tags |
|---|---|---|---|---|
| `T14.grounded` | 31/33 [95% CI 0.803934, 0.983219] | `data/groundedness_results.csv` | `python src/experiments/score_groundedness_set.py` |  |
| `T14.hedge_appropriate` | 32/33 [95% CI 0.846809, 0.994631] | `data/groundedness_results.csv` | `python src/experiments/score_groundedness_set.py` |  |
| `T14.judge.raw_agreement` | 30/33 | `data/groundedness_results.csv` | `python src/experiments/score_groundedness_set.py` |  |
| | *A high raw agreement score conceals the judge's behaviour entirely; only the kappa and the individual disagreements expose it.* | | | |
| `T14.judge.cohens_kappa` | -0.042105 | `data/groundedness_results.csv` | `python src/experiments/score_groundedness_set.py` |  |
| | *BELOW ZERO. Every disagreement was the judge substituting APPROPRIATENESS for SUPPORT, in both directions. A judge is an object of study here, never a scaling tool.* | | | |
| `T14.judge.n_disagreements` | 3 | `data/groundedness_results.csv` | `python src/experiments/score_groundedness_set.py` |  |
| `T14.context_distinct_fixes_2plus` | 23/33 | `data/groundedness_results.csv` | `python src/experiments/score_groundedness_set.py` |  |
| | *The retrieved context is heterogeneous -- which is why the near-duplicate explanation for 6C is a REJECTED hypothesis.* | | | |

### T16

| id | value | source file | regenerating command | tags |
|---|---|---|---|---|
| `T16.minilm.pooled_cliff` | 0.8 | `data/resolution_clustering_calibration_results.csv` | `python src/experiments/calibrate_resolution_clustering.py; python src/experiments/flag_automation_candidates.py; python src/experiments/score_flag_validation_set.py --pilot` |  |
| | *Recomputed from the committed per-category rows; it must equal the production threshold, and the build fails if not.* | | | |
| `T16.bge.pooled_cliff` | 0.9 | `data/resolution_clustering_calibration_results_bge-base-en-v1-5.csv` | `python src/experiments/calibrate_resolution_clustering.py; python src/experiments/flag_automation_candidates.py; python src/experiments/score_flag_validation_set.py --pilot` |  |
| | *MEASURED but NOT PROMOTED. Phase 2A found the two indistinguishable on precision; promotion is a product decision about review-queue capacity.* | | | |
| `T16.production.threshold` | 0.8 | `src/agent/config.py` | `python src/experiments/calibrate_resolution_clustering.py; python src/experiments/flag_automation_candidates.py; python src/experiments/score_flag_validation_set.py --pilot` |  |
| `T16.flagged_share.max` | Infrastructure 0.930435 | `data/automation_candidates.json` | `python src/experiments/calibrate_resolution_clustering.py; python src/experiments/flag_automation_candidates.py; python src/experiments/score_flag_validation_set.py --pilot` |  |
| `T16.flagged_share.min` | Database 0.489796 | `data/automation_candidates.json` | `python src/experiments/calibrate_resolution_clustering.py; python src/experiments/flag_automation_candidates.py; python src/experiments/score_flag_validation_set.py --pilot` |  |
| | *A genuine finding about which categories have standardised fixes -- not a defect of the clustering.* | | | |
| `T16.pilot.false_merges` | 0/12 | `data/automation_flag_validation_pilot_results.csv` | `python src/experiments/calibrate_resolution_clustering.py; python src/experiments/flag_automation_candidates.py; python src/experiments/score_flag_validation_set.py --pilot` | `case-study` |
| | *Zero false merges in the pilot. STOP verdict: the harness found nothing to separate the two configurations.* | | | |
| `T16.pilot.rule_of_three_upper` | 0.25 | `data/automation_flag_validation_pilot_results.csv` | `python src/experiments/calibrate_resolution_clustering.py; python src/experiments/flag_automation_candidates.py; python src/experiments/score_flag_validation_set.py --pilot` |  |
| | *Upper 95% bound on the false-merge rate after zero events in 12 pairs -- the honest reading of a zero count.* | | | |

---

## DO NOT CITE

Each of these was published at some point and is now retired. The parity test greps the whole of `paper/` for the retired literals and fails if one reappears outside this section.

### 12 templates / ~430 rows per template / 11 touched / 40 surviving rows

- **Why it is retired:** The superseded Finding 2 diagnostic. It grouped by scenario_id alone, which is unique only WITHIN a category, so it merged all seven categories' templates. It reached published results.
- **Use instead:** T8 -- 66 templates, 62 touched, median 62 rows, 210 of 4000 surviving.

### the cascade is worth +35.6 points

- **Why it is retired:** That figure is baseline minus Tier-1-only, i.e. the BGE-vs-TF-IDF REPRESENTATION gap. It is not what cascading buys.
- **Use instead:** T3 -- against the Tier-2-only control the cascade is one ticket worse on both sets (exact McNemar p = 1.000 each), and buys 8-18% of median per-ticket latency.

### 7A's domain AUC 0.8584

- **Why it is retired:** Computed in an interactive session, quoted at a gate, never committed, and it DOES NOT REPRODUCE. This is the concrete motivation for Phase 8A.
- **Use instead:** T11 -- 0.8727 raw / 0.8472 operating, from run_external_conformal_shift.py.

### 7A's domain AUC range 0.6706-0.9316

- **Why it is retired:** Same defect as 0.8584 -- an uncommitted interactive derivation.
- **Use instead:** T11's Design B column, which is itself no-resolution and may not be a headline.

### the LLM judge's hedge-axis kappa of 0.000

- **Why it is retired:** A degenerate agreement table on that axis. It is not evidence about the judge and is never included.
- **Use instead:** T14 -- the SUPPORT axis: raw agreement 30/33 with Cohen's kappa below zero, plus the three disagreements read individually.

### reweighting recovers 47.6% against distribution matching's ~38%

- **Why it is retired:** An ORDERED COMPARISON between two residuals that differ by less than the measurement's own noise band. Both strategies are partial; neither is shown to beat the other.
- **Use instead:** T7 -- report both readings of 6B's clause and make no ordered comparison.

### any Phase 7B Design B number as a headline

- **Why it is retired:** Design B resolved nothing, twice: one queue BLOCKED by the degeneracy rule and the rest single-class test arms.
- **Use instead:** T11's Design A rows, which are the pre-registered primary.

### Phase 7C coverage numbers as a verdict

- **Why it is retired:** The pre-registered rule fired at a TF-IDF-space domain AUC of 0.9972, so NO verdict was drawn on the primary. A blocked arm is not a null.
- **Use instead:** T11's 7C rows, every one tagged post-hoc and BLOCKED.

---

## Numbers in the documents with NO committed source

These appear in the project's documents but cannot be regenerated from any committed file. Nothing here was invented, re-derived or re-run. Each is a decision: keep it as a figure whose derivation is a script that must be re-executed, drop it, or add the missing writer in a later phase.

| number | stated in the docs | where | why there is no source |
|---|---|---|---|
| Cascade threshold table by target accuracy (the three calibration attempts, two rejected) | three attempts; 0.50 chosen at a 70-80% target | README.md, cascade calibration section | train_cascade.py prints its sweep and writes no results file, so the threshold-by-target-accuracy table cannot be regenerated. |
| Fine-tuned DistilBERT, 14-ticket benchmark | 7/14 (50.0%) | README.md, the three-way classifier comparison and Final Classification Comparison tables | train_distilbert.py writes only label_mapping.json -- no metrics file of any kind. |
| Fine-tuned DistilBERT, 45-ticket benchmark | (absent) | (never measured) | DistilBERT was never run against the 45-ticket benchmark, and the embedding comparison CSV has no DistilBERT row. |
| In-domain self-retrieval contamination rate 5.7% (10/175) | 5.7% (10/175) | README.md, the RAG threshold calibration section and What's Done vs What's Pending | calibrate_rag_similarity_threshold.py computes it but writes no column for it in either calibration CSV. |
| TF-IDF + LogReg, 14-ticket benchmark | 7/14 (50.0%) | README.md, the three-way classifier comparison and Final Classification Comparison tables | train_baseline_tfidf.py and generalization_test.py print their results and write no file; no ablation --mode no-cascade run exists for benchmark14. |

---

## Source files and their hashes at build time

| source file | sha256 |
|---|---|
| `data/ablation_baseline_results.csv` | `8282e2db4353ca57f367672cb2c57126f132e08158816880cd414fc822a9d425` |
| `data/ablation_no-cascade_results.csv` | `55ff1a39151e71b5ce8dbe37ffacd6803fcdee13ff494942693a4628667146da` |
| `data/ablation_no-rag_results.csv` | `13c06d878f8f63b404bfd8f47101e9ff9b8312c3f9e0a11a39752fdd072201da` |
| `data/ablation_tier2-only_results.csv` | `44d2537043d5b0e8bb16c28767c82f553f84707c4ec52f19f710666c86fa0ef4` |
| `data/automation_candidates.json` | `54c63f88c29a0bc0fb36d4497efed47132aaf9620fb8d959f60828acb16dc99b` |
| `data/automation_flag_validation_pilot_results.csv` | `9ccb3b2b035f717c0d2792f4d5e308675e677e020360f0076cde4fb2c0082407` |
| `data/batch_intake/batch_summary.csv` | `4133cbfd0f7af4e5e14912cd78a2219648bd8a86eb60d43b4bef3f5e65b2468b` |
| `data/calibration_reliability_data.csv` | `9e8931d71b77e7796226db38c4d5062489e7815b08000c11b2a7d86cabef48a1` |
| `data/cascade_vs_tier2_mcnemar_benchmark45.csv` | `2a9584376d1793751cb5c123a91e0b661a9cb9c0693493a913931b360f53348c` |
| `data/conformal_calibration_corrected_bge-base-en-v1-5.json` | `6709be36246a02fda1b46b0c19ab5bf73b4843e70384fa3f5264ced67f522ee2` |
| `data/conformal_calibration_results.csv` | `b7cebd98f38857ac5177f214daea895e21aa9e5fe575d703c584349f394bece2` |
| `data/conformal_novelty_results.csv` | `f35fe53274ba9b914471a518b554c208ca8b834165cbad10dba841d80a02babb` |
| `data/deferral_conformal_operating_points.csv` | `83b01e5ebc826d50afe2e7233f193d71bfdc9336fb582a703818706ee4a1818b` |
| `data/deferral_risk_coverage.csv` | `15b34c04ea5721cb2bf80af38b7ab448f08daa0a673c995d5a8b0c60aaa317b0` |
| `data/deferral_rule_summary.csv` | `87287eb7aa803a5a5150795a3a8a581d924b53d69e83bced875c2a7f602f3b25` |
| `data/drift_evaluation_null.csv` | `a3f208a650df86e19d8c758f6bd7e7b8881eca1788bc2f613e070c9b7725e2e4` |
| `data/drift_evaluation_power.csv` | `413b871f337da46b6f940d8b28861fe6c8c80be3d3c70a451f71ac0b84c2bcdb` |
| `data/drift_evaluation_summary.json` | `517cb60c693a30fe1605a862ec1e0b73fdbd4fdff4aa0fc72c188238ec193066` |
| `data/embedding_comparison/embedding_model_comparison.csv` | `82bf7d845167937aea416fcca364c9f0cb4edd177dd68c90047af5fc927975c9` |
| `data/external_tobibueck/external_conformal_designA.csv` | `29258062ddaef8712c9f3f4696d9efb75ce3428b2107ba987c497f3d258a1d78` |
| `data/external_tobibueck/external_conformal_designB.csv` | `979973c2341173b83ef38c623a283517d2dfe54aa93d9e806a5971307a4729f6` |
| `data/external_tobibueck/external_contamination.json` | `a2a40411a14ca8a6799dc3f530995958851d3f17289469dac5780c182867628f` |
| `data/external_tobibueck/external_deferral_results.csv` | `c085cc91398eb99f5a3900de6bd88ce2b9b09085c1686443af8114af7eb30231` |
| `data/external_tobibueck/external_label_noise.json` | `f5bb7308c59db0ecb1b64af9e1f41599bf66db2c0eed83c86a16a883a0f5ff13` |
| `data/external_tobibueck/external_paraphrase_accuracy_did.json` | `6d96b08ca3b5a6e6a81a23b24d1ef652a203c641655d2aa6eef047f189a15c5c` |
| `data/external_tobibueck/external_paraphrase_conformal.csv` | `4a772fe0155c278c1b2d8781189640f4648d44729173aeb638f787a9f1b0d61d` |
| `data/external_tobibueck/external_paraphrase_summary.json` | `a3be14f0d5fd9edca705978b88b54311396de705a84a2a9914c84951d9306a6c` |
| `data/external_tobibueck/profile_summary.json` | `eb91c538b2fc1379ba53b6e8bbccff5761a21d2576b2756f8bd524ab2e021c3a` |
| `data/groundedness_results.csv` | `0022b79ca69537b96cddc416bf95e14edb535d0829c2000e1994f8a5c3d31302` |
| `data/inference_latency_13th-gen-intel-r-core-tm-i5-1334u.csv` | `089b3df69e40414b60d5e26d9afa2e6211c47ad031a86470a06ed840f3ce0b65` |
| `data/rag_similarity_calibration.csv` | `69a2888b11b837d39a92da232332d50d338787c16d75ee447a13f75ccbee11e7` |
| `data/rag_similarity_calibration_combined.csv` | `63edb9663735e0ba169f1ca295d7b926678e2cfa1a48d7ca1c0bfd65d0d882cb` |
| `data/resolution_clustering_calibration_percategory_summary.csv` | `0767f84983ee548354e4b0e94805e610b3eebcecf2663adce8f29ffe99f5aef1` |
| `data/resolution_clustering_calibration_results.csv` | `fac7fd1f3e5afeb045c14dc8122c5c856b1212fca478fff61e0d5c13bb8452c3` |
| `data/resolution_clustering_calibration_results_bge-base-en-v1-5.csv` | `799faf25c3ae566146c80872228a348426852ef76f43ebd055362dac1fafc3e1` |
| `data/skewed/imbalance_sweep_results.csv` | `5d625ab6df4e642b45e8cc771f8c2bdd4821c31afbe13dd9ffe18ea42de168f4` |
| `data/sufficiency_gate_results.csv` | `ef2cd0241322c329c658c766d873335d4b50180c00fc18f78710d4060f80f5a3` |
| `data/sufficiency_gate_summary.json` | `e0e338b7fe74edcbf5820952a37e6e05675a9adaddda671e672a4de28919d891` |
| `data/weighted_conformal_results.csv` | `a68e919cfd2235698ba69b42a45d8a0f426ed2de9a44fe237abf34f56756a967` |
| `data/zeroshot_summary_gemini_gemini-flash-lite-latest_benchmark14.csv` | `5e815b5174ebe00c6fc9e9922b5e376f9f36d18c4ae66c0f90f027e7cf4faafd` |
| `data/zeroshot_summary_gemini_gemini-flash-lite-latest_benchmark45.csv` | `124bcc493f51e2cde052a294cd4791cbaf304afc723743f1243e4d7bcc6eb3c1` |
| `data/zeroshot_summary_ollama_qwen2-5-3b-instruct_benchmark14.csv` | `eea46b8a6220b6510ae08fd40520aebc8ce1328605c948c5c3e670300849625e` |
| `data/zeroshot_summary_ollama_qwen2-5-3b-instruct_benchmark45.csv` | `4e03c542fd950b102bbf7b6ca4f3e18bfe891b190d789356e1069cd7f6b62f3e` |
| `src/agent/config.py` | `53039747da1b62514a14d8bc9a57705a2a8cb3413f8dd857f5491b3df43e93e7` |
| `src/agent/orchestrator.py` | `bf0eefd156920ea0682875172b8bdf3263fc46ba963f0d06e0e9b6db2994ecd4` |

Statistical cross-check at build time: 8 Wilson intervals and 5 exact McNemar tests agreed exactly with the implementations already committed in `src/experiments/`.
