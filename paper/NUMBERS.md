# NUMBERS.md -- every number the paper may use

**GENERATED. Do not edit by hand.** Rebuild with:

```powershell
python src/experiments/build_paper_artifacts.py --force
```

Phase 7A recorded two domain AUCs that were computed in an interactive session, quoted at a gate, and never committed. They do not reproduce. Every value below therefore names the committed file it came from and the command that regenerates that file. A number that is not in this table is not a number the paper may use.

Config fingerprint at build time: `9c9a5cbcb53f`.

Conventions: proportions carry a Wilson 95% interval computed with z = 1.96, except on pre-registered case-study axes, which report counts only. Paired comparisons on the same tickets use the EXACT McNemar test. A difference smaller than its own noise band is tagged `within-band` automatically.

Tags: `post-hoc` (not a pre-registered result), `BLOCKED` (a pre-registered rule fired; no verdict), `no-resolution` (the comparison could not resolve), `degenerate` (the test has no resolution at this operating point), `case-study` (counts only, by pre-registration), `within-band` (inside the measurement's own noise band), `recorded` (measured once on another platform and NOT re-derivable here: the raw value is transcribed in `data/cross_platform_record.json` with its machine, commit, run id and first-recorded document, and every difference is computed by this builder).

Every value in this table is also a LaTeX macro in `numbers.tex` (`\nb{<id>}`), which is how `main.tex` quotes it. The paper types no number by hand.

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
| `T2.tier2.infrastructure.recall.benchmark45` | 0.5 | `data/ablation_tier2-only_results.csv` | `python src/experiments/summarize_zeroshot_baselines.py --backend ollama --model qwen2.5:3b-instruct --set both --prompt-check-against gemini` |  |
| | *The trained classifier's Infrastructure recall on the 45; the production cascade's is identical.* | | | |
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
| `T7.weighted.tier2.tfidf.worse` | 6/9 | `data/weighted_conformal_results.csv` | `python -m src.experiments.calibrate_conformal; python src/experiments/run_weighted_conformal.py` |  |
| | *Tier-2 configurations whose |coverage gap| grew under TF-IDF-space reweighting -- the cost side of the partial repair.* | | | |
| `T7.weighted.tier2.tfidf.outside_band` | 5/9 | `data/weighted_conformal_results.csv` | `python -m src.experiments.calibrate_conformal; python src/experiments/run_weighted_conformal.py` |  |

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

> THE NAMED CROSS-PHASE FINDING, final wording (settled in Phase 9A) -- the data distribution a component is fitted or calibrated on, not the method or the test applied to it, is the binding constraint. Two scoped instance classes: the CALIBRATION/REFERENCE distribution (Phase 1 Finding 4, coverage; Phase 4B's realistic-traffic arm, drift) and the TRAINING distribution (Phase 5C, classification). Phase 2A is a RELATED dataset limitation, not an instance. Phase 6C is NOT an instance.

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
| `T1.tfidf_baseline.benchmark14` | 6/14 | `data/baseline_tfidf_benchmark14.csv` | `python src/classification/generalization_test.py` |  |
| | *TF-IDF + LogReg fitted on all 4,000 rows -- the ORIGINAL configuration of the 14-ticket baseline, and a local baseline fit rather than the production Tier-1 artifact. Phase 8A.1 gave it a writer and re-ran it: the historically published 7/14 DOES NOT REPRODUCE. See the do-not-cite list.* | | | |
| `T1.tfidf_baseline.benchmark14.split_arm` | 6/14 | `data/baseline_tfidf_benchmark14.csv` | `python src/classification/generalization_test.py` | `new-measurement` |
| | *Secondary arm added in 8A.1: the same pipeline fitted on the 80/20 training split instead of all 4,000 rows. A NEW measurement, never the source for the published figure.* | | | |
| `T1.tfidf_baseline.in_distribution_accuracy` | 1 | `data/baseline_tfidf_indistribution.csv` | `python src/classification/train_baseline_tfidf.py` |  |
| | *The '100% in-distribution' red flag. Template-generated data makes this uninformative -- see FRAMING.md.* | | | |
| `T1.distilbert.benchmark14` | 7/14 | `data/distilbert_finetune_metrics.csv` | `python src/classification/train_distilbert.py --backup-existing` |  |
| | *Best-generalizing epoch of the fresh_retrain run. Phase 8A.1 gave train_distilbert.py a metrics writer and re-ran the fine-tuning from scratch.* | | | |
| `T1.distilbert.benchmark45` | 18/45 | `data/distilbert_finetune_metrics.csv` | `python src/classification/train_distilbert.py --backup-existing` | `new-measurement` |
| | *A NEW MEASUREMENT. DistilBERT had never been evaluated on the 45-ticket benchmark before Phase 8A.1 -- this is not a reproduction of anything. READ IT WITH T1.distilbert.reference.benchmark45: two independent fine-tuning runs of the same configuration disagree by three tickets on this axis, so it does not carry a single-ticket reading. No ordered comparison between the two runs -- they are two draws, not a measurement of a difference.* | | | |
| `T1.distilbert.reference.benchmark14` | 7/14 | `data/distilbert_finetune_metrics.csv` | `python src/classification/train_distilbert.py` |  |
| | *The checkpoints the ORIGINAL DistilBERT result was measured on, scored by the same code. They are gitignored model weights, so this arm is the reference the retrain is judged against, not a source a clean clone can regenerate. The 14-ticket score is IDENTICAL across both runs and all eight epochs, which is what makes it a reproduction.* | | | |
| `T1.distilbert.reference.benchmark45` | 21/45 | `data/distilbert_finetune_metrics.csv` | `python src/classification/train_distilbert.py` | `new-measurement` |
| | *The same new measurement on the original checkpoints. It differs from the retrain's by three tickets -- the honest reading is that CPU fine-tuning reproduces exactly on the 14-ticket axis and not on this one.* | | | |
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
| | *Adversarial tickets decided correctly with the RAG gate removed: only those that should proceed are right.* | | | |
| `T3.baseline.escalation_correct` | 9/9 | `data/ablation_baseline_results.csv` | `python src/experiments/run_ablation_study.py --mode {baseline,no-cascade,tier2-only,no-rag}; python src/experiments/compare_cascade_vs_tier2.py; python src/experiments/measure_inference_latency.py` |  |
| | *CORRECTED in Phase 9A: this was published as 6/9, which is the number that ESCALATED, not the number decided correctly. The baseline decides every adversarial ticket correctly.* | | | |
| `T3.baseline.adversarial_escalated` | 6/9 | `data/ablation_baseline_results.csv` | `python src/experiments/run_ablation_study.py --mode {baseline,no-cascade,tier2-only,no-rag}; python src/experiments/compare_cascade_vs_tier2.py; python src/experiments/measure_inference_latency.py` |  |
| | *The adversarial tickets that should escalate, and do.* | | | |

### T4

| id | value | source file | regenerating command | tags |
|---|---|---|---|---|
| `T4.sweep.threshold.target90` | 1.0001 | `data/cascade_threshold_sweep.csv` | `python src/experiments/plot_calibration_curves.py; python src/classification/train_cascade.py` |  |
| `T4.sweep.benchmark14_tier1_share.target90` | 0 | `data/cascade_threshold_sweep.csv` | `python src/experiments/plot_calibration_curves.py; python src/classification/train_cascade.py` |  |
| `T4.sweep.benchmark14_accuracy.target90` | 0.714286 | `data/cascade_threshold_sweep.csv` | `python src/experiments/plot_calibration_curves.py; python src/classification/train_cascade.py` |  |
| `T4.sweep.benchmark45_tier1_share.target90` | 0 | `data/cascade_threshold_sweep.csv` | `python src/experiments/plot_calibration_curves.py; python src/classification/train_cascade.py` |  |
| `T4.sweep.benchmark45_accuracy.target90` | 0.733333 | `data/cascade_threshold_sweep.csv` | `python src/experiments/plot_calibration_curves.py; python src/classification/train_cascade.py` |  |
| `T4.sweep.threshold.target80` | 0.5 | `data/cascade_threshold_sweep.csv` | `python src/experiments/plot_calibration_curves.py; python src/classification/train_cascade.py` |  |
| `T4.sweep.benchmark14_tier1_share.target80` | 0.214286 | `data/cascade_threshold_sweep.csv` | `python src/experiments/plot_calibration_curves.py; python src/classification/train_cascade.py` |  |
| `T4.sweep.benchmark14_accuracy.target80` | 0.714286 | `data/cascade_threshold_sweep.csv` | `python src/experiments/plot_calibration_curves.py; python src/classification/train_cascade.py` |  |
| `T4.sweep.benchmark45_tier1_share.target80` | 0.088889 | `data/cascade_threshold_sweep.csv` | `python src/experiments/plot_calibration_curves.py; python src/classification/train_cascade.py` |  |
| `T4.sweep.benchmark45_accuracy.target80` | 0.711111 | `data/cascade_threshold_sweep.csv` | `python src/experiments/plot_calibration_curves.py; python src/classification/train_cascade.py` |  |
| `T4.sweep.threshold.target70` | 0.5 | `data/cascade_threshold_sweep.csv` | `python src/experiments/plot_calibration_curves.py; python src/classification/train_cascade.py` |  |
| `T4.sweep.benchmark14_tier1_share.target70` | 0.214286 | `data/cascade_threshold_sweep.csv` | `python src/experiments/plot_calibration_curves.py; python src/classification/train_cascade.py` |  |
| `T4.sweep.benchmark14_accuracy.target70` | 0.714286 | `data/cascade_threshold_sweep.csv` | `python src/experiments/plot_calibration_curves.py; python src/classification/train_cascade.py` |  |
| `T4.sweep.benchmark45_tier1_share.target70` | 0.088889 | `data/cascade_threshold_sweep.csv` | `python src/experiments/plot_calibration_curves.py; python src/classification/train_cascade.py` |  |
| `T4.sweep.benchmark45_accuracy.target70` | 0.711111 | `data/cascade_threshold_sweep.csv` | `python src/experiments/plot_calibration_curves.py; python src/classification/train_cascade.py` |  |
| `T4.calibration.attempts_total` | 3 | `data/cascade_calibration_attempts.csv` | `python src/experiments/plot_calibration_curves.py; python src/classification/train_cascade.py` |  |
| | *Three attempts, two rejected: the in-distribution held-out split (every bucket ~100% accurate, so the threshold looked trustworthy), 35 hand-written tickets (too sparse), and the 175 paraphrased tickets that were adopted.* | | | |
| `T4.calibration.attempts_with_data` | 2 | `data/cascade_calibration_attempts.csv` | `python src/experiments/plot_calibration_curves.py; python src/classification/train_cascade.py` |  |
| | *Attempt 2's 35-ticket set was never committed and is absent from every revision in the repository's history, so it cannot be re-run. Its numbers remain uncited.* | | | |
| `T4.calibration.attempt1.threshold` | 0.7 | `data/cascade_calibration_attempts.csv` | `python src/experiments/plot_calibration_curves.py; python src/classification/train_cascade.py` |  |
| | *Derived on the in-distribution held-out split and REJECTED: observed accuracy is 1.0 in every populated bucket there, so the derivation has nothing to bite on.* | | | |
| `T4.calibration.attempt3.threshold` | 1.0001 | `data/cascade_calibration_attempts.csv` | `python src/experiments/plot_calibration_curves.py; python src/classification/train_cascade.py` |  |
| | *Derived at the 90% target on the 175 paraphrased tickets. 1.0001 is the 'escalate everything' sentinel: at a 90% bar no confidence band is trustworthy. The live 0.50 comes from the 70-80% bar in the sweep above.* | | | |
| `T4.tier1.ece` | 0.112214 | `data/calibration_reliability_data.csv` | `python src/experiments/plot_calibration_curves.py; python src/classification/train_cascade.py` |  |
| | *Count-weighted binned ECE, recomputed from the committed bins, on the 500-ticket IN-DISTRIBUTION production batch.* | | | |
| `T4.tier2.ece` | 0.099248 | `data/calibration_reliability_data.csv` | `python src/experiments/plot_calibration_curves.py; python src/classification/train_cascade.py` |  |
| `T4.observed_accuracy_every_bin` | 1 | `data/calibration_reliability_data.csv` | `python src/experiments/plot_calibration_curves.py; python src/classification/train_cascade.py` |  |
| | *The MINIMUM observed accuracy across all bins of both tiers. It is 1.0, so both tiers are systematically UNDER-confident here and the ECE is entirely the distance to a ceiling. In-distribution accuracy is uninformative on template-generated data -- treat a new 100% as a red flag.* | | | |
| `T4.cascade.threshold` | 0.5 | `src/agent/config.py` | `python src/experiments/plot_calibration_curves.py; python src/classification/train_cascade.py` |  |
| | *The live gate. Frozen; Phases 5-9 are measurement only.* | | | |
| `T4.reliability.n_tickets` | 500 | `data/calibration_reliability_data.csv` | `python src/experiments/plot_calibration_curves.py; python src/classification/train_cascade.py` |  |

### T5

| id | value | source file | regenerating command | tags |
|---|---|---|---|---|
| `T5.self_retrieval_rate` | 10/175 | `data/rag_self_retrieval_check.csv` | `python -m src.experiments.calibrate_rag_similarity_threshold` |  |
| | *In-domain calibration tickets whose top-1 retrieval is their OWN source row. Phase 8A.1 added the writer; the rate itself was always computed here and reproduces exactly.* | | | |
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
| `T13.realistic_traffic.null_rate.alpha0.01` | 0.005682 | `data/drift_evaluation_summary.json` | `python src/experiments/evaluate_drift_detection.py` |  |
| | *The per-ticket marginal null rate at this alpha, for the 175-ticket reference.* | | | |
| `T13.realistic_traffic.ratio_to_null.alpha0.01` | 38.2171 | `data/drift_evaluation_summary.json` | `python src/experiments/evaluate_drift_detection.py` |  |
| `T13.realistic_traffic.null_rate.alpha0.05` | 0.045455 | `data/drift_evaluation_summary.json` | `python src/experiments/evaluate_drift_detection.py` |  |
| | *The per-ticket marginal null rate at this alpha, for the 175-ticket reference.* | | | |
| `T13.realistic_traffic.ratio_to_null.alpha0.05` | 9.4286 | `data/drift_evaluation_summary.json` | `python src/experiments/evaluate_drift_detection.py` |  |
| `T13.realistic_traffic.null_rate.alpha0.1` | 0.096591 | `data/drift_evaluation_summary.json` | `python src/experiments/evaluate_drift_detection.py` |  |
| | *The per-ticket marginal null rate at this alpha, for the 175-ticket reference.* | | | |
| `T13.realistic_traffic.ratio_to_null.alpha0.1` | 5.3244 | `data/drift_evaluation_summary.json` | `python src/experiments/evaluate_drift_detection.py` |  |
| `T13.realistic_traffic.null_rate.alpha0.2` | 0.198864 | `data/drift_evaluation_summary.json` | `python src/experiments/evaluate_drift_detection.py` |  |
| | *The per-ticket marginal null rate at this alpha, for the 175-ticket reference.* | | | |
| `T13.realistic_traffic.ratio_to_null.alpha0.2` | 3.247 | `data/drift_evaluation_summary.json` | `python src/experiments/evaluate_drift_detection.py` |  |
| `T13.realistic_traffic.ratio_to_null.min` | 3.247 | `data/drift_evaluation_summary.json` | `python src/experiments/evaluate_drift_detection.py` |  |
| | *Recomputed in Phase 9A. The documents said '4-7x'; that range does not hold across all four alphas and is not quoted.* | | | |
| `T13.realistic_traffic.ratio_to_null.max` | 38.2171 | `data/drift_evaluation_summary.json` | `python src/experiments/evaluate_drift_detection.py` |  |

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

### T17

| id | value | source file | regenerating command | tags |
|---|---|---|---|---|
| `T17.passed_unsupported.count` | 2/33 | `data/groundedness_results.csv` | `python src/experiments/score_groundedness_set.py; python src/experiments/score_sufficiency_gate.py` | `case-study` |
| | *Drafts that passed the 0.67 gate and were human-labelled ungrounded. Both are 45-ticket benchmark items -- NOT adversarial tickets, as FRAMING.md once said.* | | | |
| `T17.G021.ticket` | N26 | `data/groundedness_results.csv` | `python src/experiments/score_groundedness_set.py; python src/experiments/score_sufficiency_gate.py` |  |
| `T17.G021.top_similarity` | 0.701027 | `data/groundedness_results.csv` | `python src/experiments/score_groundedness_set.py; python src/experiments/score_sufficiency_gate.py` |  |
| `T17.G021.margin_to_gate` | 0.031027 | `data/groundedness_results.csv; src/agent/config.py` | `python src/experiments/score_groundedness_set.py; python src/experiments/score_sufficiency_gate.py` |  |
| `T17.G024.ticket` | N31 | `data/groundedness_results.csv` | `python src/experiments/score_groundedness_set.py; python src/experiments/score_sufficiency_gate.py` |  |
| `T17.G024.top_similarity` | 0.674742 | `data/groundedness_results.csv` | `python src/experiments/score_groundedness_set.py; python src/experiments/score_sufficiency_gate.py` |  |
| `T17.G024.margin_to_gate` | 0.004742 | `data/groundedness_results.csv; src/agent/config.py` | `python src/experiments/score_groundedness_set.py; python src/experiments/score_sufficiency_gate.py` |  |
| `T17.escalated_adequate.count` | 1/21 | `data/sufficiency_gate_results.csv` | `python src/experiments/score_groundedness_set.py; python src/experiments/score_sufficiency_gate.py` | `case-study` |
| | *Escalated tickets whose retrieved context the 6C rater found adequate. The scalar gate errs in this direction too.* | | | |
| `T17.escalated_adequate.0.ticket` | N45 | `data/sufficiency_gate_results.csv` | `python src/experiments/score_groundedness_set.py; python src/experiments/score_sufficiency_gate.py` |  |
| `T17.escalated_adequate.0.top_similarity` | 0.639674 | `data/sufficiency_gate_results.csv` | `python src/experiments/score_groundedness_set.py; python src/experiments/score_sufficiency_gate.py` |  |
| `T17.escalated_adequate.0.margin_to_gate` | -0.030326 | `data/sufficiency_gate_results.csv; src/agent/config.py` | `python src/experiments/score_groundedness_set.py; python src/experiments/score_sufficiency_gate.py` |  |

### T18

| id | value | source file | regenerating command | tags |
|---|---|---|---|---|
| `T18.container_8b.adv_08.rag_similarity.recorded` | 0.6123799085617065 | `data/cross_platform_record.json` | `python src/experiments/measure_tier1_vocabulary_ties.py (the record file is transcribed, not regenerable)` | `recorded` |
| | *local Docker Linux container, python:3.14.3-slim (digest-pinned), on the development laptop, commit 3885393, run no CI run (local); first recorded in PROJECT_STATUS.md, Phase 8B block ('Container verification'), 2026-09-23.* | | | |
| `T18.container_8b.adv_08.rag_similarity.golden` | 0.6123800277709961 | `tests/goldens/benchmark_baseline.json; tests/goldens/adversarial_baseline.json` | `python src/experiments/measure_tier1_vocabulary_ties.py (the record file is transcribed, not regenerable)` |  |
| `T18.container_8b.adv_08.rag_similarity.delta` | -1.1921e-07 | `data/cross_platform_record.json; tests/goldens/benchmark_baseline.json; tests/goldens/adversarial_baseline.json` | `python src/experiments/measure_tier1_vocabulary_ties.py (the record file is transcribed, not regenerable)` | `recorded` |
| | *Computed here from the two raw values -- never stored.* | | | |
| `T18.container_8b.adv_08.rag_similarity.abs_delta` | 1.1921e-07 | `data/cross_platform_record.json; tests/goldens/benchmark_baseline.json; tests/goldens/adversarial_baseline.json` | `python src/experiments/measure_tier1_vocabulary_ties.py (the record file is transcribed, not regenerable)` | `recorded` |
| `T18.container_8b.adv_08.tier1_confidence.recorded` | 0.3182984770932253 | `data/cross_platform_record.json` | `python src/experiments/measure_tier1_vocabulary_ties.py (the record file is transcribed, not regenerable)` | `recorded` |
| | *local Docker Linux container, python:3.14.3-slim (digest-pinned), on the development laptop, commit 3885393, run no CI run (local); first recorded in PROJECT_STATUS.md, Phase 8B block ('Container verification'), 2026-09-23.* | | | |
| `T18.container_8b.adv_08.tier1_confidence.golden` | 0.3182984770932253 | `tests/goldens/benchmark_baseline.json; tests/goldens/adversarial_baseline.json` | `python src/experiments/measure_tier1_vocabulary_ties.py (the record file is transcribed, not regenerable)` |  |
| `T18.container_8b.adv_08.tier1_confidence.delta` | 0.0000e+00 | `data/cross_platform_record.json; tests/goldens/benchmark_baseline.json; tests/goldens/adversarial_baseline.json` | `python src/experiments/measure_tier1_vocabulary_ties.py (the record file is transcribed, not regenerable)` | `recorded` |
| | *Computed here from the two raw values -- never stored.* | | | |
| `T18.container_8b.adv_08.tier1_confidence.abs_delta` | 0.0000e+00 | `data/cross_platform_record.json; tests/goldens/benchmark_baseline.json; tests/goldens/adversarial_baseline.json` | `python src/experiments/measure_tier1_vocabulary_ties.py (the record file is transcribed, not regenerable)` | `recorded` |
| `T18.runner_91a5b38.adv_08.rag_similarity.recorded` | 0.6123793125152588 | `data/cross_platform_record.json` | `python src/experiments/measure_tier1_vocabulary_ties.py (the record file is transcribed, not regenerable)` | `recorded` |
| | *GitHub-hosted runner (gates.yml container job), commit 91a5b38, run 35799173134; first recorded in gates.yml run 35799173134 job log (verify_deployment.py output); supplied by Aryan from the log, 2026-09-23.* | | | |
| `T18.runner_91a5b38.adv_08.rag_similarity.golden` | 0.6123800277709961 | `tests/goldens/benchmark_baseline.json; tests/goldens/adversarial_baseline.json` | `python src/experiments/measure_tier1_vocabulary_ties.py (the record file is transcribed, not regenerable)` |  |
| `T18.runner_91a5b38.adv_08.rag_similarity.delta` | -7.1526e-07 | `data/cross_platform_record.json; tests/goldens/benchmark_baseline.json; tests/goldens/adversarial_baseline.json` | `python src/experiments/measure_tier1_vocabulary_ties.py (the record file is transcribed, not regenerable)` | `recorded` |
| | *Computed here from the two raw values -- never stored.* | | | |
| `T18.runner_91a5b38.adv_08.rag_similarity.abs_delta` | 7.1526e-07 | `data/cross_platform_record.json; tests/goldens/benchmark_baseline.json; tests/goldens/adversarial_baseline.json` | `python src/experiments/measure_tier1_vocabulary_ties.py (the record file is transcribed, not regenerable)` | `recorded` |
| `T18.runner_91a5b38.adv_08.tier1_confidence.recorded` | 0.31818032412549274 | `data/cross_platform_record.json` | `python src/experiments/measure_tier1_vocabulary_ties.py (the record file is transcribed, not regenerable)` | `recorded` |
| | *GitHub-hosted runner (gates.yml container job), commit 91a5b38, run 35799173134; first recorded in PROJECT_STATUS.md, Phase 8B.3 block, and src/classification/train_tier1.py comment; confirmed by Aryan from the run 35799173134 job log, 2026-09-23.* | | | |
| `T18.runner_91a5b38.adv_08.tier1_confidence.golden` | 0.3182984770932253 | `tests/goldens/benchmark_baseline.json; tests/goldens/adversarial_baseline.json` | `python src/experiments/measure_tier1_vocabulary_ties.py (the record file is transcribed, not regenerable)` |  |
| `T18.runner_91a5b38.adv_08.tier1_confidence.delta` | -1.1815e-04 | `data/cross_platform_record.json; tests/goldens/benchmark_baseline.json; tests/goldens/adversarial_baseline.json` | `python src/experiments/measure_tier1_vocabulary_ties.py (the record file is transcribed, not regenerable)` | `recorded` |
| | *Computed here from the two raw values -- never stored.* | | | |
| `T18.runner_91a5b38.adv_08.tier1_confidence.abs_delta` | 1.1815e-04 | `data/cross_platform_record.json; tests/goldens/benchmark_baseline.json; tests/goldens/adversarial_baseline.json` | `python src/experiments/measure_tier1_vocabulary_ties.py (the record file is transcribed, not regenerable)` | `recorded` |
| `T18.runner_07136ec.golden_parity.outcome` | passed | `data/cross_platform_record.json` | `python src/experiments/measure_tier1_vocabulary_ties.py (the record file is transcribed, not regenerable)` | `recorded` |
| | *GitHub-hosted runner (gates.yml full-suite job), commit 07136ec, run 35869187805.* | | | |
| `T18.golden_tickets` | 54 | `tests/goldens/benchmark_baseline.json; tests/goldens/adversarial_baseline.json` | `python src/experiments/measure_tier1_vocabulary_ties.py (the record file is transcribed, not regenerable)` |  |
| | *Every routing decision on these tickets is compared EXACTLY across platforms by tests/test_pipeline_parity.py.* | | | |
| `T18.embedding_dim` | 768 | `src/agent/config.py` | `python src/experiments/measure_tier1_vocabulary_ties.py (the record file is transcribed, not regenerable)` |  |
| `T18.gamma_n` | 4.5778e-05 | `src/experiments/compare_gate_csv.py; src/agent/config.py` | `python src/experiments/measure_tier1_vocabulary_ties.py (the record file is transcribed, not regenerable)` |  |
| | *gamma_n = n*u/(1-n*u), n = embedding dim, u = 2^-24 (Higham, section 3.1). Bounds the float32 inner product only; it does not model the encoder's own cross-platform difference.* | | | |
| `T18.tier1_tolerance` | 1.0000e-12 | `tests/test_pipeline_parity.py` | `python src/experiments/measure_tier1_vocabulary_ties.py (the record file is transcribed, not regenerable)` |  |
| `T18.headroom.similarity_distance` | 1.4575e-04 | `tests/goldens/benchmark_baseline.json; tests/goldens/adversarial_baseline.json` | `python src/experiments/measure_tier1_vocabulary_ties.py (the record file is transcribed, not regenerable)` |  |
| `T18.headroom.similarity_ratio` | 3.1838 | `tests/goldens/benchmark_baseline.json; tests/goldens/adversarial_baseline.json; src/experiments/compare_gate_csv.py` | `python src/experiments/measure_tier1_vocabulary_ties.py (the record file is transcribed, not regenerable)` |  |
| `T18.headroom.tier1_distance` | 1.2641e-02 | `tests/goldens/benchmark_baseline.json; tests/goldens/adversarial_baseline.json` | `python src/experiments/measure_tier1_vocabulary_ties.py (the record file is transcribed, not regenerable)` |  |
| `T18.ties.full4000.n_documents` | 4000 | `data/tier1_vocabulary_ties.json` | `python src/experiments/measure_tier1_vocabulary_ties.py (the record file is transcribed, not regenerable)` |  |
| `T18.ties.full4000.above_cut` | 4240 | `data/tier1_vocabulary_ties.json` | `python src/experiments/measure_tier1_vocabulary_ties.py (the record file is transcribed, not regenerable)` |  |
| `T18.ties.full4000.tied_at_cut` | 11834 | `data/tier1_vocabulary_ties.json` | `python src/experiments/measure_tier1_vocabulary_ties.py (the record file is transcribed, not regenerable)` |  |
| `T18.ties.full4000.slots_for_tied` | 760 | `data/tier1_vocabulary_ties.json` | `python src/experiments/measure_tier1_vocabulary_ties.py (the record file is transcribed, not regenerable)` |  |
| `T18.ties.full4000.max_features` | 5000 | `data/tier1_vocabulary_ties.json` | `python src/experiments/measure_tier1_vocabulary_ties.py (the record file is transcribed, not regenerable)` |  |
| `T18.ties.full4000.share` | 0.152 | `data/tier1_vocabulary_ties.json` | `python src/experiments/measure_tier1_vocabulary_ties.py (the record file is transcribed, not regenerable)` |  |
| `T18.ties.split3200.n_documents` | 3200 | `data/tier1_vocabulary_ties.json` | `python src/experiments/measure_tier1_vocabulary_ties.py (the record file is transcribed, not regenerable)` |  |
| `T18.ties.split3200.above_cut` | 4050 | `data/tier1_vocabulary_ties.json` | `python src/experiments/measure_tier1_vocabulary_ties.py (the record file is transcribed, not regenerable)` |  |
| `T18.ties.split3200.tied_at_cut` | 9622 | `data/tier1_vocabulary_ties.json` | `python src/experiments/measure_tier1_vocabulary_ties.py (the record file is transcribed, not regenerable)` |  |
| `T18.ties.split3200.slots_for_tied` | 950 | `data/tier1_vocabulary_ties.json` | `python src/experiments/measure_tier1_vocabulary_ties.py (the record file is transcribed, not regenerable)` |  |
| `T18.ties.split3200.max_features` | 5000 | `data/tier1_vocabulary_ties.json` | `python src/experiments/measure_tier1_vocabulary_ties.py (the record file is transcribed, not regenerable)` |  |
| `T18.ties.split3200.share` | 0.19 | `data/tier1_vocabulary_ties.json` | `python src/experiments/measure_tier1_vocabulary_ties.py (the record file is transcribed, not regenerable)` |  |

### T19

| id | value | source file | regenerating command | tags |
|---|---|---|---|---|
| `T19.dataset_rows` | 4000 | `data/synthetic_tickets.csv` | `python data/generate_dataset.py (read-only benchmarks; no regeneration)` |  |
| `T19.categories` | 7 | `data/synthetic_tickets.csv` | `python data/generate_dataset.py (read-only benchmarks; no regeneration)` |  |
| `T19.train_rows` | 3200 | `data/tier1_vocabulary_ties.json` | `python data/generate_dataset.py (read-only benchmarks; no regeneration)` |  |
| `T19.benchmark14.n` | 14 | `data/ablation_baseline_results_benchmark14.csv` | `python data/generate_dataset.py (read-only benchmarks; no regeneration)` |  |
| `T19.benchmark45.n` | 45 | `data/novel_tickets_expanded.json` | `python data/generate_dataset.py (read-only benchmarks; no regeneration)` |  |
| `T19.adversarial.n` | 9 | `data/adversarial_escalation_tickets.json` | `python data/generate_dataset.py (read-only benchmarks; no regeneration)` |  |
| `T19.deployment175.n` | 175 | `data/ablation_baseline_results_deployment175.csv` | `python data/generate_dataset.py (read-only benchmarks; no regeneration)` |  |
| `T19.rag_top_k` | 5 | `src/agent/config.py` | `python data/generate_dataset.py (read-only benchmarks; no regeneration)` |  |

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

### TF-IDF + LogReg scores 7/14 on the 14-ticket benchmark

- **Why it is retired:** Phase 8A.1 gave generalization_test.py a writer and re-ran it in its original configuration (TF-IDF + LogReg fitted on all 4,000 rows, seed 42). It scores 6/14, and so does the 80/20-fit arm, and so does the production Tier-1 artifact -- three independent derivations agreeing against the documented figure. The most likely explanation is that 7/14 was measured on the earlier 1,000-ticket dataset and never re-measured after the corpus was scaled to 4,000; that cannot be confirmed, because the 1,000-ticket corpus was never committed. It is recorded as a hypothesis, not a cause.
- **Use instead:** T1.tfidf_baseline.benchmark14 -- 6/14 (42.9%), from data/baseline_tfidf_benchmark14.csv.

### the production pipeline scores 6/9 on adversarial escalation (paper surface, Phases 8A-8B)

- **Why it is retired:** Found in Phase 9A. build_t3 counted adversarial tickets that ESCALATED (6) and published the count as 'escalation correct', beside a no-rag arm counted by CORRECTNESS (3/9). The two cells measured different things. Occurrence #9 of the recurring bug class: internally consistent, wrong for its context.
- **Use instead:** T3.baseline.escalation_correct -- 9/9 decided correctly (6/9 escalate, and should), against 3/9 with the RAG gate removed.

---

## Numbers in the documents with NO committed source

These appear in the project's documents but cannot be regenerated from any committed file. Nothing here was invented, re-derived or re-run. Each is a decision: keep it as a figure whose derivation is a script that must be re-executed, drop it, or add the missing writer in a later phase.

| number | stated in the docs | where | why there is no source |
|---|---|---|---|
| Cascade calibration attempt 2: 34 of 35 hand-written tickets collapsed into one confidence bucket | 34/35 in one bucket | README.md, cascade calibration section | The 35-ticket hand-written calibration set was never committed and is absent from every revision in this repository's history, so the attempt cannot be re-run and its bucket counts cannot be regenerated. Phase 8A.1 sourced attempts 1 and 3 and recorded this one as status=no_artifact in data/cascade_calibration_attempts.csv rather than inventing it. |

---

## Source files and their hashes at build time

| source file | sha256 |
|---|---|
| `data/ablation_baseline_results.csv` | `644c817d397a1128e264194aa25970a4bf9deb89e488f43062503f3ddee2c6f2` |
| `data/ablation_baseline_results_benchmark14.csv` | `aeaa7fa1620c043109bc225f3215ecd18c2934ab43cfb5750596fa4d32bce162` |
| `data/ablation_baseline_results_deployment175.csv` | `e67c5e3941bcd6dab7c617274228ed6be97da40bba87f57067906cef8f5ee7f2` |
| `data/ablation_no-cascade_results.csv` | `1fa102fa35e3f5d95e6fdbe28bff7721e553537a8d0fda82b44f0ef209d63bb0` |
| `data/ablation_no-rag_results.csv` | `9b72f1f7550cf985d8bb4c1bbccf7a3a54dab2342e8cdde5709ac1771dd9e093` |
| `data/ablation_tier2-only_results.csv` | `6127ebe45194f5447a5ca689cdd047b2b3fb62f434bcc25a5914307471d096f3` |
| `data/adversarial_escalation_tickets.json` | `61cac33ba9b53d26fec531b3ab6f71ab1a8d8b50072ef7141472503339b30a4b` |
| `data/automation_candidates.json` | `7f167736d748d8a820928e2b6f5bdbaf09468f336fc3310c1fc52d4b67f1d395` |
| `data/automation_flag_validation_pilot_results.csv` | `9165fcd0fae05c2b9890e725c3823d4c5f80318f1a16147649a81a4c967ab1da` |
| `data/baseline_tfidf_benchmark14.csv` | `932bccaaea45022e9d97bb8862623968f0ac56921c75d4ab14864a908eb4aa9d` |
| `data/baseline_tfidf_indistribution.csv` | `0837239b465ac4307188ca7541368876f84e1fa85207455c6d3ce2799f06df4c` |
| `data/batch_intake/batch_summary.csv` | `b8584ad1ea0e93b2563869ebed98a4b8f6e858ef886f5cbf1ae7b02102e36fa6` |
| `data/calibration_reliability_data.csv` | `eb2a2950ea67283e91c97e2e35754a1b61f091570351eb2b541a6db3117e11e0` |
| `data/cascade_calibration_attempts.csv` | `1e0ed2ca5a3e7a34478562df908a95262c83a14f2fdfc31435968dce775d4ef2` |
| `data/cascade_threshold_sweep.csv` | `4867ac077c0a52415792c6c3becb8d5ec53425b0e3d748349f586e80d7342c90` |
| `data/cascade_vs_tier2_mcnemar_benchmark45.csv` | `ce0f2d4091940dcdd849334d174bf5cd3054fdeb0e764e9ca0c89d37159d7aad` |
| `data/conformal_calibration_corrected_bge-base-en-v1-5.json` | `a1daaca47876dbfaff1d9eccc4f9e3febd4d8eb667bb0f0e3558040bc713f71b` |
| `data/conformal_calibration_results.csv` | `9d03ddcabfb8a94136ef3815ac526989fa2703f53a87487688ab85b4d0266c9f` |
| `data/conformal_novelty_results.csv` | `2fe429b4d5facaec3d396d1ca59a23e9b8e57671f31b452235df0e02595337ab` |
| `data/cross_platform_record.json` | `a253ad53a3303a8ecea90844d9a01fd3dbf3628d3cf34ad12eedb085bfe091d0` |
| `data/deferral_conformal_operating_points.csv` | `33e3fa1a512099a3a7d27fa0d3ffa625aa074a1c8cbd570b7cba11ebb12cca9d` |
| `data/deferral_risk_coverage.csv` | `3ae62ebf0a3120ed72b03940db153fb15b0d0b01f4547fb14aa2603375c2bea9` |
| `data/deferral_rule_summary.csv` | `c8eba59b25c8b58f0c6239dcfe40db526572c92118634fb845765d6d8ff8a332` |
| `data/distilbert_finetune_metrics.csv` | `532820a1d3e5c081ede52ae838040e428e4dc26c924ce2196377321767cf901d` |
| `data/drift_evaluation_null.csv` | `3a86cfbafcc13b656f0e89d16fd6837fd97950b275b793f7950215d274eb744f` |
| `data/drift_evaluation_power.csv` | `22b59d5586dab0ce1203993abb51f7d32df942063b7550500346e0b9653eae75` |
| `data/drift_evaluation_summary.json` | `9f174683c71359914be078915b6211d26be80021802d9afec9db8b4a712990a9` |
| `data/embedding_comparison/embedding_model_comparison.csv` | `ca61edd285d3b35baa68e23f7a0ea3b445c37c2699011f010a150528b841a934` |
| `data/external_tobibueck/external_conformal_designA.csv` | `81faba053c2bf512924cdb20d3f573aed484f91398b2d7de1fc5fe7975779412` |
| `data/external_tobibueck/external_conformal_designB.csv` | `81c1d682650947fbab11d76e3f86c307e8e0922157688219c698703fed587b85` |
| `data/external_tobibueck/external_contamination.json` | `683635ca1e8ca1fbc7e485a3f3f2e944c83cc5ad64a75664d95a266853458b36` |
| `data/external_tobibueck/external_deferral_results.csv` | `89b0404174674c7e0022558c22bd0cadb595120b3267ce167f77327a15a81aac` |
| `data/external_tobibueck/external_label_noise.json` | `96a40ca379fa24ebe5c588d0f7d94efa93939561ac5a33930405a18274868c06` |
| `data/external_tobibueck/external_paraphrase_accuracy_did.json` | `d771d05e35b65328a0e9f58483e3200d2015b3002ff4b76fe67498248d3791ea` |
| `data/external_tobibueck/external_paraphrase_conformal.csv` | `db50984138f19553b1013518d0692fb69a6cc7e685be03453c42738d35513b18` |
| `data/external_tobibueck/external_paraphrase_summary.json` | `c136c322b50878ef8cc2f275068d0bf645a580565800a91076f895fecf45bee6` |
| `data/external_tobibueck/profile_summary.json` | `76500edc875d27ac891fb6e9c0e8f83a96b31cbc2b339c9517b467d772108ee2` |
| `data/groundedness_results.csv` | `34bf0e540d75c86594ccea8d3191bf5e029982f393c748e72c43bd9535eecd71` |
| `data/inference_latency_13th-gen-intel-r-core-tm-i5-1334u.csv` | `8903c021b883c9f052faad8247ba9d77d3dd20f917db21c4573642dc282e8890` |
| `data/novel_tickets_expanded.json` | `93816c84728f097510baddc3f2f5a9ef1a0a869e980dc746804229754a8b2e75` |
| `data/rag_self_retrieval_check.csv` | `1673bbef91ccf213278e90ba90bb7291f416ff9636fa39d8f73b97157ec00ce9` |
| `data/rag_similarity_calibration.csv` | `179319f56f7f15280cbaa0c9e5cd9e662f8583bb149f0ac588b567c282a47336` |
| `data/rag_similarity_calibration_combined.csv` | `cb6e1c3177500a898ef80cfae6cb559e7b3935908a81266e3e562b0f60280874` |
| `data/resolution_clustering_calibration_percategory_summary.csv` | `19f9b901f2e23fcf3d7e9bcd48a67f99759f80bdab4a6aea3f361a84839b1491` |
| `data/resolution_clustering_calibration_results.csv` | `f8e89bd5af7cb5faa66ab4693b67e81ceccd1eeb0e942fee6868dc032273db24` |
| `data/resolution_clustering_calibration_results_bge-base-en-v1-5.csv` | `986a91e695198665fab79cf273606b5d87f04cb064ae056f2de18460c2e4fbf6` |
| `data/skewed/imbalance_sweep_results.csv` | `86ba1519eb3cdfff9b96148707e8c1ed95682db3339ca6aec24dc19e29e8dda5` |
| `data/sufficiency_gate_results.csv` | `724439ba04593cdeedbbb549eb82b8f20f0bf5c3375756c1eabb6ac3f835654c` |
| `data/sufficiency_gate_summary.json` | `0fe82de70e9c8abec7ed4aad8ab788e8f2658acf666b54641fb9ecfab876bc95` |
| `data/synthetic_tickets.csv` | `5619bc51a8d186e73c36af522857375727fe536e1d1f0e0827e5cd5a927d09af` |
| `data/tier1_vocabulary_ties.json` | `d7b864d71c33b4cfc06bfc199c0f82faf41bbefa26af3ccf5359c96b4900e7e6` |
| `data/weighted_conformal_results.csv` | `451577a3aab9daae3d4d82065bb728aebc3c4c0139cb8a7ec390d0198f5892df` |
| `data/zeroshot_summary_gemini_gemini-flash-lite-latest_benchmark14.csv` | `d217653cf8b54ef8bbf772097b5d05ec5953d93b30ea036299c2942b204c6b9b` |
| `data/zeroshot_summary_gemini_gemini-flash-lite-latest_benchmark45.csv` | `21dcecdf798c36c955867db76dfd05fb7f4e22fa25424cbf68adb35c7d58e8af` |
| `data/zeroshot_summary_ollama_qwen2-5-3b-instruct_benchmark14.csv` | `0618c6c50bff54bc5918acd92bf8dd96886b5e88f1d3ea8344bffd771d4d85d9` |
| `data/zeroshot_summary_ollama_qwen2-5-3b-instruct_benchmark45.csv` | `08b3b991607635ab5464ed61913e1bbf40b8681081333474bf41a7abf3e93014` |
| `src/agent/config.py` | `12fab906b100fdae5315ba547ce8b6c6e14a2a67b110722586859d549706b07a` |
| `src/agent/orchestrator.py` | `bf0eefd156920ea0682875172b8bdf3263fc46ba963f0d06e0e9b6db2994ecd4` |
| `src/experiments/compare_gate_csv.py` | `a8feee0ffc2bc1e83cbd14b13e39bbc1dad1d51da5b14c809a8a68161644949c` |
| `tests/goldens/adversarial_baseline.json` | `22d00d663c79afa3676c7597cf205f338783c4843bdb6d7c1479f4b27fb99343` |
| `tests/goldens/benchmark_baseline.json` | `648cadd76edd59f82fae356a93087b815bc8f717542b5be556ae96bcbd49fe96` |
| `tests/test_pipeline_parity.py` | `28835a0093e22a84140be5d1fb2e57e53e1ab2c74b58a08cdd6652b1228e3272` |

Statistical cross-check at build time: 8 Wilson intervals and 5 exact McNemar tests agreed exactly with the implementations already committed in `src/experiments/`.
