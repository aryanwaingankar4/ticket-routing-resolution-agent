# Results for the paper — revision of October 2026

The source of every value written into
`Calibrated_Escalation_IT_Ticket_Triage_v4.docx`. Each entry gives the
**fill text** (draft, applied in Phase 4), the **value**, the **source file**
and the **command** that regenerates it. Nothing here was typed from memory.

**Commands are run from the project root** with the venv active
(`.\venv\Scripts\Activate.ps1`). Every new script refuses to overwrite its
outputs without `--force`. New result files live in `data/paper_revision_oct/`.
No existing result file, threshold, model or the `v1.0` tag was changed.

Status: **Phase 1 complete** (21 fillable-now markers + 3 new offline analyses).
Phases 2 and 3 (new data) are pending; see the end of this file.

---

## New offline analyses (Phase 1)

All four are offline: 0 Gemini calls, 0 Ollama calls.

| Script | Writes | Reproduction check (FATAL on mismatch) | Result |
|---|---|---|---|
| `scripts/reproduce_cascade_calibration_curve.py` | `cascade_calibration_tickets_paraphrased175.csv`, `cascade_calibration_curve_paraphrased175.csv` | Tier 1 refitted exactly as attempt 3 (`train_cascade.train_tier1`, all 4,000 rows, `max_features` path). Every committed bucket count and accuracy in `cascade_calibration_attempts.csv` must match, and `derive_threshold_for_target()` must give the committed threshold at every target in `cascade_threshold_sweep.csv` | **Passed.** Buckets 152/13/7/3/0/0, accuracies identical; thresholds 1.0001/0.5/0.5. Production `load_tier1()` gives identical predictions, max \|Δconf\| = 7.772e-16 |
| `scripts/reproduce_ood_similarities.py` | `rag_calibration_top1_similarity.csv` | Production `load_artifacts()` + `retriever.retrieve()`. At **every** threshold of `rag_similarity_calibration_combined.csv` the reproduced counts must equal `fp_ood` and `n_in_domain_proceed`; exactly 6/45 out-of-domain tickets ≥ 0.67 | **Passed.** All 19 thresholds reproduce, 6/45 at 0.67 |
| `scripts/error_analysis_benchmark45.py` | `error_analysis_benchmark45_tickets.csv`, `confusion_matrix_cascade_benchmark45.csv`, `error_analysis_benchmark45_summary.csv` | Three committed derivations must agree: `tests/goldens/benchmark_baseline.json`, `data/ablation_baseline_results.csv` (predictions), and `data/groundedness_results.csv` (the 30 drafted tickets must be exactly the goldens' non-escalated ones, with the same categories) | **Passed.** All 45 predictions agree; drafted sets identical |
| `scripts/export_prompts.py` | `prompts.txt` | Renders each prompt with the production builder function; every placeholder must occur exactly once | **Passed** |
| `scripts/make_revision_figures.py` | `paper_docx/fig/fig4_cascade_threshold.png`, `fig5_retrieval_sweep.png`, `fig6_confusion_matrix.png` | Asserts 175/175, 13.3% at 0.67, 37.8% two grid steps below, safe range [0.6196, 0.6704] (read from `adversarial_escalation_results.csv`), and that the confusion matrix sums to 45 | **Passed.** 300 dpi, 3.4 in, Times New Roman, Okabe-Ito; the palette passed the colour validator (CVD ΔE ≥ 11) |

Run order: the first three in any order, then `make_revision_figures.py`.

---

## Markers fillable from committed files

### D1 — the seven categories (§IV-A)
- **Fill:** "The seven categories are Access Management, Application, Database, Infrastructure, Network, Security and Storage."
- **Source:** `data/synthetic_tickets.csv`, `category` column: 4,000 rows, 7 distinct values.

### Table III — dataset splits (S1–S5, plus approved restructuring)
| Marker | Fill | Source |
|---|---|---|
| S1 | 800 | `data/baseline_tfidf_indistribution.csv` (`n_test` = 800, `test_size` 0.2, `random_state` 42) |
| S2 (merged row) | Set "Paraphrased calibration"; size **175**; source "Gemini paraphrases of training rows"; used for "Cascade threshold (0.50), conformal calibration, drift reference, in-domain arm of the 0.67 sweep" | `data/calibration_tickets_paraphrased.json` (175 entries, 25 per category). It is the calibration set in `train_cascade.py:246`, `calibrate_conformal.py:82`, `build_drift_reference.py:92` and `calibrate_rag_similarity_threshold.py:65` |
| S3 | **row deleted** (merged into S2, approved issue 2) | as above |
| S4 | 45 | `data/ood_calibration_tickets.json` (45 entries); `rag_similarity_calibration_combined.csv`: `fp_ood + tn_ood` = 45 at every threshold |
| S5 | "Gemini variants of 15 hand-written off-topic and vague seeds" | `src/classification/generate_ood_calibration_set.py` (`SEEDS`: 15 seeds in buckets A–D, 3 variants each) |
| Training row (approved issue 3) | Size "3,200 / 4,000"; used for "Tier 2 and DistilBERT on the 3,200-row split; Tier 1 on all 4,000 rows" | `train_embeddings.py:263` and `train_distilbert.py:838` (80/20 split); `train_tier1.py:21` (full 4,000 rows) |

### Table IV — configurations (H1–H7)
| Marker | Fill | Source |
|---|---|---|
| H1 | "unigrams and bigrams, English stop words removed" | `train_cascade.train_tier1()` (`ngram_range=(1, 2)`, `stop_words="english"`); confirmed on the loaded production bundle |
| H2 | "max_iter = 1000; otherwise scikit-learn defaults (L2, C = 1.0, lbfgs)" | `train_cascade.py:547`; `LogisticRegression(max_iter=1000).get_params()` under scikit-learn 1.9.0 |
| H3 | "max_iter = 1000, scikit-learn defaults, fitted on the 3,200-row split" | `train_embeddings.py:263–270` |
| H4 | "API defaults (no generation config set); zero-shot runs at temperature 0.0" | `src/agent/resolver.py:80–85` (`generate_content(model=…, contents=prompt)`, no config); `run_zeroshot_baselines.py:412,487` (`"temperature": 0.0`) |
| H5 | "4 epochs, learning rate 2 × 10⁻⁵, batch size 16, maximum length 128, weight decay 0.01" | `data/distilbert_finetune_metrics.csv` (columns `num_epochs`, `learning_rate`, `batch_size`, `max_length`, `weight_decay`); `train_distilbert.py:153–157` |
| H6 | **Cell reworded (approved issue 1):** "noise band = ±2 s.d. of a binomial coverage estimate, n = 175" | `calibrate_conformal.py:357`, `coverage_sd = √(α(1−α)/n_cal)`; `data/conformal_calibration_results.csv` (`n_calibration` = 175, `coverage_sd` = 0.022678 at α = 0.10, so ±2 s.d. = 0.045) |
| H7 | "seed 42 throughout (stratified 80/20 split); latency on an Intel Core i5-1334U (12 logical CPUs), Windows 11, Python 3.14.3, median of 200 runs after 20 warm-up runs" | `data/inference_latency_13th-gen-intel-r-core-tm-i5-1334u.csv` (`cpu`, `logical_cpus`, `platform`, `python`, `n_runs`, `warmup`). **RAM was not recorded**, so it is not stated |

### T1 — cascade threshold evidence (§V-A)
- **Fill:** "The target accuracies were 0.90, 0.80 and 0.70. No threshold met 0.90, so that target would have sent every ticket to Tier 2. At 0.50, Tier 1 keeps 23 of the 175 paraphrased tickets (13.1%) and labels 19 of them correctly (82.6%)."
- **Approved addition (issue 4):** "The curve is not monotone: above 0.60 only 10 tickets remain, of which 6 are correct, so accuracy falls again at small counts. The rule takes the lowest bucket edge whose cumulative accuracy meets the target, which is 0.50."
- **Values:**
  - targets 0.9 / 0.8 / 0.7 → 1.0001 / 0.5 / 0.5 (`cascade_threshold_sweep.csv`; attempt-3 rows of `cascade_calibration_attempts.csv`);
  - at ≥ 0.50: 13 + 7 + 3 = 23 kept, 13 + 5 + 1 = 19 correct (bucket counts × accuracies, attempt 3), equal to the fine curve's 0.50 row (`n_kept` 23, `n_kept_correct` 19);
  - at ≥ 0.60: 10 kept, 6 correct (`cascade_calibration_curve_paraphrased175.csv`).
- **Benchmark side (if wanted):** Tier 1 keeps 4/45 benchmark tickets (8.9%) (`cascade_threshold_sweep.csv` `expanded45_tier1_pct` 0.0889; goldens `tier == 1` count 4).

> **Decided (2026-10-19):** add "on 10%-wide confidence buckets" to the §V-A
> phrase "chose the lowest threshold that meets the target accuracy", and add:
> "On a 0.01 grid the same rule would choose 0.43 for the 0.80 target and 0.36
> for 0.70; kept accuracy peaks at 25/30 at 0.46, so 0.50 sits near the peak."
> The live threshold and all results are unchanged. Values below.
>
> **Finding (F4).** The rule is evaluated
> only at the 10%-bucket edges (`CONFIDENCE_BIN_EDGES` 0.5, 0.6, …). On the fine
> 0.01 grid the same cumulative-accuracy rule would choose **0.43** for the 0.80
> target (35 kept, 29 correct, 0.829) and **0.36** for 0.70 (75 kept, 53 correct,
> 0.707). Kept accuracy peaks at **0.46 (25/30, 0.833)**, and 0.50 (19/23, 0.826)
> sits within one ticket of that peak. So 0.50 is "the lowest qualifying bucket
> edge", and the paper's phrase "chose the lowest threshold that meets the target
> accuracy" (§V-A) is true only on that bucket grid. Source:
> `cascade_calibration_curve_paraphrased175.csv`. The live threshold does not
> change, and no result moves.

### F4 — Fig. 4
- **File:** `paper_docx/fig/fig4_cascade_threshold.png`.
- **Plots:** the share kept (all thresholds 0.15–0.85) and the kept accuracy (only where ≥ 5 tickets are kept, i.e. up to 0.67, as stated in the figure). Dotted lines mark the three targets, the dashed line marks 0.50, and the bucket-edge points 19/23 and 6/10 are labelled.
- **Source:** `cascade_calibration_curve_paraphrased175.csv`; threshold from `settings.cascade.confidence_threshold`.

### F5 — Fig. 5
- **File:** `paper_docx/fig/fig5_retrieval_sweep.png`.
- **Plots:** the in-domain proceed rate (`n_in_domain_proceed / 175`) and the out-of-domain leakage (`ood_leakage_rate`) at every threshold of `data/rag_similarity_calibration_combined.csv`. Markers show the uneven grid (0.01 steps between 0.60 and 0.70, 0.05 steps elsewhere).
- **Reproduces:** 175/175 and 13.3% (6/45) at 0.67, and 37.8% (17/45) at 0.65.
- **Safe range:** [0.6196, 0.6704], shaded. It is the maximum `rag_similarity` of a must-escalate ticket (adv_04, 0.619577) and the minimum of a must-proceed ticket (adv_03, 0.670427), read from `data/adversarial_escalation_results.csv`.

### X2 — escalation rate on the 45-ticket set (Table VIII)
- **Fill:** "15/45 (33.3%)".
- **Source:** `tests/goldens/benchmark_baseline.json` (`actual_escalate` true for 15 rows), cross-checked: the 30 drafted tickets equal the 30 `novel-45` items in `data/groundedness_results.csv`. Summary: `error_analysis_benchmark45_summary.csv` (`escalated` 15, `drafted` 30).

### E1 / F6 — misclassification (Table IX, Fig. 6)
- **E1 fill (replaces the "Infrastructure ↔ Network" hint, approved issue 5):** "Database→Application, Infrastructure→Application and Security→Storage (2 each); 10 of the 13 errors are wrong predictions of Application or Storage."
- **Values** (`error_analysis_benchmark45_summary.csv`):
  - 2 each: Database→Application, Infrastructure→Application, Security→Storage;
  - 1 each: Access Management→Storage, Application→Database, Application→Storage, Infrastructure→Database, Network→Application, Network→Database, Network→Storage;
  - wrong predictions into Application 5, Storage 5, Database 3;
  - recall: Access Management 6/7, Application 5/7, Database 3/5, Infrastructure 3/6 (the 50% the text quotes), Network 4/7, Security 4/6, Storage 7/7.
- **F6 file:** `paper_docx/fig/fig6_confusion_matrix.png`, from `confusion_matrix_cascade_benchmark45.csv` (sums to 45, diagonal 32).

### E2 — out-of-domain tickets that leaked (Table IX)
- **Fill:** "all 6 are vague 'something is broken' messages naming no system (3 of the 15 seeds); no off-topic ticket (weather, food, HR, Finance, Facilities) leaks. Example: 'Hey there, something isn't working right on my end and I really need some assistance…' (0.725)."
- **Values** (`rag_calibration_top1_similarity.csv`, arm `ood`, `passes_gate` true):
  - ood_D1_v3 0.7249
  - ood_D4_v3 0.7073
  - ood_D4_v1 0.6927
  - ood_D2_v3 0.6883
  - ood_D4_v2 0.6815
  - ood_D2_v1 0.6805
- **Grouping:** all six are from bucket D ("vague, non-actionable", `generate_ood_calibration_set.py` `SEEDS`), seeds D1, D2 and D4. The highest bucket A–C ticket is ood_B3_v2 at 0.6651, below the gate.
- **Note:** variants of one seed are not independent (`calibrate_conformal.seed_key`), so the 6 leaks are 3 seeds, not 6 independent events.

### E3 — concrete misclassified tickets (§VII-A)
- **Fill:** "For example, N31 (Security: ransomware pop-ups demanding payment) was routed to Storage, N39 (Network: the VPN connection to the company portal fails) to Database, and N13 (Infrastructure: company websites down for the whole office) to Database."
- **Source:** `error_analysis_benchmark45_tickets.csv`; the texts come from `data/novel_tickets_expanded.json` (N-id = index + 1, the `build_groundedness_set.py:168` convention).
- **Outcomes:** N31 was drafted and labelled ungrounded, N39 was drafted and labelled grounded, and N13 was escalated (0.655).

### E4 — misrouting × escalation (§VII-B)
- **Fill:** "5 were escalated by the retrieval gate (the safe outcome) and 8 received a draft (the unsafe outcome)."
- **Approved addition (issue 6):** "The classification gate does not stop a misrouted ticket from reaching the drafter, which is a limitation of the current design."
- **Values** (`error_analysis_benchmark45_summary.csv`): misclassified escalated 5, misclassified drafted 8; for reference, correct escalated 10 and correct drafted 22.
- **Optional context, not yet in the text:** of the 8 misrouted drafts, 7 were labelled *grounded*, i.e. faithful to the retrieved tickets, which belong to the wrong team. The eighth is N31, labelled ungrounded (`misclassified_drafted_labelled_ungrounded` = 1).

### E5 — what was unsupported in N26 and N31 (§VII-C)
- **Decided (2026-10-19):** state plainly that both ungrounded drafts acknowledged the mismatch and then added outside-knowledge fixes anyway, so the model did not follow the "use only the retrieved fixes" instruction; mention that N31 was also misrouted to Storage.
- **Fill:** "In both drafts the model itself noted that the retrieved tickets did not match. N26 retrieved fixes for failing disks behind network file shares, and N31 retrieved NTFS permission fixes. It then supplied steps from outside knowledge, which the prompt forbids: clean up the local disk for N26, and disconnect the machine and escalate to incident response for N31. Both drafts asked for human review. N31 had also been misrouted to Storage."
- **Source:** `data/groundedness_results.csv` (`unsupported_span` for G021/N26 and G024/N31); `data/groundedness_set.json` (`draft`, `retrieved_resolutions`: five "failing disk in the array behind \\fsNN" fixes for G021, five "Corrected the inherited NTFS permissions" fixes for G024).

### P1, P2 — prompts (Appendix A)
- **Fill:** verbatim from `data/paper_revision_oct/prompts.txt` (`python scripts/export_prompts.py`), rendered by `src/agent/resolver.build_prompt` and `run_zeroshot_baselines.build_prompt`. The "Past ticket #k" block repeats for k = 1…5.

### R1 — reproduction steps (Appendix B)
Every script below was checked to exist on `paper-revision-oct`.
1. Environment: `python -m venv venv`, then `.\venv\Scripts\Activate.ps1`, then `pip install -r requirements.txt` (Python 3.14.3). Put `GEMINI_API_KEY` in `.env` only for the steps that call Gemini.
2. Data and models:
   - `python data/generate_dataset.py`
   - `python src/classification/train_tier1.py`
   - `python src/classification/train_embeddings.py`
   - `python src/rag/build_vector_index.py`
3. Threshold calibration:
   - `python src/classification/train_cascade.py`
   - `python -m src.experiments.calibrate_rag_similarity_threshold`
   - `python -m src.experiments.calibrate_conformal`
4. Evaluation:
   - `python src/experiments/test_adversarial_escalation.py`
   - `python src/experiments/run_ablation_study.py --mode {baseline, tier2-only, no-cascade, no-rag}`
   - `python src/experiments/compare_cascade_vs_tier2.py`
   - `python src/experiments/measure_inference_latency.py`
   - `python src/experiments/run_zeroshot_baselines.py` and `python src/experiments/compare_zeroshot_vs_tier2.py`
   - `python src/experiments/score_groundedness_set.py`
   - `python src/experiments/score_sufficiency_gate.py`
5. Revision analyses and figures:
   - `python scripts/reproduce_cascade_calibration_curve.py`
   - `python scripts/reproduce_ood_similarities.py`
   - `python scripts/error_analysis_benchmark45.py`
   - `python scripts/export_prompts.py`
   - `python scripts/make_revision_figures.py`
   - `python scripts/make_paper_figures.py`
6. Tables, numbers and checks:
   - `python src/experiments/build_paper_artifacts.py --force`
   - `pytest`
   - `pytest tests/test_paper_artifacts.py`

---

## Approved text changes, drafted for Phase 4 (not yet applied)

| # | Where | Current | Proposed |
|---|---|---|---|
| 1a | Table IV, Conformal cell | "noise band = ±2 s.d. over [[FILL-H6]] calibration draws" | "noise band = ±2 s.d. of a binomial coverage estimate, n = 175" |
| 1b | §VI-A, first bullet | "against a noise band of ±0.045 (two standard deviations over calibration draws)" | "against a noise band of ±0.045 (two standard deviations of a binomial coverage estimate, n = 175)" |
| 1c | Table VI caption | "…Only the calibration set changes between the two gap columns" | add: "±2 s.d. is the binomial standard deviation of a coverage estimate at n = 175." (the caption contains no "draws" wording today; this states the column's meaning) |
| 2 | Table III | separate "Paraphrased calibration" and "In-domain calibration" rows | one row (see S2 above) |
| 3a | §IV-C | "The classifiers are trained on 3,200 rows." | "Tier 2 and DistilBERT are trained on a stratified 3,200-row split (seed 42); the production Tier 1 is fitted on all 4,000 rows." |
| 3b | Table III, Training row | "3,200 \| … \| Fitting all trained classifiers" | "3,200 / 4,000 \| … \| Tier 2 and DistilBERT on the 3,200-row split; Tier 1 on all 4,000 rows" |
| 4 | §V-A | — | the sentence under T1 above |
| 6 | §VII-B | — | the sentence under E4 above |

Other places that mention 3,200 were checked and are correct as written: §VI-C ("a classifier fitted on 3,200 of its rows") refers to Tier 2. Pending decision: the fine-grid finding under T1.

---

## Phase 2 — real-world tickets (in progress)

**Dataset (chosen 2026-10-19: option A only).** L. S. B. Pereira, "Classification of IT Support Tickets", Zenodo, 2022.
- **DOI:** 10.5281/zenodo.7648117 (latest version; v1 is 10.5281/zenodo.7384758). **Licence:** CC BY 4.0.
- **Content:** 2,229 tickets from an IT support company in the Florianópolis (Brazil) region. Each was typed by the user who opened it and labelled into 7 categories by three IT professionals in 2020. Personal data was replaced by tags such as [NAME].
- **Fetch:** `python scripts/fetch_zenodo_tickets.py` downloads only the CSVs, `distributions.txt`, `metrics.txt` and `code.py` (as text). The record's pickles and Keras model are not downloaded. Every file is md5-pinned.
- **Reference to add:** L. S. B. Pereira, "Classification of IT support tickets," Zenodo, Dec. 2022, doi: 10.5281/zenodo.7648117. Licensed under CC BY 4.0.

**English filter** (`python scripts/prepare_zenodo_tickets.py`, after `pip install -r requirements-revision.txt`):
- **Method:** langid.py 1.1.6, unrestricted over its 97 languages and deterministic. A ticket is English when its top language is `en`. The anonymisation tags are stripped for detection only.
- **Result: 1,734 of 2,229 English (77.8%).** By source category (English / all): Active Directory 85/133, Computer-Services 46/106, EOL 103/103, Fileservice 597/684, O365 144/219, Software 158/222, Support general 601/762.
- **Sensitivity (not used):** restricted to the four languages the record names, langid calls 1,801 English. The extra tickets are short texts the unrestricted model assigns to Danish, Norwegian, etc.
- **Reference to add:** M. Lui and T. Baldwin, "langid.py: An off-the-shelf language identification tool," in Proc. ACL 2012 System Demonstrations, Jeju Island, Korea, 2012, pp. 25–30. Verified against the ACL Anthology, P12-3005.

**Duplicates.** 745 of the 2,229 texts are exact duplicates of another. Within the English set: Fileservice 360/597, Support general 217/601, EOL 101/103 and Software 34/158. They are mostly system-generated subjects such as "File Share Access - [TICKET ID] - [NAME] (…)". The scoring script reports an **all** reading and a **distinct-text** reading.

**Mapping: awaiting author confirmation.** `category_mapping.json` has `status: proposed`, and `score_realworld.py` refuses to run until it is `confirmed`. The review samples are in `mapping_review_samples.csv` (5 per category, seed 42).

**Routing:** `python scripts/run_realworld_routing.py` runs the unchanged pipeline over all 1,734 English tickets, with no LLM call. Config fingerprint 9c9a5cbcb53f.

## Pending

- **Real-world tickets (Phase 2):** S7, S8, X8–X11, and the real-world parts of X1, X3, A1, C1, L1, T0, D2. Waiting on the mapping confirmation, then scoring and at most 30 drafts.
- **Independent author-written set (Phase 3):** S6, X4–X7, and the independent parts of the same shared markers. See below.

## Phase 3 — independent set (setup done; tickets not yet written)

**Design (changed 2026-10-19):**
- **One writer, AW:** about 100 tickets, roughly 13 per category plus about 10 off-topic or vague.
- **One second labeller** (name to be confirmed) labels every ticket blind.
- **Disagreements** are resolved by discussion between the two and recorded in `adjudication.csv` (`resolved_by`, e.g. `AW+XY`). κ is computed before that discussion.
- The four-author rotation was removed.

**Files and script:**
- `data/independent_test/GUIDELINES.md`, `TEMPLATE.csv`, `SECOND_LABELS_TEMPLATE.csv`, `ADJUDICATION_TEMPLATE.csv`.
- `python scripts/eval_independent.py [--check-only]` runs with two annotators.
- `--single-annotator` runs without second labels. κ is skipped, gold is the writer's label, and the summary records `annotation = single annotator`. The script refuses this mode if second-label files exist.

**Planned fills.** Every `{…}` is filled from `data/independent_test/results/independent_summary.csv` once the set exists.
- **S6 (Table III, size):** the actual row count of `tickets_AW.csv`. **Planned: about 100.**
- **D2 (§IV-C):** "We add two evaluation sets that no language model wrote: {n} tickets written by one of the authors and labelled blind by a second annotator, and {N} real tickets from a public IT-support dataset (Section VI-E)."
- **X1, independent part (§VI-E):** "One author wrote {n} tickets, about 13 per category plus {k} off-topic or vague ones. They were written without consulting the training data, the generation templates or the benchmarks, and without LLM assistance. A second annotator labelled the category and the expected outcome of every ticket without seeing the writer's labels. Cohen's κ was {κ_cat} for category and {κ_out} for outcome, and the {d} disagreements were resolved by discussion."
  - **Single-annotator variant** (only if run with `--single-annotator`): replace the last two sentences with "The labels were assigned by the writer alone (a single annotator), so no inter-annotator agreement is reported."
- **X1, real-world part:** drafted after the mapping is confirmed.

**Text change this design implies (needs your OK, Phase 4 rule 3):**

| Where | Current | Proposed |
|---|---|---|
| Table III, independent-set row, Source | "Written blind by the authors" | "Written blind by one author; second annotator" (or "…; single annotator" in that mode) |
