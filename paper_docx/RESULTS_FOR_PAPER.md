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
**Phase 2 (real-world tickets) scored and drafted** (2026-10-21); fills drafted,
draft labels pending. Phase 3 is waiting on the author-written tickets.

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

**Mapping: confirmed by AW on 2026-10-21** (`category_mapping.json`, `status: confirmed`).
- **Primary mapping:**
  - Fileservice → Access Management. 546 of its 597 English tickets are "File Share Access" grant requests.
  - Active Directory → Access Management.
  - Software and O365 → Application.
  - Computer-Services, Support general and EOL are excluded.
- **Order of events:** the mapping was chosen from the review samples (`mapping_review_samples.csv`) before anyone looked at per-category predictions. `routing_results.csv` existed and was committed, but no per-category prediction was shown or examined before confirmation. *AW: confirm this holds for you too before the paper says so.*
- **Sensitivity (secondary, author's request):** Fileservice → Storage, everything else unchanged (`sensitivity.fileservice_storage` in the same file).
- **Covered teams:** Access Management and Application only. Database, Infrastructure, Network, Security and Storage are **not covered** by the primary mapping.

**Routing:** `python scripts/run_realworld_routing.py` runs the unchanged pipeline over all 1,734 English tickets, with no LLM call. Config fingerprint 9c9a5cbcb53f.

**Scoring** (offline; refuses to overwrite without `--force`):
```
python scripts/score_realworld.py
python scripts/score_realworld.py --variant fileservice_storage
```
- **Primary outputs:** `realworld_summary.csv`, `realworld_by_source.csv`, `realworld_confusion.csv`.
- **Sensitivity outputs:** the same three names with the suffix `_sens_fileservice_storage`.
- **Location:** all six files are in `data/external_zenodo_7648117/`.
- **Method:** Wilson intervals use z = 1.96 via `score_groundedness_set.wilson_interval`. The exact McNemar test uses `compare_cascade_vs_tier2.mcnemar_from_pairs`. Both are the committed implementations.

### Real-world results (primary mapping). Headline = distinct texts; all tickets beside it.

All values come from `realworld_summary.csv` (keys in brackets).

| Quantity | Distinct texts | All tickets |
|---|---|---|
| English tickets excluded (unmapped) | — | 750: Computer-Services 46, Support general 601, EOL 103 [`excluded.*`] |
| n (mapped) | **583** [`distinct.n`] | 984 [`all.n`] |
| Cascade correct | **115/583 = 19.7% [16.7, 23.1]** | 119/984 = 12.1% [10.2, 14.3] |
| Tier-2-only correct | 114/583 = 19.6% [16.5, 23.0] | 118/984 = 12.0% [10.1, 14.2] |
| Cascade vs Tier-2-only, exact McNemar | b = 9, c = 8, p = 1.0 | b = 9, c = 8, p = 1.0 |
| **Escalated at the 0.67 gate** | **129/583 = 22.1% [18.9, 25.7]** | 159/984 = 16.2% [14.0, 18.6] |
| **Tier-1 share (kept at 0.50)** | **238/583 = 40.8% [36.9, 44.9]** | 568/984 = 57.7% [54.6, 60.8] |
| Tier-1 kept and correct | 39/238 | 39/568 |
| Recall, Access Management | 86/319 | 90/682 |
| Recall, Application | 29/264 | 29/302 |
| Misrouted: escalated / drafted | 97 / 371 of 468 | 127 / 738 of 865 |
| Reached the drafter and correctly routed | 83/454 = 18.3% [15.0, 22.1] | 87/825 = 10.5% [8.6, 12.8] |

**Label-free gate readings over the whole English set** (no mapping needed; `english_*` keys):

| Reading | Escalated | Tier-1 share |
|---|---|---|
| 1,015 distinct texts | 223 = 22.0% [19.5, 24.6] | 352 = 34.7% [31.8, 37.7] |
| All 1,734 tickets | 261 = 15.1% [13.4, 16.8] | 822 = 47.4% [45.1, 49.8] |

**Reference points for the comparison** (existing committed files):
- **Benchmark-45:** escalation 15/45 (X2). Tier-1 share 4/45, 2 of them correct (`data/ablation_baseline_results.csv`, `tier_used`).
- **Deployment-175:** Tier-1 share 33/175 (`data/ablation_baseline_results_deployment175.csv`, `tier_used`).
- **Paraphrased-175:** Tier-1 share 23/175 (T1).

**Where the errors come from** (`realworld_by_source.csv`, with `reading` = all or distinct, and `realworld_confusion.csv`):
- **Fileservice:** of its 238 distinct texts, the pipeline sends 203 to Storage, 25 to Access Management and 8 to Security. In our generator, "Permissions incorrect on {share}" is a **Storage** template (`data/generate_dataset.py:428`), so these grant requests sit on a boundary of our own taxonomy.
- **Why 238 here but 237 in the sensitivity recall (203/237):** in `realworld_by_source.csv` "distinct" dedups within each source, so the four mapped sources sum to 82 + 238 + 140 + 124 = 584. The summary dedups across the mapped subset, giving 583: one text occurs under both Active Directory and Fileservice, and both map to Access Management. Quote 203/238 for the per-source statement and 203/237 for recall.
- **Application (Software, O365):** in the all reading, predicted mostly as Access Management (110) and Security (105). Distinct Software: 41 of 124 went to Security.
  - Observation only, not a paper number: a handful of these are vulnerability-scan tickets ("Vulnerable Internet Explorer on [SERVER]"), which our taxonomy would also file under Security. This was a keyword count, not a committed result.
- **The gate does not separate misrouted tickets:** it escalates 97/468 misrouted texts (20.7%) and 32/115 correctly routed ones (27.8%). Source: `distinct.misclassified_escalated`, `distinct.misclassified`, `distinct.escalated.k` and `distinct.cascade_correct.k`; 32 = 129 − 97.

### Sensitivity: Fileservice → Storage (one sentence in the paper + a supplement table)

All values come from `realworld_summary_sens_fileservice_storage.csv`.

| Quantity | Primary (Fileservice → AM) | Sensitivity (Fileservice → Storage) |
|---|---|---|
| Distinct: cascade correct | 115/583 = 19.7% [16.7, 23.1] | **294/583 = 50.4% [46.4, 54.5]** |
| All: cascade correct | 119/984 = 12.1% | 651/984 = 66.2% [63.1, 69.0] |
| Distinct recall | AM 86/319, App 29/264 | AM 62/82, App 29/264, Storage 203/237 |
| Distinct: drafted and correctly routed | 83/454 | 262/454 = 57.7% [53.1, 62.2] |
| Distinct: Tier-1 kept and correct | 39/238 | 199/238 |
| Escalation, Tier-1 share | unchanged (label-free) | unchanged |
| Exact McNemar, distinct | b = 9, c = 8, p = 1.0 | b = 8, c = 6, p = 0.79 |

**Sensitivity sentence (draft, §VI-E):** "The result depends heavily on one mapping decision: if the file-share access requests are mapped to Storage, where our generator files share-permission faults, distinct-text accuracy is 294/583 (50.4%, 95% CI [46.4, 54.5]); the escalation rate and Tier-1 share do not depend on the mapping."

### Decisions recorded 2026-10-21 (AW)

- **Mapping timing confirmed:** AW chose the mapping before opening `routing_results.csv`. The X1 clause "fixed before predictions were examined" stands.
- **Table VIII:** add a "Tier-1 share" column (approved).
- **Appendix C (new, in the Word file):** the Fileservice → Storage sensitivity table below.
- **Draft labels:** AW labels the 30 drafts (single annotator). If `draft_labels.csv` is not filled by Phase 4, the hand-off sentence is left out and the paper says the drafts were "not labelled".
- **Reporting:** the real-world result is stated plainly in the abstract, §VI-E, §VII-B, Limitations and Conclusion (coverage matrix below). It is linked to the recurring finding, with no softening and no overclaiming. Two teams only.
- **Limitations:** a note that the English filter let a few non-English tickets through (observation; no number).

### Drafted fills and text (real-world parts)

**Markers:**
- **S7 (Table III, size):** "583 distinct (984)"
- **S8 (Table III, source):** "Real user tickets, Pereira [ref], CC BY 4.0"
- **Table VIII, Real-world row:** X8 "583", X9 "115/583", X10 "114/583", X11 "129/583 (22.1%)". These use X2's format. The all-ticket reading goes in the X3 text.
- **Table VIII, new "Tier-1 share" column (approved structural change):**
  - Out-of-template: "4/45". Source: `data/ablation_baseline_results.csv`, `tier_used`.
  - Independent: "{all.tier1_kept.k}/{n}". Source: `independent_summary.csv`.
  - Real-world: "238/583". Source: `distinct.tier1_kept.k`.
- **D2:** "We add two evaluation sets that no language model wrote: {n} tickets written by two of the authors, each labelling the other's tickets blind, and 583 distinct real tickets from a public IT-support dataset (Section VI-E)."
- **T0:** "independent author-written set ({n} tickets); 583 real tickets [ref]"

**X1, real-world part (§VI-E):** "The real-world set is the public dataset of Pereira [ref], released under CC BY 4.0: 2,229 tickets typed by the users of an IT-support company and labelled into seven categories by three IT professionals. We kept the 1,734 tickets that langid.py [ref] identifies as English. Four source categories correspond to one of our teams, and the mapping was fixed before predictions were examined: Active Directory and Fileservice (almost entirely file-share access requests) to Access Management, and Software and O365 to Application. The other three categories (750 tickets) have no single counterpart and were excluded, so Database, Infrastructure, Network, Security and Storage are not covered. Of the 984 mapped tickets, 401 repeat an earlier text exactly, mostly system-generated subject lines, so we report the 583 distinct texts and give all 984 alongside."

**X3, real-world part (§VI-E):** "On real tickets the classifier does not transfer: the cascade routes 115 of 583 distinct texts to the right team (19.7%, 95% CI [16.7, 23.1]; 119/984, 12.1%, over all tickets), with recall 86/319 for Access Management and 29/264 for Application. Tier 2 alone does no better (114/583; exact McNemar p = 1.0), so the cascade neither helps nor hurts accuracy here. Tier 1 answers far more often than on generated text (238/583, 40.8% [36.9, 44.9], against 4/45 on the out-of-template set), but only 39 of those 238 answers are correct. The retrieval gate escalates 129/583 (22.1% [18.9, 25.7]; 159/984, 16.2%, over all tickets) but does not filter misrouted tickets: 371 of the 468 misrouted texts still reached the drafter. The result also depends on one mapping decision: if the file-share access requests are mapped to Storage, where our generator files share-permission faults, accuracy is 294/583 (50.4%, [46.4, 54.5]; Appendix C), while the escalation rate and Tier-1 share do not depend on the mapping. This is the recurring finding of this paper in its plainest form: a classifier trained only on synthetic templates does not transfer to real tickets, and the gates, calibrated on the same synthetic material, do not compensate for it."
- **Then, only if `draft_labels.csv` is filled:** "In the 30 sampled drafts, labelled by one author, the model handed the ticket to a human in {handoff.yes + handoff.no} cases, {handoff.yes} of them appropriately, and {grounded.yes}/30 drafts were grounded in the retrieved tickets." Source: `draft_labels_summary.csv`.
- **If it is not filled:** "The 30 drafts sampled from these tickets were not labelled."

**§VII-B, added after approved change 6:** "The real tickets show the same at scale. Of the 468 misrouted distinct texts, 371 passed the retrieval gate and reached the drafter. The gate escalated 97 misrouted and 32 correctly routed texts, so its similarity score does not track routing correctness. Tier 1's confidence does not either: it kept 238 of the 583 texts at the 0.50 threshold, and 39 of those were correct."

**A1 (abstract):** "On 583 distinct real user tickets from a public dataset, covering two of the seven teams, the pipeline routed 19.7% correctly (119 of all 984 tickets), equal to Tier 2 alone (p = 1.0). Accuracy rises to 50.4% if file-share requests are mapped to a different team. Tier 1 answered 238 tickets but only 39 correctly, and the retrieval gate escalated 22.1% but let 371 of 468 misrouted tickets reach the drafter, so a classifier trained only on synthetic templates does not transfer to real tickets."

**C1 (Conclusion):** "On real user-written tickets for the two teams we could test, the classifier did not transfer: 115 of 583 distinct texts (19.7%; 119/984 over all tickets) were routed correctly, Tier 2 alone did no better (p = 1.0), and Tier 1 answered 238 of them with only 39 correct. The retrieval gate did not filter the misroutes (371 of 468 reached the drafter), and accuracy depends on the category mapping (50.4% under the alternative). This is the paper's recurring finding at its starkest: the synthetic training data, not the choice of method or threshold, limits what the pipeline can do."

**L1, real-world part (Limitations):** "The real-world set removes the dependence on generated evaluation text for two teams only: Database, Infrastructure, Network, Security and Storage remain untested on real tickets. Its accuracy depends strongly on how the source categories are mapped to ours (19.7% or 50.4%), and the tickets come from a single organisation. The English filter also let a few non-English tickets through, because their system-generated subject lines were in English."

**Coverage matrix: where each required point appears**

| Point | A1 | §VI-E (X3) | §VII-B | Limitations | Conclusion |
|---|---|---|---|---|---|
| 115/583 = 19.7% (119/984 beside it) | ✓ | ✓ | | ✓ (19.7%) | ✓ |
| gate does not filter misrouted (371 of 468) | ✓ | ✓ | ✓ | | ✓ |
| mapping dependence (50.4%) | ✓ | ✓ | | ✓ | ✓ |
| Tier 1 answers 238/583, 39 right | ✓ | ✓ | ✓ | | ✓ |
| cascade = Tier 2 alone (p = 1.0) | ✓ | ✓ | | | ✓ |
| two teams only; five not covered | ✓ ("two of the seven") | ✓ (X1) | | ✓ | ✓ ("two teams") |
| link to the recurring finding | ✓ | ✓ | | | ✓ |

### Appendix C (new): Fileservice → Storage sensitivity table, for the Word file

Caption: "Real-world results under the primary mapping (file-share access requests → Access Management) and the sensitivity mapping (→ Storage). Distinct texts, n = 583. 95% Wilson intervals in brackets."

| | Primary | Sensitivity |
|---|---|---|
| Cascade correct | 115/583 (19.7%) [16.7, 23.1] | 294/583 (50.4%) [46.4, 54.5] |
| Tier 2 alone correct | 114/583 | 292/583 |
| Exact McNemar, cascade vs Tier 2 | p = 1.0 | p = 0.79 |
| Recall, Access Management | 86/319 | 62/82 |
| Recall, Application | 29/264 | 29/264 |
| Recall, Storage | — | 203/237 |
| Tier-1 answers correct | 39/238 | 199/238 |
| Drafted and correctly routed | 83/454 | 262/454 |
| Escalated at 0.67 | 129/583 (22.1%) | 129/583 (22.1%) |

Sources: `realworld_summary.csv` and `realworld_summary_sens_fileservice_storage.csv` (`distinct.*` keys). Tier 2 alone in the sensitivity column is `distinct.tier2_only_correct.k` = 292.

### Drafts on real tickets (30)

`python scripts/draft_realworld.py --limit 3` (the dry run), then `python scripts/draft_realworld.py`.
- **Calls:** 30 Gemini calls in total. The dry run's 3 were reused from the cache, so the full run made 27 live calls.
- **Model and config:** `gemini-flash-lite-latest`, config fingerprint 9c9a5cbcb53f.
- **Sample:** distinct mapped texts that the gate does not escalate (population: Active Directory 51, Fileservice 224, O365 98, Software 81). Stratified by source with equal allocation (AD 7, Fileservice 8, O365 8, Software 7), `random.Random(42)`. The sample was fixed before any call.
- **Outputs:** `data/external_zenodo_7648117/drafts.json`, plus one cached raw response per ticket in `drafts_raw/<id>.json`, keyed by prompt sha256.
- **Routing of the 30 (mechanical, from drafts.json):** 9 correctly routed, 21 misrouted.
- **Labelling sheet:** `draft_labels.csv`, with columns `id, grounded, appropriate_handoff, notes`, empty. `DRAFT_LABELS_README.md` shows each ticket, its 5 retrieved tickets and its draft, plus the label definitions. Both come from `python scripts/make_draft_label_sheet.py`. After labelling, run `python scripts/score_draft_labels.py` to write `draft_labels_summary.csv`. **Not yet labelled.**
- **Observation, not a paper number:** in at least 2 of the 30 (ids 1077 and 1212) the user's text is Spanish or German. Their English system-generated subject lines carried them through langid. Covered by the L1 sentence.

## Pending

- **Real-world tickets (Phase 2):** AW labels the 30 drafts. Everything else is drafted.
- **Independent author-written set (Phase 3):** S6, X4–X7, the Tier-1 cell, and the independent parts of the shared markers. See below.

## Phase 3 — independent set (setup done; tickets not yet written)

**Design (changed 2026-10-21):**
- **Writers:** two authors, AW and SP, about 54 tickets each (7 per category + 5 off-topic or vague).
- **Cross-labelling:** each labels the **category** of the other's tickets blind. Expected outcomes are the writer's.
- **Disagreements:** resolved by discussion and recorded in `adjudication.csv`, with `resolved_by` = `AW+SP`.
- **κ:** Cohen's κ on the first, independent labels, before adjudication. Reported pooled and per direction, with a 10,000-draw bootstrap (seed 42).
- **`NONE`:** a category is `NONE` only with `expected_outcome = escalate` (off-topic). Category accuracy is computed on rows with a real category; escalation correctness uses all rows.
- **Fallback:** `--single-annotator` is kept.

**Files** (`data/independent_test/`, exact names):
- `tickets_AW.csv`, `tickets_SP.csv`. Committed with the header only.
- `second_labels_on_AW_by_SP.csv`, `second_labels_on_SP_by_AW.csv`. Generated by `--make-second-label-files`.
- `adjudication.csv`. Pre-filled by `--make-adjudication-file`.
- `GUIDELINES.md` and the three templates.

**Commands:**
```
python scripts/eval_independent.py --check-only                # any time; works with 0, 1 or 2 files filled
python scripts/eval_independent.py --make-second-label-files   # once both ticket files are complete
python scripts/eval_independent.py --make-adjudication-file    # once both second-label files are complete
python scripts/eval_independent.py                             # final run
```

**Planned fills.** Every `{…}` is filled from `data/independent_test/results/independent_summary.csv`.
- **S6 (Table III, size):** `n_tickets`. **Planned: about 108.**
- **Table III, Source (approved 2026-10-21):** "Written blind by two authors; cross-labelled". In single-annotator mode: "…; single annotator".
- **X4–X7 (Table VIII):**
  - X4 = `n_tickets`;
  - X5 = `with_category.cascade_correct.k`/`n_with_category`;
  - X6 = `with_category.tier2_only_correct.k`/`n_with_category`;
  - X7 = `all.escalated.k`/`n_tickets` (rate).
  - The accuracy denominator excludes `NONE` rows. The caption or X3 says so.
- **Tier-1 cell:** `all.tier1_kept.k`/`n_tickets`.
- **X1, independent part (§VI-E):** "Two authors each wrote about 54 tickets, seven per category plus five off-topic or vague ones ({n} in total). They wrote without consulting the training data, the generation templates or the benchmarks, and without LLM assistance. Each author then labelled the category of the other's tickets without seeing the writer's labels. Cohen's κ before adjudication was {kappa.category} (bootstrap 95% CI [{lo}, {hi}]), and the {disagreements} disagreements were resolved by discussion. The expected outcome (escalate or proceed) was assigned by each ticket's writer."
  - **Single-annotator variant:** replace the last two sentences with "The labels were assigned by each ticket's writer alone (a single annotator), so no inter-annotator agreement is reported."
- **X3, independent part:** written once the results exist. It must include `all.outcome_correct` (escalation correctness on all rows).
