# Results for the paper — revision of October 2026

The source of every value written into
`Calibrated_Escalation_IT_Ticket_Triage_v4.docx`. Each entry gives the
**fill text** (draft, applied in Phase 4), the **value**, the **source file**
and the **command** that regenerates it. Nothing here was typed from memory.

**Commands are run from the project root** with the venv active
(`.\venv\Scripts\Activate.ps1`). Every new script refuses to overwrite its
outputs without `--force`. New result files live in `data/paper_revision_oct/`.
No existing result file, threshold, model or the `v1.0` tag was changed.

Status (2026-10-23): **Phases 1-4 complete; review fixes applied** (label-integrity check, adjudication reasons, stale sentences, 240-word abstract) and **cuts 1, 4, 7, 8 (fold) and 9 applied**; cuts 2, 3, 5 and 6 not applied. `Calibrated_Escalation_IT_Ticket_Triage_v4.docx`
is built by `python scripts/build_v4_docx.py` from v3, which is never written. Every Phase 2
and Phase 3 number in v4 is read by that script from its result file. The 30 real-ticket
drafts are **not labelled**, and the paper says so. See "Phase 4" at the end.

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
- **Second round of decisions (2026-10-21, before any labelling):**
  - The independent set's expected outcome is cross-labelled too (`second_expected_outcome`). κ is reported for category and for outcome separately, and both are adjudicated.
  - The hand-off labels are defined as below.
  - The dataset-availability sentence below is approved for §VI-E and Limitations.

**Approved sentence, verbatim, for §VI-E (Phase 4):** "We found no public, real, English IT ticket dataset whose categories match our seven teams; the Zenodo corpus was the closest real option, and our hand-written set covers all seven." It goes in §VI-E right after the X1 real-world part, before the results.
- **Reflected in Limitations,** as one clause added to the L1 real-world part (draft): "…two teams only, because no public, real, English IT ticket dataset with categories matching our seven teams was available: Database, Infrastructure, …"
- **Source:** this is the authors' statement about their dataset search, not a measured number, so no result file backs it. Phase 2 compared candidate datasets and chose this one (option A) on 2026-10-19. **The list of candidates and why each was rejected is not recorded in the repository.** If a reviewer may ask, AW should add a short note listing them.

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
- **Then, only if `draft_labels.csv` is filled:** "In the 30 sampled drafts, labelled by one author, the retrieved tickets did not fit the ticket in {context_mismatch.k} cases. In {handoff.yes.k} of those the model declined to give a fix and handed the ticket to a human, and in {handoff.no.k} it gave a fix anyway. {grounded.yes.k} of the 30 drafts were grounded in the retrieved tickets." Source: `draft_labels_summary.csv`. Optionally add the Wilson interval for `handed_off_given_mismatch`.
  - **Hand-off definition (AW, 2026-10-21):**
    - `yes`: the retrieved tickets did not fit, and the draft declined to give a fix and handed off to a human.
    - `no`: the retrieved tickets did not fit, but the draft still gave a fix.
    - `na`: the retrieved tickets did fit, so a fix is expected.
    - A generic "a human should double-check" line alone is not a hand-off.
- **If it is not filled:** "The 30 drafts sampled from these tickets were not labelled."

**§VII-B, added after approved change 6:** "The real tickets show the same at scale. Of the 468 misrouted distinct texts, 371 passed the retrieval gate and reached the drafter. The gate escalated 97 misrouted and 32 correctly routed texts, so its similarity score does not track routing correctness. Tier 1's confidence does not either: it kept 238 of the 583 texts at the 0.50 threshold, and 39 of those were correct."

**A1 (abstract):** "On 583 distinct real user tickets from a public dataset, covering two of the seven teams, the pipeline routed 19.7% correctly (119 of all 984 tickets), equal to Tier 2 alone (p = 1.0). Accuracy rises to 50.4% if file-share requests are mapped to a different team. Tier 1 answered 238 tickets but only 39 correctly, and the retrieval gate escalated 22.1% but let 371 of 468 misrouted tickets reach the drafter, so a classifier trained only on synthetic templates does not transfer to real tickets."

**C1 (Conclusion):** "On real user-written tickets for the two teams we could test, the classifier did not transfer: 115 of 583 distinct texts (19.7%; 119/984 over all tickets) were routed correctly, Tier 2 alone did no better (p = 1.0), and Tier 1 answered 238 of them with only 39 correct. The retrieval gate did not filter the misroutes (371 of 468 reached the drafter), and accuracy depends on the category mapping (50.4% under the alternative). This is the paper's recurring finding at its starkest: the synthetic training data, not the choice of method or threshold, limits what the pipeline can do."

**L1, real-world part (Limitations):** "The real-world set removes the dependence on generated evaluation text for two teams only, because no public, real, English IT ticket dataset with categories matching our seven teams was available: Database, Infrastructure, Network, Security and Storage remain untested on real tickets. Its accuracy depends strongly on how the source categories are mapped to ours (19.7% or 50.4%), and the tickets come from a single organisation. The English filter also let a few non-English tickets through, because their system-generated subject lines were in English."

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
- **Cross-labelling:** each labels **both the category and the expected outcome** of the other's tickets blind (`second_label`, `second_expected_outcome`).
- **Disagreements:** resolved by discussion and recorded in `adjudication.csv`, one row per ticket where either field disagrees. Columns: `ticket_id, author_label, second_label, final_label, author_outcome, second_outcome, final_outcome, resolved_by, reason`, with `resolved_by` = `AW+SP`.
- **κ:** Cohen's κ for category and, **separately**, for outcome, on the first, independent labels, before adjudication. Each is reported pooled and per direction, with a 10,000-draw bootstrap (seed 42).
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
- **X1, independent part (§VI-E):** "Two authors each wrote about 54 tickets, seven per category plus five off-topic or vague ones ({n} in total). They wrote without consulting the training data, the generation templates or the benchmarks, and without LLM assistance. Each author then labelled the category and the expected outcome (escalate or proceed) of the other's tickets without seeing the writer's labels. Before adjudication, Cohen's κ was {kappa.category} for category (bootstrap 95% CI [{lo}, {hi}]) and {kappa.outcome} for outcome ([{lo}, {hi}]). The {disagreements.category} category and {disagreements.outcome} outcome disagreements were resolved by discussion."
  - **Single-annotator variant:** replace the last two sentences with "The labels were assigned by each ticket's writer alone (a single annotator), so no inter-annotator agreement is reported."
- **X3, independent part:** written once the results exist. It must include `all.outcome_correct` (escalation correctness on all rows).

## Phase 3 — results (2026-10-23)

**Inputs** (`data/independent_test/`):
- `tickets_AW.csv` and `tickets_SP.csv`: 54 tickets each, 7 per category plus 5 `NONE`/escalate.
- `second_labels_on_AW_by_SP.csv`, `second_labels_on_SP_by_AW.csv` and `adjudication.csv`.

**Provenance notes:**
- **AW's statement (2026-10-23):** "An earlier --check-only run used the wrong second-label files by mistake; the files now in place are the original labels." κ below is computed from the files now in place.
- **File times (observation):** `tickets_AW.csv` was last saved at 02:28, after `second_labels_on_AW_by_SP.csv` (02:24). The script confirms the ticket text still matches the second-label file, but it cannot see whether a label column was edited after cross-labelling.
- **Label integrity check (2026-10-23, resolves the file-times note):** both ticket files were compared, row by row on `ticket_id`, with AW's read-only backups `../ticket_backup/tickets_AW.csv.xlsx` (saved 22 Oct, 20:02) and `tickets_SP.csv.xlsx` (20:13). Both backups predate the second-label files (23 Oct, 02:04–02:05).
  - **Result:** the same 54 IDs in each file. 10 `true_category` cells differ, and only in case: `none` in the backup, `NONE` now (AW050–AW054, SP050–SP054). The script requires upper-case `NONE`.
  - **Everything else matches:** no category or `expected_outcome` value differs in meaning, and no title, description, author or note differs.
  - **Backups untouched:** their sha256 was the same before and after the read. AW reviewed this and said to continue.
- **Adjudication:** each disagreement was resolved by one person, not by discussion. `resolved_by` is `SP` for AW's tickets and `AW` for SP's. In all 6 rows the final label is the writer's own. The paper says "adjudicated, and in all 6 the writer's label was kept", not "resolved by discussion".
- **Reasons that don't match their final label:** some recorded reasons do not obviously support the label that was kept. AW007's reason is "sign-in issue is for app", but the final is Access Management. SP007's reason is "forgot the password", but the final is Network.
  - **Fixed by AW (2026-10-23):** only the two `reason` cells changed. AW007 now reads "any issue related to login is solved by access management", and SP007 "password portal issue". No label changed. The re-run summary is byte-identical to the previous one.

**Command:** `python scripts/eval_independent.py` → `results/independent_summary.csv` and `independent_tickets_routed.csv`. It was run twice; apart from the adjudication keys renamed in between, all values are identical.

| Quantity | Value | Key |
|---|---|---|
| Tickets / with a team | 108 / 98 | `n_tickets`, `n_with_category` |
| κ category (before adjudication) | 0.936 [0.882, 0.979], 102/108 agreement | `kappa.category*` |
| κ category per direction | 0.936 each way (51/54) | `kappa.category.on_*` |
| κ outcome | 1.000, 108/108 | `kappa.outcome*` |
| Disagreements | 6 category, 0 outcome; 6 adjudicated, writer's label kept in 6 | `disagreements.*`, `adjudicated*` |
| Cascade correct | 74/98 = 75.5% [66.1, 83.0] | `with_category.cascade_correct.*` |
| Tier 2 alone correct | 75/98 = 76.5% [67.2, 83.8] | `with_category.tier2_only_correct.*` |
| Recall (cascade) | AM 10/14, App 9/14, DB 14/14, Infra 9/14, Net 12/14, Sec 8/14, Sto 12/14 | `with_category.recall.cascade.*` |
| Escalated at 0.67 | 40/108 = 37.0% [28.5, 46.4] | `all.escalated.*` |
| Should escalate → escalated | 8/10 | `should_escalate.escalated.*` |
| Should proceed → proceeded | 66/98 (so 32 actionable tickets escalated) | `should_proceed.proceeded.*` |
| Outcome correct | 74/108 = 68.5% [59.3, 76.5] | `all.outcome_correct.*` |
| Misrouted: escalated / drafted | 8 / 16 of 24 | `with_category.misclassified*` |
| Tier-1 share | 27/108 = 25.0% | `all.tier1_kept.*` |

## Phase 4 — v4.docx (2026-10-23)

**Build:** `python scripts/build_v4_docx.py` (`--force` to rebuild). It reads v3 and writes v4.
- **Fatal checks:** the build stops if any target text is not found exactly once, or if any source value does not match. It also checks that every R1 command names a script that exists.
- **Guarantees:** no marker or shaded run is left, and v3 is byte-identical to `HEAD` (checked with git).

**What was applied:**
- **Markers:** all 44 filled. S3 was removed together with its row (approved merge). Marker runs keep Times New Roman and their size, and lose the bold and the yellow shading.
- **Approved edits:**
  - 1a/1b/1c: noise band in Table IV, §VI-A and the Table VI caption;
  - 2: Table III merge;
  - 3a/3b: the 3,200/4,000 correction;
  - 4: §V-A bucket wording, the dip sentence and the fine-grid sentence;
  - 6: the §VII-B limitation;
  - E5 wording;
  - Table III source: "Written blind by two authors; cross-labelled";
  - the Table VIII Tier-1 column;
  - the dataset-availability sentence (§VI-E) and its Limitations clause;
  - Appendix C (Table X);
  - references [24] Pereira and [25] Lui & Baldwin.
- **Figures:** Figs. 4–6 at 3.3 in, using Fig. 1's paragraph format (centred, keep-with-next).
- **"Not labelled":** X3 ends with "The 30 drafts generated for a sample of the real tickets were not labelled." The build fails if `draft_labels.csv` contains any label.

**Where the applied text differs from the drafts above (all small; each is in the commit):**
- **A1:** "equal to Tier 2 alone" became "indistinguishable from Tier 2 alone". 115 and 114 are not equal; p = 1.0.
- **Table VI caption (1c):** the drafted "±2 s.d. is the binomial standard deviation" was wrong by a factor of two. It now reads "The ±2 s.d. column is twice the binomial standard deviation of a coverage estimate at n = 175" (2 × 0.02268 = 0.045).
- **X1, independent part:** "resolved by discussion" became "adjudicated, and in all 6 the writer's label was kept", which is what `adjudication.csv` records.
- **§VII-B:** one clause added for the independent set: "16 of the 24 misrouted independent tickets reached the drafter".
- **A1, C1 and L1:** each gains an independent-set clause in front of the approved real-world text; X3 gains the independent paragraph. All numbers come from `independent_summary.csv`.
- **R1:** adds the revision requirements file and a step 6, "Extended evaluation", listing the Phase 2 and Phase 3 scripts.

**Possibly stale sentences.** AW asked on 2026-10-23 for all three to be updated, and they are applied in v4, with numbers read from the result files. The original note follows:
1. §II "Two of the ticket-triage studies evaluate on real tickets, while our main evaluation does not (Section VIII)." This is still true, since the main evaluation is synthetic, but it reads oddly now that a real-ticket set exists.
2. §VIII "No real ticket stream has passed through the system…". Real tickets have now been routed offline, though not as a live stream.
3. §IX "established on a synthetic corpus and small, mostly LLM-generated evaluation sets". The 583 real and 108 hand-written tickets now outnumber the LLM-generated evaluation tickets.

**Applied wording for those three sentences (2026-10-23):**
- **§II:** "Two of the ticket-triage studies evaluate on real tickets, while our main evaluation is synthetic, supplemented by 108 hand-written tickets and 583 real tickets that cover only two of our seven teams (Section VI-E)."
- **§VIII:** "The system has not been deployed: 583 real tickets were routed offline, but no live ticket stream has passed through it and no help-desk user has acted on its output, …"
- **§IX:** "These conclusions are established on a synthetic corpus, small LLM-generated evaluation sets, 108 hand-written tickets and 583 real tickets that cover only two of the seven teams, and they should not be read as …"

**Abstract (2026-10-23):** rewritten to 240 words; the build fails above 240.
- **Why the whole abstract:** the build replaces the whole v3 abstract body, not just A1, so the new text keeps the abstract's bold. In the first v4, A1 had come out in regular weight.
- **What it keeps:** every point AW required: 115/583 = 19.7% with 119/984, 371 of 468 misrouted reaching the drafter, 50.4% under the other mapping, Tier 1 238 with 39 correct, p = 1.0 against Tier 2, two teams, and the recurring finding. It also keeps the independent-set result.

**Length (measured in Word, A4, two columns, 2026-10-23):**
- **Totals:** 9 pages. v3 was 7.
- **Main text plus references** ends at the foot of the right column on page 8: References start at page 8, right column, 1.8 in, and Appendix A starts at 10.2 in. So the main text is already about 8 pages including references, with no margin.
- **Appendices A–C** fill page 9.

**Proposed cuts.** AW's decision (2026-10-23): apply 1, 4, 7, 9, and 8 as a one-sentence fold; do NOT apply 2, 3, 5 or 6. See "Cuts applied" below. Savings are estimates in column-inches, measured from the Word layout; a full column is about 9.5 in.

| # | Cut | Where it goes | Est. saving |
|---|---|---|---|
| 1 | Move Table IV (configurations) to a new appendix and keep one sentence in §IV-D pointing to it | Appendix | ~4.5 in |
| 2 | Drop Fig. 2. Table VI already gives the same coverage gaps with the noise band. Keep the table | — | ~2.9 in |
| 3 | Drop Table V. Every value is already in the §V-B text (175/175, 13.3%, 37.8%, safe range, 9/9 vs 3/9) or in Table VII (latencies) | — | ~2.7 in |
| 4 | Move Table II (verification checks) to an appendix; keep §III-G as two sentences | Appendix | ~2.4 in |
| 5 | Tighten §VI-A: keep bullet 1; merge bullets 2–3 (external replication inconclusive; post-hoc accuracy evidence) into one sentence, keeping the three clauses together; move the AUC/band details to an appendix | Appendix | ~1.2 in |
| 6 | Shorten §VI-E: move the per-team recall and all-ticket CIs into Table VIII and Appendix C, so the prose keeps only the headline numbers | Table VIII / App. C | ~1.0 in |
| 7 | Move "Methodological lessons" (§VI-F, second paragraph) to an appendix | Appendix | ~1.4 in |
| 8 | Drop §III-F (automation candidates, which no result in the paper evaluates) or fold it into §III-A as one clause | — | ~0.6 in |
| 9 | Scale Fig. 1 (full width) to about 85% height | — | ~0.8 in in each column |

- **Total if all are applied:** about 1.7 columns, a little under one page. Main text plus references would end around the middle of page 8, with the appendices on pages 8–9.
- **Two cuts give most of the margin:** #1 + #2, about 7.4 in, are enough for comfortable room under 8 pages.
- **Nothing is cut** from the results the mentor asked for: thresholds, error analysis, reproducibility, and the new test sets.

### Cuts applied (2026-10-23, AW's decision)

`python scripts/build_v4_docx.py --force`, section "8b". Content is moved, never deleted, and moved tables keep their numbers.

- **Cut 1:** Table IV moved to **Appendix D: Model Configurations and Hyperparameters**. §IV-D now reads "Table IV (Appendix D) gives the configuration of every component."
- **Cut 4:** Table II moved to **Appendix E: Verification Checks**. §III-G is now two sentences: "…without raising any error, so the checks in Table II (Appendix E) run before any result is accepted. Routing decisions are compared exactly, and floating-point values within tolerances derived from their arithmetic."
- **Cut 7:** the "Methodological lessons" paragraph moved, verbatim, to **Appendix F: Methodological Lessons**. §VI-F's first paragraph ends with the pointer "Methodological lessons from these experiments are summarised in Appendix F."
- **Cut 8, folded:** the §III-F heading and its paragraph were removed, and the paragraph is kept as one sentence at the end of §III-A: "An offline job also clusters resolved tickets … as candidates for automation." "G. Verification" was relettered to "F. Verification"; nothing in the paper cited §III-F or §III-G.
- **Cut 9:** Fig. 1 scaled to 85%, both dimensions, aspect kept.
- **Layout fix, no text change:** Table III's rows are kept together (keep-with-next). Without it, the table broke after its first row.
- **Not applied:** cuts 2 (Fig. 2), 3 (Table V), 5 (§VI-A) and 6 (§VI-E).

**Cross-reference check, now part of the build:**
- Every cited Table (I–X), Fig. (1–6), Appendix (A–F) and Section (III-C, III-D, V, V-A, V-B, VI-A, VI-C, VI-D, VI-E, VIII) exists in the document.
- No "??" and no "Error! Reference" text appears.
- The document has no Word fields, so nothing can render as "??" later.

**Layout after the cuts (Word, A4, two columns):**
- **Total:** 9 pages.
- **Main text plus references:** References now start at page 8, left column (2.7 in) and end in the right column, where Appendix A starts at 1.3 in. So main text plus references is about 7.6 pages, against about 8.0 before the cuts.
- **Appendices A–F:** the rest of page 8 and page 9.

### Final edits (2026-10-23, AW)

- **Contribution 4 (§I):** added as list item 4, formatted like items 1–3 with a bold lead phrase. Text: "**An extended evaluation** on 108 tickets written and cross-labelled by two authors (κ = 0.936) and 583 real tickets from a public dataset, reported with its negative result." The numbers are read from `independent_summary.csv` and `realworld_summary.csv`.
- **Abstract:**
  - "(119 of all 984)" is now "(12.1% over all 984 tickets)". 12.1% = `all.cascade_correct` 119/984.
  - The new wording is one word longer, so to stay at or under 240 words "so it is reported as a baseline" became "so it serves as a baseline".
  - The abstract is now exactly 240 words.
- **Layout:** still 9 pages in Word. References start at page 8, left column, and Appendix A at the top of page 8's right column.
