# Black book — where every number comes from

Every number in `Black_Book_v2.docx` is listed here with the committed file (or
script) it was read from. "NUMBERS.md id" refers to `paper/NUMBERS.md`;
"RFP" refers to `paper_docx/RESULTS_FOR_PAPER.md`. Nothing was typed from memory.

## Front matter (abstract)
| Number(s) | Source |
|---|---|
| 0.50, 0.67 thresholds | `src/agent/config.py` (NUMBERS.md `F1.gate.cascade`, `F1.gate.rag`) |
| 9/9 vs 3/9 adversarial | NUMBERS.md `T3.baseline.escalation_correct`, `T3.norag.escalation_correct` |
| 32/45 vs 33/45, p = 1.000 | `T1.cascade.benchmark45`, `T1.tier2only.benchmark45`, `T3.mcnemar.benchmark45` |
| 8%–18% latency saving | `T3.latency.saving.benchmark45` (0.082268), `.deployment175` (0.181951) |
| 40/45 Gemini | `T1.zeroshot_gemini.benchmark45` |
| 31/33 grounded; κ −0.042 | `T14.grounded`, `T14.judge.cohens_kappa` |
| 108 tickets, κ 0.936, 74/98, 32 actionable escalated | `data/independent_test/results/independent_summary.csv` (`n_tickets`, `kappa.category`, `with_category.cascade_correct.k`, `should_proceed.proceeded.k` = 66 of 98) |
| 583, 19.7%, 50.4%, 371/468 | `data/external_zenodo_7648117/realworld_summary.csv` and `..._sens_fileservice_storage.csv` (`distinct.*`) |
| 456 tests | `pytest --collect-only` on this branch: "456 tests collected"; per file in `reports/black_book/data/test_counts_by_file.txt` |

## Chapter 1
| Number(s) | Source |
|---|---|
| Seven categories, ~1,000 tickets in the brief, five fields | `README.md` ("Origin"), `PROJECT_STATUS.md` (nasscom note) |
| 4,000 tickets | `data/synthetic_tickets.csv` (`T19.dataset_rows`) |
| 0.80 clustering threshold | `src/agent/config.py` (`F1.gate.clustering`) |
| 9, 175 + 45, 45, 175, 33, 108, 583, α = 0.10 (hypothesis table) | NUMBERS.md `T19.*`, `T5.*`, `T7.alpha`; RFP Phase 2/3 |
| "75% accurate" | illustrative example in the text, not a result |

## Chapter 2
| Number(s) | Source |
|---|---|
| Facts about each cited work (20,000+ tickets, κ≈0.85, 81–89%, 90 of 6,316, 172 classes, 18 benchmarks, 2%/99.9%/60%, 40%/60%/97%, 5 s vs 65 h, >80%, >96%/<25%, eleven tasks, three QA tasks, six datasets/three tasks) | the paper's published abstract, fetched during this work; see `REFERENCES_CHECK.md` |
| 768 dimensions | `T18.embedding_dim` |
| ~1 ms vs 156.4 ms | `T3.latency.tier1_median_ms` (1.0356), `T3.latency.tier2_median_ms` (156.4024) |
| 100% in distribution; 16/45 | `T1.tfidf_baseline.in_distribution_accuracy`; `T1.tier1only.benchmark45` |
| DistilBERT 18/45 and 21/45 | `T1.distilbert.benchmark45`, `T1.distilbert.reference.benchmark45` |
| Gemini 40/45; judge 30/33 | `T1.zeroshot_gemini.benchmark45`; `T14.judge.raw_agreement` |
| 108, 583 | as abstract |

## Chapter 3
| Number(s) | Source |
|---|---|
| Library versions | `requirements.txt`, `requirements-revision.txt`; Ollama, Docker, Playwright, python-docx versions read from the installed tools on 25 Oct 2026 |
| 15 requests/min, 500/day; 4 s delay | `CLAUDE.md` (Gemini free tier rules) |
| CPU, 12 logical CPUs, Windows 11, Python 3.14.3 | `data/inference_latency_13th-gen-intel-r-core-tm-i5-1334u.csv` |
| 16 GB RAM | stated by the team (not in any result file) |
| Phase dates | `git log --date=short` on this repository |
| 1.04 / 1.74 / 156.4 / 227.0 ms, 151.0×, 143.5 / 127.9 ms, 8.2% / 18.2% | latency CSV above (`median_ms`, `p95_ms`, `cascade_expected_median_ms`); NUMBERS.md `T3.latency.*` |
| 54 tickets, 10⁻¹², 4.578 × 10⁻⁵ | `T18.golden_tickets`, `T18.tier1_tolerance`, `T18.gamma_n` |
| 0.31818032412549274 vs 0.3182984770932253; 11,834 tied for 760 of 5,000 | `T18.runner_91a5b38.adv_08.tier1_confidence.recorded`, `...golden`; `T18.ties.full4000.*` |
| 59 zero-shot calls; 33 + 33; 30 drafts (27 live) | `CLAUDE.md` (Phase 5C: "59 calls for both sets"); `CLAUDE.md` Phase 2B; RFP "Drafts on real tickets" |
| Gemini prices USD 0.10/0.40 and 0.30/2.50 per million tokens | https://ai.google.dev/gemini-api/docs/pricing, checked 25 Oct 2026 |
| 30 of 45 drafted | RFP X2 / `error_analysis_benchmark45_summary.csv` (`drafted` 30) |
| 10,000 tickets/month; 1,500 / 200 tokens; ≈6,700 drafts; USD 1.5 / 6.4 | **assumptions and arithmetic stated in the report** (10,000 × 30/45; 6,700 × 1,500 × 0.10/10⁶ + 6,700 × 200 × 0.40/10⁶ ≈ 1.54; with 0.30/2.50 ≈ 6.37) |

## Chapter 4
| Number(s) | Source |
|---|---|
| 66 templates (9 per category, Database 12); 571/572 per category; High 1,684 / Medium 1,473 / Low 843; median 32 words (23–44); 962 titles; 1,271 resolutions | `reports/black_book/data/dataset_profile.csv`, written by `scripts/make_black_book_figures.py` from `data/synthetic_tickets.csv` |
| 62 of 66 templates; 210 rows | `T8.templates_touched`, `T8.rows_surviving_template_exclusion` |
| Splits table (3,200 / 4,000, 800, 175, 45, 175, 14, 45, 9, 108, 583 (984)) | paper v4 Table III; RFP S1–S8 |
| Attempt 1 buckets 5 / 515 / 280; derived 0.70 | `data/cascade_calibration_attempts.csv` (attempt 1 rows) |
| ECE 0.112 / 0.099 | `T4.tier1.ece`, `T4.tier2.ece` |
| Attempt 3 buckets 152 / 13 / 7 / 3 / 0 and accuracies; 23 kept, 19 correct; 10 kept / 6 correct above 0.60; 0.01-grid peak 25/30 at 0.46 | `data/cascade_calibration_attempts.csv`; `data/paper_revision_oct/cascade_calibration_curve_paraphrased175.csv`; RFP T1 |
| 175/175, 13.3% (6/45), 37.8% (17/45) | `data/rag_similarity_calibration_combined.csv` (`T5.*`); RFP F5 |
| 6 leaks from 3 seeds | RFP E2 (`rag_calibration_top1_similarity.csv`) |
| Safe range 0.6196–0.6704 | `data/adversarial_escalation_results.csv` (adv_04, adv_03); `T5.safe_range_*` |
| 0.35 under MiniLM | `CLAUDE.md` (`OLD_MINILM_THRESHOLD`) |
| Fingerprint 9c9a5cbcb53f | NUMBERS.md header; `GET /health` on 25 Oct 2026 |
| Demo: 57.0%, 0.30, 77.7% | screenshots `reports/black_book/fig/s_demo_*.png`; `reports/black_book/data/triage_response_example.json` (tier1_conf 0.3007, observed_value 0.7771) |
| 456 tests and the group counts 109 / 121 / 66 / 110 / 50 | `reports/black_book/data/test_counts_by_file.txt` (summed by area) |
| 3.18× headroom; "caught two real bugs"; nine occurrences | `T18.headroom.similarity_ratio`; `CLAUDE.md` |
| α ∈ {0.01, 0.05, 0.10, 0.20}; ±0.045 at n = 175 | paper v4 Table IV; `T7.noise_band_2sd` |
| 54 + 54, κ 0.936 [0.882, 0.979], 102/108, outcome κ 1.000, 6 adjudicated | `independent_summary.csv` |
| 2,229; 1,734; 984; 401; 583; English counts per category (597, 85, 158, 144; 601, 103, 46) | RFP Phase 2 ("English filter", "Mapping"); `realworld_summary.csv` |

## Chapter 5
| Number(s) | Source |
|---|---|
| Classifier table | `T1.*` (tfidf_baseline 6/14, tier1only 16/45 and 91/175, minilm 32/45, e5 27/45, distilbert 7/14, 18/45, 21/45, tier2only, cascade, zeroshot) |
| Tier-1 CI [0.232, 0.502] | `T1.tier1only.benchmark45` |
| 35.6 points | `T3.repr_gap_points` |
| Cascade table | paper v4 Table VII; latency CSV |
| Tier-1 share 4/45 (8.9%), 33/175 (18.9%) | `T12.6a.gate_coverage.*`; RFP T1 |
| Zero-shot table, p values, 45 identical prompts, 0 unparseable, Infrastructure 0.5 | 5C clause group in NUMBERS.md (`T1.zeroshot_*`, `T1.mcnemar.*`, `T2.*`) |
| Adversarial table | `data/adversarial_escalation_results.csv` (similarities rounded to 3 dp) |
| 15/45 escalated (33.3%) | RFP X2 |
| Conformal table | paper v4 Table VI; `data/conformal_calibration_results.csv` |
| 0.0009 vs −0.0072; AUC 0.9972; −10.5 / −2.4 points; DiD −0.080 [−0.136, −0.021] | Finding-1 clause group (`T11.7b.*`, `T11.7c.*`) |
| −0.144; −0.122; 5.4 s.d.; 12 comparisons, no signal | `T10.tier1.gap_deployment`; `T7.weighted.tier1.*`; 6A clause group |
| Drift 0.217–0.646, 3.2–38.2× | `T13.realistic_traffic.*` |
| Groundedness table, 91%, 85%, N45 0.640 | `T14.*`, 6C clause group (`T15.*`), `T17.escalated_adequate.*` |
| Independent table | `independent_summary.csv` |
| Real-world table and text (incl. 20.7% = 97/468, 27.8% = 32/115, 203 of 238, 12.1%) | `realworld_summary.csv`, `realworld_summary_sens_fileservice_storage.csv`; RFP "Where the errors come from" |
| Error taxonomy, recall 3/6 – 7/7, 29/45, N26 0.701, N31 0.675, 5 / 8 / 7 | paper v4 Table IX; `error_analysis_benchmark45_summary.csv`; `T17.*`; RFP E1–E5 |
| Accuracy-by-set figure (32/45, 131/175, 74/98, 115/583, 294/583) | read by `scripts/make_black_book_figures.py` from the CSVs above |
| 2.2 points per ticket at n = 45 | 1/45, arithmetic |

## Chapter 6
All numbers repeat those of Chapters 3–5 with the same sources. The
dataset-availability sentence is the approved sentence from RFP ("Approved
sentence, verbatim, for §VI-E").

## Figures
Paper figures reused unchanged: `paper_docx/fig/fig4_cascade_threshold.png`,
`fig5_retrieval_sweep.png`, `fig6_confusion_matrix.png`; `paper/figures/G1_...png`,
`F2_...png`. Every other figure is drawn by `scripts/make_black_book_figures.py`
from the files named above; screenshots by `scripts/capture_black_book_screenshots.py`.
