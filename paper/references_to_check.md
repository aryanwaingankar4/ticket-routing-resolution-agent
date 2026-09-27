# References to check (Phase 9A, filled in 9A.1)

**Hand-written. Every `\cite` in `main.tex` and `supplement.tex` is listed here.**

**9A.1:** every `references.bib` entry is now **filled** from the guide's reference list, using exactly the fields supplied. Volume, number and pages are present only where the list gave them ([19] FAISS, [20] Wilson, [21] McNemar, [23] Cohen). Edition and publisher are present only for [7] Vovk et al. and [24] Higham. Nothing was looked up or guessed. Where the list gave only a first author ("et al."), the entry reads `and others`.

**9B: characterisations resolved.** Every `% TODO-VERIFY` comment is gone. Each citing sentence now claims **no more than the cited work's title supports**. Where the old wording went further, it was softened, and none was strengthened:
- [1–3]: "LLM-based methods", no longer "classical and LLM-based".
- [5]: the distribution-free guarantees of the title, no longer a set-valued mechanism.
- [6]: "cascades can end in a human decision-maker".
- [16]: agreeableness bias, no longer "reliability" in general.
- [17]: "the efficacy of synthetic data as an evaluation benchmark has been studied", no longer "increasingly used".

[14] CRAG's sentence already named no mechanism and was kept. See `paper/AUDIT_9B.md`.

| # | key | status | cited in | claim it supports |
|---|---|---|---|---|
| 1 | `tickit` | filled (9A.1) | main: Related work, ticket triage | Automated routing/escalation of support tickets studied with LLM-based methods |
| 2 | `paket_ticket` | filled (9A.1) | main: Related work, ticket triage | Ticket classification with LLM classifiers |
| 3 | `redcap_llm_triage` | filled (9A.1) | main: Related work, ticket triage | LLM-based triage of support requests |
| 4 | `jitkrittum_cascade` | filled (9A.1) | main: Related work, cascades | Confidence-based cascades defer from a cheap to an expensive model |
| 5 | `conformal_cascade` | filled (9A.1), **arXiv** | main: Related work, cascades | Conformal variants replace the confidence score with a set-valued criterion |
| 6 | `fanconi_defer` | filled (9A.1) | main: Related work, cascades | Formulations that send inputs to a human expert |
| 7 | `vovk_alrw` | filled (9A.1) | main: Related work, conformal | Split conformal gives marginal coverage under exchangeability |
| 8 | `angelopoulos_gentle` | filled (9A.1), **arXiv** | main: Related work, conformal | Same (tutorial reference) |
| 9 | `tibshirani_covshift` | filled (9A.1) | main: Related work, conformal; Results, calibration distribution. Supplement: Weighted conformal under shift | Weighted conformal restores coverage under covariate shift given the likelihood ratio |
| 10 | `barber_beyond` | filled (9A.1) | main: Related work, conformal | Bounds on coverage loss beyond exchangeability |
| 11 | `gibbs_adaptive` | filled (9A.1). **Resolved:** Gibbs & Candès, NeurIPS 2021 | main: Related work, conformal | "adapts online". This is adaptive conformal inference, **not** Gibbs et al. 2023 (conditional guarantees). The guide settled the question 9A left open |
| 12 | `lewis_rag` | filled (9A.1) | main: Related work, RAG | Definition of retrieval-augmented generation |
| 13 | `joren_sufficient` | filled (9A.1), **arXiv** | main: Related work, RAG | RAG gated on whether retrieved context suffices |
| 14 | `crag` | filled (9A.1), **arXiv** | main: Related work, RAG | Corrective RAG acts when retrieval is poor. The key is Yan et al., not Meta's CRAG benchmark |
| 15 | `jung_trust_escalate` | filled (9A.1), **arXiv** | main: Related work, LLM judges | Judges with human-agreement guarantees that escalate when unsure |
| 16 | `jain_judges` | filled (9A.1), **arXiv** | main: Related work, LLM judges | Reliability of LLM judges (the paper is on agreeableness bias) |
| 17 | `maheshwari_synthetic` | filled (9A.1), **arXiv** | main: Related work, synthetic benchmarks | LLM-generated data used as evaluation benchmarks |
| 18 | `bge` | filled (9A.1), **arXiv** | main: System | The Tier-2 and retrieval embedding model, bge-base-en-v1.5 (C-Pack) |
| 19 | `faiss` | filled (9A.1) | main: System | FAISS inner-product index |
| 20 | `wilson` | filled (9A.1) | main: Setup, statistics | Wilson score interval |
| 21 | `mcnemar` | filled (9A.1) | main: Setup, statistics | McNemar's test for paired proportions |
| 22 | `qwen25` | filled (9A.1), **arXiv** | main: Results, zero-shot | Qwen2.5-3B, the local zero-shot model |
| 23 | `cohen_kappa` | filled (9A.1) | main: Results, groundedness; supplement: Groundedness, the judge and sufficiency | Cohen's kappa |
| 24 | `higham` | filled (9A.1) | **supplement only**: Reproducibility details | The float32 inner-product error bound $\gamma_n = nu/(1-nu)$, section 3.1 |

## References no longer in the main paper's list

- **`higham` [24]** is now cited **only in the supplement**. The γₙ derivation moved there in 9A.1. BibTeX lists only cited entries, so it drops out of the main paper's reference list while staying in the supplement's. No other reference was dropped.

## arXiv entries to check for a published version

**All 9 are for you to check on Google Scholar.** No venue has been added from memory, because nothing here may be invented. For each one, search the exact title. Where Scholar lists a peer-reviewed venue, replace the `@misc` entry with it, giving venue, year and pages only as Scholar shows them. **If no venue is listed, keep the arXiv entry as it is.**

| # | key | search for | note |
|---|---|---|---|
| [5] | `conformal_cascade` | "Conformal Cascade: Distribution-Free Accuracy Guarantees for Multi-Tier LLM Inference" | 2026 preprint, the most recent. May have no venue yet; keep arXiv if so |
| [8] | `angelopoulos_gentle` | "A Gentle Introduction to Conformal Prediction and Distribution-Free Uncertainty Quantification" | Check whether a journal or monograph version is listed |
| [13] | `joren_sufficient` | "Sufficient Context: A New Lens on Retrieval Augmented Generation Systems" | Check for a conference version |
| [14] | `crag` | "Corrective Retrieval Augmented Generation" | Check for a conference version. Do not confuse it with Meta's CRAG *benchmark* |
| [15] | `jung_trust_escalate` | "Trust or Escalate: LLM Judges with Provable Guarantees for Human Agreement" | Check for a conference version |
| [16] | `jain_judges` | "Beyond Consensus: Mitigating the Agreeableness Bias in LLM Judge Evaluations" | 2025 preprint. May have no venue yet |
| [17] | `maheshwari_synthetic` | "Efficacy of Synthetic Data as a Benchmark" | Check for a workshop or conference version |
| [18] | `bge` | "C-Pack: Packaged Resources To Advance General Chinese Embedding" | Check for a conference version. The arXiv number (2309.07597) is from your list |
| [22] | `qwen25` | "Qwen2.5 Technical Report" | Technical reports usually stay arXiv-only. Keep it unless Scholar shows otherwise |

Also add the arXiv identifier for [5], [13], [14], [15], [16] and [17] if you want it printed. Your list gave "arXiv preprint" without a number, so none was added.

## Claims that need a citation and have none yet

- The nasscom hackathon use-case brief (main: Setup). It is not a publication. Decide whether to cite it as a URL or document, and confirm its wording against the brief itself. The sentence records your account (PROJECT_STATUS.md, 2026-09-21) and was never checked against the brief document. Its marker was removed in 9B, and the check is an item on the submit checklist in `AUDIT_9B.md`.
- The external dataset `Tobi-Bueck/customer-support-tickets` (supplement: External validity). Cite its dataset card. The pinned revision is recorded in `src/experiments/fetch_external_dataset.py`.
- Gemini `gemini-flash-lite` (main: System and zero-shot). Cite the model documentation if the venue requires it.
