# References to check (Phase 9A, filled in 9A.1)

**Hand-written. Every `\cite` in `main.tex` and `supplement.tex` is listed here.**

**9A.1:** every `references.bib` entry is now **filled** from the guide's reference list, using exactly the fields supplied. Volume, number and pages are present only where the list gave them ([19] FAISS, [20] Wilson, [21] McNemar, [23] Cohen). Edition and publisher are present only for [7] Vovk et al. and [24] Higham. Nothing was looked up or guessed. Where the list gave only a first author ("et al."), the entry reads `and others`.

**Still open:** the bibliographic fields are filled, but checking each characterisation is not done. Confirming that a paper actually supports the sentence citing it is still tracked by the `% TODO-VERIFY` comments in `main.tex`. That work belongs to Phase 9B.

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

Before submission, look each of these up. If a peer-reviewed version exists, replace the arXiv entry with it:

- [5] `conformal_cascade`: Dou et al., "Conformal Cascade" (arXiv, 2026)
- [8] `angelopoulos_gentle`: Angelopoulos & Bates (arXiv:2107.07511, 2021). A later journal/monograph version may exist
- [13] `joren_sufficient`: Joren et al., "Sufficient Context" (arXiv, 2024)
- [14] `crag`: Yan et al., "Corrective Retrieval Augmented Generation" (arXiv, 2024)
- [15] `jung_trust_escalate`: Jung et al., "Trust or Escalate" (arXiv, 2024)
- [16] `jain_judges`: Jain et al., "Beyond Consensus" (arXiv, 2025)
- [17] `maheshwari_synthetic`: Maheshwari et al., "Efficacy of Synthetic Data as a Benchmark" (arXiv, 2024)
- [18] `bge`: Xiao et al., "C-Pack" (arXiv:2309.07597, 2023)
- [22] `qwen25`: Qwen Team, "Qwen2.5 Technical Report" (arXiv:2412.15115, 2024)

## Claims that need a citation and have none yet

- The nasscom hackathon use-case brief (main: Setup). It is not a publication. Decide whether to cite it as a URL or document, and confirm its wording against the brief itself (`% TODO-VERIFY` in `main.tex`).
- The external dataset `Tobi-Bueck/customer-support-tickets` (supplement: External validity). Cite its dataset card. The pinned revision is recorded in `src/experiments/fetch_external_dataset.py`.
- Gemini `gemini-flash-lite` (main: System and zero-shot). Cite the model documentation if the venue requires it.
