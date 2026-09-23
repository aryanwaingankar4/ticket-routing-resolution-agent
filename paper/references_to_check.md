# References to check (Phase 9A)

**Hand-written. Every `\cite` in `main.tex` is listed here.** No bibliographic detail in `references.bib` was invented. Each entry has only an author surname and, where you supplied one, a title. Every other field is `TODO`.

Before submission, for each key:
1. Find the paper.
2. Fill in the `.bib` fields.
3. Confirm that the paper actually supports the sentence citing it.

That sentence carries `% TODO-VERIFY` in `main.tex` wherever its characterisation of the work is unchecked.

"Look up" gives what to search for. Where a likely candidate is named, it is a **search hint from memory, not a verified reference**. Confirm it or replace it.

| key | section | claim it supports in `main.tex` | look up |
|---|---|---|---|
| `tickit` | 2 Related work: ticket triage | Automated routing/escalation of support tickets has been studied with LLM-based methods | "TickIt" ticket escalation with LLMs. Confirm full title, authors, venue, year |
| `paket_ticket` | 2: ticket triage | Ticket triage/routing with classical or LLM classifiers | Paket et al., ticket classification/triage. Confirm title and scope |
| `redcap_llm_triage` | 2: ticket triage | LLM-based triage of support or service requests | The "REDCap LLM triage" work you named. Confirm what REDCap refers to here, plus authors, title and venue |
| `jitkrittum_cascade` | 2: cascades and deferral | Confidence-based cascades defer from a cheap to an expensive model | Jitkrittum et al. Hint: "When does confidence-based cascade deferral suffice?" (unverified) |
| `conformal_cascade` | 2: cascades and deferral | Conformal variants replace the confidence score with a set-valued deferral criterion | "Conformal Cascade". Confirm authors, venue and that this is its mechanism |
| `fanconi_defer` | 2: cascades and deferral | Learning-to-defer formulations route inputs to a human expert | Fanconi et al. Confirm the paper intended and that it is learning-to-defer |
| `vovk_alrw` | 2: conformal | Split conformal gives marginal coverage under exchangeability | Vovk, Gammerman, Shafer, *Algorithmic Learning in a Random World* (hint, unverified edition) |
| `angelopoulos_gentle` | 2: conformal | Same (tutorial reference) | Angelopoulos and Bates, "A gentle introduction to conformal prediction..." (hint, unverified) |
| `tibshirani_covshift` | 2: conformal; 5.2 (weighted conformal) | Weighted conformal restores coverage under covariate shift given the likelihood ratio; our 6B reweighting follows it | Tibshirani et al., conformal prediction under covariate shift (hint: NeurIPS 2019, unverified) |
| `barber_beyond` | 2: conformal | Bounds on coverage loss beyond exchangeability | Barber et al., "Conformal prediction beyond exchangeability" (hint, unverified) |
| `gibbs_adaptive` | 2: conformal | Online adaptation of conformal under shift | Gibbs et al. **You named Gibbs et al. without a title.** Decide between adaptive conformal inference (Gibbs & Candès) and conformal prediction with conditional guarantees (Gibbs, Cherian, Candès); the sentence assumes the former |
| `lewis_rag` | 2: RAG abstention | Definition of retrieval-augmented generation | Lewis et al., retrieval-augmented generation for knowledge-intensive NLP (hint, unverified) |
| `joren_sufficient` | 2: RAG abstention | RAG can be gated on whether retrieved context suffices; motivates our 6C rater | Joren et al., "Sufficient Context". Confirm full title, venue, year |
| `crag` | 2: RAG abstention | Corrective RAG acts when retrieval is poor | "CRAG" (Corrective Retrieval Augmented Generation). Confirm authors and that it uses a retrieval evaluator. Note the name clash with Meta's CRAG benchmark |
| `jung_trust_escalate` | 2: LLM judges | Judges with human-agreement guarantees that escalate when unsure | Jung et al., "Trust or Escalate". Confirm full title and venue |
| `jain_judges` | 2: LLM judges | Reliability of LLM judges | Jain et al. **The intended paper is unknown.** Confirm which one and its scope |
| `maheshwari_synthetic` | 2: synthetic benchmarks | LLM-generated data used as evaluation benchmarks | Maheshwari et al. on synthetic data as a benchmark. Confirm title and finding |
| `bge` | 3 System | The Tier-2 and retrieval embedding model, bge-base-en-v1.5 | The BAAI / C-Pack paper introducing BGE (hint, unverified) |
| `faiss` | 3 System | FAISS inner-product index | The FAISS library paper (Johnson et al. or Douze et al., hint, unverified) |
| `qwen25` | 5.4 Zero-shot | Qwen2.5-3B, the local zero-shot model | The Qwen2.5 technical report |
| `wilson` | 4 Setup: statistics | The Wilson score interval | Wilson (1927) score interval (hint, unverified) |
| `mcnemar` | 4 Setup: statistics | McNemar's test for paired proportions | McNemar (1947) (hint, unverified) |
| `cohen_kappa` | 5.5 LLM judge | Cohen's kappa | Cohen (1960) (hint, unverified) |
| `higham` | 7 Reproducibility | The float32 inner-product error bound $\gamma_n = nu/(1-nu)$ | Higham, *Accuracy and Stability of Numerical Algorithms*. The repository cites the 2nd edition, section 3.1. Confirm edition, year and publisher |

## Claims that need a citation and have none yet

- The nasscom hackathon use-case brief (Section 4). It is not a publication. Decide whether to cite it as a URL or document, and confirm its wording against the brief itself (`% TODO-VERIFY` in `main.tex`).
- The external dataset `Tobi-Bueck/customer-support-tickets` (supplement, external validity). Cite its dataset card. The pinned revision is recorded in `src/experiments/fetch_external_dataset.py`.
- Gemini `gemini-flash-lite` (Sections 3 and 5.4). Cite the model documentation if the venue requires it.
