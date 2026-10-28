# Black book — Phase 0 outline

> **Superseded (28 Oct 2026).** The author changed the target to 50–60 pages in total and
> asked for the build to start from the edited base file. The result is
> `Black_Book_v2.docx` (57 pages); see `DECISIONS.md`, section "Revision v2". This file is
> kept as the record of the first plan.

Written 2026-10-25. The author was away and asked for all phases to run without
stopping, so this outline was not reviewed before Phase 1. Every decision taken
without review is in `DECISIONS.md`.

## Title

**Chosen (as requested):** *Calibrated Escalation for IT Ticket Triage: A Two-Gate
Retrieval-Augmented Pipeline*

Alternatives (not used; offered for the author's choice):
1. *Knowing When Not to Answer: Calibrated Escalation Gates for IT Support Ticket Triage*
2. *A Two-Gate Triage Pipeline for IT Support Tickets: Confidence-Based Routing,
   Retrieval-Gated Drafting and Human Escalation*

## Page budget

Target: about 80–83 pages from the cover to the end of the References.
The appendices (mandatory forms the author inserts, the paper, the user manual)
come after that and are counted separately (see DECISIONS.md, D-03).

| Part | Pages | Main sources |
|---|---|---|
| Cover, certificate, declaration, acknowledgement | 4 | template, CLAUDE.md, README |
| Abstract + keywords | 1 | paper v4, RESULTS_FOR_PAPER.md |
| TOC, list of tables, list of figures, abbreviations | 8 | generated Word fields |
| **Ch. 1 Introduction** | 7 | README (origin), paper v4 §I, RESULTS_FOR_PAPER |
| 1.1 Introduction | 2.5 | |
| 1.2 Problem statement (nasscom Use Case 1) | 1.5 | README "Origin", PROJECT_STATUS (nasscom note) |
| 1.3 Scope | 1 | CLAUDE.md, paper §VIII |
| 1.4 Research hypotheses | 1 | paper v4, NUMBERS.md |
| 1.5 Objectives | 0.5 | |
| 1.6 Organisation | 0.5 | |
| **Ch. 2 Literature review** | 11 | 25 paper refs + 15 verified additions (REFERENCES_CHECK.md) |
| 2.1 Background (7 subsections) | 7 | |
| 2.2 Summary, survey table, gap | 4 | |
| **Ch. 3 SRS** | 9 | requirements.txt, latency CSV, api.py, git log |
| 3.1 Software tools | 1.5 | requirements.txt |
| 3.2 Hardware | 0.5 | latency CSV (CPU, cores, OS) |
| 3.3 WBS + phase table + timeline | 2.5 | git log, PROJECT_STATUS.md |
| 3.4 Functional requirements | 1.5 | src/service/api.py, src/agent/ |
| 3.5 Non-functional requirements | 1.5 | NUMBERS.md T3, T18, CLAUDE.md |
| 3.6 Cost estimation | 1.5 | actual ₹0; Gemini list price (checked online) |
| **Ch. 4 Methodology** | 19 | |
| 4.1 Dataset | 3 | data/synthetic_tickets.csv, generate_dataset.py, Table III of v4 |
| 4.2 Architecture, agents, sequence | 3 | src/agent/*, src/service/api.py |
| 4.3 Cascade and 0.50 | 2.5 | cascade_calibration_attempts.csv, curve CSV, Fig. 4 |
| 4.4 Retrieval, 0.67, drafting prompt | 3 | rag sweep CSV, adversarial CSV, prompts.txt |
| 4.5 Conformal and drift | 2 | conformal + drift result files, NUMBERS T7/T9/T13 |
| 4.6 API, demo, deployment | 2.5 | api.py, Dockerfile, workflows, screenshots |
| 4.7 Testing and verification | 1.5 | pytest collection (456), goldens, NUMBERS T18 |
| 4.8 Extended evaluation design | 1.5 | RESULTS_FOR_PAPER Phase 2/3 |
| **Ch. 5 Results and discussion** | 17 | NUMBERS.md, RESULTS_FOR_PAPER.md, result CSVs |
| 5.1 Benchmark accuracy, classifier comparison | 2 | T1 |
| 5.2 Cascade ablation and latency | 1.5 | T3 |
| 5.3 Zero-shot baselines | 1.5 | 5C clause group |
| 5.4 Gates and adversarial set | 1.5 | T5, adversarial CSV |
| 5.5 Conformal coverage (Finding 1) | 2 | finding1 clause group, Table VI of v4 |
| 5.6 Drift | 1 | T13 |
| 5.7 Groundedness, judge, sufficiency | 1.5 | T14, 6C group |
| 5.8 Independent set | 1.5 | independent_summary.csv |
| 5.9 Real-world set | 2 | realworld_summary*.csv |
| 5.10 Error analysis | 1.5 | error_analysis files, T17 |
| 5.11 Summary | 1 | |
| **Ch. 6 Conclusion and future scope** | 3 | |
| References (IEEE) | 3 | REFERENCES_CHECK.md |
| *Appendices (counted separately)* | ~16 | |

## Figures (all numbered by chapter; regenerated from files)

New (scripts/make_black_book_figures.py): ticket lifecycle; literature map; WBS;
project timeline; dataset generation flow; category and priority distribution;
ticket length histogram; layered architecture; agent/orchestrator components;
/triage sequence; cascade decision flow; split-conformal procedure; adversarial
similarities against the gate; CI/CD and container pipeline; extended-evaluation
design and Zenodo funnel; latency; benchmark per-team recall; accuracy across
evaluation sets; independent per-team recall; real-world confusion.

Reused: paper Fig. 1 (pipeline), Figs. 4–6 (cascade curve, retrieval sweep,
confusion matrix), paper F2, F3, F4, F6, F7, G1.

Screenshots (scripts/capture_black_book_screenshots.py, Playwright): /docs,
a /triage response, the Streamlit demo on a proceed ticket and on an escalated ticket.

## Tables

Brief-to-component map; hypotheses; literature survey (18 works); related-work
comparison; software stack; hardware; phases; functional requirements;
non-functional requirements; actual cost; estimated cost; sample tickets;
dataset splits; agents; cascade calibration attempts; attempt-3 buckets;
adversarial set; drift tests; API endpoints; test suite by file; verification
checks; Zenodo mapping; classifier comparison; cascade vs Tier 2; zero-shot;
gates; conformal gaps; drift realistic traffic; groundedness; independent set;
real-world; sensitivity; error taxonomy; hypothesis outcomes; configurations.
