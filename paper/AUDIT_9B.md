# Phase 9B: pre-submission audit

**Hand-written, 2026-09-28.** The audit covers `paper/main.tex` (compiled on Overleaf at 5 pages, per the author) and `paper/supplement.tex`. It was offline: 0 Gemini calls, 0 Ollama calls, no experiment re-run, and no `NUMBERS.md` value moved.

## Verdict

**SUBMIT, once the author-only items at the bottom are ticked.** Every mechanical check passes. The remaining items need a person: an Overleaf re-compile, because 9B added a few words; the nasscom wording; the arXiv versions; and the guides' sign-off.

## Checks 1–6

| # | Check | Result |
|---|---|---|
| 1 | Every number traces to `NUMBERS.md` via `\nb{}` | **Pass.** `main.tex` has 93 `\nb` uses (85 distinct ids) and `supplement.tex` has 148. **0 undefined.** The only digit-bearing tokens outside `\nb` are names and conventions: Tier-1/Tier-2, Finding~1, Qwen2.5-3B, 3B, bge-base-en-v1.5, top-1, float32, the 95\% of a Wilson interval, the γₙ formula's `(1-nu)`, and the authors' e-mail addresses. Every table is a generated `\input` from `tables/ieee/`; 3 are in main and 24 in the supplement, and none is hand-written. Independently re-derived by a script, and `tests/test_paper_draft.py` (10/10) enforces the same rule. |
| 2 | Every empirical claim points to its support | **Pass, with two number-only claims** (see the map below). Three supplement pointers were added to `main.tex`. One sentence was added to the supplement's *Retrieval gate* section, so that the main text's adversarial counts are supported where the main text points. |
| 3 | The 7 `% TODO-VERIFY` comments | **All resolved and removed.** Four sentences were softened, one was narrowed, and two were kept. None was strengthened. See the table below. |
| 4 | arXiv entries | **9, all listed for you to check on Google Scholar** in `references_to_check.md`. No venue was invented. |
| 5 | Consistency (documents follow files) | **Fixed.** The drift ratio now reads 3.2×–38.2× (`T13.realistic_traffic.ratio_to_null.min/max`) in `CLAUDE.md` and in `PROJECT_STATUS.md`'s named-finding and 4B-1 sections. Mentions that *record* the correction were kept. The health table was refreshed (456 tests, 257 numbers, 62 sources). `PROJECT_OVERVIEW.md` has a SUPERSEDED banner. `supplement.tex`'s author block now matches `main.tex`. |
| 6 | Hygiene | **Pass.** Printed "TODO" count is 0 in both files. Retired (do-not-cite) literals: 0 of 16 appear in either file, and `test_do_not_cite_literals_do_not_reappear` passes. "novel", "state-of-the-art", "the first to/work/study/paper", "bit-identical" and "over-confident" each appear 0 times. The 4 uses of "first" are all ordinal, for example "first-tier classifier" and "The first used the in-distribution split". **The superseded adv_03/adv_05 note figures** (the old-threshold similarities) appear in no prose file under `paper/`. The one grep hit is a coincidental substring of a risk value in the F3 figure-data CSV, not a quote. |

## Claim-to-support map (`main.tex`)

| Section | Claim | Supported by |
|---|---|---|
| Intro / Zero-shot | A zero-shot LLM is more accurate than the trained classifier | Table *T1_classification_main*; supplement *Zero-shot baselines in detail* (Table T1_paired) |
| System | Gate values (cascade, RAG, clustering) | `F1.gate.*`, the frozen production config, and Fig. 1 |
| Setup | Template structure (touched, total, surviving rows) | Supplement *Corpus and evaluation sets*, Table T8 |
| Setup | TF-IDF in-distribution accuracy is at the ceiling | **Number only** (`T1.tfidf_baseline.in_distribution_accuracy`). It sits in no table; stated as a warning, not a result |
| Gates | Retrieval-gate derivation (proceed, leakage, safe range) | Supplement *Retrieval gate* (Fig. F5, Table T5) |
| Gates | Adversarial decided correctly, with and without the gate | Supplement *Retrieval gate* (sentence added in 9B) → Table T3 ablation |
| Gates | The gate errs both ways (N26/N31 pass, N45 escalated) | Supplement *Retrieval gate*, Table T17 |
| Gates | Cascade threshold attempts, the ceiling-effect ECE | Supplement *Cascade calibration and in-distribution reliability* (pointer added in 9B), Tables T4 |
| Finding 1 | Coverage gaps by tier and calibration set | Fig. 2 (coverage) and Table *T7_T10_coverage* |
| Finding 1 | External clauses (7B version shift, 7C blocked, DiD) | Supplement *External validity* (pointer added in 9B), Tables T11, Fig. F7 |
| Finding 1 | Weighted conformal is a partial repair | Supplement *Weighted conformal under shift* |
| Cascade | The cascade is one ticket worse than Tier-2-only, with the latency saving | Table *T3_compact* |
| Cascade | Tier-2 / Tier-1 latency ratio; Tier-1 median ms | **Tier-1 median is number only** (`T3.latency.tier1_median_ms`); Tier-2 median and the savings are in T3_compact |
| Zero-shot | Per-category Infrastructure recall | Supplement *Zero-shot baselines in detail*, Table T2 |
| Groundedness | Grounded share, judge κ, sufficiency 2×2 | Supplement *Groundedness, the judge and sufficiency*, Tables T14 and T15 |
| Named finding | Drift flag rates and ratio to null | Supplement *Drift detection* (pointer added in 9B), Table T13 realistic |
| Methodology / Repro | Nine silent errors; parity and derived bound | Supplement *Methodology lessons*, *Reproducibility details* |
| Limitations | External label ceiling | Supplement *External validity*, Table T11 |

**Unsupported claims: none.** Two numbers are sourced but sit in no table. Both trace to `NUMBERS.md` and are acceptable as stated.

## TODO-VERIFY resolutions

Each resolution rests only on what we know about the cited work, which is its title as supplied in the guide's reference list. Where that could not carry the old wording, the sentence was softened, never strengthened.

| # | Old sentence (claim) | Supported? | Change |
|---|---|---|---|
| 1 | Triage studied "with classical and LLM-based classifiers" [1–3] | Partly: all three titles are LLM-based, and none is classical | → "studied with LLM-based methods" |
| 2 | Conformal Cascade "replaces the confidence score with a set-valued criterion"; Fanconi "sends inputs to a human expert" | The mechanism is not established from the titles | → "give distribution-free accuracy guarantees for multi-tier cascades [5], and cascades can end in a human decision-maker [6]" |
| 3 | RAG "corrected when retrieval is poor" [14] | Yes, per the title "Corrective RAG". The sentence names no mechanism | Kept; comment removed |
| 4 | Jain et al. "study their reliability" | Too broad: the title is about agreeableness bias | → "address an agreeableness bias in judge evaluations" (narrower) |
| 5 | "LLM-generated data is increasingly used for evaluation" [17] | "Increasingly" is not supported by the title | → "The efficacy of synthetic data as an evaluation benchmark has itself been studied" |
| 6 | The nasscom brief's content (categories, fields, synthetic data) | Recorded from the author's account, never checked against the brief document | Kept; comment removed; **author must confirm** (checklist) |
| 7 | Cross-reference to the benchmark-scope correction | Already carried by the supplement's *Zero-shot baselines in detail* | Comment removed |

## Page budget note

> **Superseded 2026-09-28:** the guide now wants **6 pages** (A4, references included), not 5. The mentor revision of `main.tex` was written to that target. The 5-page note below is kept as the 9B record.

9B added about 20 words to `main.tex`: three short supplement pointers plus the softened sentences, which are roughly length-neutral. The 5-page compile predates them. **Re-compile on Overleaf.** If page 5 overflows, revert the three "(supplement, …)" pointers first; that audit trail is also kept here.

## Submit / don't-submit checklist

Mechanical, done in 9B:
- [x] Every number is a `\nb{}` macro traced to `NUMBERS.md`; 0 undefined.
- [x] No printed TODO; 0 `% TODO-VERIFY` left.
- [x] No retired literal, and no forbidden or claim-inflating wording.
- [x] Clause groups whole (`test_paper_draft.py` 10/10); paper parity green.
- [x] Every citation is in `references.bib` with only supplied fields.
- [x] Gates re-run: pytest, adversarial 9/9 with CSV byte-identical, goldens, ablation baseline.
- [x] No secret in git history. Every `GEMINI_API_KEY=` in history is a placeholder, and no `AIza…` key appears.

**Only the author can tick these. Do not submit until all are ticked:**
- [ ] **Overleaf re-compile after the mentor revision (2026-09-28): 6 pages on A4, references included.** The guide's limit moved from 5 to 6 pages; the revision targets about 5.75 pages and is uncompiled.
- [ ] **nasscom brief:** the Setup sentence matches the brief (categories, fields, "synthetic, LLM-generated").
- [ ] **arXiv entries:** each of the 9 checked on Google Scholar and replaced by a published version where one exists (`references_to_check.md`).
- [ ] **Guides' approval** (Dr. Madhvi Saxena, Dr. Aditi Saxena) of the final PDF.
- [ ] **Venue requirements:** check whether the venue needs the nasscom brief, the external dataset card or the Gemini model documentation cited (listed in `references_to_check.md`).
- [ ] The final PDF is attached to the `v1.0` GitHub Release, or placed at `paper/paper_final.pdf`.
