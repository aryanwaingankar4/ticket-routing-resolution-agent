# Camera-ready revision notes: IConAECC 2026, paper #1251

**Paper:** *Calibrated Escalation for IT Ticket Triage: What the Calibration
Distribution Decides*.

**Reviews:** 6.90, 6.60 and 6.90 out of 10, all positive. Reviewer 3
recommended minor revisions. Accepted.

**Revision date:** 2026-11-14, on branch `iconaecc-2026`.

## Ground rules kept

- **No new experiment and no new number.** The build
  (`scripts/build_iconaecc_docx.py`) still fails on any number that does not
  occur in the mentor-approved v4. The same check now also runs on the LaTeX
  output.
- **No result, limitation or the dataset sentence was weakened or removed.**
  The dataset sentence is unchanged, and the build checks for it verbatim.
- **Unchanged:**
  - the IEEE template and its styles;
  - the Acknowledgment;
  - every table;
  - every reference, with all 25 still cited.
- **Every change is made in the build script.** Nothing was edited by hand
  in the .docx file.

## Length

**The Word paper measured 7 pages.** Microsoft Word exported it to PDF and
the page count was read from the PDF metadata. That was done twice, the limit
set. The final build has one later layout fix, which is not measured (see
below).

| Build | Pages | Where the last reference ends |
|---|---|---|
| All additions | 7 | about 120 pt into page 7, left column |
| After tightening | **7** | about 30 pt into page 7, roughly three reference lines |

**After the second measurement, one layout fix was made.** In the six-author
block, columns 2 and 3 sat about 19 pt lower than column 1. Each row is now a
single paragraph, so all three columns start level.

Word places the text more tightly after that fix:
- the abstract now starts 25.2 pt higher on page 1, read from Word's
  positions, not from a page count;
- that is close to the 30 pt overflow, and page 1's two columns both gain the
  space.

**So the final Word file is 6 or 7 pages, and that count was not measured.**
Open it in Word to see which. The conference allows pages 7–8 in the final
version at USD 10 each.

**Tightening.** It saved about 105 words and about 38 pt of figure height:
- Fig. 2 width went from 3.3 in to 3.05 in, and Fig. 3 from 2.8 in to 2.55 in.
- Wording only was cut; no result or number was removed. Each cut is listed
  under "Tightening" below.

**The final paper is the LaTeX version, at 6 pages.** It was compiled on
Overleaf and fitted to 6 pages there; see "Final LaTeX (6 pages)" at the end.
The Word page count above no longer decides anything.

## Author block and title

| | Before (v4 block) | After (portal, exactly) |
|---|---|---|
| Title | *Calibrated Escalation for IT Ticket Triage: What the Calibration Distribution Decides* | unchanged; the build now checks it against the portal title |
| Order | Aryan Manoj Waingankar, Shreyansh Pandey, Parth Bagle, Swapnil Samrat, Dr. Madhvi Saxena, Dr. Aditi Saxena | **Aryan Waingankar, Shreyansh Pandey, Parth Bagle, Swapnil Samrat, Madhvi Saxena, Aditi Saxena** |
| Spelling | "Aryan Manoj Waingankar"; "Dr." before both guides | "Aryan Waingankar"; no "Dr.", as on the portal |
| Affiliation | one shared block: Department of Artificial Intelligence and Machine Learning / Symbiosis Institute of Technology, Symbiosis International (Deemed University) / Pune, India | each author: *Symbiosis Institute of Technology, Pune* / India |
| E-mail | one shared list of all six | each author's own address under their name, unchanged (matched from v4 by name) |
| Layout | names on two lines, then the shared block | the IEEE template's six-author layout, two rows of three |

The Word file's document properties now list all six authors.

## Changes for the reviewers (text only)

| # | Reviewer point | Change | Where |
|---|---|---|---|
| 1 | Synthetic-to-real gap (R1, R2, R3) | Future work: domain adaptation by few-shot fine-tuning on a small labelled real set, prompt tuning, and self-training on unlabelled real tickets. | IX, new "Future work." paragraph |
| 2 | Real-world coverage (R1, R3) | §VI-E now says that five of our seven teams have no real-ticket evaluation. The Limitations sentence now reads "covers only two of our seven teams … so the other five (Database, Infrastructure, Network, Security and Storage) have no real-ticket evaluation". Future work: a real evaluation of all seven teams with a partner IT service desk. | VI-E, VIII, IX |
| 3 | Alternative mapping, 50.4% (R2, R3) | It changes one assignment: Fileservice goes to Storage, where our generator files share-permission faults, instead of Access Management. The jump is called a semantic gap between synthetic and real categories: the classifier files real file-share access requests as Storage (Storage recall 203/237, in the supplement's Table S-I). The mapping choice is stated to be a sensitivity analysis, not a deployable setting, and a production system would need labels in its own categories. Future work: hierarchical classification (coarse category first, then team) or zero-shot prompting for unseen categories. | VI-E, IX |
| 4 | 371 of 468 misrouted tickets reached the drafter (R2) | The text now calls this "a structural limitation". It explains, by construction, why the retrieval gate cannot catch misrouting: the gate's only input is the top-1 similarity to past tickets of any team. It therefore checks whether a similar past ticket exists, not whether the predicted team is correct. This uses only facts already in the paper (Methodology; the 97/32 escalation split) and the retriever code, which searches the whole index. It proposes a cheap consistency check (predicted team against the teams of the top-5 retrieved tickets) and says plainly that it has not been tested. | VII, "Incorrect escalation decisions." |
| 5 | Gates in changing environments (R1, R2) | Future work: periodic recalibration on a rolling window of recent labelled tickets, drift monitoring that triggers recalibration, and weighted [9] or online [11] conformal methods. These existing references are cited again here. | IX |
| 6 | Deployment and operational metrics (R1) | Future work: a shadow deployment, running alongside human triage without acting, to measure resolution time, re-routing rate and cost. It replaces the old "live deployment" item. | IX |
| 7 | Computational trade-offs (R1, R2) | Collected in one place: "151.0× as much as Tier 1 per ticket (156.4 ms against 1.04 ms median)", the 8% and 18% median-latency savings, and a new sentence that a ticket escalated at the retrieval gate makes no language-model call, so drafting is paid only for tickets that pass it. | VI-B |
| 8 | Novelty framing (R3) | New closing sentence in Related Work: confidence-based [4] and conformal [5] cascades set thresholds on held-out in-distribution data, and split conformal guarantees [7], [8] assume exchangeable calibration and test tickets. We choose and re-check each threshold on deliberately shifted sets (paraphrased, adversarial, human-written, real) and report where the guarantees fail (Section VI-A). | II |
| 9 | Better groundedness evaluation (R3) | After the LLM-judge result (κ = −0.042), new future work: claim-level checks against the retrieved text, NLI-based entailment scoring, a small human-labelled set to calibrate any automated judge, and reporting judge–human agreement before using a judge. | VI-D |
| 10 | Production implications (R3) | New sentence in the conclusion: "Before production use, both thresholds must be recalibrated on the target desk's own tickets." | IX |

**The conclusion is now two paragraphs**: the conclusion itself, then a
run-in "Future work." paragraph. The old one-sentence future-work list is
folded into the new paragraph: live deployment, more human-labelled drafts
with sufficiency labels, and a larger real-ticket benchmark.

## Tightening (wording only, to offset the additions)

- **Related Work:**
  - dropped "the judge we audited shows the failure these works warn about"
    (the audit is reported in VI-D);
  - "The value of synthetic data as a benchmark has itself been studied" →
    "Synthetic data's value as a benchmark has been studied".
- **Experimental Setup:** "4,000 tickets with title, description, category,
  resolution and priority fields" → "4,000 tickets (title, description,
  category, resolution, priority)".
- **VI-C:** "Gemini (gemini-flash-lite)" → "Gemini". The model name is still
  given in Section III-A.
- **VI-E:** the list of the five uncovered teams moved to Limitations, where
  it was already given. VI-E now says "five of our seven teams have no
  real-ticket evaluation (Section VIII)".
- **IX:** dropped the opening sentences "A triage pipeline is valuable only if
  it knows when not to answer. Our two separately calibrated gates decide
  when …". The paragraph now opens "Of our two separately calibrated gates,
  the retrieval gate decides 9/9 …".

## LaTeX version (`paper_ieee_latex/`, `paper_ieee_latex.zip`)

`main.tex` was first generated by the same build script from the same
resolved content as the Word paper, so its text, tables, figures, numbers and
references were identical by construction. **It has since been replaced by
the Overleaf version fitted to 6 pages**, which differs from the Word paper
as listed in "Final LaTeX (6 pages)". The LaTeX conventions:

- `\documentclass[conference,a4paper]{IEEEtran}`;
- cross-references are `\ref`;
- citations are `\cite` (with the `cite` package, so "[1]–[3]" compresses as
  in Word);
- the 25 references are `\bibitem` entries in `main.tex`, so no BibTeX run is
  needed;
- the figures are the same PNG files.

**Two typographic differences** are IEEEtran's own:
- table cells are set ragged-right;
- the six-author block uses a small `\linebreakand` macro, defined in the
  preamble, to force two rows of three.

**What the build checks:**
- braces and environments are balanced;
- every `\ref` has a label;
- exactly the 25 bibitems are cited;
- `%` and `#` are escaped;
- no character is left without a LaTeX mapping;
- every number obeys the same v4 rule as the Word paper.

It was compiled on Overleaf with pdfLaTeX, not on this machine, which has no
LaTeX engine.

## Other files

- `Response_to_Reviewers.docx`, about 1 page, is built by
  `scripts/build_iconaecc_response.py`.
  - Its point lines **summarise** the reviews. Only Reviewer 2's phrase "a
    structural limitation that warrants deeper discussion" is quoted
    verbatim. Paste in the reviewers' own wording if wanted.
  - The two points the brief attributes to "both reviewers" (gates in
    changing environments, computational trade-offs) are listed under R1 and
    R2. **Check that attribution against the reviews.**
- `SUPPLEMENT.docx` and `SUPPLEMENT.md` were rebuilt, and their content is
  unchanged.

## Final LaTeX (6 pages)

**The final source is `paper_ieee_latex/main.tex`.** It is the Overleaf
version, compiled with pdfLaTeX (pdfTeX 1.40.29) to **6 pages**. The count was
read from the PDF's page tree, not from page images. The compiled PDF is
`paper_ieee_latex/final/Calibrated_Escalation_for_IT_Ticket_Triage__What_the_Calibration_Distribution_Decides.pdf`.
`paper_ieee_latex.zip` holds the folder: `main.tex`, the three figures and
`final/`.

**The Word version is not the final source.**
`Calibrated_Escalation_IConAECC2026.docx` keeps the text from before the
6-page fit, without the four trims below, and its page count was never
measured after the last fix. It is kept as a record.

### What the 6-page fit changed

The Overleaf `main.tex` differs from the generated one only in these ways:

1. **Float and caption spacing.** Five lines were added before
   `\begin{document}`:
   - `\textfloatsep`, `\floatsep` and `\intextsep` are 6pt plus 2pt minus 2pt;
   - `\abovecaptionskip` is 3pt;
   - `\belowcaptionskip` is 0pt.
2. **Figure widths:**
   - Fig. 1 is 3.15 in (was 3.3);
   - Fig. 2 is 2.7 in (was 3.05);
   - Fig. 3 is 2.2 in (was 2.55).

   The figure files are unchanged, byte for byte.
3. **Table font.** Four tables changed from `\footnotesize` to
   `\scriptsize`:
   - Table III, the gates at their operating points;
   - Table IV, the coverage gap;
   - Table V, cascade against Tier 2;
   - Table VI, the independent and real-world sets.

   All seven tables are now `\scriptsize`.
4. **Four wording trims**, each repeating something stated elsewhere:
   - **VII, Misclassification.** The list of the most frequent confusions
     (Database→Application, Infrastructure→Application and
     Security→Storage, 2 each) and "10 of the 13 errors predict Application
     or Storage" was replaced by a pointer to Table VII. Table VII gives
     both, unchanged.
   - **VIII, Limitations.** The clause "as no public, real, English IT
     ticket dataset with matching categories was available" was dropped.
     §VI-E still gives that reason in full: "We found no public, real,
     English IT ticket dataset whose categories match our seven teams …".
     The sentence that five teams have no real-ticket evaluation is kept.
   - **IX, Conclusion.** The scoping sentence is shorter. It now reads:
     "They rest on a synthetic corpus, small LLM-generated evaluation sets,
     108 hand-written tickets and 583 real tickets covering two of seven
     teams, and are not performance claims for a production help desk."
     Every item in the scope is kept.
   - **VI-C, zero-shot.** The clause "which does not mean better" was
     dropped after "indistinguishable from Tier 2 (p = 1.000)". The
     indistinguishable result and its p-value are kept.

### What was checked

The checks compare the Overleaf `main.tex` with the previous version
(`7c3d1ac`), and the build now runs them on every run.

- **The same rules as before** apply to the final file:
  - braces and environments are balanced;
  - every `\ref` has a label;
  - all 25 `\bibitem`s are cited;
  - `%` and `#` are escaped;
  - every number occurs in v4;
  - the portal title, the authors, the affiliation and the e-mails are
    present.
- **Numbers, counted by occurrence:**
  - only three occurrences are gone: one each of 2, 10 and 13, all from
    the misclassification trim, and all still in Table VII;
  - no number is new.
- **Citations:** every `\cite` occurrence is identical.
- **Labels and references:** every label is kept. Every `\ref` is kept, plus
  one new `\ref{tab:err}`.
- **The reference list** is identical.
- **Every reviewer-requested passage is present** (points 1–10 above), along
  with the dataset sentence and the Acknowledgment, verbatim.
- **The check fails when it should.** Each of these deliberate breakages was
  caught:
  - dropping "371 of 468";
  - dropping the shadow-deployment sentence;
  - dropping one `\cite{b9}`;
  - dropping the dataset reason in §VI-E.

### Build change

`scripts/build_iconaecc_docx.py` **no longer writes
`paper_ieee_latex/main.tex`**. A rebuild would otherwise silently replace
the 6-page source with the longer generated one. Instead, the build:

- renders the LaTeX in memory;
- checks the committed final file against it, as above;
- fails if a rebuilt figure differs from the committed PNG the PDF was
  compiled with;
- rebuilds the zip.

**Changing the paper's text now means editing `main.tex`.** That includes
running the build afterwards to check it. If the content in the build
script changes, the build reports what the final file lacks.
