# IConAECC 2026 submission (Word)

The 6-page IEEE conference version of the mentor-approved paper v4
(`paper_docx/Calibrated_Escalation_IT_Ticket_Triage_v4.docx`, branch
`paper-revision-oct`), for **IConAECC 2026** (IEEE technically co-sponsored,
IEEE Madras Section, conference record #70685). Branch `iconaecc-2026`; not
merged into `main`.

| File | What it is |
|---|---|
| `Calibrated_Escalation_IConAECC2026.docx` | **The paper. Open this in Word, check the page count, export the PDF.** |
| `SUPPLEMENT.docx`, `SUPPLEMENT.md` | Online supplement: v4's Appendices A–F unchanged, plus the two figures removed from the paper |
| `ieee_conference_template_letter.docx` | Official IEEE conference Word template (US letter), as IEEE ships it (Strict OOXML) |
| `ieee_conference_template_letter_resaved.docx` | The same file re-saved by Word as a standard (Transitional) .docx, nothing else changed |
| `ieee_conference_template_a4.docx` | The A4 template the paper is built in (derived, see below) |
| `fig/` | Fig. 1 (v4's Fig. 1, cropped) and Fig. 2 (v4's Figs. 4 and 5 as two panels) |

## Template source

- IEEE "Manuscript Templates for Conference Proceedings":
  https://www.ieee.org/conferences/publishing/templates.html
  (the conference site links the older address
  http://www.ieee.org/conferences_events/conferences/publishing/templates.html).
- The A4 file, https://www.ieee.org/content/dam/ieee-org/ieee/web/org/conferences/conference-template-a4.docx,
  **could not be downloaded from this machine**: ieee.org answers scripted
  requests with an empty bot-check page. The official **US-letter** file from
  the same page (`conference-template-letter.docx`) was already on this
  machine and is committed unchanged.
- IEEE ships it in Strict OOXML, which python-docx cannot open, so Word
  re-saved it as a Transitional .docx (`..._letter_resaved.docx`).
- `scripts/derive_ieee_a4_template.py` then changes **only the page
  geometry**, in every section, to the IEEE A4 specification: A4 paper; top
  19 mm, bottom 43 mm, left and right 14.32 mm; 4.22 mm between the two body
  columns. Styles, numbering and column counts are as IEEE ships them.
- **If you download the official A4 template by hand**, re-save it in Word
  as a .docx and save it over `ieee_conference_template_a4.docx`. Then
  re-run the build; nothing else changes.

## Conference rules (read 2026-10-30)

Sources: https://iconaecc2026.aconf.org/ (home, Author_Guidelines.html,
faq.html) and https://www.usssociety.org/conf_224650.html.

- **Initial submission:**
  - six printed pages, 10-point font, English;
  - "Initial submissions longer than SIX (6) pages will be rejected without
    review";
  - uploaded as a **PDF** through the Aconf system.
- **Final:**
  - six pages, with up to two extra pages (7 and 8) at USD 10 each;
  - IEEE-formatted, checked by IEEE PDF eXpress (conference ID 70685X, opens
    20 Nov 2026);
  - submitted as a ZIP of the Word file, separate figure files and the PDF.
- **Templates:** IEEE Word templates, A4 or US letter.
- **Review:** not stated to be blind. The author list and title in the PDF
  must **exactly match** the Aconf registration and, later, the IEEE
  copyright form. So the author block is kept as in v4.
- **Dates:**
  - the guidelines page gives the draft deadline as **2026-10-30**;
  - the home page says 31 Oct 2026;
  - acceptance 20 Nov, final paper 30 Nov, conference 18–20 Dec 2026.
- **No plagiarism percentage** is given; plagiarism and double submission
  are forbidden.
- **AI disclosure:** the conference does not mention it. IEEE's author
  policy requires AI-generated content to be disclosed in the
  acknowledgments, so the paper has an Acknowledgment section.

## Build (offline, no model, no LLM)

```powershell
.\venv\Scripts\python.exe scripts/derive_ieee_a4_template.py paper_ieee_docx/ieee_conference_template_letter_resaved.docx --force
.\venv\Scripts\python.exe scripts/make_iconaecc_figures.py --force
.\venv\Scripts\python.exe scripts/build_iconaecc_docx.py --force
```

`build_iconaecc_docx.py` copies these from v4 verbatim:
- tables;
- references;
- the author block;
- the prompts and reproduction steps.

The Phase 2/3 numbers are formatted from `build_v4_docx.load_values()`, which
re-reads and re-checks the committed result files.

The build fails if:
- **any number in the paper or supplement does not occur in v4's text**;
- template sample text remains;
- a "??" or TODO remains;
- a Table/Fig./Section reference dangles;
- the abstract is over 200 words;
- a reference is never cited.

The template numbers the sections, tables (TABLE I…), figures (Fig. 1.) and
references ([1]…). Citations are renumbered in order of first appearance.

## Page count

Measured in Microsoft Word on this machine. The PDF page count is read from
the exported PDF's metadata:

1. **First build: 8 pages.** The last reference was a third of the way down
   page 8, about 7.3 pages.
2. **Second build: 7 pages.** Page 7 held only the last reference line,
   about 6.0 pages. No table broke across a column or page, and all
   figures were at column width.
3. After that, a further ~300 words were cut and Fig. 2 was made 0.2 in
   shorter. This is **estimated at about 5.7 pages and not measured**: the
   brief allowed two measurements. **Check it in Word**.
   - If it shows 7 pages, the next cut is the Table VII leakage cell. Shorten
     its quoted example and keep the counts, then rebuild.

## What was condensed (for the mentor)

Nothing below removes a result, a number or a table that v4 has in its main
text, unless the item says it moved to the supplement.

1. **Appendices A–F moved to the online supplement** (S-A prompts, S-B
   reproduction steps, S-C mapping sensitivity, S-D configurations, S-E
   verification checks, S-F lessons). The paper points to it with one
   sentence and its URL. Every former "Appendix X" citation now reads
   "online supplement" or "supplement".
2. **v4 Fig. 2 (coverage gap) removed** from the paper, because Table IV
   (v4 Table VI) has the same numbers. It is Fig. S1 in the supplement.
3. **v4 Figs. 4 and 5 merged** into one two-panel Fig. 2: (a) cascade 0.50,
   (b) retrieval 0.67. Both are drawn by the same plotting functions from the
   same files, with panel height 2.35 in → 1.65 in. The threshold evidence
   (Table III, v4 Table V) is kept.
4. **v4 Fig. 6 (confusion matrix) moved** to the supplement (Fig. S2). The
   key confusions stay in the text: Database→Application,
   Infrastructure→Application and Security→Storage, 2 each, and 10 of 13 are
   Application or Storage. The three example tickets (N31, N39, N13) moved
   beside the matrix.
5. **Fig. 1** was cropped to its drawn content (white margins only) and moved
   from a full-width section to column width.
6. **Table I**: shorter cells, 7 pt, and the header "Escalation / abstention mechanism" became
   "Abstention mechanism". The citation [24] was dropped from the "Ours" row;
   the dataset is cited in Section IV. Tables II and VII (v4 III and IX) are
   set in 7 pt with v4's cell text unchanged.
7. **Abstract**: 240 → 189 words. These clauses were dropped:
   - "(12.1% over all 984 tickets)";
   - "indistinguishable from Tier 2 alone (p = 1.0)";
   - "Tier 1 answers 238 of them with 39 correct";
   - "so it serves as a baseline";
   - "We report negative results, an error analysis and reproduction steps".

   All of these stay in Section VI-E or VII.
8. **Introduction**: the first paragraph was shortened, and the two
   paragraphs were merged. The four contributions are shortened; contribution
   2 no longer lists "a coverage guarantee, a drift reference and a trained
   classifier", which Section VI-E's last paragraph does.
9. **Related Work**: the six run-in paragraphs became one. All 17 citations
   are kept. "Jung et al."/"Jain et al." became plain citations, and the
   positioning paragraph no longer repeats the 108/583 counts.
10. **Methodology**: v4's §III-A–F became two subsections. These were folded
    into "Pipeline and gates":
    - overview;
    - cascade;
    - retrieval gate;
    - gate independence.

    Conformal and verification were merged into the second subsection.

    These were dropped:
    - "the measured latency contains no network hops";
    - "Each stage ... declares the artifacts it needs";
    - the sentence explaining why internally consistent wrong values matter;
    - the details of the drafting prompt's fields, which are in the
      supplement.
11. **Experimental Setup**: v4's §IV-A–G became three run-in paragraphs:
    dataset, evaluation sets, statistics and reproducibility.
    - "Additional training rows add little accuracy" is now covered by
      Section VI-C.
    - The configuration table and the prompts went to the supplement.
    - The repository URL is given once, inside the supplement URL.
12. **Thresholds (§V)**: v4's wording was tightened, and every number in it
    is kept. "This threshold decides when/whether..." openers were dropped,
    and the two cautions are now one sentence.
13. **Results**:
    - **§VI-A**: the three bullets were tightened and keep all their
      numbers. The recalibration paragraph is now one sentence.
    - **§VI-B**: one sentence was dropped: "where more tickets pass the
      Tier-1 threshold and never reach the encoder".
    - **§VI-C**: the bullets became one paragraph. "A 3B model with no
      training ... is not beaten by a classifier fitted on 3,200 of its
      rows" was dropped.
    - **Lexical tier**: its 16/45 is now quoted only in §VI-B; Fig. 3 still
      shows it.
    - **§VI-D**: rephrased, with the same numbers.
    - **§VI-E**: the description was tightened. The dataset sentence and
      every number are kept verbatim.
    - **§VI-F**: became a run-in paragraph at the end of §VI-E.
14. **Error Analysis**:
    - the three subsections became run-in paragraphs;
    - the N45 sentence was dropped, because it is in Table VII;
    - "13.3% clear it" was dropped, because it is in Table III and Table VII;
    - the 26/31 sufficiency count is now in §VI-D only.
15. **Limitations**: both paragraphs are kept with every point, more
    tightly worded.
16. **Conclusion**: two paragraphs became one. The scoping sentence and the
    real-world negative result (19.7%, 50.4%, 371 of 468) are kept. Dropped
    from the conclusion only:
    - "13.3% leakage";
    - "31/33 grounded";
    - "74/98";
    - "32 of 98 actionable".

    The future-work list is shorter.
17. **Added**: the Acknowledgment (AI-assistance disclosure) and the
    supplement sentence.
