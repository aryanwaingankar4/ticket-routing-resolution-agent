# paper/ -- the generated write-up surface

**Everything in this directory is generated -- except the four hand-written draft files named below. Do not edit any generated file by hand.** Rebuild with one command, from the project root:

```powershell
.\venv\Scripts\Activate.ps1
python src/experiments/build_paper_artifacts.py --force
```

The build is offline. It makes **no Gemini call, no Ollama call, no model load, no training run and no experiment re-run** -- it reads committed result files under `data/` and writes only here.

## Why this exists

Phase 7A recorded two domain AUCs that were computed in an interactive session, quoted at a gate, and never committed. When Phase 7B specified the recomputation they did not reproduce. The conclusions survived because every value cleared the same threshold, but the figures were wrong in print for a day.

So: **every number the paper uses is regenerated from a committed result file, through code.** `tests/test_paper_artifacts.py` rebuilds this directory and fails if any value, table CSV or figure-data CSV changes -- golden parity, applied to the write-up.

## What is here

- `NUMBERS.md` -- 257 numbers, each with its source file, the command that regenerates that file, and its tags. A number that is not in this file is not a number the paper may use. It also carries the **do-not-cite list** and the **no committed source** list.
- `FRAMING.md` -- the write-up framing agreed at each phase gate, with its numbers interpolated from `NUMBERS.md` so the text cannot drift from the measurements.
- `RECONCILIATION.md` -- the four project documents audited against their source files.
- `PROVENANCE.json` -- the sha256 of every source file read, so a parity failure can distinguish 'the builder changed' from 'a result file changed'.
- `tables/` -- 146 files: each table as `.csv` and as booktabs `.tex`, plus `tables/ieee/`: column selections of those frames sized for an IEEE page, which is what the draft inputs.
- `numbers.tex` -- every `NUMBERS.md` value as a LaTeX macro, `\nb{<id>}`, with generated display variants (`@pct1`, `@r3`, `@k`, `@n`, `@ci`, ...). An unknown id is a compile error.
- `figures/` -- 7 figures, each as `.pdf` and `.png` at 300 dpi, with the data behind it as `_data.csv` and its caption as `_caption.txt`.

## Hand-written (Phase 9A) -- NOT generated

- `main.tex` -- the IEEE conference draft (IEEEtran). Every number in it is a `\nb{}` macro; `tests/test_paper_draft.py` fails on a typed digit, a partially quoted clause group or a retired phrasing.
- `supplement.tex` -- the overflow appendix, same rules.
- `references.bib` -- citation keys; every field not confirmed is `TODO`.
- `references_to_check.md` -- every citation, the claim it supports and what to look up.

No LaTeX engine is installed on the development machine. The sources are Overleaf-ready: upload this directory, set `main.tex` as the main document, and compile with pdfLaTeX + BibTeX.

## Conventions

- Figures are sized for IEEE two-column: 3.5 in single-column, 7.16 in full-width, base font 8 pt, Okabe-Ito colour-blind-safe palette, and distinct dash patterns so the figures also survive greyscale.
- Output is deterministic: fixed seed, stable row order, fixed float formatting, and PDF/PNG creation metadata stripped. A rebuild is byte-identical unless a source file changed.
- Proportions carry a Wilson 95% interval, except on pre-registered case-study axes, which report counts only. Paired comparisons on the same tickets use the exact McNemar test.

## Flags

| flag | effect |
|---|---|
| *(none)* | refuses to overwrite an existing `paper/` |
| `--force` | rebuild in place |
| `--out DIR` | write somewhere else (the parity test uses this) |
| `--no-render` | skip PDF/PNG rasterisation; still writes every table and figure-data CSV |
| `--no-audit` | skip the document reconciliation pass |
