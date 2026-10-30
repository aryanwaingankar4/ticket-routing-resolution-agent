"""Derive paper_ieee_docx/ieee_conference_template_a4.docx (IConAECC 2026).

The official IEEE A4 conference template could not be downloaded from this
machine (ieee.org refuses scripted downloads), so the A4 template is derived
from the official US-letter one, which the authors downloaded from the same
IEEE page. The two IEEE Word templates differ only in page geometry. This
script changes ONLY that geometry, in every section, to the IEEE A4
specification:

    paper A4 (210 x 297 mm); top 19 mm, bottom 43 mm, left/right 14.32 mm;
    body columns 88.9 mm wide with 4.22 mm spacing.

Styles, numbering, column counts and sample text are left exactly as IEEE
ships them. The input must be the letter template re-saved by Word as a
Transitional .docx (IEEE ships it in Strict OOXML, which python-docx cannot
read); see paper_ieee_docx/README.md. If the official A4 template is ever
downloaded by hand, re-save it the same way and use it instead of this file.

    .\\venv\\Scripts\\python.exe scripts/derive_ieee_a4_template.py SRC [--force]
"""

from __future__ import annotations

import argparse
import os

from realworld_common import PROJECT_ROOT, die

OUT = os.path.join(PROJECT_ROOT, "paper_ieee_docx",
                   "ieee_conference_template_a4.docx")
MM = 1440 / 25.4                       # twips per millimetre
A4_W, A4_H = 11906, 16838               # Word's own A4 page size in twips
TOP, BOTTOM, SIDE, COL_SPACE = (round(19 * MM), round(43 * MM),
                                round(14.32 * MM), round(4.22 * MM))


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("src", help="letter template re-saved as Transitional")
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args()
    if os.path.exists(OUT) and not args.force:
        die(f"{OUT} exists (use --force)")

    import docx
    from docx.oxml.ns import qn

    try:
        d = docx.Document(args.src)
    except KeyError:
        die("the source is Strict OOXML; re-save it in Word as a .docx "
            "(Word Document) first")
    sects = list(d.element.body.iter(qn("w:sectPr")))
    for s in sects:
        sz = s.find(qn("w:pgSz"))
        if (sz.get(qn("w:w")), sz.get(qn("w:h"))) not in (("12240", "15840"),
                                                         (str(A4_W), str(A4_H))):
            die(f"unexpected page size {sz.get(qn('w:w'))}x{sz.get(qn('w:h'))}")
        sz.set(qn("w:w"), str(A4_W))
        sz.set(qn("w:h"), str(A4_H))
        sz.set(qn("w:code"), "9")       # A4
        mar = s.find(qn("w:pgMar"))
        mar.set(qn("w:top"), str(TOP))
        mar.set(qn("w:bottom"), str(BOTTOM))
        mar.set(qn("w:left"), str(SIDE))
        mar.set(qn("w:right"), str(SIDE))
        cols = s.find(qn("w:cols"))
        if cols is not None and cols.get(qn("w:num")) == "2":
            cols.set(qn("w:space"), str(COL_SPACE))
    d.save(OUT)
    print(f"[write] {os.path.relpath(OUT, PROJECT_ROOT)}: {len(sects)} sections "
          f"set to A4, margins {TOP}/{BOTTOM}/{SIDE} twips, column space "
          f"{COL_SPACE}")


if __name__ == "__main__":
    main()
