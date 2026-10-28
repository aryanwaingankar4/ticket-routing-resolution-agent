"""Build the B.Tech project report (black book) as a Word file.

Since 28 Oct 2026 the build starts from the author's edited copy,
reports/black_book/Black_Book_EDITED_BASE.docx. Everything before its
"Chapter 1" heading (cover, certificate, declaration, acknowledgement,
abstract, contents and lists as Word fields, abbreviations) is kept exactly as
the author left it, except that any PRN differing from the cover page is
aligned to the cover and reported. Chapter 1 onwards, the references and the
appendices are regenerated from the plain-text sources in
reports/black_book/content/ and the result is saved as Black_Book_v2.docx.
`--from-template` restores the original path, which filled the department's
format file from scratch.

    .\\venv\\Scripts\\python.exe scripts/build_black_book.py            # build + update fields in Word
    .\\venv\\Scripts\\python.exe scripts/build_black_book.py --no-word  # python-docx only
    .\\venv\\Scripts\\python.exe scripts/build_black_book.py --pdf      # also export a PDF via Word

Inputs (never written):
    reports/template/B.Tech_Project_Report_Hackathon_Format.docx
    reports/black_book/content/*.txt
    reports/black_book/fig/*.png, paper_docx/fig/*.png, paper/figures/*.png
Output:
    reports/black_book/Black_Book_Calibrated_Escalation_IT_Ticket_Triage.docx

The content files use a small markup, documented in content/README.md. Figure,
table and section numbers are assigned here, by chapter, and cross-references
({fig:x}, {tab:x}, {sec:x}) and citations ([@key]) are resolved in a first
pass, so a reference to a later chapter works. The build is FATAL on an
unknown label, an unknown citation key, a reference that is never cited, a
figure or table that is never referred to in the text, or a missing image.

Captions carry SEQ fields and the contents and lists are TOC fields, so Word
fills the page numbers (Ctrl+A, F9). The document is also set to update its
fields when it is opened. On Windows with Word installed, the build opens the
file in Word once, updates every field and saves it, so it arrives complete.
"""

from __future__ import annotations

import argparse
import copy
import os
import re
import subprocess
import sys
import tempfile
import zipfile

from docx import Document
from docx.enum.table import WD_TABLE_ALIGNMENT
from docx.enum.text import WD_ALIGN_PARAGRAPH, WD_COLOR_INDEX
from docx.oxml import OxmlElement, parse_xml
from docx.oxml.ns import nsdecls, qn
from docx.shared import Inches, Pt, RGBColor

PROJECT_ROOT = os.path.abspath(
    os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
BB = os.path.join(PROJECT_ROOT, "reports", "black_book")
CONTENT = os.path.join(BB, "content")
TEMPLATE = os.path.join(PROJECT_ROOT, "reports", "template",
                        "B.Tech_Project_Report_Hackathon_Format.docx")
# Since 28 Oct 2026 the build starts from the author's edited copy, whose front
# matter (names, PRNs, date) was corrected by hand and must survive. The
# template path is kept only for --from-template.
BASE = os.path.join(BB, "Black_Book_EDITED_BASE.docx")
OUTPUT = os.path.join(BB, "Black_Book_v2.docx")

TITLE = ("Calibrated Escalation for IT Ticket Triage: "
         "A Two-Gate Retrieval-Augmented Pipeline")
STUDENTS = ["Aryan Manoj Waingankar", "Shreyansh Pandey", "Parth Bagle",
            "Swapnil Samrat"]
GUIDE = ("Dr. Madhvi Saxena", "Assistant Professor, Department of AI & ML")
CO_GUIDE = ("Dr. Aditi Saxena", "[Designation]")

TEXT_WIDTH_IN = 5.9          # A4 minus 1.3" left and 1.0" right margins
MAX_FIG_HEIGHT_IN = 7.6
CHAPTER_FILES = ["01_introduction.txt", "02_literature.txt", "03_srs.txt",
                 "04_methodology.txt", "05_results.txt", "06_conclusion.txt"]
APPENDIX_FILE = "07_appendices.txt"


def die(msg):
    print(f"\n[FATAL] {msg}")
    raise SystemExit(1)


# =========================================================================== #
# 1. Parsing the content files into blocks                                     #
# =========================================================================== #
LABEL_RE = re.compile(r"\s*\{#([a-z0-9_:-]+)\}\s*$")


def parse_file(path):
    """Return a list of (kind, payload) blocks. See content/README.md."""
    with open(path, encoding="utf-8") as fh:
        lines = fh.read().splitlines()
    blocks, para, i = [], [], 0

    def flush():
        if para:
            blocks.append(("p", " ".join(s.strip() for s in para)))
            para.clear()

    while i < len(lines):
        line = lines[i]
        s = line.strip()
        if s.startswith("%%"):                       # comment line
            i += 1
            continue
        if not s:
            flush()
            i += 1
            continue
        if s.startswith("```"):
            flush()
            code = []
            i += 1
            while i < len(lines) and not lines[i].strip().startswith("```"):
                code.append(lines[i])
                i += 1
            blocks.append(("code", code))
            i += 1
            continue
        m = re.match(r"^(#{1,4}) (.+)$", s)
        if m:
            flush()
            level, text = len(m.group(1)), m.group(2)
            lm = LABEL_RE.search(text)
            label = lm.group(1) if lm else None
            text = LABEL_RE.sub("", text)
            blocks.append((f"h{level}", (text, label)))
            i += 1
            continue
        if s.startswith("[[") and s.endswith("]]") and not s.startswith("[[PH "):
            flush()
            inner = s[2:-2]
            kind, _, rest = inner.partition(" ")
            parts = [p.strip() for p in rest.split("|")]
            if kind == "TAB":
                rows = []
                i += 1
                while i < len(lines) and lines[i].strip().startswith("|"):
                    cells = [c.strip() for c in lines[i].strip().strip("|").split("|")]
                    rows.append(cells)
                    i += 1
                blocks.append(("tab", (parts, rows)))
                continue
            blocks.append((kind.lower(), parts))
            i += 1
            continue
        if re.match(r"^- ", s) or re.match(r"^\d+\. ", s):
            flush()
            numbered = bool(re.match(r"^\d+\. ", s))
            items = []
            while i < len(lines):
                t = lines[i]
                ts = t.strip()
                if re.match(r"^- ", ts) or re.match(r"^\d+\. ", ts):
                    items.append(re.sub(r"^(- |\d+\. )", "", ts))
                elif ts and t.startswith("  ") and items:
                    items[-1] += " " + ts
                else:
                    break
                i += 1
            blocks.append(("ol" if numbered else "ul", items))
            continue
        para.append(line)
        i += 1
    flush()
    return blocks


# =========================================================================== #
# 2. Numbering pass                                                            #
# =========================================================================== #
class Numbers:
    def __init__(self):
        self.fig, self.tab, self.sec = {}, {}, {}
        self.cites = {}
        self.cited_order = []
        self.fig_refs, self.tab_refs = set(), set()


CITE_RE = re.compile(r"\[@([^\]]+)\]")
XREF_RE = re.compile(r"\{(fig|tab|sec|secn):([a-z0-9_:-]+)\}")


def iter_text(block):
    kind, payload = block
    if kind == "p":
        yield payload
    elif kind in ("ul", "ol"):
        yield from payload
    elif kind == "tab":
        parts, rows = payload
        yield parts[1] if len(parts) > 1 else ""
        for r in rows:
            yield from r
    elif kind == "fig":
        yield payload[3] if len(payload) > 3 else ""
    elif kind in ("h1", "h2", "h3", "h4"):
        yield payload[0]


LENIENT = False


def number_everything(doc_blocks, refs):
    """doc_blocks: list of (chapter_key, blocks). chapter_key is '1'..'6' or 'A'.."""
    n = Numbers()
    for key, blocks in doc_blocks:
        f = t = s2 = s3 = 0
        for kind, payload in blocks:
            if kind == "h2":
                s2 += 1
                s3 = 0
                if payload[1]:
                    n.sec[payload[1]] = f"Section {key}.{s2}"
            elif kind == "h3":
                s3 += 1
                if payload[1]:
                    n.sec[payload[1]] = f"Section {key}.{s2}.{s3}"
            elif kind == "fig":
                f += 1
                n.fig[payload[0]] = f"{key}.{f}"
            elif kind == "tab":
                t += 1
                n.tab[payload[0][0]] = f"{key}.{t}"
            elif kind == "h1" and payload[1]:
                n.sec[payload[1]] = f"Chapter {key}"
    # citation order = order of first appearance in the document
    for _, blocks in doc_blocks:
        for b in blocks:
            for text in iter_text(b):
                for m in CITE_RE.finditer(text):
                    for k in [x.strip().lstrip("@") for x in m.group(1).split(";")]:
                        if k not in refs:
                            die(f"unknown citation key [@{k}]")
                        if k not in n.cites:
                            n.cites[k] = len(n.cites) + 1
                            n.cited_order.append(k)
                for m in XREF_RE.finditer(text):
                    kind, lab = m.groups()
                    table = {"fig": n.fig, "tab": n.tab, "sec": n.sec, "secn": n.sec}[kind]
                    if lab not in table:
                        if LENIENT:
                            print(f"  [lenient] unresolved {{{kind}:{lab}}}")
                            continue
                        die(f"unknown cross-reference {{{kind}:{lab}}}")
                    if kind == "fig":
                        n.fig_refs.add(lab)
                    if kind == "tab":
                        n.tab_refs.add(lab)
    uncited = sorted(set(refs) - set(n.cites))
    if uncited and not LENIENT:
        die(f"references never cited: {uncited}")
    for lab in n.fig:
        if lab not in n.fig_refs:
            die(f"Figure {n.fig[lab]} ({lab}) is never referred to in the text")
    for lab in n.tab:
        if lab not in n.tab_refs:
            die(f"Table {n.tab[lab]} ({lab}) is never referred to in the text")
    return n


def resolve(text, n):
    def cite(m):
        keys = [x.strip().lstrip("@") for x in m.group(1).split(";")]
        nums = sorted(n.cites[k] for k in keys)
        # IEEE style: [3], [5]; a run of three or more becomes [3]-[5]
        out, i = [], 0
        while i < len(nums):
            j = i
            while j + 1 < len(nums) and nums[j + 1] == nums[j] + 1:
                j += 1
            if j - i >= 2:
                out.append(f"[{nums[i]}]–[{nums[j]}]")
            else:
                out.extend(f"[{x}]" for x in nums[i:j + 1])
            i = j + 1
        return ", ".join(out)

    def xref(m):
        kind, lab = m.groups()
        if lab not in {"fig": n.fig, "tab": n.tab, "sec": n.sec, "secn": n.sec}[kind]:
            return {"fig": "Figure ?", "tab": "Table ?", "sec": "Section ?", "secn": "?"}[kind]
        if kind == "fig":
            return f"Figure {n.fig[lab]}"
        if kind == "tab":
            return f"Table {n.tab[lab]}"
        if kind == "secn":
            return n.sec[lab].split(" ", 1)[1]
        return n.sec[lab]

    text = CITE_RE.sub(cite, text)
    return XREF_RE.sub(xref, text)


# =========================================================================== #
# 3. Low-level docx helpers                                                    #
# =========================================================================== #
INLINE_RE = re.compile(
    r"(\[\[PH .+?\]\]|\*\*.+?\*\*|`[^`]+`|(?<![\w*])\*(?!\s)[^*]+?(?<!\s)\*(?![\w*]))")


def add_runs(par, text, size=None, bold=None, italic=None, font=None):
    """Write text with **bold**, *italic*, `code` and [[PH ...]] markup."""
    for piece in INLINE_RE.split(text):
        if not piece:
            continue
        b, it, mono, ph = bold, italic, False, False
        if piece.startswith("[[PH ") and piece.endswith("]]"):
            piece, ph = "[" + piece[5:-2] + "]", True
        elif piece.startswith("**") and piece.endswith("**") and len(piece) > 4:
            piece, b = piece[2:-2], True
        elif piece.startswith("`") and piece.endswith("`") and len(piece) > 2:
            piece, mono = piece[1:-1], True
        elif piece.startswith("*") and piece.endswith("*") and len(piece) > 2:
            piece, it = piece[1:-1], True
        r = par.add_run(piece)
        if b is not None:
            r.bold = b
        if it is not None:
            r.italic = it
        if size:
            r.font.size = Pt(size)
        if mono:
            r.font.name = "Consolas"
            r._element.rPr.rFonts.set(qn("w:hAnsi"), "Consolas")
            r.font.size = Pt((size or 12) - 1.5)
        elif font:
            r.font.name = font
        if ph:
            r.font.highlight_color = WD_COLOR_INDEX.YELLOW
    return par


def field_runs(par, instr, result_text, bold=False):
    """Append a complex field {instr} with a precomputed result."""
    def run_with(child):
        r = OxmlElement("w:r")
        if bold:
            rpr = OxmlElement("w:rPr")
            rpr.append(OxmlElement("w:b"))
            r.append(rpr)
        r.append(child)
        par._p.append(r)
        return r
    b = OxmlElement("w:fldChar")
    b.set(qn("w:fldCharType"), "begin")
    run_with(b)
    it = OxmlElement("w:instrText")
    it.set(qn("xml:space"), "preserve")
    it.text = f" {instr} "
    run_with(it)
    sep = OxmlElement("w:fldChar")
    sep.set(qn("w:fldCharType"), "separate")
    run_with(sep)
    t = OxmlElement("w:t")
    t.set(qn("xml:space"), "preserve")
    t.text = result_text
    run_with(t)
    e = OxmlElement("w:fldChar")
    e.set(qn("w:fldCharType"), "end")
    run_with(e)


def set_cell_shading(cell, hex_fill):
    tcpr = cell._tc.get_or_add_tcPr()
    shd = OxmlElement("w:shd")
    shd.set(qn("w:val"), "clear")
    shd.set(qn("w:color"), "auto")
    shd.set(qn("w:fill"), hex_fill)
    tcpr.append(shd)


def keep_with_next(par, on=True):
    par.paragraph_format.keep_with_next = on


def set_text_keep_format(par, text):
    """Replace a template paragraph's text but keep its first run's formatting."""
    runs = par.runs
    if not runs:
        par.add_run(text)
        return
    runs[0].text = text
    for r in runs[1:]:
        r._element.getparent().remove(r._element)


# =========================================================================== #
# 4. Styles                                                                    #
# =========================================================================== #
def ensure_style(doc, name, base="Normal", kind=1):
    styles = doc.styles
    try:
        return styles[name]
    except KeyError:
        st = styles.add_style(name, kind)
        st.base_style = styles[base]
        return st


def spacing(pf, before=0, after=0, line=1.5):
    pf.space_before = Pt(before)
    pf.space_after = Pt(after)
    pf.line_spacing = line


def setup_styles(doc):
    tnr = "Times New Roman"

    def font(st, size, bold=False, name=tnr):
        st.font.name = name
        st.font.size = Pt(size)
        st.font.bold = bold
        st.font.small_caps = False
        st.font.color.rgb = RGBColor(0, 0, 0)
        rpr = st.element.get_or_add_rPr()
        rf = rpr.find(qn("w:rFonts"))
        if rf is None:
            rf = OxmlElement("w:rFonts")
            rpr.insert(0, rf)
        for a in ("w:ascii", "w:hAnsi", "w:cs", "w:eastAsia"):
            rf.set(qn(a), name)

    body = ensure_style(doc, "BB Body")
    font(body, 12)
    body.paragraph_format.alignment = WD_ALIGN_PARAGRAPH.JUSTIFY
    spacing(body.paragraph_format, 0, 8, 1.5)

    h1 = doc.styles["Heading 1"]
    font(h1, 18, True)
    h1.paragraph_format.alignment = WD_ALIGN_PARAGRAPH.CENTER
    h1.paragraph_format.page_break_before = True
    h1.paragraph_format.keep_with_next = True
    spacing(h1.paragraph_format, 0, 24, 1.5)

    h2 = doc.styles["Heading 2"]
    font(h2, 14, True)
    h2.paragraph_format.alignment = WD_ALIGN_PARAGRAPH.LEFT
    h2.paragraph_format.keep_with_next = True
    spacing(h2.paragraph_format, 14, 6, 1.5)

    h3 = doc.styles["Heading 3"]
    font(h3, 12, True)
    h3.paragraph_format.alignment = WD_ALIGN_PARAGRAPH.LEFT
    h3.paragraph_format.left_indent = Inches(0)
    h3.paragraph_format.first_line_indent = Inches(0)
    h3.paragraph_format.keep_with_next = True
    spacing(h3.paragraph_format, 10, 4, 1.5)

    h4 = ensure_style(doc, "BB Runin", "BB Body")
    font(h4, 12, True)
    h4.paragraph_format.keep_with_next = True
    spacing(h4.paragraph_format, 6, 2, 1.5)

    front = ensure_style(doc, "BB Front Heading")
    font(front, 14, True)
    front.font.small_caps = True
    front.paragraph_format.alignment = WD_ALIGN_PARAGRAPH.CENTER
    spacing(front.paragraph_format, 0, 18, 1.5)
    ppr = front.element.get_or_add_pPr()
    ol = OxmlElement("w:outlineLvl")
    ol.set(qn("w:val"), "0")
    ppr.append(ol)

    bullet = ensure_style(doc, "BB List", "BB Body")
    spacing(bullet.paragraph_format, 0, 4, 1.5)

    code = ensure_style(doc, "BB Code")
    font(code, 9, name="Consolas")
    code.paragraph_format.alignment = WD_ALIGN_PARAGRAPH.LEFT
    code.paragraph_format.left_indent = Inches(0.15)
    spacing(code.paragraph_format, 0, 0, 1.0)

    cell = ensure_style(doc, "BB Table Text")
    font(cell, 10)
    cell.paragraph_format.alignment = WD_ALIGN_PARAGRAPH.LEFT
    spacing(cell.paragraph_format, 1, 1, 1.0)

    fig = ensure_style(doc, "BB Figure")
    fig.paragraph_format.alignment = WD_ALIGN_PARAGRAPH.CENTER
    fig.paragraph_format.keep_with_next = True
    spacing(fig.paragraph_format, 6, 2, 1.0)

    eq = ensure_style(doc, "BB Equation", "BB Body")
    eq.paragraph_format.alignment = WD_ALIGN_PARAGRAPH.CENTER
    spacing(eq.paragraph_format, 4, 8, 1.15)

    cap = doc.styles["Caption"]
    font(cap, 11, True)
    cap.paragraph_format.alignment = WD_ALIGN_PARAGRAPH.CENTER
    spacing(cap.paragraph_format, 4, 12, 1.0)

    by_id = {st.style_id: st for st in doc.styles}
    for name, indent, bold in (("TOC1", 0, True), ("TOC2", 0.25, False),
                               ("TOC3", 0.5, False), ("TableofFigures", 0, False)):
        st = by_id[name]
        font(st, 12 if name == "TOC1" else 11.5, bold)
        st.paragraph_format.left_indent = Inches(indent)
        spacing(st.paragraph_format, 2 if name == "TOC1" else 0, 2, 1.15)
        ppr = st.element.get_or_add_pPr()
        for tag in ("w:spacing",):
            el = ppr.find(qn(tag))
            if el is not None:
                for a in ("w:beforeAutospacing", "w:afterAutospacing"):
                    if el.get(qn(a)) is not None:
                        del el.attrib[qn(a)]
    return {"body": body, "front": front}


# =========================================================================== #
# 5. Numbering definitions (bullets and numbered lists)                        #
# =========================================================================== #
class Lists:
    BULLET_ABS = 9101
    DECIMAL_ABS = 9102

    def __init__(self, doc):
        self.numbering = doc.part.numbering_part.element
        existing_abs = {int(a.get(qn("w:abstractNumId")))
                        for a in self.numbering.findall(qn("w:abstractNum"))}
        nums = [int(n.get(qn("w:numId"))) for n in self.numbering.findall(qn("w:num"))]
        self.next_num = max(nums + [9199]) + 1
        first_num = self.numbering.find(qn("w:num"))
        for abs_id, fmt, text in ((self.BULLET_ABS, "bullet", "•"),
                                  (self.DECIMAL_ABS, "decimal", "%1.")):
            if abs_id in existing_abs:      # the edited base already defines it
                continue
            xml = (f'<w:abstractNum {nsdecls("w")} w:abstractNumId="{abs_id}">'
                   f'<w:multiLevelType w:val="singleLevel"/>'
                   f'<w:lvl w:ilvl="0"><w:start w:val="1"/><w:numFmt w:val="{fmt}"/>'
                   f'<w:lvlText w:val="{text}"/><w:lvlJc w:val="left"/>'
                   f'<w:pPr><w:ind w:left="567" w:hanging="340"/></w:pPr>'
                   + ('<w:rPr><w:rFonts w:ascii="Times New Roman" w:hAnsi="Times New Roman"/></w:rPr>'
                      if fmt == "bullet" else "") +
                   '</w:lvl></w:abstractNum>')
            el = parse_xml(xml)
            if first_num is not None:
                first_num.addprevious(el)
            else:
                self.numbering.append(el)
        self.bullet_num = self._new_num(self.BULLET_ABS)

    def _new_num(self, abs_id, restart=False):
        nid = self.next_num
        self.next_num += 1
        over = ('<w:lvlOverride w:ilvl="0"><w:startOverride w:val="1"/></w:lvlOverride>'
                if restart else "")
        self.numbering.append(parse_xml(
            f'<w:num {nsdecls("w")} w:numId="{nid}"><w:abstractNumId w:val="{abs_id}"/>'
            f'{over}</w:num>'))
        return nid

    def new_numbered(self):
        return self._new_num(self.DECIMAL_ABS, restart=True)

    @staticmethod
    def apply(par, num_id):
        ppr = par._p.get_or_add_pPr()
        numpr = OxmlElement("w:numPr")
        il = OxmlElement("w:ilvl")
        il.set(qn("w:val"), "0")
        ni = OxmlElement("w:numId")
        ni.set(qn("w:val"), str(num_id))
        numpr.append(il)
        numpr.append(ni)
        ppr.append(numpr)


# =========================================================================== #
# 6. Rendering blocks                                                          #
# =========================================================================== #
class Renderer:
    def __init__(self, doc, numbers, lists):
        self.doc, self.n, self.lists = doc, numbers, lists
        self.missing_images = []

    # -- headings ---------------------------------------------------------- #
    def chapter(self, key, title):
        p = self.doc.add_paragraph(style="Heading 1")
        p.add_run(f"Chapter {key}")
        p.add_run().add_break()
        p.add_run(title)

    def appendix(self, letter, title):
        p = self.doc.add_paragraph(style="Heading 1")
        p.add_run(f"Appendix {letter}")
        p.add_run().add_break()
        p.add_run(title)

    def unnumbered_h1(self, title):
        p = self.doc.add_paragraph(style="Heading 1")
        p.add_run(title)

    def heading(self, level, text):
        p = self.doc.add_paragraph(style=f"Heading {level}")
        add_runs(p, resolve(text, self.n))

    # -- body -------------------------------------------------------------- #
    def para(self, text, style="BB Body"):
        p = self.doc.add_paragraph(style=style)
        add_runs(p, resolve(text, self.n))
        return p

    def runin(self, text):
        p = self.doc.add_paragraph(style="BB Runin")
        add_runs(p, resolve(text, self.n))

    def bullets(self, items, numbered=False):
        num = self.lists.new_numbered() if numbered else self.lists.bullet_num
        for it in items:
            p = self.doc.add_paragraph(style="BB List")
            add_runs(p, resolve(it, self.n))
            Lists.apply(p, num)

    def code(self, lines):
        if len(lines) > 16:
            die(f"code excerpt longer than 16 lines: {lines[0]!r}")
        for k, line in enumerate(lines):
            p = self.doc.add_paragraph(style="BB Code")
            r = p.add_run(line if line else " ")
            r.font.name = "Consolas"
            ppr = p._p.get_or_add_pPr()
            shd = OxmlElement("w:shd")
            shd.set(qn("w:val"), "clear")
            shd.set(qn("w:color"), "auto")
            shd.set(qn("w:fill"), "F2F2F2")
            ppr.append(shd)
            if k == 0:
                p.paragraph_format.space_before = Pt(4)
            if k == len(lines) - 1:
                p.paragraph_format.space_after = Pt(10)
            else:
                keep_with_next(p)

    def equation(self, text):
        p = self.doc.add_paragraph(style="BB Equation")
        add_runs(p, text, italic=True)

    def placeholder(self, text):
        p = self.doc.add_paragraph(style="BB Body")
        p.alignment = WD_ALIGN_PARAGRAPH.CENTER
        r = p.add_run(f"[{text}]")
        r.font.highlight_color = WD_COLOR_INDEX.YELLOW
        r.bold = True
        return p

    def page_break(self):
        p = self.doc.add_paragraph()
        p.add_run().add_break(__import__("docx").enum.text.WD_BREAK.PAGE)

    # -- captions ---------------------------------------------------------- #
    def caption(self, kind, number, text, keep_next=False):
        chap, idx = number.rsplit(".", 1)
        p = self.doc.add_paragraph(style="Caption")
        p.add_run(f"{kind} {chap}.")
        field_runs(p, f"SEQ {kind} \\* ARABIC \\s 1", idx)
        add_runs(p, ": " + resolve(text, self.n))
        if keep_next:
            keep_with_next(p)
        return p

    # -- figures ----------------------------------------------------------- #
    def figure(self, parts):
        label, path, width, cap = parts[0], parts[1], float(parts[2]), parts[3]
        full = os.path.join(PROJECT_ROOT, path)
        if not os.path.exists(full):
            die(f"missing image {path}")
        from PIL import Image
        with Image.open(full) as im:
            w_px, h_px = im.size
        width = min(width, TEXT_WIDTH_IN)
        if width * h_px / w_px > MAX_FIG_HEIGHT_IN:
            width = MAX_FIG_HEIGHT_IN * w_px / h_px
        p = self.doc.add_paragraph(style="BB Figure")
        p.add_run().add_picture(full, width=Inches(width))
        self.caption("Figure", self.n.fig[label], cap)

    # -- tables ------------------------------------------------------------ #
    def table(self, parts, rows):
        label, cap = parts[0], parts[1]
        widths = [float(x) for x in parts[2].split(",")] if len(parts) > 2 and parts[2] else None
        font_size = float(parts[3]) if len(parts) > 3 and parts[3] else 10
        ncols = len(rows[0])
        for r in rows:
            if len(r) != ncols:
                die(f"table {label}: row has {len(r)} cells, header has {ncols}: {r}")
        if widths is None:
            widths = [TEXT_WIDTH_IN / ncols] * ncols
        if abs(sum(widths) - TEXT_WIDTH_IN) > 0.05:
            scale = TEXT_WIDTH_IN / sum(widths)
            widths = [w * scale for w in widths]
        self.caption("Table", self.n.tab[label], cap, keep_next=True)
        t = self.doc.add_table(rows=len(rows), cols=ncols)
        t.style = self.doc.styles["Table Grid"]
        t.alignment = WD_TABLE_ALIGNMENT.CENTER
        tblpr = t._tbl.tblPr
        lay = OxmlElement("w:tblLayout")
        lay.set(qn("w:type"), "fixed")
        tblpr.append(lay)
        for ri, row in enumerate(rows):
            tr = t.rows[ri]
            trpr = tr._tr.get_or_add_trPr()
            cs = OxmlElement("w:cantSplit")
            trpr.append(cs)
            if ri == 0:
                th = OxmlElement("w:tblHeader")
                trpr.append(th)
            for ci, text in enumerate(row):
                c = tr.cells[ci]
                c.width = Inches(widths[ci])
                p = c.paragraphs[0]
                p.style = self.doc.styles["BB Table Text"]
                if ri == 0:
                    set_cell_shading(c, "D9E2F3")
                    add_runs(p, resolve(text, self.n), size=font_size, bold=True)
                else:
                    add_runs(p, resolve(text, self.n), size=font_size)
                if ri < len(rows) - 1 and len(rows) <= 14:
                    keep_with_next(p)
        spacer = self.doc.add_paragraph(style="BB Body")
        spacer.paragraph_format.space_after = Pt(4)
        spacer.paragraph_format.line_spacing = 1.0

    # -- dispatch ---------------------------------------------------------- #
    def render(self, blocks, chapter_key=None, appendix=False):
        for kind, payload in blocks:
            if kind == "h1":
                if appendix:
                    self.appendix(chapter_key, payload[0])
                else:
                    self.chapter(chapter_key, payload[0])
            elif kind == "h2":
                self.heading(2, self._num_heading(payload[0], 2, chapter_key))
            elif kind == "h3":
                self.heading(3, self._num_heading(payload[0], 3, chapter_key))
            elif kind == "h4":
                self.runin(payload[0])
            elif kind == "p":
                self.para(payload)
            elif kind == "ul":
                self.bullets(payload)
            elif kind == "ol":
                self.bullets(payload, numbered=True)
            elif kind == "code":
                self.code(payload)
            elif kind == "fig":
                self.figure(payload)
            elif kind == "tab":
                self.table(*payload)
            elif kind == "eq":
                self.equation(payload[0])
            elif kind == "phblock":
                self.placeholder(payload[0])
            elif kind == "pagebreak":
                self.page_break()
            elif kind == "imgpages":
                self.image_pages(payload)
            else:
                die(f"unknown block kind {kind!r}")

    _h2 = 0
    _h3 = 0

    def _num_heading(self, text, level, key):
        if level == 2:
            self._h2 += 1
            self._h3 = 0
            return f"{key}.{self._h2} {text}"
        self._h3 += 1
        return f"{key}.{self._h2}.{self._h3} {text}"

    def reset_heading_counters(self):
        self._h2 = self._h3 = 0

    def image_pages(self, parts):
        import glob
        pattern = os.path.join(PROJECT_ROOT, parts[0])
        files = sorted(glob.glob(pattern))
        if not files:
            die(f"no images match {parts[0]}")
        for k, f in enumerate(files):
            p = self.doc.add_paragraph(style="BB Figure")
            p.paragraph_format.keep_with_next = False
            if k:
                p.paragraph_format.page_break_before = True
            p.add_run().add_picture(f, height=Inches(8.6))


# =========================================================================== #
# 7. Front matter                                                              #
# =========================================================================== #
def parse_front(path):
    """Sections introduced by '@@NAME' lines."""
    out, cur = {}, None
    with open(path, encoding="utf-8") as fh:
        for line in fh.read().splitlines():
            if line.startswith("%%"):
                continue
            if line.startswith("@@"):
                cur = line[2:].strip()
                out[cur] = []
                continue
            if cur is not None:
                out[cur].append(line)
    paras = {}
    for k, lines in out.items():
        if k == "ABBREVIATIONS":           # one entry per line
            paras[k] = [ln.strip() for ln in lines if ln.strip()]
            continue
        chunks, buf = [], []
        for ln in lines:
            if ln.strip():
                buf.append(ln.strip())
            elif buf:
                chunks.append(" ".join(buf))
                buf = []
        if buf:
            chunks.append(" ".join(buf))
        paras[k] = chunks
    return paras


def remove(el):
    el.getparent().remove(el)


def fill_front_matter(doc, front, numbers):
    body = doc.element.body
    kids = list(body.iterchildren())
    P = lambda i: __import__("docx").text.paragraph.Paragraph(kids[i], doc)  # noqa: E731

    def expect(i, text):
        got = "".join(t.text or "" for t in kids[i].iter(qn("w:t")))
        if text not in got:
            die(f"template changed: element {i} should contain {text!r}, has {got!r}")

    # --- cover ------------------------------------------------------------ #
    expect(4, "PROJECT TITLE NAME")
    set_text_keep_format(P(4), f"“{TITLE}”")
    expect(16, "Student")
    p16 = P(16)
    proto = copy.deepcopy(p16.runs[0]._element)
    for r in list(p16.runs):
        remove(r._element)

    def cover_run(text, highlight=False, brk=False):
        el = copy.deepcopy(proto)
        for t in el.findall(qn("w:t")):
            el.remove(t)
        p16._p.append(el)
        run = __import__("docx").text.run.Run(el, p16)
        run.add_text(text)
        if highlight:
            run.font.highlight_color = WD_COLOR_INDEX.YELLOW
        if brk:
            run.add_break()
    for k, s in enumerate(STUDENTS):
        cover_run(f"{s}  (PRN: ")
        cover_run("[PRN]", highlight=True)
        cover_run(")", brk=k < len(STUDENTS) - 1)
    expect(23, "Prof. XYZ")
    set_text_keep_format(P(23), GUIDE[0] + "  (Guide)")
    expect(25, "Designation")
    p25 = P(25)
    set_text_keep_format(p25, GUIDE[1])
    r = p25.add_run()
    r.add_break()
    p25.add_run(CO_GUIDE[0] + "  (Co-guide), ")
    rr = p25.add_run(CO_GUIDE[1])
    rr.font.highlight_color = WD_COLOR_INDEX.YELLOW
    # The cover's spacer paragraphs make room for four names and a co-guide.
    for i in (17, 18, 19, 26, 27):
        remove(kids[i])

    # --- cover section break (no page number on the cover) ----------------- #
    final_sect = body.find(qn("w:sectPr"))
    cover_sect = copy.deepcopy(final_sect)
    for fr in cover_sect.findall(qn("w:footerReference")):
        cover_sect.remove(fr)
    for tag in ("w:pgNumType", "w:titlePg"):
        el = cover_sect.find(qn(tag))
        if el is not None:
            cover_sect.remove(el)
    ppr = P(35)._p.get_or_add_pPr()
    ppr.append(cover_sect)

    # --- certificate -------------------------------------------------------- #
    expect(41, "CERTIFICATE")
    P(41).style = doc.styles["BB Front Heading"]
    expect(42, "This is to certify")
    p42 = P(42)
    for r in list(p42.runs):
        remove(r._element)
    names = ", ".join(STUDENTS[:-1]) + " and " + STUDENTS[-1]
    add_runs(p42, f"This is to certify that the project work entitled **“{TITLE}”** is "
                  f"carried out by **{names}** in partial fulfilment for the award of the degree of "
                  f"**Bachelor of Technology** in **Artificial Intelligence & Machine Learning**, "
                  f"Symbiosis International (Deemed University), Pune, during the academic year "
                  f"2026-2027.")
    tbl = kids[47]
    rows = tbl.findall(qn("w:tr"))
    cells = rows[1].findall(qn("w:tc"))
    from docx.table import _Cell
    for tc, (name, role) in zip(cells[:2], ((GUIDE[0], "Guide"), (CO_GUIDE[0], "Co-Guide"))):
        c = _Cell(tc, None)
        p = c.paragraphs[0]
        r0 = p.runs[0] if p.runs else p.add_run()
        bold = r0.bold
        txt = p.text
        for r in list(p.runs):
            remove(r._element)
        nr = p.add_run(name)
        nr.bold = True
        nr.add_break()
        p.add_run(txt)

    # --- declaration -------------------------------------------------------- #
    expect(50, "DECLARATION")
    P(50).style = doc.styles["BB Front Heading"]
    P(50).paragraph_format.page_break_before = True
    for i in (48, 49):                  # empty spacers after the signature table
        remove(kids[i])
    expect(51, "I hereby declare")
    p51 = P(51)
    for r in list(p51.runs):
        remove(r._element)
    add_runs(p51, f"We hereby declare that the project titled **“{TITLE}”** submitted to "
                  "Symbiosis Institute of Technology, Constituent of Symbiosis International "
                  "(Deemed University), Pune, for the award of the degree of Bachelor of Technology "
                  "in Artificial Intelligence & Machine Learning is a result of original research "
                  "carried out by us. We understand that our report may be made electronically "
                  "available to the public. It is further declared that the project report or any "
                  "part thereof has not been previously submitted to any University or Institute "
                  "for the award of any degree or diploma.")
    expect(52, "Name(s) of Student(s)")
    p52 = P(52)
    p52.add_run(" " + ", ".join(STUDENTS))
    expect(53, "PRN")
    p53 = P(53)
    for k in range(len(STUDENTS)):
        if k:
            p53.add_run(", ")
        r2 = p53.add_run("[PRN]")
        r2.font.highlight_color = WD_COLOR_INDEX.YELLOW
    expect(56, "Title of the project")
    P(56).add_run(" " + TITLE)
    # The filled declaration is longer than the template's; tighten it so the
    # signature block stays on the same page.
    for i in range(51, 61):
        pf = P(i).paragraph_format
        pf.line_spacing = 1.5
        pf.space_before = Pt(0)
        pf.space_after = Pt(6)
    remove(kids[57])

    # --- acknowledgement ---------------------------------------------------- #
    expect(62, "ACKNOWLEDGEMENT")
    P(62).style = doc.styles["BB Front Heading"]
    ack = front["ACK"]
    template_ack = [kids[i] for i in range(63, 68)]
    for k, el in enumerate(template_ack):
        p = __import__("docx").text.paragraph.Paragraph(el, doc)
        for r in list(p.runs):
            remove(r._element)
        if k < len(ack):
            add_runs(p, ack[k])
    last = template_ack[-1]
    for text in ack[len(template_ack):]:
        new = copy.deepcopy(last)
        last.addnext(new)
        last = new
        p = __import__("docx").text.paragraph.Paragraph(new, doc)
        for r in list(p.runs):
            remove(r._element)
        add_runs(p, text)

    # --- abstract ----------------------------------------------------------- #
    expect(69, "ABSTRACT")
    P(69).style = doc.styles["BB Front Heading"]
    abstract = front["ABSTRACT"]
    anchor = kids[69]
    for i in range(70, 76):
        remove(kids[i])
    for text in abstract:
        p = doc.add_paragraph(style="BB Body")
        add_runs(p, text)
        anchor.addnext(p._p)
        anchor = p._p
    expect(76, "Keywords")
    p76 = P(76)
    p76.style = doc.styles["BB Body"]
    for r in list(p76.runs):
        remove(r._element)
    add_runs(p76, "**Keywords:** " + " ".join(front["KEYWORDS"]))

    # --- contents and lists: real fields replace the typed tables ----------- #
    def field_paragraph(after_el, instr, hint):
        p = doc.add_paragraph(style="BB Body")
        field_runs(p, instr, hint)
        after_el.addnext(p._p)
        return p

    for head_i, tbl_i, instr in ((79, 80, 'TOC \\o "1-3" \\h \\z \\u'),
                                 (82, 83, 'TOC \\h \\z \\c "Table"'),
                                 (85, 86, 'TOC \\h \\z \\c "Figure"')):
        P(head_i).style = doc.styles["BB Front Heading"]
        remove(kids[tbl_i])
        field_paragraph(kids[head_i], instr,
                        "Press Ctrl+A, then F9, to update this list.")
    remove(kids[87])

    # --- abbreviations ------------------------------------------------------ #
    expect(89, "LIST OF ABBREVIATIONS")
    P(89).style = doc.styles["BB Front Heading"]
    for i in range(90, 102):
        remove(kids[i])
    abbr = [ln.split("|", 1) for ln in front["ABBREVIATIONS"]]
    t = doc.add_table(rows=len(abbr) + 1, cols=2)
    t.alignment = WD_TABLE_ALIGNMENT.CENTER
    hdr = t.rows[0].cells
    for c, txt, w in ((hdr[0], "Abbreviation", 1.6), (hdr[1], "Meaning", 4.2)):
        c.width = Inches(w)
        p = c.paragraphs[0]
        p.style = doc.styles["BB Table Text"]
        add_runs(p, txt, size=11, bold=True)
    for k, (a, m) in enumerate(abbr, start=1):
        for c, txt, w in ((t.rows[k].cells[0], a.strip(), 1.6),
                          (t.rows[k].cells[1], m.strip(), 4.2)):
            c.width = Inches(w)
            p = c.paragraphs[0]
            p.style = doc.styles["BB Table Text"]
            add_runs(p, txt, size=11)
            p.paragraph_format.space_before = Pt(2)
            p.paragraph_format.space_after = Pt(2)
    kids[89].addnext(t._tbl)

    # --- front-matter section: roman numerals, number on every page --------- #
    sect_p = kids[102]
    fm_sect = sect_p.find(qn("w:pPr")).find(qn("w:sectPr"))
    tp = fm_sect.find(qn("w:titlePg"))
    if tp is not None:
        fm_sect.remove(tp)
    pg = fm_sect.find(qn("w:pgNumType"))
    pg.set(qn("w:fmt"), "lowerRoman")
    pg.set(qn("w:start"), "1")

    # --- drop the template's Chapter 1 placeholder; main section arabic ------ #
    for i in range(103, 122):
        remove(kids[i])
    tp = final_sect.find(qn("w:titlePg"))
    if tp is not None:
        final_sect.remove(tp)
    for fr in final_sect.findall(qn("w:footerReference")):
        final_sect.remove(fr)   # inherit the front matter's PAGE footer
    pg = final_sect.find(qn("w:pgNumType"))
    pg.set(qn("w:fmt"), "decimal")
    pg.set(qn("w:start"), "1")


PRN_RE = re.compile(r"\b\d{10,12}\b")


def _el_text(el):
    return "".join(t.text or "" for t in el.iter(qn("w:t")))


def prepare_base(doc):
    """Keep the edited front matter; drop Chapter 1 onwards; align PRNs.

    The cover page is the authority for PRNs (author's instruction, 28 Oct).
    Every other PRN list is compared with it in student order, and a
    differing value is replaced in place, run by run, so the formatting of
    the edited file is kept. Returns the cover PRNs and (where, who, found,
    cover) rows.
    """
    body = doc.element.body
    kids = list(body.iterchildren())
    sect = body.find(qn("w:sectPr"))

    cover_el = next((el for el in kids if "(PRN:" in _el_text(el)), None)
    if cover_el is None:
        die("base file: no cover paragraph with '(PRN: ...)' found")
    cover_txt = _el_text(cover_el)
    cover = []
    for s_ in STUDENTS:
        m = re.search(re.escape(s_) + r"\s*\(PRN:\s*(\d+)\)", cover_txt)
        if not m:
            die(f"base file: cover has no PRN for {s_}")
        cover.append(m.group(1))

    report = []
    for el in body.iter(qn("w:p")):
        if el is cover_el:
            continue
        txt = _el_text(el)
        found = PRN_RE.findall(txt)
        if not found:
            continue
        where = txt[:40].strip()
        if len(found) == len(cover):
            for k, (f, c) in enumerate(zip(found, cover)):
                report.append((where, STUDENTS[k], f, c))
                if f != c:
                    hit = False
                    for t in el.iter(qn("w:t")):
                        if t.text and f in t.text:
                            t.text = t.text.replace(f, c, 1)
                            hit = True
                            break
                    if not hit:
                        die(f"PRN {f} is split across runs in {where!r}; fix it by hand")
        else:
            for f in found:
                report.append((where, "?", f, "(not a full list: not changed)"))

    start = None
    for i, el in enumerate(kids):
        if el.tag == qn("w:p"):
            ps = el.find(qn("w:pPr"))
            st = ps.find(qn("w:pStyle")) if ps is not None else None
            if st is not None and st.get(qn("w:val")) == "Heading1" and \
                    _el_text(el).startswith("Chapter 1"):
                start = i
                break
    if start is None:
        die("base file: no 'Chapter 1' Heading 1 found")
    for el in kids[start:]:
        if el is not sect:
            body.remove(el)
    return cover, report


def references(doc, renderer, refs, numbers):
    renderer.unnumbered_h1("References")
    for key in numbers.cited_order:
        p = doc.add_paragraph(style="BB Body")
        p.paragraph_format.left_indent = Inches(0.45)
        p.paragraph_format.first_line_indent = Inches(-0.45)
        p.paragraph_format.line_spacing = 1.0
        p.paragraph_format.space_after = Pt(3)
        p.alignment = WD_ALIGN_PARAGRAPH.LEFT
        add_runs(p, f"[{numbers.cites[key]}]\t" + refs[key], size=10.5)
        tabs = p.paragraph_format.tab_stops
        tabs.add_tab_stop(Inches(0.45))


def load_refs(path):
    refs = {}
    with open(path, encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line or line.startswith("%%"):
                continue
            key, _, text = line.partition("|")
            refs[key.strip()] = text.strip()
    return refs


def set_update_fields(path):
    """Ensure <w:updateFields w:val="true"/> in word/settings.xml."""
    tmp = path + ".tmp"
    with zipfile.ZipFile(path) as zin, zipfile.ZipFile(tmp, "w", zipfile.ZIP_DEFLATED) as zout:
        for item in zin.infolist():
            data = zin.read(item.filename)
            if item.filename == "word/settings.xml":
                s = data.decode("utf-8")
                if "w:updateFields" not in s:
                    s = re.sub(r"(<w:settings[^>]*>)", r'\1<w:updateFields w:val="true"/>', s, count=1)
                data = s.encode("utf-8")
            zout.writestr(item, data)
    os.replace(tmp, path)


WORD_PS = r"""
param([string]$In, [string]$Pdf)
$w = New-Object -ComObject Word.Application
# Never quit a Word instance that already holds someone's documents.
$own = ($w.Documents.Count -eq 0)
$w.Visible = $false
$w.DisplayAlerts = 0
try {
  $d = $w.Documents.Open($In, $false, $false, $false)
  $d.Fields.Update() | Out-Null
  foreach ($t in $d.TablesOfContents) { $t.Update() | Out-Null }
  foreach ($t in $d.TablesOfFigures) { $t.Update() | Out-Null }
  $d.Repaginate()
  foreach ($t in $d.TablesOfContents) { $t.UpdatePageNumbers() | Out-Null }
  foreach ($t in $d.TablesOfFigures) { $t.UpdatePageNumbers() | Out-Null }
  $pages = $d.ComputeStatistics(2)
  Write-Output "PAGES=$pages"
  if ($d.TablesOfContents.Count -gt 0) {
    Write-Output "TOC<<"
    Write-Output $d.TablesOfContents.Item(1).Range.Text
    Write-Output ">>TOC"
  }
  $d.Save()
  if ($Pdf) { $d.ExportAsFixedFormat($Pdf, 17) }
  $d.Close(0)
} finally {
  if ($own) { $w.Quit() }
  [System.Runtime.InteropServices.Marshal]::ReleaseComObject($w) | Out-Null
}
"""


def word_update(path, pdf=None):
    if sys.platform != "win32":
        print("  (not Windows: open the file in Word and press Ctrl+A, F9)")
        return None
    with tempfile.NamedTemporaryFile("w", suffix=".ps1", delete=False, encoding="utf-8") as fh:
        fh.write(WORD_PS)
        ps1 = fh.name
    try:
        cmd = ["powershell", "-NoProfile", "-ExecutionPolicy", "Bypass", "-File", ps1,
               "-In", path]
        if pdf:
            cmd += ["-Pdf", pdf]
        out = subprocess.run(cmd, capture_output=True, text=True, timeout=900)
        if out.returncode != 0:
            print("  Word update failed; the fields update when the file is opened.")
            print(out.stdout[-2000:], out.stderr[-2000:])
            return None
        m = re.search(r"PAGES=(\d+)", out.stdout)
        toc = re.search(r"TOC<<(.*)>>TOC", out.stdout, re.S)
        if toc:
            print("  Contents (top level, with start pages):")
            for line in toc.group(1).replace("\r", "\n").split("\n"):
                line = line.strip()
                if re.match(r"^(Chapter \d|References|Appendix|[A-Z ]{6,})", line):
                    print("    " + re.sub(r"\s+", " ", line))
        return int(m.group(1)) if m else None
    finally:
        os.unlink(ps1)


# =========================================================================== #
# 8. Main                                                                      #
# =========================================================================== #
def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--no-word", action="store_true",
                    help="skip opening the file in Word to update fields")
    ap.add_argument("--pdf", action="store_true", help="also export a PDF via Word")
    ap.add_argument("--lenient", action="store_true",
                    help="partial builds: unresolved cross-references print as '?'")
    ap.add_argument("--from-template", action="store_true",
                    help="old path: fill the department template instead of the edited base")
    ap.add_argument("--chapters", default="", help="comma list of chapter numbers to include "
                    "(default: every chapter whose file exists)")
    args = ap.parse_args()
    global LENIENT
    LENIENT = args.lenient

    if not os.path.exists(TEMPLATE):
        die(f"template not found: {TEMPLATE}")
    refs = load_refs(os.path.join(CONTENT, "references.txt"))
    front = parse_front(os.path.join(CONTENT, "00_front.txt"))

    doc_blocks = []
    wanted = {c.strip() for c in args.chapters.split(",") if c.strip()}
    for k, fname in enumerate(CHAPTER_FILES, start=1):
        path = os.path.join(CONTENT, fname)
        if not os.path.exists(path) or (wanted and str(k) not in wanted):
            continue
        doc_blocks.append((str(k), parse_file(path)))
    app_path = os.path.join(CONTENT, APPENDIX_FILE)
    appendices = []
    if os.path.exists(app_path) and not wanted:
        blocks = parse_file(app_path)
        cur = None
        letter = ord("A")
        for b in blocks:
            if b[0] == "h1":
                cur = (chr(letter), [b])
                appendices.append(cur)
                letter += 1
            elif cur is not None:
                cur[1].append(b)
    # Only cite references that the included chapters use (partial builds).
    used_text = " ".join(t for _, bl in doc_blocks + appendices for b in bl for t in iter_text(b))
    used = {k.strip().lstrip("@") for m in CITE_RE.finditer(used_text) for k in m.group(1).split(";")}
    if wanted or LENIENT:
        refs = {k: v for k, v in refs.items() if k in used}
    numbers = number_everything(doc_blocks + appendices, refs)

    if args.from_template:
        doc = Document(TEMPLATE)
        setup_styles(doc)
        fill_front_matter(doc, front, numbers)
    else:
        if not os.path.exists(BASE):
            die(f"edited base not found: {BASE}")
        doc = Document(BASE)
        cover, report = prepare_base(doc)
        print("  PRNs on the cover (authority):")
        for s_, c in zip(STUDENTS, cover):
            print(f"    {s_}: {c}")
        print("  PRNs found elsewhere:")
        for where, who, found, want in report:
            flag = "" if found == want else f"  -> changed to {want}"
            print(f"    [{where}] {who}: {found}{flag}")
    lists = Lists(doc)
    r = Renderer(doc, numbers, lists)
    for key, blocks in doc_blocks:
        r.reset_heading_counters()
        r.render(blocks, chapter_key=key)
    references(doc, r, refs, numbers)
    for letter, blocks in appendices:
        r.reset_heading_counters()
        r.render(blocks, chapter_key=letter, appendix=True)

    props = doc.core_properties
    props.title = TITLE
    if args.from_template:
        props.author = ", ".join(STUDENTS)
    props.subject = "B.Tech project report, Department of AI & ML, SIT Pune, AY 2026-27"
    os.makedirs(BB, exist_ok=True)
    doc.save(OUTPUT)
    set_update_fields(OUTPUT)
    print(f"wrote {os.path.relpath(OUTPUT, PROJECT_ROOT)}")
    print(f"  figures {len(numbers.fig)}, tables {len(numbers.tab)}, references {len(numbers.cites)}")
    if not args.no_word:
        pdf = OUTPUT[:-5] + ".pdf" if args.pdf else None
        pages = word_update(OUTPUT, pdf)
        set_update_fields(OUTPUT)
        if pages:
            print(f"  Word: fields updated, {pages} pages")
        if pdf and os.path.exists(pdf):
            with open(pdf, "rb") as fh:
                raw = fh.read()
            counts = [int(x) for x in re.findall(rb"/Type\s*/Pages\b[^>]*?/Count\s+(\d+)", raw)]
            counts += [int(x) for x in re.findall(rb"/Count\s+(\d+)[^>]*?/Type\s*/Pages\b", raw)]
            print(f"  wrote {os.path.relpath(pdf, PROJECT_ROOT)}; PDF page count "
                  f"(metadata): {max(counts) if counts else 'not found'}")


if __name__ == "__main__":
    main()
