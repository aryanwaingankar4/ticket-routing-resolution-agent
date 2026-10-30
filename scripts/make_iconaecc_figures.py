"""Figs. 1 and 2 of the IConAECC 2026 paper.

Fig. 1 is v4's Fig. 1 (paper_docx/fig/fig1_pipeline.png) cropped to its
drawn content (white margins removed, nothing redrawn), so it can sit at
column width. Fig. 2 is v4's Figs. 4 and 5 as one figure.

Offline, model-free. Panel (a) is v4's Fig. 4 (cascade threshold) and panel
(b) is v4's Fig. 5 (retrieval threshold sweep), drawn by the SAME functions in
make_revision_figures.py from the same committed result files, with the same
source assertions; only the panel height changes (2.35 in -> PANEL_H_IN) and
an (a)/(b) label is added. The panels are stacked so that the figure stays at
column width (3.4 in) and its text stays at 6.5-8 pt.

    .\\venv\\Scripts\\python.exe scripts/make_iconaecc_figures.py [--force]
"""

from __future__ import annotations

import argparse
import io
import os

import make_revision_figures as mrf

OUT = os.path.join(mrf.PROJECT_ROOT, "paper_ieee_docx", "fig",
                   "fig2_thresholds.png")
FIG1_SRC = os.path.join(mrf.PROJECT_ROOT, "paper_docx", "fig", "fig1_pipeline.png")
FIG1_OUT = os.path.join(mrf.PROJECT_ROOT, "paper_ieee_docx", "fig",
                        "fig1_pipeline_cropped.png")
PANEL_H_IN = 1.65
CROP_PAD_PX = 12


def panel(plt, fig, label):
    fig.set_size_inches(mrf.WIDTH_IN, PANEL_H_IN)
    ax = fig.axes[0]
    ax.text(-0.13, 1.02, label, transform=ax.transAxes, fontsize=mrf.BASE_PT,
            fontweight="bold", va="bottom", ha="left", color=mrf.INK)
    fig.tight_layout(pad=0.3)
    buf = io.BytesIO()
    fig.savefig(buf, dpi=mrf.FIG_DPI, format="png")
    plt.close(fig)
    buf.seek(0)
    return buf


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args()
    for p in (OUT, FIG1_OUT):
        if os.path.exists(p) and not args.force:
            mrf.die(f"{p} exists (use --force)")

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from PIL import Image
    from src.agent.config import settings

    mrf.style(plt)
    a = Image.open(panel(plt, mrf.fig4(plt, settings), "(a)"))
    b = Image.open(panel(plt, mrf.fig5(plt, settings), "(b)"))
    if a.width != b.width:
        mrf.die(f"panel widths differ: {a.width} vs {b.width}")
    out = Image.new("RGB", (a.width, a.height + b.height), "white")
    out.paste(a.convert("RGB"), (0, 0))
    out.paste(b.convert("RGB"), (0, a.height))
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    out.save(OUT, dpi=(mrf.FIG_DPI, mrf.FIG_DPI))
    print(f"[write] {os.path.relpath(OUT, mrf.PROJECT_ROOT)} "
          f"({out.width}x{out.height} px)")

    from PIL import ImageChops
    src = Image.open(FIG1_SRC).convert("RGB")
    box = ImageChops.difference(src, Image.new("RGB", src.size, "white")).getbbox()
    box = (max(box[0] - CROP_PAD_PX, 0), max(box[1] - CROP_PAD_PX, 0),
           min(box[2] + CROP_PAD_PX, src.width),
           min(box[3] + CROP_PAD_PX, src.height))
    src.crop(box).save(FIG1_OUT, dpi=(mrf.FIG_DPI, mrf.FIG_DPI))
    print(f"[write] {os.path.relpath(FIG1_OUT, mrf.PROJECT_ROOT)} "
          f"(crop {box} of {src.width}x{src.height})")


if __name__ == "__main__":
    main()
