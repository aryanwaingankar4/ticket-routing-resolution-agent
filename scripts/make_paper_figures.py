"""Hand-written paper figures that the Phase 8A builder does not produce.

Mentor revision (2026-09-28). Every plotted value is read BY ID from the
committed, generated paper/NUMBERS.md -- the same file paper/numbers.tex is
generated from -- so a figure cannot disagree with the \\nb{} macros the text
quotes. No value is typed here, and no interval is recomputed: an error bar is
drawn only where NUMBERS.md already carries a Wilson 95% interval.

Writes ONLY .pdf and .png under paper/figures/. Paper parity
(tests/test_paper_artifacts.py) does not byte-compare pixels, and this script
lives outside paper/ so the parity test does not see a file the builder does
not produce.

    .\\venv\\Scripts\\python.exe scripts/make_paper_figures.py
"""

from __future__ import annotations

import os
import re
import sys

PROJECT_ROOT = os.path.abspath(
    os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
NUMBERS_MD = os.path.join(PROJECT_ROOT, "paper", "NUMBERS.md")
FIG_DIR = os.path.join(PROJECT_ROOT, "paper", "figures")

# Same style constants as src/experiments/build_paper_artifacts.py, so the new
# figure matches the generated ones.
FIG_DPI = 300
BASE_FONT_PT = 8
TICK_FONT_PT = 7
OKABE_ITO = ["#0072B2", "#E69F00", "#009E73", "#D55E00", "#CC79A7",
             "#56B4E9", "#F0E442", "#000000"]

# (label, NUMBERS.md id, group). Order is the plotting order, top to bottom.
BENCHMARK45_METHODS = [
    ("TF-IDF + LogReg (Tier-1 only)", "T1.tier1only.benchmark45", "trained"),
    ("DistilBERT, fine-tuned (retrain)", "T1.distilbert.benchmark45",
     "trained"),
    ("DistilBERT, fine-tuned (original)",
     "T1.distilbert.reference.benchmark45", "trained"),
    ("E5 + LogReg", "T1.e5.benchmark45", "trained"),
    ("MiniLM + LogReg", "T1.minilm.benchmark45", "trained"),
    ("Cascade (production)", "T1.cascade.benchmark45", "production"),
    ("BGE + LogReg (Tier-2 only)", "T1.tier2only.benchmark45", "trained"),
    ("Zero-shot Qwen2.5-3B", "T1.zeroshot_qwen.benchmark45", "zeroshot"),
    ("Zero-shot Gemini flash-lite", "T1.zeroshot_gemini.benchmark45",
     "zeroshot"),
]
GROUP_STYLE = {
    "trained": (OKABE_ITO[0], "trained classifier"),
    "production": (OKABE_ITO[2], "production cascade"),
    "zeroshot": (OKABE_ITO[1], "zero-shot LLM (no gate)"),
}

_ROW = re.compile(r"^\| `([^`]+)` \| ([^|]*) \|", re.M)
_FRAC = re.compile(r"^(\d+)/(\d+)(?: \[95% CI ([0-9.]+), ([0-9.]+)\])?$")


def read_numbers():
    with open(NUMBERS_MD, "r", encoding="utf-8") as fh:
        return {key: value.strip() for key, value in _ROW.findall(fh.read())}


def parse_fraction(numbers, key):
    if key not in numbers:
        sys.exit(f"ERROR: {key} is not in paper/NUMBERS.md -- rebuild the "
                 "paper artifacts or fix the id; nothing is plotted from a "
                 "typed value.")
    match = _FRAC.match(numbers[key])
    if not match:
        sys.exit(f"ERROR: {key} = {numbers[key]!r} is not 'k/n' or "
                 "'k/n [95% CI lo, hi]'.")
    k, n = int(match.group(1)), int(match.group(2))
    ci = ((float(match.group(3)), float(match.group(4)))
          if match.group(3) else None)
    return k, n, ci


def figure_benchmark45_accuracy(numbers):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.rcParams.update({
        "font.family": "sans-serif", "font.sans-serif": ["DejaVu Sans"],
        "font.size": BASE_FONT_PT, "axes.labelsize": BASE_FONT_PT,
        "xtick.labelsize": TICK_FONT_PT, "ytick.labelsize": TICK_FONT_PT,
        "legend.fontsize": TICK_FONT_PT, "axes.grid": True,
        "grid.alpha": 0.3, "grid.linewidth": 0.5,
        "axes.spines.top": False, "axes.spines.right": False,
        "savefig.dpi": FIG_DPI, "savefig.bbox": "tight",
        "pdf.fonttype": 42, "svg.hashsalt": "mentor-revision",
    })

    # Second, independent derivation of the denominator: every row must be
    # scored on the benchmark NUMBERS.md says has T19.benchmark45.n tickets.
    n_expected = int(numbers["T19.benchmark45.n"])
    rows = []
    for label, key, group in BENCHMARK45_METHODS:
        k, n, ci = parse_fraction(numbers, key)
        if n != n_expected:
            sys.exit(f"ERROR: {key} is scored out of {n}, but "
                     f"T19.benchmark45.n is {n_expected}.")
        rows.append((label, k, n, ci, group))

    fig, ax = plt.subplots(figsize=(3.45, 2.55))
    ys = list(range(len(rows)))[::-1]
    for y, (label, k, n, ci, group) in zip(ys, rows):
        acc = k / n
        color = GROUP_STYLE[group][0]
        ax.barh(y, acc, height=0.62, color=color, zorder=2)
        if ci is not None:
            ax.errorbar(acc, y, xerr=[[acc - ci[0]], [ci[1] - acc]],
                        fmt="none", ecolor="#333333", elinewidth=0.7,
                        capsize=1.8, zorder=3)
        right = ci[1] if ci is not None else acc
        ax.text(right + 0.012, y, f"{k}/{n}", va="center",
                fontsize=TICK_FONT_PT - 0.5, color="#333333")
    ax.set_yticks(ys)
    ax.set_yticklabels([r[0] for r in rows])
    ax.set_xlim(0, 1.08)
    ax.set_xlabel(f"Accuracy on the {n_expected}-ticket out-of-template "
                  "benchmark")
    ax.grid(axis="y", visible=False)
    handles = [plt.Rectangle((0, 0), 1, 1, color=c)
               for c, _ in GROUP_STYLE.values()]
    ax.legend(handles, [name for _, name in GROUP_STYLE.values()],
              loc="lower center", bbox_to_anchor=(0.42, 1.0), ncol=3,
              frameon=False, borderpad=0.2, handlelength=1.0,
              columnspacing=0.8, fontsize=TICK_FONT_PT - 0.5)

    os.makedirs(FIG_DIR, exist_ok=True)
    base = os.path.join(FIG_DIR, "G1_benchmark45_accuracy_all_methods")
    fig.savefig(base + ".pdf", metadata={"CreationDate": None})
    fig.savefig(base + ".png", metadata={"Software": None,
                                         "Creation Time": None})
    plt.close(fig)
    return base


def main():
    numbers = read_numbers()
    base = figure_benchmark45_accuracy(numbers)
    print(f"wrote {os.path.relpath(base, PROJECT_ROOT)}.pdf / .png")


if __name__ == "__main__":
    main()
