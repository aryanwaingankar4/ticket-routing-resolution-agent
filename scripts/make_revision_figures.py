"""Figures 4-6 for the Word paper (paper revision, Oct 2026).

Offline, model-free. Every plotted value is read from a committed result file;
nothing is typed here. The script asserts the three values the paper's text
quotes for the retrieval sweep (175/175, 13.3%, 37.8%) and the adversarial
safe range [0.6196, 0.6704] before drawing.

  Fig. 4  data/paper_revision_oct/cascade_calibration_curve_paraphrased175.csv
          (scripts/reproduce_cascade_calibration_curve.py)
  Fig. 5  data/rag_similarity_calibration_combined.csv
          + data/adversarial_escalation_results.csv (safe range)
  Fig. 6  data/paper_revision_oct/confusion_matrix_cascade_benchmark45.csv
          (scripts/error_analysis_benchmark45.py)

PNG, 300 dpi, 3.4 in wide, serif, Okabe-Ito colours. Writes only under
paper_docx/fig/ and refuses to overwrite without --force.

    .\\venv\\Scripts\\python.exe scripts/make_revision_figures.py
"""

from __future__ import annotations

import argparse
import csv
import os
import sys

PROJECT_ROOT = os.path.abspath(
    os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

DATA = os.path.join(PROJECT_ROOT, "data")
REV = os.path.join(DATA, "paper_revision_oct")
FIG_DIR = os.path.join(PROJECT_ROOT, "paper_docx", "fig")
CURVE_CSV = os.path.join(REV, "cascade_calibration_curve_paraphrased175.csv")
CALIB_CSV = os.path.join(DATA, "cascade_calibration_attempts.csv")
RAG_CSV = os.path.join(DATA, "rag_similarity_calibration_combined.csv")
ADV_CSV = os.path.join(DATA, "adversarial_escalation_results.csv")
CM_CSV = os.path.join(REV, "confusion_matrix_cascade_benchmark45.csv")
OUTS = {
    "f4": os.path.join(FIG_DIR, "fig4_cascade_threshold.png"),
    "f5": os.path.join(FIG_DIR, "fig5_retrieval_sweep.png"),
    "f6": os.path.join(FIG_DIR, "fig6_confusion_matrix.png"),
}

FIG_DPI = 300
WIDTH_IN = 3.4
BASE_PT, TICK_PT, NOTE_PT = 8, 7, 6.5
BLUE, VERMILLION, GREEN, ORANGE = "#0072B2", "#D55E00", "#009E73", "#E69F00"
INK, MUTED, GRID = "#222222", "#666666", "#DDDDDD"
# Below this many kept tickets one ticket moves kept accuracy by >= 20 points,
# so the accuracy line is not drawn there (stated in the figure).
MIN_KEPT_FOR_ACCURACY = 5


def die(msg):
    print(f"\n[FATAL] {msg}")
    raise SystemExit(1)


def read_csv(path):
    with open(path, newline="", encoding="utf-8") as fh:
        return list(csv.DictReader(fh))


def style(plt):
    plt.rcParams.update({
        "font.family": "serif",
        "font.serif": ["Times New Roman", "DejaVu Serif"],
        "font.size": BASE_PT, "axes.labelsize": BASE_PT,
        "xtick.labelsize": TICK_PT, "ytick.labelsize": TICK_PT,
        "legend.fontsize": TICK_PT, "axes.edgecolor": MUTED,
        "axes.labelcolor": INK, "xtick.color": MUTED, "ytick.color": MUTED,
        "text.color": INK, "axes.spines.top": False,
        "axes.spines.right": False, "savefig.dpi": FIG_DPI,
        "mathtext.fontset": "stix",
    })


def fig4(plt, settings):
    rows = read_csv(CURVE_CSV)
    thr = [float(r["threshold"]) for r in rows]
    share = [float(r["share_kept"]) for r in rows]
    acc_x = [float(r["threshold"]) for r in rows
             if int(r["n_kept"]) >= MIN_KEPT_FOR_ACCURACY]
    acc_y = [float(r["kept_accuracy"]) for r in rows
             if int(r["n_kept"]) >= MIN_KEPT_FOR_ACCURACY]
    by_thr = {r["threshold"]: r for r in rows}
    targets = sorted({float(r["target_accuracy"]) for r in read_csv(CALIB_CSV)
                      if r["attempt"] == "3"}
                     | {float(r["target_accuracy"]) for r in
                        read_csv(os.path.join(DATA, "cascade_threshold_sweep.csv"))})
    live = settings.cascade.confidence_threshold

    fig, ax = plt.subplots(figsize=(WIDTH_IN, 2.35))
    ax.grid(axis="y", color=GRID, linewidth=0.5)
    ax.set_axisbelow(True)
    for t in targets:
        ax.axhline(t, color=MUTED, linewidth=0.6, linestyle=":")
        ax.text(0.845, t + 0.012, f"target {t:.2f}", fontsize=NOTE_PT,
                color=MUTED, va="bottom", ha="right")
    ax.plot(acc_x, acc_y, color=BLUE, linewidth=1.5, label="accuracy of kept")
    ax.plot(thr, share, color=VERMILLION, linewidth=1.5,
            label="share kept")
    ax.axvline(live, color=INK, linewidth=0.9, linestyle="--")
    # The committed 10%-bucket edges, labelled with their counts.
    for edge in ("0.50", "0.60"):
        r = by_thr[edge]
        x, y = float(edge), float(r["kept_accuracy"])
        ax.plot([x], [y], marker="o", markersize=4, color=BLUE,
                markeredgecolor="white", markeredgewidth=0.6, zorder=5)
        ax.annotate(f"{r['n_kept_correct']}/{r['n_kept']}", (x, y),
                    xytext=(4, 4), textcoords="offset points",
                    fontsize=NOTE_PT, color=INK)
    ax.text(live + 0.008, 0.03, f"live {live:.2f}", fontsize=NOTE_PT,
            color=INK)
    ax.set_xlim(0.15, 0.85)
    ax.set_ylim(0, 1.0)
    ax.set_xlabel("Tier-1 confidence threshold")
    ax.set_ylabel("Proportion of 175 tickets")
    ax.legend(loc="lower left", bbox_to_anchor=(0.0, 0.0), frameon=False,
              handlelength=1.6)
    ax.text(0.845, 0.27, f"accuracy drawn where ≥{MIN_KEPT_FOR_ACCURACY} "
            "tickets are kept", fontsize=NOTE_PT, color=MUTED, ha="right")
    fig.tight_layout(pad=0.3)
    return fig


def fig5(plt, settings):
    rows = read_csv(RAG_CSV)
    rows.sort(key=lambda r: float(r["threshold"]))
    thr = [float(r["threshold"]) for r in rows]
    n_in = [int(r["n_in_domain_proceed"]) + int(r["n_in_domain_escalate"])
            for r in rows]
    if len(set(n_in)) != 1:
        die("in-domain total varies across thresholds")
    n_in = n_in[0]
    proceed = [int(r["n_in_domain_proceed"]) / n_in for r in rows]
    leak = [float(r["ood_leakage_rate"]) for r in rows]
    gate = settings.rag.similarity_threshold
    by = {float(r["threshold"]): r for r in rows}

    # The three values the paper quotes, asserted rather than trusted.
    g = by[gate]
    two_below = thr[thr.index(gate) - 2]
    if int(g["n_in_domain_proceed"]) != n_in or n_in != 175:
        die(f"in-domain proceed at {gate} is {g['n_in_domain_proceed']}/{n_in}")
    if round(100 * float(g["ood_leakage_rate"]), 1) != 13.3:
        die(f"leakage at {gate} is {g['ood_leakage_rate']}")
    if round(100 * float(by[two_below]["ood_leakage_rate"]), 1) != 37.8:
        die(f"leakage at {two_below} is {by[two_below]['ood_leakage_rate']}")

    adv = read_csv(ADV_CSV)
    must_esc = [float(r["rag_similarity"]) for r in adv
                if r["expected_escalate"] == "True"]
    must_go = [float(r["rag_similarity"]) for r in adv
               if r["expected_escalate"] == "False"]
    lo, hi = max(must_esc), min(must_go)
    if (round(lo, 4), round(hi, 4)) != (0.6196, 0.6704):
        die(f"adversarial safe range is [{lo}, {hi}]")
    print(f"  [ok] F5: {n_in}/{n_in} proceed and "
          f"{100 * float(g['ood_leakage_rate']):.1f}% leak at {gate}; "
          f"{100 * float(by[two_below]['ood_leakage_rate']):.1f}% at "
          f"{two_below}; safe range [{lo:.4f}, {hi:.4f}]")

    fig, ax = plt.subplots(figsize=(WIDTH_IN, 2.35))
    ax.grid(axis="y", color=GRID, linewidth=0.5)
    ax.set_axisbelow(True)
    ax.axvspan(lo, hi, color=GREEN, alpha=0.18, linewidth=0,
               label="adversarial safe range")
    ax.plot(thr, proceed, color=BLUE, linewidth=1.5, marker="o",
            markersize=2.6, label="in-domain tickets that proceed")
    ax.plot(thr, leak, color=VERMILLION, linewidth=1.5, marker="s",
            markersize=2.6, label="out-of-domain leakage")
    ax.axvline(gate, color=INK, linewidth=0.9, linestyle="--")
    for x, dx, ha in ((gate, 5, "left"), (two_below, -5, "right")):
        y = float(by[x]["ood_leakage_rate"])
        ax.annotate(f"{100 * y:.1f}%", (x, y), xytext=(dx, 2),
                    textcoords="offset points", fontsize=NOTE_PT, color=INK,
                    ha=ha)
    ax.text(gate + 0.006, 0.30, f"live {gate:.2f}", fontsize=NOTE_PT,
            color=INK)
    ax.set_xlim(min(thr), max(thr))
    ax.set_ylim(0, 1.03)
    ax.set_xlabel("Top-1 cosine similarity threshold")
    ax.set_ylabel("Proportion")
    ax.legend(loc="lower left", bbox_to_anchor=(0.0, 0.04), frameon=False,
              handlelength=1.6)
    fig.tight_layout(pad=0.3)
    return fig


def fig6(plt):
    import numpy as np
    with open(CM_CSV, newline="", encoding="utf-8") as fh:
        rdr = list(csv.reader(fh))
    cats = rdr[0][1:]
    if [r[0] for r in rdr[1:]] != cats:
        die("confusion matrix rows and columns are not in the same order")
    m = np.array([[int(x) for x in r[1:]] for r in rdr[1:]])
    if m.sum() != 45:
        die(f"confusion matrix sums to {m.sum()}, not 45")

    fig, ax = plt.subplots(figsize=(WIDTH_IN, 3.0))
    ax.imshow(m, cmap="Blues", vmin=0, vmax=m.max())
    for i in range(len(cats)):
        for j in range(len(cats)):
            if m[i, j]:
                dark = m[i, j] > 0.55 * m.max()
                ax.text(j, i, str(m[i, j]), ha="center", va="center",
                        fontsize=TICK_PT,
                        color="white" if dark else INK,
                        fontweight="bold" if i == j else "normal")
    ax.set_xticks(range(len(cats)))
    ax.set_yticks(range(len(cats)))
    ax.set_xticklabels(cats, rotation=40, ha="right")
    ax.set_yticklabels(cats)
    ax.set_xlabel("Predicted category")
    ax.set_ylabel("True category")
    ax.tick_params(length=0)
    for s in ax.spines.values():
        s.set_visible(False)
    ax.set_xticks(np.arange(-0.5, len(cats)), minor=True)
    ax.set_yticks(np.arange(-0.5, len(cats)), minor=True)
    ax.grid(which="minor", color="white", linewidth=1.5)
    ax.tick_params(which="minor", length=0)
    fig.tight_layout(pad=0.3)
    return fig


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args()
    existing = [p for p in OUTS.values() if os.path.exists(p)]
    if existing and not args.force:
        die("Figures exist (use --force to overwrite):\n    "
            + "\n    ".join(existing))

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from src.agent.config import settings

    style(plt)
    os.makedirs(FIG_DIR, exist_ok=True)
    for key, fig in (("f4", fig4(plt, settings)), ("f5", fig5(plt, settings)),
                     ("f6", fig6(plt))):
        fig.savefig(OUTS[key], dpi=FIG_DPI, metadata={"Software": None})
        plt.close(fig)
        print(f"[write] {os.path.relpath(OUTS[key], PROJECT_ROOT)}")


if __name__ == "__main__":
    main()
