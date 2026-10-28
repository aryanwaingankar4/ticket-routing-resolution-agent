"""Figures for the B.Tech project report (black book).

Offline and model-free: 0 Gemini calls, 0 Ollama calls, no model load. Every
plotted value is read from a committed result or data file; the diagrams carry
only names and the three live thresholds, which are read from
src/agent/config.py. Writes PNGs (200 dpi, serif, Okabe-Ito colours, the same
palette as scripts/make_revision_figures.py) to reports/black_book/fig/ and a
small profile of the corpus to reports/black_book/data/dataset_profile.csv.

    .\\venv\\Scripts\\python.exe scripts/make_black_book_figures.py

Re-running overwrites the outputs; they are derived files.
"""

from __future__ import annotations

import csv
import os
import sys

PROJECT_ROOT = os.path.abspath(
    os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

import matplotlib  # noqa: E402
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch  # noqa: E402

from src.agent.config import settings  # noqa: E402
from src.experiments.score_groundedness_set import wilson_interval  # noqa: E402

DATA = os.path.join(PROJECT_ROOT, "data")
OUT = os.path.join(PROJECT_ROOT, "reports", "black_book", "fig")
OUT_DATA = os.path.join(PROJECT_ROOT, "reports", "black_book", "data")

BLUE, VERMILLION, GREEN, ORANGE = "#0072B2", "#D55E00", "#009E73", "#E69F00"
SKY, PURPLE = "#56B4E9", "#CC79A7"
INK, MUTED, GRID = "#222222", "#666666", "#DDDDDD"
FILL_A, FILL_B, FILL_C, FILL_D = "#E8F1F8", "#FDF0E6", "#E6F5F0", "#F2F2F2"
DPI = 200

CASCADE = settings.cascade.confidence_threshold
RAG = settings.rag.similarity_threshold
CLUSTER = settings.clustering.resolution_similarity_threshold
CATS = ["Access Management", "Application", "Database", "Infrastructure",
        "Network", "Security", "Storage"]


def die(msg):
    print(f"\n[FATAL] {msg}")
    raise SystemExit(1)


def read_csv(path):
    with open(path, newline="", encoding="utf-8") as fh:
        return list(csv.DictReader(fh))


def kv(path):
    return {r["quantity"]: r["value"] for r in read_csv(path)}


def style():
    plt.rcParams.update({
        "font.family": "serif",
        "font.serif": ["Times New Roman", "DejaVu Serif"],
        "font.size": 10, "axes.labelsize": 10, "axes.titlesize": 10,
        "xtick.labelsize": 9, "ytick.labelsize": 9, "legend.fontsize": 9,
        "axes.edgecolor": MUTED, "axes.labelcolor": INK,
        "xtick.color": MUTED, "ytick.color": MUTED, "text.color": INK,
        "axes.spines.top": False, "axes.spines.right": False,
        "savefig.dpi": DPI, "mathtext.fontset": "stix",
    })


def save(fig, name):
    path = os.path.join(OUT, name)
    fig.savefig(path, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"  wrote {os.path.relpath(path, PROJECT_ROOT)}")


# --------------------------------------------------------------------------- #
# Diagram helpers                                                              #
# --------------------------------------------------------------------------- #
def canvas(w, h):
    fig, ax = plt.subplots(figsize=(w, h))
    ax.set_xlim(0, w)
    ax.set_ylim(0, h)
    ax.axis("off")
    return fig, ax


def box(ax, x, y, w, h, text, fill=FILL_A, edge=BLUE, size=9, bold=False,
        round_=True, lw=1.2):
    style_ = "round,pad=0.02,rounding_size=0.08" if round_ else "square,pad=0.02"
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle=style_, fc=fill,
                                ec=edge, lw=lw))
    ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=size,
            fontweight="bold" if bold else "normal", wrap=True)


def arrow(ax, x1, y1, x2, y2, text="", color=INK, size=8, dashed=False,
          offset=(0, 0.12), rad=0.0):
    ax.add_patch(FancyArrowPatch((x1, y1), (x2, y2), arrowstyle="-|>",
                                 mutation_scale=11, color=color, lw=1.1,
                                 linestyle="--" if dashed else "-",
                                 connectionstyle=f"arc3,rad={rad}"))
    if text:
        ax.text((x1 + x2) / 2 + offset[0], (y1 + y2) / 2 + offset[1], text,
                ha="center", va="bottom", fontsize=size, color=color)


def diamond(ax, cx, cy, w, h, text, size=8.5):
    xs = [cx, cx + w / 2, cx, cx - w / 2, cx]
    ys = [cy + h / 2, cy, cy - h / 2, cy, cy + h / 2]
    ax.fill(xs, ys, fc=FILL_B, ec=ORANGE, lw=1.2)
    ax.text(cx, cy, text, ha="center", va="center", fontsize=size)


# --------------------------------------------------------------------------- #
# Chapter 1                                                                    #
# --------------------------------------------------------------------------- #
def fig_ticket_lifecycle():
    fig, ax = canvas(9.0, 3.4)
    box(ax, 0.1, 1.35, 1.4, 0.8, "Employee\nraises ticket", FILL_D, MUTED)
    box(ax, 2.0, 1.35, 1.6, 0.8, "Service desk\n(triage)", FILL_A, BLUE, bold=True)
    box(ax, 4.3, 2.35, 1.7, 0.75, "Wrong team\n(misrouted)", FILL_B, VERMILLION)
    box(ax, 4.3, 0.35, 1.7, 0.75, "Correct team", FILL_C, GREEN)
    box(ax, 6.7, 2.35, 2.1, 0.75, "Reassignment:\nwaiting, re-reading", FILL_B,
        VERMILLION)
    box(ax, 6.7, 0.35, 2.1, 0.75, "Diagnosis and\nresolution", FILL_C, GREEN)
    arrow(ax, 1.5, 1.75, 2.0, 1.75)
    arrow(ax, 3.6, 1.95, 4.3, 2.7, "confident\nbut wrong", VERMILLION, offset=(-0.25, 0.0))
    arrow(ax, 3.6, 1.55, 4.3, 0.75, "right first time", GREEN, offset=(-0.45, -0.32))
    arrow(ax, 6.0, 2.72, 6.7, 2.72)
    arrow(ax, 7.75, 2.35, 7.75, 1.1, "transfer", VERMILLION, offset=(0.35, -0.05))
    arrow(ax, 6.0, 0.72, 6.7, 0.72)
    ax.text(4.5, 3.25, "Every transfer adds a queue wait before any work starts",
            ha="center", fontsize=9, color=MUTED, style="italic")
    save(fig, "f_ticket_lifecycle.png")


# --------------------------------------------------------------------------- #
# Chapter 2                                                                    #
# --------------------------------------------------------------------------- #
def fig_literature_map():
    fig, ax = canvas(9.0, 4.6)
    box(ax, 3.35, 1.95, 2.3, 0.85, "This project:\ncalibrated escalation\nfor ticket triage",
        FILL_A, BLUE, size=9, bold=True)
    areas = [
        (0.1, 3.55, "Ticket triage and ITSM\nrouting, classification"),
        (3.35, 3.65, "Text representations\nTF-IDF, BERT, sentence\nembeddings, BGE"),
        (6.6, 3.55, "Cascades, deferral and\nselective prediction"),
        (0.1, 0.25, "Conformal prediction\nand distribution shift"),
        (3.35, 0.1, "RAG and abstention\nsufficient context, CRAG"),
        (6.6, 0.25, "LLM-as-judge and\nsynthetic benchmarks"),
    ]
    for x, y, t in areas:
        box(ax, x, y, 2.3, 0.85, t, FILL_D, MUTED, size=8.5)
    cx, cy = 4.5, 2.375
    for x, y, _ in areas:
        ax.plot([x + 1.15, cx], [y + 0.425, cy], color=GRID, lw=1.4, zorder=0)
    save(fig, "f_literature_map.png")


# --------------------------------------------------------------------------- #
# Chapter 3                                                                    #
# --------------------------------------------------------------------------- #
WBS = [
    ("1 Data", ["1.1 Synthetic corpus", "1.2 Fixed benchmarks",
                "1.3 Calibration sets", "1.4 Independent and\nreal-world sets"]),
    ("2 Models", ["2.1 Tier 1 (TF-IDF)", "2.2 Tier 2 (BGE)",
                  "2.3 Baselines", "2.4 Cascade threshold"]),
    ("3 Retrieval and\ndrafting", ["3.1 FAISS index", "3.2 Retrieval gate",
                                   "3.3 Grounded prompt", "3.4 Groundedness study"]),
    ("4 Engineering", ["4.1 Agents and\norchestrator", "4.2 HTTP service",
                       "4.3 Streamlit demo", "4.4 Docker and CI"]),
    ("5 Evaluation", ["5.1 Ablation, latency", "5.2 Zero-shot baselines",
                      "5.3 Conformal, drift", "5.4 Error analysis"]),
    ("6 Write-up", ["6.1 Lab notebook", "6.2 IEEE paper",
                    "6.3 Project report", "6.4 Reproduction kit"]),
]


def fig_wbs():
    fig, ax = canvas(10.0, 5.2)
    box(ax, 3.4, 4.45, 3.2, 0.62, "Calibrated escalation for IT ticket triage",
        FILL_A, BLUE, size=9.5, bold=True)
    n = len(WBS)
    colw = 10.0 / n
    ax.plot([colw / 2, 10.0 - colw / 2], [4.15, 4.15], color=MUTED, lw=1)
    ax.plot([5.0, 5.0], [4.45, 4.15], color=MUTED, lw=1)
    for i, (head, kids) in enumerate(WBS):
        x = i * colw + 0.08
        w = colw - 0.16
        ax.plot([x + w / 2, x + w / 2], [4.15, 3.85], color=MUTED, lw=1)
        box(ax, x, 3.25, w, 0.6, head, FILL_B, ORANGE, size=8.5, bold=True)
        for j, k in enumerate(kids):
            y = 2.45 - j * 0.72
            box(ax, x + 0.08, y, w - 0.16, 0.6, k, FILL_D, MUTED, size=7.6)
        ax.plot([x + 0.04, x + 0.04], [3.25, 2.45 - 3 * 0.72 + 0.3], color=GRID,
                lw=1, zorder=0)
    save(fig, "f_wbs.png")


# Phase windows taken from `git log --date=short` on this repository. A phase
# runs from its first commit to its last; the dates are the commit dates.
PHASES = [
    ("Prototype (dataset, models, RAG, demo)", "2026-07-28", "2026-08-15"),
    ("BGE migration, 0.67 recalibration", "2026-08-26", "2026-09-01"),
    ("Phase 0: pipeline consolidation", "2026-09-13", "2026-09-13"),
    ("Phases 1-2: conformal, flags, groundedness", "2026-09-14", "2026-09-14"),
    ("Phase 3: Tier-1 artifact, agents, API", "2026-09-14", "2026-09-15"),
    ("Phase 4: decision log, drift evaluation", "2026-09-15", "2026-09-20"),
    ("Phase 5: corrections, ablation, zero-shot", "2026-09-20", "2026-09-21"),
    ("Phase 6: deferral, weighted conformal, 6C", "2026-09-21", "2026-09-22"),
    ("Phase 7: external validity", "2026-09-22", "2026-09-22"),
    ("Phase 8: paper artifacts, Docker, CI", "2026-09-22", "2026-09-23"),
    ("Phase 9: paper draft, audit, v1.0", "2026-09-23", "2026-09-28"),
    ("Paper revision (independent, real-world)", "2026-10-19", "2026-10-23"),
    ("Project report", "2026-10-25", "2026-10-25"),
]


def fig_timeline():
    import datetime as dt
    d = lambda s: dt.date.fromisoformat(s)  # noqa: E731
    t0 = d(PHASES[0][1])
    fig, ax = plt.subplots(figsize=(9.0, 4.4))
    for i, (name, a, b) in enumerate(PHASES):
        start = (d(a) - t0).days
        dur = (d(b) - d(a)).days + 1
        ax.barh(i, dur, left=start, height=0.55, color=BLUE if i < 11 else ORANGE)
    ax.set_yticks(range(len(PHASES)))
    ax.set_yticklabels([p[0] for p in PHASES])
    ax.invert_yaxis()
    ticks = ["2026-07-28", "2026-08-15", "2026-09-01", "2026-09-14",
             "2026-09-28", "2026-10-12", "2026-10-25"]
    ax.set_xticks([(d(t) - t0).days for t in ticks])
    ax.set_xticklabels([d(t).strftime("%d %b") for t in ticks])
    ax.grid(axis="x", color=GRID, lw=0.8)
    ax.set_axisbelow(True)
    ax.set_xlabel("Commit date (from git log)")
    save(fig, "f_timeline.png")


# --------------------------------------------------------------------------- #
# Chapter 4: dataset                                                           #
# --------------------------------------------------------------------------- #
def dataset_profile():
    import pandas as pd
    df = pd.read_csv(os.path.join(DATA, "synthetic_tickets.csv"))
    if len(df) != 4000 or df.category.nunique() != 7:
        die(f"corpus is {len(df)} rows / {df.category.nunique()} categories")
    df["n_words"] = (df.title + " " + df.description).str.split().str.len()
    templates = df.groupby("category").scenario_id.nunique()
    if int(templates.sum()) != 66:
        die(f"expected 66 (category, scenario_id) templates, got {templates.sum()}")
    rows = [("rows", len(df)), ("categories", df.category.nunique()),
            ("templates", int(templates.sum())),
            ("distinct_titles", df.title.nunique()),
            ("distinct_descriptions", df.description.nunique()),
            ("distinct_resolutions", df.resolution.nunique()),
            ("words_mean", round(df.n_words.mean(), 1)),
            ("words_median", int(df.n_words.median())),
            ("words_min", int(df.n_words.min())),
            ("words_max", int(df.n_words.max()))]
    rows += [(f"category.{c}", int(n)) for c, n in df.category.value_counts().sort_index().items()]
    rows += [(f"priority.{p}", int(n)) for p, n in df.priority.value_counts().items()]
    rows += [(f"templates.{c}", int(n)) for c, n in templates.items()]
    for c in CATS:
        sub = df[df.category == c].priority.value_counts()
        rows += [(f"priority_by_category.{c}.{p}", int(sub.get(p, 0)))
                 for p in ["High", "Medium", "Low"]]
    os.makedirs(OUT_DATA, exist_ok=True)
    with open(os.path.join(OUT_DATA, "dataset_profile.csv"), "w", newline="",
              encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(["quantity", "value"])
        w.writerows(rows)
    print("  wrote reports/black_book/data/dataset_profile.csv")
    return df


def fig_dataset_distribution(df):
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(9.0, 3.4),
                                 gridspec_kw={"width_ratios": [1.4, 1]})
    counts = df.category.value_counts().reindex(CATS)
    a1.barh(range(7), counts.values, color=BLUE, height=0.6)
    a1.set_yticks(range(7))
    a1.set_yticklabels(CATS)
    a1.invert_yaxis()
    for i, v in enumerate(counts.values):
        a1.text(v + 8, i, str(v), va="center", fontsize=8.5)
    a1.set_xlim(0, 700)
    a1.set_xlabel("Tickets")
    a1.set_title("(a) Tickets per category", loc="left")
    pri = ["High", "Medium", "Low"]
    shades = ["#08519C", "#4292C6", "#9ECAE1"]
    left = [0] * 7
    for p, col in zip(pri, shades):
        vals = [int((df[(df.category == c)].priority == p).sum()) for c in CATS]
        a2.barh(range(7), vals, left=left, color=col, height=0.6, label=p,
                edgecolor="white", linewidth=1)
        left = [l + v for l, v in zip(left, vals)]
    a2.set_yticks(range(7))
    a2.set_yticklabels([])
    a2.invert_yaxis()
    a2.set_xlabel("Tickets")
    a2.set_title("(b) Priority mix per category", loc="left")
    a2.legend(frameon=False, ncol=3, loc="lower center", bbox_to_anchor=(0.5, -0.38))
    save(fig, "f_dataset_distribution.png")


def fig_ticket_length(df):
    fig, ax = plt.subplots(figsize=(6.5, 2.8))
    lo, hi = int(df.n_words.min()), int(df.n_words.max())
    ax.hist(df.n_words, bins=range(lo, hi + 2), color=BLUE, edgecolor="white")
    ax.axvline(df.n_words.median(), color=VERMILLION, ls="--", lw=1.2)
    ax.text(df.n_words.median() + 4.2, ax.get_ylim()[1] * 0.9,
            f"median {int(df.n_words.median())} words", color=VERMILLION, fontsize=9)
    ax.set_xlabel("Words in title + description")
    ax.set_ylabel("Tickets")
    ax.grid(axis="y", color=GRID, lw=0.8)
    ax.set_axisbelow(True)
    save(fig, "f_ticket_length.png")


def fig_dataset_flow():
    fig, ax = canvas(9.0, 3.0)
    box(ax, 0.1, 1.1, 1.75, 0.9, "66 linked templates\n(title, symptom,\nresolution)", FILL_D, MUTED, size=8.5)
    box(ax, 2.3, 1.1, 1.75, 0.9, "Placeholder context\nresolved once per\nticket (seed 42)", FILL_A, BLUE, size=8.5)
    box(ax, 4.5, 1.1, 1.75, 0.9, "Priority drawn from a\nper-category\ndistribution", FILL_A, BLUE, size=8.5)
    box(ax, 6.7, 1.1, 2.2, 0.9, "Validation gates\n(balance, no empty field,\nduplicate check)", FILL_B, ORANGE, size=8.5)
    for x in (1.85, 4.05, 6.25):
        arrow(ax, x, 1.55, x + 0.45, 1.55)
    ax.text(7.8, 0.55, "data/synthetic_tickets.csv: 4,000 rows", ha="center",
            fontsize=9, fontweight="bold")
    arrow(ax, 7.8, 1.1, 7.8, 0.75)
    ax.text(4.5, 2.55, "Offline generator (data/generate_dataset.py), no network calls",
            ha="center", fontsize=9, color=MUTED, style="italic")
    save(fig, "f_dataset_flow.png")


# --------------------------------------------------------------------------- #
# Chapter 4: architecture                                                      #
# --------------------------------------------------------------------------- #
def fig_architecture():
    fig, ax = canvas(9.5, 6.7)
    layers = [
        (5.35, "Presentation", FILL_D, MUTED,
         ["Streamlit demo", "FastAPI service\n/triage, /agents/*", "Batch and\nexperiment scripts"]),
        (3.85, "Orchestration", FILL_A, BLUE,
         ["Orchestrator: both gates,\nstep trace, escalation reasons", "Pipeline facade\npipeline.run()"]),
        (2.35, "Agents (decide nothing)", FILL_C, GREEN,
         ["Classification agent\nTier 1 -> Tier 2", "Retrieval agent\nFAISS top-5", "Resolution agent\nGemini drafting"]),
        (0.85, "Artifacts and configuration", FILL_B, ORANGE,
         ["Tier-1 bundle\n+ manifest", "BGE encoder\n+ Tier-2 model", "FAISS index\n+ metadata", "Frozen config\n(thresholds)"]),
    ]
    for y, name, fill, edge, items in layers:
        ax.add_patch(FancyBboxPatch((0.05, y - 0.1), 9.4, 1.2,
                                    boxstyle="round,pad=0.02,rounding_size=0.1",
                                    fc="white", ec=edge, lw=1, ls="--"))
        ax.text(0.2, y + 0.95, name, fontsize=9, fontweight="bold", color=edge)
        n = len(items)
        w = (9.0 - 0.3 * (n - 1)) / n
        for i, t in enumerate(items):
            box(ax, 0.25 + i * (w + 0.3), y, w, 0.8, t, fill, edge, size=8.5)
    for y in (5.25, 3.75, 2.25):
        arrow(ax, 4.75, y, 4.75, y - 0.4)
    ax.text(4.75, 0.25, f"Live thresholds: cascade {CASCADE:.2f} | retrieval {RAG:.2f} | "
            f"clustering {CLUSTER:.2f} (offline job)", ha="center", fontsize=8.5, color=MUTED)
    save(fig, "f_architecture.png")


def fig_agents():
    fig, ax = canvas(9.0, 4.2)
    box(ax, 3.0, 3.1, 3.0, 0.8, "Orchestrator\n(owns every routing decision)", FILL_A, BLUE, bold=True)
    agents = [(0.2, "ClassificationAgent\nrequires: tier1, encoder,\ntier2 classifier"),
              (3.2, "RetrievalAgent\nrequires: encoder,\nindex, metadata"),
              (6.2, "ResolutionAgent\nrequires: gemini client\n(only if drafting)")]
    for x, t in agents:
        box(ax, x, 1.35, 2.6, 1.0, t, FILL_C, GREEN, size=8.5)
        arrow(ax, 4.5, 3.1, x + 1.3, 2.35)
    box(ax, 1.2, 0.1, 6.6, 0.75, "Artifacts (loaded once by artifacts.load_artifacts(), three hard guards)",
        FILL_B, ORANGE, size=8.5)
    for x, _ in agents:
        arrow(ax, x + 1.3, 1.35, x + 1.3, 0.85, dashed=True, color=MUTED)
    box(ax, 7.0, 3.15, 1.9, 0.7, "StepTrace per agent\n(incl. skipped)", FILL_D, MUTED, size=8)
    arrow(ax, 6.0, 3.5, 7.0, 3.5, color=MUTED)
    save(fig, "f_agents.png")


def fig_sequence():
    fig, ax = canvas(9.5, 6.6)
    parts = ["Client", "FastAPI\n/triage", "Orchestrator", "Classification\nagent",
             "Retrieval\nagent", "Resolution\nagent (Gemini)"]
    xs = [0.75 + i * 1.6 for i in range(len(parts))]
    for x, p in zip(xs, parts):
        box(ax, x - 0.68, 5.85, 1.36, 0.62, p, FILL_A, BLUE, size=8.5)
        ax.plot([x, x], [0.2, 5.85], color=GRID, lw=1.2, ls="--", zorder=0)
    msgs = [
        (0, 1, 5.45, "POST /triage {title, description}"),
        (1, 2, 5.05, "pipeline.run(ticket)"),
        (2, 3, 4.65, "classify"),
        (3, 2, 4.25, f"category, tier (Tier 1 kept if conf >= {CASCADE:.2f})"),
        (2, 4, 3.85, "retrieve top-5"),
        (4, 2, 3.45, "neighbours, top-1 similarity"),
    ]
    for a, b, y, t in msgs:
        arrow(ax, xs[a], y, xs[b], y, t, color=INK if a < b else MUTED, size=7.8,
              dashed=a > b, offset=(0, 0.04))
    ax.add_patch(FancyBboxPatch((xs[2] - 0.72, 1.0), xs[5] - xs[2] + 1.4, 2.15,
                                boxstyle="square,pad=0", fc="none", ec=ORANGE, lw=1))
    ax.text(xs[2] - 0.7, 2.95, f"alt  [top-1 similarity >= {RAG:.2f} and drafting requested]",
            fontsize=8, color=ORANGE)
    arrow(ax, xs[2], 2.55, xs[5], 2.55, "draft resolution (prompt with 5 past tickets)", size=7.8, offset=(0, 0.04))
    arrow(ax, xs[5], 2.15, xs[2], 2.15, "grounded suggestion", color=MUTED, dashed=True, size=7.8, offset=(0, 0.04))
    ax.plot([xs[2] - 0.75, xs[5] + 0.75], [1.85, 1.85], color=ORANGE, lw=0.8, ls=":")
    ax.text(xs[2] - 0.7, 1.6, f"[else: similarity < {RAG:.2f}]  escalate to a human; Gemini is never called",
            fontsize=8, color=ORANGE)
    ax.text(xs[2] - 0.7, 1.2, "resolution step recorded as 'skipped' in the trace", fontsize=8, color=MUTED)
    arrow(ax, xs[2], 0.7, xs[1], 0.7, "PipelineResult + trace", color=MUTED, dashed=True, size=7.8, offset=(0, 0.04))
    arrow(ax, xs[1], 0.3, xs[0], 0.3, "HTTP 200 TriageResponse", color=MUTED, dashed=True, size=7.8, offset=(0, 0.04))
    save(fig, "f_sequence.png")


def fig_cascade_flow():
    fig, ax = canvas(9.0, 3.3)
    box(ax, 0.1, 1.25, 1.3, 0.8, "Ticket text", FILL_D, MUTED)
    box(ax, 1.9, 1.25, 1.7, 0.8, "Tier 1\nTF-IDF + LogReg\n(~1 ms)", FILL_A, BLUE, size=8.5)
    diamond(ax, 4.75, 1.65, 1.9, 1.2, f"top-class\nprob >= {CASCADE:.2f}?")
    box(ax, 6.4, 2.25, 2.4, 0.75, "Accept Tier-1 label", FILL_C, GREEN, size=8.5)
    box(ax, 6.4, 0.3, 2.4, 0.85, "Tier 2\nBGE + LogReg (~156 ms)", FILL_A, BLUE, size=8.5)
    arrow(ax, 1.4, 1.65, 1.9, 1.65)
    arrow(ax, 3.6, 1.65, 3.8, 1.65)
    arrow(ax, 5.7, 1.85, 6.4, 2.6, "yes", GREEN, offset=(-0.15, 0.0))
    arrow(ax, 5.7, 1.45, 6.4, 0.75, "no", VERMILLION, offset=(-0.2, -0.3))
    ax.text(4.5, 3.05, "Either way, the category then goes to retrieval and the retrieval gate",
            ha="center", fontsize=8.5, color=MUTED, style="italic")
    save(fig, "f_cascade_flow.png")


def fig_conformal():
    fig, ax = canvas(9.0, 3.0)
    steps = [("Calibration set\n(n = 175 tickets\nwith labels)", FILL_D, MUTED),
             ("Score each ticket:\ns = 1 - p(true label)", FILL_A, BLUE),
             ("q = empirical quantile\nat level\nceil((n+1)(1-a))/n", FILL_A, BLUE),
             ("New ticket: keep every\nlabel with\n1 - p(label) <= q", FILL_C, GREEN),
             ("Prediction set;\ncoverage >= 1 - a\nif exchangeable", FILL_B, ORANGE)]
    w = 1.6
    for i, (t, f, e) in enumerate(steps):
        box(ax, 0.1 + i * 1.8, 0.9, w, 1.2, t, f, e, size=8.2)
        if i:
            arrow(ax, 0.1 + i * 1.8 - 0.2, 1.5, 0.1 + i * 1.8, 1.5)
    ax.text(4.5, 2.55, "Split conformal prediction, measured offline for each tier (a = target error rate)",
            ha="center", fontsize=9, color=MUTED, style="italic")
    save(fig, "f_conformal.png")


def fig_cicd():
    fig, ax = canvas(9.5, 3.8)
    box(ax, 0.1, 1.5, 1.4, 0.8, "git push /\npull request", FILL_D, MUTED)
    box(ax, 2.0, 2.4, 2.5, 0.9, "ci.yml\npytest -m \"not slow\"\n+ paper parity", FILL_A, BLUE, size=8.5)
    box(ax, 2.0, 0.45, 2.5, 0.9, "gates.yml (manual)\nfull suite, adversarial gate,\nablation CSV checks", FILL_A, BLUE, size=8.5)
    box(ax, 5.1, 0.45, 2.1, 0.9, "docker build\n(builds its own\nartifacts)", FILL_C, GREEN, size=8.5)
    box(ax, 7.6, 0.45, 1.8, 0.9, "verify_deployment.py\nover HTTP", FILL_B, ORANGE, size=8.5)
    box(ax, 5.1, 2.4, 2.1, 0.9, "No step can spend\nGemini quota", FILL_D, MUTED, size=8.5)
    arrow(ax, 1.5, 2.0, 2.0, 2.8)
    arrow(ax, 1.5, 1.8, 2.0, 0.9)
    ax.text(1.0, 0.95, "manual\ntrigger", fontsize=7.5, ha="center", color=MUTED)
    arrow(ax, 4.5, 0.9, 5.1, 0.9)
    arrow(ax, 7.2, 0.9, 7.6, 0.9)
    ax.plot([4.5, 5.1], [2.85, 2.85], color=GRID, lw=1)
    save(fig, "f_cicd.png")


def fig_extended_design():
    fig, ax = canvas(9.5, 4.6)
    ax.text(2.2, 4.35, "(a) Independent author-written set", ha="center", fontsize=9.5, fontweight="bold")
    left = [(3.45, "Two authors write 54 tickets each\n(7 per team + 5 off-topic/vague)"),
            (2.45, "Each labels the OTHER's tickets blind:\ncategory and expected outcome"),
            (1.45, "Cohen's kappa on first labels;\ndisagreements adjudicated"),
            (0.45, "Unchanged pipeline routes all 108")]
    for y, t in left:
        box(ax, 0.2, y, 4.0, 0.7, t, FILL_A, BLUE, size=8.3)
    for y in (3.45, 2.45, 1.45):
        arrow(ax, 2.2, y, 2.2, y - 0.3)
    ax.text(7.1, 4.35, "(b) Real-world tickets (Zenodo, CC BY 4.0)", ha="center", fontsize=9.5, fontweight="bold")
    funnel = [("2,229 tickets, 7 source categories", 4.2),
              ("1,734 English (langid.py)", 3.6),
              ("984 in the 4 mapped categories", 3.0),
              ("583 distinct texts (headline)", 2.4)]
    for i, (t, w) in enumerate(funnel):
        y = 3.45 - i * 1.0
        box(ax, 7.1 - w / 2, y, w, 0.7, t, FILL_B if i < 3 else FILL_C,
            ORANGE if i < 3 else GREEN, size=8.3)
        if i:
            arrow(ax, 7.1, y + 1.0, 7.1, y + 0.7)
    save(fig, "f_extended_design.png")


# --------------------------------------------------------------------------- #
# Charts from result files                                                     #
# --------------------------------------------------------------------------- #
def fig_adversarial():
    rows = read_csv(os.path.join(DATA, "adversarial_escalation_results.csv"))
    if len(rows) != 9 or not all(r["pass_fail"] == "PASS" for r in rows):
        die("adversarial results are not 9/9 PASS")
    fig, ax = plt.subplots(figsize=(8.0, 3.6))
    rows = sorted(rows, key=lambda r: float(r["rag_similarity"]))
    must = [float(r["rag_similarity"]) for r in rows if r["expected_escalate"] == "True"]
    ok = [float(r["rag_similarity"]) for r in rows if r["expected_escalate"] == "False"]
    lo, hi = max(must), min(ok)
    ax.axvspan(lo, hi, color=GREEN, alpha=0.15, lw=0)
    ax.axvline(RAG, color=INK, ls="--", lw=1.1)
    ax.text(RAG + 0.002, 8.6, f"gate {RAG:.2f}", fontsize=8.5)
    for i, r in enumerate(rows):
        s = float(r["rag_similarity"])
        esc = r["expected_escalate"] == "True"
        ax.scatter(s, i, s=50, color=VERMILLION if esc else BLUE, zorder=3,
                   marker="o" if esc else "s", edgecolor="white", linewidth=1.5)
    labels = [f"{r['id']}  " + (r["text"] if len(r["text"]) < 44 else r["text"][:41] + "...")
              for r in rows]
    ax.set_xlim(0.50, 0.71)
    ax.set_xticks([0.52, 0.56, 0.60, 0.64, 0.68])
    ax.set_yticks(range(len(rows)))
    ax.set_yticklabels(labels, fontsize=7.8)
    ax.tick_params(axis="y", length=0)
    ax.spines["left"].set_visible(False)
    ax.set_xlabel("Top-1 cosine similarity to a past ticket")
    ax.scatter([], [], color=VERMILLION, marker="o", label="must escalate")
    ax.scatter([], [], color=BLUE, marker="s", label="must proceed")
    ax.legend(frameon=False, loc="lower right")
    ax.text((lo + hi) / 2, -0.95, f"safe range [{lo:.4f}, {hi:.4f}]", ha="center",
            fontsize=8, color=GREEN)
    ax.set_ylim(-1.3, 8.9)
    save(fig, "f_adversarial.png")


def fig_latency():
    rows = read_csv(os.path.join(DATA, "inference_latency_13th-gen-intel-r-core-tm-i5-1334u.csv"))
    by = {(r["measurement"], r["eval_set"]): r for r in rows}
    t1 = float(by[("tier1", "")]["median_ms"])
    t2 = float(by[("tier2", "")]["median_ms"])
    b45 = float(by[("cascade_expected", "benchmark45")]["cascade_expected_median_ms"])
    d175 = float(by[("cascade_expected", "deployment175")]["cascade_expected_median_ms"])
    labels = ["Tier 1 alone", "Tier 2 alone", "Cascade,\nbenchmark (45)", "Cascade,\ndeployment (175)"]
    vals = [t1, t2, b45, d175]
    fig, ax = plt.subplots(figsize=(6.5, 3.0))
    ax.bar(range(4), vals, color=[GREEN, BLUE, ORANGE, ORANGE], width=0.55)
    for i, v in enumerate(vals):
        ax.text(i, v + 3, f"{v:.2f} ms" if v < 10 else f"{v:.1f} ms", ha="center", fontsize=9)
    ax.set_xticks(range(4))
    ax.set_xticklabels(labels)
    ax.set_ylabel("Median per-ticket latency (ms)")
    ax.set_ylim(0, 185)
    ax.grid(axis="y", color=GRID, lw=0.8)
    ax.set_axisbelow(True)
    save(fig, "f_latency.png")


def recall_bars(pairs, title_note, name, ymax_label):
    fig, ax = plt.subplots(figsize=(7.5, 3.0))
    ks = [p[1] for p in pairs]
    ns = [p[2] for p in pairs]
    rates = [k / n for k, n in zip(ks, ns)]
    ax.bar(range(len(pairs)), rates, color=BLUE, width=0.55)
    for i, (k, n) in enumerate(zip(ks, ns)):
        ax.text(i, k / n + 0.02, f"{k}/{n}", ha="center", fontsize=8.5)
    ax.set_xticks(range(len(pairs)))
    ax.set_xticklabels([p[0].replace(" ", "\n") for p in pairs])
    ax.set_ylim(0, 1.12)
    ax.set_ylabel(ymax_label)
    ax.grid(axis="y", color=GRID, lw=0.8)
    ax.set_axisbelow(True)
    save(fig, name)


def fig_benchmark_recall():
    s = kv(os.path.join(DATA, "paper_revision_oct", "error_analysis_benchmark45_summary.csv"))
    pairs = []
    for c in CATS:
        k, n = s[f"recall.{c}"].split("/")
        pairs.append((c, int(k), int(n)))
    if sum(p[1] for p in pairs) != 32 or sum(p[2] for p in pairs) != 45:
        die("benchmark recall does not sum to 32/45")
    recall_bars(pairs, "", "f_benchmark_recall.png", "Recall (cascade)")


def fig_independent_recall():
    s = kv(os.path.join(DATA, "independent_test", "results", "independent_summary.csv"))
    pairs = []
    for c in CATS:
        k, n = s[f"with_category.recall.cascade.{c}"].split("/")
        pairs.append((c, int(k), int(n)))
    if sum(p[1] for p in pairs) != 74 or sum(p[2] for p in pairs) != 98:
        die("independent recall does not sum to 74/98")
    recall_bars(pairs, "", "f_independent_recall.png", "Recall (cascade)")


def fig_accuracy_by_set():
    ind = kv(os.path.join(DATA, "independent_test", "results", "independent_summary.csv"))
    rw = kv(os.path.join(DATA, "external_zenodo_7648117", "realworld_summary.csv"))
    rws = kv(os.path.join(DATA, "external_zenodo_7648117", "realworld_summary_sens_fileservice_storage.csv"))
    bench = [r for r in read_csv(os.path.join(DATA, "ablation_baseline_results.csv"))
             if r["section"] == "classification"]
    dep = read_csv(os.path.join(DATA, "ablation_baseline_results_deployment175.csv"))
    sets = [("Out-of-template\nbenchmark", sum(r["correct"] == "True" for r in bench), len(bench)),
            ("Deployment-style\nset", sum(r["correct"] == "True" for r in dep), len(dep)),
            ("Independent\n(author-written)", int(ind["with_category.cascade_correct.k"]),
             int(ind["with_category.cascade_correct.n"])),
            ("Real-world,\nprimary mapping", int(rw["distinct.cascade_correct.k"]),
             int(rw["distinct.cascade_correct.n"])),
            ("Real-world,\nsensitivity mapping", int(rws["distinct.cascade_correct.k"]),
             int(rws["distinct.cascade_correct.n"]))]
    expected = [(32, 45), (131, 175), (74, 98), (115, 583), (294, 583)]
    if [(k, n) for _, k, n in sets] != expected:
        die(f"accuracy-by-set values moved: {sets}")
    fig, ax = plt.subplots(figsize=(8.0, 3.2))
    for i, (lab, k, n) in enumerate(sets):
        lo, hi = wilson_interval(k, n)
        ax.errorbar(i, k / n, yerr=[[k / n - lo], [hi - k / n]], fmt="o",
                    color=BLUE if i < 3 else VERMILLION, capsize=4, ms=7, lw=1.3)
        ax.text(i + 0.12, k / n, f"{k}/{n}", va="center", fontsize=8.5)
    ax.set_xticks(range(len(sets)))
    ax.set_xticklabels([s[0] for s in sets])
    ax.set_ylim(0, 1)
    ax.set_xlim(-0.4, len(sets) - 0.4)
    ax.set_ylabel("Cascade routing accuracy\n(Wilson 95% interval)")
    ax.grid(axis="y", color=GRID, lw=0.8)
    ax.set_axisbelow(True)
    save(fig, "f_accuracy_by_set.png")


def fig_realworld_confusion():
    rows = read_csv(os.path.join(DATA, "external_zenodo_7648117", "realworld_confusion.csv"))
    key = "true \\ predicted"
    mat = [[int(r[c]) for c in CATS] for r in rows]
    if sum(map(sum, mat)) != 984:
        die("real-world confusion does not sum to 984")
    fig, ax = plt.subplots(figsize=(8.0, 2.4))
    import numpy as np
    m = np.array(mat, dtype=float)
    ax.imshow(m / m.sum(axis=1, keepdims=True), cmap="Blues", vmin=0, vmax=1, aspect="auto")
    for i in range(m.shape[0]):
        for j in range(m.shape[1]):
            v = int(m[i, j])
            share = m[i, j] / m[i].sum()
            ax.text(j, i, str(v), ha="center", va="center", fontsize=9,
                    color="white" if share > 0.55 else INK)
    ax.set_xticks(range(7))
    ax.set_xticklabels([c.replace(" ", "\n") for c in CATS], fontsize=8.5)
    ax.set_yticks(range(len(rows)))
    ax.set_yticklabels([r[key] for r in rows])
    ax.set_xlabel("Predicted team (cascade)")
    ax.set_ylabel("Mapped true team")
    for s in ax.spines.values():
        s.set_visible(False)
    save(fig, "f_realworld_confusion.png")


# --------------------------------------------------------------------------- #
# Added 28 Oct 2026 for the condensed report (v2)                              #
# --------------------------------------------------------------------------- #
TAXONOMY = [
    ("Routing a ticket", [
        ("Rule-based routing", "Diao et al."),
        ("Classical ML (TF-IDF + LR/SVM)", "Salton & Buckley; Joachims"),
        ("Fine-tuned transformers", "BERT; DistilBERT; Paket et al."),
        ("Sentence embeddings", "Sentence-BERT; E5; BGE"),
        ("LLM-based triage", "TickIt; Carroll et al.; Qwen2.5"),
    ]),
    ("Supporting the resolution", [
        ("Retrieval-augmented generation", "Lewis et al."),
        ("RAG with abstention", "Sufficient context; CRAG"),
    ]),
    ("Deciding when to stop", [
        ("Cascades and deferral", "Viola & Jones; Jitkrittum; Fanconi; Conformal Cascade"),
        ("Selective classification, calibration", "Chow; Geifman & El-Yaniv; Guo et al."),
        ("Conformal prediction under shift", "Vovk; Tibshirani; Barber; Gibbs & Candes"),
        ("Drift detection", "Gama et al.; Rabanser et al."),
    ]),
    ("Evaluating the outputs", [
        ("LLM-as-judge", "Zheng; Jung; Jain"),
        ("Synthetic benchmarks", "Maheshwari et al."),
    ]),
]


def fig_taxonomy():
    n_leaves = sum(len(k) for _, k in TAXONOMY)
    h = 0.52 * n_leaves + 0.6
    fig, ax = canvas(10.0, h)
    y = h - 0.55
    centres = []
    for head, leaves in TAXONOMY:
        ys = []
        for name, works in leaves:
            box(ax, 4.55, y - 0.2, 2.75, 0.4, name, FILL_C, GREEN, size=8.2)
            ax.text(7.45, y, works, va="center", fontsize=7.8, color=MUTED, style="italic")
            ax.plot([4.2, 4.55], [y, y], color=MUTED, lw=0.9)
            ys.append(y)
            y -= 0.52
        mid = (ys[0] + ys[-1]) / 2
        ax.plot([4.2, 4.2], [ys[0], ys[-1]], color=MUTED, lw=0.9)
        box(ax, 2.0, mid - 0.24, 2.0, 0.48, head, FILL_B, ORANGE, size=8.6, bold=True)
        ax.plot([4.0, 4.2], [mid, mid], color=MUTED, lw=0.9)
        centres.append(mid)
    root_y = (centres[0] + centres[-1]) / 2
    box(ax, 0.05, root_y - 0.45, 1.6, 0.9, "Related work\nfor IT ticket\ntriage", FILL_A, BLUE,
        size=8.8, bold=True)
    ax.plot([1.8, 1.8], [centres[0], centres[-1]], color=MUTED, lw=0.9)
    ax.plot([1.65, 1.8], [root_y, root_y], color=MUTED, lw=0.9)
    for c in centres:
        ax.plot([1.8, 2.0], [c, c], color=MUTED, lw=0.9)
    save(fig, "f_taxonomy.png")


def layer(ax, y, h, name, edge):
    ax.add_patch(FancyBboxPatch((0.05, y), 7.35, h, boxstyle="round,pad=0.02,rounding_size=0.08",
                                fc="white", ec=edge, lw=1, ls="--"))
    ax.text(0.15, y + h - 0.2, name, fontsize=8.6, fontweight="bold", color=edge)


def row(ax, y, items, fill, edge, x0=0.2, x1=7.25, h=0.62, size=7.8):
    n = len(items)
    gap = 0.15
    w = (x1 - x0 - gap * (n - 1)) / n
    for i, t in enumerate(items):
        box(ax, x0 + i * (w + gap), y, w, h, t, fill, edge, size=size)


def fig_architecture_full():
    fig, ax = canvas(10.0, 8.6)
    # main stack, top to bottom
    layer(ax, 7.3, 1.2, "Presentation layer", MUTED)
    row(ax, 7.4, ["Streamlit demo\n(src/app)", "Interactive API docs\n(/docs, Swagger UI)",
                  "Batch and experiment\nscripts"], FILL_D, MUTED)
    layer(ax, 5.85, 1.3, "Service layer (FastAPI, src/service/api.py)", BLUE)
    row(ax, 5.95, ["GET\n/health", "POST\n/agents/classify", "POST\n/agents/retrieve",
                   "POST\n/agents/resolve", "POST\n/policy/rag-gate", "POST\n/triage"],
        FILL_A, BLUE, size=7.2)
    layer(ax, 3.55, 2.15, "Agent layer (src/agent)", GREEN)
    box(ax, 0.2, 4.75, 7.05, 0.6,
        f"Orchestrator: Gate 1 (Tier-1 confidence >= {CASCADE:.2f}), Gate 2 (top-1 similarity >= {RAG:.2f})\n"
        "records a step trace and the escalation reason", FILL_C, GREEN, size=7.8, bold=True)
    row(ax, 3.7, ["Classification agent\nTier 1 -> Tier 2", "Retrieval agent\nFAISS top-5",
                  "Resolution agent\ngrounded drafting"], FILL_C, GREEN)
    layer(ax, 2.05, 1.35, "Model layer", ORANGE)
    row(ax, 2.15, ["Tier 1: TF-IDF +\nlogistic regression", "BGE encoder\n(bge-base-en-v1.5)",
                   "Tier 2: logistic\nregression on BGE", "Gemini flash-lite\n(external API)"],
        FILL_B, ORANGE, size=7.6)
    layer(ax, 0.1, 1.4, "Data layer", PURPLE)
    row(ax, 0.25, ["Synthetic corpus\n4,000 tickets", "FAISS index +\naligned metadata",
                   "Resolved-ticket\nstore (by category)", "Frozen config +\nTier-1 manifest"],
        "#F7EAF2", PURPLE, h=0.75, size=7.6)
    for y in (7.3, 5.85, 3.55, 2.05):
        arrow(ax, 3.7, y, 3.7, y - 0.12)
    # offline path
    ax.add_patch(FancyBboxPatch((7.65, 4.3), 2.3, 3.2, boxstyle="round,pad=0.02,rounding_size=0.08",
                                fc="white", ec=SKY, lw=1, ls="--"))
    ax.text(7.75, 7.25, "Offline path", fontsize=8.6, fontweight="bold", color=SKY)
    for k, t in enumerate(["Resolved tickets", f"Cluster resolution\ntexts at {CLUSTER:.2f}",
                           "Automation\ncandidates", "Human review"]):
        box(ax, 7.85, 6.55 - k * 0.7, 1.9, 0.5, t, "#EAF5FC", SKY, size=7.6)
        if k:
            arrow(ax, 8.8, 6.55 - (k - 1) * 0.7, 8.8, 6.55 - k * 0.7 + 0.5)
    # devops
    ax.add_patch(FancyBboxPatch((7.65, 0.1), 2.3, 3.95, boxstyle="round,pad=0.02,rounding_size=0.08",
                                fc="white", ec=INK, lw=1, ls="--"))
    ax.text(7.75, 3.8, "DevOps layer", fontsize=8.6, fontweight="bold", color=INK)
    for k, t in enumerate(["pytest: 456 tests,\ngolden parity", "Adversarial gate\n(9/9 required)",
                           "Docker image builds\nits own artifacts", "GitHub Actions:\nci.yml, gates.yml"]):
        box(ax, 7.85, 2.95 - k * 0.85, 1.9, 0.65, t, FILL_D, INK, size=7.4)
    save(fig, "f_architecture_full.png")


def fig_workflow():
    fig, ax = canvas(7.4, 9.6)
    cx = 2.6
    box(ax, cx - 1.2, 8.85, 2.4, 0.55, "New ticket\n(title + description)", FILL_D, MUTED, size=8.2)
    box(ax, cx - 1.2, 7.85, 2.4, 0.55, "Tier 1: TF-IDF +\nlogistic regression", FILL_A, BLUE, size=8.2)
    diamond(ax, cx, 6.75, 2.6, 1.0, f"Tier-1 confidence\n>= {CASCADE:.2f}?", size=8)
    box(ax, 4.6, 6.48, 2.5, 0.55, "Tier 2: BGE embedding +\nlogistic regression", FILL_A, BLUE, size=8)
    box(ax, cx - 1.2, 5.45, 2.4, 0.55, "Category assigned\n(one of seven teams)", FILL_C, GREEN, size=8.2)
    box(ax, cx - 1.2, 4.45, 2.4, 0.55, "Retrieve 5 nearest\nresolved tickets (FAISS)", FILL_A, BLUE, size=8.2)
    diamond(ax, cx, 3.35, 2.6, 1.0, f"Top-1 similarity\n>= {RAG:.2f}?", size=8)
    box(ax, 4.6, 3.08, 2.5, 0.55, "Escalate to a human\n(no LLM call; step 'skipped')", FILL_B, VERMILLION, size=7.8)
    diamond(ax, cx, 2.0, 2.6, 0.95, "Drafting\nrequested?", size=8)
    box(ax, 4.6, 1.73, 2.5, 0.55, "Proceed without draft\n(category + neighbours)", FILL_D, MUTED, size=7.8)
    box(ax, cx - 1.2, 0.75, 2.4, 0.55, "Gemini drafts a resolution\nfrom the 5 past tickets", FILL_C, GREEN, size=7.8)
    box(ax, 4.6, 0.05, 2.5, 0.55, "Response: decision, reason,\nthreshold, value, trace", FILL_A, BLUE, size=7.8)
    arrow(ax, cx, 8.85, cx, 8.4)
    arrow(ax, cx, 7.85, cx, 7.25)
    arrow(ax, cx + 1.3, 6.75, 4.6, 6.75, "no", VERMILLION, offset=(0, 0.04))
    arrow(ax, cx, 6.25, cx, 6.0, "yes", GREEN, offset=(0.25, -0.12))
    arrow(ax, 5.85, 6.48, cx + 1.2, 5.75, rad=-0.2)
    arrow(ax, cx, 5.45, cx, 5.0)
    arrow(ax, cx, 4.45, cx, 3.85)
    arrow(ax, cx + 1.3, 3.35, 4.6, 3.35, "no", VERMILLION, offset=(0, 0.04))
    arrow(ax, cx, 2.85, cx, 2.48, "yes", GREEN, offset=(0.25, -0.1))
    arrow(ax, cx + 1.3, 2.0, 4.6, 2.0, "no", MUTED, offset=(0, 0.04))
    arrow(ax, cx, 1.52, cx, 1.3, "yes", GREEN, offset=(0.25, -0.12))
    arrow(ax, cx + 1.2, 0.95, 4.6, 0.4, rad=0.15)
    arrow(ax, 7.1, 3.3, 7.1, 0.38, color=MUTED, dashed=True, rad=-0.25)
    arrow(ax, 5.85, 1.73, 5.85, 0.6, color=MUTED, dashed=True)
    ax.text(0.05, 0.25, "A known agent failure at any step\nreturns an escalation (HTTP 200).",
            fontsize=7.6, color=MUTED, style="italic")
    save(fig, "f_workflow.png")


def fig_devworkflow():
    steps = [("1 Dataset", "template generator,\n4,000 tickets"),
             ("2 Classifiers", "TF-IDF, MiniLM, BGE,\nDistilBERT"),
             ("3 RAG", "FAISS retrieval,\ngrounded drafting"),
             ("4 Calibration", "cascade 0.50,\nretrieval 0.67"),
             ("5 API and agents", "orchestrator,\nFastAPI service"),
             ("6 Testing and CI", "456 tests, goldens,\nDocker, Actions"),
             ("7 Evaluation", "ablations, baselines,\nindependent, real"),
             ("8 Paper and report", "IEEE paper v4,\nthis report")]
    fig, ax = canvas(10.0, 3.6)
    w, h = 2.15, 1.1
    for i, (t, d) in enumerate(steps):
        r, c = divmod(i, 4)
        c = c if r == 0 else 3 - c
        x = 0.15 + c * 2.5
        y = 2.3 - r * 1.9
        box(ax, x, y, w, h, f"{t}\n{d}", FILL_A if i < 4 else FILL_C, BLUE if i < 4 else GREEN, size=8.2)
        if i in (0, 1, 2):
            arrow(ax, x + w, y + h / 2, x + 2.5, y + h / 2)
        if i == 3:
            arrow(ax, x + w / 2, y, x + w / 2, y - 0.8)
        if i in (4, 5, 6):
            arrow(ax, x, y + h / 2, x - 0.35, y + h / 2)
    ax.text(5.0, 1.65, "each phase: written plan -> review -> build -> report -> review gate",
            ha="center", fontsize=8.2, color=MUTED, style="italic")
    save(fig, "f_devworkflow.png")


def fig_eval_pipeline():
    fig, ax = canvas(10.0, 6.4)
    ax.text(1.4, 6.15, "Data", ha="center", fontsize=9.5, fontweight="bold")
    ax.text(5.0, 6.15, "Use", ha="center", fontsize=9.5, fontweight="bold")
    ax.text(8.55, 6.15, "Measured", ha="center", fontsize=9.5, fontweight="bold")
    src = [("Synthetic corpus, 4,000\n(train 3,200 / held-out 800)", 5.3, FILL_D, MUTED),
           ("Paraphrased calibration, 175", 4.45, FILL_B, ORANGE),
           ("Out-of-domain calibration, 45", 3.8, FILL_B, ORANGE),
           ("Deployment-style set, 175", 3.15, FILL_B, ORANGE),
           ("Benchmarks: 14 and 45\nout-of-template; adversarial 9", 2.3, FILL_C, GREEN),
           ("Independent set, 108\n(two authors, cross-labelled)", 1.3, FILL_C, GREEN),
           ("Zenodo real tickets:\n583 distinct (984 mapped)", 0.3, FILL_C, GREEN)]
    for t, y, f, e in src:
        hh = 0.75 if "\n" in t else 0.5
        box(ax, 0.05, y, 2.7, hh, t, f, e, size=7.8)
    box(ax, 3.6, 4.95, 2.8, 0.8, "Fit Tier 1 (all 4,000) and\nTier 2 (3,200); build index", FILL_D, MUTED, size=8)
    box(ax, 3.6, 3.35, 2.8, 1.0, "Calibrate thresholds:\ncascade 0.50, retrieval 0.67;\nconformal sets", FILL_B, ORANGE, size=8)
    box(ax, 3.6, 0.9, 2.8, 1.6, "Run the UNCHANGED pipeline\n(no refitting, no\nrecalibration)", FILL_C, GREEN, size=8, bold=True)
    for y in (5.65,):
        arrow(ax, 2.75, y, 3.6, 5.35)
    for y in (4.7, 4.05, 3.4):
        arrow(ax, 2.75, y, 3.6, 3.85)
    for y in (2.65, 1.65, 0.65):
        arrow(ax, 2.75, y, 3.6, 1.7)
    arrow(ax, 5.0, 4.95, 5.0, 4.35)
    arrow(ax, 5.0, 3.35, 5.0, 2.5)
    mets = ["Accuracy with Wilson 95% CI", "Exact McNemar (paired)", "Leakage / proceed rate",
            "Coverage gap vs noise band", "Escalation correctness", "Cohen's kappa", "Latency (warm median)"]
    for k, m in enumerate(mets):
        box(ax, 7.2, 5.25 - k * 0.78, 2.7, 0.55, m, FILL_A, BLUE, size=7.8)
    arrow(ax, 6.4, 3.85, 7.2, 3.85)
    arrow(ax, 6.4, 1.7, 7.2, 1.7)
    save(fig, "f_eval_pipeline.png")


def fig_demo_pair():
    from PIL import Image
    a = Image.open(os.path.join(OUT, "s_demo_escalated.png")).convert("RGB")
    b = Image.open(os.path.join(OUT, "s_demo_proceed.png")).convert("RGB")
    h = max(a.height, b.height)
    gap = 30
    out = Image.new("RGB", (a.width + b.width + gap, h), "white")
    out.paste(a, (0, 0))
    out.paste(b, (a.width + gap, 0))
    out.save(os.path.join(OUT, "f_demo_pair.png"))
    print("  wrote reports/black_book/fig/f_demo_pair.png")


def main():
    os.makedirs(OUT, exist_ok=True)
    style()
    print("Black-book figures:")
    fig_taxonomy()
    fig_architecture_full()
    fig_workflow()
    fig_devworkflow()
    fig_eval_pipeline()
    if os.path.exists(os.path.join(OUT, "s_demo_escalated.png")):
        fig_demo_pair()
    fig_ticket_lifecycle()
    fig_literature_map()
    fig_wbs()
    fig_timeline()
    df = dataset_profile()
    fig_dataset_distribution(df)
    fig_ticket_length(df)
    fig_dataset_flow()
    fig_architecture()
    fig_agents()
    fig_sequence()
    fig_cascade_flow()
    fig_conformal()
    fig_cicd()
    fig_extended_design()
    fig_adversarial()
    fig_latency()
    fig_benchmark_recall()
    fig_independent_recall()
    fig_accuracy_by_set()
    fig_realworld_confusion()
    print("Done.")


if __name__ == "__main__":
    main()
