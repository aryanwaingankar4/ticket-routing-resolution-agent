"""Phase 8A -- build EVERY paper table, figure and number from committed files.

WHY THIS SCRIPT EXISTS
======================
In Phase 7A two domain AUCs (0.8584, and the range 0.6706-0.9316) were computed
in an interactive session, quoted at a gate, and never committed. When Phase 7B
specified the recomputation they DID NOT REPRODUCE -- the re-derived values are
0.8727 raw / 0.8472 operating, and five variants chasing the recorded figure
span 0.8637-0.8727. The conclusions survived only because every value cleared
the same threshold, but the figures were wrong in print for a day.

That is this project's recurring bug class -- a value that is internally
consistent but wrong for its context -- reaching the WRITE-UP rather than the
code. Load guards answered it for artifacts, goldens answered it for routing,
and tests/test_contamination_structure.py answered it for grouping keys. This
script is the same answer for the paper:

    EVERY NUMBER THE PAPER USES IS REGENERATED FROM A COMMITTED RESULT FILE,
    THROUGH CODE. NOTHING IS RETYPED BY HAND.

The rule is enforced structurally, not by discipline. Every value that reaches
paper/NUMBERS.md passes through emit(), which requires an existing committed
source path, and tests/test_paper_artifacts.py runs an AST lint that FAILS if
any emit() call passes a numeric literal as its value.

WHAT IT DOES NOT DO
===================
It re-runs NOTHING. No Gemini call, no Ollama call, no model load, no training,
no experiment. It reads committed result files and writes only under paper/.
Because of that it cannot move a published number; the only thing it can reveal
is that a DOCUMENT disagreed with its source file, and in that case the
document is what gets fixed.

USAGE
=====
    python src/experiments/build_paper_artifacts.py              # refuses to overwrite
    python src/experiments/build_paper_artifacts.py --force      # rebuild in place
    python src/experiments/build_paper_artifacts.py --out DIR --no-render

--no-render skips PDF/PNG rasterisation and writes tables plus the data CSV
behind every figure. That is what the parity test uses: rendering is not what
carries a number.

Run from the project root -- src.experiments.* resolves as an implicit
namespace package, the same way every other script in this directory does.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import random
import re
import sys
import traceback
from dataclasses import dataclass, field

PROJECT_ROOT = os.path.abspath(
    os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

import numpy as np                                            # noqa: E402
import pandas as pd                                           # noqa: E402
from scipy.stats import binomtest                             # noqa: E402

from src.agent.config import settings, config_fingerprint     # noqa: E402

# ---------------------------------------------------------------------------
# --- presentation constants ---
#
# The ONLY numeric literals allowed to steer output in this file. Everything
# else must come from a source file. The AST lint in tests/test_paper_artifacts
# .py enforces that no emit() call passes a numeric literal as its value; this
# block is the explicit, reviewable exception list for geometry and style.
# ---------------------------------------------------------------------------
SEED = 42

IEEE_SINGLE_COL_IN = 3.5      # IEEE two-column: single-column figure width
IEEE_DOUBLE_COL_IN = 7.16     # IEEE two-column: full-width figure
FIG_DPI = 300
BASE_FONT_PT = 8
TICK_FONT_PT = 7
LEGEND_FONT_PT = 7

# z for a 95% interval. The repo contains TWO conventions -- 1.96 in
# score_groundedness_set.py and score_sufficiency_gate.py, and the exact
# normal quantile 1.959963984540054 in summarize_zeroshot_baselines.py. They
# agree to ~3.5e-6, so no published figure is affected at reported precision.
# The paper adopts 1.96, matching the majority and matching every CI already
# committed in a 2B/6C source file. See RECONCILIATION.md, "Convention notes".
Z_95 = 1.96
Z_95_EXACT = 1.959963984540054

# Okabe-Ito: colour-blind-safe qualitative palette. Paired with distinct dash
# patterns so every figure also survives greyscale printing.
OKABE_ITO = [
    "#000000",  # black
    "#E69F00",  # orange
    "#56B4E9",  # sky blue
    "#009E73",  # bluish green
    "#F0E442",  # yellow
    "#0072B2",  # blue
    "#D55E00",  # vermillion
    "#CC79A7",  # reddish purple
]
DASHES = ["-", "--", "-.", ":", (0, (3, 1, 1, 1)), (0, (5, 1)), (0, (1, 1))]
# Okabe-Ito's yellow is too pale to read as a line on white, so multi-series
# line charts draw from this ordered subset instead of the raw palette.
SERIES_COLORS = ["#0072B2", "#D55E00", "#009E73", "#CC79A7", "#E69F00",
                 "#56B4E9", "#000000"]
# --- end presentation constants ---


# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
DATA = os.path.join(PROJECT_ROOT, "data")
EXT = os.path.join(DATA, "external_tobibueck")
DEFAULT_OUT = os.path.join(PROJECT_ROOT, "paper")

DOCS_AUDITED = ["README.md", "PROJECT_STATUS.md", "CLAUDE.md",
                "PROJECT_OVERVIEW.md"]


def rel(path: str) -> str:
    """Repo-relative, forward-slashed -- the form that goes into NUMBERS.md."""
    return os.path.relpath(path, PROJECT_ROOT).replace(os.sep, "/")


def src(*parts: str) -> str:
    """A source path under data/, verified to exist. Missing is FATAL.

    A skipped table is exactly the silent failure this phase exists to stop,
    so there is deliberately no "if it exists" branch anywhere downstream.
    """
    path = os.path.join(DATA, *parts)
    if not os.path.isfile(path):
        raise FileNotFoundError(
            f"Required source file is missing: {rel(path)}\n"
            f"  Phase 8A reads committed results only and never regenerates "
            f"them.\n"
            f"  Re-run the experiment that produces it, then rebuild.")
    return path


def sha256_file(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


# ---------------------------------------------------------------------------
# Statistics -- ONE implementation each, cross-checked against the repo's own
# ---------------------------------------------------------------------------
def wilson_interval(k, n, z=Z_95):
    """Wilson score interval for a binomial proportion.

    Used only where the source file does NOT already carry a committed CI.
    Where it does, the committed value is read verbatim, so a published number
    can never be silently replaced by a recomputation under a different z.
    """
    if n == 0:
        return (float("nan"), float("nan"))
    p = k / n
    denom = 1.0 + (z * z) / n
    centre = p + (z * z) / (2 * n)
    margin = z * math.sqrt((p * (1 - p) + (z * z) / (4 * n)) / n)
    return ((centre - margin) / denom, (centre + margin) / denom)


def exact_mcnemar(b, c):
    """Two-sided EXACT McNemar: a binomial test on the discordant pairs.

    Exact, never the chi-squared approximation -- the discordant counts in this
    project are routinely in the single digits, where the approximation is not
    trustworthy.
    """
    n_disc = b + c
    if n_disc == 0:
        return 1.0
    return float(binomtest(b, n_disc, 0.5, alternative="two-sided").pvalue)


def rule_of_three_upper(n):
    """Upper 95% bound on a rate after observing ZERO events in n trials."""
    return 3.0 / n


def cross_check_statistics():
    """Rule 6: check every number against a second, independent derivation.

    The repo already contains three implementations of these tests. Rather
    than trust a fourth, this asserts the new one against all of them before a
    single table is written. A disagreement is FATAL.
    """
    from src.experiments.score_groundedness_set import (
        wilson_interval as repo_wilson_a)
    from src.experiments.summarize_zeroshot_baselines import (
        wilson_interval as repo_wilson_b)
    from src.experiments.compare_cascade_vs_tier2 import mcnemar_from_pairs

    checks = [(0, 1), (2, 2), (26, 31), (31, 33), (39, 54), (132, 175),
              (10441, 10441), (1, 175)]
    for k, n in checks:
        mine = wilson_interval(k, n, Z_95)
        theirs_a = repo_wilson_a(k, n, Z_95)
        theirs_b = repo_wilson_b(k, n, Z_95)
        for other, name in ((theirs_a, "score_groundedness_set"),
                            (theirs_b, "summarize_zeroshot_baselines")):
            for i in range(2):
                if abs(mine[i] - other[i]) > 1e-12:
                    raise AssertionError(
                        f"Wilson interval disagrees with {name} at "
                        f"k={k}, n={n}: {mine} vs {other}")

    # The two repo Wilson functions differ ONLY in their default z. Assert
    # that, so the convention note in RECONCILIATION.md stays true.
    a_default = repo_wilson_a(26, 31)
    b_default = repo_wilson_b(26, 31)
    if abs(a_default[0] - b_default[0]) < 1e-12:
        raise AssertionError(
            "The two repo Wilson defaults were expected to differ (z=1.96 vs "
            "the exact quantile). They now agree -- update the convention "
            "note in build_paper_artifacts.py and RECONCILIATION.md.")
    if abs(repo_wilson_a(26, 31, Z_95_EXACT)[0] - b_default[0]) > 1e-12:
        raise AssertionError(
            "The two repo Wilson implementations differ by more than their "
            "default z. That is a real algebraic disagreement and must be "
            "resolved before any CI is published.")

    # Exact McNemar against the shared pair-splitting helper.
    for b, c in ((0, 1), (1, 1), (3, 4), (7, 1), (0, 0)):
        pairs = ([({"correct": True}, {"correct": False})] * b +
                 [({"correct": False}, {"correct": True})] * c +
                 [({"correct": True}, {"correct": True})] * 5)
        theirs = mcnemar_from_pairs(pairs, binomtest)
        mine = exact_mcnemar(b, c)
        if abs(mine - float(theirs["p_value"])) > 1e-12:
            raise AssertionError(
                f"exact_mcnemar disagrees with compare_cascade_vs_tier2 at "
                f"b={b}, c={c}: {mine} vs {theirs['p_value']}")

    return {
        "wilson_checks": len(checks),
        "mcnemar_checks": 5,
        "z_paper": Z_95,
        "z_alternative_in_repo": Z_95_EXACT,
    }


# ---------------------------------------------------------------------------
# Emitted numbers
# ---------------------------------------------------------------------------
TAG_POST_HOC = "post-hoc"
TAG_BLOCKED = "BLOCKED"
TAG_NO_RESOLUTION = "no-resolution"
TAG_DEGENERATE = "degenerate"
TAG_CASE_STUDY = "case-study"
TAG_WITHIN_BAND = "within-band"

VALID_TAGS = {TAG_POST_HOC, TAG_BLOCKED, TAG_NO_RESOLUTION, TAG_DEGENERATE,
              TAG_CASE_STUDY, TAG_WITHIN_BAND}


@dataclass
class Number:
    id: str
    value: str
    source: str
    command: str
    tags: list = field(default_factory=list)
    group: str = ""
    note: str = ""
    # The value before formatting. The reconciliation audit needs it so an
    # anchor can ask "is the document's figure a correct ROUNDING of this?"
    # rather than "is it the same string?".
    raw: object = None


class Emitter:
    """Collects every number destined for NUMBERS.md.

    emit() is the choke point. It refuses a value with no committed source,
    refuses an unknown tag, refuses a Wilson CI on a pre-registered case-study
    axis, and auto-tags a difference smaller than its own noise band.
    """

    def __init__(self):
        self.numbers: list[Number] = []
        self._ids: set[str] = set()
        self.current_command = ""
        self.current_group = ""

    def emit(self, number_id, value, source, command=None, tags=None,
             group=None, note="", ci=None, band=None, diff=None):
        if number_id in self._ids:
            raise ValueError(f"duplicate NUMBERS.md id: {number_id}")
        self._ids.add(number_id)

        tags = list(tags or [])
        for tag in tags:
            if tag not in VALID_TAGS:
                raise ValueError(f"unknown tag {tag!r} on {number_id}")

        if isinstance(source, str):
            sources = [source]
        else:
            sources = list(source)
        for path in sources:
            abs_path = os.path.join(PROJECT_ROOT, path)
            if not os.path.isfile(abs_path):
                raise ValueError(
                    f"{number_id}: source {path!r} does not exist. Every "
                    f"number must come from a committed file.")

        if ci is not None and TAG_CASE_STUDY in tags:
            raise ValueError(
                f"{number_id}: a confidence interval was requested on a "
                f"pre-registered case-study axis. Those report counts only "
                f"-- see the 6C pre-registration.")

        # Structural, not editorial: a difference inside its own noise band is
        # tagged automatically so it can never be written up as a comparison.
        if band is not None and diff is not None:
            if abs(diff) < abs(band):
                tags.append(TAG_WITHIN_BAND)

        text = value if isinstance(value, str) else format_value(value)
        if ci is not None:
            text = f"{text} [95% CI {format_value(ci[0])}, " \
                   f"{format_value(ci[1])}]"

        self.numbers.append(Number(
            id=number_id,
            value=text,
            source="; ".join(sources),
            command=command or self.current_command,
            tags=sorted(set(tags)),
            group=group or self.current_group,
            note=note,
            raw=value,
        ))

    __call__ = emit


def format_value(value):
    """One formatting rule for the whole paper surface.

    Determinism matters more than prettiness: the parity test compares
    NUMBERS.md byte for byte.
    """
    if isinstance(value, str):
        return value
    if isinstance(value, (bool, np.bool_)):
        return "true" if value else "false"
    if value is None:
        return "n/a"
    if isinstance(value, (int, np.integer)):
        return str(int(value))
    value = float(value)
    if math.isnan(value):
        return "nan"
    return f"{value:.6f}".rstrip("0").rstrip(".") if abs(value) < 1 \
        else f"{value:.4f}".rstrip("0").rstrip(".")


def frac(k, n):
    """A k/n count, formatted the way the project writes them."""
    return f"{int(k)}/{int(n)}"


# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------
TABLES = []
FIGURES = []


def table(table_id, title, sources, command):
    def deco(fn):
        TABLES.append({"id": table_id, "title": title, "sources": sources,
                       "command": command, "fn": fn})
        return fn
    return deco


def figure(fig_id, title, sources, command, width, caption):
    def deco(fn):
        FIGURES.append({"id": fig_id, "title": title, "sources": sources,
                        "command": command, "fn": fn, "width": width,
                        "caption": caption})
        return fn
    return deco


def slug(text):
    out = re.sub(r"[^a-z0-9]+", "_", text.lower()).strip("_")
    return out


# ---------------------------------------------------------------------------
# Writers
# ---------------------------------------------------------------------------
def write_csv(df, path):
    df.to_csv(path, index=False, lineterminator="\n")


LATEX_ESCAPES = {
    "&": r"\&", "%": r"\%", "$": r"\$", "#": r"\#", "_": r"\_",
    "{": r"\{", "}": r"\}", "~": r"\textasciitilde{}",
    "^": r"\textasciicircum{}", "\\": r"\textbackslash{}",
}


def latex_escape(text):
    text = str(text)
    out = []
    for ch in text:
        out.append(LATEX_ESCAPES.get(ch, ch))
    return "".join(out)


def write_latex(df, path, table_id, title, caption_extra=""):
    """booktabs, written directly -- df.to_latex's default ruling is not it."""
    cols = list(df.columns)
    align = "".join(
        "r" if pd.api.types.is_numeric_dtype(df[c]) else "l" for c in cols)
    lines = [
        "% Generated by src/experiments/build_paper_artifacts.py -- do not "
        "edit by hand.",
        r"\begin{table}[t]",
        r"\centering",
        r"\caption{" + latex_escape(title) +
        (" " + latex_escape(caption_extra) if caption_extra else "") + "}",
        r"\label{tab:" + table_id.lower() + "}",
        r"\begin{tabular}{" + align + "}",
        r"\toprule",
        " & ".join(latex_escape(c) for c in cols) + r" \\",
        r"\midrule",
    ]
    for _, row in df.iterrows():
        cells = []
        for c in cols:
            v = row[c]
            if isinstance(v, float) and math.isnan(v):
                cells.append("--")
            else:
                cells.append(latex_escape(
                    format_value(v) if isinstance(v, (int, float, np.integer,
                                                      np.floating)) else v))
        lines.append(" & ".join(cells) + r" \\")
    lines += [r"\bottomrule", r"\end{tabular}", r"\end{table}", ""]
    with open(path, "w", encoding="utf-8", newline="\n") as fh:
        fh.write("\n".join(lines))


# ---------------------------------------------------------------------------
# Small readers shared by several builders
# ---------------------------------------------------------------------------
def read_json(path):
    with open(path, "r", encoding="utf-8") as fh:
        return json.load(fh)


def ablation_count(path, section="classification"):
    """(correct, n) from an ablation results CSV.

    The 45-ticket baseline file carries BOTH the classification rows and the
    9 escalation rows in one CSV under a `section` column; counting the whole
    file would silently report 41/54. That is the shape of bug this project
    keeps meeting, so the section filter is explicit.
    """
    df = pd.read_csv(path)
    if "section" in df.columns:
        df = df[df["section"] == section]
        if df.empty:
            raise ValueError(f"{rel(path)}: no rows in section {section!r}")
    return int(df["correct"].sum()), int(len(df))


def mcnemar_from_discordant_csv(path, right_token):
    """(b, c, p) from a committed discordant-pairs CSV.

    Those CSVs list ONLY discordant pairs, one row each, with a `direction`
    column naming which side was right. b counts the rows where the left-hand
    method was the right one.
    """
    df = pd.read_csv(path)
    directions = df["direction"].astype(str)
    c = int(directions.str.contains(right_token).sum())
    b = int(len(df) - c)
    return b, c, exact_mcnemar(b, c)


def zs_summary(backend_slug, model_slug, bench):
    """The long-form zero-shot summary CSV as a {(metric, category): value}."""
    path = src(f"zeroshot_summary_{backend_slug}_{model_slug}_{bench}.csv")
    df = pd.read_csv(path, keep_default_na=False)
    out = {}
    for _, row in df.iterrows():
        key = (row["metric"], row["category"])
        raw = row["value"]
        try:
            out[key] = float(raw)
        except (TypeError, ValueError):
            out[key] = raw
    return out, rel(path)


GEMINI_SLUG = ("gemini", "gemini-flash-lite-latest")
QWEN_SLUG = ("ollama", "qwen2-5-3b-instruct")

NO_SOURCE = []


def no_source(label, where, why, value_in_docs):
    """Record a number the docs state that no committed file can produce.

    Deliberately NOT a fallback: nothing is invented, nothing is re-run, and
    the value never enters NUMBERS.md as though it had a source.
    """
    NO_SOURCE.append({"number": label, "stated_in_docs": value_in_docs,
                      "doc_location": where, "why_no_source": why})
    return "n/a (no committed source)"


# ---------------------------------------------------------------------------
# T1 -- classification and baselines on the two fixed benchmarks
# ---------------------------------------------------------------------------
@table("T1", "Classification accuracy on the fixed benchmarks, with exact "
             "paired tests",
       ["data/embedding_comparison/embedding_model_comparison.csv",
        "data/ablation_baseline_results.csv",
        "data/ablation_tier2-only_results.csv",
        "data/ablation_no-cascade_results.csv",
        "data/zeroshot_summary_gemini_gemini-flash-lite-latest_benchmark45.csv",
        "data/zeroshot_summary_ollama_qwen2-5-3b-instruct_benchmark45.csv"],
       "python src/experiments/run_ablation_study.py --mode baseline; "
       "python src/experiments/summarize_zeroshot_baselines.py")
def build_t1(emit):
    emb_path = src("embedding_comparison", "embedding_model_comparison.csv")
    emb = pd.read_csv(emb_path)

    b45_base = src("ablation_baseline_results.csv")
    b14_base = src("ablation_baseline_results_benchmark14.csv")
    d175_base = src("ablation_baseline_results_deployment175.csv")
    b45_t2 = src("ablation_tier2-only_results.csv")
    b14_t2 = src("ablation_tier2-only_results_benchmark14.csv")
    d175_t2 = src("ablation_tier2-only_results_deployment175.csv")
    b45_t1 = src("ablation_no-cascade_results.csv")
    d175_t1 = src("ablation_no-cascade_results_deployment175.csv")

    cascade45 = ablation_count(b45_base)
    cascade14 = ablation_count(b14_base)
    cascade175 = ablation_count(d175_base)
    t2_45 = ablation_count(b45_t2)
    t2_14 = ablation_count(b14_t2)
    t2_175 = ablation_count(d175_t2)
    t1_45 = ablation_count(b45_t1)
    t1_175 = ablation_count(d175_t1)

    # Rule 6 -- a second, independent derivation of the same number. The BGE
    # row of the embedding comparison and the tier2-only ablation were
    # produced by different scripts on different days; if they ever disagree,
    # one of them is stale and the paper must not quietly pick a winner.
    bge_row = emb[emb["model_name"] == "BAAI/bge-base-en-v1.5"].iloc[0]
    if int(bge_row["bench46_correct_count"]) != t2_45[0]:
        raise AssertionError(
            f"BGE 45-ticket count disagrees between "
            f"{rel(emb_path)} ({int(bge_row['bench46_correct_count'])}) and "
            f"{rel(b45_t2)} ({t2_45[0]}). One of them is stale.")
    if int(bge_row["benchmark_correct_count"]) != t2_14[0]:
        raise AssertionError(
            f"BGE 14-ticket count disagrees between {rel(emb_path)} and "
            f"{rel(b14_t2)}.")

    zs_g45, zs_g45_path = zs_summary(*GEMINI_SLUG, "benchmark45")
    zs_g14, zs_g14_path = zs_summary(*GEMINI_SLUG, "benchmark14")
    zs_q45, zs_q45_path = zs_summary(*QWEN_SLUG, "benchmark45")
    zs_q14, zs_q14_path = zs_summary(*QWEN_SLUG, "benchmark14")

    def zs_counts(summary):
        return (int(summary[("n_correct", "")]), int(summary[("n", "")]),
                int(summary[("n_unparseable", "")]))

    g45, g14 = zs_counts(zs_g45), zs_counts(zs_g14)
    q45, q14 = zs_counts(zs_q45), zs_counts(zs_q14)

    minilm = emb[emb["model_name"] == "all-MiniLM-L6-v2"].iloc[0]
    e5 = emb[emb["model_name"] == "intfloat/e5-base-v2"].iloc[0]

    tfidf14 = no_source(
        "TF-IDF + LogReg, 14-ticket benchmark",
        "README.md, the three-way classifier comparison and Final Classification Comparison tables",
        "train_baseline_tfidf.py and generalization_test.py print their "
        "results and write no file; no ablation --mode no-cascade run exists "
        "for benchmark14.", "7/14 (50.0%)")
    distil14 = no_source(
        "Fine-tuned DistilBERT, 14-ticket benchmark",
        "README.md, the three-way classifier comparison and Final Classification Comparison tables",
        "train_distilbert.py writes only label_mapping.json -- no metrics "
        "file of any kind.", "7/14 (50.0%)")
    distil45 = no_source(
        "Fine-tuned DistilBERT, 45-ticket benchmark", "(never measured)",
        "DistilBERT was never run against the 45-ticket benchmark, and the "
        "embedding comparison CSV has no DistilBERT row.", "(absent)")

    rows = []

    def add(method, kind, c14, c45, c175, note=""):
        is_str45 = isinstance(c45, str)
        rows.append({
            "method": method,
            "kind": kind,
            "benchmark14": c14 if isinstance(c14, str) else frac(*c14),
            "benchmark45": c45 if is_str45 else frac(*c45),
            "benchmark45_pct": (float("nan") if is_str45
                                else 100.0 * c45[0] / c45[1]),
            "benchmark45_ci_low": (float("nan") if is_str45
                                   else wilson_interval(*c45, Z_95)[0]),
            "benchmark45_ci_high": (float("nan") if is_str45
                                    else wilson_interval(*c45, Z_95)[1]),
            "deployment175": c175 if isinstance(c175, str) else frac(*c175),
            "note": note,
        })

    add("TF-IDF + LogReg (Tier-1 only)", "trained", tfidf14, t1_45, t1_175)
    add("Fine-tuned DistilBERT", "trained", distil14, distil45,
        "n/a (no committed source)")
    add("all-MiniLM-L6-v2 + LogReg", "trained",
        (int(minilm["benchmark_correct_count"]), 14),
        (int(minilm["bench46_correct_count"]), 45), "n/a (not run)")
    add("intfloat/e5-base-v2 + LogReg", "trained",
        (int(e5["benchmark_correct_count"]), 14),
        (int(e5["bench46_correct_count"]), 45), "n/a (not run)")
    add("BGE-base-en-v1.5 + LogReg (Tier-2 only)", "trained",
        t2_14, t2_45, t2_175, "production representation")
    add("Cascade (Tier-1 -> Tier-2), production", "trained",
        cascade14, cascade45, cascade175, "the shipped configuration")
    add("Zero-shot Gemini flash-lite", "zero-shot LLM",
        (g14[0], g14[1]), (g45[0], g45[1]), "n/a (not run)",
        f"{g45[2]} unparseable on 45, {g14[2]} on 14")
    add("Zero-shot Qwen2.5-3B (local CPU)", "zero-shot LLM",
        (q14[0], q14[1]), (q45[0], q45[1]), "n/a (not run)",
        f"{q45[2]} unparseable on 45, {q14[2]} on 14")

    df = pd.DataFrame(rows)

    # ---- the paired tests, every one exact McNemar on the same tickets ----
    pairs = []

    def add_pair(label, path, right_token, left_name, right_name, eval_set):
        b, c, p = mcnemar_from_discordant_csv(path, right_token)
        pairs.append({"comparison": label, "eval_set": eval_set,
                      "left": left_name, "right": right_name,
                      "b_left_only_right": b, "c_right_only_right": c,
                      "n_discordant": b + c, "exact_mcnemar_p": p,
                      "test": "exact McNemar (binomial)"})
        return p

    p_cas45 = add_pair(
        "cascade vs Tier-2-only",
        src("cascade_vs_tier2_mcnemar_benchmark45.csv"),
        "cascade_wrong_tier2_right", "cascade", "Tier-2-only", "benchmark45")
    p_cas175 = add_pair(
        "cascade vs Tier-2-only",
        src("cascade_vs_tier2_mcnemar_deployment175.csv"),
        "cascade_wrong_tier2_right", "cascade", "Tier-2-only", "deployment175")
    p_g45 = add_pair(
        "zero-shot Gemini vs Tier-2-only",
        src(f"zeroshot_vs_tier2_mcnemar_{GEMINI_SLUG[0]}_{GEMINI_SLUG[1]}"
            f"_benchmark45.csv"),
        "zeroshot_right", "Tier-2-only", "zero-shot Gemini", "benchmark45")
    p_g14 = add_pair(
        "zero-shot Gemini vs Tier-2-only",
        src(f"zeroshot_vs_tier2_mcnemar_{GEMINI_SLUG[0]}_{GEMINI_SLUG[1]}"
            f"_benchmark14.csv"),
        "zeroshot_right", "Tier-2-only", "zero-shot Gemini", "benchmark14")
    p_q45 = add_pair(
        "zero-shot Qwen2.5-3B vs Tier-2-only",
        src(f"zeroshot_vs_tier2_mcnemar_{QWEN_SLUG[0]}_{QWEN_SLUG[1]}"
            f"_benchmark45.csv"),
        "zeroshot_right", "Tier-2-only", "zero-shot Qwen2.5-3B", "benchmark45")
    p_q14 = add_pair(
        "zero-shot Qwen2.5-3B vs Tier-2-only",
        src(f"zeroshot_vs_tier2_mcnemar_{QWEN_SLUG[0]}_{QWEN_SLUG[1]}"
            f"_benchmark14.csv"),
        "zeroshot_right", "Tier-2-only", "zero-shot Qwen2.5-3B", "benchmark14")
    p_qg45 = add_pair(
        "zero-shot Qwen2.5-3B vs zero-shot Gemini",
        src(f"zeroshot_vs_zeroshot_mcnemar_{QWEN_SLUG[0]}_{QWEN_SLUG[1]}_vs_"
            f"{GEMINI_SLUG[0]}_{GEMINI_SLUG[1]}_benchmark45.csv"),
        "gemini_right", "zero-shot Qwen2.5-3B", "zero-shot Gemini",
        "benchmark45")
    p_qg14 = add_pair(
        "zero-shot Qwen2.5-3B vs zero-shot Gemini",
        src(f"zeroshot_vs_zeroshot_mcnemar_{QWEN_SLUG[0]}_{QWEN_SLUG[1]}_vs_"
            f"{GEMINI_SLUG[0]}_{GEMINI_SLUG[1]}_benchmark14.csv"),
        "gemini_right", "zero-shot Qwen2.5-3B", "zero-shot Gemini",
        "benchmark14")

    pairs_df = pd.DataFrame(pairs)

    # ---- emitted numbers ----
    emit("T1.cascade.benchmark45", frac(*cascade45), rel(b45_base),
         ci=wilson_interval(*cascade45, Z_95))
    emit("T1.cascade.benchmark14", frac(*cascade14), rel(b14_base))
    emit("T1.cascade.deployment175", frac(*cascade175), rel(d175_base),
         ci=wilson_interval(*cascade175, Z_95))
    emit("T1.tier2only.benchmark45", frac(*t2_45), rel(b45_t2),
         ci=wilson_interval(*t2_45, Z_95))
    emit("T1.tier2only.benchmark14", frac(*t2_14), rel(b14_t2))
    emit("T1.tier2only.deployment175", frac(*t2_175), rel(d175_t2),
         ci=wilson_interval(*t2_175, Z_95))
    emit("T1.tier1only.benchmark45", frac(*t1_45), rel(b45_t1),
         ci=wilson_interval(*t1_45, Z_95),
         note="Tier-1 answering everything -- the TF-IDF representation, NOT "
              "a measure of what cascading is worth.")
    emit("T1.tier1only.deployment175", frac(*t1_175), rel(d175_t1),
         ci=wilson_interval(*t1_175, Z_95))
    emit("T1.minilm.benchmark45",
         frac(int(minilm["bench46_correct_count"]), 45), rel(emb_path))
    emit("T1.e5.benchmark45", frac(int(e5["bench46_correct_count"]), 45),
         rel(emb_path))
    emit("T1.zeroshot_gemini.benchmark45", frac(g45[0], g45[1]), zs_g45_path,
         group="5C", ci=(zs_g45[("accuracy_unparseable_wrong_ci_low", "")],
                         zs_g45[("accuracy_unparseable_wrong_ci_high", "")]))
    emit("T1.zeroshot_gemini.benchmark14", frac(g14[0], g14[1]), zs_g14_path,
         group="5C")
    emit("T1.zeroshot_qwen.benchmark45", frac(q45[0], q45[1]), zs_q45_path,
         group="5C", ci=(zs_q45[("accuracy_unparseable_wrong_ci_low", "")],
                         zs_q45[("accuracy_unparseable_wrong_ci_high", "")]))
    emit("T1.zeroshot_qwen.benchmark14", frac(q14[0], q14[1]), zs_q14_path,
         group="5C")
    emit("T1.zeroshot_gemini.unparseable.benchmark45", g45[2], zs_g45_path,
         group="5C")
    emit("T1.zeroshot_qwen.unparseable.benchmark45", q45[2], zs_q45_path,
         group="5C")
    emit("T1.mcnemar.cascade_vs_tier2.benchmark45", p_cas45,
         rel(src("cascade_vs_tier2_mcnemar_benchmark45.csv")),
         note="The cascade is one ticket WORSE and indistinguishable.")
    emit("T1.mcnemar.cascade_vs_tier2.deployment175", p_cas175,
         rel(src("cascade_vs_tier2_mcnemar_deployment175.csv")))
    emit("T1.mcnemar.gemini_vs_tier2.benchmark45", p_g45, zs_g45_path,
         group="5C",
         note="Gemini is distinguishably better than the trained Tier-2 here.")
    emit("T1.mcnemar.gemini_vs_tier2.benchmark14", p_g14, zs_g14_path,
         group="5C")
    emit("T1.mcnemar.qwen_vs_tier2.benchmark45", p_q45, zs_q45_path,
         group="5C",
         note="INDISTINGUISHABLE from Tier-2, which is not 'better'.")
    emit("T1.mcnemar.qwen_vs_tier2.benchmark14", p_q14, zs_q14_path,
         group="5C")
    emit("T1.mcnemar.qwen_vs_gemini.benchmark45", p_qg45, zs_q45_path,
         group="5C",
         note="6 tickets apart and NOT significant; the benchmark is "
              "Gemini-generated, so the Qwen arm is the partial control for "
              "authorship.")
    emit("T1.mcnemar.qwen_vs_gemini.benchmark14", p_qg14, zs_q14_path,
         group="5C")

    return {"main": df, "paired_tests": pairs_df}


# ---------------------------------------------------------------------------
# T2 -- Phase 5C per-category detail
# ---------------------------------------------------------------------------
@table("T2", "Zero-shot per-category behaviour (Phase 5C)",
       ["data/zeroshot_summary_gemini_gemini-flash-lite-latest_benchmark45.csv",
        "data/zeroshot_summary_ollama_qwen2-5-3b-instruct_benchmark45.csv",
        "data/zeroshot_summary_gemini_gemini-flash-lite-latest_benchmark14.csv",
        "data/zeroshot_summary_ollama_qwen2-5-3b-instruct_benchmark14.csv"],
       "python src/experiments/summarize_zeroshot_baselines.py --backend "
       "ollama --model qwen2.5:3b-instruct --set both "
       "--prompt-check-against gemini")
def build_t2(emit):
    arms = {
        "gemini": (GEMINI_SLUG, "zero-shot Gemini flash-lite"),
        "qwen": (QWEN_SLUG, "zero-shot Qwen2.5-3B"),
    }
    rows = []
    loaded = {}
    for key, (slugs, label) in arms.items():
        for bench in ("benchmark45", "benchmark14"):
            summary, path = zs_summary(*slugs, bench)
            loaded[(key, bench)] = (summary, path)
            cats = sorted({c for (m, c) in summary if m == "support" and c})
            for cat in cats:
                support = summary[("support", cat)]
                recall = summary[("recall", cat)]
                precision = summary.get(("precision", cat), float("nan"))
                rows.append({
                    "arm": label,
                    "eval_set": bench,
                    "category": cat,
                    "support": int(support),
                    "recall": recall,
                    "n_recalled": int(round(recall * support)),
                    "precision": precision,
                    "predicted_this_category":
                        ("0" if (isinstance(precision, float)
                                 and math.isnan(precision)) else ""),
                })
    df = pd.DataFrame(rows).sort_values(
        ["arm", "eval_set", "category"], kind="mergesort").reset_index(
            drop=True)

    q45, q45_path = loaded[("qwen", "benchmark45")]
    q14, q14_path = loaded[("qwen", "benchmark14")]
    g45, g45_path = loaded[("gemini", "benchmark45")]

    emit("T2.qwen.database.recall.benchmark45",
         frac(round(q45[("recall", "Database")] * q45[("support", "Database")]),
              q45[("support", "Database")]),
         q45_path, group="5C",
         note="Qwen2.5-3B predicted Database for no ticket at all; the "
              "precision cell is undefined, not zero.")
    emit("T2.qwen.database.recall.benchmark14",
         frac(round(q14[("recall", "Database")] * q14[("support", "Database")]),
              q14[("support", "Database")]),
         q14_path, group="5C")
    emit("T2.qwen.application.precision.benchmark45",
         q45[("precision", "Application")], q45_path, group="5C",
         note="Everything Qwen could not place went to Application.")
    emit("T2.qwen.infrastructure.recall.benchmark45",
         q45[("recall", "Infrastructure")], q45_path, group="5C",
         note="Infrastructure fails at ~50% for the trained model AND both "
              "LLMs -- a property of the register, not of any one method.")
    emit("T2.gemini.infrastructure.recall.benchmark45",
         g45[("recall", "Infrastructure")], g45_path, group="5C")
    emit("T2.prompt_identity.benchmark45",
         q45[("prompt_identity_verified_vs_gemini", "")], q45_path,
         group="5C",
         note="Byte-identical prompts, verified by sha256 in both directions.")

    return {"main": df}


# ---------------------------------------------------------------------------
# T3 -- the ablation, the missing control, and what the cascade actually buys
# ---------------------------------------------------------------------------
@table("T3", "Ablation across the four modes, with the Tier-2-only control "
             "and the latency it buys",
       ["data/ablation_baseline_results.csv",
        "data/ablation_no-cascade_results.csv",
        "data/ablation_tier2-only_results.csv",
        "data/ablation_no-rag_results.csv",
        "data/cascade_vs_tier2_mcnemar_benchmark45.csv",
        "data/inference_latency_13th-gen-intel-r-core-tm-i5-1334u.csv"],
       "python src/experiments/run_ablation_study.py --mode {baseline,"
       "no-cascade,tier2-only,no-rag}; "
       "python src/experiments/compare_cascade_vs_tier2.py; "
       "python src/experiments/measure_inference_latency.py")
def build_t3(emit):
    base45 = src("ablation_baseline_results.csv")
    nocas45 = src("ablation_no-cascade_results.csv")
    t2only45 = src("ablation_tier2-only_results.csv")
    norag = src("ablation_no-rag_results.csv")
    base175 = src("ablation_baseline_results_deployment175.csv")
    nocas175 = src("ablation_no-cascade_results_deployment175.csv")
    t2only175 = src("ablation_tier2-only_results_deployment175.csv")

    esc_base = pd.read_csv(base45)
    esc_base = esc_base[esc_base["section"] == "escalation"]
    esc_correct = int((esc_base["escalated"].astype(str).str.lower()
                       == "true").sum())
    norag_df = pd.read_csv(norag)
    norag_correct = int(norag_df["escalation_correct"].astype(bool).sum())

    b45 = ablation_count(base45)
    n45 = ablation_count(nocas45)
    t45 = ablation_count(t2only45)
    b175 = ablation_count(base175)
    n175 = ablation_count(nocas175)
    t175 = ablation_count(t2only175)

    rows = [
        {"mode": "baseline (production cascade + RAG gate)",
         "what_it_is": "the shipped pipeline",
         "benchmark45": frac(*b45), "deployment175": frac(*b175),
         "adversarial9_escalation": frac(esc_correct, len(esc_base))},
        {"mode": "no-cascade (Tier-1 answers everything)",
         "what_it_is": "the TF-IDF REPRESENTATION, not the cascade's value",
         "benchmark45": frac(*n45), "deployment175": frac(*n175),
         "adversarial9_escalation": "n/a"},
        {"mode": "tier2-only (the control that isolates the cascade)",
         "what_it_is": "BGE answering everything",
         "benchmark45": frac(*t45), "deployment175": frac(*t175),
         "adversarial9_escalation": "n/a"},
        {"mode": "no-rag (RAG similarity gate removed)",
         "what_it_is": "the escalation gate's value",
         "benchmark45": "n/a", "deployment175": "n/a",
         "adversarial9_escalation": frac(norag_correct, len(norag_df))},
    ]
    df = pd.DataFrame(rows)

    # The arithmetic the ablation's headline used to be read as. Computed here
    # so the paper can state precisely what it is and is not.
    repr_gap_pts = 100.0 * (b45[0] / b45[1] - n45[0] / n45[1])
    cascade_delta_45 = b45[0] - t45[0]
    cascade_delta_175 = b175[0] - t175[0]

    _, _, p45 = mcnemar_from_discordant_csv(
        src("cascade_vs_tier2_mcnemar_benchmark45.csv"),
        "cascade_wrong_tier2_right")
    _, _, p175 = mcnemar_from_discordant_csv(
        src("cascade_vs_tier2_mcnemar_deployment175.csv"),
        "cascade_wrong_tier2_right")

    lat_path = src("inference_latency_13th-gen-intel-r-core-tm-i5-1334u.csv")
    lat = pd.read_csv(lat_path)
    t1_row = lat[lat["measurement"] == "tier1"].iloc[0]
    t2_row = lat[lat["measurement"] == "tier2"].iloc[0]
    exp_rows = lat[lat["measurement"] == "cascade_expected"]

    lat_rows = []
    for _, r in exp_rows.iterrows():
        saving = -float(r["delta_ms"]) / float(r["tier2_only_median_ms"])
        lat_rows.append({
            "eval_set": r["eval_set"],
            "tier1_share": float(r["tier1_share"]),
            "cascade_expected_median_ms": float(r["cascade_expected_median_ms"]),
            "tier2_only_median_ms": float(r["tier2_only_median_ms"]),
            "delta_ms": float(r["delta_ms"]),
            "median_latency_saving": saving,
        })
    lat_df = pd.DataFrame(lat_rows)

    ratio = float(t2_row["median_ms"]) / float(t1_row["median_ms"])

    emit("T3.repr_gap_points", repr_gap_pts, rel(base45),
         note="baseline MINUS Tier-1-only. This is the BGE-vs-TF-IDF "
              "REPRESENTATION gap. It is NOT what cascading is worth -- see "
              "the do-not-cite list.")
    emit("T3.cascade_delta.benchmark45", cascade_delta_45, rel(base45),
         note="The cascade is one ticket worse than the Tier-2-only control.")
    emit("T3.cascade_delta.deployment175", cascade_delta_175, rel(base175))
    emit("T3.mcnemar.benchmark45", p45,
         rel(src("cascade_vs_tier2_mcnemar_benchmark45.csv")))
    emit("T3.mcnemar.deployment175", p175,
         rel(src("cascade_vs_tier2_mcnemar_deployment175.csv")))
    emit("T3.latency.tier1_median_ms", float(t1_row["median_ms"]), rel(lat_path))
    emit("T3.latency.tier2_median_ms", float(t2_row["median_ms"]), rel(lat_path))
    emit("T3.latency.tier1_p95_ms", float(t1_row["p95_ms"]), rel(lat_path))
    emit("T3.latency.tier2_p95_ms", float(t2_row["p95_ms"]), rel(lat_path))
    emit("T3.latency.tier2_over_tier1", ratio, rel(lat_path),
         note="Warm, batch size 1, on the recorded CPU.")
    emit("T3.latency.saving.benchmark45",
         float(lat_df[lat_df.eval_set == "benchmark45"]
               ["median_latency_saving"].iloc[0]), rel(lat_path),
         note="This -- not accuracy -- is what the cascade buys.")
    emit("T3.latency.saving.deployment175",
         float(lat_df[lat_df.eval_set == "deployment175"]
               ["median_latency_saving"].iloc[0]), rel(lat_path))
    emit("T3.norag.escalation_correct", frac(norag_correct, len(norag_df)),
         rel(norag),
         note="Removing the RAG gate costs the adversarial set; the baseline "
              f"scores {frac(esc_correct, len(esc_base))} there.")
    emit("T3.baseline.escalation_correct", frac(esc_correct, len(esc_base)),
         rel(base45))

    return {"main": df, "latency": lat_df}


# ---------------------------------------------------------------------------
# T4 -- cascade calibration and reliability
# ---------------------------------------------------------------------------
@table("T4", "Cascade threshold calibration and tier reliability",
       ["data/calibration_reliability_data.csv"],
       "python src/experiments/plot_calibration_curves.py")
def build_t4(emit):
    path = src("calibration_reliability_data.csv")
    bins = pd.read_csv(path)

    rows = []
    eces = {}
    for tier, grp in bins.groupby("tier"):
        n_total = int(grp["n_tickets_in_bin"].sum())
        ece = 0.0
        for _, r in grp.iterrows():
            w = int(r["n_tickets_in_bin"]) / n_total
            ece += w * abs(float(r["mean_predicted_confidence"])
                           - float(r["observed_accuracy"]))
        eces[int(tier)] = ece
        rows.append({
            "tier": int(tier),
            "representation": "TF-IDF + LogReg" if int(tier) == 1
                              else "BGE-base-en-v1.5 + LogReg",
            "n_tickets": n_total,
            "n_bins_non_empty": int(len(grp)),
            "expected_calibration_error": ece,
            "mean_confidence": float(
                (grp["mean_predicted_confidence"] * grp["n_tickets_in_bin"]
                 ).sum() / n_total),
            "observed_accuracy": float(
                (grp["observed_accuracy"] * grp["n_tickets_in_bin"]
                 ).sum() / n_total),
        })
    df = pd.DataFrame(rows).sort_values("tier").reset_index(drop=True)

    # The live gate, read from config rather than retyped.
    gate = float(settings.cascade.confidence_threshold)

    no_source(
        "Cascade threshold table by target accuracy (the three calibration "
        "attempts, two rejected)", "README.md, cascade calibration section",
        "train_cascade.py prints its sweep and writes no results file, so the "
        "threshold-by-target-accuracy table cannot be regenerated.",
        "three attempts; 0.50 chosen at a 70-80% target")

    emit("T4.tier1.ece", eces[1], rel(path),
         note="Count-weighted binned ECE, recomputed from the committed "
              "bins, on the 500-ticket IN-DISTRIBUTION production batch.")
    emit("T4.tier2.ece", eces[2], rel(path))
    emit("T4.observed_accuracy_every_bin",
         float(bins["observed_accuracy"].min()), rel(path),
         note="The MINIMUM observed accuracy across all bins of both tiers. "
              "It is 1.0, so both tiers are systematically UNDER-confident "
              "here and the ECE is entirely the distance to a ceiling. "
              "In-distribution accuracy is uninformative on "
              "template-generated data -- treat a new 100% as a red flag.")
    emit("T4.cascade.threshold", gate, "src/agent/config.py",
         note="The live gate. Frozen; Phases 5-9 are measurement only.")
    emit("T4.reliability.n_tickets", int(
        bins[bins.tier == 1]["n_tickets_in_bin"].sum()), rel(path))

    return {"main": df, "bins": bins}


# ---------------------------------------------------------------------------
# T5 -- RAG similarity threshold derivation
# ---------------------------------------------------------------------------
@table("T5", "RAG similarity threshold: the in-domain/OOD trade-off around "
             "the chosen 0.67",
       ["data/rag_similarity_calibration.csv",
        "data/rag_similarity_calibration_combined.csv"],
       "python -m src.experiments.calibrate_rag_similarity_threshold")
def build_t5(emit):
    comb_path = src("rag_similarity_calibration_combined.csv")
    comb = pd.read_csv(comb_path)
    base_path = src("rag_similarity_calibration.csv")

    gate = float(settings.rag.similarity_threshold)
    df = comb[["threshold", "n_in_domain_proceed", "n_in_domain_escalate",
               "ood_leakage_rate", "precision_combined", "recall_combined",
               "f1_combined"]].copy()
    df["is_live_gate"] = df["threshold"] == gate
    df = df.sort_values("threshold", ascending=False).reset_index(drop=True)

    at_gate = comb[comb["threshold"] == gate].iloc[0]

    # The adversarial "safe range" comes from the 9 hand-written tickets'
    # measured similarities: the widest band that keeps every must-escalate
    # ticket out and every must-proceed ticket in.
    adv_path = src("ablation_no-rag_results.csv")
    adv = pd.read_csv(adv_path)
    must_escalate = adv[adv["expected_escalate"].astype(bool)]
    must_proceed = adv[~adv["expected_escalate"].astype(bool)]
    safe_low = float(must_escalate["top_similarity"].max())
    safe_high = float(must_proceed["top_similarity"].min())

    lower_row = comb[comb["threshold"] < gate].sort_values(
        "threshold", ascending=False)
    leak_below = float(lower_row.iloc[1]["ood_leakage_rate"])
    thr_below = float(lower_row.iloc[1]["threshold"])

    no_source(
        "In-domain self-retrieval contamination rate 5.7% (10/175)",
        "README.md, the RAG threshold calibration section and What's Done vs What's Pending",
        "calibrate_rag_similarity_threshold.py computes it but writes no "
        "column for it in either calibration CSV.", "5.7% (10/175)")

    emit("T5.gate", gate, "src/agent/config.py")
    emit("T5.ood_leakage_at_gate", float(at_gate["ood_leakage_rate"]),
         rel(comb_path))
    emit("T5.ood_leakage_below_gate", leak_below, rel(comb_path),
         note=f"At threshold {format_value(thr_below)} -- dropping the gate "
              f"two steps nearly triples OOD leakage.")
    emit("T5.in_domain_proceed_at_gate", frac(
        int(at_gate["n_in_domain_proceed"]),
        int(at_gate["n_in_domain_proceed"]) +
        int(at_gate["n_in_domain_escalate"])), rel(comb_path))
    emit("T5.safe_range_low", safe_low, rel(adv_path),
         note="Highest similarity among the adversarial tickets that MUST "
              "escalate.")
    emit("T5.safe_range_high", safe_high, rel(adv_path),
         note="Lowest similarity among those that must proceed. The gate at "
              "0.67 sits inside this band, and it errs in BOTH directions -- "
              "see FRAMING.md.")

    return {"main": df, "raw_in_domain": pd.read_csv(base_path)}


# ---------------------------------------------------------------------------
# T6 -- training-set skew and batch intake
# ---------------------------------------------------------------------------
@table("T6", "Class-imbalance sweep and batch-intake behaviour",
       ["data/skewed/imbalance_sweep_results.csv",
        "data/batch_intake/batch_summary.csv"],
       "python src/experiments/run_imbalance_sweep.py; "
       "python src/experiments/process_ticket_batch.py")
def build_t6(emit):
    sweep_path = src("skewed", "imbalance_sweep_results.csv")
    sweep = pd.read_csv(sweep_path)
    batch_path = src("batch_intake", "batch_summary.csv")
    batch = pd.read_csv(batch_path)

    sweep_out = sweep[[
        "skew_level", "am_train_count", "tfidf_heldout_acc",
        "minilm_heldout_acc", "tfidf_bench_14", "minilm_bench_14",
        "tfidf_bench_45", "minilm_bench_45", "cascade_safety_net_failures",
    ]].copy().sort_values("am_train_count").reset_index(drop=True)

    batch_out = batch.copy()

    worst = sweep.sort_values("am_train_count").iloc[0]
    best = sweep.sort_values("am_train_count").iloc[-1]

    emit("T6.skew.min_am_rows", int(worst["am_train_count"]), rel(sweep_path))
    emit("T6.skew.max_am_rows", int(best["am_train_count"]), rel(sweep_path))
    emit("T6.skew.heldout_acc_at_min_rows", float(worst["minilm_heldout_acc"]),
         rel(sweep_path),
         note="In-distribution accuracy stays ~1.0 across the whole sweep -- "
              "template-generated data makes it uninformative.")
    emit("T6.skew.cascade_safety_net_failures_total",
         int(sweep["cascade_safety_net_failures"].sum()), rel(sweep_path))
    emit("T6.batch.volume_total", int(batch["volume_received"].sum()),
         rel(batch_path))
    emit("T6.batch.auto_resolved_total",
         int(batch["auto_resolved_count"].sum()), rel(batch_path))
    emit("T6.batch.gemini_failed_total",
         int(batch["gemini_call_failed_count"].sum()), rel(batch_path),
         note="Produced under MiniLM. Re-running under BGE would silently "
              "invalidate the 0.80 clustering calibration -- see "
              "PROJECT_STATUS.md, Known risks.")

    return {"imbalance_sweep": sweep_out, "batch_intake": batch_out}


# ---------------------------------------------------------------------------
# Conformal helpers
# ---------------------------------------------------------------------------
CONF_PRIMARY = {"score_function": "lac", "label_filter": "all",
                "mondrian": False}


def conformal_slice(df, calibration_source, contamination=None):
    sel = df[(df["calibration_source"] == calibration_source)
             & (df["score_function"] == CONF_PRIMARY["score_function"])
             & (df["label_filter"] == CONF_PRIMARY["label_filter"])
             & (~df["mondrian"].astype(bool))]
    if contamination is not None:
        sel = sel[sel["contamination"] == contamination]
    return sel.sort_values(["tier", "alpha"], ascending=[True, False])


# ---------------------------------------------------------------------------
# T7 -- Finding 1, and Phase 6B's weighted repair of it
# ---------------------------------------------------------------------------
@table("T7", "Finding 1: split-conformal coverage transfer by representation, "
             "with the Phase 6B weighted arm",
       ["data/conformal_calibration_results.csv",
        "data/weighted_conformal_results.csv"],
       "python -m src.experiments.calibrate_conformal; "
       "python src/experiments/run_weighted_conformal.py")
def build_t7(emit):
    conf_path = src("conformal_calibration_results.csv")
    conf = pd.read_csv(conf_path)
    sel = conformal_slice(conf, "in_domain", "contaminated")

    main = sel[["tier", "alpha", "n_calibration", "nominal_coverage",
                "coverage_calibration", "coverage_benchmark", "coverage_gap",
                "coverage_sd", "mean_set_size", "singleton_rate",
                "empty_rate", "full_rate"]].copy()
    main["noise_band_2sd"] = 2.0 * main["coverage_sd"]
    main["gap_outside_band"] = (main["coverage_gap"].abs()
                                > main["noise_band_2sd"])
    main["gap_in_sd"] = main["coverage_gap"] / main["coverage_sd"]
    main = main.reset_index(drop=True)

    w_path = src("weighted_conformal_results.csv")
    w = pd.read_csv(w_path)
    weighted = w[["space", "tier", "alpha", "weighting", "domain_auc",
                  "n_eff", "coverage_benchmark", "coverage_gap",
                  "noise_band_2sd", "gap_outside_band", "delta_vs_unweighted",
                  "delta_outside_band", "recovery_fraction_post_hoc",
                  "degenerate_reason", "is_primary", "post_hoc"]].copy()
    weighted["blocked"] = weighted["degenerate_reason"].notna()
    weighted = weighted.sort_values(
        ["space", "tier", "alpha", "weighting"],
        kind="mergesort").reset_index(drop=True)

    alpha = float(settings.conformal.alpha)
    t1 = main[(main.tier == "tier1") & (main.alpha == alpha)].iloc[0]
    t2 = main[(main.tier == "tier2") & (main.alpha == alpha)].iloc[0]

    emit("T7.alpha", alpha, "src/agent/config.py",
         note="settings.conformal.enabled is False -- measurement only, and "
              "it gates nothing in production.")
    emit("T7.tier1.coverage_gap", float(t1["coverage_gap"]), rel(conf_path),
         group="finding1",
         note="TF-IDF loses this much coverage moving from the calibration "
              "set to the benchmark.")
    emit("T7.tier2.coverage_gap", float(t2["coverage_gap"]), rel(conf_path),
         group="finding1")
    emit("T7.noise_band_2sd", float(t1["noise_band_2sd"]), rel(conf_path),
         group="finding1",
         note="+/-2 s.d. on the calibration draw at n=175.")
    emit("T7.tier1.gap_in_sd", float(t1["gap_in_sd"]), rel(conf_path),
         group="finding1")
    emit("T7.tier2.gap_in_sd", float(t2["gap_in_sd"]), rel(conf_path),
         group="finding1",
         band=float(t2["noise_band_2sd"]), diff=float(t2["coverage_gap"]))
    emit("T7.tier1.mean_set_size", float(t1["mean_set_size"]), rel(conf_path))
    emit("T7.tier2.mean_set_size", float(t2["mean_set_size"]), rel(conf_path))
    emit("T7.tier1.singleton_rate", float(t1["singleton_rate"]), rel(conf_path))
    emit("T7.tier2.singleton_rate", float(t2["singleton_rate"]), rel(conf_path))

    # ---- 6B: the weighted arm. Two readings, both labelled. ----
    wt1 = weighted[(weighted.space == "tfidf") & (weighted.tier == "tier1")
                   & (weighted.alpha == alpha)
                   & (weighted.weighting == "weighted")
                   & (weighted["is_primary"].astype(bool))]
    if wt1.empty:
        raise AssertionError(
            "No PRIMARY TF-IDF-space weighted tier1 row at the configured "
            f"alpha in {rel(w_path)} -- 6B's headline cannot be emitted. The "
            "row must be marked is_primary by the experiment, never chosen "
            "here.")
    if wt1["coverage_gap"].nunique() != 1:
        raise AssertionError(
            "6B's primary rows disagree on coverage_gap; the headline is "
            "ambiguous and must not be published.")
    wt1 = wt1.iloc[0]
    emit("T7.weighted.tier1.coverage_gap", float(wt1["coverage_gap"]),
         rel(w_path), group="6B",
         note="A PARTIAL REPAIR. Never 'the shift is correctable by "
              "covariate reweighting'.")
    emit("T7.weighted.tier1.delta_vs_unweighted",
         float(wt1["delta_vs_unweighted"]), rel(w_path), group="6B",
         note="READING 1 of the pre-registered clause: the CHANGE cleared "
              f"the +/-2 s.d. band ({format_value(wt1['delta_outside_band'])}).")
    emit("T7.weighted.tier1.residual_in_sd",
         float(wt1["coverage_gap"]) / float(t1["coverage_sd"]), rel(w_path),
         group="6B",
         note="READING 2 of the same clause: the RESIDUAL is still this many "
              "s.d. below nominal. Both readings are reported together.")

    bge_blocked = weighted[(weighted.space == "bge")
                           & weighted["degenerate_reason"].notna()]
    if bge_blocked.empty:
        raise AssertionError(
            "6B's BGE arm is expected to be blocked by its degeneracy rule; "
            f"no blocked BGE row found in {rel(w_path)}.")
    emit("T7.weighted.bge.domain_auc",
         float(bge_blocked["domain_auc"].iloc[0]), rel(w_path),
         tags=[TAG_BLOCKED, TAG_DEGENERATE], group="6B",
         note="The density ratio is ill-posed in this space. The BGE arm's "
              "flattering numbers are in the CSV as BLOCKED, not as a "
              "result. The 0.9908 is itself a statement of the named finding.")

    return {"finding1": main, "weighted_6b": weighted}


# ---------------------------------------------------------------------------
# T8 -- Finding 2, with the Phase 5A correction
# ---------------------------------------------------------------------------
@table("T8", "Finding 2: template-level contamination of the calibration set "
             "(Phase 5A corrected diagnostic)",
       ["data/conformal_calibration_corrected_bge-base-en-v1-5.json",
        "data/conformal_calibration_results.csv"],
       "python -m src.experiments.calibrate_conformal")
def build_t8(emit):
    json_path = src("conformal_calibration_corrected_bge-base-en-v1-5.json")
    payload = read_json(json_path)
    cs = payload["contamination_structure"]

    diag = pd.DataFrame([
        {"quantity": "template key", "corrected_value": cs["template_key"],
         "superseded_value": "scenario_id alone"},
        {"quantity": "templates in the dataset",
         "corrected_value": cs["total_templates"], "superseded_value": 12},
        {"quantity": "templates touched by the 175 calibration tickets",
         "corrected_value": cs["calibration_templates"],
         "superseded_value": 11},
        {"quantity": "median rows per template",
         "corrected_value": cs["median_rows_per_template"],
         "superseded_value": "~430"},
        {"quantity": "rows surviving whole-template exclusion",
         "corrected_value": cs["rows_surviving_template_exclusion"],
         "superseded_value": 40},
        {"quantity": "dataset rows", "corrected_value": cs["dataset_rows"],
         "superseded_value": cs["dataset_rows"]},
    ])

    conf_path = src("conformal_calibration_results.csv")
    conf = pd.read_csv(conf_path)
    clean = conformal_slice(conf, "in_domain", "clean")
    dirty = conformal_slice(conf, "in_domain", "contaminated")
    merged = dirty.merge(
        clean, on=["tier", "alpha"], suffixes=("_contaminated", "_clean"))
    movement = merged[["tier", "alpha", "coverage_benchmark_contaminated",
                       "coverage_benchmark_clean",
                       "coverage_gap_contaminated", "coverage_gap_clean"]
                      ].copy()
    movement["gap_movement"] = (movement["coverage_gap_clean"]
                                - movement["coverage_gap_contaminated"])
    movement["noise_band_2sd"] = 2.0 * merged["coverage_sd_contaminated"]
    movement = movement.sort_values(["tier", "alpha"], ascending=[True, False]
                                    ).reset_index(drop=True)
    max_move = float(movement["gap_movement"].abs().max())

    emit("T8.templates_total", int(cs["total_templates"]), rel(json_path),
         group="finding2",
         note="A template is (category, scenario_id). scenario_id alone is "
              "unique only WITHIN a category.")
    emit("T8.templates_touched", int(cs["calibration_templates"]),
         rel(json_path), group="finding2")
    emit("T8.median_rows_per_template", int(cs["median_rows_per_template"]),
         rel(json_path), group="finding2")
    emit("T8.rows_surviving_template_exclusion",
         int(cs["rows_surviving_template_exclusion"]), rel(json_path),
         group="finding2",
         note="Removing whole templates would leave this many of "
              f"{cs['dataset_rows']} rows -- the set cannot be "
              "de-contaminated by filtering.")
    emit("T8.dataset_rows", int(cs["dataset_rows"]), rel(json_path),
         group="finding2")
    emit("T8.max_gap_movement_clean_vs_contaminated", max_move, rel(conf_path),
         group="finding2",
         note="Excluding contaminated rows moves the coverage gap by at most "
              "this much anywhere on the grid -- Finding 2's conclusion is "
              "unchanged by the 5A correction, and that measurement excludes "
              "by row id, not by template.")

    return {"diagnostic": diag, "coverage_movement": movement}


# ---------------------------------------------------------------------------
# T9 -- Finding 3: conformal p-values as a novelty detector
# ---------------------------------------------------------------------------
@table("T9", "Finding 3: conformal p-values detect novelty that the "
             "similarity gate does not",
       ["data/conformal_novelty_results.csv"],
       "python -m src.experiments.calibrate_conformal")
def build_t9(emit):
    path = src("conformal_novelty_results.csv")
    df = pd.read_csv(path).sort_values(
        ["contamination", "alpha"], ascending=[True, False]).reset_index(
            drop=True)

    alpha = float(settings.conformal.alpha)
    row = df[(df.alpha == alpha) & (df.contamination == "contaminated")].iloc[0]

    emit("T9.false_escalation_in_domain",
         float(row["false_escalation_rate_in_domain"]), rel(path),
         note="Tracks alpha by construction -- the guarantee is marginal "
              "over the calibration draw, not conditional on it.")
    emit("T9.ood_detection_seed_level", float(row["ood_detection_rate_seeds"]),
         rel(path))
    emit("T9.ood_detection_variant_level",
         float(row["ood_detection_rate_variants"]), rel(path))
    emit("T9.adversarial_flagged",
         frac(int(row["adversarial_flagged"]), int(row["adversarial_total"])),
         rel(path))
    emit("T9.min_attainable_p", float(row["min_attainable_p"]), rel(path),
         note="1/(n_cal+1) at n=175. No p-value can go below it, which is "
              "why the KS test reads discreteness rather than drift.")

    return {"main": df}


# ---------------------------------------------------------------------------
# T10 -- Finding 4: in-domain vs deployment-distribution calibration
# ---------------------------------------------------------------------------
@table("T10", "Finding 4: the calibration distribution, not the method, is "
              "the binding constraint",
       ["data/conformal_calibration_results.csv"],
       "python -m src.experiments.calibrate_conformal")
def build_t10(emit):
    path = src("conformal_calibration_results.csv")
    conf = pd.read_csv(path)

    in_dom = conformal_slice(conf, "in_domain", "contaminated")
    deploy = conformal_slice(conf, "deployment")
    if deploy.empty:
        raise AssertionError(
            f"No deployment-source conformal rows in {rel(path)} -- "
            "Finding 4 cannot be tabulated.")

    cols = ["tier", "alpha", "n_calibration", "coverage_calibration",
            "coverage_benchmark", "coverage_gap", "coverage_sd",
            "mean_set_size", "singleton_rate"]
    merged = in_dom[cols].merge(deploy[cols],
                                on=["tier", "alpha"],
                                suffixes=("_in_domain", "_deployment"))
    merged["gap_improvement"] = (merged["coverage_gap_deployment"].abs()
                                 - merged["coverage_gap_in_domain"].abs())
    merged = merged.sort_values(["tier", "alpha"], ascending=[True, False]
                                ).reset_index(drop=True)

    alpha = float(settings.conformal.alpha)
    t1 = merged[(merged.tier == "tier1") & (merged.alpha == alpha)].iloc[0]

    emit("T10.tier1.gap_in_domain", float(t1["coverage_gap_in_domain"]),
         rel(path), group="named-finding")
    emit("T10.tier1.gap_deployment", float(t1["coverage_gap_deployment"]),
         rel(path), group="named-finding",
         note="Same method, same alpha, same benchmark -- only the "
              "calibration DISTRIBUTION changed.")
    emit("T10.deployment.n_calibration",
         int(t1["n_calibration_deployment"]), rel(path))

    return {"main": merged}


# ---------------------------------------------------------------------------
# T11 -- external validity: 7A profile, 7B replication, 7C paraphrase shift
# ---------------------------------------------------------------------------
@table("T11", "External validity (Phases 7A-7C): profile, version shift, "
              "paraphrase shift",
       ["data/external_tobibueck/profile_summary.json",
        "data/external_tobibueck/external_conformal_designA.csv",
        "data/external_tobibueck/external_conformal_designB.csv",
        "data/external_tobibueck/external_paraphrase_conformal.csv",
        "data/external_tobibueck/external_paraphrase_summary.json",
        "data/external_tobibueck/external_paraphrase_accuracy_did.json",
        "data/external_tobibueck/external_contamination.json",
        "data/external_tobibueck/external_label_noise.json"],
       "python src/experiments/profile_external_dataset.py; "
       "python src/experiments/run_external_conformal_shift.py; "
       "python src/experiments/run_paraphrase_shift_conformal.py")
def build_t11(emit):
    prof_path = src("external_tobibueck", "profile_summary.json")
    prof = read_json(prof_path)
    nd = prof["near_duplicates_bge"]
    ctrl = prof["near_duplicates_bge_our_corpus_control"]

    profile = pd.DataFrame([
        {"quantity": "rows parsed (all languages)",
         "external": prof["rows_parsed_total"], "our_corpus": "n/a"},
        {"quantity": "English rows used",
         "external": prof["rows_english_total"], "our_corpus": ctrl["n_rows"]},
        {"quantity": "multiple of the 45-ticket benchmark",
         "external": prof["benchmark45_multiple"], "our_corpus": "n/a"},
        {"quantity": "distinct texts after exact de-duplication",
         "external": prof["exact_duplicates"]["n_unique_texts"],
         "our_corpus": "n/a"},
        {"quantity": "BGE >= 0.95 near-duplicate rate",
         "external": nd["near_duplicate_rate"],
         "our_corpus": ctrl["near_duplicate_rate"]},
        {"quantity": "queues meeting the 300-row minimum",
         "external": frac(prof["queues_meeting_min_rows"],
                          prof["queues_total"]), "our_corpus": "n/a"},
    ])

    a_path = src("external_tobibueck", "external_conformal_designA.csv")
    design_a = pd.read_csv(a_path).sort_values(
        ["test_arm", "tier", "alpha"], ascending=[True, True, False]
    ).reset_index(drop=True)

    b_path = src("external_tobibueck", "external_conformal_designB.csv")
    design_b_raw = pd.read_csv(b_path)
    design_b = (design_b_raw.groupby(["variant", "status"])
                .size().reset_index(name="n_rows")
                .sort_values(["variant", "status"]).reset_index(drop=True))

    para_path = src("external_tobibueck", "external_paraphrase_conformal.csv")
    para = pd.read_csv(para_path).sort_values(
        ["arm", "tier", "alpha"], ascending=[True, True, False]
    ).reset_index(drop=True)
    psum_path = src("external_tobibueck", "external_paraphrase_summary.json")
    psum = read_json(psum_path)
    did_path = src("external_tobibueck",
                   "external_paraphrase_accuracy_did.json")
    did = read_json(did_path)
    contam_path = src("external_tobibueck", "external_contamination.json")
    contam = read_json(contam_path)
    noise_path = src("external_tobibueck", "external_label_noise.json")
    noise = read_json(noise_path)

    manip = psum["manipulation"]
    manipulation = pd.DataFrame([
        {"quantity": "TF-IDF mean cosine, original vs paraphrase",
         "value": manip["tfidf_mean_cosine"]},
        {"quantity": "BGE mean cosine, original vs paraphrase",
         "value": manip["bge_mean_cosine"]},
        {"quantity": "TF-IDF-space domain AUC (the blocking statistic)",
         "value": manip["tfidf_domain_auc"]},
        {"quantity": "BGE-space domain AUC", "value": manip["bge_domain_auc"]},
        {"quantity": "pre-registered blocking floor",
         "value": manip["floor_auc"]},
        {"quantity": "7B version-shift BGE AUC, for reference",
         "value": manip["reference_7b_version_shift_bge_auc"]},
    ])

    alpha = float(settings.conformal.alpha)
    a_t1 = design_a[(design_a.tier == "tier1") & (design_a.alpha == alpha)
                    & (design_a.test_arm == "full")].iloc[0]
    a_t2 = design_a[(design_a.tier == "tier2") & (design_a.alpha == alpha)
                    & (design_a.test_arm == "full")].iloc[0]

    emit("T11.7a.rows_english", int(prof["rows_english_total"]),
         rel(prof_path))
    emit("T11.7a.near_duplicate_rate_external",
         float(nd["near_duplicate_rate"]), rel(prof_path), group="7A-control")
    emit("T11.7a.near_duplicate_rate_our_corpus",
         float(ctrl["near_duplicate_rate"]), rel(prof_path),
         group="7A-control",
         note="OUR corpus is MORE near-duplicated than the external one. A "
              "redundancy rate is meaningless without saying what it is high "
              "relative to.")
    emit("T11.7a.distinct_texts",
         int(prof["exact_duplicates"]["n_unique_texts"]), rel(prof_path))
    emit("T11.7a.is_real_production_data",
         "false -- independently generated data, a different generator, not "
         "reality", rel(prof_path))

    emit("T11.7b.tier1.coverage_gap", float(a_t1["coverage_gap"]), rel(a_path),
         group="finding1",
         note="Under a VERSION shift, Tier-1 transfers as well as Tier-2.")
    emit("T11.7b.tier2.coverage_gap", float(a_t2["coverage_gap"]), rel(a_path),
         group="finding1",
         band=float(a_t2["noise_band_2sd"]),
         diff=float(a_t1["coverage_gap"]) - float(a_t2["coverage_gap"]))
    emit("T11.7b.noise_band_2sd", float(a_t1["noise_band_2sd"]), rel(a_path),
         group="finding1")
    emit("T11.7b.n_test", int(a_t1["n_test"]), rel(a_path))
    emit("T11.7b.n_calibration", int(a_t1["n_calibration"]), rel(a_path))
    emit("T11.7b.domain_auc_operating",
         float(a_t1["domain_auc_operating"]), rel(a_path),
         note="Replaces the two uncommitted 7A figures -- see the "
              "do-not-cite list.")
    emit("T11.7b.rows_inside_band",
         frac(int((~design_a[design_a.test_arm == "full"]["gap_outside_band"]
                   .astype(bool)).sum()),
              int(len(design_a[design_a.test_arm == "full"]))),
         rel(a_path), group="finding1")
    emit("T11.7b.boundary_contamination_rate",
         float(contam["near_neighbour_rate"]), rel(contam_path),
         note=f"Low because the reference set is "
              f"{contam['n_reference']} rows, not because the corpus changed.")
    emit("T11.7b.label_ceiling_tier1",
         float(noise["tier1"]["accuracy_test"]), rel(noise_path),
         note="Generator-assigned, UNAUDITED labels. Tier-1 and Tier-2 are "
              "only ~2.6 points apart here, against 35.6 on our corpus -- "
              "there is little representation gap for 7B to find, which is a "
              "limitation of 7B as evidence, not a rescue of Finding 1.")
    emit("T11.7b.label_ceiling_tier2",
         float(noise["tier2"]["accuracy_test"]), rel(noise_path))

    n_blocked = int((design_b_raw["status"] == "BLOCKED_AUC_GE_0.95").sum())
    blocked_queues = sorted(set(
        design_b_raw[design_b_raw["status"] == "BLOCKED_AUC_GE_0.95"]
        ["held_out_queue"].dropna()))
    emit("T11.7b.designB.blocked_queue", "; ".join(blocked_queues), rel(b_path),
         tags=[TAG_BLOCKED, TAG_NO_RESOLUTION],
         note="Design B resolved nothing, twice, for two named reasons; no "
              "Design B number may be a headline.")
    emit("T11.7b.designB.single_class_rows",
         int((design_b_raw["status"] == "single_class_test_arm").sum()),
         rel(b_path), tags=[TAG_NO_RESOLUTION])
    emit("T11.7b.designB.blocked_rows", n_blocked, rel(b_path),
         tags=[TAG_BLOCKED, TAG_NO_RESOLUTION])

    emit("T11.7c.verdict", psum["verdict"], rel(psum_path),
         tags=[TAG_BLOCKED], group="finding1",
         note="The pre-registered degeneracy rule fired, so NO verdict is "
              "drawn on the primary. A blocked arm is not a null.")
    emit("T11.7c.tfidf_domain_auc", float(manip["tfidf_domain_auc"]),
         rel(psum_path), tags=[TAG_BLOCKED], group="finding1")
    emit("T11.7c.tfidf_mean_cosine", float(manip["tfidf_mean_cosine"]),
         rel(psum_path),
         note="The manipulation WORKED -- a near-1.0 AUC here means it was "
              "strong, which is why the rule was mis-specified for this "
              "design. It was kept anyway; see FRAMING.md.")
    emit("T11.7c.bge_mean_cosine", float(manip["bge_mean_cosine"]),
         rel(psum_path))
    emit("T11.7c.n_pairs", int(psum["n_pairs"]), rel(psum_path))
    emit("T11.7c.accuracy.tier1_change", float(did["tier1_accuracy_change"]),
         rel(did_path), tags=[TAG_POST_HOC], group="finding1",
         note="The mechanism IS visible in accuracy even though the "
              "coverage primary is blocked.")
    emit("T11.7c.accuracy.tier2_change", float(did["tier2_accuracy_change"]),
         rel(did_path), tags=[TAG_POST_HOC], group="finding1")
    emit("T11.7c.did_point", float(did["did_point"]), rel(did_path),
         tags=[TAG_POST_HOC], group="finding1",
         ci=(did["did_ci_lo"], did["did_ci_hi"]),
         note="10,000 paired bootstrap draws, seed 42. Excludes zero. Still "
              "POST-HOC -- it does not replace the blocked primary.")

    # 7C reports a COMBINED band: at n=286 the test-sampling term dominates.
    p_t1 = para[(para.arm == "paraphrased") & (para.tier == "tier1")
                & (para.alpha == alpha)].iloc[0]
    emit("T11.7c.calibration_only_band_2sd",
         float(p_t1["calibration_only_band_2sd"]), rel(para_path),
         note="Quoting this alone on a 286-ticket test arm understates "
              "uncertainty by roughly 2x.")
    emit("T11.7c.combined_band_2sd", float(p_t1["combined_band_2sd"]),
         rel(para_path),
         note="Calibration term AND test-sampling term. This is the band 7C "
              "reports.")
    emit("T11.7c.tier1.coverage_gap_posthoc", float(p_t1["coverage_gap"]),
         rel(para_path), tags=[TAG_POST_HOC, TAG_BLOCKED], group="finding1",
         note="POST-HOC only. Never quote 7C coverage as a verdict.")

    return {"profile_7a": profile, "design_a_7b": design_a,
            "design_b_7b": design_b, "manipulation_7c": manipulation,
            "paraphrase_conformal_7c": para}


# ---------------------------------------------------------------------------
# T12 -- deferral rules, ours (6A) and external (7B)
# ---------------------------------------------------------------------------
@table("T12", "Conformal deferral vs the confidence incumbent, at the live "
              "gate and by AURC",
       ["data/deferral_rule_summary.csv",
        "data/deferral_conformal_operating_points.csv",
        "data/external_tobibueck/external_deferral_results.csv"],
       "python src/experiments/compare_deferral_rules.py; "
       "python src/experiments/compare_deferral_rules_external.py")
def build_t12(emit):
    path = src("deferral_rule_summary.csv")
    df = pd.read_csv(path)
    ours = df[["tier", "eval_set", "rule", "n", "live_gate_coverage",
               "degenerate_at_gate", "d_risk_at_live_gate",
               "d_risk_at_live_ci_low", "d_risk_at_live_ci_high",
               "verdict_at_live_gate", "aurc", "d_aurc_vs_confidence",
               "d_aurc_ci_low", "d_aurc_ci_high"]].copy()
    ours["aurc_is_headline"] = False
    ours = ours.sort_values(["tier", "eval_set", "rule"],
                            kind="mergesort").reset_index(drop=True)

    ext_path = src("external_tobibueck", "external_deferral_results.csv")
    ext = pd.read_csv(ext_path)
    ext_out = ext[["tier", "rule", "n_test", "accuracy", "live_gate_coverage",
                   "coverage_used", "risk_at_gate", "degenerate_at_gate",
                   "is_incumbent", "risk_delta_vs_incumbent", "risk_signal",
                   "aurc", "aurc_delta_vs_incumbent", "aurc_signal"]].copy()
    ext_out = ext_out.sort_values(["tier", "rule"],
                                  kind="mergesort").reset_index(drop=True)

    comparisons = ours[ours["d_risk_at_live_gate"].notna()]
    n_signal = int((comparisons["verdict_at_live_gate"].astype(str)
                    != "no signal on the gated axis").sum())
    n_degenerate = int(ours["degenerate_at_gate"].astype(str).str.lower()
                       .eq("yes").sum())
    n_configs = int(ours.groupby(["tier", "eval_set"]).ngroups)
    degenerate_configs = int(
        ours[ours["degenerate_at_gate"].astype(str).str.lower() == "yes"]
        .groupby(["tier", "eval_set"]).ngroups)

    # "yes" is the only value that means a signal; "no" is a resolved
    # comparison that found none, and NaN is the incumbent's own row.
    ext_signals = ext_out[ext_out["risk_signal"].astype(str).str.lower()
                          == "yes"]
    ext_resolved = ext_out[ext_out["risk_signal"].notna()]

    emit("T12.6a.comparisons_total", int(len(comparisons)), rel(path),
         group="6A")
    emit("T12.6a.comparisons_with_signal", n_signal, rel(path), group="6A",
         note="ZERO. The honest claim is 'no evidence either way on the "
              "gated axis' -- never 'conformal is worse' on our data.")
    emit("T12.6a.degenerate_configurations",
         frac(degenerate_configs, n_configs), rel(path), group="6A",
         tags=[TAG_DEGENERATE],
         note="Every rule accepts the same tickets at the live gate, so the "
              "test has no resolution there.")
    emit("T12.6a.gate_coverage.benchmark45",
         float(ours[ours.eval_set == "benchmark45"]
               ["live_gate_coverage"].iloc[0]), rel(path), group="6A")
    emit("T12.6a.gate_coverage.deployment175",
         float(ours[ours.eval_set == "deployment175"]
               ["live_gate_coverage"].iloc[0]), rel(path), group="6A")
    emit("T12.6a.degenerate_rows", n_degenerate, rel(path),
         tags=[TAG_DEGENERATE], group="6A")
    emit("T12.7b.gate_coverage", float(ext_out["live_gate_coverage"].iloc[0]),
         rel(ext_path),
         note="The gated axis RESOLVES here -- hundreds of tickets at the "
              "gate against 6A's four.")
    emit("T12.7b.n_at_gate", int(round(
        float(ext_out["live_gate_coverage"].iloc[0])
        * float(ext_out["n_test"].iloc[0]))), rel(ext_path))
    emit("T12.7b.risk_signals",
         frac(len(ext_signals), len(ext_resolved)), rel(ext_path),
         note="All favouring the confidence incumbent -- on THAT corpus, "
              "with a transplanted gate and ~35% label accuracy. This must "
              "never be merged with 6A's reading.")

    return {"ours_6a": ours, "external_7b": ext_out,
            "conformal_operating_points":
                pd.read_csv(src("deferral_conformal_operating_points.csv"))}


# ---------------------------------------------------------------------------
# T13 -- drift detection (Phase 4B-1)
# ---------------------------------------------------------------------------
@table("T13", "Drift detection: measured null false-alarm rates and the "
              "realistic-traffic arm",
       ["data/drift_evaluation_null.csv", "data/drift_evaluation_power.csv",
        "data/drift_evaluation_summary.json"],
       "python src/experiments/evaluate_drift_detection.py")
def build_t13(emit):
    null_path = src("drift_evaluation_null.csv")
    null = pd.read_csv(null_path)
    sum_path = src("drift_evaluation_summary.json")
    summary = read_json(sum_path)

    grouped = (null.groupby(["signal", "test", "window"])
               .agg(rate_min=("rate", "min"), rate_max=("rate", "max"),
                    n_points=("rate", "size"),
                    n_eligible=("eligible",
                                lambda s: int(s.astype(bool).sum())))
               .reset_index()
               .sort_values(["signal", "test", "window"])
               .reset_index(drop=True))
    grouped["usable_as_an_alarm"] = grouped["n_eligible"] > 0

    realistic = pd.DataFrame(summary["realistic_traffic_per_ticket"])

    n_eligible = len(summary["eligible_operating_points"])
    n_points = int(len(null))

    def band(test, window=None):
        sel = grouped[grouped.test == test]
        if window is not None:
            sel = sel[sel.window == window]
        return float(sel["rate_min"].min()), float(sel["rate_max"].max())

    cond_lo, cond_hi = band("conditional_binomial")
    marg_lo, marg_hi = band("marginal_binomial", 200)
    ks_lo, ks_hi = band("ks", 25)
    ks_lo2, ks_hi2 = band("ks", 200)

    emit("T13.eligible_operating_points", frac(n_eligible, n_points),
         rel(sum_path),
         note="Verdict: MEASURED, NOT SHIPPED. Nothing was promoted into "
              "config; settings.drift.enabled stays False.")
    emit("T13.conditional_binomial.null_rate_min", cond_lo, rel(null_path),
         note="Holds at every window -- the only Signal A test that does.")
    emit("T13.conditional_binomial.null_rate_max", cond_hi, rel(null_path))
    emit("T13.marginal_binomial.null_rate_at_w200_min", marg_lo,
         rel(null_path),
         note="Against a nominal 0.05 -- usable only at W <= 50.")
    emit("T13.marginal_binomial.null_rate_at_w200_max", marg_hi,
         rel(null_path))
    emit("T13.ks.null_rate_at_w25", ks_lo, rel(null_path),
         note="The KS test reads the p-values' 1/176 discreteness, not "
              "drift. Descriptive read only -- never an alarm.")
    emit("T13.ks.null_rate_at_w200", ks_hi2, rel(null_path))
    for row in summary["realistic_traffic_per_ticket"]:
        key = format_value(row.get("alpha"))
        emit(f"T13.realistic_traffic.flag_rate.alpha{key}",
             float(row.get("rate", row.get("flag_rate"))), rel(sum_path),
             note="Deployment-register tickets are LEGITIMATE, not drift. A "
                  "monitor on the in-domain reference would alarm "
                  "continuously; the binding constraint is what the "
                  "reference is made of.")

    return {"null_rates": grouped, "realistic_traffic": realistic,
            "null_raw": null}


def cohen_kappa_2x2(a, b, c, d):
    """Cohen's kappa for a 2x2 agreement table.

    a = both "positive", d = both "negative", b and c the disagreements.
    Reported ONLY for the support/groundedness axis. The hedge axis's kappa
    is deliberately never computed here -- see the do-not-cite list.
    """
    n = a + b + c + d
    po = (a + d) / n
    p_row1 = (a + b) / n
    p_col1 = (a + c) / n
    pe = p_row1 * p_col1 + (1 - p_row1) * (1 - p_col1)
    if abs(1 - pe) < 1e-15:
        return float("nan")
    return (po - pe) / (1 - pe)


# ---------------------------------------------------------------------------
# T14 -- Phase 2B: resolution groundedness, and the audited LLM judge
# ---------------------------------------------------------------------------
@table("T14", "Resolution groundedness and the audited LLM judge (Phase 2B)",
       ["data/groundedness_results.csv"],
       "python src/experiments/score_groundedness_set.py")
def build_t14(emit):
    path = src("groundedness_results.csv")
    g = pd.read_csv(path)
    n = int(len(g))

    grounded = int((g["human_label"] == "grounded").sum())
    hedge_ok = int(g["human_hedge_appropriate"].astype(bool).sum())
    judge_grounded = int((g["judge_label"] == "grounded").sum())
    agree = int(g["agreement"].astype(bool).sum())

    # The 2x2 that the aggregate agreement score conceals.
    a = int(((g["human_label"] == "grounded")
             & (g["judge_label"] == "grounded")).sum())
    b = int(((g["human_label"] == "grounded")
             & (g["judge_label"] == "ungrounded")).sum())
    c = int(((g["human_label"] == "ungrounded")
             & (g["judge_label"] == "grounded")).sum())
    d = int(((g["human_label"] == "ungrounded")
             & (g["judge_label"] == "ungrounded")).sum())
    kappa = cohen_kappa_2x2(a, b, c, d)

    # Rule 6: a second, independent derivation of the same agreement count.
    if a + d != agree:
        raise AssertionError(
            f"{rel(path)}: the agreement column says {agree} but the "
            f"human/judge cross-tabulation says {a + d}.")

    main = pd.DataFrame([
        {"quantity": "drafts judged", "value": frac(n, n)},
        {"quantity": "grounded (human label)", "value": frac(grounded, n)},
        {"quantity": "hedge appropriate (human label)",
         "value": frac(hedge_ok, n)},
        {"quantity": "grounded (LLM judge)", "value": frac(judge_grounded, n)},
        {"quantity": "raw human/judge agreement", "value": frac(agree, n)},
    ])

    confusion = pd.DataFrame([
        {"human_label": "grounded", "judge_label": "grounded", "n": a},
        {"human_label": "grounded", "judge_label": "ungrounded", "n": b},
        {"human_label": "ungrounded", "judge_label": "grounded", "n": c},
        {"human_label": "ungrounded", "judge_label": "ungrounded", "n": d},
    ])

    disagreements = g[~g["agreement"].astype(bool)][
        ["item_id", "ticket_id", "top_similarity", "human_label",
         "judge_label", "label_reason"]].reset_index(drop=True)

    emit("T14.grounded", frac(grounded, n), rel(path),
         ci=wilson_interval(grounded, n, Z_95))
    emit("T14.hedge_appropriate", frac(hedge_ok, n), rel(path),
         ci=wilson_interval(hedge_ok, n, Z_95))
    emit("T14.judge.raw_agreement", frac(agree, n), rel(path),
         note="A high raw agreement score conceals the judge's behaviour "
              "entirely; only the kappa and the individual disagreements "
              "expose it.")
    emit("T14.judge.cohens_kappa", kappa, rel(path),
         note="BELOW ZERO. Every disagreement was the judge substituting "
              "APPROPRIATENESS for SUPPORT, in both directions. A judge is an "
              "object of study here, never a scaling tool.")
    emit("T14.judge.n_disagreements", int(len(disagreements)), rel(path))
    emit("T14.context_distinct_fixes_2plus",
         frac(int((g["context_distinct_fixes"] >= 2).sum()), n), rel(path),
         note="The retrieved context is heterogeneous -- which is why the "
              "near-duplicate explanation for 6C is a REJECTED hypothesis.")

    return {"main": main, "judge_confusion": confusion,
            "disagreements": disagreements}


# ---------------------------------------------------------------------------
# T15 -- Phase 6C: the retrieval-sufficiency gate
# ---------------------------------------------------------------------------
@table("T15", "A retrieval-sufficiency check as a second gate (Phase 6C)",
       ["data/sufficiency_gate_summary.json",
        "data/sufficiency_gate_results.csv"],
       "python src/experiments/score_sufficiency_gate.py")
def build_t15(emit):
    sum_path = src("sufficiency_gate_summary.json")
    s = read_json(sum_path)
    res_path = src("sufficiency_gate_results.csv")
    res = pd.read_csv(res_path)

    pop = s["population"]
    primary = s["primary"]
    tbl = primary["two_by_two"]
    caught_k, caught_n = primary["caught_of_ungrounded"]
    flagged_k, flagged_n = primary["flagged_of_grounded"]

    # Rule 6: the 2x2 must reproduce from the per-ticket CSV, independently.
    eligible = res[res["arm"] == "eligible"] if "eligible" in set(
        res["arm"].astype(str)) else res[res["human_label"].notna()]
    recomputed = {
        "insufficient_and_ungrounded": int(
            ((eligible["gemini_verdict"] == "INSUFFICIENT")
             & (eligible["human_label"] == "ungrounded")).sum()),
        "insufficient_and_grounded": int(
            ((eligible["gemini_verdict"] == "INSUFFICIENT")
             & (eligible["human_label"] == "grounded")).sum()),
        "sufficient_and_ungrounded": int(
            ((eligible["gemini_verdict"] == "SUFFICIENT")
             & (eligible["human_label"] == "ungrounded")).sum()),
        "sufficient_and_grounded": int(
            ((eligible["gemini_verdict"] == "SUFFICIENT")
             & (eligible["human_label"] == "grounded")).sum()),
    }
    for key, value in tbl.items():
        if recomputed.get(key) != value:
            raise AssertionError(
                f"6C 2x2 cell {key} disagrees: summary says {value}, "
                f"{rel(res_path)} gives {recomputed.get(key)}.")

    gemini_2x2 = pd.DataFrame([
        {"rater_verdict": "INSUFFICIENT", "human_label": "ungrounded",
         "n": tbl["insufficient_and_ungrounded"]},
        {"rater_verdict": "INSUFFICIENT", "human_label": "grounded",
         "n": tbl["insufficient_and_grounded"]},
        {"rater_verdict": "SUFFICIENT", "human_label": "ungrounded",
         "n": tbl["sufficient_and_ungrounded"]},
        {"rater_verdict": "SUFFICIENT", "human_label": "grounded",
         "n": tbl["sufficient_and_grounded"]},
    ])

    qwen_eligible = eligible[eligible["qwen_verdict"].notna()]
    qwen_2x2 = pd.DataFrame([
        {"rater_verdict": v, "human_label": h,
         "n": int(((qwen_eligible["qwen_verdict"] == v)
                   & (qwen_eligible["human_label"] == h)).sum())}
        for v in ("INSUFFICIENT", "SUFFICIENT")
        for h in ("ungrounded", "grounded")])

    cross = s["secondary_cross_family"]
    stab = s["secondary_stability"]
    rag = s["secondary_rag_gate_agreement"]

    secondaries = pd.DataFrame([
        {"secondary": "cross-family agreement (Qwen2.5-3B vs Gemini)",
         "value": json.dumps(cross, sort_keys=True)},
        {"secondary": "stability at temperature 0",
         "value": json.dumps(stab, sort_keys=True)},
        {"secondary": "agreement with the live RAG gate",
         "value": json.dumps(rag, sort_keys=True)},
    ])

    # The pre-registered case-study axis: 2 positives. COUNTS ONLY -- the
    # emitter refuses a CI here, which is the point.
    emit("T15.caught_of_ungrounded", frac(caught_k, caught_n), rel(sum_path),
         tags=[TAG_CASE_STUDY], group="6C",
         note="Caught BOTH. Never quote this without the false-flag count "
              "beside it.")
    emit("T15.flagged_of_grounded", frac(flagged_k, flagged_n), rel(sum_path),
         group="6C",
         ci=tuple(primary["false_flag_wilson95"]),
         note="The false-flag proportion, measured against 2B's groundedness "
              "labels -- which are an OUTCOME PROXY for draft support, not a "
              "direct sufficiency label. That is 6C's binding limitation.")
    emit("T15.false_flag_proportion",
         float(primary["false_flag_proportion"]), rel(sum_path), group="6C")
    emit("T15.escalated_of_eligible",
         frac(tbl["insufficient_and_ungrounded"]
              + tbl["insufficient_and_grounded"], pop["eligible"]),
         rel(sum_path), group="6C",
         note="A gate escalating this many of the tickets that currently "
              "reach the resolver suppresses ~85% of auto-resolution to "
              "recover two bad drafts, and nothing can be tuned.")
    emit("T15.population.eligible", int(pop["eligible"]), rel(sum_path),
         group="6C")
    emit("T15.population.escalated", int(pop["escalated"]), rel(sum_path),
         group="6C")

    return {"gemini_2x2": gemini_2x2, "qwen_2x2": qwen_2x2,
            "secondaries": secondaries,
            "disagreements": pd.DataFrame(primary["disagreements"])}


# ---------------------------------------------------------------------------
# T16 -- resolution clustering and automation flagging
# ---------------------------------------------------------------------------
@table("T16", "Resolution clustering thresholds and automation flagging",
       ["data/resolution_clustering_calibration_results.csv",
        "data/resolution_clustering_calibration_results_bge-base-en-v1-5.csv",
        "data/resolution_clustering_calibration_percategory_summary.csv",
        "data/automation_candidates.json",
        "data/automation_flag_validation_pilot_results.csv"],
       "python src/experiments/calibrate_resolution_clustering.py; "
       "python src/experiments/flag_automation_candidates.py; "
       "python src/experiments/score_flag_validation_set.py --pilot")
def build_t16(emit):
    mini_path = src("resolution_clustering_calibration_results.csv")
    bge_path = src(
        "resolution_clustering_calibration_results_bge-base-en-v1-5.csv")
    percat_path = src(
        "resolution_clustering_calibration_percategory_summary.csv")
    cand_path = src("automation_candidates.json")
    pilot_path = src("automation_flag_validation_pilot_results.csv")

    def pooled_cliff(path, label):
        df = pd.read_csv(path)
        pooled = (df.groupby("threshold")
                  .agg(tp=("tp", "sum"), fp=("fp", "sum"), fn=("fn", "sum"))
                  .reset_index())
        pooled["precision"] = pooled["tp"] / (pooled["tp"] + pooled["fp"])
        pooled["recall"] = pooled["tp"] / (pooled["tp"] + pooled["fn"])
        pooled["model"] = label
        perfect = pooled[pooled["precision"] >= 1.0]
        cliff = float(perfect["threshold"].min()) if not perfect.empty \
            else float("nan")
        return pooled, cliff

    mini_pooled, mini_cliff = pooled_cliff(mini_path, "all-MiniLM-L6-v2")
    bge_pooled, bge_cliff = pooled_cliff(bge_path, "BAAI/bge-base-en-v1.5")
    pooled = pd.concat([mini_pooled, bge_pooled], ignore_index=True)
    pooled = pooled[["model", "threshold", "tp", "fp", "fn", "precision",
                     "recall"]].sort_values(
        ["model", "threshold"], ascending=[True, False]).reset_index(drop=True)

    percat = pd.read_csv(percat_path)

    cand = read_json(cand_path)
    n_by_cat = pd.read_csv(percat_path).set_index("category")["n_tickets"]
    flag_rows = []
    for category, clusters in cand.items():
        if category == "_meta":
            continue
        flagged = sum(int(cl["size"]) for cl in clusters)
        total = int(n_by_cat[category])
        flag_rows.append({"category": category,
                          "n_clusters_flagged": len(clusters),
                          "n_tickets_in_flagged_clusters": flagged,
                          "n_tickets": total,
                          "flagged_share": flagged / total})
    flags = pd.DataFrame(flag_rows).sort_values(
        "flagged_share", ascending=False).reset_index(drop=True)

    pilot = pd.read_csv(pilot_path)
    n_pairs = int(len(pilot))
    n_false_merges = int(pilot["false_merge_by"].notna().sum())

    live = float(settings.clustering.resolution_similarity_threshold)
    if mini_cliff != live:
        raise AssertionError(
            f"The pooled MiniLM cliff edge recomputes to {mini_cliff} but the "
            f"production threshold is {live}. One of them is wrong.")

    top = flags.iloc[0]
    bottom = flags.iloc[-1]

    emit("T16.minilm.pooled_cliff", mini_cliff, rel(mini_path),
         note="Recomputed from the committed per-category rows; it must "
              "equal the production threshold, and the build fails if not.")
    emit("T16.bge.pooled_cliff", bge_cliff, rel(bge_path),
         note="MEASURED but NOT PROMOTED. Phase 2A found the two "
              "indistinguishable on precision; promotion is a product "
              "decision about review-queue capacity.")
    emit("T16.production.threshold", live, "src/agent/config.py")
    emit("T16.flagged_share.max",
         f"{top['category']} {format_value(top['flagged_share'])}",
         rel(cand_path))
    emit("T16.flagged_share.min",
         f"{bottom['category']} {format_value(bottom['flagged_share'])}",
         rel(cand_path),
         note="A genuine finding about which categories have standardised "
              "fixes -- not a defect of the clustering.")
    emit("T16.pilot.false_merges", frac(n_false_merges, n_pairs),
         rel(pilot_path), tags=[TAG_CASE_STUDY],
         note="Zero false merges in the pilot. STOP verdict: the harness "
              "found nothing to separate the two configurations.")
    emit("T16.pilot.rule_of_three_upper", rule_of_three_upper(n_pairs),
         rel(pilot_path),
         note="Upper 95% bound on the false-merge rate after zero events in "
              f"{n_pairs} pairs -- the honest reading of a zero count.")

    return {"pooled_cliffs": pooled, "per_category": percat,
            "automation_flags": flags, "pilot": pilot}


# ---------------------------------------------------------------------------
# Figure style
# ---------------------------------------------------------------------------
def apply_style():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.rcParams.update({
        "font.family": "sans-serif",
        # DejaVu Sans ships with matplotlib. Naming a system font here would
        # make the rendered output depend on the machine.
        "font.sans-serif": ["DejaVu Sans"],
        "font.size": BASE_FONT_PT,
        "axes.titlesize": BASE_FONT_PT,
        "axes.labelsize": BASE_FONT_PT,
        "xtick.labelsize": TICK_FONT_PT,
        "ytick.labelsize": TICK_FONT_PT,
        "legend.fontsize": LEGEND_FONT_PT,
        "axes.prop_cycle": plt.cycler(color=OKABE_ITO),
        "axes.grid": True,
        "grid.alpha": 0.3,
        "grid.linewidth": 0.5,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "figure.dpi": FIG_DPI,
        "savefig.dpi": FIG_DPI,
        "savefig.bbox": "tight",
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
        "svg.hashsalt": "phase8a",
    })
    return plt


def save_figure(fig, base_path):
    """PDF + PNG, both with creation metadata stripped.

    A creation date in the file would make every rebuild differ, which would
    make the parity test useless on the one axis it is meant to protect.
    """
    fig.savefig(base_path + ".pdf", metadata={"CreationDate": None})
    fig.savefig(base_path + ".png",
                metadata={"Software": None, "Creation Time": None})


# ---------------------------------------------------------------------------
# F1 -- architecture
# ---------------------------------------------------------------------------
@figure("F1", "System architecture",
        ["src/agent/orchestrator.py", "src/agent/config.py"],
        "python src/experiments/build_paper_artifacts.py",
        IEEE_DOUBLE_COL_IN,
        "Sequential pipeline with agent boundaries and an HTTP surface. The "
        "agents are independently ADDRESSABLE, not independently running: "
        "POST /triage orchestrates in process. This is not a distributed or "
        "autonomous multi-agent system.")
def build_f1(emit):
    cascade_gate = float(settings.cascade.confidence_threshold)
    rag_gate = float(settings.rag.similarity_threshold)
    cluster_gate = float(settings.clustering.resolution_similarity_threshold)

    # x, y are the box centres; w and h its full width and height, in the
    # figure's own units. The geometry lives here rather than in the drawing
    # code so the _data.csv pins the diagram's structure AND its layout, and
    # the parity test fails if either moves.
    nodes = pd.DataFrame([
        {"node_id": "ticket", "label": "ticket text", "kind": "input",
         "agent": "-", "x": 1.05, "y": 6.35, "w": 1.9, "h": 0.52},
        {"node_id": "tier1", "label": "Tier 1: TF-IDF + LogReg",
         "kind": "stage", "agent": "ClassifierAgent",
         "x": 1.05, "y": 5.35, "w": 1.9, "h": 0.52},
        {"node_id": "gate1",
         "label": "Tier-1 confidence\n>= "
                  + format_value(cascade_gate) + "?",
         "kind": "gate", "agent": "orchestrator",
         "x": 1.05, "y": 4.25, "w": 1.9, "h": 0.62},
        {"node_id": "tier2", "label": "Tier 2: BGE + LogReg", "kind": "stage",
         "agent": "ClassifierAgent", "x": 3.45, "y": 4.25, "w": 1.9,
         "h": 0.52},
        {"node_id": "retrieve", "label": "FAISS top-5 retrieval",
         "kind": "stage", "agent": "RetrieverAgent",
         "x": 1.05, "y": 3.15, "w": 1.9, "h": 0.52},
        {"node_id": "gate2",
         "label": "top-1 similarity\n>= " + format_value(rag_gate) + "?",
         "kind": "gate", "agent": "orchestrator",
         "x": 1.05, "y": 2.05, "w": 1.9, "h": 0.62},
        {"node_id": "resolve",
         "label": "Gemini drafts a\ngrounded resolution",
         "kind": "stage", "agent": "ResolverAgent",
         "x": 3.45, "y": 2.05, "w": 1.9, "h": 0.62},
        {"node_id": "escalate",
         "label": "ESCALATE TO HUMAN\n(Gemini is never called)",
         "kind": "terminal", "agent": "orchestrator",
         "x": 1.05, "y": 0.95, "w": 1.9, "h": 0.62},
        {"node_id": "cluster",
         "label": "resolved tickets -> embed\nresolution -> union-find\nat "
                  + format_value(cluster_gate),
         "kind": "offline", "agent": "offline analysis",
         "x": 5.85, "y": 4.25, "w": 2.0, "h": 0.8},
        {"node_id": "flags",
         "label": "automation candidates\nfor human review",
         "kind": "offline", "agent": "offline analysis",
         "x": 5.85, "y": 2.85, "w": 2.0, "h": 0.62},
    ])

    # port: which side of each box an edge leaves from and arrives at.
    edges = pd.DataFrame([
        {"src": "ticket", "dst": "tier1", "label": "", "port": "v"},
        {"src": "tier1", "dst": "gate1", "label": "", "port": "v"},
        {"src": "gate1", "dst": "retrieve", "label": "yes: category",
         "port": "v"},
        {"src": "gate1", "dst": "tier2", "label": "no", "port": "h"},
        {"src": "tier2", "dst": "retrieve", "label": "category",
         "port": "elbow"},
        {"src": "retrieve", "dst": "gate2", "label": "", "port": "v"},
        {"src": "gate2", "dst": "resolve", "label": "yes", "port": "h"},
        {"src": "gate2", "dst": "escalate", "label": "no", "port": "v"},
        {"src": "cluster", "dst": "flags", "label": "offline", "port": "v"},
    ])

    data = nodes.copy()
    data["edges_out"] = data["node_id"].map(
        edges.groupby("src")["dst"].apply(lambda s: ";".join(sorted(s))))
    data["edges_out"] = data["edges_out"].fillna("")

    emit("F1.gate.cascade", cascade_gate, "src/agent/config.py")
    emit("F1.gate.rag", rag_gate, "src/agent/config.py")
    emit("F1.gate.clustering", cluster_gate, "src/agent/config.py")

    def draw():
        plt = apply_style()
        from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

        fig, ax = plt.subplots(figsize=(IEEE_DOUBLE_COL_IN, 3.3))
        ax.set_axis_off()
        ax.grid(False)
        ax.set_xlim(0.0, 7.0)
        ax.set_ylim(0.45, 6.85)

        style = {
            "input": dict(fc="#FFFFFF", ec="#000000", ls="--"),
            "stage": dict(fc="#DCEEF8", ec="#0072B2", ls="-"),
            "gate": dict(fc="#FDF0D5", ec="#E69F00", ls="-"),
            "terminal": dict(fc="#F6DED8", ec="#D55E00", ls="-"),
            "offline": dict(fc="#E4F1EA", ec="#009E73", ls="-"),
        }
        geom = {}
        for _, nd in nodes.iterrows():
            st = style[nd["kind"]]
            ax.add_patch(FancyBboxPatch(
                (nd["x"] - nd["w"] / 2, nd["y"] - nd["h"] / 2),
                nd["w"], nd["h"],
                boxstyle="round,pad=0.01,rounding_size=0.06",
                linewidth=0.9, **st))
            ax.text(nd["x"], nd["y"], nd["label"], ha="center", va="center",
                    fontsize=6.4, linespacing=1.4)
            geom[nd["node_id"]] = (nd["x"], nd["y"], nd["w"] / 2, nd["h"] / 2)

        def arrow(start, end, rad=0.0):
            ax.add_patch(FancyArrowPatch(
                start, end, arrowstyle="-|>", mutation_scale=7,
                linewidth=0.85, color="#333333",
                connectionstyle="arc3,rad=" + str(rad)))

        for _, ed in edges.iterrows():
            sx, sy, shw, shh = geom[ed["src"]]
            dx, dy, dhw, dhh = geom[ed["dst"]]
            if ed["port"] == "v":
                start, end = (sx, sy - shh), (dx, dy + dhh)
                arrow(start, end)
                if ed["label"]:
                    ax.text(sx + 0.08, (start[1] + end[1]) / 2, ed["label"],
                            fontsize=5.8, color="#333333", ha="left",
                            va="center")
            elif ed["port"] == "h":
                start, end = (sx + shw, sy), (dx - dhw, dy)
                arrow(start, end)
                if ed["label"]:
                    ax.text((start[0] + end[0]) / 2, sy + 0.08, ed["label"],
                            fontsize=5.8, color="#333333", ha="center",
                            va="bottom")
            else:  # elbow: leave the bottom, arrive at the right-hand side
                start, end = (sx, sy - shh), (dx + dhw, dy)
                arrow(start, end, rad=-0.3)
                if ed["label"]:
                    ax.text(sx - 0.14, dy + 0.34, ed["label"], fontsize=5.8,
                            color="#333333", ha="right", va="center")

        # The framing sentence lives in the CAPTION, not inside the axes --
        # an IEEE figure should not carry its own prose.
        ax.text(5.85, 5.15, "offline analysis path", fontsize=6.6,
                ha="center", va="center", color="#009E73", weight="bold")
        ax.text(5.85, 2.15,
                "not in the live request path;\nthe output is a human "
                "review queue",
                fontsize=5.8, ha="center", va="center", color="#555555",
                style="italic")
        fig.tight_layout()
        return fig

    return data, draw


# ---------------------------------------------------------------------------
# F2 -- Finding 1
# ---------------------------------------------------------------------------
@figure("F2", "Coverage gap by representation (Finding 1)",
        ["data/conformal_calibration_results.csv"],
        "python -m src.experiments.calibrate_conformal",
        IEEE_SINGLE_COL_IN,
        "Split-conformal coverage transfer from the 175-ticket calibration "
        "set to the 45-ticket benchmark, on OUR corpus. The shaded band is "
        "+/-2 s.d. on the calibration draw. External replication is "
        "inconclusive -- see Table T11.")
def build_f2(emit):
    path = src("conformal_calibration_results.csv")
    sel = conformal_slice(pd.read_csv(path), "in_domain", "contaminated")
    data = sel[["tier", "alpha", "coverage_gap", "coverage_sd",
                "coverage_benchmark", "nominal_coverage"]].copy()
    data["noise_band_2sd"] = 2.0 * data["coverage_sd"]
    data = data.sort_values(["tier", "alpha"]).reset_index(drop=True)

    def draw():
        plt = apply_style()
        fig, ax = plt.subplots(figsize=(IEEE_SINGLE_COL_IN, 2.5))
        labels = {"tier1": "Tier-1 (TF-IDF)", "tier2": "Tier-2 (BGE)"}
        for i, (tier, grp) in enumerate(data.groupby("tier")):
            grp = grp.sort_values("alpha")
            ax.plot(grp["alpha"], grp["coverage_gap"], marker="o",
                    markersize=3.2, linewidth=1.2, linestyle=DASHES[i],
                    color=OKABE_ITO[5 if tier == "tier2" else 6],
                    label=labels.get(tier, tier))
        band = data.groupby("alpha")["noise_band_2sd"].max().sort_index()
        ax.fill_between(band.index, -band.values, band.values,
                        color="#999999", alpha=0.22, linewidth=0,
                        label="+/-2 s.d. (calibration draw)")
        ax.axhline(0.0, color="#000000", linewidth=0.7)
        ax.set_xlabel(r"target error rate $\alpha$")
        ax.set_ylabel("coverage gap\n(benchmark - nominal)")
        ax.set_xscale("log")
        ax.set_xticks(sorted(data["alpha"].unique()))
        ax.get_xaxis().set_major_formatter(
            plt.matplotlib.ticker.ScalarFormatter())
        ax.get_xaxis().set_minor_formatter(plt.matplotlib.ticker.NullFormatter())
        ax.legend(loc="lower right", frameon=False)
        fig.tight_layout()
        return fig

    return data, draw


# ---------------------------------------------------------------------------
# F3 -- risk-coverage curves (6A)
# ---------------------------------------------------------------------------
@figure("F3", "Risk-coverage curves for four deferral rules (Phase 6A)",
        ["data/deferral_risk_coverage.csv", "data/deferral_rule_summary.csv"],
        "python src/experiments/compare_deferral_rules.py",
        IEEE_DOUBLE_COL_IN,
        "The vertical line marks the live gate's MEASURED operating "
        "coverage. AURC averages over coverages the system never runs at, so "
        "it is shown but is not the headline.")
def build_f3(emit):
    rc_path = src("deferral_risk_coverage.csv")
    rc = pd.read_csv(rc_path)
    sm_path = src("deferral_rule_summary.csv")
    sm = pd.read_csv(sm_path)
    gates = sm.groupby(["tier", "eval_set"])["live_gate_coverage"].first()
    degen = (sm.assign(d=sm["degenerate_at_gate"].astype(str).str.lower()
                       .eq("yes"))
             .groupby(["tier", "eval_set"])["d"].any())

    data = rc.sort_values(["tier", "eval_set", "rule", "coverage"],
                          kind="mergesort").reset_index(drop=True)

    def draw():
        plt = apply_style()
        panels = sorted(gates.index.tolist())
        fig, axes = plt.subplots(
            1, len(panels), figsize=(IEEE_DOUBLE_COL_IN, 2.4), sharey=True)
        axes = np.atleast_1d(axes)
        for ax, key in zip(axes, panels):
            tier, eval_set = key
            sub = data[(data.tier == tier) & (data.eval_set == eval_set)]
            for i, (rule, grp) in enumerate(sub.groupby("rule")):
                grp = grp.sort_values("coverage")
                ax.plot(grp["coverage"], grp["risk"], linewidth=1.0,
                        linestyle=DASHES[i % len(DASHES)],
                        color=SERIES_COLORS[i % len(SERIES_COLORS)],
                        label=rule)
            ax.axvline(float(gates[key]), color="#000000", linewidth=0.8,
                       linestyle=":")
            title = f"{tier}, {eval_set}"
            if bool(degen[key]):
                title += "\n(degenerate at the gate)"
                ax.axvspan(0.0, float(gates[key]), color="#999999",
                           alpha=0.12, linewidth=0, hatch="///")
            ax.set_title(title, fontsize=6.4)
            ax.set_xlabel("coverage")
        axes[0].set_ylabel("risk among accepted")
        axes[-1].legend(loc="upper left", frameon=False, ncol=2)
        fig.tight_layout()
        return fig

    return data, draw


# ---------------------------------------------------------------------------
# F4 -- reliability diagrams
# ---------------------------------------------------------------------------
@figure("F4", "Reliability diagrams for both cascade tiers",
        ["data/calibration_reliability_data.csv"],
        "python src/experiments/plot_calibration_curves.py",
        IEEE_DOUBLE_COL_IN,
        "Count-weighted binned ECE on the 500-ticket in-distribution "
        "production batch. Observed accuracy is 1.0 in EVERY bin, so both "
        "tiers are systematically UNDER-confident here and the whole ECE is "
        "the gap between confidence and a ceiling. That is a property of "
        "template-generated data, not evidence that the models are well "
        "calibrated on real traffic -- which is why both escalation gates "
        "are calibrated empirically rather than read off a probability.")
def build_f4(emit):
    path = src("calibration_reliability_data.csv")
    bins = pd.read_csv(path).sort_values(
        ["tier", "bin_lower"]).reset_index(drop=True)

    def draw():
        plt = apply_style()
        tiers = sorted(bins["tier"].unique())
        fig, axes = plt.subplots(1, len(tiers),
                                 figsize=(IEEE_DOUBLE_COL_IN, 2.5),
                                 sharey=True)
        axes = np.atleast_1d(axes)
        for ax, tier in zip(axes, tiers):
            grp = bins[bins.tier == tier]
            n_total = grp["n_tickets_in_bin"].sum()
            ece = float((grp["n_tickets_in_bin"] / n_total
                         * (grp["mean_predicted_confidence"]
                            - grp["observed_accuracy"]).abs()).sum())
            ax.plot([0, 1.06], [0, 1.06], color="#000000", linewidth=0.7,
                    linestyle="--", label="perfect calibration")
            ax.axhline(1.0, color="#D55E00", linewidth=0.7, linestyle=":")
            ax.annotate("observed accuracy = 1.0 in every bin -- a ceiling, "
                        "not calibration", (0.455, 1.048),
                        fontsize=5.6, color="#D55E00")
            ax.plot(grp["mean_predicted_confidence"],
                    grp["observed_accuracy"], marker="o", markersize=3.2,
                    linewidth=1.2, color=OKABE_ITO[5 if tier == 2 else 6],
                    label=f"Tier-{int(tier)}")
            for j, (_, r) in enumerate(grp.iterrows()):
                ax.annotate(f"n={int(r['n_tickets_in_bin'])}",
                            (r["mean_predicted_confidence"],
                             r["observed_accuracy"]),
                            textcoords="offset points",
                            xytext=(0, -11 if j % 2 == 0 else -19),
                            ha="center", fontsize=5.4, color="#555555")
            ax.set_title(
                f"Tier-{int(tier)} "
                f"({'TF-IDF' if tier == 1 else 'BGE'}), ECE = {ece:.4f}",
                fontsize=6.6)
            ax.set_xlabel("mean predicted confidence")
            ax.set_xlim(0.44, 1.06)
            ax.set_ylim(0.44, 1.10)
        axes[0].set_ylabel("observed accuracy")
        axes[0].legend(loc="lower right", frameon=False)
        fig.tight_layout()
        return fig

    return bins, draw


# ---------------------------------------------------------------------------
# F5 -- OOD leakage vs threshold
# ---------------------------------------------------------------------------
@figure("F5", "OOD leakage against the RAG similarity threshold",
        ["data/rag_similarity_calibration_combined.csv",
         "data/ablation_no-rag_results.csv"],
        "python -m src.experiments.calibrate_rag_similarity_threshold",
        IEEE_SINGLE_COL_IN,
        "The shaded span is the adversarial safe range. The gate sits inside "
        "it, and it errs in BOTH directions -- see FRAMING.md.")
def build_f5(emit):
    comb_path = src("rag_similarity_calibration_combined.csv")
    comb = pd.read_csv(comb_path)
    adv = pd.read_csv(src("ablation_no-rag_results.csv"))
    safe_low = float(adv[adv["expected_escalate"].astype(bool)]
                     ["top_similarity"].max())
    safe_high = float(adv[~adv["expected_escalate"].astype(bool)]
                      ["top_similarity"].min())
    gate = float(settings.rag.similarity_threshold)

    data = comb[["threshold", "ood_leakage_rate", "n_in_domain_proceed",
                 "n_in_domain_escalate"]].copy()
    data["in_domain_proceed_rate"] = (
        data["n_in_domain_proceed"]
        / (data["n_in_domain_proceed"] + data["n_in_domain_escalate"]))
    data = data.sort_values("threshold").reset_index(drop=True)

    def draw():
        plt = apply_style()
        fig, ax = plt.subplots(figsize=(IEEE_SINGLE_COL_IN, 2.4))
        ax.axvspan(safe_low, safe_high, color="#009E73", alpha=0.14,
                   linewidth=0, label="adversarial safe range")
        ax.plot(data["threshold"], data["ood_leakage_rate"], marker="o",
                markersize=2.8, linewidth=1.2, color=OKABE_ITO[6],
                label="OOD leakage")
        ax.plot(data["threshold"], data["in_domain_proceed_rate"],
                marker="s", markersize=2.8, linewidth=1.2,
                linestyle="--", color=OKABE_ITO[5],
                label="in-domain proceed rate")
        ax.axvline(gate, color="#000000", linewidth=0.9, linestyle=":")
        ax.annotate(f"gate = {format_value(gate)}", (gate, 0.90),
                    textcoords="offset points", xytext=(4, 0), fontsize=6.0)
        ax.set_xlabel("similarity threshold")
        ax.set_ylabel("rate")
        ax.set_xlim(0.55, 0.86)
        ax.legend(loc="lower left", frameon=False)
        fig.tight_layout()
        return fig

    return data, draw


# ---------------------------------------------------------------------------
# F6 -- drift null false-alarm rate vs window
# ---------------------------------------------------------------------------
@figure("F6", "Measured null false-alarm rate against window size",
        ["data/drift_evaluation_null.csv"],
        "python src/experiments/evaluate_drift_detection.py",
        IEEE_SINGLE_COL_IN,
        "Signal A. The marginal binomial's 'alpha by construction' holds "
        "only at small windows; the calibration-conditional test holds "
        "everywhere. The KS curve reads the p-values' discreteness, not "
        "drift.")
def build_f6(emit):
    path = src("drift_evaluation_null.csv")
    null = pd.read_csv(path)
    sig_a = null[null["signal"] == "A"]
    data = (sig_a.groupby(["test", "window"])
            .agg(rate_mean=("rate", "mean"), rate_min=("rate", "min"),
                 rate_max=("rate", "max"))
            .reset_index()
            .sort_values(["test", "window"]).reset_index(drop=True))

    def draw():
        plt = apply_style()
        fig, ax = plt.subplots(figsize=(IEEE_SINGLE_COL_IN, 2.4))
        labels = {"marginal_binomial": "marginal binomial",
                  "conditional_binomial": "calibration-conditional binomial",
                  "ks": "KS (descriptive only)"}
        for i, (test, grp) in enumerate(data.groupby("test")):
            grp = grp.sort_values("window")
            ax.plot(grp["window"], grp["rate_mean"], marker="o",
                    markersize=3.0, linewidth=1.2,
                    linestyle=DASHES[i % len(DASHES)],
                    color=SERIES_COLORS[i % len(SERIES_COLORS)],
                    label=labels.get(test, test))
            ax.fill_between(grp["window"], grp["rate_min"], grp["rate_max"],
                            color=SERIES_COLORS[i % len(SERIES_COLORS)],
                            alpha=0.15, linewidth=0)
        ax.axhline(0.05, color="#000000", linewidth=0.8, linestyle=":")
        ax.annotate("nominal 0.05", (25, 0.055), fontsize=6.0)
        ax.set_xlabel("window size (tickets)")
        ax.set_ylabel("null false-alarm rate")
        ax.set_xscale("log")
        ax.set_xticks(sorted(data["window"].unique()))
        ax.get_xaxis().set_major_formatter(
            plt.matplotlib.ticker.ScalarFormatter())
        ax.get_xaxis().set_minor_formatter(plt.matplotlib.ticker.NullFormatter())
        ax.legend(loc="upper left", frameon=False)
        fig.tight_layout()
        return fig

    return data, draw


# ---------------------------------------------------------------------------
# F7 -- accuracy under paraphrase shift (7C, post-hoc)
# ---------------------------------------------------------------------------
@figure("F7", "Accuracy under a paraphrase shift (Phase 7C, POST-HOC)",
        ["data/external_tobibueck/external_paraphrase_accuracy_did.json"],
        "python src/experiments/run_paraphrase_shift_conformal.py",
        IEEE_SINGLE_COL_IN,
        "POST-HOC. The pre-registered coverage primary was BLOCKED by its "
        "own degeneracy rule; this accuracy contrast does not replace it.")
def build_f7(emit):
    path = src("external_tobibueck", "external_paraphrase_accuracy_did.json")
    did = read_json(path)
    acc = did["accuracy"]

    data = pd.DataFrame([
        {"tier": "tier1", "arm": "original",
         "accuracy": acc["original_tier1"]},
        {"tier": "tier1", "arm": "paraphrased",
         "accuracy": acc["paraphrased_tier1"]},
        {"tier": "tier2", "arm": "original",
         "accuracy": acc["original_tier2"]},
        {"tier": "tier2", "arm": "paraphrased",
         "accuracy": acc["paraphrased_tier2"]},
    ])
    data["n_pairs"] = did["n_pairs"]
    data["post_hoc"] = True

    def draw():
        plt = apply_style()
        fig, ax = plt.subplots(figsize=(IEEE_SINGLE_COL_IN, 2.4))
        labels = {"tier1": "Tier-1 (TF-IDF)", "tier2": "Tier-2 (BGE)"}
        for i, tier in enumerate(["tier1", "tier2"]):
            sub = data[data.tier == tier]
            ax.plot([0, 1], sub["accuracy"].tolist(), marker="o",
                    markersize=4.0, linewidth=1.4,
                    linestyle=DASHES[i],
                    color=OKABE_ITO[6 if tier == "tier1" else 5],
                    label=labels[tier])
            ax.annotate(
                f"{did[f'{tier}_accuracy_change']:+.4f}",
                (1, sub["accuracy"].iloc[1]), textcoords="offset points",
                xytext=(5, -3), fontsize=6.0,
                color=OKABE_ITO[6 if tier == "tier1" else 5])
        ax.set_xticks([0, 1])
        ax.set_xticklabels(["original", "paraphrased"])
        ax.set_ylabel(f"accuracy (n = {did['n_pairs']} pairs)")
        ax.set_xlim(-0.15, 1.45)
        ax.set_title(
            f"DiD {did['did_point']:.4f} "
            f"[{did['did_ci_lo']:.4f}, {did['did_ci_hi']:.4f}]  POST-HOC",
            fontsize=6.4)
        ax.legend(loc="lower left", frameon=False)
        fig.tight_layout()
        return fig

    return data, draw


# ---------------------------------------------------------------------------
# Clause groups -- numbers that may never be quoted without their clause
# ---------------------------------------------------------------------------
CLAUSES = {
    "finding1": (
        "FINDING 1 -- the three clauses travel together, and none is quoted "
        "alone:\n"
        "  (1) MEASURED ON OUR CORPUS: split-conformal coverage transfers "
        "for the BGE representation but not the lexical one.\n"
        "  (2) EXTERNAL REPLICATION IS INCONCLUSIVE: Phase 7B applied a "
        "VERSION shift -- the wrong shift type for a mechanism that is "
        "surface-vocabulary change -- and Phase 7C applied the right type "
        "but was BLOCKED by its own pre-registered degeneracy rule. After "
        "7B and 7C, Finding 1 has been shown neither to hold nor to fail "
        "outside its original corpus. Never write '7B refutes Finding 1'.\n"
        "  (3) POST-HOC EVIDENCE OF THE MECHANISM IN ACCURACY on the "
        "external corpus: difference-in-differences -0.0804, 95% CI "
        "[-0.1364, -0.0210]. Post-hoc, and it does not replace the blocked "
        "primary."),
    "finding2": (
        "FINDING 2 -- the 175-ticket calibration set cannot be "
        "de-contaminated. Memorisation is TEMPLATE-level, and a template is "
        "(category, scenario_id), never scenario_id alone. The pre-Phase-5A "
        "diagnostic is superseded and appears only in the do-not-cite list; "
        "Finding 2's conclusion survived the correction because the coverage "
        "measurement excludes by row id, not by template."),
    "named-finding": (
        "THE NAMED CROSS-PHASE FINDING -- the calibration/reference "
        "distribution, not the test or the method, is the binding "
        "constraint. Instances: Phase 1 Finding 4 (coverage), Phase 4B's "
        "realistic-traffic arm (drift), Phase 5C (classification, which "
        "widened it to the TRAINING distribution as well). Phase 2A is a "
        "RELATED corpus limitation, not an instance. Phase 6C is NOT an "
        "instance. Phase 9A settles the final wording."),
    "5C": (
        "PHASE 5C -- the correct sentence is: a 3B model on a laptop CPU "
        "with no training on this corpus is not beaten by a classifier "
        "fitted on 3,200 of its rows, across two vendors. NEVER 'LLMs beat "
        "the pipeline'. It is zero-shot LLM vs trained classifier, which is "
        "not like-for-like, and what it measures is what a template-"
        "generated corpus is worth on out-of-template phrasing. The "
        "readings differ by pair and must not be merged: Gemini is "
        "distinguishably better than Tier-2 on the 45; Qwen2.5-3B is "
        "INDISTINGUISHABLE from it, which is not 'better'. The benchmark is "
        "Gemini-generated, so the Qwen arm is the partial control for "
        "authorship."),
    "6A": (
        "PHASE 6A -- the honest claim is 'no evidence either way on the "
        "gated axis', NEVER 'conformal is worse' on our data. All twelve "
        "comparisons returned no signal at the live gate's measured "
        "operating coverage, and three of four configurations are "
        "degenerate there. An AURC win is never a reason to promote: this "
        "project gates on risk at the operating coverage, and AURC averages "
        "over coverages the system never runs at."),
    "6B": (
        "PHASE 6B -- weighted conformal is a PARTIAL REPAIR, NOT A "
        "CORRECTION. Never write 'the shift is correctable by covariate "
        "reweighting'. Both readings are reported together: the CHANGE "
        "cleared the +/-2 s.d. band, and the RESIDUAL is still 5-6 s.d. "
        "below nominal. No ordered comparison between reweighting and "
        "distribution matching -- the residuals differ by less than the "
        "band. The BGE arm is BLOCKED, not a result."),
    "6C": (
        "PHASE 6C -- the headline is 'catches both, and is still not usable "
        "as a gate', never 'the sufficiency check caught the two cases the "
        "gate missed', which is true and misleading alone. 6C is NOT an "
        "instance of the named finding; the near-duplicate explanation is a "
        "REJECTED hypothesis, refuted by 2B's own distinct-fix diagnostic, "
        "and the mechanism is a MEASUREMENT-TARGET MISMATCH between "
        "sufficiency and groundedness. The binding limitation is the LABEL, "
        "not the rater. The N45 counter-case travels with the result: the "
        "scalar errs in both directions."),
    "7A-control": (
        "PHASE 7A's CONTROL LESSON -- never quote a redundancy or "
        "similarity rate without stating what it is high RELATIVE TO. Our "
        "own corpus is more near-duplicated than the external one. High "
        "near-duplication is a property of template-generated corpora "
        "generally, not a flaw of that dataset."),
}


# ---------------------------------------------------------------------------
# Do-not-cite list
# ---------------------------------------------------------------------------
DO_NOT_CITE = [
    {"retired": "12 templates / ~430 rows per template / 11 touched / 40 "
                "surviving rows",
     "literals": ["12 templates", "430 rows", "40/4000", "40 of 4000"],
     "why": "The superseded Finding 2 diagnostic. It grouped by scenario_id "
            "alone, which is unique only WITHIN a category, so it merged all "
            "seven categories' templates. It reached published results.",
     "use_instead": "T8 -- 66 templates, 62 touched, median 62 rows, 210 of "
                    "4000 surviving."},
    {"retired": "the cascade is worth +35.6 points",
     "literals": ["+35.6", "35.6 points"],
     "why": "That figure is baseline minus Tier-1-only, i.e. the "
            "BGE-vs-TF-IDF REPRESENTATION gap. It is not what cascading buys.",
     "use_instead": "T3 -- against the Tier-2-only control the cascade is one "
                    "ticket worse on both sets (exact McNemar p = 1.000 "
                    "each), and buys 8-18% of median per-ticket latency."},
    {"retired": "7A's domain AUC 0.8584",
     "literals": ["0.8584"],
     "why": "Computed in an interactive session, quoted at a gate, never "
            "committed, and it DOES NOT REPRODUCE. This is the concrete "
            "motivation for Phase 8A.",
     "use_instead": "T11 -- 0.8727 raw / 0.8472 operating, from "
                    "run_external_conformal_shift.py."},
    {"retired": "7A's domain AUC range 0.6706-0.9316",
     "literals": ["0.6706", "0.9316"],
     "why": "Same defect as 0.8584 -- an uncommitted interactive derivation.",
     "use_instead": "T11's Design B column, which is itself no-resolution and "
                    "may not be a headline."},
    {"retired": "the LLM judge's hedge-axis kappa of 0.000",
     "literals": ["hedge kappa 0.000", "kappa 0.000", "κ 0.000"],
     "why": "A degenerate agreement table on that axis. It is not evidence "
            "about the judge and is never included.",
     "use_instead": "T14 -- the SUPPORT axis: raw agreement 30/33 with "
                    "Cohen's kappa below zero, plus the three disagreements "
                    "read individually."},
    {"retired": "reweighting recovers 47.6% against distribution matching's "
                "~38%",
     "literals": ["47.6%", "47.6 vs", "recovery 47.6"],
     "why": "An ORDERED COMPARISON between two residuals that differ by less "
            "than the measurement's own noise band. Both strategies are "
            "partial; neither is shown to beat the other.",
     "use_instead": "T7 -- report both readings of 6B's clause and make no "
                    "ordered comparison."},
    {"retired": "any Phase 7B Design B number as a headline",
     "literals": [],
     "why": "Design B resolved nothing, twice: one queue BLOCKED by the "
            "degeneracy rule and the rest single-class test arms.",
     "use_instead": "T11's Design A rows, which are the pre-registered "
                    "primary."},
    {"retired": "Phase 7C coverage numbers as a verdict",
     "literals": [],
     "why": "The pre-registered rule fired at a TF-IDF-space domain AUC of "
            "0.9972, so NO verdict was drawn on the primary. A blocked arm "
            "is not a null.",
     "use_instead": "T11's 7C rows, every one tagged post-hoc and BLOCKED."},
]


# ---------------------------------------------------------------------------
# NUMBERS.md
# ---------------------------------------------------------------------------
def write_numbers_md(path, numbers, provenance, stats_check):
    lines = [
        "# NUMBERS.md -- every number the paper may use",
        "",
        "**GENERATED. Do not edit by hand.** Rebuild with:",
        "",
        "```powershell",
        "python src/experiments/build_paper_artifacts.py --force",
        "```",
        "",
        "Phase 7A recorded two domain AUCs that were computed in an "
        "interactive session, quoted at a gate, and never committed. They do "
        "not reproduce. Every value below therefore names the committed file "
        "it came from and the command that regenerates that file. A number "
        "that is not in this table is not a number the paper may use.",
        "",
        f"Config fingerprint at build time: `{config_fingerprint()}`.",
        "",
        "Conventions: proportions carry a Wilson 95% interval computed with "
        f"z = {Z_95}, except on pre-registered case-study axes, which report "
        "counts only. Paired comparisons on the same tickets use the EXACT "
        "McNemar test. A difference smaller than its own noise band is "
        f"tagged `{TAG_WITHIN_BAND}` automatically.",
        "",
        "Tags: "
        f"`{TAG_POST_HOC}` (not a pre-registered result), "
        f"`{TAG_BLOCKED}` (a pre-registered rule fired; no verdict), "
        f"`{TAG_NO_RESOLUTION}` (the comparison could not resolve), "
        f"`{TAG_DEGENERATE}` (the test has no resolution at this operating "
        "point), "
        f"`{TAG_CASE_STUDY}` (counts only, by pre-registration), "
        f"`{TAG_WITHIN_BAND}` (inside the measurement's own noise band).",
        "",
        "---",
        "",
    ]

    by_group = {}
    for num in numbers:
        by_group.setdefault(num.group or "", []).append(num)

    def emit_rows(rows):
        out = ["| id | value | source file | regenerating command | tags |",
               "|---|---|---|---|---|"]
        for num in rows:
            tags = ", ".join(f"`{t}`" for t in num.tags) if num.tags else ""
            out.append(
                f"| `{num.id}` | {num.value} | `{num.source}` | "
                f"`{num.command}` | {tags} |")
            if num.note:
                out.append(f"| | *{num.note}* | | | |")
        out.append("")
        return out

    lines.append("## Clause groups -- numbers that may never be quoted alone")
    lines.append("")
    for group_id in sorted(CLAUSES):
        rows = by_group.get(group_id, [])
        lines.append(f"### Clause group `{group_id}`")
        lines.append("")
        lines.append("> " + CLAUSES[group_id].replace("\n", "\n> "))
        lines.append("")
        if rows:
            lines += emit_rows(rows)
        else:
            lines.append("*(no numbers currently carry this clause)*")
            lines.append("")

    lines.append("---")
    lines.append("")
    lines.append("## Ungrouped numbers, by table and figure")
    lines.append("")
    ungrouped = by_group.get("", [])
    prefixes = sorted({n.id.split(".")[0] for n in ungrouped},
                      key=lambda s: (s[0], int(re.sub(r"\D", "", s) or 0)))
    for prefix in prefixes:
        rows = [n for n in ungrouped if n.id.split(".")[0] == prefix]
        lines.append(f"### {prefix}")
        lines.append("")
        lines += emit_rows(rows)

    lines.append("---")
    lines.append("")
    lines.append("## DO NOT CITE")
    lines.append("")
    lines.append(
        "Each of these was published at some point and is now retired. The "
        "parity test greps the whole of `paper/` for the retired literals "
        "and fails if one reappears outside this section.")
    lines.append("")
    for item in DO_NOT_CITE:
        lines.append(f"### {item['retired']}")
        lines.append("")
        lines.append(f"- **Why it is retired:** {item['why']}")
        lines.append(f"- **Use instead:** {item['use_instead']}")
        lines.append("")

    lines.append("---")
    lines.append("")
    lines.append("## Numbers in the documents with NO committed source")
    lines.append("")
    if NO_SOURCE:
        lines.append(
            "These appear in the project's documents but cannot be "
            "regenerated from any committed file. Nothing here was invented, "
            "re-derived or re-run. Each is a decision: keep it as a figure "
            "whose derivation is a script that must be re-executed, drop it, "
            "or add the missing writer in a later phase.")
        lines.append("")
        lines.append("| number | stated in the docs | where | why there is "
                     "no source |")
        lines.append("|---|---|---|---|")
        for item in sorted(NO_SOURCE, key=lambda d: d["number"]):
            lines.append(
                f"| {item['number']} | {item['stated_in_docs']} | "
                f"{item['doc_location']} | {item['why_no_source']} |")
    else:
        lines.append("*(none)*")
    lines.append("")

    lines.append("---")
    lines.append("")
    lines.append("## Source files and their hashes at build time")
    lines.append("")
    lines.append("| source file | sha256 |")
    lines.append("|---|---|")
    for source_path in sorted(provenance):
        lines.append(f"| `{source_path}` | `{provenance[source_path]}` |")
    lines.append("")
    lines.append(
        f"Statistical cross-check at build time: "
        f"{stats_check['wilson_checks']} Wilson intervals and "
        f"{stats_check['mcnemar_checks']} exact McNemar tests agreed exactly "
        f"with the implementations already committed in "
        f"`src/experiments/`.")
    lines.append("")

    with open(path, "w", encoding="utf-8", newline="\n") as fh:
        fh.write("\n".join(lines))
    return len(lines)


# ---------------------------------------------------------------------------
# FRAMING.md -- the agreed write-up framing, carried forward for Phase 9A
# ---------------------------------------------------------------------------
def write_framing_md(path, numbers):
    idx = {n.id: n.value for n in numbers}

    def val(number_id):
        if number_id not in idx:
            raise KeyError(
                f"FRAMING.md refers to {number_id}, which was never emitted. "
                f"Framing text may not carry a number that has no source.")
        return idx[number_id]

    lines = [
        "# FRAMING.md -- how the results must be written up",
        "",
        "**GENERATED. Do not edit by hand.** Every number below is "
        "interpolated from `NUMBERS.md`, so this file cannot drift away from "
        "the measurements it describes.",
        "",
        "This file exists for Phase 9A. Each item was agreed at a phase gate "
        "and is not reopened here.",
        "",
        "---",
        "",
        "## 1. The corpus is an object of study, not a given",
        "",
        "The nasscom brief recommended a synthetic, LLM-generated dataset. "
        "We followed it, at "
        f"{val('T8.dataset_rows')} rows, and the experimental-setup section "
        "opens by saying so. The corpus-as-object-of-study framing follows "
        "immediately: Phase 2A, Finding 2 and Phase 5C are measurements of "
        "what that corpus is worth, not incidental limitations.",
        "",
        "In-distribution accuracy is uninformative on template-generated "
        "data -- every model scores about 100%. Only the 14- and 45-ticket "
        "benchmarks measure anything real, and a new 100% in-distribution "
        "number is a red flag rather than a success.",
        "",
        "## 2. The named cross-phase finding",
        "",
        "> " + CLAUSES["named-finding"],
        "",
        "Its sharpest single instance is Finding 4: the same method at the "
        "same alpha against the same benchmark moves from a coverage gap of "
        f"{val('T10.tier1.gap_in_domain')} to "
        f"{val('T10.tier1.gap_deployment')} when only the calibration "
        "DISTRIBUTION changes.",
        "",
        "Phase 9A settles the final wording. Phase 5C widened the mechanism "
        "from the calibration/reference distribution to the training "
        "distribution as well; the two original instances are unchanged and "
        "are not weakened by the addition.",
        "",
        "## 3. The contribution is the calibrated escalation machinery, not "
        "accuracy",
        "",
        "> " + CLAUSES["5C"],
        "",
        "The contrast is the argument, so the zero-shot baseline goes in the "
        "paper prominently and is never buried. Zero-shot Gemini scores "
        f"{val('T1.zeroshot_gemini.benchmark45')} against the trained "
        f"Tier-2's {val('T1.tier2only.benchmark45')} "
        f"(exact McNemar p = {val('T1.mcnemar.gemini_vs_tier2.benchmark45')}) "
        "while having no abstention guarantee, no calibrated gate and no "
        "cost model. Qwen2.5-3B, running locally on a laptop CPU, scores "
        f"{val('T1.zeroshot_qwen.benchmark45')} and is indistinguishable "
        f"from Tier-2 (p = {val('T1.mcnemar.qwen_vs_tier2.benchmark45')}).",
        "",
        "## 4. Methodological findings",
        "",
        "### Ungated and averaged metrics manufacture significance",
        "",
        "> " + CLAUSES["6A"],
        "",
        f"At the live gate, {val('T12.6a.degenerate_configurations')} "
        "configurations are degenerate and "
        f"{val('T12.6a.comparisons_with_signal')} of "
        f"{val('T12.6a.comparisons_total')} comparisons show a signal. Phase "
        "2A's decision-rule inversion is the second instance of the same "
        "shape.",
        "",
        "### In-distribution calibration is a ceiling effect, not "
        "calibration",
        "",
        "On the 500-ticket in-distribution batch the observed accuracy is "
        f"{val('T4.observed_accuracy_every_bin')} in every confidence bin of "
        "both tiers. The reported ECE -- Tier-1 "
        f"{val('T4.tier1.ece')}, Tier-2 {val('T4.tier2.ece')} -- is "
        "therefore entirely the distance between the model's confidence and "
        "a ceiling, and both tiers are systematically UNDER-confident there. "
        "It is not evidence that either is well calibrated on real traffic. "
        "This is the same ceiling effect as the ~100% in-distribution "
        "accuracy, and it is why both escalation gates were calibrated "
        "empirically against benchmark behaviour rather than read off a "
        "predicted probability.",
        "",
        "### An LLM judge must never be used unaudited",
        "",
        f"Raw agreement with the human labels was "
        f"{val('T14.judge.raw_agreement')} and Cohen's kappa was "
        f"{val('T14.judge.cohens_kappa')}. Every disagreement was the judge "
        "substituting APPROPRIATENESS for SUPPORT, in both directions. An "
        "aggregate agreement score would not have revealed this; only "
        "labelling the full population and reading each disagreement did. A "
        "judge is an object of study in this project, never a scaling tool.",
        "",
        "### A degeneracy rule must be justified per design",
        "",
        "Phase 7C's blocking rule was copied from Phase 6B, where it guarded "
        "a DENSITY-RATIO estimate. 7C computes no density ratio and uses the "
        "AUC only as a manipulation check, where a near-1.0 value means the "
        "manipulation was STRONG. The critique is sound and the gate still "
        "DECLINED to unblock, because revising a pre-registered rule after "
        "seeing the results is precisely what the discipline exists to "
        "prevent. It is recorded as a specification error, not acted on. The "
        "counter-reading is recorded too: a rewrite a classifier identifies "
        "with near-certainty may be a DIFFERENT corpus rather than a shifted "
        "one.",
        "",
        "### A number quoted at a gate must come from a committed script",
        "",
        "Phase 7A's two design domain AUCs were computed interactively, "
        "quoted at a gate, never committed, and do not reproduce. The "
        "conclusions were unaffected because every value cleared the same "
        "threshold, but the figures were wrong in print for a day. This is "
        "the motivation for Phase 8A and belongs in the reproducibility "
        "section.",
        "",
        "## 5. The escalation gate errs in both directions",
        "",
        f"The RAG similarity gate sits at {val('F1.gate.rag')}, inside the "
        f"adversarial safe range [{val('T5.safe_range_low')}, "
        f"{val('T5.safe_range_high')}]. It is not a perfect separator and "
        "the paper says so in both directions: adversarial tickets G021 and "
        "G024 pass the gate when they should not, and ticket N45 "
        "(top similarity 0.6397) is escalated although Phase 6C's rater "
        "found its retrieved context adequate. A scalar threshold on a "
        "single similarity is the simplest thing that works, not a claim "
        "that it is sufficient.",
        "",
        "## 6. Separability does not imply score shift",
        "",
        "Phase 6B's cross-fitted domain classifier separates the "
        "calibration set from the deployment set MORE easily in BGE space "
        f"({val('T7.weighted.bge.domain_auc')}) than in TF-IDF space -- and "
        "BGE is nevertheless the space whose coverage transfers. The BGE "
        "arm's own numbers are BLOCKED, not a result; the AUC itself is a "
        "quantitative statement of the named finding. Cross-reference "
        "Finding 1.",
        "",
        "> " + CLAUSES["6B"],
        "",
        "## 7. Finding 1, in full",
        "",
        "> " + CLAUSES["finding1"].replace("\n", "\n> "),
        "",
        f"On our corpus, at alpha = {val('T7.alpha')}: Tier-1's coverage gap "
        f"is {val('T7.tier1.coverage_gap')} against Tier-2's "
        f"{val('T7.tier2.coverage_gap')}, with a +/-2 s.d. band of "
        f"{val('T7.noise_band_2sd')}. On the external corpus under a version "
        f"shift the two are indistinguishable "
        f"({val('T11.7b.tier1.coverage_gap')} against "
        f"{val('T11.7b.tier2.coverage_gap')}, band "
        f"{val('T11.7b.noise_band_2sd')}) -- but that corpus has almost no "
        "representation gap to find "
        f"(Tier-1 {val('T11.7b.label_ceiling_tier1')} vs Tier-2 "
        f"{val('T11.7b.label_ceiling_tier2')}), which weakens 7B as "
        "evidence and is a limitation, not a rescue.",
        "",
        "A test-arm coverage band must include the test-sampling term. At "
        "Phase 7C's n = 286 the calibration-only band is "
        f"{val('T11.7c.calibration_only_band_2sd')} while the combined band "
        f"is {val('T11.7c.combined_band_2sd')}; quoting the calibration term "
        "alone on a small test arm understates uncertainty by roughly 2x.",
        "",
        "## 8. Phase 6C",
        "",
        "> " + CLAUSES["6C"],
        "",
        f"The rater caught {val('T15.caught_of_ungrounded')} of the "
        "human-labelled ungrounded drafts and flagged "
        f"{val('T15.flagged_of_grounded')} of the grounded ones. A gate "
        f"escalating {val('T15.escalated_of_eligible')} of the tickets that "
        "currently reach the resolver is not usable, and there is nothing to "
        "tune.",
        "",
        "## 9. Phase 2A and the clustering question",
        "",
        "Phase 2A is a RELATED corpus limitation, not an instance of the "
        "named finding. Its pilot found "
        f"{val('T16.pilot.false_merges')} false merges, which bounds the "
        f"false-merge rate above by {val('T16.pilot.rule_of_three_upper')} "
        "by the rule of three -- the honest reading of a zero count. "
        "Promotion of BGE clustering is now a product decision about "
        "review-queue capacity, not a calibration one, and must not be "
        "reopened as a calibration question.",
        "",
        "## 10. What the system is, described honestly",
        "",
        "A sequential pipeline with two independently calibrated confidence "
        "gates, agent boundaries and an HTTP surface. The agents are "
        "independently ADDRESSABLE, not independently running: POST /triage "
        "orchestrates in process. It is never described as a distributed or "
        "autonomous multi-agent system. Gemini never decides the category; "
        "classification is entirely the trained models', and Gemini's only "
        "job is resolution generation grounded in retrieved tickets.",
        "",
        "> " + CLAUSES["7A-control"],
        "",
        "The external dataset is INDEPENDENTLY GENERATED DATA, NOT REAL "
        "PRODUCTION DATA -- its card advertises a synthetic generator from "
        "the same author. It tests whether findings survive a DIFFERENT "
        "GENERATOR, not whether they survive reality. Any write-up must say "
        "so in those words.",
        "",
    ]
    with open(path, "w", encoding="utf-8", newline="\n") as fh:
        fh.write("\n".join(lines))


# ---------------------------------------------------------------------------
# Reconciliation audit
# ---------------------------------------------------------------------------
# Pass A anchors. Each is a place where a document states a number that a
# committed file also produces. The regex captures the document's version; the
# build compares it against the emitted value at the document's own precision.
# An anchor that no longer matches anywhere is itself reported -- a silently
# vanishing anchor would hide a drift.
ANCHORS = [
    # (label, emitted id, regex the DOCUMENTS are searched with, the value
    #  the documents state). The check has two halves: the stated value must
    #  be a correct rounding of what the committed file now produces, and the
    #  regex must still match somewhere. An anchor that stops matching is
    #  reported too -- a silently vanishing anchor would hide a drift.
    # --- classification and baselines ---
    ("cascade, benchmark45", "T1.cascade.benchmark45",
     r"\b32\s*/\s*45\b", "32/45"),
    ("Tier-2-only, benchmark45", "T1.tier2only.benchmark45",
     r"\b33\s*/\s*45\b", "33/45"),
    ("Tier-1-only, benchmark45", "T1.tier1only.benchmark45",
     r"\b16\s*/\s*45\b", "16/45"),
    ("cascade, deployment175", "T1.cascade.deployment175",
     r"\b131\s*/\s*175\b", "131/175"),
    ("Tier-2-only, deployment175", "T1.tier2only.deployment175",
     r"\b132\s*/\s*175\b", "132/175"),
    ("Tier-1-only, deployment175", "T1.tier1only.deployment175",
     r"\b91\s*/\s*175\b", "91/175"),
    ("zero-shot Gemini, benchmark45", "T1.zeroshot_gemini.benchmark45",
     r"\b40\s*/\s*45\b", "40/45"),
    ("zero-shot Gemini, benchmark14", "T1.zeroshot_gemini.benchmark14",
     r"\b14\s*/\s*14\b", "14/14"),
    ("zero-shot Qwen, benchmark45", "T1.zeroshot_qwen.benchmark45",
     r"\b34\s*/\s*45\b", "34/45"),
    ("zero-shot Qwen, benchmark14", "T1.zeroshot_qwen.benchmark14",
     r"\b12\s*/\s*14\b", "12/14"),
    ("Gemini vs Tier-2, exact McNemar",
     "T1.mcnemar.gemini_vs_tier2.benchmark45",
     r"p\s*=\s*0\.0391", "0.0391"),
    ("Qwen vs Gemini, exact McNemar",
     "T1.mcnemar.qwen_vs_gemini.benchmark45",
     r"p\s*=\s*0\.070", "0.070"),
    ("Qwen Application precision", "T2.qwen.application.precision.benchmark45",
     r"Application precision\D{0,30}47", "47"),

    # --- the ablation and what the cascade buys ---
    ("cascade vs Tier-2, benchmark45", "T3.mcnemar.benchmark45",
     r"p\s*=\s*1\.000", "1.000"),
    ("Tier-1 warm latency", "T3.latency.tier1_median_ms",
     r"\b1\.04\s*ms", "1.04"),
    ("Tier-2 warm latency", "T3.latency.tier2_median_ms",
     r"\b156\.40\s*ms", "156.40"),
    ("Tier-2 / Tier-1 latency ratio", "T3.latency.tier2_over_tier1",
     r"\b151\s*[x×]", "151"),

    # --- calibration ---
    ("Tier-1 ECE", "T4.tier1.ece", r"\b0\.1122\b", "0.1122"),
    ("Tier-2 ECE", "T4.tier2.ece", r"\b0\.0992\b", "0.0992"),
    ("OOD leakage at the gate", "T5.ood_leakage_at_gate",
     r"13\.3\s*%", "13.3"),
    ("OOD leakage two steps below", "T5.ood_leakage_below_gate",
     r"37\.8\s*%", "37.8"),
    ("adversarial safe range, low", "T5.safe_range_low",
     r"\b0\.6196\b", "0.6196"),

    # --- Finding 1 and Phase 6B ---
    ("Finding 1, Tier-1 gap", "T7.tier1.coverage_gap",
     r"-\s*0\.2333|−\s*0\.2333", "-0.2333"),
    ("Finding 1, Tier-2 gap", "T7.tier2.coverage_gap",
     r"-\s*0\.0111|−\s*0\.0111", "-0.0111"),
    ("6B, reweighted Tier-1 gap", "T7.weighted.tier1.coverage_gap",
     r"-\s*0\.1222|−\s*0\.1222", "-0.1222"),
    ("6B, BGE-arm domain AUC (BLOCKED)", "T7.weighted.bge.domain_auc",
     r"\b0\.9908\b", "0.9908"),

    # --- Finding 2, corrected in Phase 5A ---
    ("templates in the dataset", "T8.templates_total",
     r"\b66\s+templates\b", "66"),
    ("templates touched by the calibration set", "T8.templates_touched",
     r"\b62\s+of\s+them\b|touches\s+62\b", "62"),
    ("rows surviving template exclusion",
     "T8.rows_surviving_template_exclusion",
     r"\b210\s*/\s*4000\b|\b210\s+of\s+4000\b", "210"),

    # --- external validity ---
    ("external English rows", "T11.7a.rows_english",
     r"\b28,?261\b", "28261"),
    ("external near-duplicate rate", "T11.7a.near_duplicate_rate_external",
     r"79\.39\s*%", "79.39"),
    ("our corpus near-duplicate rate",
     "T11.7a.near_duplicate_rate_our_corpus", r"85\.20\s*%", "85.20"),
    ("7B, Tier-1 gap", "T11.7b.tier1.coverage_gap",
     r"\+\s*0\.0009\b", "0.0009"),
    ("7B, Tier-2 gap", "T11.7b.tier2.coverage_gap",
     r"-\s*0\.0072\b|−\s*0\.0072\b", "-0.0072"),
    ("7B, noise band", "T11.7b.noise_band_2sd", r"\b0\.0165\b", "0.0165"),
    ("7B, operating domain AUC", "T11.7b.domain_auc_operating",
     r"\b0\.8472\b", "0.8472"),
    ("7B, boundary contamination",
     "T11.7b.boundary_contamination_rate", r"1\.46\s*%", "1.46"),
    ("7C, TF-IDF domain AUC (BLOCKED)", "T11.7c.tfidf_domain_auc",
     r"\b0\.9972\b", "0.9972"),
    ("7C, TF-IDF cosine", "T11.7c.tfidf_mean_cosine",
     r"\b0\.2489\b", "0.2489"),
    ("7C, pairs", "T11.7c.n_pairs", r"\b286\b", "286"),
    ("7C, difference-in-differences (post-hoc)", "T11.7c.did_point",
     r"-\s*0\.0804\b|−\s*0\.0804\b", "-0.0804"),

    # --- deferral, drift, generation, clustering ---
    ("6A, gate coverage on benchmark45", "T12.6a.gate_coverage.benchmark45",
     r"8\.9\s*%", "8.9"),
    ("6A, gate coverage on deployment175",
     "T12.6a.gate_coverage.deployment175", r"18\.9\s*%", "18.9"),
    ("drift eligible operating points", "T13.eligible_operating_points",
     r"\b25\s*/\s*68\b", "25/68"),
    ("drift realistic traffic, alpha 0.05",
     "T13.realistic_traffic.flag_rate.alpha0.05", r"\b0\.429\b", "0.429"),
    ("drift realistic traffic, alpha 0.20",
     "T13.realistic_traffic.flag_rate.alpha0.2", r"\b0\.646\b", "0.646"),
    ("2B, grounded drafts", "T14.grounded", r"\b31\s*/\s*33\b", "31/33"),
    ("2B, judge raw agreement", "T14.judge.raw_agreement",
     r"90\.9\s*%", "30/33"),
    ("2B, judge Cohen's kappa", "T14.judge.cohens_kappa",
     r"-\s*0\.042\b|−\s*0\.042\b", "-0.042"),
    ("6C, caught (case study)", "T15.caught_of_ungrounded",
     r"\b2\s*(?:of|/)\s*2\b", "2/2"),
    ("6C, false flags", "T15.flagged_of_grounded",
     r"\b26\s*(?:of|/)\s*31\b", "26/31"),
    ("6C, false-flag proportion", "T15.false_flag_proportion",
     r"\b0\.839\b", "0.839"),
    ("clustering production threshold", "T16.production.threshold",
     r"\b0\.80\b", "0.80"),
    ("clustering BGE pooled cliff", "T16.bge.pooled_cliff",
     r"pooled cliff 0\.90|BGE@0\.90", "0.90"),
    ("2A pilot false merges", "T16.pilot.false_merges",
     r"\b0\s*/\s*12\b", "0/12"),
]


def _anchor_agrees(expected, number):
    """Is the document's stated figure a correct rounding of the source's?

    Counts (k/n) and short text are compared as strings; everything else is
    compared numerically at the document's own precision, and also against
    the value expressed as a percentage, because the documents use both.
    """
    emitted = number.value.replace(" ", "")
    if "/" in expected or not re.fullmatch(r"-?\d+(?:\.\d+)?", expected):
        return expected.replace(" ", "") in emitted
    raw = number.raw
    if isinstance(raw, str):
        return expected in emitted
    try:
        raw = float(raw)
    except (TypeError, ValueError):
        return expected in emitted
    places = len(expected.split(".")[1]) if "." in expected else 0
    target = float(expected)
    return (round(raw, places) == target
            or round(raw * 100.0, places) == target)


STAT_TOKEN_RE = re.compile(
    r"(?<![\w.])(?:"
    r"-?\d+\.\d{3,}"          # a statistic-shaped decimal
    r"|\d{1,4}\s*/\s*\d{1,5}"  # a k/n count
    r"|\d+\.\d%"              # a one-decimal percentage
    r")(?![\w])")


def printed_forms(value_text):
    """Every way a value might legitimately appear in prose."""
    forms = {value_text.strip()}
    match = re.match(r"^(-?\d+(?:\.\d+)?)", value_text.strip())
    if match:
        raw = float(match.group(1))
        for places in (2, 3, 4, 6):
            forms.add(f"{raw:.{places}f}")
            forms.add(f"{raw:.{places}f}".rstrip("0").rstrip("."))
            forms.add(f"{abs(raw):.{places}f}")
        for places in (1, 2):
            forms.add(f"{raw * 100:.{places}f}")
            forms.add(f"{abs(raw) * 100:.{places}f}")
    for part in re.findall(r"\d+\s*/\s*\d+", value_text):
        forms.add(part.replace(" ", ""))
    return {f for f in forms if f}


def run_reconciliation(numbers, out_path):
    docs = {}
    for name in DOCS_AUDITED:
        doc_path = os.path.join(PROJECT_ROOT, name)
        if not os.path.isfile(doc_path):
            continue
        with open(doc_path, "r", encoding="utf-8") as fh:
            docs[name] = fh.read()

    idx = {n.id: n for n in numbers}

    # ---- Pass A: anchored ----
    anchor_rows = []
    for label, number_id, pattern, expected in ANCHORS:
        if number_id not in idx:
            anchor_rows.append({"anchor": label, "id": number_id,
                                "status": "NUMBER NOT EMITTED",
                                "expected": expected, "found_in": ""})
            continue
        number = idx[number_id]
        agrees = _anchor_agrees(expected, number)
        hits = [name for name, text in docs.items()
                if re.search(pattern, text)]
        if not agrees:
            status = ("MISMATCH: the documents say " + expected
                      + " but the source file now gives " + number.value)
        elif not hits:
            status = "anchor absent from every document"
        else:
            status = "ok"
        anchor_rows.append({"anchor": label, "id": number_id,
                            "status": status, "expected": expected,
                            "found_in": ", ".join(sorted(hits))})

    # ---- Pass B: sweep ----
    known = set()
    for num in numbers:
        known |= printed_forms(num.value)
    retired = set()
    for item in DO_NOT_CITE:
        for lit in item["literals"]:
            retired.add(lit.strip())

    sweep_rows = []
    for name, text in docs.items():
        seen = {}
        for match in STAT_TOKEN_RE.finditer(text):
            token = match.group(0).replace(" ", "")
            seen[token] = seen.get(token, 0) + 1
        for token, count in seen.items():
            if token in known:
                verdict = "matched to a committed source"
            elif any(token in lit or lit in token for lit in retired):
                verdict = "DO-NOT-CITE literal"
            else:
                verdict = "unmatched"
            sweep_rows.append({"document": name, "token": token,
                               "occurrences": count, "verdict": verdict})
    sweep = pd.DataFrame(sweep_rows)

    lines = [
        "# RECONCILIATION.md -- the documents against their source files",
        "",
        "**GENERATED. Do not edit by hand.**",
        "",
        "Phase 8A's rule: where a document and a committed result file "
        "disagree, the DOCUMENT is wrong and is fixed. No result file was "
        "edited to match a document, and no number was invented to give a "
        "document a source.",
        "",
        "## Convention notes",
        "",
        "The repository contains **two conventions for a 95% z**: "
        f"`{Z_95}` in `src/experiments/score_groundedness_set.py` and "
        f"`src/experiments/score_sufficiency_gate.py`, and the exact normal "
        f"quantile `{Z_95_EXACT}` in "
        "`src/experiments/summarize_zeroshot_baselines.py`. The two "
        "implementations are otherwise algebraically identical -- the build "
        "asserts that -- and they differ by about 3.5e-6, so **no published "
        "figure is affected at reported precision**. The paper adopts "
        f"`{Z_95}`. Confidence intervals that a source file already carries "
        "are read verbatim rather than recomputed, so a published interval "
        "can never be silently restated under a different z.",
        "",
        "---",
        "",
        "## Pass A -- anchored checks",
        "",
        "For each anchor, the document's stated value is compared against "
        "the value the committed file currently produces. This is the pass "
        "that would catch a Phase 7A-style drift.",
        "",
        "| anchor | id | expected | status | found in |",
        "|---|---|---|---|---|",
    ]
    for row in anchor_rows:
        lines.append(
            f"| {row['anchor']} | `{row['id']}` | {row['expected']} | "
            f"{row['status']} | {row['found_in']} |")

    n_mismatch = sum(1 for r in anchor_rows if r["status"].startswith(
        "MISMATCH"))
    lines += [
        "",
        f"**{n_mismatch} anchored mismatch(es).**",
        "",
        "---",
        "",
        "## Pass B -- sweep",
        "",
        "Every statistic-shaped token in the four documents -- decimals with "
        "three or more places, `k/n` counts, and one-decimal percentages -- "
        "classified against the emitted set. Prose numbers, dates, line "
        "counts and file sizes are not statistics, so an `unmatched` token "
        "is triage output, not a defect.",
        "",
    ]
    if not sweep.empty:
        counts = (sweep.groupby(["document", "verdict"])["occurrences"]
                  .sum().reset_index())
        lines.append("| document | verdict | occurrences |")
        lines.append("|---|---|---|")
        for _, r in counts.sort_values(["document", "verdict"]).iterrows():
            lines.append(f"| {r['document']} | {r['verdict']} | "
                         f"{int(r['occurrences'])} |")
        lines.append("")
        retired_hits = sweep[sweep["verdict"] == "DO-NOT-CITE literal"]
        lines.append("### DO-NOT-CITE literals still present in the documents")
        lines.append("")
        if retired_hits.empty:
            lines.append("*(none)*")
        else:
            lines.append("These are expected in the lab notebook, which "
                         "records superseded results on purpose. They must "
                         "not appear in the paper.")
            lines.append("")
            lines.append("| document | token | occurrences |")
            lines.append("|---|---|---|")
            for _, r in retired_hits.sort_values(
                    ["document", "token"]).iterrows():
                lines.append(f"| {r['document']} | `{r['token']}` | "
                             f"{int(r['occurrences'])} |")
        lines.append("")

    lines += [
        "---",
        "",
        "## Numbers with NO committed machine-readable source",
        "",
        "Nothing below was invented, re-derived or re-run. Each is a "
        "decision about the paper.",
        "",
    ]
    if NO_SOURCE:
        lines.append("| number | stated in the docs | where | why there is "
                     "no source |")
        lines.append("|---|---|---|---|")
        for item in sorted(NO_SOURCE, key=lambda d: d["number"]):
            lines.append(
                f"| {item['number']} | {item['stated_in_docs']} | "
                f"{item['doc_location']} | {item['why_no_source']} |")
    else:
        lines.append("*(none)*")
    lines.append("")

    with open(out_path, "w", encoding="utf-8", newline="\n") as fh:
        fh.write("\n".join(lines))

    return {"anchors": anchor_rows, "sweep": sweep,
            "n_anchor_mismatches": n_mismatch}


# ---------------------------------------------------------------------------
# paper/README.md
# ---------------------------------------------------------------------------
def write_paper_readme(path, n_table_files, n_numbers):
    # NOTE: the counts below must not depend on --no-render, or the parity
    # test would compare a rendered build against an unrendered one and fail
    # for a reason that has nothing to do with a number moving.
    lines = [
        "# paper/ -- the generated write-up surface",
        "",
        "**Everything in this directory is generated. Do not edit any of it "
        "by hand.** Rebuild the whole directory with one command, from the "
        "project root:",
        "",
        "```powershell",
        ".\\venv\\Scripts\\Activate.ps1",
        "python src/experiments/build_paper_artifacts.py --force",
        "```",
        "",
        "The build is offline. It makes **no Gemini call, no Ollama call, no "
        "model load, no training run and no experiment re-run** -- it reads "
        "committed result files under `data/` and writes only here.",
        "",
        "## Why this exists",
        "",
        "Phase 7A recorded two domain AUCs that were computed in an "
        "interactive session, quoted at a gate, and never committed. When "
        "Phase 7B specified the recomputation they did not reproduce. The "
        "conclusions survived because every value cleared the same "
        "threshold, but the figures were wrong in print for a day.",
        "",
        "So: **every number the paper uses is regenerated from a committed "
        "result file, through code.** `tests/test_paper_artifacts.py` "
        "rebuilds this directory and fails if any value, table CSV or "
        "figure-data CSV changes -- golden parity, applied to the write-up.",
        "",
        "## What is here",
        "",
        f"- `NUMBERS.md` -- {n_numbers} numbers, each with its source file, "
        "the command that regenerates that file, and its tags. A number that "
        "is not in this file is not a number the paper may use. It also "
        "carries the **do-not-cite list** and the **no committed source** "
        "list.",
        "- `FRAMING.md` -- the write-up framing agreed at each phase gate, "
        "with its numbers interpolated from `NUMBERS.md` so the text cannot "
        "drift from the measurements.",
        "- `RECONCILIATION.md` -- the four project documents audited against "
        "their source files.",
        "- `PROVENANCE.json` -- the sha256 of every source file read, so a "
        "parity failure can distinguish 'the builder changed' from 'a result "
        "file changed'.",
        f"- `tables/` -- {n_table_files} files: each table as `.csv` and "
        "as booktabs `.tex`.",
        f"- `figures/` -- {len(FIGURES)} figures, each as `.pdf` and `.png` "
        "at 300 dpi, with the data behind it as `_data.csv` and its caption "
        "as `_caption.txt`.",
        "",
        "## Conventions",
        "",
        "- Figures are sized for IEEE two-column: "
        f"{IEEE_SINGLE_COL_IN} in single-column, {IEEE_DOUBLE_COL_IN} in "
        "full-width, base font "
        f"{BASE_FONT_PT} pt, Okabe-Ito colour-blind-safe palette, and "
        "distinct dash patterns so the figures also survive greyscale.",
        "- Output is deterministic: fixed seed, stable row order, fixed "
        "float formatting, and PDF/PNG creation metadata stripped. A rebuild "
        "is byte-identical unless a source file changed.",
        "- Proportions carry a Wilson 95% interval, except on pre-registered "
        "case-study axes, which report counts only. Paired comparisons on "
        "the same tickets use the exact McNemar test.",
        "",
        "## Flags",
        "",
        "| flag | effect |",
        "|---|---|",
        "| *(none)* | refuses to overwrite an existing `paper/` |",
        "| `--force` | rebuild in place |",
        "| `--out DIR` | write somewhere else (the parity test uses this) |",
        "| `--no-render` | skip PDF/PNG rasterisation; still writes every "
        "table and figure-data CSV |",
        "| `--no-audit` | skip the document reconciliation pass |",
        "",
    ]
    with open(path, "w", encoding="utf-8", newline="\n") as fh:
        fh.write("\n".join(lines))


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------
def parse_args(argv=None):
    parser = argparse.ArgumentParser(
        description="Phase 8A -- build every paper table, figure and number "
                    "from committed result files. Offline; spends no quota.")
    parser.add_argument("--force", action="store_true",
                        help="rebuild in place over an existing output "
                             "directory")
    parser.add_argument("--out", default=DEFAULT_OUT,
                        help="output directory (default: paper/)")
    parser.add_argument("--no-render", action="store_true",
                        help="skip PDF/PNG rasterisation; tables and "
                             "figure-data CSVs are still written")
    parser.add_argument("--no-audit", action="store_true",
                        help="skip the document reconciliation pass")
    return parser.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    random.seed(SEED)
    np.random.seed(SEED)

    out_dir = os.path.abspath(args.out)
    tables_dir = os.path.join(out_dir, "tables")
    figures_dir = os.path.join(out_dir, "figures")

    if os.path.isdir(out_dir) and os.listdir(out_dir) and not args.force:
        print(f"[abort] {rel(out_dir)} already exists and is not empty.")
        print("        Re-run with --force to rebuild it in place. Refusing "
              "to overwrite a published surface by accident.")
        return 1

    print("[step0] cross-checking the statistics against the repo's own "
          "implementations")
    stats_check = cross_check_statistics()
    print(f"        {stats_check['wilson_checks']} Wilson intervals and "
          f"{stats_check['mcnemar_checks']} exact McNemar tests agree "
          f"exactly")
    print(f"        config fingerprint: {config_fingerprint()}")
    print(f"        production gates (frozen): cascade "
          f"{settings.cascade.confidence_threshold}, RAG "
          f"{settings.rag.similarity_threshold}, clustering "
          f"{settings.clustering.resolution_similarity_threshold}")
    if settings.conformal.enabled or settings.drift.enabled:
        raise AssertionError(
            "settings.conformal.enabled or settings.drift.enabled is True. "
            "Phases 5-9 are measurement only and the paper surface describes "
            "a frozen production configuration -- refusing to build.")

    os.makedirs(tables_dir, exist_ok=True)
    os.makedirs(figures_dir, exist_ok=True)

    emitter = Emitter()
    provenance = {}
    table_files, figure_files = [], []

    def record_sources(sources):
        for source_path in sources:
            abs_path = os.path.join(PROJECT_ROOT, source_path)
            if os.path.isfile(abs_path):
                provenance[source_path] = sha256_file(abs_path)

    print(f"[step1] building {len(TABLES)} tables")
    for spec in TABLES:
        emitter.current_command = spec["command"]
        emitter.current_group = ""
        frames = spec["fn"](emitter.emit)
        record_sources(spec["sources"])
        for key, frame in frames.items():
            base = f"{spec['id']}_{slug(key)}"
            csv_path = os.path.join(tables_dir, base + ".csv")
            tex_path = os.path.join(tables_dir, base + ".tex")
            write_csv(frame, csv_path)
            write_latex(frame, tex_path, base, spec["title"],
                        "" if key == "main" else f"({key.replace('_', ' ')})")
            table_files += [rel(csv_path), rel(tex_path)]
        print(f"        {spec['id']}: {len(frames)} frame(s)")

    print(f"[step2] building {len(FIGURES)} figures"
          f"{' (data only, --no-render)' if args.no_render else ''}")
    for spec in FIGURES:
        emitter.current_command = spec["command"]
        emitter.current_group = ""
        data, draw = spec["fn"](emitter.emit)
        record_sources(spec["sources"])
        base = os.path.join(figures_dir, f"{spec['id']}_{slug(spec['title'])}")
        write_csv(data, base + "_data.csv")
        figure_files.append(rel(base + "_data.csv"))
        with open(base + "_caption.txt", "w", encoding="utf-8",
                  newline="\n") as fh:
            fh.write(spec["caption"] + "\n")
        figure_files.append(rel(base + "_caption.txt"))
        if not args.no_render:
            plt = apply_style()
            fig = draw()
            save_figure(fig, base)
            plt.close(fig)
            figure_files += [rel(base + ".pdf"), rel(base + ".png")]
        print(f"        {spec['id']}: {len(data)} data row(s)")

    print(f"[step3] writing NUMBERS.md ({len(emitter.numbers)} numbers)")
    numbers_path = os.path.join(out_dir, "NUMBERS.md")
    n_lines = write_numbers_md(numbers_path, emitter.numbers, provenance,
                               stats_check)

    print("[step4] writing FRAMING.md")
    write_framing_md(os.path.join(out_dir, "FRAMING.md"), emitter.numbers)

    audit = None
    if not args.no_audit:
        print("[step5] reconciling the documents against their source files")
        audit = run_reconciliation(
            emitter.numbers, os.path.join(out_dir, "RECONCILIATION.md"))
        print(f"        {audit['n_anchor_mismatches']} anchored mismatch(es)")

    print("[step6] writing PROVENANCE.json and README.md")
    with open(os.path.join(out_dir, "PROVENANCE.json"), "w",
              encoding="utf-8", newline="\n") as fh:
        json.dump({
            "phase": "8A",
            "config_fingerprint": config_fingerprint(),
            "seed": SEED,
            "z_95": Z_95,
            "production_gates": {
                "cascade_confidence_threshold":
                    float(settings.cascade.confidence_threshold),
                "rag_similarity_threshold":
                    float(settings.rag.similarity_threshold),
                "clustering_resolution_similarity_threshold":
                    float(settings.clustering.resolution_similarity_threshold),
                "conformal_enabled": bool(settings.conformal.enabled),
                "drift_enabled": bool(settings.drift.enabled),
            },
            "n_numbers": len(emitter.numbers),
            "n_tables": len(TABLES),
            "n_figures": len(FIGURES),
            "sources": provenance,
        }, fh, indent=2, sort_keys=True)
        fh.write("\n")

    write_paper_readme(os.path.join(out_dir, "README.md"), len(table_files),
                       len(emitter.numbers))

    print()
    print("=" * 72)
    print(f"  tables          : {len(TABLES)} "
          f"({len([f for f in table_files if f.endswith('.csv')])} CSV + "
          f"{len([f for f in table_files if f.endswith('.tex')])} LaTeX)")
    print(f"  figures         : {len(FIGURES)}"
          f"{' (data only)' if args.no_render else ' (PDF + PNG + data)'}")
    print(f"  NUMBERS.md      : {len(emitter.numbers)} numbers, "
          f"{n_lines} lines")
    print(f"  sources hashed  : {len(provenance)}")
    print(f"  no-source list  : {len(NO_SOURCE)}")
    if audit is not None:
        print(f"  anchor mismatch : {audit['n_anchor_mismatches']}")
    print(f"  output          : {rel(out_dir)}")
    print("=" * 72)
    if NO_SOURCE:
        print()
        print("Numbers stated in the documents with NO committed source "
              "(nothing was invented or re-run):")
        for item in sorted(NO_SOURCE, key=lambda d: d["number"]):
            print(f"  - {item['number']}: {item['stated_in_docs']} "
                  f"({item['doc_location']})")
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except Exception as exc:                                  # noqa: BLE001
        print()
        print(f"[FATAL] {type(exc).__name__}: {exc}")
        print()
        print("Phase 8A reads committed result files only. It never "
              "regenerates one, never invents a value, and never falls back "
              "to a hard-coded number -- so a missing or changed source is a "
              "hard stop, by design.")
        if os.environ.get("PAPER_ARTIFACTS_TRACEBACK"):
            traceback.print_exc()
        sys.exit(1)
