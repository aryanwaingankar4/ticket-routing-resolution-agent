"""Build the IConAECC 2026 submission and its online supplement (Word).

Offline, model-free, no LLM. Writes, under paper_ieee_docx/:

    Calibrated_Escalation_IConAECC2026.docx   the 6-page paper, built in the
                                              IEEE A4 conference template
    SUPPLEMENT.docx, SUPPLEMENT.md            v4's appendices, moved

Content comes only from the mentor-approved v4
(paper_docx/Calibrated_Escalation_IT_Ticket_Triage_v4.docx):

  * tables, references, the author block, the prompts, the reproduction steps,
    the limitations and the groundedness paragraph are READ from v4 and
    copied verbatim (Table I's cells are shortened, without changing a
    number);
  * the Phase 2 / Phase 3 numbers in the condensed prose are formatted from
    build_v4_docx.load_values(), which re-reads and re-checks the committed
    result files, exactly as v4's build does;
  * every number in the output (paper and supplement) must occur in v4's own
    text: a number that v4 does not contain is FATAL, so condensing cannot
    introduce or change a value.

Sections, tables and figures are numbered by the template's own styles
(Heading 1/2, table head, figure caption, references); the text refers to
them by key, and the build resolves the keys, renumbers the citations in
order of first appearance and fails on any dangling reference, any leftover
template text, "??" or TODO.

    .\\venv\\Scripts\\python.exe scripts/build_iconaecc_docx.py [--force]
"""

from __future__ import annotations

import argparse
import copy
import os
import re

import build_v4_docx as v4b
from realworld_common import PROJECT_ROOT, die

OUT_DIR = os.path.join(PROJECT_ROOT, "paper_ieee_docx")
TEMPLATE = os.path.join(OUT_DIR, "ieee_conference_template_a4.docx")
V4 = v4b.V4
OUT = os.path.join(OUT_DIR, "Calibrated_Escalation_IConAECC2026.docx")
SUPP_DOCX = os.path.join(OUT_DIR, "SUPPLEMENT.docx")
SUPP_MD = os.path.join(OUT_DIR, "SUPPLEMENT.md")
FIG = v4b.FIG
FIG1 = os.path.join(OUT_DIR, "fig", "fig1_pipeline_cropped.png")
FIG2 = os.path.join(OUT_DIR, "fig", "fig2_thresholds.png")
FIG3 = os.path.join(FIG, "fig3_accuracy.png")
REPO_URL = "https://github.com/aryanwaingankar4/ticket-routing-resolution-agent"
SUPP_URL = REPO_URL + "/blob/iconaecc-2026/paper_ieee_docx/SUPPLEMENT.md"
COL_IN, WIDE_FIG_IN = 3.3, 5.3
ABSTRACT_MAX_WORDS = 200
DATASET_SENTENCE = (
    "We found no public, real, English IT ticket dataset whose categories "
    "match our seven teams; the Zenodo corpus was the closest real option, "
    "and our hand-written set covers all seven.")
ACK = (
    "The authors used AI assistants (Anthropic Claude) for code development, "
    "experiment scripting and language editing. All experiments were run and "
    "checked by the authors, and every reported number is regenerated from "
    "committed result files. The independent test tickets were written and "
    "labelled by the authors without AI assistance.")
SUPP_POINTER = ("Prompts, configurations, reproduction steps and additional "
                f"tables are in the online supplement at {SUPP_URL}.")
# Template sample text that must not survive into the paper.
TEMPLATE_LEFTOVERS = ("Paper Title", "Given Name", "dept. name", "This template",
                      "(Heading", "use style", "Identify applicable funding",
                      "IEEE conference templates contain guidance",
                      "Eason", "Table Head", "Example of a figure caption")


# ------------------------------------------------------------- reading v4
class V4Source:
    def __init__(self, path):
        import docx
        from docx.oxml.ns import qn
        self.qn = qn
        self.doc = docx.Document(path)
        body = self.doc.element.body
        self.items = []                       # ("p", text, el) / ("tbl", rows, el)
        for el in body:
            tag = el.tag.split("}")[1]
            if tag == "p":
                self.items.append(("p", self.ptext(el), el))
            elif tag == "tbl":
                self.items.append(("tbl", self.table_rows(el), el))
        self.texts = [t for k, t, _ in self.items if k == "p"]
        self.full = "\n".join(
            t if k == "p" else "\n".join(" | ".join(r) for r in t)
            for k, t, _ in self.items)

    def ptext(self, el):
        return "".join(t.text or "" for t in el.iter(self.qn("w:t")))

    def table_rows(self, tbl):
        rows = []
        for tr in tbl.findall(self.qn("w:tr")):
            rows.append(["\n".join(self.ptext(p) for p in tc.findall(self.qn("w:p")))
                         for tc in tr.findall(self.qn("w:tc"))])
        return rows

    def index(self, pred, what):
        hits = [i for i, (k, t, _) in enumerate(self.items)
                if k == "p" and pred(t)]
        if len(hits) != 1:
            die(f"v4: expected one paragraph for {what}, found {len(hits)}")
        return hits[0]

    def para(self, start):
        return self.items[self.index(lambda t: t.startswith(start), start)][1]

    def table(self, number):
        """(caption text, rows) of v4's TABLE <number>."""
        i = self.index(lambda t: t.strip() == f"TABLE {number}", number)
        cap, tbl = self.items[i + 1], self.items[i + 2]
        if cap[0] != "p" or tbl[0] != "tbl":
            die(f"v4: TABLE {number} is not caption + table")
        return cap[1], tbl[1]

    def between(self, first, last):
        a = self.index(lambda t: t == first, first)
        b = self.index(lambda t: t == last, last)
        return [t for k, t, _ in self.items[a + 1:b] if k == "p"]

    def fig_caption(self, n):
        return self.para(f"Fig. {n}. ")[len(f"Fig. {n}. "):]

    def references(self):
        refs = self.between("References", "Appendix A: Prompts")
        if len(refs) != 25:
            die(f"v4: expected 25 references, found {len(refs)}")
        return refs

    def authors(self):
        a = self.index(lambda t: t.startswith("Abstract—"), "abstract")
        lines = [t for k, t, _ in self.items[1:a] if k == "p" and t.strip()]
        if len(lines) != 8 or not lines[0].startswith("Aryan Manoj Waingankar"):
            die(f"v4: author block not as expected: {lines}")
        return lines


NUM_RE = re.compile(r"\d+(?:[.,]\d+)*")


def numbers_in(text):
    """Numeric tokens, ignoring citation brackets and cross-reference labels."""
    text = re.sub(r"\[\d+\](?:–\[\d+\])?", " ", text)
    text = re.sub(r"\b(?:Fig\.|Figs\.|Table|TABLE) (?:S-)?[IVX\d]+\b", " ", text)
    return set(NUM_RE.findall(text))


# ---------------------------------------------------------- the content
def H1(key, text):
    return ("h1", key, text)


def H2(key, text):
    return ("h2", key, text)


def P(text, lead=None):
    return ("p", text, lead)


def BUL(items):
    return ("bul", items)


def TAB(key, caption, rows, widths, size=None, subheads=()):
    return ("tbl", key, caption, rows, widths, size, subheads)


def FIGC(key, path, caption, width=COL_IN):
    return ("fig", key, path, caption, width, False)


def FIGW(key, path, caption, width=WIDE_FIG_IN):
    return ("fig", key, path, caption, width, True)


def content(v, src):
    pct = v4b.pct
    F = v4b.fills(v)[0]
    p_rw = "1.0" if v["rw_p"] == 1.0 else f"{v['rw_p']:.2f}"
    rw_pct, sens_pct = pct(v["rw_ok"], v["rw_n"]), pct(v["sens_ok"], v["rw_n"])
    ind_pct = pct(v["ind_ok"], v["ind_ncat"])
    v4b.check(v["ind_dis_out"] == 0 and v["ind_adj_writer"] == v["ind_adj"],
              "the adjudication wording assumes writer labels kept")

    abstract = (
        "A triage system for IT support tickets must route each ticket to the "
        "right team and recognise when to hand it to a person. Our pipeline "
        "has two separately calibrated escalation gates: a 0.50 confidence "
        "threshold on a fast lexical classifier decides whether a slower "
        "embedding classifier is consulted, and a 0.67 cosine-similarity "
        "threshold on retrieved past tickets decides whether a language model "
        "may draft a resolution. On a 9-ticket adversarial set the retrieval "
        "gate decides every ticket correctly, against 3 of 9 without it. A "
        "zero-shot LLM classifies more accurately (40/45 against 33/45) but "
        f"cannot abstain. On {v['ind_n']} tickets written and cross-labelled "
        f"by two authors (κ = {v['k_cat']:.3f}), the cascade routes "
        f"{v['ind_ok']} of {v['ind_ncat']} correctly, but the gate also "
        f"escalates {v['ind_act_esc']} actionable tickets. On {v['rw_n']} "
        "distinct real tickets covering two of seven teams, it routes "
        f"{rw_pct} correctly ({sens_pct} under an alternative mapping), and "
        f"{v['rw_mis_draft']} of {v['rw_mis']} misrouted tickets reach the "
        "drafter: a classifier trained only on synthetic templates does not "
        "transfer to real tickets, and the data a component is trained or "
        "calibrated on limits it more than the method chosen. Conclusions "
        "are scoped to small evaluation sets.")
    keywords = src.para("Index Terms—")[len("Index Terms—"):]

    cap_rel, rel = src.table("I")
    ours = rel[-1]
    v4b.check(ours[0] == "Ours" and f"({v['ind_n']} tickets)" in ours[3]
              and f"{v['rw_n']} real tickets" in ours[3], "Table I 'Ours' row")
    # Table I, shortened cells (no number changed; the citation [24] in the
    # 'Ours' row is dropped, the dataset being cited in Section IV).
    rel_short = [
        ["Work", "Task", "Abstention mechanism", "Evaluated on"],
        ["TickIt [1]", "Ticket escalation", "Routes tickets to support teams",
         "Deployment on Volcano Engine"],
        ["Paket et al. [2]", "Service desk ticket classification",
         "not stated", "Turkish service desk tickets, 172 classes"],
        ["Carroll et al. [3]", "REDCap ticket triage and drafting",
         "not stated", "Real REDCap tickets, expert-rated"],
        ["Jitkrittum et al. [4]", "Cascade deferral",
         "Confidence-based and post-hoc deferral", "not stated"],
        ["Conformal Cascade [5]", "Multi-tier LLM inference",
         "Defer unless conformal set is a singleton",
         "Multiple-choice benchmarks"],
        ["Fanconi et al. [6]", "Human–AI decision-making",
         "Deferral, then abstention to a human expert",
         "General and medical QA"],
        ["Sufficient context [13]", "RAG error analysis",
         "Sufficient-context guided abstention", "Several models and datasets"],
        ["CRAG [14]", "Robust RAG", "Retrieval evaluator triggers corrective "
         "actions", "Short- and long-form generation"],
        ["Ours", "IT ticket triage",
         "Two calibrated gates (confidence, retrieval)",
         f"Synthetic corpus; 45-ticket out-of-template set; {v['ind_n']} "
         f"author-written; {v['rw_n']} real tickets"],
    ]
    if [r[0] for r in rel_short] != [r[0] for r in rel]:
        die("Table I rows no longer match v4")

    cap_splits, splits = src.table("III")
    cap_gates, gates = src.table("V")
    cap_cov, cov = src.table("VI")
    cap_casc, casc = src.table("VII")
    cap_ext, ext = src.table("VIII")
    cap_err, err = src.table("IX")
    want = {"Independent": [str(v["ind_n"]), f"{v['ind_ok']}/{v['ind_ncat']}",
                            f"{v['ind_t2']}/{v['ind_ncat']}",
                            f"{v['ind_esc']}/{v['ind_n']} "
                            f"({pct(v['ind_esc'], v['ind_n'])})",
                            f"{v['ind_t1']}/{v['ind_n']}"],
            "Real-world": [str(v["rw_n"]), f"{v['rw_ok']}/{v['rw_n']}",
                           f"{v['rw_t2']}/{v['rw_n']}",
                           f"{v['rw_esc']}/{v['rw_n']} "
                           f"({pct(v['rw_esc'], v['rw_n'])})",
                           f"{v['rw_t1']}/{v['rw_n']}"]}
    for r in ext[1:]:
        if r[0] in want and r[1:] != want[r[0]]:
            die(f"v4 Table VIII row {r[0]} differs from the result files: {r}")

    groundedness = src.para("Every benchmark and adversarial ticket")
    # Limitations: v4's two paragraphs, tightened (every point kept).
    lim_src = (src.para("Synthetic data and small evaluation sets.")
               + src.para("No live evaluation and few human quality labels."))
    for need in ("[0.590, 0.840]", "±0.045", "[0.804, 0.983]", "30/33",
                 "−0.042"):
        v4b.check(need in lim_src, f"v4 limitations contain {need}")

    blocks = [
        H1("intro", "Introduction"),
        P("A triage system that routes a ticket confidently but wrongly costs "
          "more than one that asks for help, so when it declines to answer "
          "matters as much as its accuracy. In our pipeline escalation to a "
          "human is a first-class "
          "output: a two-tier cascade assigns one of seven teams, similar "
          "resolved tickets are retrieved, and a language model may draft a "
          "resolution only when the closest past ticket is similar enough. "
          "Both thresholds come from measured calibration data (Section "
          "{S:thr}), and a zero-shot LLM, more accurate but unable to abstain "
          "(Section {S:zs}), is kept as a baseline. The contributions are:"),
        BUL([
            ("A two-gate escalation pipeline", " whose thresholds are "
             "calibrated independently, with the evidence for each value and "
             "the calibration attempts we rejected."),
            ("A recurring finding:", " the data distribution used to fit or "
             "calibrate a component constrains it more than the method or "
             "test applied to it."),
            ("Transparent baselines and error analysis:", " a controlled "
             "ablation, zero-shot LLM baselines from two vendors, an audit of "
             "an LLM judge, and a breakdown of misclassifications, wrong "
             "escalations and unsupported drafts."),
            ("An extended evaluation", f" on {v['ind_n']} tickets written and "
             f"cross-labelled by two authors (κ = {v['k_cat']:.3f}) and "
             f"{v['rw_n']} real tickets from a public dataset, reported with "
             "its negative result."),
        ]),

        H1("rw", "Related Work"),
        P("LLM-based methods route and escalate IT and customer-support "
          "tickets [1]–[3] and mainly report routing accuracy; our focus is "
          "the decision to abstain. Confidence-based cascades pass difficult "
          "inputs to a more capable model [4], conformal variants give "
          "distribution-free guarantees for multi-tier cascades [5], and a "
          "cascade can end with a human [6]. Split conformal prediction "
          "guarantees marginal coverage under exchangeability [7], [8]; "
          "weighted conformal prediction restores it under covariate shift "
          "when the likelihood ratio is known [9], and later work bounds the "
          "loss when exchangeability fails [10] and adapts online [11]; our "
          "shift is a change in writing style. A retrieval-augmented "
          "generation (RAG) system [12] can abstain when the retrieved context "
          "is insufficient [13] or correct itself when retrieval is poor "
          "[14]. LLM judges can come with guaranteed human agreement [15] and "
          "show an agreeableness bias [16]; the judge we audited shows the "
          "failure these works warn about. The value of synthetic data as a "
          "benchmark has itself been studied [17]. Table {T:rel} compares the "
          "closest works using only their abstracts: two ticket-triage "
          "studies evaluate on real tickets, while our main evaluation is "
          "synthetic (Section {S:ext}), and none combines gates calibrated on "
          "measured data with generation permitted only after retrieval "
          "passes such a gate."),
        TAB("rel", cap_rel, rel_short, [1050, 1150, 1400, 1400], size=14),

        H1("meth", "Methodology"),
        H2("pipe", "Pipeline and gates"),
        P("Fig. {F:pipe} shows the pipeline: sequential, with agent "
          "boundaries and an HTTP interface, not an autonomous multi-agent "
          "system. One orchestrator takes every routing decision and traces "
          "every stage, so a skipped resolution step is direct evidence that "
          "no language model was called; a failed stage escalates the ticket. "
          "Tier 1 is TF-IDF with "
          "logistic regression over a committed vocabulary; its label is "
          "accepted if its highest class probability is at least 0.50, and "
          "otherwise Tier 2, logistic regression over frozen BGE embeddings "
          "[18] (bge-base-en-v1.5, 768 dimensions), classifies the ticket. "
          "The five nearest resolved tickets are retrieved from a FAISS "
          "inner-product index [19]. Below a top-1 cosine similarity of 0.67 "
          "the ticket is escalated and no language model is called; "
          "otherwise Gemini (gemini-flash-lite) drafts a resolution based "
          "only on the retrieved resolutions, recommending a human check when "
          "they do not fit. The language model never assigns the category. "
          "The cascade threshold depends only on Tier-1 confidence, so the "
          "Tier-2 encoder can be replaced without recalibrating it; the "
          "retrieval threshold must be recalibrated with the encoder. "
          "Offline, resolved "
          "tickets are clustered by resolution-text similarity at 0.80 into "
          "automation candidates for a reviewer."),
        FIGC("pipe", FIG1, src.fig_caption(1)),
        H2("conf", "Conformal measurement and verification"),
        P("To test whether a finite-sample guarantee could replace the "
          "cascade threshold, we calibrate split conformal prediction [7], "
          "[8] for each tier on a held-out set; its prediction sets have "
          "marginal miscoverage at most α when calibration and test tickets "
          "are exchangeable. It runs offline and gates nothing. Automated "
          "checks (online supplement) run before any result is accepted: "
          "routing decisions are compared exactly, and floating-point values "
          "within tolerances derived from their arithmetic."),

        H1("setup", "Experimental Setup"),
        P("Following the nasscom use-case brief, which specified seven "
          "categories, routing, resolution suggestion, confidence-based "
          "escalation and automation flagging and recommended a synthetic, "
          "LLM-generated dataset, we generated 4,000 tickets with title, "
          "description, category, resolution and priority fields. "
          + F["D1"] + " The synthetic corpus, a requirement of the brief, "
          "limits what can be claimed (Section {S:lim}): the "
          "generation templates coincide with the resolution classes, so "
          "clustering precision cannot be measured, and memorisation is "
          "template-level (the calibration set touches 62 of the 66 "
          "templates, and removing whole templates would leave 210 of the "
          "4,000 rows), so the corpus cannot be decontaminated by filtering.",
          lead="Dataset."),
        P("Held-out accuracy on generated data is uninformative (the TF-IDF "
          "baseline scores 100%). Table {T:splits} lists every split. Tier 2 "
          "and DistilBERT are trained on a stratified 3,200-row split (seed "
          "42); the production Tier 1 is fitted on all 4,000 rows. Evaluation "
          "uses fixed, read-only sets written outside the templates: a "
          "hand-written 14-ticket benchmark, a 45-ticket out-of-template "
          "benchmark, a 9-ticket adversarial set of off-topic, vague and slang "
          "tickets labelled with the expected escalate-or-proceed outcome, "
          "and a 175-ticket deployment-distribution set; the last two "
          "Gemini-generated. "
          + F["D2"].replace("(Section VI-E)", "(Section {S:ext})"),
          lead="Evaluation sets."),
        TAB("splits", cap_splits, splits, [1150, 800, 1350, 1700], size=14),
        P("Proportions carry Wilson 95% intervals [20], paired comparisons "
          "use the exact McNemar test [21], and a difference smaller than its "
          "noise band is never reported as an ordering. Latency is the warm "
          "median per ticket at batch size one. Every number is produced by "
          "scripts from committed result files, and a parity test fails if a "
          "quoted number differs from its source. Code, data and results are "
          "in a public repository. " + SUPP_POINTER,
          lead="Statistics and reproducibility."),

        H1("thr", "Threshold Selection and Calibration Evidence"),
        H2("thr_c", "Cascade threshold (0.50)"),
        P("Of three calibration attempts we kept one. On an in-distribution "
          "split accuracy was 1.0 in every confidence bin, so both tiers "
          "appeared under-confident (expected calibration error 0.112 for "
          "Tier 1, 0.099 for Tier 2) and any threshold looked safe; a second "
          "attempt, on a small hand-written set, was rejected and never "
          "committed. The third used paraphrased tickets and chose the "
          "lowest threshold at which the accuracy of the tickets Tier 1 keeps "
          "meets the target on 10%-wide confidence buckets (Fig. "
          "{F:thr}(a)). No threshold met the 0.90 target; the 0.80 and 0.70 "
          "targets both gave 0.50, the live value, at which Tier 1 keeps 23 "
          "of the 175 tickets (13.1%) and labels 19 correctly (82.6%). The "
          "curve is not monotone: above 0.60 only 10 tickets remain, 6 "
          "correct. On a 0.01 grid the rule would choose 0.43 and 0.36; kept "
          "accuracy peaks at 25/30 at 0.46, so 0.50 sits near the peak."),
        H2("thr_r", "Retrieval threshold (0.67)"),
        P("Swept over in-domain and out-of-domain calibration tickets (Fig. "
          "{F:thr}(b)), it lets all 175/175 in-domain tickets proceed at 0.67 "
          "while 13.3% of out-of-domain tickets leak through; two grid steps "
          "lower, leakage almost triples to 37.8%. On the adversarial set any "
          "threshold between the highest similarity of a ticket that must "
          "escalate (0.6196) and the lowest of one that must proceed (0.6704) "
          "separates the groups; 0.67 lies inside, and 9/9 adversarial "
          "tickets are decided correctly with the gate against 3/9 without "
          "it (Table {T:gates}). As 0.67 lies near the top of the safe range, "
          "a small encoder change could move a ticket across it, and one "
          "threshold is the simplest rule that works here, not a claim that "
          "one number suffices in general."),
        FIGC("thr", FIG2,
             "Threshold calibration. (a) Cascade threshold on paraphrased "
             "tickets; the dashed line marks the live threshold of 0.50. (b) "
             "Retrieval threshold sweep; the shaded band is the adversarial "
             "safe range [0.6196, 0.6704], and the dashed line marks the live "
             "threshold of 0.67."),
        TAB("gates", cap_gates, gates, [3700, 1300],
            subheads=("Retrieval gate", "Cascade gate")),

        H1("res", "Results"),
        H2("cov", "Coverage transfer and the calibration distribution"),
        P("Split-conformal coverage on the 45-ticket benchmark (Table "
          "{T:cov}) gives three linked statements."),
        BUL([
            ("", "On our corpus, coverage transfers for the BGE "
             "representation but not for the lexical one: at α = 0.10 the gap "
             "is −0.233 for Tier 1 and −0.011 for Tier 2, against a noise band "
             "of ±0.045 (two standard deviations of a binomial coverage "
             "estimate, n = 175)."),
            ("", "External replication is inconclusive. On an independently "
             "generated corpus (not real production data), a version shift "
             "left the tiers indistinguishable (0.0009 against −0.0072, band "
             "±0.0165), but that is the wrong kind of shift for a "
             "vocabulary-level effect, and that corpus shows almost no "
             "representation gap (label accuracy 34.8% against 37.3%). A "
             "paraphrase shift was blocked by our own pre-registered "
             "degeneracy rule (domain AUC 0.9972), so the finding has been "
             "shown neither to hold nor to fail outside its corpus."),
            ("", "Post-hoc, the paraphrase shift cost Tier 1 10.5 accuracy "
             "points against 2.4 for Tier 2 (difference-in-differences "
             "−0.08042, 95% CI [−0.136364, −0.020979]); this does not replace "
             "the blocked test."),
        ]),
        P("Recalibrating on the deployment-distribution set, with nothing "
          "else changed, moves the Tier-1 gap from −0.233 to −0.144 (for Tier "
          "2 the benefit appears as smaller prediction sets), and "
          "density-ratio reweighting [9] is also only a partial repair, so "
          "the lexical tier's guarantee cannot be quoted on out-of-template "
          "text and conformal prediction remains a measurement, not a gate."),
        TAB("cov", cap_cov, cov, [450, 550, 1100, 1150, 900, 850]),
        H2("lat", "The cascade saves latency, not accuracy"),
        P("Tier 1 alone scores 16/45, 35.6 points below the full pipeline, "
          "but that is the BGE-versus-TF-IDF representation gap, not the "
          "value of cascading. Against the fair control, Tier 2 answering "
          "every ticket (Table {T:casc}), the cascade is one ticket worse on "
          "each set (32/45 against 33/45, and 131/175 against 132/175; exact "
          "McNemar p = 1.000 for both). It buys speed: Tier 2 costs 151.0× as "
          "much as Tier 1 (156.4 ms against 1.04 ms median), so median "
          "latency falls by 8% on the benchmark and 18% on the deployment "
          "set."),
        TAB("casc", cap_casc, casc, [900, 800, 750, 850, 950, 750]),
        H2("zs", "Zero-shot LLM baselines"),
        P("Gemini (gemini-flash-lite) and Qwen2.5-3B, a 3-billion-parameter "
          "open-weight model run on a laptop CPU [22], received byte-identical "
          "prompts (Fig. {F:acc}). Gemini scores 40/45 (95% CI [0.765, "
          "0.952]), significantly better than Tier 2 at 33/45 (p = 0.039); "
          "Qwen2.5-3B scores 34/45 ([0.613, 0.858]), indistinguishable from "
          "Tier 2 (p = 1.000), which does not mean better; Qwen2.5-3B against "
          "Gemini gives p = 0.070, not significant. As the benchmark was "
          "generated with Gemini, Qwen is a "
          "partial control for authorship, and Gemini's margin cannot be "
          "attributed to capability alone. This is not like-for-like: it "
          "measures what a template-generated training set is worth on "
          "differently phrased tickets, not whether an LLM outperforms the "
          "pipeline, whose contribution is the calibrated escalation neither "
          "zero-shot model provides. Among trained classifiers, "
          "representation matters more than model family: two other frozen "
          "encoders with the same logistic head score 32/45 and 27/45, and a "
          "fine-tuned DistilBERT 18/45 and 21/45 in two runs that differ, so "
          "we order neither."),
        FIGC("acc", FIG3, src.fig_caption(3), width=2.8),
        H2("grd", "Groundedness, an LLM judge, and a sufficiency gate"),
        P("All 33 benchmark and adversarial tickets that passed the "
          "retrieval gate received a draft, and a human labelled 31/33 as "
          "grounded in the retrieved text (95% CI [0.804, 0.983]). An LLM "
          "judge using the same rubric agreed on 30/33 yet reached Cohen's "
          "κ = −0.042 [23], every disagreement being the judge rating "
          "appropriateness instead of support, so we treat an LLM judge as "
          "something to audit, not a way to scale labelling. A "
          "retrieval-sufficiency rater that never sees the draft caught 2/2 "
          "ungrounded drafts but flagged 26/31 grounded ones, so it cannot be "
          "used as a gate."),
        H2("ext", "Extended evaluation on independent and real-world tickets"),
        P("To test whether these results depend on Gemini-written evaluation "
          f"text, the unchanged pipeline was run on two more sets (Table "
          f"{{T:ext}}). Two authors each wrote {v['ind_per_author']} tickets, "
          "seven per category plus five off-topic or vague ones "
          f"({v['ind_n']} in total), without the training data, templates, "
          "benchmarks or LLM assistance, and each labelled the other's "
          f"tickets blind. Before adjudication Cohen's κ was {v['k_cat']:.3f} "
          "for "
          f"category (bootstrap 95% CI {v['k_cat_ci']}; {v['k_cat_agree']} "
          f"agreement) and {v['k_out']:.3f} for outcome "
          f"({v['k_out_agree']}); in all {v['ind_adj_writer']} adjudicated "
          "disagreements the writer's label was kept. The real-world set is "
          f"the CC BY 4.0 dataset of Pereira [{v4b.REF_PEREIRA}]: 2,229 "
          "user-typed tickets of an IT-support company, labelled by three IT "
          "professionals, of which langid.py "
          f"[{v4b.REF_LANGID}] identifies {v['rw_english']:,} as English. A "
          "mapping fixed before predictions were examined sends Active "
          "Directory and Fileservice (almost entirely file-share access "
          "requests) to Access Management and Software and O365 to "
          f"Application; the other three categories ({v['rw_excluded']} "
          "tickets) have no counterpart, so Database, Infrastructure, "
          "Network, Security and Storage are not covered. Of the "
          f"{v['rw_all_n']} mapped tickets {v['rw_dup']} repeat an earlier "
          f"text exactly, so we report the {v['rw_n']} distinct texts, with "
          f"all {v['rw_all_n']} alongside. " + DATASET_SENTENCE),
        TAB("ext", cap_ext, ext, [1250, 450, 800, 750, 1000, 750]),
        P(f"On the independent set the cascade routes {v['ind_ok']} of the "
          f"{v['ind_ncat']} tickets that belong to a team correctly "
          f"({ind_pct}, 95% CI {v['ind_ok_ci']}; Tier 2 alone "
          f"{v['ind_t2']}/{v['ind_ncat']}), similar to the benchmark's "
          f"32/45, with per-team recall from {v['ind_rec_sec']} (Security) to "
          f"{v['ind_rec_db']} (Database). The gate escalates "
          f"{v['ind_esc']}/{v['ind_n']} ({pct(v['ind_esc'], v['ind_n'])} "
          f"{v['ind_esc_ci']}): {v['ind_se']} of the {v['ind_se_n']} "
          f"off-topic or vague tickets, but also {v['ind_act_esc']} of the "
          f"{v['ind_sp_n']} actionable ones, so the escalate-or-proceed "
          f"decision is correct for {v['ind_out']}/{v['ind_n']} "
          f"({pct(v['ind_out'], v['ind_n'])} {v['ind_out_ci']}). On real "
          f"tickets the classifier does not transfer: the cascade routes "
          f"{v['rw_ok']} of {v['rw_n']} distinct texts correctly ({rw_pct}, "
          f"95% CI {v['rw_ok_ci']}; {v['rw_all_ok']}/{v['rw_all_n']}, "
          f"{pct(v['rw_all_ok'], v['rw_all_n'])}, over all tickets), with "
          f"recall {v['rw_rec_am']} for Access Management and "
          f"{v['rw_rec_app']} for Application, and Tier 2 alone does no "
          f"better ({v['rw_t2']}/{v['rw_n']}; exact McNemar p = {p_rw}). "
          f"Tier 1 answers {v['rw_t1']}/{v['rw_n']} "
          f"({pct(v['rw_t1'], v['rw_n'])} {v['rw_t1_ci']}, against 4/45 on "
          f"generated text), but only {v['rw_t1_ok']} correctly. The gate "
          f"escalates {v['rw_esc']}/{v['rw_n']} "
          f"({pct(v['rw_esc'], v['rw_n'])} {v['rw_esc_ci']}; "
          f"{v['rw_all_esc']}/{v['rw_all_n']}, "
          f"{pct(v['rw_all_esc'], v['rw_all_n'])}, over all tickets) but does "
          f"not filter misroutes: {v['rw_mis_draft']} of the {v['rw_mis']} "
          "misrouted texts still reached the drafter. Mapping the file-share "
          "access requests to Storage instead, where our generator files "
          f"share-permission faults, gives {v['sens_ok']}/{v['rw_n']} "
          f"({sens_pct}, {v['sens_ok_ci']}; online supplement); the "
          "escalation rate and Tier-1 share do not depend on the mapping. A "
          "classifier trained only on synthetic templates does not transfer "
          "to real tickets, and gates calibrated on the same material do not "
          f"compensate. The {v['n_drafts']} drafts generated for a sample of "
          "the real tickets were not labelled."),
        P("The data a component is fitted or calibrated on constrains it more "
          "than the method or test applied to it: beyond the coverage gap "
          "(Section {S:cov}), legitimate deployment-style tickets trigger the "
          "in-domain drift reference at rates of 0.217 to 0.646 "
          "across four α levels, 3.2 to 38.2 times the per-ticket null, so a "
          "monitor on that reference would alarm continuously whatever the "
          "test, α or window, and training on the corpus transfers to "
          "out-of-template phrasing no better than no training at all "
          "(Section {S:zs}). Methodological lessons are in the supplement.",
          lead="The recurring finding."),

        H1("err", "Error Analysis"),
        P("Table {T:err} groups the errors of the live pipeline by type."),
        TAB("err", cap_err, err, [1300, 600, 3100], size=14),
        P("The cascade misroutes 13 of the 45 benchmark tickets (confusion "
          "matrix in the online supplement); the most frequent confusions are "
          "Database→Application, Infrastructure→Application and "
          "Security→Storage (2 each), and 10 of the 13 errors predict "
          "Application or Storage. Infrastructure is hardest: recall is 50% "
          "for the trained classifier and for both zero-shot LLMs, which "
          "suggests ambiguous wording rather than one weak method. Tier 1 "
          "alone misclassifies 29 of 45, mostly where the wording shares no "
          "vocabulary with the templates (examples in the supplement).",
          lead="Misclassification."),
        P("Of the 13 misclassified benchmark tickets, 5 were escalated "
          "(safe) and 8 drafted (unsafe): the classification gate does not "
          "stop a misrouted ticket from reaching the drafter, a limitation of "
          f"the current design, seen at scale in {v['ind_mis_draft']} of "
          f"{v['ind_mis']} misrouted independent and {v['rw_mis_draft']} of "
          f"{v['rw_mis']} misrouted real texts. The gate escalated "
          f"{v['rw_mis_esc']} misrouted and {v['rw_ok_esc']} correctly routed "
          "real texts, so its similarity does not track routing correctness; "
          f"nor does Tier-1 confidence ({v['rw_t1_ok']} of the {v['rw_t1']} "
          "real texts kept at 0.50 were correct).",
          lead="Incorrect escalation decisions."),
        P("In both ungrounded drafts the model noted that the retrieved "
          "tickets did not match (N26 retrieved fixes for failing disks "
          "behind network file shares, N31 NTFS permission fixes), then "
          "supplied steps from outside knowledge, which the prompt forbids, "
          "and asked for human review; N31 had also been misrouted to "
          "Storage. Neither the sufficiency rater nor the LLM judge (Section "
          "{S:grd}) can yet replace human review.",
          lead="Unsupported resolutions."),

        H1("lim", "Limitations"),
        P("All training data is synthetic and template-generated, and the "
          "evaluation sets are small: at 45 tickets the Wilson interval on "
          "Tier 2's 33/45 is [0.590, 0.840] and the conformal noise band at "
          "α = 0.10 is ±0.045, so most comparisons between strong methods "
          "cannot be ordered. The benchmark and deployment set were written "
          "by Gemini, which also drafts resolutions and forms one baseline, "
          "and the external transfer corpus is itself generated. The "
          f"independent set's {v['ind_n']} tickets were written and labelled "
          "by two authors who know the categories, so it is small and is not "
          "real traffic. The real-world set covers two teams only, as no "
          "public, real, English IT ticket dataset with matching categories "
          "was available: Database, "
          "Infrastructure, Network, Security and Storage remain untested on "
          "real tickets. Its accuracy depends strongly on the category "
          f"mapping ({rw_pct} or {sens_pct}), the tickets come from one "
          "organisation, and the English filter let a few non-English "
          "tickets through, because their system-generated subject lines "
          "were in English.", lead="Synthetic data and small evaluation sets."),
        P(f"The system has not been deployed: {v['rw_n']} real tickets were "
          "routed offline, but no live stream has passed through it and no "
          "help-desk user has acted on its output, so its effect on "
          "resolution time, misrouting cost and user trust is unmeasured, and "
          "the drift monitor has no deployment-traffic reference. Resolution "
          "quality rests on human labels for only 33 drafts; an LLM judge "
          "could not scale them, its 30/33 raw agreement concealing a Cohen's "
          "κ of −0.042, so the share of grounded drafts is known only within "
          "[0.804, 0.983].",
          lead="No live evaluation and few human quality labels."),

        H1("concl", "Conclusion and Future Work"),
        P("A triage pipeline is valuable only if it knows when not to answer. "
          "Our two separately calibrated gates decide when a lexical "
          "classifier may answer alone and when a language model may draft a "
          "fix grounded in past tickets. The retrieval gate decides 9/9 "
          "adversarial tickets correctly against 3/9 without it, and the "
          "cascade cuts median latency by 8% to 18% without improving "
          "accuracy. These conclusions are established on a synthetic corpus, "
          f"small LLM-generated evaluation sets, {v['ind_n']} hand-written "
          f"tickets and {v['rw_n']} real tickets that cover only two of the "
          "seven teams, and they should not be read as performance claims for "
          "a production help desk. On real tickets the classifier did not "
          f"transfer ({rw_pct} routed correctly, {sens_pct} under the "
          f"alternative mapping; {v['rw_mis_draft']} of {v['rw_mis']} "
          "misroutes reached the drafter): the synthetic training data, not "
          "the choice of method or threshold, limits what the pipeline can "
          "do. Future work is live deployment, to recalibrate both gates on "
          "real tickets and measure misrouting cost; more human-labelled "
          "drafts, with sufficiency as well as support labels; and a larger "
          "real-ticket benchmark, to test the recurring finding outside our "
          "corpus."),
    ]
    return {"abstract": abstract, "keywords": keywords, "blocks": blocks,
            "title": src.texts[0], "authors": src.authors(),
            "refs": src.references()}


# ----------------------------------------------------- numbering + refs
ROMAN = ["I", "II", "III", "IV", "V", "VI", "VII", "VIII", "IX", "X", "XI",
         "XII"]


def resolve(c):
    """Number sections/tables/figures, resolve {S:..}/{T:..}/{F:..} keys,
    renumber citations by first appearance, and return the new reference
    order and a label map for the checks."""
    sec, tab, fig = {}, {}, {}
    h1 = h2 = 0
    for b in c["blocks"]:
        if b[0] == "h1":
            h1, h2 = h1 + 1, 0
            sec[b[1]] = ROMAN[h1 - 1]
        elif b[0] == "h2":
            h2 += 1
            sec[b[1]] = f"{ROMAN[h1 - 1]}-{'ABCDEFGH'[h2 - 1]}"
        elif b[0] == "tbl":
            tab[b[1]] = ROMAN[len(tab)]
        elif b[0] == "fig":
            fig[b[1]] = str(len(fig) + 1)
    maps = {"S": sec, "T": tab, "F": fig}

    def sub_keys(text):
        def rep(m):
            kind, key = m.group(1), m.group(2)
            if key not in maps[kind]:
                die(f"unknown {kind} key {key!r}")
            return maps[kind][key]
        return re.sub(r"\{([STF]):(\w+)\}", rep, text)

    def walk(fn):
        out = []
        for b in c["blocks"]:
            if b[0] == "p":
                out.append(("p", fn(b[1]), b[2]))
            elif b[0] == "bul":
                out.append(("bul", [(h, fn(t)) for h, t in b[1]]))
            elif b[0] == "tbl":
                out.append(b[:2] + (fn(b[2]), [[fn(x) for x in r] for r in b[3]])
                           + b[4:])
            elif b[0] == "fig":
                out.append(b[:3] + (fn(b[3]),) + b[4:])
            else:
                out.append(b)
        return out

    c["blocks"] = walk(sub_keys)
    # Table and figure order of first citation must equal appearance order.
    flow = " ".join(texts_of(c["blocks"], cells=False))
    for kind, word, m in (("T", "Table", tab), ("F", "Fig.", fig)):
        first = []
        for x in re.findall(rf"{re.escape(word)} (\w+)", flow):
            x = x
            if x not in first:
                first.append(x)
        if first != list(m.values()):
            die(f"{word} cited in order {first}, numbered {list(m.values())}")
    # Citations: order of first appearance across text, tables and captions.
    order = []
    for t in texts_of(c["blocks"]):
        for a, b in re.findall(r"\[(\d+)\](?:–\[(\d+)\])?", t):
            rng = range(int(a), int(b) + 1) if b else [int(a)]
            for n in rng:
                if n not in order:
                    order.append(n)
    if sorted(order) != list(range(1, 26)):
        die(f"not every reference is cited: {sorted(set(range(1, 26)) - set(order))}")
    new = {old: k + 1 for k, old in enumerate(order)}

    def renum(text):
        def rng(m):
            a, b = int(m.group(1)), int(m.group(2))
            if [new[x] for x in range(a, b + 1)] != list(range(new[a], new[b] + 1)):
                die(f"citation range [{a}]–[{b}] is no longer consecutive")
            return f"[{new[a]}]–[{new[b]}]"
        text = re.sub(r"\[(\d+)\]–\[(\d+)\]", rng, text)
        return re.sub(r"\[(\d+)\](?!–)", lambda m: f"[{new[int(m.group(1))]}]",
                      text) if "–[" not in text else _renum_mixed(text, new)

    c["blocks"] = walk(renum)
    c["refs"] = [c["refs"][old - 1] for old in order]
    return maps, new


def _renum_mixed(text, new):
    """Renumber single citations in text that also holds renumbered ranges."""
    parts = re.split(r"(\[\d+\]–\[\d+\])", text)
    return "".join(p if re.fullmatch(r"\[\d+\]–\[\d+\]", p) else
                   re.sub(r"\[(\d+)\]", lambda m: f"[{new[int(m.group(1))]}]", p)
                   for p in parts)


def texts_of(blocks, cells=True):
    out = []
    for b in blocks:
        if b[0] in ("h1", "h2"):
            out.append(b[2])
        elif b[0] == "p":
            out.append((b[2] or "") + " " + b[1])
        elif b[0] == "bul":
            out.extend(h + t for h, t in b[1])
        elif b[0] == "tbl":
            out.append(b[2])
            if cells:
                out.extend(x for r in b[3] for x in r)
        elif b[0] == "fig":
            out.append(b[3])
    return out


# ------------------------------------------------------------ docx writer
class Writer:
    def __init__(self, template):
        import docx
        from docx.oxml.ns import qn
        from docx.shared import Inches, Pt
        self.docx, self.qn, self.Inches, self.Pt = docx, qn, Inches, Pt
        try:
            self.d = docx.Document(template)
        except KeyError:
            die("the template is Strict OOXML; see paper_ieee_docx/README.md")
        body = self.d.element.body
        W = self.W = lambda t: qn("w:" + t)  # noqa: E731
        sects = list(body.iter(W("sectPr")))
        self.sect_title = copy.deepcopy(sects[0])
        two = [s for s in sects if s.find(W("cols")) is not None
               and s.find(W("cols")).get(W("num")) == "2"]
        if len(two) != 1:
            die("template: expected one two-column section")
        self.sect_two = copy.deepcopy(two[0])
        self.sect_one = copy.deepcopy(sects[-1])   # one column, continuous
        for s in (self.sect_title, self.sect_two, self.sect_one):
            for tag in ("headerReference", "footerReference", "titlePg"):
                for x in s.findall(W(tag)):
                    s.remove(x)
        tbls = list(body.iter(W("tbl")))
        self.tbl_pr = copy.deepcopy(tbls[0].find(W("tblPr")))
        for x in self.tbl_pr.findall(W("tblW")):
            x.set(W("w"), "0")
            x.set(W("type"), "auto")
        final = body.find(W("sectPr"))
        for x in final.findall(W("headerReference")) + final.findall(
                W("footerReference")) + final.findall(W("titlePg")):
            final.remove(x)
        for el in list(body):
            if el is not final:
                body.remove(el)
        self.body = body

    def para(self, text="", style="Body Text", lead=None, size=None,
             italic_lead=False, align=None, keep_next=False):
        p = self.d.add_paragraph(style=style)
        if lead:
            r = p.add_run(lead + " ")
            r.bold = not italic_lead
            r.italic = italic_lead
            if size:
                r.font.size = self.Pt(size)
        if text:
            r = p.add_run(text)
            if size:
                r.font.size = self.Pt(size)
        if align is not None:
            p.alignment = align
        if keep_next:
            p.paragraph_format.keep_with_next = True
        return p

    def end_section(self, p, sect):
        ppr = p._p.get_or_add_pPr()
        ppr.append(copy.deepcopy(sect))

    def no_numbering(self, p):
        ppr = p._p.get_or_add_pPr()
        num = ppr.makeelement(self.W("numPr"), {})
        for tag, val in (("ilvl", "0"), ("numId", "0")):
            e = num.makeelement(self.W(tag), {self.W("val"): val})
            num.append(e)
        st = ppr.find(self.W("pStyle"))
        (st.addnext(num) if st is not None else ppr.insert(0, num))

    def figure(self, path, width_in, caption, label=None):
        from docx.enum.text import WD_ALIGN_PARAGRAPH
        p = self.d.add_paragraph(style="Normal")
        p.alignment = WD_ALIGN_PARAGRAPH.CENTER
        pf = p.paragraph_format
        pf.keep_with_next, pf.first_line_indent = True, 0
        pf.space_before, pf.space_after = self.Pt(4), self.Pt(2)
        p.add_run().add_picture(path, width=self.Inches(width_in))
        c = self.d.add_paragraph(style="figure caption")
        if label:
            self.no_numbering(c)
            c.add_run(label + " ")
        c.add_run(caption)
        return c

    def table(self, caption, rows, widths, size=None, subheads=(), label=None):
        W, qn = self.W, self.qn
        h = self.d.add_paragraph(style="table head")
        h.paragraph_format.keep_with_next = True
        if label:
            self.no_numbering(h)
            h.add_run(label + ". ")
        h.add_run(caption)
        t = self.d.add_table(rows=len(rows), cols=len(rows[0]))
        tbl = t._tbl
        tbl.remove(tbl.tblPr)
        tbl.insert(0, copy.deepcopy(self.tbl_pr))
        for g, w in zip(tbl.tblGrid.findall(W("gridCol")), widths):
            g.set(W("w"), str(w))
        for i, (row, tr) in enumerate(zip(rows, t.rows)):
            trpr = tr._tr.get_or_add_trPr()
            trpr.append(trpr.makeelement(W("cantSplit"), {}))
            if i == 0:
                trpr.append(trpr.makeelement(W("tblHeader"), {}))
            merged = row[0] in subheads and not any(x.strip() for x in row[1:])
            cells = tr.cells
            if merged:
                cells[0].merge(cells[-1])
                cells = [tr.cells[0]]
                row = [row[0]]
            for j, (cell, text) in enumerate(zip(cells, row)):
                tcw = cell._tc.get_or_add_tcPr().get_or_add_tcW()
                tcw.set(W("w"), str(sum(widths) if merged else widths[j]))
                tcw.set(W("type"), "dxa")
                p = cell.paragraphs[0]
                p.style = self.d.styles["table col head" if i == 0
                                        else "table copy"]
                lines = text.split("\n")
                for k, line in enumerate(lines):
                    r = p.add_run(line)
                    if k < len(lines) - 1:
                        r.add_break()
                    if size:
                        r.font.size = self.Pt(size / 2)
                    if merged:
                        r.italic = True
                if i < len(rows) - 1:
                    p.paragraph_format.keep_with_next = True
        return t

    def save(self, path):
        self.d.save(path)


def write_paper(c, path):
    from docx.enum.text import WD_ALIGN_PARAGRAPH
    w = Writer(TEMPLATE)
    w.para(c["title"], style="paper title")
    names = c["authors"][:2]
    for k, line in enumerate(c["authors"]):
        p = w.para(style="Author")
        pf = p.paragraph_format
        pf.space_before = w.Pt(6 if k in (0, 2) else 0)
        pf.space_after = w.Pt(0)
        r = p.add_run(line)
        if line not in names:
            r.font.size = w.Pt(9)
        if k in (2, 3):
            r.italic = True
    w.end_section(p, w.sect_title)
    a = w.para(style="Abstract")
    r = a.add_run("Abstract")
    r.italic = True
    a.add_run("—" + c["abstract"])
    k = w.para(style="Keywords")
    k.add_run("Index Terms—" + c["keywords"])
    last = k
    for b in c["blocks"]:
        if b[0] == "h1":
            last = w.para(b[2], style="Heading 1")
        elif b[0] == "h2":
            last = w.para(b[2], style="Heading 2")
        elif b[0] == "p":
            last = w.para(b[1], lead=b[2])
        elif b[0] == "bul":
            for lead, text in b[1]:
                last = w.para(text.lstrip(), style="bullet list",
                              lead=lead.rstrip() or None)
        elif b[0] == "tbl":
            _, key, cap, rows, widths, size, subheads = b
            w.table(cap, rows, widths, size=size, subheads=subheads)
            last = None
        elif b[0] == "fig":
            _, key, fpath, cap, width, wide = b
            if wide:
                if last is None:
                    last = w.para(style="Normal")
                w.end_section(last, w.sect_two)
            last = w.figure(fpath, width, cap)
            if wide:
                w.end_section(last, w.sect_one)
    w.para("Acknowledgment", style="Heading 5")
    w.para(ACK)
    w.para("References", style="Heading 5")
    for ref in c["refs"]:
        last = w.para(ref, style="references")
    w.end_section(last, w.sect_two)
    tail = w.para(style="Normal")
    tail.paragraph_format.space_before = tail.paragraph_format.space_after = w.Pt(0)
    tail.add_run("").font.size = w.Pt(1)
    del WD_ALIGN_PARAGRAPH
    cp = w.d.core_properties
    cp.title, cp.author = c["title"], "; ".join(
        x.strip() for x in ", ".join(names).split(",") if x.strip())
    cp.subject = cp.keywords = cp.comments = cp.category = ""
    cp.last_modified_by = cp.author
    w.save(path)
    return w


# -------------------------------------------------------------- supplement
def supplement(src, v):
    """Blocks of the online supplement: v4's appendices, unchanged."""
    prompts = src.between("Appendix A: Prompts", "Appendix B: Reproduction Steps")
    steps = src.between("Appendix B: Reproduction Steps",
                        "Appendix C: Sensitivity of the Real-World Result to "
                        "the Category Mapping")
    zs = next(i for i, t in enumerate(prompts)
              if t.startswith("Zero-shot classification prompt."))
    cap_x, tx = src.table("X")
    cap_iv, tiv = src.table("IV")
    cap_ii, tii = src.table("II")
    sens = src.para("Table X repeats the real-world evaluation").replace(
        "Table X", "Table S-I").replace("Section VI-E",
                                        "Section VI-E of the paper")
    lessons = src.para("Methodological lessons.")
    return [
        ("h", "S-A. Prompts"),
        ("p", prompts[0][len("Drafting prompt."):].strip(), "Drafting prompt."),
        ("code", prompts[1:zs]),
        ("p", prompts[zs][len("Zero-shot classification prompt."):].strip(),
         "Zero-shot classification prompt."),
        ("code", prompts[zs + 1:]),
        ("h", "S-B. Reproduction Steps"),
        *[("p", s, None) for s in steps],
        ("h", "S-C. Sensitivity of the Real-World Result to the Category "
              "Mapping"),
        ("p", sens, None),
        ("tbl", "TABLE S-I", cap_x, tx, [3900, 2900, 2900]),
        ("h", "S-D. Model Configurations and Hyperparameters"),
        ("tbl", "TABLE S-II", cap_iv, tiv, [2000, 7700]),
        ("h", "S-E. Verification Checks"),
        ("tbl", "TABLE S-III", cap_ii, tii, [1900, 6300, 1500]),
        ("h", "S-F. Methodological Lessons"),
        ("p", lessons[len("Methodological lessons."):].strip(),
         "Methodological lessons."),
        ("h", "S-G. Additional Figures"),
        ("fig", "Fig. S1.", os.path.join(FIG, "fig2_coverage.png"),
         src.fig_caption(2) + " The same values are in Table IV of the "
         "paper."),
        ("fig", "Fig. S2.", os.path.join(FIG, "fig6_confusion_matrix.png"),
         src.fig_caption(6)),
        ("p", v4b.fills(v)[0]["E3"], "Examples of misrouted tickets."),
    ]


SUPP_INTRO = (
    "Online supplement to the paper above, submitted to IConAECC 2026. It "
    "holds the appendices of the full version (v4), moved here unchanged to "
    "meet the six-page limit, and two figures removed from the paper. Every "
    "number is regenerated from the committed result files in the "
    f"repository, {REPO_URL}.")


def write_supplement(sb, title, path_docx, path_md):
    from docx.enum.text import WD_ALIGN_PARAGRAPH
    w = Writer(TEMPLATE)
    tp = w.para(title, style="paper title")
    tp.runs[0].font.size = w.Pt(16)
    w.para("Online Supplement", style="Author", align=WD_ALIGN_PARAGRAPH.CENTER)
    w.para(SUPP_INTRO)
    md = [f"# {title}", "", "## Online Supplement", "", SUPP_INTRO, ""]
    for b in sb:
        if b[0] == "h":
            h = w.para(b[1], style="Heading 1")
            w.no_numbering(h)
            md += [f"## {b[1]}", ""]
        elif b[0] == "p":
            w.para(b[1], lead=b[2])
            md += [(f"**{b[2]}** " if b[2] else "") + b[1], ""]
        elif b[0] == "code":
            for line in b[1]:
                p = w.para(line or " ", style="Normal", size=8)
                p.runs[0].font.name = "Courier New"
                p.paragraph_format.space_after = w.Pt(0)
            md += ["```text", *b[1], "```", ""]
        elif b[0] == "tbl":
            _, label, cap, rows, widths = b
            w.table(cap, rows, widths, label=label)
            w.para(style="Normal")
            md += [f"**{label}. {cap}**", "",
                   "| " + " | ".join(rows[0]) + " |",
                   "|" + "---|" * len(rows[0])]
            md += ["| " + " | ".join(x.replace("|", "\\|").replace("\n", " ")
                                     for x in r) + " |" for r in rows[1:]]
            md.append("")
        elif b[0] == "fig":
            _, label, fpath, cap = b
            w.figure(fpath, 4.5, cap, label=label)
            rel = os.path.relpath(fpath, OUT_DIR).replace("\\", "/")
            md += [f"![{label}]({rel})", "", f"*{label} {cap}*", ""]
    # Supplement is one column throughout.
    final = w.body.find(w.W("sectPr"))
    w.body.replace(final, copy.deepcopy(w.sect_one))
    w.save(path_docx)
    with open(path_md, "w", encoding="utf-8", newline="\n") as fh:
        fh.write("\n".join(md).rstrip() + "\n")
    del WD_ALIGN_PARAGRAPH
    return md


# ------------------------------------------------------------------ checks
def check_output(path, src, maps, supp_md, abstract):
    import docx
    d = docx.Document(path)
    body_text = []
    for el in d.element.body.iter():
        if el.tag.endswith("}p"):
            body_text.append("".join(t.text or "" for t in el.iter()
                                     if t.tag.endswith("}t")))
    full = "\n".join(body_text)
    paras = [p.text for p in d.paragraphs]
    for bad in ("??", "TODO", "[[", "{S:", "{T:", "{F:", "Error! Reference"):
        if bad in full:
            die(f"output contains {bad!r}")
    for left in TEMPLATE_LEFTOVERS:
        if left in full:
            die(f"template sample text left in the paper: {left!r}")
    for s in d.sections:
        if s._sectPr.findall(d.element.body[0].tag.replace("}p", "}footerReference")) \
                or s._sectPr.findall(d.element.body[0].tag.replace("}p", "}headerReference")):
            die("a section still references a header or footer")
    for need in (DATASET_SENTENCE, ACK, SUPP_URL, "19.7%", "50.4%",
                 "371 of the 468"):
        if need not in full:
            die(f"required text missing: {need[:60]!r}")
    # Every number in the paper and the supplement occurs in approved v4.
    v4nums = numbers_in(src.full)
    extra = sorted(numbers_in(full) - v4nums)
    extra_s = sorted(numbers_in("\n".join(supp_md)) - v4nums)
    allowed = {"2026"}                      # in v4 too; listed for clarity
    if set(extra) - allowed or set(extra_s) - allowed:
        die(f"numbers not in v4: paper {extra}, supplement {extra_s}")
    # Cross-references resolve.
    secs = set(maps["S"].values())
    for x in re.findall(r"Section ([IVX]+(?:-[A-H])?)\b", full):
        if x not in secs:
            die(f"Section {x} cited but not present")
    tabs, figs = set(maps["T"].values()), set(maps["F"].values())
    for x in re.findall(r"Tables? ([IVX]+)\b", full):
        if x not in tabs:
            die(f"Table {x} cited but not present")
    for x in re.findall(r"Fig\. (\d+)", full):
        if x not in figs:
            die(f"Fig. {x} cited but not present")
    if "Appendix" in full:
        die("the paper still cites an appendix")
    n_abs = len(abstract.split())
    if n_abs > ABSTRACT_MAX_WORDS:
        die(f"abstract is {n_abs} words (limit {ABSTRACT_MAX_WORDS})")
    cites = sorted({int(x) for x in re.findall(r"\[(\d+)\]", full)})
    if cites != list(range(1, 26)):
        die(f"citations in the paper are {cites}")
    words = sum(len(t.split()) for t in paras)
    return n_abs, words


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args()
    for p in (OUT, SUPP_DOCX, SUPP_MD):
        if os.path.exists(p) and not args.force:
            die(f"{p} exists (use --force)")
    for p in (TEMPLATE, V4, FIG1, FIG2, FIG3):
        if not os.path.exists(p):
            die(f"missing input {p}")

    v = v4b.load_values()                   # re-checks the committed files
    src = V4Source(V4)
    c = content(v, src)
    maps, new = resolve(c)
    write_paper(c, OUT)
    sb = supplement(src, v)
    md = write_supplement(sb, c["title"], SUPP_DOCX, SUPP_MD)
    n_abs, words = check_output(OUT, src, maps, md, c["abstract"])
    print("[refs] new order (v4 number -> new): "
          + ", ".join(f"{o}->{n}" for o, n in new.items() if o != n))
    print(f"[tables] {maps['T']}")
    print(f"[figs] {maps['F']}")
    print(f"[ok] abstract {n_abs} words; paper {words} words; every number "
          "occurs in v4; no template text, ??, TODO or dangling reference")
    for p in (OUT, SUPP_DOCX, SUPP_MD):
        print(f"[write] {os.path.relpath(p, PROJECT_ROOT)}")


if __name__ == "__main__":
    main()
