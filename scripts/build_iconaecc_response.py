"""Write paper_ieee_docx/Response_to_Reviewers.docx (IConAECC 2026, #1251).

Offline. About one page: for each reviewer, each point in one line, then what
changed and where (section number and name in the camera-ready paper). Every
number quoted here is already in the paper.

The point lines summarise the reviews; only Reviewer 2's phrase "a structural
limitation that warrants deeper discussion" is quoted verbatim. Paste the
reviewers' exact wording over a summary line if it is wanted.

    .\\venv\\Scripts\\python.exe scripts/build_iconaecc_response.py [--force]
"""

from __future__ import annotations

import argparse
import os

from realworld_common import PROJECT_ROOT, die

OUT = os.path.join(PROJECT_ROOT, "paper_ieee_docx", "Response_to_Reviewers.docx")
TITLE = ("Calibrated Escalation for IT Ticket Triage: What the Calibration "
         "Distribution Decides")

# Section labels as built by scripts/build_iconaecc_docx.py.
S2 = "Section II, Related Work"
S6B = "Section VI-B, The cascade saves latency, not accuracy"
S6D = "Section VI-D, Groundedness, an LLM judge, and a sufficiency gate"
S6E = "Section VI-E, Extended evaluation on independent and real-world tickets"
S7 = "Section VII, Error Analysis"
S8 = "Section VIII, Limitations"
S9 = "Section IX, Conclusion and Future Work"

REVIEWS = [
    ("Reviewer 1", [
        ("The gap between synthetic training data and real tickets.",
         "Added domain adaptation as future work: few-shot fine-tuning on a "
         f"small labelled real set, prompt tuning, and self-training on "
         f"unlabelled real tickets ({S9})."),
        ("Real-world coverage is limited.",
         "We now state that the Zenodo set covers only two of our seven "
         f"teams, so five teams have no real-ticket evaluation ({S6E}; {S8}), "
         "and add a real evaluation of all seven teams with a partner IT "
         f"service desk as future work ({S9})."),
        ("Deployment and operational metrics.",
         "Added a shadow deployment as future work: the system runs "
         "alongside human triage without acting, to measure resolution "
         f"time, re-routing rate and cost ({S9})."),
        ("How the gates behave in a changing environment.",
         "Added periodic recalibration on a rolling window of recent "
         "labelled tickets, drift monitoring that triggers recalibration, "
         f"and weighted or online conformal methods as future work ({S9})."),
        ("Computational trade-offs.",
         "The costs are now in one place: Tier 2 costs 151.0× Tier 1 per "
         "ticket (156.4 ms against 1.04 ms median), the cascade saves 8–18% "
         "of median latency, and a ticket escalated at the retrieval gate "
         f"makes no language-model call ({S6B})."),
    ]),
    ("Reviewer 2", [
        ("The synthetic-to-real gap.", f"As for Reviewer 1 ({S9})."),
        ("The alternative mapping (50.4%) needs explaining.",
         "We now explain that it changes one assignment, Fileservice to "
         "Storage instead of Access Management, and state that it is a "
         "sensitivity analysis, not a deployable setting; a production "
         f"system would need labels in its own categories ({S6E})."),
        ("371 of 468 misrouted tickets reached the drafter: “a structural "
         "limitation that warrants deeper discussion”.",
         "Added why the retrieval gate cannot catch misrouting: it checks "
         "whether a similar past ticket exists, not whether the predicted "
         "team is correct. We propose a cheap consistency check (predicted "
         "team against the teams of the top-5 retrieved tickets) and say it "
         f"is untested ({S7})."),
        ("Gates in changing environments.", f"As for Reviewer 1 ({S9})."),
        ("Computational trade-offs.", f"As for Reviewer 1 ({S6B})."),
    ]),
    ("Reviewer 3 (minor revisions)", [
        ("The synthetic-to-real gap.", f"As for Reviewer 1 ({S9})."),
        ("Real-world coverage.", f"As for Reviewer 1 ({S6E}; {S8}; {S9})."),
        ("The 50.4% mapping reflects a semantic gap between synthetic and "
         "real categories.",
         f"We now say so ({S6E}) and add hierarchical classification "
         "(coarse category first, then team) or zero-shot prompting for "
         f"unseen categories as future work ({S9})."),
        ("Novelty relative to confidence-based cascades and conformal "
         "methods.",
         "Added a contrast: those methods set thresholds on held-out "
         "in-distribution data and assume exchangeability; we choose and "
         "re-check thresholds on deliberately shifted sets (paraphrased, "
         "adversarial, human-written, real) and report where the guarantees "
         f"fail ({S2})."),
        ("Groundedness evaluation beyond an LLM judge (κ = −0.042).",
         "Added future work: claim-level checks against the retrieved text, "
         "NLI-based entailment scoring, a small human-labelled set to "
         "calibrate any automated judge, and reporting judge–human agreement "
         f"before a judge is used ({S6D})."),
        ("Implications for production use.",
         "The conclusion now states that both thresholds must be "
         f"recalibrated on the target desk's own tickets before production "
         f"use ({S9})."),
    ]),
]

INTRO = (
    "We thank the reviewers for their positive and constructive reviews. "
    "All changes are to the text: no new experiment was run, and every "
    "number in the revision already appeared in the reviewed paper or in "
    "the committed result files it reports. Section numbers refer to the "
    "camera-ready version. The title and the author list now match the "
    "conference registration.")


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args()
    if os.path.exists(OUT) and not args.force:
        die(f"{OUT} exists (use --force)")
    import docx
    from docx.shared import Cm, Pt
    d = docx.Document()
    sec = d.sections[0]
    sec.page_width, sec.page_height = Cm(21.0), Cm(29.7)
    sec.left_margin = sec.right_margin = Cm(2.0)
    sec.top_margin = sec.bottom_margin = Cm(1.8)
    st = d.styles["Normal"]
    st.font.name, st.font.size = "Times New Roman", Pt(10)
    st.paragraph_format.space_after = Pt(0)

    def para(text="", bold=False, italic=False, size=None, after=0, before=0,
             indent=None):
        p = d.add_paragraph()
        pf = p.paragraph_format
        pf.space_after, pf.space_before = Pt(after), Pt(before)
        if indent is not None:
            pf.left_indent = Cm(indent)
        if text:
            r = p.add_run(text)
            r.bold, r.italic = bold, italic
            if size:
                r.font.size = Pt(size)
        return p

    para("Response to the Reviewers — IConAECC 2026, Paper #1251",
         bold=True, size=12)
    para(TITLE, italic=True, after=4)
    para(INTRO, after=4)
    for who, points in REVIEWS:
        para(who, bold=True, before=4, after=1)
        for k, (point, reply) in enumerate(points, 1):
            p = para(indent=0.3)
            p.add_run(f"{k}. ").bold = True
            p.add_run(point).italic = True
            q = para(indent=0.8, after=2)
            q.add_run("Change: ").bold = True
            q.add_run(reply)
    d.core_properties.title = "Response to the Reviewers, IConAECC 2026 #1251"
    d.core_properties.author = "Aryan Waingankar"
    d.save(OUT)
    print(f"[write] {os.path.relpath(OUT, PROJECT_ROOT)}")


if __name__ == "__main__":
    main()
