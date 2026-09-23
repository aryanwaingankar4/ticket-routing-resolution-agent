"""Phase 9A -- the draft (paper/main.tex, paper/supplement.tex) types no number.

tests/test_paper_artifacts.py pins the GENERATED paper surface. This file pins
the HAND-WRITTEN draft that sits on top of it: NUMBERS.md made every number
traceable, and paper/numbers.tex makes a \\nb{} macro the ONLY way a number can
reach the draft.

Every check here is fast. It reads committed files, rebuilds nothing and loads
no model, so it runs in the `-m "not slow"` subset and therefore in ci.yml.
No LaTeX engine is installed on the development machine, so structural balance
and label resolution are also checked here rather than by a compile.
"""

from __future__ import annotations

import json
import os
import re
import sys

import pytest

PROJECT_ROOT = os.path.abspath(
    os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
PAPER_DIR = os.path.join(PROJECT_ROOT, "paper")
DRAFTS = ("main.tex", "supplement.tex")

pytestmark = pytest.mark.skipif(
    not os.path.isfile(os.path.join(PAPER_DIR, "main.tex")),
    reason="paper/main.tex has not been written yet")

_NB_USE = re.compile(r"\\nb\{([^}]*)\}")
_NB_DEF = re.compile(r"^\\nb@def\{([^}]*)\}", re.M)


def _read(name):
    with open(os.path.join(PAPER_DIR, name), "r", encoding="utf-8") as fh:
        return fh.read()


def _strip_comments(text):
    return "\n".join(re.sub(r"(?<!\\)%.*", "", line)
                     for line in text.splitlines())


def _body(text):
    marker = r"\begin{document}"
    assert marker in text, "draft has no \\begin{document}"
    return text.split(marker, 1)[1]


def _defined_macros():
    return set(_NB_DEF.findall(_read("numbers.tex")))


def test_numbers_tex_carries_every_numbers_md_id():
    """Every NUMBERS.md value is quotable, and nothing else is."""
    # Only the five-column number rows; the source-hash table at the end of
    # NUMBERS.md also starts with a backticked cell.
    ids = set(re.findall(r"^\| `([^`]+)` \|(?:[^|\n]*\|){4}\s*$",
                         _read("NUMBERS.md"), re.M))
    base = {key for key in _defined_macros() if "@" not in key}
    assert ids, "no ids parsed from NUMBERS.md"
    assert base == ids, (
        f"numbers.tex and NUMBERS.md disagree: only in NUMBERS.md "
        f"{sorted(ids - base)[:10]}, only in numbers.tex "
        f"{sorted(base - ids)[:10]}")


def test_every_number_the_draft_quotes_exists():
    defined = _defined_macros()
    missing = [f"{name}: \\nb{{{key}}}"
               for name in DRAFTS
               for key in _NB_USE.findall(_strip_comments(_read(name)))
               if key not in defined]
    assert not missing, (
        "the draft quotes numbers that numbers.tex does not define (a "
        f"compile error): {missing}")


# Name tokens that legitimately contain digits. Anything else with a digit in
# the body of a draft is a typed number.
_ALLOWED_DIGIT_TOKENS = [
    r"Tier-[12]", r"Finding~?\d", r"Qwen2\.5-3B", r"\b3B(?:-parameter)?\b",
    r"bge-base-en-v1\.5", r"float(?:32|64)", r"top-1", r"\bF1\b",
    r"adv\\_08",
    # Conventions, not measurements: the 95% of a Wilson interval, and the
    # constant in the gamma_n formula.
    r"\b95\\%", r"\(1-nu\)",
]
_STRIP_COMMANDS = re.compile(
    r"\\(?:nb|ref|label|cite|input|includegraphics|bibliography|"
    r"bibliographystyle|graphicspath)(?:\[[^\]]*\])?\{[^{}]*\}")


def test_the_draft_types_no_number():
    """The Phase 9A rule, enforced: a digit in prose is a retyped number."""
    offenders = []
    for name in DRAFTS:
        body = _STRIP_COMMANDS.sub(" ", _strip_comments(_body(_read(name))))
        for token in _ALLOWED_DIGIT_TOKENS:
            body = re.sub(token, " ", body)
        offenders += [f"{name}: {line.strip()[:120]}"
                      for line in body.splitlines() if re.search(r"\d", line)]
    assert not offenders, (
        "digits typed into the draft -- quote them through \\nb{} instead:\n"
        + "\n".join(offenders))


def test_the_lint_catches_a_typed_number():
    """The lint above must be able to fail, or it proves nothing."""
    probe = "Tier-1 scores 32/45 on the benchmark."
    body = _STRIP_COMMANDS.sub(" ", probe)
    for token in _ALLOWED_DIGIT_TOKENS:
        body = re.sub(token, " ", body)
    assert re.search(r"\d", body), "the no-number lint no longer detects 32/45"


def test_every_input_and_figure_the_draft_names_exists():
    missing = []
    for name in DRAFTS:
        text = _strip_comments(_read(name))
        missing += [f"{name}: \\input{{{p}}}"
                    for p in re.findall(r"\\input\{([^}]*)\}", text)
                    if not os.path.isfile(os.path.join(PAPER_DIR, p))]
        missing += [f"{name}: figures/{p}"
                    for p in re.findall(
                        r"\\includegraphics(?:\[[^\]]*\])?\{([^}]*)\}", text)
                    if not os.path.isfile(os.path.join(PAPER_DIR, "figures",
                                                       p))]
    assert not missing, f"the draft names files that do not exist: {missing}"


def test_the_draft_is_structurally_balanced():
    for name in DRAFTS:
        text = _strip_comments(_read(name)).replace(r"\{", "").replace(
            r"\}", "")
        assert text.count("{") == text.count("}"), f"{name}: unbalanced braces"
        stack = []
        for kind, env in re.findall(r"\\(begin|end)\{([^}]*)\}", text):
            if kind == "begin":
                stack.append(env)
            else:
                assert stack and stack[-1] == env, (
                    f"{name}: \\end{{{env}}} closes "
                    f"{stack[-1] if stack else 'nothing'}")
                stack.pop()
        assert not stack, f"{name}: unclosed environments {stack}"

        labels = re.findall(r"\\label\{([^}]*)\}", text)
        for path in re.findall(r"\\input\{(tables/[^}]*)\}", text):
            labels += re.findall(r"\\label\{([^}]*)\}", _read(path))
        refs = set(re.findall(r"\\ref\{([^}]*)\}", text))
        assert refs <= set(labels), (
            f"{name}: references to undefined labels "
            f"{sorted(refs - set(labels))}")


def test_every_cite_key_is_in_the_bib_and_the_checklist():
    bib = _read("references.bib")
    checklist = _read("references_to_check.md")
    keys = set(re.findall(r"^@\w+\{([^,]+),", bib, re.M))
    missing = []
    for name in DRAFTS:
        for group in re.findall(r"\\cite\{([^}]*)\}",
                                _strip_comments(_read(name))):
            for key in (k.strip() for k in group.split(",")):
                if key not in keys:
                    missing.append(f"{key} (not in references.bib)")
                if f"`{key}`" not in checklist:
                    missing.append(f"{key} (not in references_to_check.md)")
    assert not missing, f"citation bookkeeping incomplete: {missing}"


# A clause group is quoted WHOLE: if a section of the draft quotes any number
# from the group, every required companion appears in that same section.
CLAUSE_COMPANIONS = {
    # group: (trigger ids, required ids). Quoting any trigger in a section
    # requires every required id in that same section.
    "finding1": (["T7.tier1.coverage_gap", "T7.tier2.coverage_gap",
                  "T11.7b.tier1.coverage_gap", "T11.7c.tfidf_domain_auc",
                  "T11.7c.did_point"],
                 ["T7.tier1.coverage_gap", "T7.tier2.coverage_gap",
                  "T7.noise_band_2sd", "T11.7b.tier1.coverage_gap",
                  "T11.7c.tfidf_domain_auc", "T11.7c.did_point"]),
    "5C": (["T1.zeroshot_gemini.benchmark45", "T1.zeroshot_qwen.benchmark45",
            "T1.mcnemar.gemini_vs_tier2.benchmark45",
            "T1.mcnemar.qwen_vs_tier2.benchmark45"],
           ["T1.zeroshot_gemini.benchmark45", "T1.zeroshot_qwen.benchmark45",
            "T1.mcnemar.gemini_vs_tier2.benchmark45",
            "T1.mcnemar.qwen_vs_tier2.benchmark45"]),
    "6B": (["T7.weighted.tier1.delta_vs_unweighted",
            "T7.weighted.tier1.residual_in_sd",
            "T7.weighted.tier1.coverage_gap"],
           ["T7.weighted.tier1.delta_vs_unweighted",
            "T7.weighted.tier1.residual_in_sd"]),
    # The catch is never quoted without the false flags and the N45
    # counter-case; N45 alone (as a gate error) triggers nothing.
    "6C": (["T15.caught_of_ungrounded", "T15.flagged_of_grounded"],
           ["T15.caught_of_ungrounded", "T15.flagged_of_grounded",
            "T17.escalated_adequate.0.ticket"]),
    "6A": (["T12.6a.comparisons_with_signal",
            "T12.6a.degenerate_configurations"],
           ["T12.6a.comparisons_with_signal",
            "T12.6a.degenerate_configurations"]),
    "5B": (["T3.cascade_delta.benchmark45", "T3.mcnemar.benchmark45"],
           ["T3.cascade_delta.benchmark45", "T3.mcnemar.benchmark45",
            "T3.latency.saving.benchmark45"]),
    "7A-control": (["T11.7a.near_duplicate_rate_external",
                    "T11.7a.near_duplicate_rate_our_corpus"],
                   ["T11.7a.near_duplicate_rate_external",
                    "T11.7a.near_duplicate_rate_our_corpus"]),
}


def test_clause_groups_are_quoted_whole():
    problems = []
    for name in DRAFTS:
        sections = re.split(r"\\(?:sub)?section\*?\{",
                            _strip_comments(_read(name)))
        for idx, section in enumerate(sections):
            used = {key.split("@")[0] for key in _NB_USE.findall(section)}
            for group, (triggers, required) in CLAUSE_COMPANIONS.items():
                if any(t in used for t in triggers):
                    missing = [r for r in required if r not in used]
                    if missing:
                        problems.append(
                            f"{name} section {idx}: clause group {group} "
                            f"quoted partially, missing {missing}")
    assert not problems, "\n".join(problems)


def test_the_draft_avoids_retired_and_forbidden_wording():
    """The Phase 9B audit list, and the words that need Aryan's approval."""
    forbidden = [
        r"bit-identical", r"over-?confident", r"\bnovel\b",
        r"state[- ]of[- ]the[- ]art", r"\bthe first (?:to|work|study|paper)\b",
        r"LLMs beat the pipeline", r"refutes Finding",
        r"correctable by covariate reweighting", r"conformal is worse",
        r"MiniLM-era", r"adv\\_0[35]",
    ]
    offenders = [f"{name}: {m.group(0)!r}"
                 for name in DRAFTS
                 for pattern in forbidden
                 for m in re.finditer(pattern, _strip_comments(_read(name)),
                                      re.I)]
    assert not offenders, f"forbidden wording in the draft: {offenders}"


def test_tier1_vocabulary_ties_rederive():
    """data/tier1_vocabulary_ties.json, re-derived from the dataset.

    Also checks that the committed vocabulary contains every term the corpus
    puts strictly ABOVE the cut -- the part the data did determine.
    """
    sys.path.insert(0, PROJECT_ROOT)
    from src.experiments.measure_tier1_vocabulary_ties import (
        MAX_FEATURES, load_texts, tie_profile)
    import numpy as np
    from sklearn.feature_extraction.text import CountVectorizer

    with open(os.path.join(PROJECT_ROOT, "data", "tier1_vocabulary_ties.json"),
              "r", encoding="utf-8") as fh:
        committed = json.load(fh)
    texts, _labels, _n = load_texts()
    assert tie_profile(texts) == committed["full4000"]

    counts = CountVectorizer(ngram_range=(1, 2), stop_words="english")
    tfs = np.asarray(counts.fit_transform(texts).sum(axis=0)).ravel()
    terms = counts.get_feature_names_out()
    cut = np.sort(tfs)[::-1][MAX_FEATURES - 1]
    above = set(terms[tfs > cut])
    with open(os.path.join(PROJECT_ROOT, "data", "tier1_vocabulary.txt"), "r",
              encoding="utf-8") as fh:
        vocabulary = {line.rstrip("\n") for line in fh if line.strip()}
    assert len(above) == committed["full4000"]["above_cut"]
    assert above <= vocabulary, (
        f"{len(above - vocabulary)} data-determined terms are missing from "
        "the committed vocabulary")
