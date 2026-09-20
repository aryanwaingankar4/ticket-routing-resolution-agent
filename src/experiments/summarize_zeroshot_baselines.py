# src/experiments/summarize_zeroshot_baselines.py
"""
Phase 5C -- descriptive summary of a zero-shot baseline arm.

WHY THIS EXISTS. compare_zeroshot_vs_tier2.py answers "are these two
distinguishable?" (exact McNemar on the discordant pairs). It does not answer
"how precise is this estimate?", "where does the model fail?", or "how much of
the gap is format rather than knowledge?". With a small local model as a second
arm, those three become the difference between a readable result and a bare
percentage.

DUAL ACCOUNTING FOR UNPARSEABLE OUTPUT. A model that cannot emit the requested
format is wrong in production, so counting unparseable as WRONG is the primary,
honest number. But if a weak model's gap is mostly format, saying only that
hides the interesting part. Both are therefore always reported, each with its
own denominator stated, so they cannot be confused for one another. The parser
never coerces a near-miss ("Networking" -> "Network"), so an unparseable rate is
a real measurement rather than an artefact of being strict.

PROMPT IDENTITY IS PROVEN, NOT ASSERTED. build_prompt() is backend-agnostic, so
two arms get a byte-identical prompt by construction. This script nevertheless
compares the prompt_sha256 recorded in each arm's cached raw responses and
fails fatally on any mismatch. Rule 6 of this project: check every count you
rely on against a second, independent derivation. "The code path guarantees it"
is not a second derivation.

Offline: reads CSVs and the raw-response cache only. No models, no API, no
quota.

Usage (from the project root):
    python src/experiments/summarize_zeroshot_baselines.py \
        --backend ollama --model qwen2.5:3b-instruct --set both

    # also prove the prompt matched the Gemini arm's, ticket by ticket
    python src/experiments/summarize_zeroshot_baselines.py \
        --backend ollama --model qwen2.5:3b-instruct --set both \
        --prompt-check-against gemini
"""

import os
import sys
import csv
import json
import math
import argparse
import traceback

THIS_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(THIS_DIR, os.pardir, os.pardir))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from src.experiments.compare_cascade_vs_tier2 import (  # noqa: E402
    _banner, _fatal,
)
from src.experiments.compare_zeroshot_vs_tier2 import (  # noqa: E402
    DEFAULT_GEMINI_MODEL, DEFAULT_OLLAMA_MODEL, load_zeroshot_rows,
)
from src.experiments.run_zeroshot_baselines import (  # noqa: E402
    CATEGORIES, EVAL_SETS, _slug, cache_path,
)

DATA_DIR = os.path.join(PROJECT_ROOT, "data")

# Two-sided normal quantile for a 95% interval. Hard-coded rather than pulled
# from scipy so the script has no import-time dependency on it; the test suite
# checks every interval this produces against
# scipy.stats.binomtest(...).proportion_ci(method="wilson").
Z_95 = 1.959963984540054


def wilson_interval(successes, n, z=Z_95):
    """Wilson score interval for a binomial proportion.

    Preferred over the normal approximation because these samples are small
    (n = 14 and n = 45) and proportions sit near the boundary, where the normal
    interval can run outside [0, 1] and undercover badly.
    """
    if n <= 0:
        return (float("nan"), float("nan"))
    if successes < 0 or successes > n:
        _fatal("Wilson interval given {k} successes out of {n}.".format(
            k=successes, n=n))

    p = float(successes) / float(n)
    z2n = (z * z) / float(n)
    centre = (p + z2n / 2.0) / (1.0 + z2n)
    half = (z / (1.0 + z2n)) * math.sqrt(
        p * (1.0 - p) / float(n) + (z * z) / (4.0 * float(n) * float(n))
    )
    return (max(0.0, centre - half), min(1.0, centre + half))


def quantile(sorted_values, q):
    """Nearest-rank quantile. Small n, so interpolation would overstate it."""
    if not sorted_values:
        return float("nan")
    idx = int(math.ceil(q * len(sorted_values))) - 1
    return sorted_values[max(0, min(idx, len(sorted_values) - 1))]


def confusion(rows):
    """{expected: {predicted_or_UNPARSEABLE: count}} over the seven categories."""
    labels = list(CATEGORIES) + ["UNPARSEABLE"]
    table = {e: {p: 0 for p in labels} for e in CATEGORIES}
    for r in rows:
        exp = r["expected"]
        if exp not in table:
            _fatal("Row {i} has expected label {e!r}, which is not one of the "
                   "seven categories.".format(i=r["ticket_id"], e=exp))
        pred = r["predicted"] if r["parsed_ok"] else "UNPARSEABLE"
        if pred not in table[exp]:
            _fatal("Row {i} predicted {p!r}, which is not a category. The "
                   "parser should have rejected it.".format(
                       i=r["ticket_id"], p=pred))
        table[exp][pred] += 1
    return table, labels


def per_category(table):
    """Precision, recall and support per category, from the confusion table."""
    out = {}
    for cat in CATEGORIES:
        support = sum(table[cat].values())
        tp = table[cat][cat]
        predicted_as = sum(table[e][cat] for e in CATEGORIES)
        out[cat] = {
            "support": support,
            "tp": tp,
            "predicted_as": predicted_as,
            "recall": (tp / support) if support else float("nan"),
            "precision": (tp / predicted_as) if predicted_as else float("nan"),
        }
    return out


def latency_stats(rows):
    """Timing from LIVE rows only.

    A cached row carries the ORIGINAL run's elapsed_s. Mixing cached and live
    timings would produce a number that is internally consistent and describes
    no run that ever happened -- this project's recurring bug shape.
    """
    live = sorted(r["elapsed_s"] for r in rows if r.get("source") == "live")
    return {
        "n_live": len(live),
        "median_s": quantile(live, 0.5),
        "p90_s": quantile(live, 0.90),
        "min_s": live[0] if live else float("nan"),
        "max_s": live[-1] if live else float("nan"),
    }


def load_rows_with_source(backend, model, eval_set):
    """load_zeroshot_rows() plus the 'source' column it does not carry.

    Reuses the shared loader -- including its rule-6 guard that recomputes the
    headline from the raw predictions -- rather than re-reading the CSV.
    """
    rows, path = load_zeroshot_rows(backend, model, eval_set)
    with open(path, "r", newline="", encoding="utf-8") as fh:
        raw = list(csv.DictReader(fh))
    if len(raw) != len(rows):
        _fatal("Row count changed while re-reading {p}.".format(p=path))
    for row, r in zip(rows, raw):
        if row["ticket_id"] != r["ticket_id"]:
            _fatal("Row order changed while re-reading {p}.".format(p=path))
        row["source"] = r["source"]
    return rows, path


def check_prompt_identity(backend, model, other_backend, other_model, eval_set,
                          rows):
    """Prove both arms saw a byte-identical prompt, ticket by ticket.

    Returns (n_checked, n_missing). Fatal on any hash mismatch.
    """
    n_checked = 0
    n_missing = 0
    for r in rows:
        mine = cache_path(backend, model, eval_set, r["ticket_id"])
        theirs = cache_path(other_backend, other_model, eval_set,
                            r["ticket_id"])
        if not (os.path.isfile(mine) and os.path.isfile(theirs)):
            n_missing += 1
            continue
        try:
            with open(mine, "r", encoding="utf-8") as fh:
                a = json.load(fh)
            with open(theirs, "r", encoding="utf-8") as fh:
                b = json.load(fh)
        except Exception as exc:
            _fatal("Could not read a cached response for {t}: {e}".format(
                t=r["ticket_id"], e=repr(exc)))
        if a.get("prompt_sha256") != b.get("prompt_sha256"):
            _fatal(
                "PROMPT MISMATCH on {t} ({s}):\n"
                "  {ba}: {ha}\n  {bb}: {hb}\n"
                "The two arms did not answer the same question, so no "
                "comparison between them is valid.".format(
                    t=r["ticket_id"], s=eval_set,
                    ba=backend, ha=a.get("prompt_sha256"),
                    bb=other_backend, hb=b.get("prompt_sha256"))
            )
        n_checked += 1
    return n_checked, n_missing


def summarise_set(backend, model, eval_set, prompt_check_against,
                  prompt_check_model=None):
    rows, path = load_rows_with_source(backend, model, eval_set)
    n = len(rows)

    n_correct = sum(1 for r in rows if r["correct"])
    n_unparseable = sum(1 for r in rows if not r["parsed_ok"])
    n_parsed = n - n_unparseable
    # Among the answers that PARSED, how many were right. An unparseable answer
    # is excluded from both numerator and denominator here -- that is what makes
    # this the "if the format were fixed" number rather than the honest one.
    n_correct_parsed = sum(1 for r in rows if r["parsed_ok"] and r["correct"])

    acc_strict = (n_correct / n) if n else float("nan")
    lo_s, hi_s = wilson_interval(n_correct, n)
    acc_excl = (n_correct_parsed / n_parsed) if n_parsed else float("nan")
    lo_e, hi_e = wilson_interval(n_correct_parsed, n_parsed)

    table, labels = confusion(rows)
    cats = per_category(table)
    lat = latency_stats(rows)

    # Rule 6: the confusion table is an independent recount of the headline.
    diag = sum(table[c][c] for c in CATEGORIES)
    if diag != n_correct:
        _fatal(
            "Confusion diagonal ({d}) disagrees with the correct count ({c}) "
            "for {p}. One of them is wrong; do not use this summary.".format(
                d=diag, c=n_correct, p=path))
    total_cells = sum(sum(v.values()) for v in table.values())
    if total_cells != n:
        _fatal("Confusion table holds {t} rows but the file has {n}.".format(
            t=total_cells, n=n))

    prompt_checked = prompt_missing = None
    if prompt_check_against:
        other_model = prompt_check_model or (
            DEFAULT_GEMINI_MODEL if prompt_check_against == "gemini"
            else DEFAULT_OLLAMA_MODEL)
        prompt_checked, prompt_missing = check_prompt_identity(
            backend, model, prompt_check_against, other_model, eval_set, rows)
        # A check that verified NOTHING must not be recorded as if it passed:
        # "0 verified" and "all verified" would otherwise look equally green.
        if prompt_checked == 0 and prompt_missing:
            _fatal(
                "Prompt identity check verified 0 of {n} tickets against the "
                "{o} arm ({m}) -- every cached response was missing on one "
                "side.\n"
                "  Pass --prompt-check-model with the tag that arm actually "
                "ran under, or drop --prompt-check-against.".format(
                    n=prompt_missing, o=prompt_check_against, m=other_model)
            )

    return {
        "path": path, "n": n, "n_correct": n_correct,
        "n_unparseable": n_unparseable, "n_parsed": n_parsed,
        "n_correct_parsed": n_correct_parsed,
        "acc_strict": acc_strict, "ci_strict": (lo_s, hi_s),
        "acc_excl": acc_excl, "ci_excl": (lo_e, hi_e),
        "table": table, "labels": labels, "cats": cats, "lat": lat,
        "prompt_checked": prompt_checked, "prompt_missing": prompt_missing,
        "prompt_other": prompt_check_against,
    }


def print_summary(backend, model, eval_set, s):
    _banner("{b} / {m}  --  {s}".format(b=backend, m=model, s=eval_set))
    print("source: " + s["path"])
    print("")

    print("ACCURACY (both accountings, each with its own denominator)")
    print("  unparseable counted WRONG  : {c}/{n} = {a:.2%}   "
          "95% Wilson [{lo:.2%}, {hi:.2%}]   <- the honest number".format(
              c=s["n_correct"], n=s["n"], a=s["acc_strict"],
              lo=s["ci_strict"][0], hi=s["ci_strict"][1]))
    if s["n_unparseable"]:
        print("  unparseable EXCLUDED       : {c}/{n} = {a:.2%}   "
              "95% Wilson [{lo:.2%}, {hi:.2%}]   <- 'if the format were "
              "fixed'".format(
                  c=s["n_correct_parsed"], n=s["n_parsed"], a=s["acc_excl"],
                  lo=s["ci_excl"][0], hi=s["ci_excl"][1]))
    else:
        print("  unparseable EXCLUDED       : identical -- 0 unparseable "
              "answers, so the two accountings cannot differ.")
    print("  unparseable                : {u}/{n} = {r:.1%}".format(
        u=s["n_unparseable"], n=s["n"],
        r=(s["n_unparseable"] / s["n"]) if s["n"] else float("nan")))

    print("")
    print("LATENCY (live rows only; cached rows carry the original run's "
          "timing)")
    lat = s["lat"]
    if lat["n_live"]:
        print("  n={n}  median {me:.2f}s  p90 {p9:.2f}s  min {mi:.2f}s  "
              "max {ma:.2f}s".format(n=lat["n_live"], me=lat["median_s"],
                                     p9=lat["p90_s"], mi=lat["min_s"],
                                     ma=lat["max_s"]))
    else:
        print("  no live rows in this file (every response came from cache)")

    print("")
    print("CONFUSION (rows = expected, columns = predicted)")
    short = {c: c[:6] for c in s["labels"]}
    header = " " * 20 + "".join("{:>8}".format(short[c]) for c in s["labels"])
    print(header)
    for exp in CATEGORIES:
        line = "{:<20}".format(exp[:19])
        for pred in s["labels"]:
            v = s["table"][exp][pred]
            line += "{:>8}".format(v if v else ".")
        print(line)

    print("")
    print("PER CATEGORY")
    print("  {:<20}{:>8}{:>10}{:>10}".format(
        "category", "support", "recall", "precision"))
    for cat in CATEGORIES:
        c = s["cats"][cat]
        prec = ("{:.0%}".format(c["precision"])
                if c["predicted_as"] else "n/a")
        print("  {:<20}{:>8}{:>10}{:>10}".format(
            cat[:19], c["support"], "{:.0%}".format(c["recall"]), prec))

    if s["prompt_other"]:
        print("")
        print("PROMPT IDENTITY vs the {o} arm: {c} ticket(s) verified "
              "byte-identical (sha256), {m} not cached on both sides.".format(
                  o=s["prompt_other"], c=s["prompt_checked"],
                  m=s["prompt_missing"]))


def write_summary(backend, model, eval_set, s, force):
    out_path = os.path.join(
        DATA_DIR,
        "zeroshot_summary_{b}_{m}_{s}.csv".format(
            b=backend, m=_slug(model), s=eval_set),
    )
    if os.path.isfile(out_path) and not force:
        _fatal(
            "Refusing to overwrite an existing result:\n  {p}\n"
            "Pass --force if you mean to replace it.".format(p=out_path)
        )

    rows = [
        ("n", "", s["n"]),
        ("n_correct", "", s["n_correct"]),
        ("n_unparseable", "", s["n_unparseable"]),
        ("accuracy_unparseable_wrong", "", "{:.6f}".format(s["acc_strict"])),
        ("accuracy_unparseable_wrong_ci_low", "",
         "{:.6f}".format(s["ci_strict"][0])),
        ("accuracy_unparseable_wrong_ci_high", "",
         "{:.6f}".format(s["ci_strict"][1])),
        ("n_parsed", "", s["n_parsed"]),
        ("n_correct_parsed", "", s["n_correct_parsed"]),
        ("accuracy_unparseable_excluded", "", "{:.6f}".format(s["acc_excl"])),
        ("accuracy_unparseable_excluded_ci_low", "",
         "{:.6f}".format(s["ci_excl"][0])),
        ("accuracy_unparseable_excluded_ci_high", "",
         "{:.6f}".format(s["ci_excl"][1])),
        ("latency_n_live", "", s["lat"]["n_live"]),
        ("latency_median_s", "", "{:.4f}".format(s["lat"]["median_s"])),
        ("latency_p90_s", "", "{:.4f}".format(s["lat"]["p90_s"])),
        ("latency_min_s", "", "{:.4f}".format(s["lat"]["min_s"])),
        ("latency_max_s", "", "{:.4f}".format(s["lat"]["max_s"])),
    ]
    if s["prompt_other"]:
        rows.append(("prompt_identity_verified_vs_" + s["prompt_other"], "",
                     s["prompt_checked"]))
    for cat in CATEGORIES:
        c = s["cats"][cat]
        rows.append(("support", cat, c["support"]))
        rows.append(("recall", cat, "{:.6f}".format(c["recall"])))
        rows.append(("precision", cat, "{:.6f}".format(c["precision"])))
    for exp in CATEGORIES:
        for pred in s["labels"]:
            rows.append(("confusion",
                         "{e}->{p}".format(e=exp, p=pred),
                         s["table"][exp][pred]))

    try:
        with open(out_path, "w", newline="", encoding="utf-8") as fh:
            writer = csv.writer(fh)
            writer.writerow(["metric", "category", "value"])
            for row in rows:
                writer.writerow(list(row))
    except Exception as exc:
        _fatal("Failed to write {p}: {e}".format(p=out_path, e=repr(exc)))
    return out_path


def parse_args(argv):
    parser = argparse.ArgumentParser(
        description=("Descriptive summary of a zero-shot baseline arm: "
                     "Wilson intervals, dual unparseable accounting, "
                     "per-category confusion, latency. Offline.")
    )
    parser.add_argument("--backend", required=True,
                        choices=("gemini", "ollama"))
    parser.add_argument("--model", default=None,
                        help="Model tag as it appears in the result filename.")
    parser.add_argument("--set", dest="eval_set", default="benchmark45",
                        choices=EVAL_SETS + ("both",))
    parser.add_argument("--prompt-check-against", default=None,
                        choices=("gemini", "ollama"),
                        help="Prove, ticket by ticket, that this arm's cached "
                             "prompts are byte-identical to that arm's.")
    parser.add_argument("--prompt-check-model", default=None,
                        help="Model tag of the arm named by "
                             "--prompt-check-against. Needed whenever that arm "
                             "is not the default model for its backend -- "
                             "otherwise the cache files are looked up under "
                             "the wrong name and every ticket is reported as "
                             "'not cached on both sides'.")
    parser.add_argument("--force", action="store_true")
    return parser.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    model = args.model or (DEFAULT_GEMINI_MODEL if args.backend == "gemini"
                           else DEFAULT_OLLAMA_MODEL)
    if args.prompt_check_against == args.backend:
        _fatal("--prompt-check-against names the same backend being "
               "summarised; that comparison proves nothing.")

    sets = EVAL_SETS if args.eval_set == "both" else (args.eval_set,)
    for eval_set in sets:
        s = summarise_set(args.backend, model, eval_set,
                          args.prompt_check_against,
                          args.prompt_check_model)
        print_summary(args.backend, model, eval_set, s)
        out_path = write_summary(args.backend, model, eval_set, s, args.force)
        print("")
        print("Summary written: " + out_path)
        print("")

    print("READ THIS WITH THE NUMBERS: a zero-shot LLM against a classifier")
    print("trained on 3,200 rows is NOT like-for-like. What 5C measures is")
    print("what the training corpus is worth, not which method is better.")


if __name__ == "__main__":
    try:
        main()
    except SystemExit:
        raise
    except Exception:
        print("")
        print("=" * 70)
        print("UNEXPECTED ERROR -- full traceback follows:")
        print("=" * 70)
        traceback.print_exc()
        sys.exit(1)
