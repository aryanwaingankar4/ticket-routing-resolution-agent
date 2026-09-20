# src/experiments/compare_cascade_vs_tier2.py
"""
Phase 5B -- the paired comparison that isolates the CASCADE.

The published ablation claim ("the cascade threshold is worth +35.6 accuracy
points") is `baseline` minus `no-cascade`, and `no-cascade` is Tier-1 (TF-IDF)
answering every ticket. That difference is dominated by the BGE-vs-TF-IDF
representation gap; it is not the value of cascading.

The comparison that isolates cascading is `baseline` (cascade @ 0.50) against
`tier2-only` (the production BGE classifier answers everything). The two runs
differ in exactly one thing -- whether Tier-1 was allowed to keep a ticket --
so every difference between them is attributable to the cascade.

They are also PAIRED: the same tickets, scored twice. An unpaired accuracy
difference of one or two tickets at n=45 says nothing, so this script runs an
EXACT McNemar test (a two-sided binomial on the discordant pairs) rather than
quoting the raw gap. The chi-squared form of McNemar is not used: it is a
large-sample approximation and is invalid at these counts.

Reads the CSVs written by run_ablation_study.py; runs no models and calls no
API. Offline, zero Gemini quota.

Usage (from the project root):
    python src/experiments/run_ablation_study.py --mode baseline
    python src/experiments/run_ablation_study.py --mode tier2-only
    python src/experiments/compare_cascade_vs_tier2.py

    python src/experiments/run_ablation_study.py --mode baseline    --set deployment175
    python src/experiments/run_ablation_study.py --mode tier2-only  --set deployment175
    python src/experiments/compare_cascade_vs_tier2.py --set deployment175
"""

import os
import sys
import csv
import argparse
import traceback

THIS_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(THIS_DIR, os.pardir, os.pardir))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

DATA_DIR = os.path.join(PROJECT_ROOT, "data")

DEFAULT_EVAL_SET = "benchmark45"
VALID_SETS = (DEFAULT_EVAL_SET, "deployment175")
EVAL_SET_SIZES = {DEFAULT_EVAL_SET: 45, "deployment175": 175}


def _banner(text):
    rule = "=" * 70
    print(rule)
    print(text)
    print(rule)


def _fatal(message):
    print("")
    print("ERROR: " + str(message))
    sys.exit(1)


def _import_scipy():
    try:
        from scipy.stats import binomtest
    except Exception as exc:
        _fatal(
            "Failed to import scipy.stats.binomtest. Is the venv activated? "
            "scipy is pinned in requirements.txt. " + repr(exc)
        )
    return binomtest


def ablation_csv_path(mode, eval_set):
    suffix = "" if eval_set == DEFAULT_EVAL_SET else "_" + eval_set
    return os.path.join(
        DATA_DIR, "ablation_{m}_results{s}.csv".format(m=mode, s=suffix)
    )


def _to_bool(value):
    """CSV round-trips booleans as the strings 'True'/'False'."""
    if isinstance(value, bool):
        return value
    text = str(value).strip().lower()
    if text in ("true", "1"):
        return True
    if text in ("false", "0"):
        return False
    _fatal("Could not read {v!r} as a boolean.".format(v=value))


def load_classification_rows(mode, eval_set):
    """Read one ablation CSV down to its classification rows.

    baseline on benchmark45 writes a two-section CSV (classification rows plus
    the 9-ticket escalation rows) keyed by 'section'/'key'. Every other file is
    classification-only and keyed by 'index'. Both shapes are normalised here
    to (index, text, expected, predicted, tier1_conf, tier_used, correct).
    """
    path = ablation_csv_path(mode, eval_set)
    if not os.path.isfile(path):
        _fatal(
            "Missing ablation result:\n  {p}\n\n"
            "Produce it first, from the project root:\n"
            "  python src/experiments/run_ablation_study.py --mode {m}"
            "{extra}".format(
                p=path, m=mode,
                extra="" if eval_set == DEFAULT_EVAL_SET
                else " --set " + eval_set,
            )
        )

    try:
        with open(path, "r", newline="", encoding="utf-8") as fh:
            raw = list(csv.DictReader(fh))
    except Exception as exc:
        _fatal("Failed to read {p}: {e}".format(p=path, e=repr(exc)))

    if not raw:
        _fatal("{p} contains no rows.".format(p=path))

    rows = []
    if "section" in raw[0]:
        for r in raw:
            if r.get("section") != "classification":
                continue
            rows.append(
                {
                    "index": int(r["key"]),
                    "text": r["text"],
                    "expected": r["expected"],
                    "predicted": r["predicted"],
                    "tier1_conf": float(r["tier1_conf"]),
                    "tier_used": int(r["tier_used"]),
                    "correct": _to_bool(r["correct"]),
                }
            )
    else:
        for r in raw:
            rows.append(
                {
                    "index": int(r["index"]),
                    "text": r["text"],
                    "expected": r["expected"],
                    "predicted": r["predicted"],
                    "tier1_conf": float(r["tier1_conf"]),
                    "tier_used": int(r["tier_used"]),
                    "correct": _to_bool(r["correct"]),
                }
            )

    expected_n = EVAL_SET_SIZES[eval_set]
    if len(rows) != expected_n:
        _fatal(
            "{p} has {n} classification rows; expected {e} for set {s}. "
            "Re-run the ablation for this set.".format(
                p=path, n=len(rows), e=expected_n, s=eval_set
            )
        )

    # Second, independent derivation of this file's own headline accuracy
    # (rule 6): recount from the raw predictions rather than trusting the
    # 'correct' column the writer produced.
    recount = sum(1 for r in rows if r["predicted"] == r["expected"])
    from_column = sum(1 for r in rows if r["correct"])
    if recount != from_column:
        _fatal(
            "{p}: the 'correct' column says {a} correct but recomputing "
            "predicted == expected gives {b}. The file is internally "
            "inconsistent; do not use it.".format(
                p=path, a=from_column, b=recount
            )
        )

    return rows, path


def pair_rows(cascade_rows, tier2_rows):
    """Pair the two runs by index, proving the pairing rather than assuming it."""
    if len(cascade_rows) != len(tier2_rows):
        _fatal(
            "Row-count mismatch: cascade has {a}, tier2-only has {b}. The two "
            "runs must cover the same set.".format(
                a=len(cascade_rows), b=len(tier2_rows)
            )
        )

    by_index_t2 = {r["index"]: r for r in tier2_rows}
    pairs = []
    for c in cascade_rows:
        t2 = by_index_t2.get(c["index"])
        if t2 is None:
            _fatal(
                "Ticket index {i} is present in the cascade run but missing "
                "from tier2-only.".format(i=c["index"])
            )
        # The index alone could line up while the underlying tickets differ
        # (a stale CSV from another set, for instance). Pin the pairing on the
        # ticket text and its label, which is what actually has to match.
        if c["text"] != t2["text"] or c["expected"] != t2["expected"]:
            _fatal(
                "Ticket index {i} does not agree between the two runs:\n"
                "  cascade   expected={ce!r} text={ct!r}\n"
                "  tier2only expected={te!r} text={tt!r}\n"
                "One of the CSVs is stale. Re-run both modes for this "
                "set.".format(
                    i=c["index"], ce=c["expected"], ct=c["text"][:60],
                    te=t2["expected"], tt=t2["text"][:60],
                )
            )
        pairs.append((c, t2))
    return pairs


def analyse(pairs, binomtest):
    n = len(pairs)
    both_right = both_wrong = 0
    b_rows = []   # cascade right, tier2-only wrong
    c_rows = []   # cascade wrong, tier2-only right

    for casc, t2 in pairs:
        cr, tr = casc["correct"], t2["correct"]
        if cr and tr:
            both_right += 1
        elif not cr and not tr:
            both_wrong += 1
        elif cr and not tr:
            b_rows.append((casc, t2))
        else:
            c_rows.append((casc, t2))

    b, c = len(b_rows), len(c_rows)

    # STRUCTURAL CHECK. On any ticket the cascade escalated, both runs used the
    # SAME Tier-2 prediction, so they cannot disagree. Every discordant pair
    # must therefore be a ticket Tier-1 kept (tier_used == 1 in the cascade
    # run). If that does not hold, the two CSVs are not what they claim to be.
    bad = [
        casc["index"]
        for casc, _ in (b_rows + c_rows)
        if casc["tier_used"] != 1
    ]
    if bad:
        _fatal(
            "Discordant tickets {idx} were answered by Tier-2 in the cascade "
            "run, so both configurations used the same prediction and could "
            "not disagree. One of the CSVs is stale or mismatched.".format(
                idx=bad
            )
        )

    n_discordant = b + c
    if n_discordant == 0:
        p_value = 1.0
        test_note = (
            "No discordant pairs: the two configurations made identical "
            "predictions on every ticket. McNemar is undefined and reported "
            "as p = 1.0."
        )
    else:
        p_value = float(binomtest(b, n_discordant, 0.5).pvalue)
        test_note = (
            "Exact McNemar: two-sided binomial on {d} discordant pair(s), "
            "b={b} (cascade right / Tier-2-only wrong), c={c} (the "
            "reverse).".format(d=n_discordant, b=b, c=c)
        )

    n_tier1 = sum(1 for casc, _ in pairs if casc["tier_used"] == 1)
    tier1_correct = sum(
        1 for casc, _ in pairs if casc["tier_used"] == 1 and casc["correct"]
    )
    # What Tier-2 would have scored on exactly the tickets Tier-1 kept --
    # the like-for-like read on whether keeping them was a good idea.
    tier2_on_kept = sum(
        1 for casc, t2 in pairs if casc["tier_used"] == 1 and t2["correct"]
    )

    return {
        "n": n,
        "cascade_correct": both_right + b,
        "tier2_correct": both_right + c,
        "both_right": both_right,
        "both_wrong": both_wrong,
        "b": b,
        "c": c,
        "b_rows": b_rows,
        "c_rows": c_rows,
        "p_value": p_value,
        "test_note": test_note,
        "n_tier1": n_tier1,
        "tier1_correct": tier1_correct,
        "tier2_on_kept": tier2_on_kept,
    }


def write_report(res, eval_set, force):
    out_path = os.path.join(
        DATA_DIR, "cascade_vs_tier2_mcnemar_{s}.csv".format(s=eval_set)
    )
    if os.path.isfile(out_path) and not force:
        _fatal(
            "Refusing to overwrite an existing result:\n  {p}\n"
            "Pass --force if you really mean to replace it.".format(p=out_path)
        )

    fieldnames = [
        "index",
        "text",
        "expected",
        "cascade_predicted",
        "cascade_correct",
        "tier2only_predicted",
        "tier2only_correct",
        "tier1_conf",
        "cascade_tier_used",
        "direction",
    ]
    rows = []
    for direction, bucket in (
        ("cascade_right_tier2_wrong", res["b_rows"]),
        ("cascade_wrong_tier2_right", res["c_rows"]),
    ):
        for casc, t2 in bucket:
            rows.append(
                {
                    "index": casc["index"],
                    "text": casc["text"],
                    "expected": casc["expected"],
                    "cascade_predicted": casc["predicted"],
                    "cascade_correct": casc["correct"],
                    "tier2only_predicted": t2["predicted"],
                    "tier2only_correct": t2["correct"],
                    "tier1_conf": casc["tier1_conf"],
                    "cascade_tier_used": casc["tier_used"],
                    "direction": direction,
                }
            )

    try:
        with open(out_path, "w", newline="", encoding="utf-8") as fh:
            writer = csv.DictWriter(fh, fieldnames=fieldnames)
            writer.writeheader()
            for row in rows:
                writer.writerow(row)
    except Exception as exc:
        _fatal("Failed to write {p}: {e}".format(p=out_path, e=repr(exc)))
    return out_path


def parse_args(argv):
    parser = argparse.ArgumentParser(
        description=(
            "Exact McNemar test of the cascade against Tier-2 alone, on the "
            "CSVs written by run_ablation_study.py. Offline; no models, no "
            "Gemini."
        )
    )
    parser.add_argument(
        "--set",
        dest="eval_set",
        default=DEFAULT_EVAL_SET,
        choices=VALID_SETS,
        help="Which evaluation set's results to compare.",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Overwrite an existing report file.",
    )
    return parser.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    eval_set = args.eval_set
    binomtest = _import_scipy()

    _banner("Cascade vs Tier-2-only  --  exact McNemar  ({s})".format(s=eval_set))

    cascade_rows, cascade_path = load_classification_rows("baseline", eval_set)
    tier2_rows, tier2_path = load_classification_rows("tier2-only", eval_set)
    print("cascade    : " + cascade_path)
    print("tier2-only : " + tier2_path)
    print("")

    pairs = pair_rows(cascade_rows, tier2_rows)
    res = analyse(pairs, binomtest)

    n = res["n"]
    _banner("RESULT")
    print(
        "Cascade    (Tier-1 @0.50 -> Tier-2) : {c}/{n} = {a:.2%}".format(
            c=res["cascade_correct"], n=n, a=res["cascade_correct"] / n
        )
    )
    print(
        "Tier-2 only (BGE answers everything): {c}/{n} = {a:.2%}".format(
            c=res["tier2_correct"], n=n, a=res["tier2_correct"] / n
        )
    )
    delta = (res["cascade_correct"] - res["tier2_correct"]) / n * 100.0
    print(
        "Difference                          : {d:+d} ticket(s), "
        "{p:+.2f} points".format(
            d=res["cascade_correct"] - res["tier2_correct"], p=delta
        )
    )
    print("")
    print("Paired 2x2 contingency:")
    print("                          tier2-only right   tier2-only wrong")
    print("  cascade right           {a:>16d}   {b:>16d}".format(
        a=res["both_right"], b=res["b"]))
    print("  cascade wrong           {c:>16d}   {d:>16d}".format(
        c=res["c"], d=res["both_wrong"]))
    print("")
    print(res["test_note"])
    print("Exact McNemar p = {p:.6f}".format(p=res["p_value"]))
    if res["p_value"] > 0.05:
        print(
            "  => NOT statistically distinguishable at alpha=0.05. The "
            "accuracy difference\n     between the cascade and Tier-2 alone "
            "is within what this sample can resolve."
        )
    else:
        print("  => Distinguishable at alpha=0.05.")

    print("")
    _banner("WHERE THE CASCADE ACTS")
    print(
        "Tier-1 answered {k}/{n} tickets ({pct:.1f}%); Tier-2 answered the "
        "rest.".format(k=res["n_tier1"], n=n, pct=res["n_tier1"] / n * 100.0)
    )
    if res["n_tier1"]:
        print(
            "On those {k} kept tickets: Tier-1 got {t1} right, Tier-2 would "
            "have got {t2} right.".format(
                k=res["n_tier1"], t1=res["tier1_correct"], t2=res["tier2_on_kept"]
            )
        )
    print(
        "Every discordant ticket is by construction one Tier-1 kept -- "
        "verified above."
    )

    print("")
    _banner("DISCORDANT TICKETS ({d})".format(d=res["b"] + res["c"]))
    if not (res["b"] or res["c"]):
        print("None: the two configurations agreed on every ticket.")
    for label, bucket in (
        ("CASCADE RIGHT, tier2-only wrong", res["b_rows"]),
        ("CASCADE WRONG, tier2-only right", res["c_rows"]),
    ):
        for casc, t2 in bucket:
            print("")
            print("[{lab}]  index {i}".format(lab=label, i=casc["index"]))
            print("  tier1_conf      : {c:.4f}  (cascade used tier {t})".format(
                c=casc["tier1_conf"], t=casc["tier_used"]))
            print("  expected        : " + casc["expected"])
            print("  cascade said    : " + casc["predicted"])
            print("  tier2-only said : " + t2["predicted"])
            print("  text            : " + casc["text"])

    out_path = write_report(res, eval_set, args.force)
    print("")
    print("Report written: " + out_path)


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
