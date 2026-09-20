# src/experiments/compare_zeroshot_vs_tier2.py
"""
Phase 5C -- zero-shot LLM against the trained classifier, paired.

Phase 5B left the project's accuracy claim resting on Tier-2 alone (BGE +
LogReg). This script tests that claim against the baseline a reviewer will
raise: a zero-shot LLM given the same seven categories and nothing else.

PAIRED, and exact. The same tickets are scored by both, so the comparison is a
McNemar on the discordant pairs, not a difference of two accuracies. The test
itself is imported from compare_cascade_vs_tier2.mcnemar_from_pairs -- one
implementation, shared with Phase 5B, because two would eventually disagree
and the disagreement would be invisible.

WHAT THIS IS NOT. A zero-shot LLM against a classifier trained on 3,200 rows
is not like-for-like, in exactly the way Phase 5B found baseline-vs-Tier-1-only
was not. Whichever way the numbers fall, the claim is "zero-shot LLM vs
TRAINED classifier", and what is really being measured is what the training
corpus is worth. That sentence belongs beside every number this script prints.

Offline: reads CSVs only. No models, no API, zero quota.

Usage (from the project root):
    python src/experiments/run_ablation_study.py --mode tier2-only --set benchmark45
    python src/experiments/run_zeroshot_baselines.py --backend gemini --set both
    python src/experiments/compare_zeroshot_vs_tier2.py --backend gemini
    python src/experiments/compare_zeroshot_vs_tier2.py --backend gemini --set benchmark14
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

from src.experiments.compare_cascade_vs_tier2 import (  # noqa: E402
    _banner, _fatal, _import_scipy, load_classification_rows,
    mcnemar_from_pairs,
)
from src.experiments.run_zeroshot_baselines import _slug  # noqa: E402

DATA_DIR = os.path.join(PROJECT_ROOT, "data")

VALID_SETS = ("benchmark45", "benchmark14")
DEFAULT_GEMINI_MODEL = "gemini-flash-lite-latest"
DEFAULT_OLLAMA_MODEL = "qwen2.5:7b-instruct"


def _to_bool(value):
    text = str(value).strip().lower()
    if text in ("true", "1"):
        return True
    if text in ("false", "0"):
        return False
    _fatal("Could not read {v!r} as a boolean.".format(v=value))


def load_zeroshot_rows(backend, model, eval_set):
    path = os.path.join(
        DATA_DIR,
        "zeroshot_{b}_{m}_{s}.csv".format(b=backend, m=_slug(model),
                                          s=eval_set),
    )
    if not os.path.isfile(path):
        _fatal(
            "Missing zero-shot result:\n  {p}\n\n"
            "Produce it first:\n"
            "  python src/experiments/run_zeroshot_baselines.py "
            "--backend {b} --set {s}".format(p=path, b=backend, s=eval_set)
        )
    try:
        with open(path, "r", newline="", encoding="utf-8") as fh:
            raw = list(csv.DictReader(fh))
    except Exception as exc:
        _fatal("Failed to read {p}: {e}".format(p=path, e=repr(exc)))

    rows = []
    for r in raw:
        rows.append({
            "index": int(r["index"]),
            "ticket_id": r["ticket_id"],
            "text": r["text"],
            "expected": r["expected"],
            "predicted": r["predicted"],
            "parsed_ok": _to_bool(r["parsed_ok"]),
            "correct": _to_bool(r["correct"]),
            "elapsed_s": float(r["elapsed_s"]),
        })

    # Rule 6: recompute the headline from the raw predictions instead of
    # trusting the 'correct' column that produced it. An unparseable answer
    # must never count as correct, which is the specific way this file could
    # be internally consistent and wrong.
    recount = sum(1 for r in rows
                  if r["parsed_ok"] and r["predicted"] == r["expected"])
    from_column = sum(1 for r in rows if r["correct"])
    if recount != from_column:
        _fatal(
            "{p}: the 'correct' column says {a} but recomputing from "
            "predictions gives {b}. Do not use this file.".format(
                p=path, a=from_column, b=recount)
        )
    return rows, path


def pair_by_index(zs_rows, t2_rows):
    """Pair on index, then prove the pairing on the ticket text and label."""
    if len(zs_rows) != len(t2_rows):
        _fatal(
            "Row-count mismatch: zero-shot has {a}, tier2-only has {b}.".format(
                a=len(zs_rows), b=len(t2_rows))
        )
    by_index = {r["index"]: r for r in t2_rows}
    pairs = []
    for z in zs_rows:
        t2 = by_index.get(z["index"])
        if t2 is None:
            _fatal("Ticket index {i} missing from the tier2-only run.".format(
                i=z["index"]))
        if z["text"] != t2["text"] or z["expected"] != t2["expected"]:
            _fatal(
                "Ticket index {i} does not agree between the two files:\n"
                "  zero-shot  expected={ze!r} text={zt!r}\n"
                "  tier2-only expected={te!r} text={tt!r}\n"
                "One file is stale. Re-run both for this set.".format(
                    i=z["index"], ze=z["expected"], zt=z["text"][:60],
                    te=t2["expected"], tt=t2["text"][:60])
            )
        pairs.append((z, t2))
    return pairs


def write_report(res, backend, model, eval_set, force):
    out_path = os.path.join(
        DATA_DIR,
        "zeroshot_vs_tier2_mcnemar_{b}_{m}_{s}.csv".format(
            b=backend, m=_slug(model), s=eval_set),
    )
    if os.path.isfile(out_path) and not force:
        _fatal(
            "Refusing to overwrite an existing result:\n  {p}\n"
            "Pass --force if you mean to replace it.".format(p=out_path)
        )
    fieldnames = ["ticket_id", "text", "expected", "zeroshot_predicted",
                  "zeroshot_correct", "tier2only_predicted",
                  "tier2only_correct", "direction"]
    rows = []
    for direction, bucket in (
        ("zeroshot_right_tier2_wrong", res["b_rows"]),
        ("zeroshot_wrong_tier2_right", res["c_rows"]),
    ):
        for z, t2 in bucket:
            rows.append({
                "ticket_id": z["ticket_id"],
                "text": z["text"],
                "expected": z["expected"],
                "zeroshot_predicted": z["predicted"] if z["parsed_ok"]
                else "UNPARSEABLE",
                "zeroshot_correct": z["correct"],
                "tier2only_predicted": t2["predicted"],
                "tier2only_correct": t2["correct"],
                "direction": direction,
            })
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
        description=("Exact McNemar: a zero-shot LLM against the trained "
                     "Tier-2 classifier, on the same tickets. Offline.")
    )
    parser.add_argument("--backend", required=True,
                        choices=("gemini", "ollama"))
    parser.add_argument("--set", dest="eval_set", default="benchmark45",
                        choices=VALID_SETS)
    parser.add_argument("--model", default=None,
                        help="Model tag as it appears in the result filename.")
    parser.add_argument("--force", action="store_true")
    return parser.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    model = args.model or (DEFAULT_GEMINI_MODEL if args.backend == "gemini"
                           else DEFAULT_OLLAMA_MODEL)
    binomtest = _import_scipy()
    eval_set = args.eval_set

    _banner("Zero-shot {b} ({m})  vs  trained Tier-2  --  {s}".format(
        b=args.backend, m=model, s=eval_set))

    zs_rows, zs_path = load_zeroshot_rows(args.backend, model, eval_set)
    t2_rows, t2_path = load_classification_rows("tier2-only", eval_set)
    print("zero-shot  : " + zs_path)
    print("tier2-only : " + t2_path)
    print("")

    pairs = pair_by_index(zs_rows, t2_rows)
    res = mcnemar_from_pairs(pairs, binomtest)
    n = res["n"]

    n_unparseable = sum(1 for r in zs_rows if not r["parsed_ok"])
    median_s = sorted(r["elapsed_s"] for r in zs_rows)[len(zs_rows) // 2]

    _banner("RESULT")
    print("Zero-shot {b:<8}            : {c}/{n} = {a:.2%}".format(
        b=args.backend, c=res["left_correct"], n=n,
        a=res["left_correct"] / n))
    print("Trained Tier-2 (BGE+LogReg) : {c}/{n} = {a:.2%}".format(
        c=res["right_correct"], n=n, a=res["right_correct"] / n))
    delta = res["left_correct"] - res["right_correct"]
    print("Difference                  : {d:+d} ticket(s), {p:+.2f} points"
          .format(d=delta, p=delta / n * 100.0))
    print("")
    print("Paired 2x2 contingency:")
    print("                        tier2 right      tier2 wrong")
    print("  zero-shot right     {a:>13d}    {b:>13d}".format(
        a=res["both_right"], b=res["b"]))
    print("  zero-shot wrong     {c:>13d}    {d:>13d}".format(
        c=res["c"], d=res["both_wrong"]))
    print("")
    print(res["test_note"].replace("left", "zero-shot").replace(
        "right wrong", "Tier-2 wrong"))
    print("Exact McNemar p = {p:.6f}".format(p=res["p_value"]))
    if res["p_value"] <= 0.05:
        print("  => Distinguishable at alpha=0.05.")
    else:
        print("  => NOT distinguishable at alpha=0.05 on this sample.")

    print("")
    print("Unparseable zero-shot answers : {u}/{n}".format(u=n_unparseable, n=n))
    print("Median wall clock per ticket  : {s:.2f} s  (Tier-2 measured at "
          "0.156 s in Phase 5B)".format(s=median_s))

    print("")
    _banner("DISCORDANT TICKETS ({d})".format(d=res["n_discordant"]))
    for label, bucket in (
        ("ZERO-SHOT RIGHT, Tier-2 wrong", res["b_rows"]),
        ("ZERO-SHOT WRONG, Tier-2 right", res["c_rows"]),
    ):
        for z, t2 in bucket:
            print("")
            print("[{lab}]  {i}".format(lab=label, i=z["ticket_id"]))
            print("  expected   : " + z["expected"])
            print("  zero-shot  : " + (z["predicted"] if z["parsed_ok"]
                                       else "UNPARSEABLE"))
            print("  tier2-only : " + t2["predicted"])
            print("  text       : " + z["text"])

    out_path = write_report(res, args.backend, model, eval_set, args.force)
    print("")
    print("Report written: " + out_path)
    print("")
    print("READ THIS WITH THE NUMBER: a zero-shot LLM against a classifier")
    print("trained on 3,200 rows is NOT like-for-like. What is measured here")
    print("is what the training corpus is worth, not which method is better.")


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
