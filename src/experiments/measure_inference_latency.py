# src/experiments/measure_inference_latency.py
"""
Phase 5B -- warm per-ticket inference latency, Tier-1 vs Tier-2.

WHY THIS EXISTS. The cascade's efficiency claim previously rested on a Phase 3A
measurement of 1.56s to FIT Tier-1 against 0.014s to LOAD it. That is a startup
number and says nothing about what a ticket costs to classify. The cascade only
pays off if answering with Tier-1 is meaningfully cheaper than answering with
Tier-2 AT INFERENCE TIME, on tickets that have already been loaded into a warm
process.

So this measures exactly that: models loaded once, a warmup discarded, then N
single-ticket classifications per tier, reported as median and p95.

  Tier-1 = vectorizer.transform([text]) + classifier.predict_proba
  Tier-2 = embedder.encode([text])      + classifier.predict

Batch size is 1 on purpose: that is what a per-request service does, and it is
the only setting in which "Tier-1 saved us a Tier-2 call" means anything.

The cascade's expected per-ticket cost is reported too, and it is NOT the
Tier-1 cost: the cascade ALWAYS runs Tier-1 and then runs Tier-2 as well
whenever Tier-1 is not confident enough, so its cost is

    tier1 + (share escalated) * tier2

which is strictly greater than Tier-2 alone unless Tier-1 keeps a large share.

Offline. No Gemini. Loads only through src/agent/artifacts.py.

Usage (from the project root):
    python src/experiments/measure_inference_latency.py
    python src/experiments/measure_inference_latency.py --runs 200 --warmup 20
"""

import os
import sys
import csv
import json
import platform
import argparse
import statistics
import traceback
from time import perf_counter_ns

THIS_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(THIS_DIR, os.pardir, os.pardir))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

DATA_DIR = os.path.join(PROJECT_ROOT, "data")
BENCHMARK_JSON_PATH = os.path.join(DATA_DIR, "novel_tickets_expanded.json")

DEFAULT_RUNS = 200
DEFAULT_WARMUP = 20

# Measured Tier-1 share, from the Phase 5B ablation runs. Used only to report
# the cascade's expected per-ticket cost; it is not itself measured here.
TIER1_SHARE = {
    "benchmark45": 4.0 / 45.0,
    "deployment175": 33.0 / 175.0,
}


def _banner(text):
    rule = "=" * 70
    print(rule)
    print(text)
    print(rule)


def _fatal(message):
    print("")
    print("ERROR: " + str(message))
    sys.exit(1)


def describe_cpu():
    """Name the CPU the numbers were produced on.

    A latency number without its hardware is not a result, so this is read at
    run time rather than hardcoded. Falls back progressively and never raises.
    """
    name = ""
    if platform.system() == "Windows":
        try:
            import winreg

            key = winreg.OpenKey(
                winreg.HKEY_LOCAL_MACHINE,
                r"HARDWARE\DESCRIPTION\System\CentralProcessor\0",
            )
            try:
                name = str(winreg.QueryValueEx(key, "ProcessorNameString")[0])
            finally:
                winreg.CloseKey(key)
        except Exception:
            name = ""
    if not name:
        name = platform.processor() or platform.machine() or "unknown CPU"
    return " ".join(name.split())


def cpu_slug(name):
    keep = []
    for ch in name.lower():
        if ch.isalnum():
            keep.append(ch)
        elif keep and keep[-1] != "-":
            keep.append("-")
    return "".join(keep).strip("-") or "unknown-cpu"


def load_benchmark_texts():
    """Ticket text for timing, from the READ-ONLY 45-ticket benchmark.

    Cycling through 45 different tickets rather than repeating one keeps the
    measurement from being a property of a single string's length.
    """
    if not os.path.isfile(BENCHMARK_JSON_PATH):
        _fatal(
            "Missing the 45-ticket benchmark:\n  {p}".format(
                p=BENCHMARK_JSON_PATH
            )
        )
    try:
        with open(BENCHMARK_JSON_PATH, "r", encoding="utf-8") as fh:
            data = json.load(fh)
    except Exception as exc:
        _fatal("Failed to read novel_tickets_expanded.json: " + repr(exc))

    texts = [r["text"] for r in data if isinstance(r, dict) and r.get("text")]
    if len(texts) != 45:
        _fatal(
            "Expected exactly 45 benchmark tickets; found {n}. This set is "
            "read-only and must not have changed.".format(n=len(texts))
        )
    return texts


def time_calls(fn, texts, runs, warmup, label):
    """Run fn(text) warmup+runs times; return the timed durations in ms."""
    print("  {lab}: {w} warmup + {r} timed runs...".format(
        lab=label, w=warmup, r=runs))
    for i in range(warmup):
        fn(texts[i % len(texts)])

    durations_ms = []
    for i in range(runs):
        text = texts[i % len(texts)]
        t0 = perf_counter_ns()
        fn(text)
        t1 = perf_counter_ns()
        durations_ms.append((t1 - t0) / 1e6)
    return durations_ms


def summarise(durations_ms):
    ordered = sorted(durations_ms)
    n = len(ordered)
    # Nearest-rank p95: the smallest value at or above 95% of the sample.
    p95_index = max(0, min(n - 1, -(-95 * n // 100) - 1))
    return {
        "n": n,
        "median_ms": statistics.median(ordered),
        "p95_ms": ordered[p95_index],
        "mean_ms": statistics.fmean(ordered),
        "min_ms": ordered[0],
        "max_ms": ordered[-1],
    }


def parse_args(argv):
    parser = argparse.ArgumentParser(
        description=(
            "Warm per-ticket inference latency for Tier-1 and Tier-2, batch "
            "size 1. Offline; no Gemini."
        )
    )
    parser.add_argument("--runs", type=int, default=DEFAULT_RUNS,
                        help="Timed runs per tier (default {d}).".format(
                            d=DEFAULT_RUNS))
    parser.add_argument("--warmup", type=int, default=DEFAULT_WARMUP,
                        help="Discarded warmup runs per tier (default "
                             "{d}).".format(d=DEFAULT_WARMUP))
    parser.add_argument("--force", action="store_true",
                        help="Overwrite an existing result file.")
    return parser.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    if args.runs < 1 or args.warmup < 0:
        _fatal("--runs must be >= 1 and --warmup >= 0.")

    cpu = describe_cpu()

    _banner("Warm per-ticket inference latency  --  Tier-1 vs Tier-2")
    print("CPU          : " + cpu)
    print("Logical CPUs : {n}".format(n=os.cpu_count()))
    print("Python       : " + platform.python_version())
    print("Platform     : {s} {r}".format(s=platform.system(),
                                          r=platform.release()))
    print("Batch size   : 1 (per-request, not throughput)")
    print("")

    try:
        from src.agent import artifacts as artifacts_mod
    except Exception as exc:
        _fatal("Failed to import src/agent/artifacts.py: " + repr(exc))

    print("Loading artifacts (this is the cost the measurement EXCLUDES)...")
    try:
        art = artifacts_mod.load_artifacts(require_gemini=False)
    except Exception as exc:
        _fatal(
            "Failed to load artifacts via load_artifacts(): {e}\n"
            "Build them first (see CLAUDE.md 'Core pipeline').".format(
                e=repr(exc)
            )
        )

    texts = load_benchmark_texts()

    tier1_vec = art.tier1_vectorizer
    tier1_clf = art.tier1_classifier
    tier2_clf = art.tier2_classifier
    embedder = art.embedder

    def tier1_call(text):
        return tier1_clf.predict_proba(tier1_vec.transform([text]))

    def tier2_call(text):
        emb = embedder.encode([text], show_progress_bar=False,
                              convert_to_numpy=True)
        return tier2_clf.predict(emb)

    print("")
    print("Timing (models already loaded and warm):")
    t1 = summarise(time_calls(tier1_call, texts, args.runs, args.warmup,
                              "Tier-1 (TF-IDF + LogReg)"))
    t2 = summarise(time_calls(tier2_call, texts, args.runs, args.warmup,
                              "Tier-2 (BGE + LogReg)"))

    print("")
    _banner("RESULT")
    print("{h:<28} {med:>12} {p95:>12} {mean:>12}".format(
        h="", med="median (ms)", p95="p95 (ms)", mean="mean (ms)"))
    for label, s in (("Tier-1 (TF-IDF + LogReg)", t1),
                     ("Tier-2 (BGE + LogReg)", t2)):
        print("{h:<28} {med:>12.3f} {p95:>12.3f} {mean:>12.3f}".format(
            h=label, med=s["median_ms"], p95=s["p95_ms"], mean=s["mean_ms"]))

    ratio = t2["median_ms"] / t1["median_ms"] if t1["median_ms"] else float("nan")
    saved_ms = t2["median_ms"] - t1["median_ms"]
    print("")
    print("Tier-2 costs {r:.1f}x Tier-1 at the median "
          "({s:.3f} ms more per ticket).".format(r=ratio, s=saved_ms))

    print("")
    _banner("WHAT THE CASCADE ACTUALLY COSTS")
    print(
        "The cascade ALWAYS runs Tier-1, then runs Tier-2 as well whenever\n"
        "Tier-1 is below 0.50. So its expected per-ticket cost is\n"
        "  tier1 + (1 - tier1_share) * tier2\n"
    )
    rows = []
    for set_name, share in sorted(TIER1_SHARE.items()):
        cascade_ms = t1["median_ms"] + (1.0 - share) * t2["median_ms"]
        delta = cascade_ms - t2["median_ms"]
        print(
            "{s:<15} Tier-1 answers {p:>5.1f}%  ->  cascade {c:7.3f} ms  vs  "
            "Tier-2 alone {t:7.3f} ms  ({d:+.3f} ms)".format(
                s=set_name, p=share * 100.0, c=cascade_ms,
                t=t2["median_ms"], d=delta,
            )
        )
        rows.append({
            "eval_set": set_name,
            "tier1_share": round(share, 6),
            "cascade_expected_median_ms": round(cascade_ms, 4),
            "tier2_only_median_ms": round(t2["median_ms"], 4),
            "delta_ms": round(delta, 4),
        })

    out_path = os.path.join(
        DATA_DIR, "inference_latency_{c}.csv".format(c=cpu_slug(cpu))
    )
    if os.path.isfile(out_path) and not args.force:
        _fatal(
            "Refusing to overwrite an existing result:\n  {p}\n"
            "Pass --force if you really mean to replace it.".format(p=out_path)
        )

    fieldnames = [
        "measurement", "eval_set", "n_runs", "warmup",
        "median_ms", "p95_ms", "mean_ms", "min_ms", "max_ms",
        "tier1_share", "cascade_expected_median_ms", "tier2_only_median_ms",
        "delta_ms", "cpu", "logical_cpus", "python", "platform", "batch_size",
    ]
    env = {
        "cpu": cpu,
        "logical_cpus": os.cpu_count(),
        "python": platform.python_version(),
        "platform": "{s} {r}".format(s=platform.system(), r=platform.release()),
        "batch_size": 1,
    }
    out_rows = []
    for label, s in (("tier1", t1), ("tier2", t2)):
        row = {"measurement": label, "eval_set": "", "n_runs": s["n"],
               "warmup": args.warmup}
        for k in ("median_ms", "p95_ms", "mean_ms", "min_ms", "max_ms"):
            row[k] = round(s[k], 4)
        row.update({"tier1_share": "", "cascade_expected_median_ms": "",
                    "tier2_only_median_ms": "", "delta_ms": ""})
        row.update(env)
        out_rows.append(row)
    for r in rows:
        row = {"measurement": "cascade_expected", "n_runs": "", "warmup": "",
               "median_ms": "", "p95_ms": "", "mean_ms": "", "min_ms": "",
               "max_ms": ""}
        row.update(r)
        row.update(env)
        out_rows.append(row)

    try:
        with open(out_path, "w", newline="", encoding="utf-8") as fh:
            writer = csv.DictWriter(fh, fieldnames=fieldnames)
            writer.writeheader()
            for row in out_rows:
                writer.writerow(row)
    except Exception as exc:
        _fatal("Failed to write {p}: {e}".format(p=out_path, e=repr(exc)))

    print("")
    print("CSV written: " + out_path)
    print("")
    print(
        "LIMITATION, recorded with the number: one machine, one process, "
        "batch size 1,\nCPU only, no other load controlled for. This bounds "
        "per-ticket inference cost\non this hardware; it is not a throughput "
        "or a served-latency measurement."
    )


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
