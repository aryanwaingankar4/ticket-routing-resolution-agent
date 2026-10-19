"""Fine-grained cascade-threshold curve on the 175 paraphrased calibration tickets.

Paper revision (Oct 2026), Fig. 4. Offline: no Gemini, no Ollama, no BGE.

The committed record of calibration attempt 3
(data/cascade_calibration_attempts.csv) holds only 10%-wide confidence
buckets, so it gives three populated thresholds. This script re-derives the
per-ticket Tier-1 confidences that produced those buckets and writes the curve
at a fine step -- but ONLY after proving it is the same measurement:

  1. Tier-1 is refitted EXACTLY as train_cascade.main() fitted it for attempt
     3: train_cascade.train_tier1() on all 4,000 rows, max_features path, no
     committed vocabulary. Nothing is reimplemented; the functions are imported.
  2. FATAL unless every bucket count and every bucket accuracy equals the
     committed attempt-3 rows exactly (152/13/7/3/0/0), and unless
     derive_threshold_for_target() returns the committed derived threshold at
     every target accuracy in data/cascade_threshold_sweep.csv.
  3. Cross-check (an independent derivation): the persisted production Tier-1
     (artifacts.load_tier1(), fitted against the committed vocabulary) must
     give identical predictions and confidences within 1e-12.

Writes only under data/paper_revision_oct/ and refuses to overwrite without
--force.

    .\\venv\\Scripts\\python.exe scripts/reproduce_cascade_calibration_curve.py
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

DATA_DIR = os.path.join(PROJECT_ROOT, "data")
OUT_DIR = os.path.join(DATA_DIR, "paper_revision_oct")
ATTEMPTS_CSV = os.path.join(DATA_DIR, "cascade_calibration_attempts.csv")
SWEEP_CSV = os.path.join(DATA_DIR, "cascade_threshold_sweep.csv")
OUT_TICKETS = os.path.join(OUT_DIR, "cascade_calibration_tickets_paraphrased175.csv")
OUT_CURVE = os.path.join(OUT_DIR, "cascade_calibration_curve_paraphrased175.csv")

# The fine grid. Step 0.01 from 0.15 (just above 1/7, the floor of a 7-class
# max-probability) to 0.90 (above the highest populated bucket).
GRID_LO, GRID_HI, GRID_STEP_HUNDREDTHS = 15, 90, 1
CROSS_CHECK_TOL = 1e-12


def die(msg):
    print(f"\n[FATAL] {msg}")
    raise SystemExit(1)


def read_attempt3():
    with open(ATTEMPTS_CSV, newline="", encoding="utf-8") as fh:
        rows = [r for r in csv.DictReader(fh) if r["attempt"] == "3"]
    if not rows:
        die(f"No attempt-3 rows in {ATTEMPTS_CSV}.")
    return rows


def read_sweep():
    with open(SWEEP_CSV, newline="", encoding="utf-8") as fh:
        return [(float(r["target_accuracy"]), float(r["derived_threshold"]))
                for r in csv.DictReader(fh)]


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--force", action="store_true",
                    help="overwrite existing outputs")
    args = ap.parse_args()

    existing = [p for p in (OUT_TICKETS, OUT_CURVE) if os.path.exists(p)]
    if existing and not args.force:
        die("Outputs exist (use --force to overwrite):\n    "
            + "\n    ".join(existing))

    import numpy as np
    from src.classification import train_cascade as tc
    from src.agent.artifacts import load_tier1

    # ---- 1. attempt 3, in its original configuration ----------------------
    df = tc.load_and_validate_csv(tc.CSV_PATH).reset_index(drop=True)
    text_all = tc.build_text_series(df).tolist()
    y_all = df["category"].astype(str).to_numpy()
    print(f"[fit] Tier-1 on all {len(df)} rows (max_features path, as attempt 3)")
    vec, clf = tc.train_tier1(text_all, y_all)

    calib = tc.load_calibration_set()
    texts = [t["text"] for t in calib]
    expected = np.array([t["expected"] for t in calib])
    preds, conf = tc.get_tier1_confidence(vec, clf, texts)
    correct = (np.asarray(preds) == expected).astype(int)
    print(f"[score] {len(texts)} paraphrased calibration tickets")

    # ---- 2. FATAL reproduction checks --------------------------------------
    committed = read_attempt3()
    mine = tc.bucket_stats(conf, correct)
    if len(mine) != len(committed):
        die(f"bucket count differs: {len(mine)} vs {len(committed)} committed")
    for m, c in zip(mine, committed):
        c_cnt = int(c["count"])
        c_acc = float(c["observed_accuracy"]) if c["observed_accuracy"] else None
        m_acc = m["observed_accuracy"] if m["observed_accuracy"] != "" else None
        if m["bucket"] != c["bucket"] or m["count"] != c_cnt or m_acc != c_acc:
            die(f"bucket {c['bucket']}: reproduced count={m['count']} "
                f"acc={m_acc}, committed count={c_cnt} acc={c_acc}")
        print(f"  [ok] {m['bucket']:>7}: n={m['count']:>3}  acc={m_acc}")
    if int(committed[0]["n_tickets"]) != len(texts):
        die(f"n_tickets {committed[0]['n_tickets']} != {len(texts)}")

    for target, committed_thr in read_sweep():
        derived, _ = tc.derive_threshold_for_target(conf, correct, target)
        if derived != committed_thr:
            die(f"target {target}: derived {derived}, committed {committed_thr}")
        print(f"  [ok] target {target:.2f} -> threshold {derived}")

    # ---- 3. cross-check against the persisted production Tier-1 ------------
    p_vec, p_clf = load_tier1()
    p_preds, p_conf = tc.get_tier1_confidence(p_vec, p_clf, texts)
    if not np.array_equal(np.asarray(p_preds), np.asarray(preds)):
        die("production Tier-1 predictions differ from the attempt-3 refit")
    max_delta = float(np.max(np.abs(np.asarray(p_conf) - np.asarray(conf))))
    if max_delta > CROSS_CHECK_TOL:
        die(f"production Tier-1 confidence differs by {max_delta:.3e} "
            f"> {CROSS_CHECK_TOL}")
    print(f"  [ok] production Tier-1 agrees: predictions identical, "
          f"max |dconf| = {max_delta:.3e}")

    # ---- 4. write ----------------------------------------------------------
    os.makedirs(OUT_DIR, exist_ok=True)
    with open(OUT_TICKETS, "w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(["id", "expected", "tier1_predicted", "tier1_conf",
                    "correct"])
        for t, p, c, k in zip(calib, preds, conf, correct):
            w.writerow([t["id"], t["expected"], p, repr(float(c)), int(k)])

    with open(OUT_CURVE, "w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(["threshold", "n_total", "n_kept", "n_kept_correct",
                    "share_kept", "kept_accuracy"])
        for h in range(GRID_LO, GRID_HI + 1, GRID_STEP_HUNDREDTHS):
            thr = h / 100.0
            mask = conf >= thr
            n_kept = int(mask.sum())
            n_ok = int(correct[mask].sum())
            w.writerow([f"{thr:.2f}", len(texts), n_kept, n_ok,
                        repr(n_kept / len(texts)),
                        repr(n_ok / n_kept) if n_kept else ""])

    print(f"\n[write] {os.path.relpath(OUT_TICKETS, PROJECT_ROOT)}")
    print(f"[write] {os.path.relpath(OUT_CURVE, PROJECT_ROOT)}")


if __name__ == "__main__":
    main()
