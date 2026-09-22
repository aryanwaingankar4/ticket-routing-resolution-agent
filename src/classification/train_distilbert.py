#!/usr/bin/env python3
"""
src/classification/train_distilbert.py

Fine-tunes distilbert-base-uncased as a 7-class IT-support-ticket classifier.

CHANGED BEHAVIOR (per-epoch checkpointing + generalization tracking):
  * After EVERY epoch it (1) saves a full checkpoint to a PER-EPOCH folder
    models/distilbert_ticket_classifier/epoch_{N}/ (all epochs kept on disk,
    never overwritten), (2) reloads that just-saved checkpoint and runs the
    SAME 14-ticket generalization test against it, and (3) prints a live
    per-epoch summary line.
  * At the very end it prints a summary table across ALL epochs and
    automatically identifies the BEST-GENERALIZING epoch (not the best
    in-distribution epoch, which saturates early via memorization).
  * The final 3-way comparison is computed against the BEST-GENERALIZING
    epoch's score, alongside:
        1. TF-IDF + LogisticRegression:          7/14  (50.0%)
        2. Frozen MiniLM embeddings + LogReg:    10/14  (71.4%)
        3. Fine-tuned DistilBERT (best epoch):    X/14  (Y.Y%)

  * Resume logic: if the LAST epoch folder (epoch_4/) already contains a valid
    checkpoint, training is treated as already completed and the script skips
    straight to re-running the summary/comparison across whichever epoch_N/
    folders exist on disk, without retraining.

CPU-ONLY. No GPU/CUDA assumed anywhere.

Run from the project root:
    python src/classification/train_distilbert.py
"""

import os
import sys
import json
import argparse
import csv
import json
import shutil
import time
import random

# ----------------------------------------------------------------------------
# 0. Dependency guard (ImportError -> actionable pip command, no raw traceback)
# ----------------------------------------------------------------------------
try:
    import numpy as np
    import pandas as pd
except ImportError as e:
    print("[ERROR] Missing a core dependency: {}".format(e))
    print("        Install it with:  pip install pandas numpy scikit-learn")
    sys.exit(1)

try:
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import (
        accuracy_score,
        f1_score,
        classification_report,
        confusion_matrix,
    )
except ImportError as e:
    print("[ERROR] scikit-learn is required for metrics/split: {}".format(e))
    print("        Install it with:  pip install scikit-learn")
    sys.exit(1)

try:
    import torch
    from torch.utils.data import Dataset, DataLoader
    from transformers import (
        DistilBertTokenizerFast,
        DistilBertForSequenceClassification,
    )
except ImportError as e:
    print("[ERROR] transformers and/or torch are not installed: {}".format(e))
    print("        Install them with:  pip install transformers torch")
    sys.exit(1)


# ----------------------------------------------------------------------------
# 9. Reproducibility: seed everything to 42
# ----------------------------------------------------------------------------
SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
# Phase 8A.1: seed everything else that is reachable. CPU fine-tuning is not
# guaranteed to be bit-reproducible across library versions even so, which is
# exactly why the metrics file records the torch/transformers versions beside
# every number it carries.
os.environ.setdefault("PYTHONHASHSEED", str(SEED))
try:
    torch.cuda.manual_seed_all(SEED)
except Exception:
    pass
# Keep torch deterministic-ish on CPU; harmless if already default.
try:
    torch.use_deterministic_algorithms(False)  # avoid errors on ops without det impl
except Exception:
    pass


def seeded_generator():
    """A DataLoader shuffle generator with its own fixed seed.

    Without this, shuffling draws from the global torch RNG, whose state
    depends on everything that consumed it earlier in the process -- so the
    batch order changes if anything upstream changes. Pinning it here is the
    difference between "seeded" and "reproducible".
    """
    gen = torch.Generator()
    gen.manual_seed(SEED)
    return gen


# ----------------------------------------------------------------------------
# 1. Device: CPU-ONLY. Never touch CUDA.
# ----------------------------------------------------------------------------
DEVICE = torch.device("cpu")
# Cap CPU threads to something sane so a laptop stays usable; comment out if
# you want torch to grab all cores.
try:
    torch.set_num_threads(max(1, os.cpu_count() or 1))
except Exception:
    pass


# ----------------------------------------------------------------------------
# Paths (resolve relative to THIS file; project root = two dirs up)
# ----------------------------------------------------------------------------
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, "..", ".."))
CSV_PATH = os.path.join(PROJECT_ROOT, "data", "synthetic_tickets.csv")
MODEL_DIR = os.path.join(PROJECT_ROOT, "models", "distilbert_ticket_classifier")

# Phase 8A.1. This script wrote only label_mapping.json, so the published
# "fine-tuned DistilBERT, 7/14" had no committed machine-readable source and
# the 45-ticket benchmark had never been run against it at all. It now writes
# a metrics file covering both benchmarks, for every epoch.
#
# The 45-ticket set is READ-ONLY, like every benchmark in this project.
EXPANDED_JSON_PATH = os.path.join(PROJECT_ROOT, "data",
                                  "novel_tickets_expanded.json")
METRICS_CSV_PATH = os.path.join(PROJECT_ROOT, "data",
                                "distilbert_finetune_metrics.csv")
EXPECTED_EXPANDED_COUNT = 45

MODEL_NAME = "distilbert-base-uncased"

# ----------------------------------------------------------------------------
# Hyperparameters (CPU-friendly, ~10-20 min budget)
# ----------------------------------------------------------------------------
NUM_EPOCHS = 4
BATCH_SIZE = 16
MAX_LENGTH = 128
LEARNING_RATE = 2e-5
WEIGHT_DECAY = 0.01

# 7 expected categories (used to validate NOVEL_TICKETS "expected" labels).
EXPECTED_CATEGORIES = {
    "Infrastructure",
    "Application",
    "Security",
    "Database",
    "Storage",
    "Network",
    "Access Management",
}

# Network error markers (same style as the embeddings scripts).
NETWORK_MARKERS = (
    "connection",
    "timeout",
    "timed out",
    "network",
    "proxy",
    "huggingface.co",
    "max retries",
    "temporary failure",
    "name resolution",
    "unreachable",
    "ssl",
    "connectionerror",
    "newconnectionerror",
)


# ----------------------------------------------------------------------------
# The EXACT 14 novel tickets (copied verbatim). Do not edit.
# ----------------------------------------------------------------------------
NOVEL_TICKETS = [
    {"text": "I got a new laptop and now the system keeps saying my username "
             "or password is wrong even though I'm sure it's right. Can someone "
             "reset me so I can get back in?",
     "expected": "Access Management"},
    {"text": "My manager said I should be able to see the finance shared folder "
             "but it just says I'm not allowed. Can you give me permission to open it?",
     "expected": "Access Management"},
    {"text": "The wifi in the third floor meeting room keeps dropping every few "
             "minutes so we can't run our video calls. Everyone else in the "
             "building seems fine.",
     "expected": "Network"},
    {"text": "Pages take forever to load today and sometimes just time out. My "
             "colleague next to me has the same slowness on her machine too.",
     "expected": "Network"},
    {"text": "I keep getting a message that there's no room left to save my files "
             "and I can't download the report I need. It says the drive is full.",
     "expected": "Storage"},
    {"text": "I tried to save my presentation but it won't let me because it says "
             "I've run out of space in my folder. How do I free some up?",
     "expected": "Storage"},
    {"text": "When I try to pull up last month's customer records the whole thing "
             "just spins and then shows an error about not being able to reach the "
             "records system.",
     "expected": "Database"},
    {"text": "The report tool says it can't find the numbers it needs and mentions "
             "something about a broken connection to where the data is kept.",
     "expected": "Database"},
    {"text": "I got a weird email pretending to be from HR asking me to type in my "
             "password on a link, and I think I might have clicked it by mistake. "
             "What should I do?",
     "expected": "Security"},
    {"text": "My antivirus popped up a warning that something suspicious was blocked "
             "and now I'm worried my computer has a virus on it.",
     "expected": "Security"},
    {"text": "Every time I open the expense program it freezes on the loading screen "
             "and then closes itself. I've tried restarting but it keeps crashing.",
     "expected": "Application"},
    {"text": "The invoicing software gives me an error and shuts down whenever I "
             "click the print button. Nothing prints at all.",
     "expected": "Application"},
    {"text": "None of the company websites are working for anyone in the office this "
             "morning - it looks like one of the main servers might be down.",
     "expected": "Infrastructure"},
    {"text": "The whole team can't reach any of our internal tools and someone said "
             "a machine in the server room overheated and shut off overnight.",
     "expected": "Infrastructure"},
]


# ----------------------------------------------------------------------------
# Per-epoch checkpoint folder helper
# ----------------------------------------------------------------------------
def epoch_dir(n):
    """Folder for a specific epoch's checkpoint."""
    return os.path.join(MODEL_DIR, "epoch_{}".format(n))


# ----------------------------------------------------------------------------
# Phase 8A.1: the 45-ticket benchmark, and the metrics file.
# ----------------------------------------------------------------------------
def load_expanded_benchmark():
    """Load the read-only 45-ticket benchmark, with the project's count gate.

    The count check is not decoration: silently scoring 44 or 46 tickets would
    produce a plausible, internally-consistent, wrong number -- this repo's
    recurring bug class.
    """
    if not os.path.exists(EXPANDED_JSON_PATH):
        print("[ERROR] 45-ticket benchmark not found at:\n        {}".format(
            EXPANDED_JSON_PATH))
        print("        This file is read-only project data and must exist.")
        sys.exit(1)
    try:
        with open(EXPANDED_JSON_PATH, "r", encoding="utf-8") as fh:
            data = json.load(fh)
    except Exception as e:  # noqa: BLE001
        print("[ERROR] Failed to read {}: {}".format(EXPANDED_JSON_PATH, e))
        sys.exit(1)
    if not isinstance(data, list) or len(data) != EXPECTED_EXPANDED_COUNT:
        print("[ERROR] {} must contain exactly {} records, found {}.".format(
            EXPANDED_JSON_PATH, EXPECTED_EXPANDED_COUNT,
            len(data) if isinstance(data, list) else "a non-list"))
        sys.exit(1)
    for i, rec in enumerate(data):
        if not isinstance(rec, dict) or "text" not in rec \
                or "expected" not in rec:
            print("[ERROR] {} record #{} is missing 'text' or "
                  "'expected'.".format(EXPANDED_JSON_PATH, i))
            sys.exit(1)
        if rec["expected"] not in EXPECTED_CATEGORIES:
            print("[ERROR] {} record #{} has expected label {!r}, which is "
                  "not a training category.".format(
                      EXPANDED_JSON_PATH, i, rec["expected"]))
            sys.exit(1)
    return data


def score_benchmark(model, tokenizer, id2label, tickets, verbose=False):
    """Score a model against a list of {text, expected} tickets.

    ONE implementation for both benchmarks, so the 14- and 45-ticket numbers
    cannot drift apart through a copied-and-edited scoring loop.
    """
    texts = [t["text"] for t in tickets]
    expected = [t["expected"] for t in tickets]
    pred_ids = predict_ids(model, tokenizer, texts)
    pred_labels = [id2label[i] for i in pred_ids]

    correct = 0
    for want, got, text in zip(expected, pred_labels, texts):
        is_correct = (want == got)
        if is_correct:
            correct += 1
        if verbose:
            tag = "[CORRECT]" if is_correct else "[WRONG]  "
            print("{} expected={} predicted={}".format(tag, want, got))
            print("    {}".format(text))

    n = len(tickets)
    pct = 100.0 * correct / n if n else 0.0
    return correct, n, pct


def environment_stamp():
    """Library versions that can move a fine-tuning result on their own."""
    try:
        import transformers
        transformers_version = transformers.__version__
    except Exception:  # noqa: BLE001
        transformers_version = "unknown"
    return {
        "torch_version": torch.__version__,
        "transformers_version": transformers_version,
        "seed": SEED,
        "device": str(DEVICE),
        "num_epochs": NUM_EPOCHS,
        "batch_size": BATCH_SIZE,
        "max_length": MAX_LENGTH,
        "learning_rate": LEARNING_RATE,
        "weight_decay": WEIGHT_DECAY,
    }


def write_metrics_csv(run_label, epoch_records, path=METRICS_CSV_PATH,
                      append=False):
    """One row per epoch, with the environment stamped on every row.

    `append` lets the evaluate-existing pass and the fresh retrain land in one
    file without either overwriting the other.
    """
    stamp = environment_stamp()
    fieldnames = ["run", "epoch", "val_accuracy",
                  "benchmark14_correct", "benchmark14_total",
                  "benchmark45_correct", "benchmark45_total",
                  "is_best_epoch"] + sorted(stamp)
    exists = os.path.exists(path)
    mode = "a" if (append and exists) else "w"
    with open(path, mode, encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        if mode == "w":
            writer.writeheader()
        for rec in epoch_records:
            row = {
                "run": run_label,
                "epoch": rec["epoch"],
                "val_accuracy": ("" if rec.get("val_acc") is None
                                 else "{:.6f}".format(rec["val_acc"])),
                "benchmark14_correct": rec["gen_correct"],
                "benchmark14_total": rec["gen_n"],
                "benchmark45_correct": rec.get("gen45_correct", ""),
                "benchmark45_total": rec.get("gen45_n", ""),
                "is_best_epoch": int(bool(rec.get("is_best"))),
            }
            row.update(stamp)
            writer.writerow(row)
    return path


# ============================================================================
# CSV validation (same gates/messages style as the other scripts)
# ============================================================================
def load_and_validate_csv():
    if not os.path.exists(CSV_PATH):
        print("[ERROR] CSV not found at: {}".format(CSV_PATH))
        print("        Expected data/synthetic_tickets.csv relative to project root.")
        sys.exit(1)

    try:
        df = pd.read_csv(CSV_PATH)
    except Exception as e:
        print("[ERROR] Failed to read CSV: {}".format(e))
        sys.exit(1)

    required_cols = ["id", "title", "description", "category",
                     "resolution", "priority"]
    missing = [c for c in required_cols if c not in df.columns]
    if missing:
        print("[ERROR] CSV is missing required columns: {}".format(missing))
        print("        Required columns: {}".format(required_cols))
        sys.exit(1)

    # No NaNs in the fields we actually use for training.
    for col in ["title", "description", "category"]:
        n_nan = df[col].isna().sum()
        if n_nan > 0:
            print("[ERROR] Column '{}' contains {} missing value(s). "
                  "Please clean the CSV.".format(col, n_nan))
            sys.exit(1)

    categories = sorted(df["category"].unique().tolist())
    if len(categories) < 7:
        print("[ERROR] Expected at least 7 categories, found {}: {}".format(
            len(categories), categories))
        sys.exit(1)

    # Enough rows per class to stratify an 80/20 split (need >= 2 per class,
    # realistically >= 5 to have a test representative).
    class_counts = df["category"].value_counts()
    too_small = class_counts[class_counts < 5]
    if len(too_small) > 0:
        print("[ERROR] Some categories have too few rows to stratify safely:")
        for cat, cnt in too_small.items():
            print("          {}: {} rows".format(cat, cnt))
        print("        Each category needs at least ~5 rows.")
        sys.exit(1)

    print("[ok] CSV loaded: {} rows, {} categories.".format(
        len(df), len(categories)))
    return df, categories


def validate_novel_tickets(training_categories):
    """Every NOVEL_TICKETS 'expected' label must match a real training category."""
    training_set = set(training_categories)
    mismatches = []
    for i, t in enumerate(NOVEL_TICKETS):
        exp = t["expected"]
        if exp not in training_set:
            mismatches.append((i, exp))
    if mismatches:
        print("[ERROR] NOVEL_TICKETS contains 'expected' labels that are NOT "
              "real training categories:")
        for idx, lab in mismatches:
            print("          ticket #{}: expected='{}'".format(idx, lab))
        print("        Training categories are: {}".format(
            sorted(training_set)))
        sys.exit(1)
    print("[ok] All 14 novel-ticket expected labels match training categories.")


# ============================================================================
# Dataset
# ============================================================================
class TicketDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, max_length):
        self.encodings = tokenizer(
            list(texts),
            truncation=True,
            padding="max_length",
            max_length=max_length,
            return_tensors="pt",
        )
        self.labels = torch.tensor(list(labels), dtype=torch.long)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        item = {k: v[idx] for k, v in self.encodings.items()}
        item["labels"] = self.labels[idx]
        return item


# ============================================================================
# Checkpoint helpers (now PER-EPOCH aware)
# ============================================================================
def checkpoint_is_valid(ckpt_dir):
    """A valid checkpoint = model weights + config + tokenizer + label_mapping."""
    if not os.path.isdir(ckpt_dir):
        return False
    if not os.path.exists(os.path.join(ckpt_dir, "label_mapping.json")):
        return False
    has_weights = any(
        os.path.exists(os.path.join(ckpt_dir, f))
        for f in ("model.safetensors", "pytorch_model.bin")
    )
    has_config = os.path.exists(os.path.join(ckpt_dir, "config.json"))
    has_tokenizer = os.path.exists(os.path.join(ckpt_dir, "tokenizer_config.json"))
    return has_weights and has_config and has_tokenizer


def load_checkpoint(ckpt_dir):
    with open(os.path.join(ckpt_dir, "label_mapping.json"), "r",
              encoding="utf-8") as f:
        mapping = json.load(f)
    label2id = {k: int(v) for k, v in mapping["label2id"].items()}
    id2label = {int(k): v for k, v in mapping["id2label"].items()}

    tokenizer = DistilBertTokenizerFast.from_pretrained(ckpt_dir)
    model = DistilBertForSequenceClassification.from_pretrained(ckpt_dir)
    model.to(DEVICE)
    model.eval()
    return model, tokenizer, label2id, id2label


def save_checkpoint(ckpt_dir, model, tokenizer, label2id, id2label):
    os.makedirs(ckpt_dir, exist_ok=True)
    model.save_pretrained(ckpt_dir)
    tokenizer.save_pretrained(ckpt_dir)
    mapping = {
        "label2id": {k: int(v) for k, v in label2id.items()},
        "id2label": {str(k): v for k, v in id2label.items()},
    }
    with open(os.path.join(ckpt_dir, "label_mapping.json"), "w",
              encoding="utf-8") as f:
        json.dump(mapping, f, indent=2, ensure_ascii=False)
    print("[ok] Checkpoint saved to: {}".format(ckpt_dir))
    print("     - model.save_pretrained() / tokenizer.save_pretrained()")
    print("     - label_mapping.json")


# ============================================================================
# Prediction helper (batched, CPU)
# ============================================================================
@torch.no_grad()
def predict_ids(model, tokenizer, texts, max_length=MAX_LENGTH, batch_size=32):
    model.eval()
    preds = []
    texts = list(texts)
    for start in range(0, len(texts), batch_size):
        chunk = texts[start:start + batch_size]
        enc = tokenizer(
            chunk,
            truncation=True,
            padding="max_length",
            max_length=max_length,
            return_tensors="pt",
        )
        enc = {k: v.to(DEVICE) for k, v in enc.items()}
        logits = model(**enc).logits
        batch_preds = torch.argmax(logits, dim=-1).cpu().numpy().tolist()
        preds.extend(batch_preds)
    return preds


# ============================================================================
# Evaluate on a DataLoader (returns avg loss + accuracy + preds/trues)
# ============================================================================
@torch.no_grad()
def evaluate_loader(model, loader):
    model.eval()
    total_loss = 0.0
    n_batches = 0
    all_preds, all_trues = [], []
    for batch in loader:
        batch = {k: v.to(DEVICE) for k, v in batch.items()}
        out = model(**batch)
        total_loss += out.loss.item()
        n_batches += 1
        preds = torch.argmax(out.logits, dim=-1).cpu().numpy().tolist()
        trues = batch["labels"].cpu().numpy().tolist()
        all_preds.extend(preds)
        all_trues.extend(trues)
    avg_loss = total_loss / max(1, n_batches)
    acc = accuracy_score(all_trues, all_preds)
    return avg_loss, acc, all_preds, all_trues


# ============================================================================
# Confusion matrix printer (rows=true, cols=predicted), same style as before
# ============================================================================
def print_confusion_matrix(y_true, y_pred, label_names):
    cm = confusion_matrix(y_true, y_pred, labels=list(range(len(label_names))))
    # Short header codes so the grid stays readable.
    codes = []
    seen = {}
    for name in label_names:
        code = "".join(w[0] for w in name.split())[:3].upper()
        # de-dup
        base = code
        k = 1
        while code in seen:
            k += 1
            code = base + str(k)
        seen[code] = True
        codes.append(code)

    print("\nConfusion matrix (rows=true, cols=predicted):")
    print("Legend: " + ", ".join(
        "{}={}".format(c, n) for c, n in zip(codes, label_names)))
    colw = 6
    header = " " * 8 + "".join("{:>{w}}".format(c, w=colw) for c in codes)
    print(header)
    for i, row in enumerate(cm):
        line = "{:>8}".format(codes[i])
        line += "".join("{:>{w}}".format(int(v), w=colw) for v in row)
        print(line)


# ============================================================================
# Generalization test on the 14 novel tickets.
#   verbose=True  -> prints the full per-ticket CORRECT/WRONG breakdown
#                    (used for the final report of the best epoch)
#   verbose=False -> silent, just returns (correct, n, pct) for the live
#                    per-epoch summary line
# ============================================================================
def run_generalization_test(model, tokenizer, id2label, verbose=False):
    # Phase 8A.1: delegates to score_benchmark() so the 14- and 45-ticket
    # numbers come from one scoring loop rather than two copies of it.
    return score_benchmark(model, tokenizer, id2label, NOVEL_TICKETS,
                           verbose=verbose)


# ============================================================================
# Final 3-way comparison, computed against a chosen (best) epoch's score.
# ============================================================================
def read_tfidf_benchmark14():
    """The TF-IDF baseline's 14-ticket score, READ FROM ITS RESULT FILE.

    Phase 8A.1: this was a hardcoded 7/14 -- a retyped number with no source,
    and re-running the baseline showed it is now 6/14. Reading the committed
    file means this comparison cannot silently go stale again. Returns None if
    the baseline has not been run, and the caller says so rather than guessing.
    """
    path = os.path.join(PROJECT_ROOT, "data", "baseline_tfidf_benchmark14.csv")
    if not os.path.exists(path):
        return None
    try:
        with open(path, "r", encoding="utf-8") as fh:
            for row in csv.DictReader(fh):
                if row["fit_scope"] == "full4000" \
                        and row["ticket_index"] == "TOTAL":
                    return int(row["n_correct"]), int(row["n_total"])
    except Exception:  # noqa: BLE001
        return None
    return None


def print_three_way_comparison(correct, n, pct, best_epoch):
    tfidf = read_tfidf_benchmark14()
    minilm_correct, minilm_pct = 10, 71.4

    print("\n" + "=" * 70)
    print(" 3-WAY GENERALIZATION COMPARISON (same 14 novel tickets)")
    print("=" * 70)
    print("(DistilBERT figure below is the BEST-GENERALIZING epoch: "
          "epoch {})".format(best_epoch))
    if tfidf is None:
        print("TF-IDF baseline:            not available -- run "
              "generalization_test.py to produce "
              "data/baseline_tfidf_benchmark14.csv")
    else:
        tfidf_correct, tfidf_n = tfidf
        tfidf_pct = 100.0 * tfidf_correct / tfidf_n
        print("TF-IDF baseline:            {}/{} ({:.1f}%)".format(
            tfidf_correct, tfidf_n, tfidf_pct))
    print("Frozen embeddings (MiniLM): {}/14 ({:.1f}%)".format(
        minilm_correct, minilm_pct))
    print("Fine-tuned DistilBERT:      {}/{} ({:.1f}%)".format(
        correct, n, pct))

    print("\nDeltas (Fine-tuned DistilBERT vs each baseline):")
    if tfidf is not None:
        print("  vs TF-IDF:            {:+d} tickets  "
              "({:+.1f} pct points)".format(
                  correct - tfidf_correct, pct - tfidf_pct))
    d_minilm = correct - minilm_correct
    print("  vs MiniLM embeddings: {:+d} tickets  ({:+.1f} pct points)".format(
        d_minilm, pct - minilm_pct))

    if correct > minilm_correct:
        print("\n=> DistilBERT generalizes best of the three on this test.")
    elif correct == minilm_correct:
        print("\n=> DistilBERT ties the MiniLM embeddings on this test.")
    else:
        print("\n=> DistilBERT does NOT beat the MiniLM embeddings here - "
              "likely memorizing template patterns. Check the per-epoch "
              "val curve above for overfitting.")


# ============================================================================
# End-of-run summary table + best-generalizing-epoch selection.
#   epoch_records: list of dicts with keys:
#       epoch, val_acc (0-1 or None), gen_correct, gen_n, gen_pct
# Returns the record of the best-generalizing epoch.
# ============================================================================
def print_summary_and_pick_best(epoch_records):
    print("\n" + "=" * 70)
    print(" PER-EPOCH SUMMARY (all checkpoints kept on disk)")
    print("=" * 70)
    print("  Epoch | In-Dist Val Acc | Generalization Score")
    print("  ------+-----------------+----------------------")
    for r in epoch_records:
        if r["val_acc"] is None:
            val_str = "   n/a   "
        else:
            val_str = "{:6.2f}%".format(r["val_acc"] * 100.0)
        gen_str = "{}/{} ({:.1f}%)".format(
            r["gen_correct"], r["gen_n"], r["gen_pct"])
        print("    {:<2}  |     {:^9}   |   {}".format(
            r["epoch"], val_str, gen_str))

    # Best = highest generalization correct-count. Tie-break: earliest epoch,
    # since later epochs at the same score are the more memorized/less
    # preferable checkpoint.
    best = None
    for r in epoch_records:
        if best is None or r["gen_correct"] > best["gen_correct"]:
            best = r
    # (iteration order is ascending epoch, so first-seen max == earliest epoch)

    best_folder = epoch_dir(best["epoch"])
    print("\n=> BEST GENERALIZING CHECKPOINT: Epoch {} ({}/{}, {:.1f}%)".format(
        best["epoch"], best["gen_correct"], best["gen_n"], best["gen_pct"]))
    print("    Recommendation: use {}/ for".format(best_folder))
    print("    inference, not the final epoch, since later epochs show declining "
          "or")
    print("    flat generalization while in-distribution accuracy is already "
          "saturated")
    print("    (memorization, not improvement).")
    return best


# ============================================================================
# Rebuild the summary from whatever epoch_N folders already exist on disk
# (used by the "training already completed" resume path).
# ============================================================================
def summarize_existing_checkpoints(run_label="existing_checkpoints",
                                   append_metrics=False):
    expanded = load_expanded_benchmark()
    epoch_records = []
    found_any = False
    for n in range(1, NUM_EPOCHS + 1):
        ckpt_dir = epoch_dir(n)
        if not checkpoint_is_valid(ckpt_dir):
            continue
        found_any = True
        try:
            model, tokenizer, label2id, id2label = load_checkpoint(ckpt_dir)
        except Exception as e:
            print("[warn] Could not load {} ({}); skipping it.".format(
                ckpt_dir, e))
            continue
        gen_correct, gen_n, gen_pct = run_generalization_test(
            model, tokenizer, id2label, verbose=False)
        gen45_correct, gen45_n, gen45_pct = score_benchmark(
            model, tokenizer, id2label, expanded, verbose=False)
        # In-dist val acc is not recomputed here (no split reload needed for
        # the resume summary); mark as n/a.
        print("Epoch {}/{}: in-dist val_acc=n/a (resumed) | "
              "benchmark14={}/{} ({:.1f}%) | benchmark45={}/{} "
              "({:.1f}%)".format(
                  n, NUM_EPOCHS, gen_correct, gen_n, gen_pct,
                  gen45_correct, gen45_n, gen45_pct))
        epoch_records.append({
            "epoch": n,
            "val_acc": None,
            "gen_correct": gen_correct,
            "gen_n": gen_n,
            "gen_pct": gen_pct,
            "gen45_correct": gen45_correct,
            "gen45_n": gen45_n,
            "gen45_pct": gen45_pct,
        })
        # free memory between checkpoints
        del model, tokenizer

    if not found_any:
        print("[ERROR] Last-epoch folder looked valid but no epoch_N "
              "checkpoints could be summarized.")
        sys.exit(1)

    best = print_summary_and_pick_best(epoch_records)
    for rec in epoch_records:
        rec["is_best"] = (rec["epoch"] == best["epoch"])
    path = write_metrics_csv(run_label, epoch_records,
                             append=append_metrics)
    print("\n[write] Per-epoch metrics ({}) -> {}".format(run_label, path))

    # Re-run the verbose generalization report + 3-way comparison on the best.
    best_dir = epoch_dir(best["epoch"])
    b_model, b_tok, b_l2i, b_i2l = load_checkpoint(best_dir)
    print("\n" + "=" * 70)
    print(" GENERALIZATION TEST (best epoch {} - 14 novel tickets)".format(
        best["epoch"]))
    print("=" * 70)
    correct, n, pct = run_generalization_test(b_model, b_tok, b_i2l, verbose=True)
    print("\nGeneralization score: {}/{} ({:.1f}%)".format(correct, n, pct))
    print_three_way_comparison(correct, n, pct, best["epoch"])


# ============================================================================
# Main
# ============================================================================
def parse_args():
    parser = argparse.ArgumentParser(
        description="Fine-tune DistilBERT for IT-support ticket "
                    "classification (CPU).")
    parser.add_argument(
        "--backup-existing", action="store_true",
        help="Move an existing models/distilbert_ticket_classifier/ aside "
             "(never delete it) and train from scratch, instead of resuming "
             "from it. Added in Phase 8A.1 so a reproduction run cannot "
             "silently re-report the previous run's checkpoints.")
    parser.add_argument(
        "--run-label", default=None,
        help="Label recorded in the 'run' column of "
             "data/distilbert_finetune_metrics.csv. Defaults to "
             "'existing_checkpoints' when resuming and 'fresh_retrain' when "
             "training.")
    parser.add_argument(
        "--append-metrics", action="store_true",
        help="Append to the metrics CSV instead of overwriting it, so the "
             "evaluate-existing pass and a later retrain land in one file.")
    return parser.parse_args()


def main():
    args = parse_args()
    print("=" * 70)
    print(" Fine-tune DistilBERT for IT-support ticket classification (CPU)")
    print("=" * 70)
    print("[device] Using device: {}".format(DEVICE))
    if DEVICE.type != "cpu":
        print("[warn] This script is intended for CPU-only runs.")
    print("[info] Project root: {}".format(PROJECT_ROOT))
    print("[info] Model dir:    {}".format(MODEL_DIR))

    # ---- Load + validate data -------------------------------------------
    df, categories = load_and_validate_csv()
    validate_novel_tickets(categories)

    # ---- Build input text and labels ------------------------------------
    df = df.copy()
    df["input_text"] = (df["title"].astype(str) + " " +
                        df["description"].astype(str))
    X = df["input_text"].tolist()
    y_str = df["category"].tolist()

    # Stable label ordering -> deterministic id assignment.
    label_names = sorted(set(y_str))
    label2id = {name: i for i, name in enumerate(label_names)}
    id2label = {i: name for name, i in label2id.items()}
    y = [label2id[c] for c in y_str]

    # ---- IDENTICAL split as the other two scripts -----------------------
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=SEED, stratify=y
    )
    print("[split] Train: {}  |  Test (held-out): {}".format(
        len(X_train), len(X_test)))

    # Phase 8A.1: the 45-ticket benchmark, loaded (and count-gated) up front
    # so a missing or malformed file fails before any training time is spent.
    expanded_benchmark = load_expanded_benchmark()
    print("[benchmark] Loaded {} expanded benchmark tickets.".format(
        len(expanded_benchmark)))

    # ---- Phase 8A.1: optionally move an existing run aside --------------
    if args.backup_existing and os.path.isdir(MODEL_DIR):
        stamp = time.strftime("%Y%m%d-%H%M%S")
        backup = "{}.pre-8a1-{}".format(MODEL_DIR, stamp)
        shutil.move(MODEL_DIR, backup)
        print("[backup] Moved the previous checkpoints to:\n"
              "         {}".format(backup))
        print("         They are MOVED, not deleted -- the previous run's "
              "artifacts remain on disk.")

    # ---- Resume path: last-epoch folder already complete? ---------------
    last_epoch_dir = epoch_dir(NUM_EPOCHS)
    if checkpoint_is_valid(last_epoch_dir):
        print("[checkpoint found] '{}' already contains a valid checkpoint - "
              "treating training as ALREADY COMPLETED.".format(last_epoch_dir))
        print("                   Skipping training and re-running the summary "
              "comparison across existing epoch_N/ folders.")
        print("                   Delete the models/distilbert_ticket_classifier "
              "folder if you want to force a full retrain.")
        summarize_existing_checkpoints(
            run_label=args.run_label or "existing_checkpoints",
            append_metrics=args.append_metrics)
        print("\n[done] All results printed above.")
        return

    # ---- Load tokenizer + model from HF (network-guarded) ---------------
    print("[download] Loading '{}' tokenizer and model from Hugging Face..."
          .format(MODEL_NAME))
    try:
        tokenizer = DistilBertTokenizerFast.from_pretrained(MODEL_NAME)
        model = DistilBertForSequenceClassification.from_pretrained(
            MODEL_NAME,
            num_labels=len(label_names),
            id2label=id2label,
            label2id=label2id,
        )
    except Exception as e:
        msg = str(e).lower()
        if any(marker in msg for marker in NETWORK_MARKERS):
            print("[ERROR] Network/download problem while fetching the model:")
            print("        {}".format(e))
            print("        Check your internet connection and proxy settings.")
            print("        The model must download once from huggingface.co.")
        else:
            print("[ERROR] Failed to load model/tokenizer: {}".format(e))
        sys.exit(1)

    model.to(DEVICE)

    # ---- Datasets / loaders ---------------------------------------------
    train_ds = TicketDataset(X_train, y_train, tokenizer, MAX_LENGTH)
    val_ds = TicketDataset(X_test, y_test, tokenizer, MAX_LENGTH)
    # Phase 8A.1: the shuffle draws from an explicitly seeded generator rather
    # than the global RNG, so batch order does not depend on what consumed the
    # global stream earlier in the process.
    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True,
                              generator=seeded_generator())
    val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False)

    optimizer = torch.optim.AdamW(
        model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY
    )

    steps_per_epoch = len(train_loader)
    total_steps = steps_per_epoch * NUM_EPOCHS
    # Rough CPU estimate: ~0.4-0.9s per step for distilbert @ bs16/len128.
    est_min = total_steps * 0.4 / 60.0
    est_max = total_steps * 0.9 / 60.0
    # Per-epoch generalization test: reload checkpoint from disk + predict on
    # 14 short tickets. Reload dominates; budget ~5-15s per epoch.
    gen_overhead_min = NUM_EPOCHS * 5.0 / 60.0
    gen_overhead_max = NUM_EPOCHS * 15.0 / 60.0
    print("[plan] epochs={}, batch_size={}, max_length={}, lr={}, wd={}"
          .format(NUM_EPOCHS, BATCH_SIZE, MAX_LENGTH,
                  LEARNING_RATE, WEIGHT_DECAY))
    print("[plan] {} steps/epoch x {} epochs = {} total steps".format(
        steps_per_epoch, NUM_EPOCHS, total_steps))
    print("[plan] per-epoch checkpoint save + 14-ticket generalization test "
          "adds a little CPU time each epoch.")
    print("[estimate] Expected wall-clock: ~{:.0f}-{:.0f} minutes on a "
          "typical laptop CPU".format(
              est_min + gen_overhead_min, est_max + gen_overhead_max))
    print("           (training ~{:.0f}-{:.0f} min + per-epoch generalization "
          "tests ~{:.0f}-{:.0f} min).".format(
              est_min, est_max, gen_overhead_min, gen_overhead_max))
    print("[note] If this is much slower, reduce epochs or batch_size.")

    # ---- Training loop --------------------------------------------------
    # After each epoch: save per-epoch checkpoint, reload it from disk, run the
    # 14-ticket generalization test, print a live summary line, and record it.
    epoch_records = []
    train_start = time.time()
    try:
        for epoch in range(1, NUM_EPOCHS + 1):
            print("\nEpoch {}/{}...".format(epoch, NUM_EPOCHS))
            model.train()
            running_loss = 0.0
            epoch_t0 = time.time()

            for step, batch in enumerate(train_loader, start=1):
                batch = {k: v.to(DEVICE) for k, v in batch.items()}
                optimizer.zero_grad()
                out = model(**batch)
                loss = out.loss
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                running_loss += loss.item()

                if step % 10 == 0 or step == steps_per_epoch:
                    elapsed = time.time() - epoch_t0
                    print("   epoch {} step {}/{}  avg_loss={:.4f}  "
                          "({:.1f}s elapsed)".format(
                              epoch, step, steps_per_epoch,
                              running_loss / step, elapsed))

            train_loss = running_loss / max(1, steps_per_epoch)
            val_loss, val_acc, _, _ = evaluate_loader(model, val_loader)
            print("Epoch {}/{} complete - train_loss={:.4f}, "
                  "val_loss={:.4f}, val_accuracy={:.2f}%".format(
                      epoch, NUM_EPOCHS, train_loss,
                      val_loss, val_acc * 100.0))

            # ---- (1) Save this epoch's checkpoint to its OWN folder ------
            ckpt_dir = epoch_dir(epoch)
            save_checkpoint(ckpt_dir, model, tokenizer, label2id, id2label)

            # ---- (2) Reload that just-saved checkpoint & run the SAME ----
            #          14-ticket generalization test against it.
            ep_model, ep_tok, ep_l2i, ep_i2l = load_checkpoint(ckpt_dir)
            gen_correct, gen_n, gen_pct = run_generalization_test(
                ep_model, ep_tok, ep_i2l, verbose=False)
            # Phase 8A.1: the 45-ticket benchmark, scored from the same
            # reloaded checkpoint by the same scoring loop.
            gen45_correct, gen45_n, gen45_pct = score_benchmark(
                ep_model, ep_tok, ep_i2l, expanded_benchmark, verbose=False)
            del ep_model, ep_tok  # free the reloaded copy

            # ---- (3) Live per-epoch summary line -------------------------
            print("Epoch {}/{}: in-dist val_acc={:.2f}% | "
                  "benchmark14={}/{} ({:.1f}%) | benchmark45={}/{} "
                  "({:.1f}%)".format(
                      epoch, NUM_EPOCHS, val_acc * 100.0,
                      gen_correct, gen_n, gen_pct,
                      gen45_correct, gen45_n, gen45_pct))

            epoch_records.append({
                "epoch": epoch,
                "val_acc": val_acc,
                "gen_correct": gen_correct,
                "gen_n": gen_n,
                "gen_pct": gen_pct,
                "gen45_correct": gen45_correct,
                "gen45_n": gen45_n,
                "gen45_pct": gen45_pct,
            })

    except (MemoryError, RuntimeError) as e:
        msg = str(e).lower()
        if isinstance(e, MemoryError) or "out of memory" in msg or "oom" in msg:
            print("\n[ERROR] Ran out of memory during training.")
            print("        Reduce BATCH_SIZE (e.g. 16 -> 8 -> 4) near the "
                  "top of this script and rerun.")
            print("        You can also lower MAX_LENGTH (e.g. 128 -> 64).")
            sys.exit(1)
        raise  # re-raise anything that isn't a memory problem

    elapsed_total = time.time() - train_start
    print("\n[done] Training finished in {:.1f} minutes "
          "({:.0f} seconds).".format(elapsed_total / 60.0, elapsed_total))

    # ========================================================================
    # Summary table across ALL epochs + best-generalizing-epoch selection.
    # ========================================================================
    best = print_summary_and_pick_best(epoch_records)

    # ========================================================================
    # 7a. IN-DISTRIBUTION results for the BEST epoch's checkpoint
    #     (reload from disk so we report exactly what's saved for inference).
    # ========================================================================
    best_dir = epoch_dir(best["epoch"])
    b_model, b_tok, b_l2i, b_i2l = load_checkpoint(best_dir)
    b_label_names = [b_i2l[i] for i in range(len(b_i2l))]

    print("\n" + "=" * 70)
    print(" IN-DISTRIBUTION TEST (held-out split) - best epoch {}".format(
        best["epoch"]))
    print("=" * 70)
    test_preds = predict_ids(b_model, b_tok, X_test)
    acc = accuracy_score(y_test, test_preds)
    macro_f1 = f1_score(y_test, test_preds, average="macro")
    weighted_f1 = f1_score(y_test, test_preds, average="weighted")

    print("Accuracy:     {:.4f}  ({:.2f}%)".format(acc, acc * 100.0))
    print("Macro F1:     {:.4f}".format(macro_f1))
    print("Weighted F1:  {:.4f}".format(weighted_f1))
    print("\nClassification report:")
    print(classification_report(
        y_test, test_preds,
        labels=list(range(len(b_label_names))),
        target_names=b_label_names,
        zero_division=0,
    ))
    print_confusion_matrix(y_test, test_preds, b_label_names)

    # ========================================================================
    # 7b. GENERALIZATION TEST (verbose per-ticket) on the BEST epoch
    # ========================================================================
    print("\n" + "=" * 70)
    print(" GENERALIZATION TEST (best epoch {} - 14 hand-written novel "
          "tickets)".format(best["epoch"]))
    print("=" * 70)
    correct, n, pct = run_generalization_test(b_model, b_tok, b_i2l, verbose=True)
    print("\nGeneralization score: {}/{} ({:.1f}%)".format(correct, n, pct))

    # Second, independent derivation of the best epoch's 14-ticket count: the
    # loop above recorded it from a checkpoint reloaded mid-training, and this
    # is a fresh scoring pass over the same saved checkpoint. They must agree.
    recorded14 = next(r["gen_correct"] for r in epoch_records
                      if r["epoch"] == best["epoch"])
    if recorded14 != correct:
        print("[ERROR] Best epoch's 14-ticket count disagrees between the "
              "training loop ({}) and this re-scoring pass ({}).".format(
                  recorded14, correct))
        sys.exit(1)

    # ========================================================================
    # 7c. FINAL 3-WAY COMPARISON (against the BEST-generalizing epoch)
    # ========================================================================
    print_three_way_comparison(correct, n, pct, best["epoch"])

    # ---- Phase 8A.1: write the per-epoch metrics ------------------------
    for rec in epoch_records:
        rec["is_best"] = (rec["epoch"] == best["epoch"])
    metrics_path = write_metrics_csv(
        args.run_label or "fresh_retrain", epoch_records,
        append=args.append_metrics)
    print("\n[write] Per-epoch metrics -> {}".format(metrics_path))
    print("[check] Best epoch's 14-ticket count confirmed by an independent "
          "re-scoring pass.")

    print("\n[done] All results printed above.")


if __name__ == "__main__":
    main()
