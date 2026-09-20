# src/experiments/run_zeroshot_baselines.py
"""
Phase 5C -- zero-shot LLM classification baselines.

WHY THIS EXISTS. Phase 5B reduced the cascade to a latency optimisation, so the
project's accuracy claim now rests entirely on Tier-2 alone (BGE + LogReg,
33/45 on the benchmark). That number has never been compared against the
obvious alternative a reviewer will raise: just ask an LLM. This script
produces that comparison on the two fixed, read-only benchmarks.

TWO BACKENDS, ONE PROMPT. Gemini and a local Ollama model are driven through
one interface, one prompt builder and one parser. A script per vendor is how
two baselines quietly stop being comparable -- which is the same failure Phase
5B found in the ablation, where the published "cascade gain" was really a
comparison against a different model.

WHAT THIS IS NOT. A zero-shot LLM against a classifier trained on 3,200 rows is
NOT like-for-like. The honest claim is "zero-shot LLM vs trained classifier",
and the interesting quantity is what the training data is worth. Few-shot
prompting is deliberately out of scope.

QUOTA. Gemini free tier is 15 req/min, 500/day. Every raw response is cached to
disk BEFORE it is parsed, keyed by prompt hash, so a crash or a re-score never
re-spends quota. Always dry-run with --limit 3 first. Ollama is local and free.

Usage (from the project root):
    # 3 calls, writes *.dryrun.csv, never touches a real result file
    python src/experiments/run_zeroshot_baselines.py --backend gemini --limit 3

    # local model, no quota at all
    python src/experiments/run_zeroshot_baselines.py --backend ollama --set both

    # the real Gemini run: 59 calls
    python src/experiments/run_zeroshot_baselines.py --backend gemini --set both
"""

import os
import sys
import csv
import json
import time
import errno
import ctypes
import hashlib
import argparse
import traceback
import urllib.error
import urllib.request

THIS_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(THIS_DIR, os.pardir, os.pardir))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

DATA_DIR = os.path.join(PROJECT_ROOT, "data")
RAW_CACHE_DIR = os.path.join(DATA_DIR, "zeroshot_raw")

BENCHMARK45_PATH = os.path.join(DATA_DIR, "novel_tickets_expanded.json")

# The seven routing categories, in a FIXED order so the prompt is byte-stable
# across runs. This list is cross-checked at run time against the production
# classifier's own classes_ and against the benchmarks' labels -- see
# validate_categories(). It is a guard, not a second source of truth.
CATEGORIES = [
    "Infrastructure", "Application", "Security", "Database",
    "Storage", "Network", "Access Management",
]

EVAL_SETS = ("benchmark45", "benchmark14")
EXPECTED_SET_SIZES = {"benchmark45": 45, "benchmark14": 14}

# Gemini pacing and safety. The delay floor is a project invariant (CLAUDE.md:
# keep it at or above 4s; this project's memory of it says 4.5s).
GEMINI_CALL_DELAY_SEC = 4.5
MAX_RETRIES = 3
BACKOFF_SCHEDULE = [5, 15, 30]
# Hard ceiling so a prompt bug cannot burn the daily quota. 59 tickets + slack.
MAX_TOTAL_GEMINI_CALLS = 80

OLLAMA_URL = "http://localhost:11434"
DEFAULT_OLLAMA_MODEL = "qwen2.5:7b-instruct"
# A category name is a handful of tokens. Capping this stops a rambling model
# from turning a 10-minute run into an hour.
OLLAMA_NUM_PREDICT = 48
OLLAMA_TIMEOUT_SEC = 300

# Minimum free physical RAM (MB) before a local model is allowed to run. A
# model that swaps produces a latency number that is meaningless AND plausible,
# which is exactly this project's recurring bug shape -- so the run aborts
# rather than silently measuring the page file. Floors are the resident weights
# plus ~0.7 GB of working margin.
MODEL_RAM_FLOOR_MB = {
    "qwen2.5:3b-instruct": 3277,   # ~2.5 GB weights + margin
    "qwen2.5:7b-instruct": 6656,   # ~5.9 GB weights + margin
}
DEFAULT_RAM_FLOOR_MB = 3277


def _banner(text):
    rule = "=" * 70
    print(rule)
    print(text)
    print(rule)


def _fatal(message):
    print("")
    print("ERROR: " + str(message))
    sys.exit(1)


def _slug(text):
    keep = []
    for ch in str(text).lower():
        if ch.isalnum():
            keep.append(ch)
        elif keep and keep[-1] != "-":
            keep.append("-")
    return "".join(keep).strip("-") or "unknown"


# --------------------------------------------------------------------------
# Memory preflight -- refuse to measure a swapping model
# --------------------------------------------------------------------------
class _MEMORYSTATUSEX(ctypes.Structure):
    _fields_ = [
        ("dwLength", ctypes.c_ulong),
        ("dwMemoryLoad", ctypes.c_ulong),
        ("ullTotalPhys", ctypes.c_ulonglong),
        ("ullAvailPhys", ctypes.c_ulonglong),
        ("ullTotalPageFile", ctypes.c_ulonglong),
        ("ullAvailPageFile", ctypes.c_ulonglong),
        ("ullTotalVirtual", ctypes.c_ulonglong),
        ("ullAvailVirtual", ctypes.c_ulonglong),
        ("ullAvailExtendedVirtual", ctypes.c_ulonglong),
    ]


def available_ram_mb():
    """Free physical RAM in MB, or None if it cannot be read.

    ctypes + GlobalMemoryStatusEx is stdlib and exact, so this adds no
    dependency -- psutil is deliberately not in requirements.txt.
    """
    try:
        status = _MEMORYSTATUSEX()
        status.dwLength = ctypes.sizeof(_MEMORYSTATUSEX)
        ok = ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(status))
        if not ok:
            return None
        return int(status.ullAvailPhys // (1024 * 1024))
    except Exception:  # noqa: BLE001 -- non-Windows, or the call is unavailable
        return None


def require_free_ram(model, allow_low_ram=False):
    """Abort unless there is headroom for `model`. Returns the reading in MB.

    Never silently proceeds: an unreadable value is itself fatal, because
    "could not check" and "checked and it was fine" must not look the same.
    """
    floor = MODEL_RAM_FLOOR_MB.get(model, DEFAULT_RAM_FLOOR_MB)
    avail = available_ram_mb()

    if avail is None:
        if allow_low_ram:
            print("  WARNING: free RAM could not be read; --allow-low-ram "
                  "given, proceeding anyway.")
            return None
        _fatal(
            "Could not read free physical memory, so the {m} run cannot be "
            "shown to fit in RAM.\n"
            "  A swapping model produces a meaningless latency number and a "
            "plausible one.\n"
            "  Pass --allow-low-ram to run anyway (the result is then marked "
            "as unfit for timing).".format(m=model)
        )

    print("  free RAM : {a} MB (floor for {m} is {f} MB)".format(
        a=avail, m=model, f=floor))

    if avail < floor:
        if allow_low_ram:
            print("  WARNING: {a} MB is below the {f} MB floor. "
                  "--allow-low-ram given, proceeding -- LATENCY FROM THIS RUN "
                  "IS NOT TRUSTWORTHY.".format(a=avail, f=floor))
            return avail
        _fatal(
            "Not enough free RAM for {m}: {a} MB available, {f} MB needed.\n"
            "  Refusing to start rather than swap -- a swapping model gives a "
            "meaningless latency number that still looks plausible.\n"
            "  Close some applications and re-run, or pass --allow-low-ram to "
            "accept an untrustworthy timing.".format(
                m=model, a=avail, f=floor)
        )
    return avail


# --------------------------------------------------------------------------
# The benchmarks (READ-ONLY -- loaded, never written)
# --------------------------------------------------------------------------
def load_eval_set(name):
    """Return [{id, text, expected}] for one fixed benchmark."""
    if name == "benchmark45":
        if not os.path.isfile(BENCHMARK45_PATH):
            _fatal("Missing the 45-ticket benchmark:\n  " + BENCHMARK45_PATH)
        try:
            with open(BENCHMARK45_PATH, "r", encoding="utf-8") as fh:
                data = json.load(fh)
        except Exception as exc:
            _fatal("Failed to read novel_tickets_expanded.json: " + repr(exc))
        records = [
            {"id": "b45_%02d" % i, "text": r["text"], "expected": r["expected"]}
            for i, r in enumerate(data)
        ]
    elif name == "benchmark14":
        try:
            from src.classification.generalization_test import NOVEL_TICKETS
        except Exception as exc:
            _fatal(
                "Failed to import NOVEL_TICKETS from "
                "src/classification/generalization_test.py: " + repr(exc)
            )
        records = [
            {"id": "b14_%02d" % i, "text": r["text"], "expected": r["expected"]}
            for i, r in enumerate(NOVEL_TICKETS)
        ]
    else:
        _fatal("Unknown evaluation set: " + repr(name))

    expected_n = EXPECTED_SET_SIZES[name]
    if len(records) != expected_n:
        _fatal(
            "{n} must contain EXACTLY {e} tickets; found {f}. These benchmarks "
            "are read-only reference points and must not have changed.".format(
                n=name, e=expected_n, f=len(records)
            )
        )
    for r in records:
        if not isinstance(r["text"], str) or not r["text"].strip():
            _fatal("{n}: ticket {i} has empty text.".format(n=name, i=r["id"]))
    return records


def validate_categories(records):
    """Cross-check CATEGORIES two independent ways before spending anything.

    Rule 6: a label set that is wrong for its context but internally consistent
    would produce a plausible accuracy for a prompt that never offered the
    right answer. So the prompt's category list is checked against BOTH the
    production classifier's own classes_ AND the benchmarks' labels.
    """
    try:
        from src.agent.artifacts import load_tier1
        _, tier1_clf = load_tier1()
        model_classes = set(str(c) for c in tier1_clf.classes_)
    except Exception as exc:
        _fatal(
            "Could not load the persisted Tier-1 to verify the category list: "
            "{e}\nRun: python src/classification/train_tier1.py".format(
                e=repr(exc)
            )
        )

    declared = set(CATEGORIES)
    if declared != model_classes:
        _fatal(
            "The prompt's category list does not match the production "
            "classifier's classes.\n"
            "  only in prompt : {a}\n"
            "  only in model  : {b}\n"
            "Refusing to ask an LLM to choose from a different label set than "
            "the classifier it is being compared against.".format(
                a=sorted(declared - model_classes),
                b=sorted(model_classes - declared),
            )
        )

    labels = set(r["expected"] for r in records)
    unknown = labels - declared
    if unknown:
        _fatal(
            "Benchmark labels not present in the category list: {u}".format(
                u=sorted(unknown)
            )
        )
    print("categories verified: {n}, matching the Tier-1 classes and every "
          "benchmark label".format(n=len(CATEGORIES)))


# --------------------------------------------------------------------------
# The one prompt, and the one parser
# --------------------------------------------------------------------------
def build_prompt(text):
    """The zero-shot classification prompt. Identical for every backend.

    Deliberately close to the verification prompt already used against Gemini
    in generate_deployment_calibration_set.py, so 5C is not introducing a new
    prompt style alongside the project's existing one.
    """
    cats = "\n".join("  - " + c for c in CATEGORIES)
    return (
        "Classify this IT support ticket into exactly one category.\n"
        "\n"
        "TICKET:\n"
        "{text}\n"
        "\n"
        "CATEGORIES:\n"
        "{cats}\n"
        "\n"
        "Answer with STRICT JSON, exactly:\n"
        '{{"category": "<one of the categories above, copied exactly>"}}\n'
    ).format(text=text, cats=cats)


_NORMALISED = {c.strip().casefold(): c for c in CATEGORIES}


def parse_response(raw):
    """(predicted, parsed_ok). NEVER coerces a near-miss into a category.

    Two accepted shapes: the requested JSON object, or a bare category name.
    A bare name is a different FORMAT, not a different answer, so accepting it
    is not coercion. Anything else is recorded as unparseable and kept
    verbatim -- an unparseable rate is itself a result, and quietly mapping
    "Networking" onto "Network" would inflate the baseline with answers the
    model did not give.
    """
    if raw is None:
        return None, False

    text = str(raw).strip()
    if not text:
        return None, False

    # Strip a ```json fence if the model added one.
    if text.startswith("```"):
        lines = [ln for ln in text.splitlines() if not ln.strip().startswith("```")]
        text = "\n".join(lines).strip()

    candidate = None
    try:
        obj = json.loads(text)
        if isinstance(obj, dict) and "category" in obj:
            candidate = obj["category"]
    except (ValueError, TypeError):
        candidate = None

    if candidate is None:
        candidate = text

    if not isinstance(candidate, str):
        return None, False

    hit = _NORMALISED.get(candidate.strip().casefold())
    if hit is None:
        return None, False
    return hit, True


# --------------------------------------------------------------------------
# Backends -- one interface: .name, .model, .generate(prompt) -> raw text
# --------------------------------------------------------------------------
class GeminiBackend:
    name = "gemini"
    spends_quota = True
    # Part of the shared backend interface; hosted inference has no local
    # weights to identify and no local memory to run short of.
    model_digest = None
    allow_low_ram = False
    free_ram_mb = None

    def __init__(self):
        from src.agent.artifacts import build_gemini_client
        from src.agent.config import settings

        self._settings = settings
        self.model = settings.models.gemini_model
        try:
            self._client = build_gemini_client()
        except Exception as exc:
            _fatal(
                "Could not build the Gemini client: {e}\n"
                "GEMINI_API_KEY must be set in .env at the project "
                "root.".format(e=repr(exc))
            )
        self.calls = 0

    @staticmethod
    def _is_rate_limit(exc):
        blob = repr(exc).lower()
        return ("429" in blob or "rate" in blob or "quota" in blob
                or "resource_exhausted" in blob)

    def generate(self, prompt):
        if self.calls >= MAX_TOTAL_GEMINI_CALLS:
            _fatal(
                "Gemini call budget exhausted ({n}). This is a guard against a "
                "prompt bug burning the daily quota; every response so far is "
                "cached, so a re-run resumes without re-spending.".format(
                    n=MAX_TOTAL_GEMINI_CALLS
                )
            )

        for attempt in range(MAX_RETRIES + 1):
            try:
                self.calls += 1
                response = self._client.models.generate_content(
                    model=self.model,
                    contents=prompt,
                    config={
                        "temperature": 0.0,
                        "response_mime_type": "application/json",
                    },
                )
            except Exception as exc:  # noqa: BLE001
                if self._is_rate_limit(exc) and attempt < MAX_RETRIES:
                    wait = BACKOFF_SCHEDULE[min(attempt,
                                                len(BACKOFF_SCHEDULE) - 1)]
                    print("      rate limit (attempt {a}); waiting "
                          "{w}s".format(a=attempt + 1, w=wait))
                    time.sleep(wait)
                    continue
                raise RuntimeError("Gemini call failed: {e}".format(e=exc))
            return getattr(response, "text", None)
        return None

    def pace(self):
        time.sleep(GEMINI_CALL_DELAY_SEC)


class OllamaBackend:
    name = "ollama"
    spends_quota = False

    def __init__(self, model, allow_low_ram=False):
        self.model = model
        self.calls = 0
        self.model_digest = None
        self.allow_low_ram = bool(allow_low_ram)
        self._check_server()
        self.free_ram_mb = require_free_ram(self.model, self.allow_low_ram)

    def _check_server(self):
        try:
            with urllib.request.urlopen(OLLAMA_URL + "/api/tags", timeout=10) as r:
                tags = json.loads(r.read().decode("utf-8"))
        except urllib.error.URLError as exc:
            _fatal(
                "Could not reach Ollama at {u} ({e}).\n"
                "  Install it:  winget install Ollama.Ollama\n"
                "  Then start it (it runs as a background service once "
                "installed) and pull the model:\n"
                "      ollama pull {m}".format(u=OLLAMA_URL, e=exc, m=self.model)
            )
        except Exception as exc:  # noqa: BLE001
            _fatal("Unexpected error talking to Ollama: " + repr(exc))

        entries = tags.get("models", []) or []
        available = [m.get("name", "") for m in entries]
        # EXACT match, plus Ollama's ":latest" convention. A prefix match was
        # tried first and is wrong: asking for qwen2.5:7b-instruct with only
        # qwen2.5:3b-instruct pulled passed a guard that claims to have
        # verified the model is installed. That is this project's recurring
        # shape -- a check that is internally consistent and wrong for its
        # context -- so the guard must be exact.
        wanted = (self.model, self.model + ":latest")
        match = next((m for m in entries if m.get("name", "") in wanted), None)
        if match is None:
            _fatal(
                "Ollama is running but has no model named exactly {m!r}.\n"
                "  Installed: {a}\n"
                "  Pull it with:  ollama pull {m}".format(
                    m=self.model, a=available or "(none)"
                )
            )
        # Record WHICH weights answered, not just the tag that was asked for.
        self.model_digest = match.get("digest")

    def generate(self, prompt):
        payload = json.dumps({
            "model": self.model,
            "prompt": prompt,
            "stream": False,
            "format": "json",
            "options": {
                "temperature": 0.0,
                "seed": 42,
                "num_predict": OLLAMA_NUM_PREDICT,
            },
        }).encode("utf-8")
        req = urllib.request.Request(
            OLLAMA_URL + "/api/generate",
            data=payload,
            headers={"Content-Type": "application/json"},
        )
        self.calls += 1
        try:
            with urllib.request.urlopen(req, timeout=OLLAMA_TIMEOUT_SEC) as r:
                body = json.loads(r.read().decode("utf-8"))
        except Exception as exc:  # noqa: BLE001
            raise RuntimeError("Ollama call failed: {e}".format(e=exc))
        return body.get("response")

    def pace(self):
        return  # local, free, no pacing needed


# --------------------------------------------------------------------------
# Raw-response cache -- written BEFORE parsing, so quota is never re-spent
# --------------------------------------------------------------------------
def cache_path(backend_name, model, eval_set, ticket_id):
    return os.path.join(
        RAW_CACHE_DIR,
        "{b}__{m}__{s}__{t}.json".format(
            b=backend_name, m=_slug(model), s=eval_set, t=ticket_id
        ),
    )


def prompt_hash(prompt):
    return hashlib.sha256(prompt.encode("utf-8")).hexdigest()


def read_cache(path, expected_hash):
    """Return the cached record, or None if absent or built from another prompt."""
    if not os.path.isfile(path):
        return None
    try:
        with open(path, "r", encoding="utf-8") as fh:
            rec = json.load(fh)
    except Exception:
        return None
    if rec.get("prompt_sha256") != expected_hash:
        # The prompt changed, so the cached answer is for a different question.
        return None
    return rec


def write_cache(path, record):
    try:
        os.makedirs(RAW_CACHE_DIR, exist_ok=True)
        with open(path, "w", encoding="utf-8") as fh:
            json.dump(record, fh, ensure_ascii=False, indent=1)
    except OSError as exc:
        if exc.errno != errno.EEXIST:
            _fatal("Failed to write the raw cache at {p}: {e}".format(
                p=path, e=repr(exc)))


# --------------------------------------------------------------------------
# The run
# --------------------------------------------------------------------------
def run_one_set(backend, eval_set, limit, dry_run):
    records = load_eval_set(eval_set)
    validate_categories(records)

    if limit is not None:
        records = records[:limit]

    print("")
    _banner("{b} / {m}  ->  {s}  ({n} ticket(s))".format(
        b=backend.name, m=backend.model, s=eval_set, n=len(records)))

    rows = []
    n_live_calls = 0
    n_cached = 0
    for i, rec in enumerate(records):
        prompt = build_prompt(rec["text"])
        phash = prompt_hash(prompt)
        path = cache_path(backend.name, backend.model, eval_set, rec["id"])

        cached = read_cache(path, phash)
        if cached is not None:
            raw = cached.get("raw")
            elapsed = float(cached.get("elapsed_s", 0.0))
            n_cached += 1
            origin = "cache"
        else:
            if n_live_calls > 0:
                backend.pace()
            t0 = time.perf_counter()
            raw = backend.generate(prompt)
            elapsed = time.perf_counter() - t0
            n_live_calls += 1
            origin = "live"
            write_cache(path, {
                "backend": backend.name,
                "model": backend.model,
                "model_digest": backend.model_digest,
                "eval_set": eval_set,
                "ticket_id": rec["id"],
                "prompt_sha256": phash,
                "raw": raw,
                "elapsed_s": round(elapsed, 4),
                # Stamped so a swapped run can never later be mistaken for a
                # clean latency measurement.
                "low_ram_override": bool(backend.allow_low_ram),
                "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
            })

        predicted, parsed_ok = parse_response(raw)
        correct = bool(parsed_ok and predicted == rec["expected"])

        print("  [{i:>3}/{n}] {t}  {o:<5}  {e:>7.2f}s  -> {p}{flag}".format(
            i=i + 1, n=len(records), t=rec["id"], o=origin, e=elapsed,
            p=predicted if parsed_ok else "UNPARSEABLE",
            flag="" if correct else "   (expected %s)" % rec["expected"],
        ))

        rows.append({
            "index": i,
            "ticket_id": rec["id"],
            "text": rec["text"],
            "expected": rec["expected"],
            "predicted": predicted if parsed_ok else "",
            "parsed_ok": parsed_ok,
            "correct": correct,
            "elapsed_s": round(elapsed, 4),
            "source": origin,
            "raw_response": "" if raw is None else str(raw).replace("\n", " "),
        })

    n_correct = sum(1 for r in rows if r["correct"])
    n_unparseable = sum(1 for r in rows if not r["parsed_ok"])
    return rows, {
        "backend": backend.name,
        "model": backend.model,
        "eval_set": eval_set,
        "n": len(rows),
        "n_correct": n_correct,
        "n_unparseable": n_unparseable,
        "live_calls": n_live_calls,
        "cached": n_cached,
    }


def write_results(rows, backend, eval_set, dry_run, force):
    suffix = ".dryrun" if dry_run else ""
    out_path = os.path.join(
        DATA_DIR,
        "zeroshot_{b}_{m}_{s}{x}.csv".format(
            b=backend.name, m=_slug(backend.model), s=eval_set, x=suffix
        ),
    )
    if os.path.isfile(out_path) and not force and not dry_run:
        _fatal(
            "Refusing to overwrite an existing result:\n  {p}\n"
            "Pass --force if you really mean to replace it. (The raw cache "
            "means a re-run costs no quota.)".format(p=out_path)
        )

    fieldnames = ["index", "ticket_id", "text", "expected", "predicted",
                  "parsed_ok", "correct", "elapsed_s", "source",
                  "raw_response"]
    try:
        os.makedirs(DATA_DIR, exist_ok=True)
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
            "Zero-shot LLM classification baselines on the fixed benchmarks. "
            "One prompt, one parser, two backends."
        )
    )
    parser.add_argument("--backend", required=True,
                        choices=("gemini", "ollama"))
    parser.add_argument("--set", dest="eval_set", default="benchmark45",
                        choices=EVAL_SETS + ("both",))
    parser.add_argument("--model", default=None,
                        help="Ollama model tag (default {d}). Ignored for "
                             "gemini, which uses "
                             "settings.models.gemini_model.".format(
                                 d=DEFAULT_OLLAMA_MODEL))
    parser.add_argument("--limit", type=int, default=None,
                        help="Only the first N tickets. Implies a dry run: "
                             "results go to *.dryrun.csv.")
    parser.add_argument("--force", action="store_true",
                        help="Overwrite an existing result CSV.")
    parser.add_argument("--allow-low-ram", action="store_true",
                        help="Run a local model even without the RAM headroom "
                             "to avoid swapping. The timing from such a run is "
                             "NOT trustworthy and every cached response is "
                             "stamped low_ram_override=true.")
    return parser.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    dry_run = args.limit is not None

    _banner("Phase 5C -- zero-shot LLM baselines")
    print("backend : " + args.backend)
    print("set     : " + args.eval_set)
    if dry_run:
        print("MODE    : DRY RUN (--limit {n}); results go to "
              "*.dryrun.csv".format(n=args.limit))

    if args.backend == "gemini":
        backend = GeminiBackend()
        sets = EVAL_SETS if args.eval_set == "both" else (args.eval_set,)
        planned = sum(EXPECTED_SET_SIZES[s] for s in sets)
        if dry_run:
            planned = min(planned, args.limit * len(sets))
        print("model   : " + backend.model)
        print("QUOTA   : up to {n} live call(s), {d}s apart, every raw "
              "response cached before parsing".format(
                  n=planned, d=GEMINI_CALL_DELAY_SEC))
    else:
        backend = OllamaBackend(args.model or DEFAULT_OLLAMA_MODEL,
                                allow_low_ram=args.allow_low_ram)
        sets = EVAL_SETS if args.eval_set == "both" else (args.eval_set,)
        print("model   : " + backend.model + "   (local, no quota)")
        print("digest  : " + str(backend.model_digest))

    summaries = []
    for eval_set in sets:
        rows, summary = run_one_set(backend, eval_set, args.limit, dry_run)
        out_path = write_results(rows, backend, eval_set, dry_run, args.force)
        summary["csv"] = out_path
        summaries.append(summary)

    print("")
    _banner("SUMMARY")
    for s in summaries:
        acc = (s["n_correct"] / s["n"]) if s["n"] else 0.0
        print("{b} / {m}  {s:<12} {c}/{n} = {a:.2%}   unparseable {u}   "
              "(live {lc}, cached {ca})".format(
                  b=s["backend"], m=s["model"], s=s["eval_set"],
                  c=s["n_correct"], n=s["n"], a=acc, u=s["n_unparseable"],
                  lc=s["live_calls"], ca=s["cached"]))
        print("   CSV: " + s["csv"])

    total_live = sum(s["live_calls"] for s in summaries)
    if backend.spends_quota:
        # Rule 6: reconcile the spend two independent ways -- the backend's own
        # counter against the number of live calls the loop made.
        print("")
        print("Gemini live calls this run: {a} (backend counter {b})".format(
            a=total_live, b=backend.calls))
        if backend.calls < total_live:
            _fatal(
                "Call accounting disagrees: the loop made {a} live calls but "
                "the backend counted {b}. Do not trust this run's quota "
                "figures.".format(a=total_live, b=backend.calls)
            )

    if dry_run:
        print("")
        print("DRY RUN complete. Inspect the raw_response column before "
              "spending the full budget.")


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
