"""
Generate a DEPLOYMENT-DISTRIBUTION calibration set for conformal prediction.

WHY THIS EXISTS
---------------
Conformal prediction guarantees coverage only under exchangeability between
calibration and deployment data. The existing 175-ticket calibration set
(data/calibration_tickets_paraphrased.json) violates that badly:

  * It is Gemini paraphrases of rows the models were TRAINED on.
  * Measured consequence (README "Conformal Prediction"): calibrating on it
    gives nominal 90% coverage but 66.7% on the plain-English benchmark for
    Tier-1 -- a 23.3-point shortfall against a 4.5-point noise band.
  * It cannot be repaired by filtering. Memorisation is at TEMPLATE level:
    12 templates, ~430 rows each, and the set touches 11 of them. Removing
    source rows changes nothing; removing whole templates would leave 40 of
    4,000 rows.

The fix is not cleaning the old set -- it is calibrating on data drawn like
deployment traffic. This script generates that set.

DESIGN DECISIONS
----------------
n = 175 (25 per category), matching the existing in-domain set EXACTLY in
size and per-class balance. That is deliberate: holding n and class balance
fixed means any coverage difference between the two sets is attributable to
the DISTRIBUTION alone, not to sample size or the quantile's granularity.
(It is also the affordable choice -- each ticket costs two Gemini calls, so
300 tickets would be 600+ calls against a 500/day free-tier cap.)

DISJOINTNESS FROM THE BENCHMARK IS ENFORCED, NOT HOPED FOR. The 45-ticket
benchmark is the evaluation set for the very coverage number this calibration
set exists to improve. If the two overlap, that measurement is meaningless.
Two independent defences:
  1. Every benchmark text (and every NOVEL_TICKETS text) goes into the
     prompt's avoid-list.
  2. Every accepted candidate is embedded with the production model and
     rejected if its cosine similarity to ANY benchmark ticket exceeds
     MAX_BENCHMARK_SIMILARITY. The maximum observed similarity is reported.

SCOPE ANCHORS FOR ALL SEVEN CATEGORIES. The expanded-benchmark build hit a
labelling bug where Gemini wrote physical-facilities tickets for
"Infrastructure" because the prompt never said what the word means in THIS
project. That fix previously existed only for Infrastructure; here it is
generalised. The Security/Access-Management and Storage/Access-Management
boundaries get explicit negative constraints too, because those are precisely
where the old calibration set's self-consistency failures clustered.

BENCHMARK POLICY ON DISAGREEMENT, NOT CALIBRATION POLICY. The old set kept
records where Gemini's own guess disagreed with the assigned label (21/175,
"flagged"). Label noise breaks conformal coverage directly, so here a
disagreement means the candidate is DISCARDED and regenerated.

USAGE
-----
Always dry-run first -- it costs ~14 calls and surfaces a bad prompt before
the full budget is spent:

    python -m src.classification.generate_deployment_calibration_set \
        --per-category 1

Then the real run (~350-400 calls, ~30 min at 4.5s spacing):

    python -m src.classification.generate_deployment_calibration_set

Resumable: progress is checkpointed after every accepted ticket, so a quota
failure can be continued by re-running the same command.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time

import numpy as np

_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(_THIS_DIR, os.pardir, os.pardir))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from src.agent.config import require_gemini_api_key, settings   # noqa: E402
from src.agent.logging_setup import ensure_utf8_console          # noqa: E402

ensure_utf8_console()

DATA_DIR = os.path.join(PROJECT_ROOT, "data")
BENCHMARK_JSON = os.path.join(DATA_DIR, "novel_tickets_expanded.json")
OUTPUT_JSON = os.path.join(DATA_DIR, "deployment_calibration_tickets.json")
REJECTED_JSON = os.path.join(DATA_DIR,
                             "deployment_calibration_rejected.json")
CHECKPOINT_JSON = os.path.join(DATA_DIR,
                               ".deployment_calibration_checkpoint.json")

CATEGORIES = [
    "Infrastructure", "Application", "Security", "Database",
    "Storage", "Network", "Access Management",
]
PER_CATEGORY = 25
STYLE_ANCHORS = 3

GEMINI_CALL_DELAY_SEC = 4.5
MAX_RETRIES = 3
BACKOFF_SCHEDULE = [5, 15, 30]

# Cosine similarity above which a candidate is considered too close to a
# benchmark ticket to be safe as calibration data.
MAX_BENCHMARK_SIMILARITY = 0.90

# Cosine similarity above which a candidate is too close to a ticket ALREADY
# ACCEPTED for the same category.
#
# Near-duplicate calibration points are not exchangeable draws: they inflate n
# without adding information and skew the empirical quantile toward whatever
# score region they cluster in. This is the same defect already documented for
# the OOD set (15 seeds x 3 near-identical variants), and it is worse here,
# because a calibration set determines the quantile rather than merely being
# scored against it.
#
# 0.95 is not arbitrary. Measured over the first full run, within-category
# similarity has p50=0.729, p95=0.892, p99=0.949 -- legitimate same-category
# tickets routinely reach 0.89, so 0.95 cuts the anomalous top ~1% without
# suppressing real diversity.
MAX_INTERNAL_SIMILARITY = 0.95

# Hard stop so a prompt bug cannot silently burn the daily quota.
MAX_TOTAL_CALLS = 520


def _banner(text):
    bar = "=" * 74
    print("\n" + bar)
    print(text)
    print(bar)


def _fatal(message):
    print("\n" + "!" * 74)
    print("FATAL: " + message)
    print("!" * 74)
    sys.exit(1)


# --------------------------------------------------------------------------- #
# Scope anchors -- what each category MEANS in this project
# --------------------------------------------------------------------------- #
SCOPE_ANCHORS = {
    "Infrastructure": """\
"Infrastructure" here means COMPUTE / SERVER infrastructure only: the servers,
virtual machines, containers, Kubernetes nodes and load balancers that run the
company's systems, plus the compute resources they depend on -- CPU, memory,
disk throughput, system-clock sync, scheduled/automated jobs, and automatic
scaling of capacity. In-scope problems sound like: an internal system down or
crawling for everyone; a server that will not come back after a restart;
things crashing because they ran out of memory; part of the platform
unreachable though the machines look "on"; overnight scheduled jobs not
running; timestamps out of sync; capacity not scaling up under load.
DO NOT write about physical office issues (lighting, doors, elevators,
plumbing, heating/cooling) -- out of scope. DO NOT write about one person's
laptop, monitor or printer -- this is the shared platform, not one desk.

CRITICAL: the ticket MUST contain a clear cue that the problem is with the
SHARED MACHINES THEMSELVES -- running out of memory, a server not coming
back up, the whole platform unreachable, capacity not growing under load,
machines overloaded or grinding. A ticket that only says "the overnight job
did not run" or "the scheduled task failed", with no machine-level cue,
reads as an Application or Database problem to any reader and will be
rejected. Name the machine-level symptom, not just the downstream effect.""",

    "Application": """\
"Application" here means a specific named business APPLICATION misbehaving --
the software itself, not the machine under it. In-scope problems sound like: a
particular tool crashing when you upload a file; a dashboard extremely slow to
load; a page throwing errors right after an update was released; its
connection to another product (single sign-on, video calls, payroll, a
payments provider) failing; automatic report emails from that tool not going
out.
DO NOT write about the server being down (that is Infrastructure), the data
itself being missing (Database), or not being allowed in (Access Management).
Name or clearly imply ONE business application.""",

    "Security": """\
"Security" here means a security THREAT or INCIDENT -- something suspicious or
hostile, or a control that has lapsed. In-scope problems sound like: repeated
failed login attempts hammering an application from outside; a suspicious
email or attachment; a machine behaving as though it is infected; a security
certificate expiring or warning users; a scan reporting a vulnerability; data
possibly going somewhere it should not.
DO NOT write a routine request to be GIVEN access, a password reset, or a
locked-out account -- those are Access Management, not Security. The
distinction is threat-or-lapse versus ordinary permissions administration.""",

    "Database": """\
"Database" here means the DATA LAYER -- the databases behind the applications.
In-scope problems sound like: an application reporting it cannot find records
that should exist; queries or reports that have become very slow; a database
running out of connections; copies of a database drifting out of sync;
routine database maintenance falling behind; a restore or backup of data.
DO NOT write about shared drives or file shares (that is Storage), the
application's own screens misbehaving (Application), or the server hardware
(Infrastructure). This is about stored records and the database itself.""",

    "Storage": """\
"Storage" here means SHARED FILE STORAGE -- network drives, file shares and
the space on them. In-scope problems sound like: a shared team drive that will
not open or has disappeared; a new machine that cannot connect to the team
drive; running out of space on a shared area; a request for more space; files
on a share being unexpectedly read-only; a shared folder being very slow to
browse.
DO NOT write about being denied PERMISSION to a folder -- if the complaint is
"I am not allowed", that is Access Management. Storage is about the drive
itself: missing, full, unmountable or slow. Also not databases (Database).""",

    "Network": """\
"Network" here means CONNECTIVITY between people and systems. In-scope
problems sound like: an office or site with connectivity dropping in and out;
everything feeling slow to reach a particular location; the VPN not
connecting from home; a firewall blocking a port or a specific service; web
addresses not resolving; wireless dropping in one building.
DO NOT write about an application being broken once you have reached it
(Application), or a server being down (Infrastructure). Network is about
getting there, not what happens when you arrive.""",

    "Access Management": """\
"Access Management" here means IDENTITY AND PERMISSIONS administration --
ordinary, legitimate requests and problems about who can get into what.
In-scope problems sound like: being locked out after too many failed
attempts; needing a password reset; asking to be granted access to a system
or folder; needing elevated or temporary admin rights; being added to or
removed from a group; a new joiner needing accounts set up or a leaver
needing them removed; multi-factor authentication not letting someone in.
DO NOT write about an ATTACK or suspicious activity -- that is Security. This
category is routine permissions work requested by a legitimate user.""",
}


# --------------------------------------------------------------------------- #
# Loading
# --------------------------------------------------------------------------- #
def load_benchmark():
    if not os.path.isfile(BENCHMARK_JSON):
        _fatal(f"45-ticket benchmark not found:\n    {BENCHMARK_JSON}")
    with open(BENCHMARK_JSON, "r", encoding="utf-8") as fh:
        return json.load(fh)


def load_style_anchors():
    """Few-shot style examples grouped by category, from NOVEL_TICKETS.

    NOVEL_TICKETS is the project's fixed 14-ticket hand-written benchmark and
    is read-only. It is used here purely as a REGISTER example -- the plain,
    non-technical voice real users write in.
    """
    from src.classification.generalization_test import NOVEL_TICKETS

    anchors = {c: [] for c in CATEGORIES}
    for t in NOVEL_TICKETS:
        cat = t.get("expected")
        if cat in anchors and isinstance(t.get("text"), str):
            anchors[cat].append(t["text"].strip())

    empty = [c for c in CATEGORIES if not anchors[c]]
    if empty:
        _fatal(f"No style anchors found for: {empty}")
    return anchors


def load_checkpoint():
    if not os.path.isfile(CHECKPOINT_JSON):
        return {"accepted": [], "rejected": [], "calls": 0}
    with open(CHECKPOINT_JSON, "r", encoding="utf-8") as fh:
        return json.load(fh)


def save_checkpoint(state):
    with open(CHECKPOINT_JSON, "w", encoding="utf-8") as fh:
        json.dump(state, fh, indent=2)
        fh.write("\n")


# --------------------------------------------------------------------------- #
# Gemini
# --------------------------------------------------------------------------- #
def build_client():
    try:
        from google import genai
    except ImportError:
        _fatal("google-genai is not installed.  pip install google-genai")
    return genai.Client(api_key=require_gemini_api_key())


def _is_rate_limit(exc):
    text = f"{type(exc).__name__} {exc}".lower()
    return any(m in text for m in ("rate limit", "ratelimit", "quota", "429",
                                   "resource_exhausted",
                                   "resource exhausted"))


def _strip_fences(raw):
    s = raw.strip()
    if s.startswith("```"):
        s = s.split("\n", 1)[1] if "\n" in s else s
        if s.rstrip().endswith("```"):
            s = s.rstrip()[:-3]
    return s.strip()


def call_gemini_json(client, prompt, temperature, state):
    """One Gemini call returning a parsed JSON object, with retry/backoff."""
    if state["calls"] >= MAX_TOTAL_CALLS:
        _fatal(
            f"Call budget exhausted ({MAX_TOTAL_CALLS}). This is a guard "
            "against a prompt bug burning the daily quota. Progress is "
            "checkpointed -- re-run to continue if this was expected."
        )

    for attempt in range(MAX_RETRIES + 1):
        try:
            state["calls"] += 1
            response = client.models.generate_content(
                model=settings.models.gemini_model,
                contents=prompt,
                config={
                    "temperature": temperature,
                    "response_mime_type": "application/json",
                },
            )
        except Exception as exc:  # noqa: BLE001
            if _is_rate_limit(exc) and attempt < MAX_RETRIES:
                wait = BACKOFF_SCHEDULE[min(attempt,
                                            len(BACKOFF_SCHEDULE) - 1)]
                print(f"      rate limit (attempt {attempt + 1}); waiting "
                      f"{wait}s")
                time.sleep(wait)
                continue
            raise RuntimeError(f"Gemini call failed: {exc}") from exc

        raw = getattr(response, "text", None)
        if not raw:
            raise RuntimeError("Gemini returned an empty/blocked response.")
        try:
            parsed = json.loads(_strip_fences(raw))
        except json.JSONDecodeError as exc:
            raise RuntimeError(
                f"Could not parse Gemini output as JSON: {raw!r}") from exc
        if not isinstance(parsed, dict):
            raise RuntimeError(f"Expected a JSON object, got: {parsed!r}")
        return parsed

    raise RuntimeError("Gemini retries exhausted on the rate-limit path.")


# --------------------------------------------------------------------------- #
# Prompts
# --------------------------------------------------------------------------- #
def generation_prompt(category, style_examples, avoid_texts):
    styles = "\n".join(f"  - {t}" for t in style_examples[:STYLE_ANCHORS])
    avoid = "\n".join(f"  - {t[:160]}" for t in avoid_texts[-40:])

    return f"""\
You write realistic IT support tickets in the voice of an ordinary, \
non-technical employee.

{SCOPE_ANCHORS[category]}

REGISTER -- write like these real examples (plain, everyday, slightly \
rambling; the person does NOT know the technical vocabulary):
{styles}

HARD RULES:
  - Write ONE new ticket for the category "{category}", and nothing else.
  - Plain everyday language. No jargon, no system names like PRD-APP-01, no
    error codes, no acronyms the average employee would not use.
  - Describe the SYMPTOM as the user experiences it, not the diagnosis.
  - 1 to 3 sentences.
  - It must be unmistakably "{category}" per the scope above, and must NOT
    read as any other category.

DO NOT duplicate or closely paraphrase any of these existing tickets:
{avoid}

Return STRICT JSON, exactly:
{{"text": "<the ticket>"}}
"""


def verification_prompt(text):
    cats = "\n".join(f"  - {c}" for c in CATEGORIES)
    return f"""\
Classify this IT support ticket into exactly one category.

TICKET:
{text}

CATEGORIES:
{cats}

Answer with STRICT JSON, exactly:
{{"category": "<one of the categories above, copied exactly>"}}
"""


# --------------------------------------------------------------------------- #
# Generation loop
# --------------------------------------------------------------------------- #
def embed_one(text, embedder):
    """L2-normalised embedding of a single candidate."""
    v = np.asarray(embedder.encode([text], convert_to_numpy=True),
                   dtype=np.float32)
    v /= (np.linalg.norm(v, axis=1, keepdims=True) + 1e-12)
    return v[0]


def max_similarity(vec, matrix):
    """Highest cosine similarity between vec and any row of matrix."""
    if matrix is None or len(matrix) == 0:
        return 0.0
    return float((matrix @ vec).max())


def run(per_category, dry_run):
    _banner("DEPLOYMENT-DISTRIBUTION CALIBRATION SET")
    print(f"target: {per_category} per category x {len(CATEGORIES)} = "
          f"{per_category * len(CATEGORIES)} tickets")
    print(f"cost:   2 Gemini calls per accepted ticket, "
          f"{GEMINI_CALL_DELAY_SEC}s apart")
    print("policy: a self-consistency disagreement DISCARDS the candidate "
          "(benchmark policy)")

    benchmark = load_benchmark()
    style = load_style_anchors()
    state = load_checkpoint()

    if state["accepted"]:
        print(f"\n[resume] {len(state['accepted'])} ticket(s) already "
              f"accepted, {state['calls']} calls spent")

    _banner("Loading the production embedder (for benchmark-overlap checks)")
    from src.agent.artifacts import load_artifacts
    artifacts = load_artifacts(require_gemini=False)
    embedder = artifacts.embedder

    bench_texts = [t["text"] for t in benchmark]
    bench_emb = np.asarray(
        embedder.encode(bench_texts, convert_to_numpy=True,
                        batch_size=64, show_progress_bar=False),
        dtype=np.float32)
    bench_emb /= (np.linalg.norm(bench_emb, axis=1, keepdims=True) + 1e-12)
    print(f"  embedded {len(bench_texts)} benchmark tickets for comparison")

    client = build_client()

    from src.classification.generalization_test import NOVEL_TICKETS
    base_avoid = ([t["text"] for t in NOVEL_TICKETS] + bench_texts)

    # Embeddings of already-accepted tickets, per category, so a resumed run
    # enforces the internal near-duplicate guard against earlier work too.
    accepted_vectors = {}
    for cat in CATEGORIES:
        prior = [a["text"] for a in state["accepted"]
                 if a["expected"] == cat]
        if not prior:
            continue
        m = np.asarray(embedder.encode(prior, convert_to_numpy=True,
                                       batch_size=64,
                                       show_progress_bar=False),
                       dtype=np.float32)
        m /= (np.linalg.norm(m, axis=1, keepdims=True) + 1e-12)
        accepted_vectors[cat] = m
    if accepted_vectors:
        print(f"  internal-duplicate guard seeded from "
              f"{sum(len(v) for v in accepted_vectors.values())} "
              f"existing ticket(s)")

    _banner("Generating")
    for category in CATEGORIES:
        have = [a for a in state["accepted"] if a["expected"] == category]
        while len(have) < per_category:
            avoid = base_avoid + [a["text"] for a in state["accepted"]
                                  if a["expected"] == category]
            n = len(have) + 1
            print(f"  [{category}] {n}/{per_category} ...", end=" ",
                  flush=True)

            try:
                gen = call_gemini_json(
                    client, generation_prompt(category, style[category],
                                              avoid),
                    temperature=0.95, state=state)
            except RuntimeError as exc:
                save_checkpoint(state)
                _fatal(f"Generation failed: {exc}\n"
                       "  Progress is checkpointed; re-run to continue.")
            time.sleep(GEMINI_CALL_DELAY_SEC)

            text = str(gen.get("text", "")).strip()
            if not text:
                print("empty, retrying")
                continue

            vec = embed_one(text, embedder)

            sim = max_similarity(vec, bench_emb)
            if sim > MAX_BENCHMARK_SIMILARITY:
                print(f"REJECT (benchmark overlap {sim:.3f})")
                state["rejected"].append({
                    "category": category, "text": text,
                    "reason": "benchmark_overlap",
                    "max_benchmark_similarity": sim})
                save_checkpoint(state)
                continue

            same_cat = accepted_vectors.get(category)
            internal = max_similarity(vec, same_cat)
            if internal > MAX_INTERNAL_SIMILARITY:
                print(f"REJECT (near-duplicate of an accepted "
                      f"{category} ticket, {internal:.3f})")
                state["rejected"].append({
                    "category": category, "text": text,
                    "reason": "internal_near_duplicate",
                    "max_internal_similarity": internal,
                    "max_benchmark_similarity": sim})
                save_checkpoint(state)
                continue

            try:
                ver = call_gemini_json(client, verification_prompt(text),
                                       temperature=0.0, state=state)
            except RuntimeError as exc:
                save_checkpoint(state)
                _fatal(f"Verification failed: {exc}\n"
                       "  Progress is checkpointed; re-run to continue.")
            time.sleep(GEMINI_CALL_DELAY_SEC)

            guess = str(ver.get("category", "")).strip()
            if guess != category:
                print(f"REJECT (self-consistency: Gemini said {guess!r})")
                state["rejected"].append({
                    "category": category, "text": text,
                    "reason": "self_consistency",
                    "gemini_guess": guess,
                    "max_benchmark_similarity": sim})
                save_checkpoint(state)
                continue

            state["accepted"].append({
                "id": f"depcal_{len(state['accepted']) + 1:04d}",
                "text": text,
                "expected": category,
                "flagged": False,
                "gemini_guess": guess,
                "max_benchmark_similarity": sim,
                "max_internal_similarity": internal,
            })
            accepted_vectors[category] = (
                vec[None, :] if same_cat is None
                else np.vstack([same_cat, vec[None, :]]))
            have = [a for a in state["accepted"]
                    if a["expected"] == category]
            save_checkpoint(state)
            print(f"ok (sim {sim:.3f}, {state['calls']} calls)")

    # ---- Write ------------------------------------------------------------ #
    _banner("Writing")
    accepted = state["accepted"]
    out = OUTPUT_JSON if not dry_run else OUTPUT_JSON.replace(
        ".json", "_dryrun.json")
    with open(out, "w", encoding="utf-8") as fh:
        json.dump(accepted, fh, indent=2, ensure_ascii=False)
        fh.write("\n")
    print(f"  {out}  ({len(accepted)} tickets)")

    with open(REJECTED_JSON, "w", encoding="utf-8") as fh:
        json.dump(state["rejected"], fh, indent=2, ensure_ascii=False)
        fh.write("\n")
    print(f"  {REJECTED_JSON}  ({len(state['rejected'])} rejected)")

    sims = [a["max_benchmark_similarity"] for a in accepted]
    n_self = sum(1 for r in state["rejected"]
                 if r["reason"] == "self_consistency")
    n_overlap = sum(1 for r in state["rejected"]
                    if r["reason"] == "benchmark_overlap")
    n_internal = sum(1 for r in state["rejected"]
                     if r["reason"] == "internal_near_duplicate")

    _banner("Summary")
    print(f"  accepted                  : {len(accepted)}")
    print(f"  rejected (self-consistency): {n_self}")
    print(f"  rejected (benchmark overlap): {n_overlap}")
    print(f"  rejected (internal near-duplicate): {n_internal}")
    print(f"  Gemini calls spent        : {state['calls']}")
    if sims:
        print(f"  max similarity to any benchmark ticket: {max(sims):.4f} "
              f"(cap {MAX_BENCHMARK_SIMILARITY})")
        print(f"  mean similarity to nearest benchmark  : "
              f"{float(np.mean(sims)):.4f}")

    for c in CATEGORIES:
        print(f"    {c:<20} {sum(1 for a in accepted if a['expected'] == c)}")

    if not dry_run and len(accepted) == PER_CATEGORY * len(CATEGORIES):
        if os.path.isfile(CHECKPOINT_JSON):
            os.remove(CHECKPOINT_JSON)
            print("\n  checkpoint cleared (run complete)")


def main():
    ap = argparse.ArgumentParser(
        description="Generate a deployment-distribution calibration set.")
    ap.add_argument("--per-category", type=int, default=PER_CATEGORY,
                    help=f"tickets per category (default {PER_CATEGORY}); "
                         "use 1 for a cheap dry run")
    args = ap.parse_args()

    dry = args.per_category < PER_CATEGORY
    if dry:
        print(f"\n*** DRY RUN: {args.per_category} per category "
              f"(~{args.per_category * len(CATEGORIES) * 2} calls) ***")
    run(args.per_category, dry_run=dry)


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\nInterrupted. Progress is checkpointed; re-run to continue.")
        sys.exit(130)
