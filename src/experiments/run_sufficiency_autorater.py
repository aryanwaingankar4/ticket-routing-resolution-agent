# src/experiments/run_sufficiency_autorater.py
"""
run_sufficiency_autorater.py
============================

Phase 6C, step 2 (Gemini, SPENDS QUOTA) and step 3 (local Ollama, free): rate
whether the retrieved context is SUFFICIENT to resolve each ticket.

THE QUESTION
------------
The production RAG gate is one scalar: top-1 BGE cosine >= 0.67, else escalate.
Phase 2B found two tickets that cleared it and still produced an ungrounded
draft -- G021 (N26) at 0.7010 and G024 (N31) at 0.6747. 6C asks whether a rater
that sees ONLY the ticket and its retrieved context -- never a draft -- flags
those two.

TWO BACKENDS, ONE PROMPT. build_prompt() is backend-agnostic and the rubric is
a single module constant, so both arms get byte-identical text by construction
-- the same discipline as 5C, and for the same reason: agreement between two
raters is only interpretable if both answered the same question.

WHAT THE RATER NEVER SEES
-------------------------
The similarity score, the 0.67 threshold, which arm the ticket is in, any
draft, and any Phase 2B label. Showing the similarity would leak the very gate
6C is tested against. The bundle built by build_sufficiency_context.py already
separates those into sufficiency_context_key.json; this script reads only the
items.

PRE-REGISTRATION -- FIXED BEFORE ANY CALL
-----------------------------------------
Primary, as COUNTS not rates: over the 33 eligible tickets, the full 2x2 of
rater verdict against the 2B human label. Of the 2 human-labelled ungrounded
drafts (G021, G024), how many are flagged INSUFFICIENT; of the 31 grounded, how
many are wrongly flagged. With 2 positives NO rate, proportion, interval, kappa
or significance test is estimable on the primary axis -- the primary result is a
CASE STUDY by construction and is reported as counts plus every disagreement
quoted.

Pre-registered secondaries: agreement with the live RAG gate on the 21 tickets
it already escalates; the same rubric on local qwen2.5:3b-instruct as a
cross-family check; and verdict stability on G021 and G024 at 3 repeats each.

**THE PRIMARY VERDICT FOR EVERY ITEM IS THE FIRST CALL ONLY (rep 1).** The
stability repeats are reported separately and NEVER change it. There is NO
majority vote: a majority over repeats would let a sampling artefact rewrite a
pre-registered primary, which is the same move as revising a rule after seeing
the results -- precisely what this project refuses to do. `temperature: 0.0` is
recorded inside every cached record, so if a repeat differs from its first call
that is reported as a DETERMINISM FINDING about the model, never as a reason to
revise the primary. This clause was fixed at the plan gate, before any call.

NO DEGENERACY OR BLOCKING RULE -- and that is deliberate. Phase 7C's rule was
carried over from 6B, where >= 0.95 domain AUC guarded a DENSITY-RATIO estimate
that becomes ill-posed near separability. 7C computed no density ratio, so the
threshold gated a quantity that did not exist there, and the gate declined to
revise it after the fact. 6C estimates no density ratio and fits no model, so
there is no ill-posedness threshold to import. What can actually go wrong here
is a population too small to support a rate, and that is DECLARED above rather
than gated.

QUOTA
-----
Gemini: 54 items x 1 call, plus 2 extra calls each on G021's and G024's items =
58 unique calls. --limit 3 FIRST, always. call_delay_sec 4.5 (project floor 4).
Every raw response is cached to data/sufficiency_raw/ keyed by prompt sha256
BEFORE parsing, so a crash or a re-score never re-spends. MAX_TOTAL_GEMINI_CALLS
is a hard in-code ceiling so a prompt bug cannot burn the daily budget.

Ollama is local and free. It loads NO embedding model, and neither does this
script -- build_sufficiency_context.py persisted the context precisely so that
BGE and a local generator are never resident at once. That overlap is what got
7C's run killed for memory. After the Ollama arm: `ollama stop
qwen2.5:3b-instruct`, confirm /api/ps is empty, THEN do any BGE work.

MEASUREMENT ONLY. Production frozen: cascade 0.50, RAG 0.67, clustering 0.80,
settings.conformal.enabled and settings.drift.enabled both False.
"""

from __future__ import annotations

import argparse
import datetime as _dt
import json
import os
import sys

_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(_THIS_DIR, os.pardir, os.pardir))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from src.agent.config import settings, config_fingerprint          # noqa: E402
from src.agent.logging_setup import ensure_utf8_console            # noqa: E402
from src.experiments.run_zeroshot_baselines import (               # noqa: E402
    GeminiBackend,
    OllamaBackend,
    OLLAMA_TIMEOUT_SEC,
    OLLAMA_URL,
    _slug,
    prompt_hash,
    read_cache,
    write_cache,
)

ensure_utf8_console()

DATA_DIR = os.path.join(PROJECT_ROOT, "data")
BUNDLE_PATH = os.path.join(DATA_DIR, "sufficiency_context_bundle.json")
RAW_DIR = os.path.join(DATA_DIR, "sufficiency_raw")

DEFAULT_OLLAMA_MODEL = "qwen2.5:3b-instruct"

# A verdict plus one sentence. 5C's 48 is sized for a bare category name and
# would truncate the reason; 7C's 512 is sized for a paraphrase.
SUFFICIENCY_NUM_PREDICT = 192
SEED = 42

# Hard ceiling: 58 planned unique calls + slack. A prompt bug must not be able
# to burn the daily 500. Every response so far is cached, so a re-run resumes.
MAX_TOTAL_GEMINI_CALLS = 70

VERDICTS = ("SUFFICIENT", "INSUFFICIENT")

# --------------------------------------------------------------------------
# THE RUBRIC. One definition, used VERBATIM by both backends. Fixed at the
# plan gate before any call was made, as Phase 2B fixed its own rubric.
# --------------------------------------------------------------------------
SUFFICIENCY_RUBRIC = """\
You are judging whether RETRIEVED CONTEXT IS SUFFICIENT to resolve a new IT
support ticket.

You are given a new ticket and the five past tickets that were retrieved for
it, each with the problem it reported and the resolution that fixed it. You are
NOT given any drafted resolution, and you must not invent one or judge one.

Judge ONLY this: do the retrieved past tickets contain enough to resolve the
new ticket?

  "SUFFICIENT"
      At least one retrieved ticket addresses the same underlying problem the
      new ticket reports, so that applying its resolution -- substituting the
      new ticket's own hostnames, app names, users or paths -- would plausibly
      resolve it.

  "INSUFFICIENT"
      No retrieved ticket addresses the new ticket's underlying problem. They
      solve adjacent, superficially similar, or same-category-but-different
      problems, or the new ticket needs a step or a piece of information that
      none of them covers. Resolving it would require knowledge from outside
      the retrieved set.

Four clarifications, each of which changes the answer:

  TOPICAL OVERLAP IS NOT SUFFICIENCY. The same product, category, error family
  or vocabulary is not the same problem. "Both are about disk space" is not
  enough; the retrieved fix must address the problem this ticket reports.

  DISAGREEMENT IS NOT INSUFFICIENCY. If the retrieved resolutions conflict with
  each other but at least one squarely addresses the new ticket's problem, the
  context is SUFFICIENT.

  COUNT DOES NOT MATTER. One relevant retrieved ticket among five is
  SUFFICIENT. Five irrelevant ones are INSUFFICIENT.

  DO NOT RATE QUALITY. Whether a retrieved resolution is good practice, safe,
  or how you would have fixed it, is not the question. A crude but applicable
  fix is SUFFICIENT.
"""

ANSWER_INSTRUCTION = """\
Reply with ONLY a JSON object, no prose and no code fence:
{"verdict": "SUFFICIENT" | "INSUFFICIENT", "reason": "<one sentence; when \
INSUFFICIENT, name what the context lacks>"}
"""


def _banner(text):
    rule = "=" * 70
    print("\n" + rule + "\n" + text + "\n" + rule)


def _fatal(message):
    print("\n[ERROR] " + message + "\n", file=sys.stderr)
    sys.exit(1)


class SufficiencyOllamaBackend(OllamaBackend):
    """5C's backend with a generation budget big enough for verdict + reason.

    Subclassed rather than parameterised in place, for the same reason 7C
    subclassed it: run_zeroshot_baselines.py produced published results and its
    `generate` must keep emitting exactly the bytes it emitted then.
    """

    def generate(self, prompt):
        import urllib.request

        payload = json.dumps({
            "model": self.model,
            "prompt": prompt,
            "stream": False,
            "format": "json",
            "options": {
                "temperature": 0.0,
                "seed": SEED,
                "num_predict": SUFFICIENCY_NUM_PREDICT,
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


# --------------------------------------------------------------------------
# The prompt -- ONE builder, both backends
# --------------------------------------------------------------------------
def build_prompt(item):
    parts = [
        SUFFICIENCY_RUBRIC,
        "\n================ NEW TICKET ================\n",
        item["ticket_text"].strip(),
        "\n\n================ RETRIEVED PAST TICKETS ================",
    ]
    for r in item["retrieved"]:
        parts.append(
            "\n--- Retrieved #{n} (category: {c}) ---\n"
            "Title: {t}\nProblem: {d}\nResolution: {res}".format(
                n=r["rank"], c=r["category"], t=r["title"],
                d=(r.get("description") or "").strip(),
                res=(r.get("resolution") or "").strip(),
            )
        )
    parts.append("\n\n================ YOUR ANSWER ================\n")
    parts.append(ANSWER_INSTRUCTION)
    return "".join(parts)


def parse_verdict(raw):
    """Extract the JSON object. Tolerates a code fence or surrounding prose."""
    t = (raw or "").strip()
    if not t:
        raise ValueError("empty response")
    if t.startswith("```"):
        t = t.split("```")[1] if "```" in t[3:] else t[3:]
        if t.lstrip().lower().startswith("json"):
            t = t.lstrip()[4:]
    start, end = t.find("{"), t.rfind("}")
    if start == -1 or end == -1 or end <= start:
        raise ValueError("no JSON object in reply: " + t[:200])
    obj = json.loads(t[start:end + 1])
    verdict = str(obj.get("verdict", "")).strip().upper()
    if verdict not in VERDICTS:
        raise ValueError("invalid verdict: " + repr(obj.get("verdict")))
    return {
        "verdict": verdict,
        "reason": str(obj.get("reason", "") or "").strip(),
    }


# --------------------------------------------------------------------------
# Cache -- one file per (backend, model, item, repeat), written BEFORE parsing
# --------------------------------------------------------------------------
def cache_path(backend_name, model, item_id, rep):
    return os.path.join(
        RAW_DIR,
        "{b}__{m}__{i}__rep{r}.json".format(
            b=backend_name, m=_slug(model), i=item_id, r=rep),
    )


def output_path(backend_name, model, only=None, limit=None):
    """Where this arm's verdicts go.

    The three run shapes get THREE DISTINCT names so none can overwrite
    another: the full pass, the --limit dry run (whose record is the
    provenance for what was seen before the full pass) and the --only
    stability run. tests/test_sufficiency_gate.py pins that they cannot
    collide.
    """
    name = "sufficiency_verdicts_{b}_{m}.json".format(
        b=backend_name, m=_slug(model))
    if only is not None:
        name = name.replace(".json", ".repeats.json")
    elif limit is not None:
        name = name.replace(".json", ".partial.json")
    return os.path.join(DATA_DIR, name)


def load_bundle():
    if not os.path.isfile(BUNDLE_PATH):
        _fatal(
            "Missing the context bundle:\n    {p}\n"
            "Build it first (offline, zero quota):\n"
            "    python src/experiments/build_sufficiency_context.py".format(
                p=BUNDLE_PATH)
        )
    with open(BUNDLE_PATH, "r", encoding="utf-8") as fh:
        doc = json.load(fh)
    items = doc.get("items", [])
    meta = doc.get("_meta", {})
    if not items:
        _fatal("The context bundle carries no items: " + BUNDLE_PATH)
    if not meta.get("retrieval_verified_against_2b"):
        _fatal(
            "The bundle does not record that its retrieval was verified "
            "against Phase 2B's recorded retrieval. 6C rates the SAME context "
            "the human labelled; rebuild the bundle."
        )
    return meta, items


def rate_items(backend, items, repeats, cached_only):
    """Return records for every (item, rep). rep 1 is the primary verdict."""
    os.makedirs(RAW_DIR, exist_ok=True)

    records = []
    n_live = n_cached = n_unparseable = 0

    total = sum(repeats.get(it["item_id"], 1) for it in items)
    print("calls to make (before cache) : %d over %d item(s)"
          % (total, len(items)))

    done = 0
    for item in items:
        iid = item["item_id"]
        prompt = build_prompt(item)
        phash = prompt_hash(prompt)

        for rep in range(1, repeats.get(iid, 1) + 1):
            done += 1
            path = cache_path(backend.name, backend.model, iid, rep)
            cached = read_cache(path, phash)

            if cached is None and cached_only:
                _fatal(
                    "--cached-only, but {i} rep{r} has no cached response for "
                    "this exact prompt (sha256 {h}).\n"
                    "A cache miss or a prompt-hash mismatch is FATAL in this "
                    "mode rather than a silent regeneration: it means the "
                    "prompt changed, so the cached answers are for a different "
                    "question.".format(i=iid, r=rep, h=phash[:12])
                )

            if cached is not None:
                rec = cached
                n_cached += 1
                mark = "cached"
            else:
                raw = backend.generate(prompt)
                rec = {
                    "phase": "6C",
                    "backend": backend.name,
                    "model": backend.model,
                    "model_digest": backend.model_digest,
                    "item_id": iid,
                    "repeat": rep,
                    "is_primary": rep == 1,
                    "prompt_sha256": phash,
                    "temperature": 0.0,
                    "seed": SEED if backend.name == "ollama" else None,
                    "raw": raw,
                    "timestamp": _dt.datetime.now().isoformat(timespec="seconds"),
                    "config_fingerprint": config_fingerprint(),
                }
                if getattr(backend, "allow_low_ram", False):
                    rec["low_ram_override"] = True
                # Written BEFORE parsing, so a parse bug never re-spends quota.
                write_cache(path, rec)
                n_live += 1
                mark = "LIVE"
                backend.pace()

            try:
                parsed = parse_verdict(rec.get("raw"))
                verdict, reason, err = parsed["verdict"], parsed["reason"], ""
            except Exception as exc:  # noqa: BLE001
                verdict, reason = "", ""
                err = "{t}: {e}".format(t=type(exc).__name__, e=str(exc)[:160])
                n_unparseable += 1

            records.append({
                "item_id": iid,
                "repeat": rep,
                "is_primary": rep == 1,
                "verdict": verdict,
                "reason": reason,
                "parse_error": err,
                "temperature": rec.get("temperature"),
                "backend": rec.get("backend"),
                "model": rec.get("model"),
                "model_digest": rec.get("model_digest"),
                "prompt_sha256": rec.get("prompt_sha256"),
                "timestamp": rec.get("timestamp"),
            })

            print("  [%2d/%2d] %s rep%d  %-13s %-6s %s"
                  % (done, total, iid, rep, verdict or "UNPARSEABLE", mark,
                     (err or reason)[:60]))

    return records, n_live, n_cached, n_unparseable


def parse_args(argv=None):
    p = argparse.ArgumentParser(
        description=("Phase 6C: rate retrieval sufficiency. Gemini SPENDS "
                     "QUOTA; ollama is local and free."))
    p.add_argument("--backend", choices=("gemini", "ollama"), required=True)
    p.add_argument("--model", default=None,
                   help=("Ollama model. 6C's pre-registered secondary arm is "
                         + DEFAULT_OLLAMA_MODEL + "."))
    p.add_argument("--limit", type=int, default=None,
                   help="Rate only the first N items. ALWAYS --limit 3 first.")
    p.add_argument("--repeats", type=int, default=1,
                   help=("Calls per item, for the stability secondary. rep 1 "
                         "is always the primary verdict; repeats never change "
                         "it and are never majority-voted."))
    p.add_argument("--only", default=None,
                   help="Comma-separated item ids to restrict the run to.")
    p.add_argument("--cached-only", action="store_true", dest="cached_only",
                   help=("Refuse every live call; a cache miss or prompt-hash "
                         "mismatch is fatal. Free re-score."))
    p.add_argument("--allow-low-ram", action="store_true",
                   dest="allow_low_ram",
                   help=("Ollama only. Overrides the RAM floor and stamps "
                         "low_ram_override into every response."))
    p.add_argument("--force", action="store_true",
                   help="Overwrite this arm's verdict file.")
    return p.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)

    _banner("Phase 6C -- retrieval-sufficiency autorater (%s)" % args.backend)
    print("Measurement only. Production frozen: cascade %.2f / RAG %.2f / "
          "clustering %.2f" % (settings.cascade.confidence_threshold,
                               settings.rag.similarity_threshold,
                               settings.clustering.resolution_similarity_threshold))
    print("conformal enabled : %s   drift enabled : %s"
          % (settings.conformal.enabled, settings.drift.enabled))
    print("config fingerprint: %s" % config_fingerprint())
    print("rubric            : module constant SUFFICIENCY_RUBRIC, verbatim")
    print("PRIMARY VERDICT   : rep 1 only -- repeats never revise it, "
          "no majority vote")

    meta, items = load_bundle()
    print("bundle            : %d items (%d eligible + %d escalated), "
          "seed %s" % (meta.get("items"), meta.get("eligible"),
                       meta.get("escalated"), meta.get("seed")))

    if args.only:
        wanted = [x.strip() for x in args.only.split(",") if x.strip()]
        by_id = {it["item_id"]: it for it in items}
        missing = [w for w in wanted if w not in by_id]
        if missing:
            _fatal("--only names unknown item ids: " + ", ".join(missing))
        items = [by_id[w] for w in wanted]
        print("--only            : " + ", ".join(wanted))

    if args.limit is not None:
        if args.limit < 1:
            _fatal("--limit must be at least 1.")
        items = items[:args.limit]
        print("MODE              : DRY RUN, first %d item(s) only" % len(items))

    repeats = {it["item_id"]: max(1, args.repeats) for it in items}

    if args.backend == "gemini":
        if args.model:
            _fatal("--model applies to the ollama backend only; the Gemini "
                   "model comes from settings.models.gemini_model.")
        planned = sum(repeats.values())
        if planned > MAX_TOTAL_GEMINI_CALLS:
            _fatal(
                "This run would make up to {p} Gemini calls, above the in-code "
                "ceiling of {c}. Raise it deliberately or narrow the run.".format(
                    p=planned, c=MAX_TOTAL_GEMINI_CALLS)
            )
        backend = GeminiBackend()
        print("backend           : Gemini %s -- SPENDS QUOTA (delay 4.5s, "
              "cap %d)" % (backend.model, MAX_TOTAL_GEMINI_CALLS))
    else:
        model = args.model or DEFAULT_OLLAMA_MODEL
        if args.cached_only:
            # No server contact, no RAM check: nothing will be generated.
            class _CachedOnly:
                name = "ollama"
                model_digest = None
                allow_low_ram = False

                def __init__(self, m):
                    self.model = m
                    self.calls = 0

                def generate(self, prompt):
                    raise AssertionError("--cached-only must not generate")

                def pace(self):
                    return
            backend = _CachedOnly(model)
            print("backend           : ollama %s -- CACHED ONLY, zero calls"
                  % model)
        else:
            from src.experiments.paraphrase_external_tickets import (
                require_ram_accounting_for_resident,
            )
            # 7C's corrected preflight: 5C's require_free_ram double-counts a
            # model Ollama already holds. It still fails closed -- an
            # unreadable residency counts as zero resident.
            if not args.allow_low_ram:
                require_ram_accounting_for_resident(model)
            backend = SufficiencyOllamaBackend(
                model, allow_low_ram=args.allow_low_ram)
            print("backend           : ollama %s (digest %s) -- local, free"
                  % (model, (backend.model_digest or "?")[:16]))

    dry = args.limit is not None or args.only is not None
    out_path = output_path(backend.name, backend.model,
                           only=args.only, limit=args.limit)
    if os.path.exists(out_path) and not args.force:
        _fatal(
            "Refusing to overwrite:\n    {p}\nRe-run with --force if that is "
            "what you intend. The raw cache is untouched either way, so "
            "nothing is re-spent.".format(p=out_path)
        )

    _banner("Rating")
    records, n_live, n_cached, n_bad = rate_items(
        backend, items, repeats, args.cached_only)

    _banner("Summary")
    print("records         : %d" % len(records))
    print("live calls      : %d" % n_live)
    print("from cache      : %d" % n_cached)
    print("unparseable     : %d" % n_bad)
    # Second, independent derivation of the spend: the backend's own counter.
    print("backend.calls   : %d  (independent count of live calls)"
          % backend.calls)
    if backend.calls != n_live:
        _fatal(
            "Live-call count disagrees: the loop counted {a} but the backend "
            "counted {b}. Two derivations of a spend must agree.".format(
                a=n_live, b=backend.calls)
        )

    primary = [r for r in records if r["is_primary"]]
    counts = {}
    for r in primary:
        counts[r["verdict"] or "UNPARSEABLE"] = (
            counts.get(r["verdict"] or "UNPARSEABLE", 0) + 1)
    print("primary verdicts: " + json.dumps(counts))

    hashes = sorted({r["prompt_sha256"] for r in records})
    doc = {
        "_meta": {
            "phase": "6C",
            "purpose": ("Sufficiency verdicts. rep 1 is the pre-registered "
                        "primary; higher repeats are the stability secondary "
                        "and never revise it."),
            "backend": backend.name,
            "model": backend.model,
            "model_digest": backend.model_digest,
            "temperature": 0.0,
            "dry_run": bool(dry),
            "items": len(items),
            "records": len(records),
            "live_calls": n_live,
            "cached": n_cached,
            "unparseable": n_bad,
            "repeats_requested": args.repeats,
            "distinct_prompt_hashes": len(hashes),
            "rubric_source": ("run_sufficiency_autorater.SUFFICIENCY_RUBRIC "
                              "(verbatim, shared by both backends)"),
            "bundle_fingerprint": meta.get("config_fingerprint"),
            "config_fingerprint": config_fingerprint(),
            "primary_rule": ("rep 1 only; no majority vote; a differing repeat "
                             "at temperature 0 is a determinism finding, not a "
                             "reason to revise the primary"),
        },
        "records": records,
    }
    with open(out_path, "w", encoding="utf-8") as fh:
        json.dump(doc, fh, indent=2, ensure_ascii=False)
    print("\n[write] " + out_path)
    print("[cache] " + RAW_DIR)


if __name__ == "__main__":
    main()
