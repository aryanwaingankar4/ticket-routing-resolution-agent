# src/experiments/run_groundedness_judge.py
"""
run_groundedness_judge.py
=========================

Phase 2B, step 2: run the LLM judge over the same items the human labels.

WHAT THIS SCRIPT IS -- AND IS NOT
----------------------------------
This judge is NOT a labour-saving device. The whole eligible population is 33
items, and the human labels all 33. So the judge is not scaling anything: it
is the OBJECT OF STUDY. Its agreement with the human labels is reported as a
result in its own right -- how far an LLM judge can be trusted on grounded
resolution text in this domain.

That is why it uses the rubric text VERBATIM from
build_groundedness_set.RUBRIC_TEXT rather than restating it. Agreement between
the judge and the human is only interpretable if both answered the same
question; a paraphrased rubric would quietly make the comparison meaningless.

Its verdicts are written to a SEPARATE file so they cannot leak into the
human labelling pass.

COST
----
One Gemini call per item. --limit N for a cheap prompt check first.

This script never writes to groundedness_set.json, so it cannot disturb
labels already entered there.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time

_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(_THIS_DIR, os.pardir, os.pardir))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from src.agent.artifacts import load_artifacts
from src.agent.config import settings
from src.agent.errors import classify_llm_exception
from src.experiments.build_groundedness_set import RUBRIC_LABELS, RUBRIC_TEXT

DATA_DIR = os.path.join(PROJECT_ROOT, "data")
SET_PATH = os.path.join(DATA_DIR, "groundedness_set.json")
JUDGE_PATH = os.path.join(DATA_DIR, "groundedness_judge.json")

GEMINI_CALL_DELAY_SEC = 4.0
MAX_ATTEMPTS = 3


def die(message):
    print("\n[ERROR] " + message + "\n", file=sys.stderr)
    sys.exit(1)


def build_judge_prompt(item):
    parts = [RUBRIC_TEXT, "\n\n================ NEW TICKET ================\n",
             item["ticket_text"],
             "\n\n================ DRAFTED RESOLUTION ================\n",
             item["draft"],
             "\n\n================ RETRIEVED PAST RESOLUTIONS ================"]
    for r in item["retrieved_resolutions"]:
        parts.append("\n--- Retrieved #%d (category: %s) ---\nTitle: %s\n"
                     "Resolution: %s"
                     % (r["rank"], r["category"], r["title"], r["resolution"]))
    parts.append(
        "\n\n================ YOUR ANSWER ================\n"
        "Reply with ONLY a JSON object, no prose and no code fence:\n"
        '{"label": "grounded" | "partially_grounded" | "ungrounded", '
        '"hedge_appropriate": true | false, '
        '"unsupported_span": "<the offending text, or empty string>", '
        '"reason": "<one sentence>"}\n'
    )
    return "".join(parts)


def parse_verdict(text):
    """Extract the JSON object. Tolerates a code fence or surrounding prose."""
    t = (text or "").strip()
    if t.startswith("```"):
        t = t.split("```")[1] if "```" in t[3:] else t[3:]
        if t.lstrip().lower().startswith("json"):
            t = t.lstrip()[4:]
    start, end = t.find("{"), t.rfind("}")
    if start == -1 or end == -1 or end <= start:
        raise ValueError("no JSON object in judge reply: " + t[:200])
    obj = json.loads(t[start:end + 1])
    label = str(obj.get("label", "")).strip()
    if label not in RUBRIC_LABELS:
        raise ValueError("judge returned an invalid label: " + repr(label))
    hedge = obj.get("hedge_appropriate")
    if not isinstance(hedge, bool):
        raise ValueError("hedge_appropriate was not a boolean: " + repr(hedge))
    return {
        "label": label,
        "hedge_appropriate": hedge,
        "unsupported_span": str(obj.get("unsupported_span", "") or ""),
        "reason": str(obj.get("reason", "") or ""),
    }


def parse_args():
    p = argparse.ArgumentParser(
        description="Run the LLM groundedness judge. One Gemini call per item.")
    p.add_argument("--limit", type=int, default=None,
                   help="Judge only the first N items (prompt check).")
    return p.parse_args()


def main():
    args = parse_args()

    print("=" * 74)
    print("Phase 2B / step 2 -- LLM judge (object of study, not a scaler)")
    print("=" * 74)

    if not os.path.exists(SET_PATH):
        die("Missing the labelling set:\n    " + SET_PATH
            + "\nBuild it first with:\n"
              "    python src/experiments/build_groundedness_set.py")

    set_doc = json.load(open(SET_PATH, encoding="utf-8"))
    items = set_doc.get("items", [])
    if not items:
        die("The labelling set contains no items: " + SET_PATH)

    if set_doc.get("_meta", {}).get("dry_run"):
        print("[warn] the labelling set is marked dry_run -- these are a")
        print("       partial set of drafts. Judging them is fine for a")
        print("       prompt check, but do not score it as the real run.")

    print("Gemini model : " + settings.models.gemini_model)
    print("items        : %d%s" % (len(items),
                                   ("  (limit %d)" % args.limit)
                                   if args.limit else ""))
    print("rubric       : imported verbatim from build_groundedness_set.py")

    artifacts = load_artifacts(require_gemini=True)
    client = artifacts.gemini_client
    if client is None:
        die("No Gemini client was loaded. Check GEMINI_API_KEY in .env.")

    todo = items[:args.limit] if args.limit else items
    verdicts = []
    failures = []

    print("\n" + "=" * 74)
    print("Judging %d item(s)  [%d Gemini call(s)]" % (len(todo), len(todo)))
    print("=" * 74)

    for n, item in enumerate(todo, start=1):
        prompt = build_judge_prompt(item)
        verdict = None
        last_err = None
        for attempt in range(1, MAX_ATTEMPTS + 1):
            try:
                resp = client.models.generate_content(
                    model=settings.models.gemini_model, contents=prompt)
                verdict = parse_verdict(resp.text)
                break
            except Exception as exc:  # noqa: BLE001
                last_err = exc
                kind = type(exc).__name__
                try:
                    kind = type(classify_llm_exception(exc)).__name__
                except Exception:  # noqa: BLE001
                    pass
                print("    [%s] attempt %d/%d failed (%s)"
                      % (item["item_id"], attempt, MAX_ATTEMPTS, kind))
                if attempt < MAX_ATTEMPTS:
                    time.sleep(GEMINI_CALL_DELAY_SEC * attempt)

        if verdict is None:
            failures.append((item["item_id"], str(last_err)[:200]))
            print("  [%2d/%2d] %s  FAILED" % (n, len(todo), item["item_id"]))
        else:
            verdicts.append(dict(item_id=item["item_id"], **verdict))
            print("  [%2d/%2d] %s  %-18s hedge_ok=%s"
                  % (n, len(todo), item["item_id"], verdict["label"],
                     verdict["hedge_appropriate"]))

        if n < len(todo):
            time.sleep(GEMINI_CALL_DELAY_SEC)

    print("\n[judge] verdicts : %d" % len(verdicts))
    print("[judge] failures : %d" % len(failures))
    for iid, msg in failures:
        print("        %s: %s" % (iid, msg))

    doc = {
        "_meta": {
            "purpose": ("LLM judge verdicts for Phase 2B. Studied against the "
                        "human labels, never substituted for them."),
            "dry_run": bool(args.limit) or bool(
                set_doc.get("_meta", {}).get("dry_run")),
            "gemini_model": settings.models.gemini_model,
            "rubric_source": "build_groundedness_set.RUBRIC_TEXT (verbatim)",
            "items_judged": len(verdicts),
            "failures": len(failures),
            "warning": ("Do not read this while labelling "
                        "data/groundedness_set.json."),
        },
        "verdicts": verdicts,
    }
    with open(JUDGE_PATH, "w", encoding="utf-8") as fh:
        json.dump(doc, fh, indent=2, ensure_ascii=False)
    print("\n[write] " + JUDGE_PATH)

    print("\n" + "=" * 74)
    print("DONE")
    print("=" * 74)


if __name__ == "__main__":
    main()
