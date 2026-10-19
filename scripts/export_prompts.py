"""Export the exact text of the two prompts for the paper's Appendix A.

Paper revision (Oct 2026), P1 and P2. Offline, no model, no API call.

Nothing is retyped: each prompt is rendered by the function that builds it
in production -- src.agent.resolver.build_prompt() for drafting and
run_zeroshot_baselines.build_prompt() for zero-shot classification -- with
placeholder field values. Each placeholder must occur exactly as many times as
it was inserted, or the export is FATAL (a placeholder that also occurs in
the fixed text would make the substitution ambiguous).

Writes data/paper_revision_oct/prompts.txt; refuses to overwrite without
--force.

    .\\venv\\Scripts\\python.exe scripts/export_prompts.py
"""

from __future__ import annotations

import argparse
import importlib.util
import os
import sys

PROJECT_ROOT = os.path.abspath(
    os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

OUT = os.path.join(PROJECT_ROOT, "data", "paper_revision_oct", "prompts.txt")
ZEROSHOT = os.path.join(PROJECT_ROOT, "src", "experiments",
                        "run_zeroshot_baselines.py")

# A sentinel similarity: build_prompt() formats it with :.3f.
SIM_SENTINEL, SIM_TEXT = 0.123, "0.123"


def die(msg):
    print(f"\n[FATAL] {msg}")
    raise SystemExit(1)


def check_once(text, needle, expected):
    n = text.count(needle)
    if n != expected:
        die(f"placeholder {needle!r} occurs {n} times, expected {expected}")


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args()
    if os.path.exists(OUT) and not args.force:
        die(f"Output exists (use --force to overwrite):\n    {OUT}")

    from src.agent.resolver import build_prompt as drafting_prompt
    from src.agent.schemas import RetrievalResult, RetrievedTicket, TicketIn

    ticket = TicketIn(title="<NEW TICKET TITLE>",
                      description="<NEW TICKET DESCRIPTION>")
    one = RetrievedTicket(id="<ID>", title="<PAST TITLE>",
                          description="<PAST DESCRIPTION>",
                          category="<PAST CATEGORY>",
                          resolution="<PAST RESOLUTION>",
                          similarity=SIM_SENTINEL)
    draft = drafting_prompt(ticket, RetrievalResult(retrieved=[one]))
    for needle in ("<NEW TICKET TITLE>", "<NEW TICKET DESCRIPTION>",
                   "<PAST TITLE>", "<PAST DESCRIPTION>", "<PAST CATEGORY>",
                   "<PAST RESOLUTION>", SIM_TEXT):
        check_once(draft, needle, 1)
    draft = draft.replace(SIM_TEXT, "<SIMILARITY, 3 d.p.>")

    spec = importlib.util.spec_from_file_location("zs", ZEROSHOT)
    zs = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(zs)
    zero = zs.build_prompt("<TICKET TEXT>")
    check_once(zero, "<TICKET TEXT>", 1)

    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    with open(OUT, "w", encoding="utf-8", newline="\n") as fh:
        fh.write("=== P1: drafting prompt (src/agent/resolver.py "
                 "build_prompt) ===\n")
        fh.write("The 'Past ticket #k' block is repeated for each of the top-5 "
                 "retrieved tickets, k = 1..5.\n\n")
        fh.write(draft)
        fh.write("\n\n=== P2: zero-shot classification prompt "
                 "(src/experiments/run_zeroshot_baselines.py build_prompt) ===\n\n")
        fh.write(zero)
    print(f"[write] {os.path.relpath(OUT, PROJECT_ROOT)}")


if __name__ == "__main__":
    main()
