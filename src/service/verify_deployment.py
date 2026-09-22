"""
verify_deployment.py -- does a RUNNING deployment serve what this repo says?

WHY THIS EXISTS (Phase 8B)
--------------------------
Two numbers get quoted whenever a container is declared good: the
config_fingerprint it reports, and the routing decision it produces for a known
ticket. This project's rule after the Phase 7A record correction is that a
number quoted at a gate comes from a committed script, never from an
interactive session -- two domain AUCs that were computed at a prompt, quoted at
a gate and never committed turned out not to reproduce. So the container check
is a file, not a sequence of curl commands in a terminal someone closed.

WHAT IT CHECKS
--------------
1. FINGERPRINT. /health's config_fingerprint against config_fingerprint()
   computed right here from src/agent/config.py. This is the check that catches
   a deployment serving stale configuration: the same value is stamped on every
   PipelineResult, so a mismatch means stored results and the live service
   disagree about what the thresholds are.

2. NO-KEY SURFACE. /health, /agents/classify, /agents/retrieve,
   /policy/rag-gate and /triage must all answer without GEMINI_API_KEY.
   /agents/resolve is the ONLY endpoint that needs one, and without a key it
   must answer 503 with a message saying so -- not a 500, and not a plausible
   empty draft.

3. ADV_08 OVER HTTP. One adversarial ticket, end to end, compared against TWO
   independently recorded derivations of the same decision:

       data/adversarial_escalation_results.csv     (6 dp, the regression gate)
       tests/goldens/adversarial_baseline.json     (full precision, the goldens)

   Checking against two is the point. This project's recurring bug is a value
   that is wrong for its context but internally consistent, and a single
   reference cannot tell you that you are comparing against a stale copy of
   itself.

WHY ADV_08 SPECIFICALLY
-----------------------
It is the one adversarial ticket that exercises BOTH gates: Tier-1 confidence
0.3183 is below the 0.50 cascade threshold, so the cascade falls through to
Tier-2; top similarity 0.6124 is below the 0.67 RAG threshold, so the ticket
escalates. A container that got either artifact wrong moves one of those two
numbers.

Run from the project root, against a container or a local uvicorn:

    python src/service/verify_deployment.py
    python src/service/verify_deployment.py --base-url http://localhost:8000

Exit code 0 means every check passed. Nothing is written anywhere.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import sys
import urllib.error
import urllib.request

PROJECT_ROOT = os.path.dirname(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, PROJECT_ROOT)

DATA_DIR = os.path.join(PROJECT_ROOT, "data")
ADVERSARIAL_CSV = os.path.join(DATA_DIR, "adversarial_escalation_results.csv")
ADVERSARIAL_JSON = os.path.join(DATA_DIR, "adversarial_escalation_tickets.json")
GOLDEN_JSON = os.path.join(PROJECT_ROOT, "tests", "goldens",
                           "adversarial_baseline.json")

TICKET_ID = "adv_08"

# The CSV records 6 decimal places; the goldens record full float precision.
CSV_TOLERANCE = 5e-7

# TWO TOLERANCES AGAINST THE GOLDENS, FOR TWO KINDS OF NUMBER.
#
# tier1_conf comes from TF-IDF + LogisticRegression in float64 and is
# bit-identical across platforms: the container reproduces the golden's
# 0.3182984770932253 exactly. It is checked exactly, and it should stay that
# way -- if that one ever moves, something real moved.
#
# The retrieval similarity comes from a float32 BGE forward pass and a FAISS
# inner product. Those accumulate in an order set by the BLAS kernel and the
# SIMD width of the machine, so the LAST DIGITS ARE PLATFORM-SPECIFIC. Measured
# in Phase 8B: the Linux container returns 0.6123799085617065 where the Windows
# goldens record 0.6123800277709961 -- a fixed -1.192e-07 offset, identical on
# every repeat, against a distance of +5.762e-02 from the 0.67 gate this value
# feeds. The margin is 483,352x the offset.
#
# So the exact check was the WRONG TEST for this number: it asserted a
# guarantee the pipeline does not make. The tolerance below is float32's, it is
# documented here and in the README rather than quietly widened, and the
# measured delta is printed on every run whether it passes or not.
GOLDEN_TOLERANCE_EXACT = 1e-12
GOLDEN_TOLERANCE_FLOAT32 = 1e-6

_failures: list[str] = []


# --------------------------------------------------------------------------- #
# Output helpers. Clear messages, not tracebacks -- this may run in front of an
# audience, and it certainly runs in CI.                                       #
# --------------------------------------------------------------------------- #
def _banner(text: str) -> None:
    print("\n" + "=" * 74)
    print(text)
    print("=" * 74)


def _ok(text: str) -> None:
    print(f"  [PASS] {text}")


def _fail(text: str) -> None:
    _failures.append(text)
    print(f"  [FAIL] {text}")


def _fatal(text: str) -> "None":
    print(f"\nFATAL: {text}\n")
    sys.exit(2)


# --------------------------------------------------------------------------- #
# HTTP. stdlib only: this script must run inside a bare container as happily as
# in the project venv.                                                         #
# --------------------------------------------------------------------------- #
def _request(url: str, payload: dict | None = None, timeout: float = 120.0):
    """Return (status_code, parsed_json_or_text). Never raises on 4xx/5xx."""
    data = None
    headers = {"Accept": "application/json"}
    if payload is not None:
        data = json.dumps(payload).encode("utf-8")
        headers["Content-Type"] = "application/json"

    req = urllib.request.Request(url, data=data, headers=headers,
                                 method="POST" if payload is not None else "GET")
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            body = resp.read().decode("utf-8")
            status = resp.status
    except urllib.error.HTTPError as exc:
        body = exc.read().decode("utf-8")
        status = exc.code
    except urllib.error.URLError as exc:
        _fatal(f"Could not reach {url}: {exc.reason}\n"
               f"  Is the service running? Start it with:\n"
               f"      docker run --rm -p 8000:8000 ticket-triage:8b\n"
               f"  or  uvicorn src.service.api:app --port 8000")
    try:
        return status, json.loads(body)
    except json.JSONDecodeError:
        return status, body


# --------------------------------------------------------------------------- #
# Reference values.                                                            #
# --------------------------------------------------------------------------- #
def _load_adversarial_ticket() -> dict:
    with open(ADVERSARIAL_JSON, "r", encoding="utf-8") as fh:
        tickets = json.load(fh)
    for ticket in tickets:
        if ticket.get("id") == TICKET_ID:
            return ticket
    _fatal(f"{TICKET_ID} is not in {ADVERSARIAL_JSON}")


def _load_csv_reference() -> dict:
    with open(ADVERSARIAL_CSV, "r", encoding="utf-8", newline="") as fh:
        for row in csv.DictReader(fh):
            if row["id"] == TICKET_ID:
                return {
                    "tier1_confidence": float(row["tier1_confidence"]),
                    "rag_similarity": float(row["rag_similarity"]),
                    "predicted_category": row["predicted_category"],
                    "escalated": row["actual_escalate"].strip().lower() == "true",
                }
    _fatal(f"{TICKET_ID} is not in {ADVERSARIAL_CSV}")


def _load_golden_reference() -> dict:
    with open(GOLDEN_JSON, "r", encoding="utf-8") as fh:
        golden = json.load(fh)
    for row in golden["rows"]:
        if row["id"] == TICKET_ID:
            return {
                "tier1_confidence": float(row["tier1_confidence"]),
                "rag_similarity": float(row["rag_similarity"]),
                "predicted_category": row["predicted_category"],
                "escalated": bool(row["actual_escalate"]),
                "tier": int(row["tier"]),
            }
    _fatal(f"{TICKET_ID} is not in {GOLDEN_JSON}")


# --------------------------------------------------------------------------- #
# Check 1 -- the fingerprint.                                                  #
# --------------------------------------------------------------------------- #
def check_fingerprint(base_url: str) -> dict:
    from src.agent.config import config_fingerprint

    _banner("CHECK 1 -- config_fingerprint: deployment vs this repository")

    status, health = _request(f"{base_url}/health")
    if status != 200 or not isinstance(health, dict):
        _fatal(f"GET /health returned {status}: {health}")

    served = health.get("config_fingerprint", "")
    local = config_fingerprint()

    print(f"  served by {base_url:<28} : {served}")
    print(f"  computed from src/agent/config.py : {local}")

    if served != local:
        _fail(f"config_fingerprint MISMATCH -- the deployment is serving a "
              f"different configuration than this checkout describes "
              f"(served {served}, local {local}). This is the stale-artifact "
              f"signal; investigate it, do not widen a tolerance.")
    else:
        _ok(f"identical ({served})")

    print(f"\n  embedding_model : {health.get('embedding_model')}")
    print(f"  embedding_dim   : {health.get('embedding_dim')}")
    print(f"  index_ntotal    : {health.get('index_ntotal')}")
    print(f"  metadata_rows   : {health.get('metadata_rows')}")
    print(f"  cascade / rag   : {health.get('cascade_confidence_threshold')}"
          f" / {health.get('rag_similarity_threshold')}")
    print(f"  tier1 rows      : {health.get('tier1_fitted_rows')}")
    print(f"  resolution      : "
          f"{'available' if health.get('resolution_available') else 'unavailable (no key)'}")

    if health.get("index_ntotal") != health.get("metadata_rows"):
        _fail("index.ntotal != metadata rows -- the deployment's index and "
              "metadata are not aligned")
    return health


# --------------------------------------------------------------------------- #
# Check 2 -- the surface that must work with no API key.                       #
# --------------------------------------------------------------------------- #
def check_no_key_surface(base_url: str, health: dict) -> None:
    _banner("CHECK 2 -- the endpoints that must serve WITHOUT GEMINI_API_KEY")

    ticket = {"title": "Laptop will not boot after the overnight update",
              "ticket_id": "verify_deployment_probe"}

    status, classification = _request(f"{base_url}/agents/classify", ticket)
    if status == 200:
        _ok(f"/agents/classify -> 200 ({classification.get('category')}, "
            f"tier {classification.get('tier')})")
    else:
        _fail(f"/agents/classify -> {status} (expected 200): {classification}")

    status, retrieval = _request(f"{base_url}/agents/retrieve", ticket)
    if status == 200:
        _ok(f"/agents/retrieve -> 200 "
            f"({len(retrieval.get('retrieved', []))} neighbours)")
    else:
        _fail(f"/agents/retrieve -> {status} (expected 200): {retrieval}")
        retrieval = {"retrieved": []}

    status, gate = _request(f"{base_url}/policy/rag-gate",
                            {"retrieval": retrieval})
    if status == 200:
        _ok(f"/policy/rag-gate -> 200 (escalate={gate.get('escalate')}, "
            f"threshold={gate.get('threshold_applied')})")
    else:
        _fail(f"/policy/rag-gate -> {status} (expected 200): {gate}")

    status, triage = _request(f"{base_url}/triage", ticket)
    if status == 200:
        _ok(f"/triage -> 200 (ok={triage.get('ok')})")
    else:
        _fail(f"/triage -> {status} (expected 200): {triage}")

    # /agents/resolve is the one endpoint that spends quota, so it is the one
    # endpoint allowed to refuse without a key -- and it must refuse with 503
    # ("unavailable"), never 500 ("we crashed") and never a plausible draft.
    #
    # It takes TWO body parameters, `ticket` and `retrieval`, so FastAPI embeds
    # them: a bare ticket body is a 422, which is the service correctly
    # rejecting a malformed request rather than the refusal being tested here.
    # Sending the wrong shape was this script's own first bug.
    status, resolve = _request(f"{base_url}/agents/resolve",
                               {"ticket": ticket, "retrieval": retrieval})
    key_present = bool(health.get("resolution_available"))
    if key_present:
        print("  NOTE: this deployment HAS a key, so /agents/resolve was "
              "expected to work and would spend quota; not asserted here.")
        _ok(f"/agents/resolve -> {status} (deployment is keyed)")
    elif status == 503:
        _ok("/agents/resolve -> 503 with no key, as documented")
    else:
        _fail(f"/agents/resolve -> {status} with no key (expected 503): "
              f"{resolve}")


# --------------------------------------------------------------------------- #
# Check 3 -- adv_08 end to end, against two recorded derivations.              #
# --------------------------------------------------------------------------- #
def _compare(label: str, observed: float, expected: float,
             tolerance: float) -> None:
    delta = abs(observed - expected)
    if delta <= tolerance:
        _ok(f"{label}: {observed!r} matches {expected!r} "
            f"(|delta| = {delta:.3e} <= {tolerance:.0e})")
    else:
        _fail(f"{label}: deployment produced {observed!r}, the recorded value "
              f"is {expected!r} (|delta| = {delta:.3e} > {tolerance:.0e}). "
              f"A routing number moved -- report it, do not widen this "
              f"tolerance.")


def check_adv_08(base_url: str) -> None:
    # The thresholds live in config and are read from it, never restated here.
    from src.agent.config import settings

    _banner(f"CHECK 3 -- {TICKET_ID} over HTTP vs two recorded derivations")

    ticket = _load_adversarial_ticket()
    csv_ref = _load_csv_reference()
    golden_ref = _load_golden_reference()

    print(f"  ticket : {ticket['text']!r}")
    print(f"  refs   : data/adversarial_escalation_results.csv (6 dp) and "
          f"tests/goldens/adversarial_baseline.json (full precision)")

    # Exactly how the regression gate builds it:
    # src/experiments/test_adversarial_escalation.py:659
    status, response = _request(
        f"{base_url}/triage",
        {"title": ticket["text"], "ticket_id": ticket["id"]},
    )
    if status != 200:
        _fatal(f"POST /triage returned {status}: {response}")
    if not response.get("ok"):
        _fatal(f"/triage reported a failure rather than a decision: "
               f"{response.get('failure')}")

    result = response["result"]
    classification = result["classification"]
    retrieved = result["retrieval"]["retrieved"]
    decision = result["decision"]

    observed_tier = int(classification["tier"])
    observed_tier1 = float(classification["tier1_conf"])
    observed_sim = float(retrieved[0]["similarity"]) if retrieved else 0.0
    observed_cat = classification["category"]
    observed_esc = bool(decision["escalated"])

    print(f"\n  deployment: tier {observed_tier}, tier1_conf {observed_tier1!r}, "
          f"similarity {observed_sim!r},\n"
          f"              category {observed_cat!r}, escalated {observed_esc}")

    if observed_tier == golden_ref["tier"]:
        _ok(f"tier {observed_tier} (the cascade fell through to Tier-2, as "
            f"recorded)")
    else:
        _fail(f"tier: deployment says {observed_tier}, goldens say "
              f"{golden_ref['tier']}")

    if observed_cat == csv_ref["predicted_category"] == golden_ref["predicted_category"]:
        _ok(f"predicted_category {observed_cat!r} (both references agree)")
    else:
        _fail(f"predicted_category: deployment {observed_cat!r}, CSV "
              f"{csv_ref['predicted_category']!r}, golden "
              f"{golden_ref['predicted_category']!r}")

    if observed_esc == csv_ref["escalated"] == golden_ref["escalated"]:
        _ok(f"escalated={observed_esc} (both references agree)")
    else:
        _fail(f"escalated: deployment {observed_esc}, CSV "
              f"{csv_ref['escalated']}, golden {golden_ref['escalated']}")

    _compare("tier1_conf vs CSV      ", observed_tier1,
             csv_ref["tier1_confidence"], CSV_TOLERANCE)
    # Exact: TF-IDF in float64 is bit-identical across platforms.
    _compare("tier1_conf vs goldens  ", observed_tier1,
             golden_ref["tier1_confidence"], GOLDEN_TOLERANCE_EXACT)
    _compare("similarity vs CSV      ", observed_sim,
             csv_ref["rag_similarity"], CSV_TOLERANCE)
    # float32 tolerance, and the delta is printed either way. See the note by
    # GOLDEN_TOLERANCE_FLOAT32.
    _compare("similarity vs goldens  ", observed_sim,
             golden_ref["rag_similarity"], GOLDEN_TOLERANCE_FLOAT32)

    sim_delta = observed_sim - golden_ref["rag_similarity"]
    margin = settings.rag.similarity_threshold - observed_sim
    print(f"\n  cross-platform delta on the embedding-derived similarity: "
          f"{sim_delta:+.3e}\n"
          f"  distance from the 0.67 gate it feeds:                     "
          f"{margin:+.3e}  ({abs(margin / sim_delta):,.0f}x larger)"
          if sim_delta else
          f"\n  cross-platform delta on the embedding-derived similarity: 0")

    # The two gates adv_08 is here to exercise, checked as inequalities rather
    # than restated as constants.
    if observed_tier1 < settings.cascade.confidence_threshold:
        _ok(f"tier1_conf {observed_tier1:.6f} < cascade threshold "
            f"{settings.cascade.confidence_threshold} -- Tier-2 was reached")
    else:
        _fail(f"tier1_conf {observed_tier1:.6f} is NOT below the cascade "
              f"threshold {settings.cascade.confidence_threshold}")

    if observed_sim < settings.rag.similarity_threshold:
        _ok(f"similarity {observed_sim:.6f} < RAG threshold "
            f"{settings.rag.similarity_threshold} -- escalated, Gemini not called")
    else:
        _fail(f"similarity {observed_sim:.6f} is NOT below the RAG threshold "
              f"{settings.rag.similarity_threshold}")

    if result.get("suggestion") is not None:
        _fail("an escalating ticket came back with a drafted suggestion -- "
              "Gemini must never be called below the RAG gate")
    else:
        _ok("no suggestion drafted (Gemini was not called)")


# --------------------------------------------------------------------------- #
def main() -> int:
    parser = argparse.ArgumentParser(
        description="Verify a running deployment against this repository's "
                    "recorded configuration and routing decisions.")
    parser.add_argument("--base-url", default="http://localhost:8000",
                        help="Base URL of the running service "
                             "(default: http://localhost:8000)")
    args = parser.parse_args()
    base_url = args.base_url.rstrip("/")

    _banner(f"verify_deployment.py -- {base_url}")

    health = check_fingerprint(base_url)
    check_no_key_surface(base_url, health)
    check_adv_08(base_url)

    _banner("SUMMARY")
    if _failures:
        print(f"  {len(_failures)} check(s) FAILED:\n")
        for failure in _failures:
            print(f"    - {failure}")
        print("\n  A failure here is a real difference between the deployment "
              "and this\n  checkout. Investigate it; do not paper over it.\n")
        return 1

    print("  All checks passed. The deployment serves this checkout's "
          "configuration\n  and reproduces its recorded routing decision.\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
