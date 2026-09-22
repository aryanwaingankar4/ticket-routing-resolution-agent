# src/experiments/paraphrase_external_tickets.py
"""
Phase 7C, step 1 -- paraphrase external tickets into plain register.

WHY THIS EXISTS. Phase 7B tested Finding 1 on the external corpus under a
VERSION SHIFT WITHIN ONE GENERATOR and found no contrast between the tiers.
That may not settle anything, because Finding 1 was measured under a
PARAPHRASE / REGISTER SHIFT -- benchmark tickets rewritten out of the training
templates' voice. TF-IDF's failure mechanism is SURFACE-VOCABULARY CHANGE:
when the words move, a bag-of-n-grams model's scores move with them and its
conformal quantile stops transferring. A version shift need not change surface
vocabulary at all, so 7B may never have exercised the mechanism it was
testing.

This script builds the missing like-for-like test set. It generates nothing
that is scored here -- scoring is run_paraphrase_shift_conformal.py, so a
re-score never costs a regeneration.

ZERO GEMINI QUOTA. Paraphrasing runs on local Ollama, the same
qwen2.5:3b-instruct that is 5C's published local arm.

THE PAIRING IS THE POINT. Every retained paraphrase keeps its original, so the
two arms are the SAME TICKETS with the same labels, and the comparison is
paired rather than two independent samples. An unpaired version of this test
would confound the register change with whatever else differs between two
draws.

GUARDS, PRE-REGISTERED
----------------------
* A paraphrase at BGE cosine >= 0.95 to its source WAS NOT REWRITTEN, and is
  discarded. The count is reported, never silently absorbed.
* An unparseable or empty response is discarded and counted SEPARATELY, so
  "the model refused" can never look like "the model paraphrased faithfully".
* Mean source-paraphrase similarity is reported in both spaces. The
  manipulation check itself lives in the scoring script, where the TF-IDF
  space -- the space the mechanism lives in -- is measured properly.

CACHING. Every raw response is written to data/external_paraphrase_raw/ keyed
by PROMPT HASH before any parsing happens, so a crash never re-does finished
work and editing the prompt invalidates old answers by construction. The cache
is committed, as 5C's is: generated text feeding a published result has to be
auditable.

Run from the project root (offline, no Gemini calls):
    python src/experiments/paraphrase_external_tickets.py --limit 3   # ALWAYS FIRST
    python src/experiments/paraphrase_external_tickets.py
    python src/experiments/paraphrase_external_tickets.py --force
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time

import numpy as np
import pandas as pd

THIS_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(THIS_DIR, os.pardir, os.pardir))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from src.agent.config import settings, config_fingerprint          # noqa: E402
from src.agent.logging_setup import ensure_utf8_console            # noqa: E402
from src.experiments.calibrate_conformal import _banner, _fatal    # noqa: E402
from src.experiments.fetch_external_dataset import (               # noqa: E402
    ENGLISH_CSV,
    OUT_DIR,
)
from src.experiments.profile_external_dataset import (             # noqa: E402
    EMB_NPY,
    NEAR_DUP_THRESHOLD,
)
from src.experiments.run_external_conformal_shift import (         # noqa: E402
    SPLITS_JSON,
    build_texts,
)
from src.experiments.run_zeroshot_baselines import (               # noqa: E402
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
RAW_DIR = os.path.join(DATA_DIR, "external_paraphrase_raw")
PARAPHRASE_JSON = os.path.join(OUT_DIR, "external_paraphrase_set.json")

MODEL = "qwen2.5:3b-instruct"
SAMPLE_SIZE = 300
SEED = 42

# A category name fits in 48 tokens; a 370-character ticket body rewritten does
# not. This is the ONLY thing 7C changes about the 5C backend -- every guard
# (exact model-name match, RAM floor, digest recording) is inherited unchanged.
PARAPHRASE_NUM_PREDICT = 512


class ParaphraseOllamaBackend(OllamaBackend):
    """5C's backend with a generation budget big enough for a paraphrase.

    Subclassed rather than parameterised in place: run_zeroshot_baselines.py
    produced published results and its `generate` must keep emitting exactly
    the bytes it emitted then.
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
                "num_predict": PARAPHRASE_NUM_PREDICT,
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


def token_lengths(texts, embedder):
    """Subword token counts under the PRODUCTION ENCODER'S OWN tokenizer.

    "Length in tokens" is ambiguous without naming a tokenizer, and the one
    that matters here is BGE's, because that is what actually sees the text.
    Returns None when the tokenizer cannot be reached, so a missing reading is
    never silently reported as zero.
    """
    tok = getattr(embedder, "tokenizer", None)
    if tok is None:
        return None
    return np.array([len(tok.encode(str(t), add_special_tokens=False))
                     for t in texts], dtype=np.float64)


def word_lengths(texts):
    """Whitespace word counts -- the second, independent derivation."""
    return np.array([len(str(t).split()) for t in texts], dtype=np.float64)


def length_ratio_stats(source_texts, para_texts, embedder):
    """Paraphrase/original length ratios, pre-registered 7C secondary.

    A large length change could move coverage for BOTH tiers independently of
    vocabulary -- shorter text means fewer features fire and flatter
    probabilities -- so the manipulation is not fully described by similarity
    alone.

    Per-pair ratios are reported as mean and median; the aggregate ratio
    (total/total) is reported too, because the two answer different questions
    and a gap between them reveals a few extreme pairs driving the mean.
    """
    out = {}
    for name, fn in (("token", lambda t: token_lengths(t, embedder)),
                     ("word", lambda t: word_lengths(t))):
        src = fn(source_texts)
        par = fn(para_texts)
        if src is None or par is None:
            out[name] = None
            continue
        safe = src > 0
        ratios = np.full(len(src), np.nan)
        ratios[safe] = par[safe] / src[safe]
        finite = ratios[np.isfinite(ratios)]
        out[name] = {
            "mean_ratio": float(finite.mean()) if len(finite) else None,
            "median_ratio": float(np.median(finite)) if len(finite) else None,
            "aggregate_ratio": (float(par.sum() / src.sum())
                                if src.sum() > 0 else None),
            "mean_source_length": float(src.mean()),
            "mean_paraphrase_length": float(par.mean()),
            "n_zero_length_sources": int((~safe).sum()),
        }
    return out


def resident_cpu_mb(model):
    """System RAM Ollama ALREADY holds for `model`, from /api/ps.

    Returns None when the answer cannot be read -- never 0.0, because
    "could not check" and "checked and nothing is loaded" must not look the
    same. Only the non-VRAM part counts: a model in GPU memory does not
    relieve system-RAM pressure.
    """
    import urllib.request

    try:
        with urllib.request.urlopen(OLLAMA_URL + "/api/ps", timeout=10) as r:
            blob = json.loads(r.read().decode("utf-8"))
    except Exception:  # noqa: BLE001
        return None

    entries = blob.get("models")
    if entries is None:
        return None

    wanted = (model, model + ":latest")
    for entry in entries:
        if entry.get("name", "") in wanted:
            total = float(entry.get("size", 0))
            vram = float(entry.get("size_vram", 0))
            return max(0.0, (total - vram)) / 1048576.0
    return 0.0


def require_ram_accounting_for_resident(model):
    """The 5C floor check, corrected for a model that is ALREADY LOADED.

    WHY THIS EXISTS, and why it is not --allow-low-ram. 5C's
    require_free_ram() compares AVAILABLE ram against a floor sized for
    LOADING the weights. Once Ollama is already holding the model, that floor
    double-counts: it demands headroom to load something that is loaded, and
    refuses a run that would never swap. Measured here: 2,275 MB available
    with 2,064 MB already resident, against a 3,277 MB floor -- a refusal with
    4,339 MB genuinely in play.

    The fix is to make the CHECK CORRECT, not to bypass it. --allow-low-ram
    would have proceeded while stamping every response `low_ram_override`,
    permanently marking these paraphrases as produced under an untrustworthy
    configuration -- which would be false, and unrecoverable without
    regenerating them.

    The conservative direction is preserved throughout: an unreadable
    residency reading counts as zero resident, so the check falls back to 5C's
    strict behaviour rather than passing on an assumption.
    """
    from src.experiments.run_zeroshot_baselines import (
        DEFAULT_RAM_FLOOR_MB, MODEL_RAM_FLOOR_MB, available_ram_mb,
    )

    floor = MODEL_RAM_FLOOR_MB.get(model, DEFAULT_RAM_FLOOR_MB)
    avail = available_ram_mb()
    if avail is None:
        _fatal("Could not read free physical memory, so this run cannot be "
               "shown to fit in RAM. Refusing to start.")

    resident = resident_cpu_mb(model)
    resident_known = resident is not None
    effective = float(avail) + (resident if resident_known else 0.0)

    print("  free RAM        : {a} MB".format(a=avail))
    print("  already resident: {r} (Ollama /api/ps)".format(
        r=("{v:.0f} MB".format(v=resident) if resident_known
           else "UNREADABLE -- counted as 0")))
    print("  effective       : {e:.0f} MB against a {f} MB floor".format(
        e=effective, f=floor))

    if effective < floor:
        _fatal(
            "Not enough RAM for {m}: {a} MB free plus {r:.0f} MB already "
            "resident = {e:.0f} MB, against a {f} MB floor.\n"
            "  Refusing to start rather than swap.\n"
            "  Close some applications and re-run. Do NOT pass "
            "--allow-low-ram: it would stamp every paraphrase as produced "
            "under an untrustworthy configuration.".format(
                m=model, a=avail, r=(resident or 0.0), e=effective, f=floor))

    print("  [ok] fits without swapping")
    return {
        "available_mb": int(avail),
        "resident_cpu_mb": (float(resident) if resident_known else None),
        "resident_readable": bool(resident_known),
        "effective_mb": float(effective),
        "floor_mb": int(floor),
        "check": "corrected_for_resident_model",
    }


def build_paraphrase_prompt(text):
    """Plain-register rewrite, meaning and subject matter preserved.

    The instruction targets SURFACE VOCABULARY specifically, because that is
    the mechanism under test: a rewrite that keeps the technical nouns would
    leave TF-IDF's features largely intact and the experiment would measure
    nothing. It is also told NOT to add information, because a paraphrase that
    invents detail changes the label's correctness rather than its wording.
    """
    return (
        "Rewrite this customer support message in plain, everyday English, as "
        "an ordinary non-technical person would say it out loud to a friend.\n"
        "\n"
        "RULES:\n"
        "  - Keep the meaning and the subject matter exactly the same.\n"
        "  - Replace technical and formal wording with ordinary words.\n"
        "  - Do not add any information that is not already there.\n"
        "  - Do not add a greeting, a sign-off, or any commentary.\n"
        "  - Keep it roughly the same length.\n"
        "\n"
        "MESSAGE:\n"
        "{text}\n"
        "\n"
        'Reply with JSON only, in the form {{"paraphrase": "..."}}.'
    ).format(text=text)


def parse_paraphrase(raw):
    """Pull the paraphrase out of the model's JSON. Returns None on failure.

    Returning None rather than raising is deliberate: an unparseable response
    is a DATA POINT with its own pre-registered count, not a crash.
    """
    if not raw:
        return None
    try:
        blob = json.loads(raw)
    except Exception:  # noqa: BLE001
        return None
    if not isinstance(blob, dict):
        return None
    value = blob.get("paraphrase")
    if not isinstance(value, str):
        return None
    value = value.strip()
    return value or None


def cache_path(ticket_id):
    return os.path.join(RAW_DIR, "paraphrase__{m}__{t}.json".format(
        m=_slug(MODEL), t=ticket_id))


def stratified_sample(labels, size, seed=SEED):
    """Proportional allocation by queue, every queue represented.

    max(1, round(...)) rather than a plain round: a queue that rounds to zero
    would silently vanish from the sample, and the whole point of stratifying
    is that it does not.
    """
    rng = np.random.default_rng(seed)
    labels = np.asarray(labels)
    total = len(labels)
    picked = []
    for value in sorted(set(labels.tolist())):
        pos = np.flatnonzero(labels == value)
        take = max(1, int(round(size * len(pos) / total)))
        take = min(take, len(pos))
        picked.extend(rng.choice(pos, size=take, replace=False).tolist())
    return np.array(sorted(picked))


def run(limit=None, force=False, cached_only=False):
    _banner("PHASE 7C step 1 - PARAPHRASE EXTERNAL TICKETS INTO PLAIN REGISTER")
    print("Zero Gemini calls. Local Ollama only.")
    print("Measurement only: conformal.enabled={c}, drift.enabled={d}".format(
        c=settings.conformal.enabled, d=settings.drift.enabled))
    if settings.conformal.enabled or settings.drift.enabled:
        _fatal("This phase is measurement only; both flags must stay False.")

    dry = limit is not None
    if cached_only:
        print("CACHED-ONLY: no generation. Every ticket must already be in "
              "the cache,")
        print("and every cached prompt hash must match the rebuilt prompt.")
    if os.path.isfile(PARAPHRASE_JSON) and not force and not dry:
        _fatal("Refusing to overwrite an existing result:\n  {p}\n"
               "Pass --force if you mean to replace it.".format(
                   p=PARAPHRASE_JSON))

    for path in (ENGLISH_CSV, EMB_NPY, SPLITS_JSON):
        if not os.path.isfile(path):
            _fatal("Missing {p}\n  Phase 7B must have run first.".format(
                p=os.path.relpath(path, PROJECT_ROOT)))

    # ---- Step 1: the sample --------------------------------------------- #
    _banner("STEP 1 - Sampling from 7B's version-400 test arm")
    df = pd.read_csv(ENGLISH_CSV, low_memory=False)
    with open(SPLITS_JSON, encoding="utf-8") as fh:
        manifest = json.load(fh)

    test_idx = np.asarray(manifest["test_row_indices"], dtype=np.int64)
    texts = build_texts(df)
    labels = df["queue"].astype(str).to_numpy()
    test_labels = labels[test_idx]

    sample_pos = stratified_sample(test_labels, SAMPLE_SIZE)
    sample_idx = test_idx[sample_pos]
    print("  test arm {t} -> sample {n} (seed {s}, proportional by queue)"
          .format(t=len(test_idx), n=len(sample_idx), s=SEED))
    for queue in sorted(set(test_labels.tolist())):
        print("    {q:34} {n:>4}".format(
            q=queue[:34], n=int(np.sum(labels[sample_idx] == queue))))

    if dry:
        sample_idx = sample_idx[:limit]
        print("\n  DRY RUN: {n} ticket(s) only. Nothing is written except the "
              "raw cache.".format(n=len(sample_idx)))

    # ---- Step 2: the backend, constructed LAZILY ------------------------- #
    # Deliberately not built up front. A fully cached re-score would otherwise
    # demand RAM headroom to load a model it is never going to call, which
    # makes the cheap, free path fail for a reason that does not apply to it.
    _banner("STEP 2 - Local backend (constructed only if a live call is "
            "needed)")
    print("  num_predict {n} (5C uses 48; a paraphrase does not fit in that)"
          .format(n=PARAPHRASE_NUM_PREDICT))

    state = {"backend": None, "ram": None}

    def get_backend():
        if state["backend"] is None:
            state["ram"] = require_ram_accounting_for_resident(MODEL)
            # allow_low_ram=True disables only the backend's OWN floor check,
            # which the corrected check above has already superseded. Nothing
            # is stamped low_ram_override: the real RAM facts go into every
            # cached record instead.
            backend = ParaphraseOllamaBackend(MODEL, allow_low_ram=True)
            print("  model {m} | digest {d}".format(
                m=MODEL, d=(backend.model_digest or "?")[:16]))
            state["backend"] = backend
        return state["backend"]

    # ---- Step 3: generate ------------------------------------------------ #
    _banner("STEP 3 - Paraphrasing ({n} tickets)".format(n=len(sample_idx)))
    # write_cache() creates 5C's RAW_CACHE_DIR, not ours -- it takes an
    # explicit path but makes only the directory it knows about. Ours is
    # created here so a first run does not lose its first generation.
    os.makedirs(RAW_DIR, exist_ok=True)

    records = []
    n_live = 0
    n_cached = 0
    cached_digests = set()
    cache_misses = []
    cache_hash_mismatches = []
    t0 = time.perf_counter()

    for i, row_idx in enumerate(sample_idx):
        ticket_id = int(row_idx)
        source = texts[ticket_id]
        prompt = build_paraphrase_prompt(source)
        phash = prompt_hash(prompt)
        path = cache_path(ticket_id)

        cached = read_cache(path, phash)
        if cached is None and cached_only:
            # Distinguish the two ways a cache read fails. read_cache() returns
            # None for both, and in a normal run both simply trigger a
            # regeneration -- but in a cached-only pass a HASH MISMATCH means
            # the cached answer was written for a different prompt, which is a
            # correctness problem, not a missing file.
            if os.path.isfile(path):
                cache_hash_mismatches.append(ticket_id)
            else:
                cache_misses.append(ticket_id)
            continue

        if cached is not None:
            raw = cached.get("response")
            elapsed = cached.get("elapsed_s")
            digest = cached.get("model_digest")
            if digest:
                cached_digests.add(digest)
            n_cached += 1
        else:
            backend = get_backend()
            call_t0 = time.perf_counter()
            try:
                raw = backend.generate(prompt)
            except RuntimeError as exc:
                print("    [error] ticket {t}: {e}".format(t=ticket_id, e=exc))
                raw = None
            elapsed = time.perf_counter() - call_t0
            n_live += 1
            write_cache(path, {
                "phase": "7C",
                "backend": "ollama",
                "model": MODEL,
                "model_digest": backend.model_digest,
                "row_index": ticket_id,
                "queue": str(labels[ticket_id]),
                "prompt_sha256": phash,
                "response": raw,
                "elapsed_s": elapsed,
                "num_predict": PARAPHRASE_NUM_PREDICT,
                "seed": SEED,
                "ram": state["ram"],
            })

        paraphrase = parse_paraphrase(raw)
        records.append({
            "row_index": ticket_id,
            "queue": str(labels[ticket_id]),
            "source": source,
            "paraphrase": paraphrase,
            "parse_ok": paraphrase is not None,
            "elapsed_s": elapsed,
            "from_cache": cached is not None,
        })

        if dry:
            print("\n  ---- ticket {t}  [{q}] ----".format(
                t=ticket_id, q=labels[ticket_id]))
            print("  SOURCE:\n    " + source.replace("\\n", " ")[:600])
            print("  PARAPHRASE:\n    "
                  + (paraphrase or "(UNPARSEABLE)")[:600])
            print("  elapsed {e:.1f}s".format(e=elapsed or 0.0))
        elif (i + 1) % 10 == 0 or i + 1 == len(sample_idx):
            rate = (i + 1) / max(time.perf_counter() - t0, 1e-9)
            left = (len(sample_idx) - i - 1) / rate / 60.0
            print("    {d}/{n}  {r:.2f}/s  ~{m:.0f} min left".format(
                d=i + 1, n=len(sample_idx), r=rate, m=left), flush=True)

    if cached_only:
        if cache_hash_mismatches:
            _fatal(
                "{n} cached record(s) carry a prompt_sha256 that does not "
                "match the rebuilt prompt: {t}\n"
                "  Those answers were produced for a DIFFERENT prompt and "
                "must not be scored.\n"
                "  Re-run without --cached-only to regenerate them.".format(
                    n=len(cache_hash_mismatches),
                    t=cache_hash_mismatches[:10]))
        if cache_misses:
            _fatal(
                "{n} ticket(s) are not in the cache: {t}\n"
                "  --cached-only forbids generating them. Re-run without it."
                .format(n=len(cache_misses), t=cache_misses[:10]))
        print("  [ok] {n}/{t} cached records verified: every prompt_sha256 "
              "matches the rebuilt prompt".format(n=n_cached,
                                                  t=len(sample_idx)))
        if n_live:
            _fatal("--cached-only made {n} live call(s); this is a bug."
                   .format(n=n_live))

    total_s = time.perf_counter() - t0
    live = [r for r in records if not r["from_cache"] and r["elapsed_s"]]
    median_s = (float(np.median([r["elapsed_s"] for r in live]))
                if live else None)
    print("\n  {l} live call(s), {c} from cache, {t:.1f}s total".format(
        l=n_live, c=n_cached, t=total_s))
    if median_s is not None:
        print("  median live call {m:.1f}s  ->  full {n}-ticket run would be "
              "~{h:.1f}h".format(m=median_s, n=SAMPLE_SIZE,
                                 h=median_s * SAMPLE_SIZE / 3600.0))

    if dry:
        _banner("DRY RUN COMPLETE - nothing written but the raw cache")
        print("Review the paraphrases above before the full run.")
        return records

    # ---- Step 4: guards -------------------------------------------------- #
    _banner("STEP 4 - Guards")
    # One set of weights must have produced the whole set. Two digests would
    # mean the cache spans a model change, which makes the arm a mixture of
    # two generators rather than one.
    if len(cached_digests) > 1:
        _fatal("The cache spans {n} different model digests: {d}\n"
               "  These paraphrases were not all produced by the same "
               "weights.".format(n=len(cached_digests),
                                 d=sorted(cached_digests)))
    cached_digest = next(iter(cached_digests), None)
    n_unparseable = sum(1 for r in records if not r["parse_ok"])
    usable = [r for r in records if r["parse_ok"]]
    print("  unparseable or empty: {n} / {t}".format(
        n=n_unparseable, t=len(records)))

    from src.agent.artifacts import load_artifacts
    artifacts = load_artifacts(require_gemini=False)

    src_emb = np.load(EMB_NPY)[[r["row_index"] for r in usable]]
    par_emb = np.asarray(artifacts.embedder.encode(
        [r["paraphrase"] for r in usable], convert_to_numpy=True,
        batch_size=32, show_progress_bar=False), dtype=np.float32)

    a = src_emb / np.linalg.norm(src_emb, axis=1, keepdims=True)
    b = par_emb / np.linalg.norm(par_emb, axis=1, keepdims=True)
    sims = np.sum(a * b, axis=1)

    too_similar = sims >= NEAR_DUP_THRESHOLD
    for rec, sim in zip(usable, sims):
        rec["bge_similarity_to_source"] = float(sim)
    survivors = [r for r, keep in zip(usable, ~too_similar) if keep]

    # Rule 6: the discard count two independent ways.
    n_discarded_mask = int(too_similar.sum())
    n_discarded_arith = len(records) - len(survivors) - n_unparseable
    if n_discarded_mask != n_discarded_arith:
        _fatal("Discard counts disagree: mask {a} vs arithmetic {b}".format(
            a=n_discarded_mask, b=n_discarded_arith))

    print("  not rewritten (BGE >= {h}): {n} / {t}  [two derivations agree]"
          .format(h=NEAR_DUP_THRESHOLD, n=n_discarded_mask, t=len(usable)))
    print("  mean source-paraphrase BGE similarity: {m:.4f} "
          "(median {d:.4f}, p05 {p:.4f})".format(
              m=float(sims.mean()), d=float(np.median(sims)),
              p=float(np.percentile(sims, 5))))
    lengths = length_ratio_stats([r["source"] for r in survivors],
                                 [r["paraphrase"] for r in survivors],
                                 artifacts.embedder)
    for space in ("token", "word"):
        st = lengths.get(space)
        if st is None:
            print("  length ({s}): UNREADABLE".format(s=space))
            continue
        print("  length ratio ({s}): mean {m:.4f}, median {d:.4f}, "
              "aggregate {a:.4f}  ({x:.1f} -> {y:.1f})".format(
                  s=space, m=st["mean_ratio"], d=st["median_ratio"],
                  a=st["aggregate_ratio"], x=st["mean_source_length"],
                  y=st["mean_paraphrase_length"]))

    print("  SURVIVING PAIRED SET: {n}".format(n=len(survivors)))

    if len(survivors) < 150:
        print("\n  [!] Below the pre-registered floor of 150. The scoring "
              "script will report")
        print("      `insufficient_n_after_guards` and NO RESOLUTION rather "
              "than a noisy estimate.")

    # ---- Step 5: write --------------------------------------------------- #
    _banner("STEP 5 - Writing the paired set")
    blob = {
        "phase": "7C",
        "measurement_only": True,
        "is_real_production_data": False,
        "model": MODEL,
        # From the live backend when one was needed, otherwise from the cache
        # records -- so a fully cached run still records WHICH weights
        # produced these paraphrases rather than leaving it blank.
        "model_digest": (state["backend"].model_digest
                         if state["backend"] is not None else cached_digest),
        "ram": state["ram"],
        "num_predict": PARAPHRASE_NUM_PREDICT,
        "seed": SEED,
        "config_fingerprint": config_fingerprint(),
        "sample_size_requested": SAMPLE_SIZE,
        "n_sampled": len(records),
        "n_unparseable": n_unparseable,
        "n_discarded_not_rewritten": n_discarded_mask,
        "n_surviving": len(survivors),
        "near_duplicate_threshold": NEAR_DUP_THRESHOLD,
        "mean_bge_similarity_to_source": float(sims.mean()),
        "median_bge_similarity_to_source": float(np.median(sims)),
        "p05_bge_similarity_to_source": float(np.percentile(sims, 5)),
        "length_ratios": lengths,
        "median_live_call_s": median_s,
        "pairs": [
            {
                "row_index": r["row_index"],
                "queue": r["queue"],
                "source": r["source"],
                "paraphrase": r["paraphrase"],
                "bge_similarity_to_source": r["bge_similarity_to_source"],
            }
            for r in survivors
        ],
    }
    with open(PARAPHRASE_JSON, "w", encoding="utf-8") as fh:
        json.dump(blob, fh, ensure_ascii=False, indent=1)
    print("[write] {p}  ({n} pairs)".format(
        p=os.path.relpath(PARAPHRASE_JSON, PROJECT_ROOT), n=len(survivors)))

    _banner("DONE - step 1 complete")
    print("Next: python src/experiments/run_paraphrase_shift_conformal.py")
    return survivors


def parse_args(argv):
    parser = argparse.ArgumentParser(
        description="Phase 7C step 1 -- paraphrase external tickets.")
    parser.add_argument("--limit", type=int, default=None,
                        help="dry run on the first N sampled tickets")
    parser.add_argument("--force", action="store_true",
                        help="overwrite the existing paired set")
    parser.add_argument("--cached-only", action="store_true",
                        dest="cached_only",
                        help="score from the cache and refuse any live call")
    return parser.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    run(limit=args.limit, force=args.force, cached_only=args.cached_only)
    return 0


if __name__ == "__main__":
    sys.exit(main())
