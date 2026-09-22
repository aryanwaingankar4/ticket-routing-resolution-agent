# AI-Powered Intelligent Ticket Routing & Resolution Agent

A three-layer AI system for IT support ticket triage: **classify** the ticket
into the right team, **retrieve** similar past tickets and suggest a grounded
resolution, **flag recurring issues for automation**, and **escalate to a
human** whenever the system isn't confident enough to act on its own. Built
as a B.Tech final-year AI/ML project, based on a nasscom hackathon use-case
brief, with an ongoing extension toward an IEEE-style research paper.

---

## Architecture

1. **Classification Layer** — "which team should this ticket go to?"
   Categories: Infrastructure, Application, Security, Database, Storage,
   Network, Access Management (7 total).
2. **RAG Layer** — "how was this solved before?" Retrieves similar past
   tickets via FAISS and asks Gemini to draft a grounded resolution.
3. **Agentic Layer** — two parts:
   - **Confidence-based escalation** — escalates to a human whenever
     confidence is too low, at two separate decision points
     (classification and retrieval), rather than letting the system guess.
   - **Automation-flagging** — separately, clusters *resolved* tickets by
     resolution similarity to surface recurring issues worth turning into
     self-service automation.
4. **Orchestration Layer** (Phase 3) — the three stages are independent
   agents with declared dependencies (`src/agent/agents.py`), coordinated by
   an orchestrator that owns every routing decision
   (`src/agent/orchestrator.py`), each individually addressable over HTTP
   (`src/service/api.py`). The agents are independently *addressable*, not
   independently *running*: `/triage` orchestrates in-process. See "Phase 3 —
   the agent architecture".

**Tech stack:** Python, scikit-learn, sentence-transformers, FAISS, Gemini
API (`gemini-flash-lite-latest`, via the `google-genai` SDK), FastAPI +
uvicorn, Streamlit, pandas/numpy.

**Repo:** `aryanwaingankar4/ticket-routing-resolution-agent`

---

## The Core Idea

The throughline across this whole project: **every escalation or automation
decision is governed by a calibrated confidence signal, measured against
real data — never assumed.** This shows up independently in three separate
places:

- The RAG layer's similarity threshold (**0.67**, BGE embeddings) — below
  this, the LLM call is skipped and the ticket goes to a human instead.
  Derived by intersecting the safe range required by the 9-ticket
  adversarial escalation set with the value in that range that minimizes
  leakage against a dedicated 45-ticket out-of-domain calibration set (see
  "RAG Similarity Threshold Recalibration" below for the full story).
- The cascade classifier's confidence threshold (**0.50** at a realistic
  70–80% target-accuracy bar) — below this, classification escalates from
  the cheap TF-IDF model to the stronger embeddings-based model.
- The resolution-clustering automation-flagging threshold (**0.80**) — a
  calibrated cliff-edge finding, chosen because a false "these two tickets
  share a fix" claim is costlier than a missed automation opportunity.

A fourth strand now sits on top of these: **conformal prediction**, which
upgrades the claim from *calibrated* to *provably risk-controlled* — and,
in the process, measures where that guarantee silently fails. See
"Conformal Prediction" below.

That consistency — three independent calibration exercises converging on
the same design philosophy — is the actual research contribution, more so
than any single accuracy number. A recurring pattern across all three is
just as important as a finding in its own right: **naive calibration
methods have repeatedly needed a second, harder look** — an in-distribution
split that hid a real error, a 35-ticket calibration set too sparse to
trust, and (most recently) a 175-ticket in-domain-only calibration set
whose cliff-edge turned out to be a low-N artifact that even survived the
initial addition of negative-class data. Every one of these failures was
caught by insisting on evidence over assumption, not by getting it right
the first time.

---

## What Was Built, and What Was Learned

### Dataset

`data/generate_dataset.py` generates synthetic IT support tickets. Each
"scenario" is a linked `(title_template, symptom_phrase, resolution_text)`
tuple so fields stay semantically consistent, and one shared per-ticket
context dict resolves all placeholders (`{app}`, `{srv}`, `{db}`, etc.)
exactly once so entities never disagree across fields. Fixed random seed
(`42`) throughout for reproducibility. The dataset was scaled from an
initial 1,000 tickets to a final **4,000 tickets** (~571–572 per category,
9 scenario templates per category).

### The 14-ticket generalization benchmark

Early on, a TF-IDF + Logistic Regression baseline hit 100% in-distribution
accuracy — a red flag, not a success, on template-generated data prone to
lexical memorization. To get an honest read on real generalization, a
**14-ticket benchmark** was built by hand: 2 tickets per category, written
in plain, non-technical, everyday language that never appears in any
training template. This became the single fixed reference point for the
whole project — defined once in `src/classification/generalization_test.py`
as `NOVEL_TICKETS`, treated as read-only for the entire project, and reused
verbatim everywhere else.

Results against it:

| Method | In-Distribution Accuracy | 14-Ticket Generalization |
|---|---|---|
| TF-IDF + Logistic Regression | 100.0% | 7/14 (50.0%) |
| Frozen MiniLM embeddings + Logistic Regression | 100.0% | 10/14 (71.4%) |
| Fine-tuned DistilBERT (best epoch) | 100.0% | 7/14 (50.0%) at 1,000 tickets; 7/14 (tie, epoch-1 best) after re-testing at 4,000 tickets |

DistilBERT showed a classic overfitting signature at every dataset size
tested — 100% in-distribution accuracy within 1–2 epochs while
generalization plateaued or declined. Re-running the MiniLM benchmark after
scaling the dataset 1,000 → 4,000 tickets produced the *exact same* score
(10/14, same 4 tickets wrong) — proof the ceiling was representation-limited
(the frozen embedding model's own resolution), not data-volume-limited.
MiniLM was selected for production ahead of the later BGE swap (see
"Embedding Model: MiniLM → BGE" below) on measured generalization
performance, not in-distribution accuracy — the same standard later applied
to the BGE decision.

### RAG layer (retrieval + Gemini-grounded resolutions)

`build_vector_index.py` builds a FAISS `IndexFlatIP` index over
L2-normalized embeddings (cosine similarity via inner product), with an
aligned metadata JSON and a built-in sanity check (querying the index with
a ticket's own embedding must retrieve itself at similarity ~1.0).
Production artifacts are model-aware and filename-suffixed
(`ticket_index_bge-base-en-v1-5.faiss`,
`ticket_metadata_bge-base-en-v1-5.json`) after a silent-stale-cache risk
was caught during the BGE swap — every loading function in this project
now checks `index.ntotal == len(metadata)` and fails loudly if they're out
of sync, rather than silently returning misaligned results.

`suggest_resolution.py` retrieves the top-5 similar past tickets and, if
the best match clears `SIMILARITY_THRESHOLD`, asks Gemini to draft a
resolution grounded in those retrieved tickets. Below that threshold, the
LLM call is skipped entirely and the ticket is escalated to a human. Gemini
never decides the ticket's category — classification is handled entirely
by the trained models above; Gemini's role is strictly resolution
generation.

(Uses the current `google-genai` SDK — `from google import genai` — not the
deprecated `google-generativeai` package, and `gemini-flash-lite-latest`
after the originally-used model was retired for new users.)

### Embedding Model: MiniLM → BGE

Production classification, RAG retrieval, and cascade Tier-2 were swapped
from `all-MiniLM-L6-v2` (384-dim) to `BAAI/bge-base-en-v1.5` (768-dim), on
the strength of the validated 45-ticket benchmark result (33/45 = 73.3% vs
MiniLM's 32/45 = 71.1% — see "Final Classification Comparison" below).
Verified correct via two independently-trained BGE classifiers matching
exactly on the 45-ticket benchmark.

**Real bug caught during the swap, and its consequence:** BGE's cosine
similarity scores run systematically higher than MiniLM's for the same
ticket pairs. The RAG similarity threshold, originally tuned for MiniLM at
0.35, did not transfer — a maximally-unrelated "weather" test ticket in the
live Streamlit demo scored 57% top similarity under BGE, well above the old
0.35 cutoff, meaning the human-escalation gate would have silently failed
to fire. `test_adversarial_escalation.py` initially still passed (3/9 at
the stale threshold, later diagnosed) because it turned out to be loading
its own hardcoded, un-migrated MiniLM constants (embedding model name,
FAISS index path, metadata path, classifier path) independent of the
rest of the BGE-updated pipeline — the same class of bug as the
silent-stale-cache risk above. Both were fixed together; see "RAG
Similarity Threshold Recalibration" below for the full threshold story
that followed.

### Cascade classifier — confidence-based tier routing

`train_cascade.py` implements a two-tier cascade: Tier-1 (cheap TF-IDF)
resolves a ticket directly if confident enough; otherwise it escalates to
Tier-2 (embedding-based). This mirrors the RAG layer's "escalate when
unsure" philosophy, applied one layer down at model-selection time. The
cascade confidence threshold (0.50) was verified, via code review, to
depend only on Tier-1/TF-IDF confidence — not on which embedding model
Tier-2 uses — so it did not need re-calibration when the embedding model
was swapped from MiniLM to BGE.

Calibrating the confidence threshold took three attempts, and the failures
were as informative as the eventual success:

1. **In-distribution held-out split** — rejected. Showed 100% accuracy in
   every confidence bucket, producing a threshold that looked trustworthy
   but missed a confidently-wrong real-world prediction entirely.
2. **35 hand-written calibration tickets** — rejected. Too sparse (34/35
   collapsed into one bucket), so the resulting threshold was driven by
   statistical noise rather than a real signal.
3. **175 Gemini-paraphrased calibration tickets** (25/category, independent
   of the 14-ticket benchmark, with a self-consistency label check — 21/175
   flagged for review, kept anyway since ground truth was preserved) —
   adopted. Dense enough to correctly reveal Tier-1's real overconfidence.

A threshold-derivation logic bug was also caught and fixed along the way:
the scan was breaking on the first small/noisy bucket it failed, before
ever checking better buckets further down.

Final accuracy/efficiency tradeoff, swept across target-reliability bars:

| Target Acc | Threshold | Novel Tier-1 Resolution Rate | Novel Accuracy |
|---:|---:|---:|---:|
| 90% | 1.00 | 0.0% | 71.4% |
| 80% | 0.50 | 21.4% | 71.4% |
| 70% | 0.50 | 21.4% | 71.4% |

At a strict 90% bar, no benefit exists — the cascade collapses to pure
Tier-2. At a relaxed 70–80% bar, a real threshold (0.50) emerges, routing
~21% of real-world tickets through the cheap tier at zero accuracy cost.
The contribution isn't a big accuracy jump — it's a validated calibration
methodology that catches what naive calibration would silently miss.

### RAG Similarity Threshold Recalibration

After the MiniLM → BGE swap, the RAG similarity threshold was set to a
**provisional 0.65** (re-verified 9/9 on the adversarial escalation set
once the stale-constants bug above was fixed), explicitly documented
in-code as provisional pending full recalibration. Recalibrating it
properly turned into its own multi-stage investigation.

**Attempt 1 — reuse the existing 175-ticket in-domain calibration set.**
The same set used to calibrate the cascade confidence threshold was fed
through a per-ticket (not pairwise) precision/recall sweep, adapting the
corrected `find_cliff_edge()` logic already validated in the category-
specific resolution-clustering script. This *ran* cleanly, but the result
exposed a methodology problem, not a code problem: every threshold from
0.35 through 0.65 produced identical results (all 175 tickets "proceed"),
because the 175-ticket set was built entirely from legitimate, in-domain
support tickets — it contains no out-of-domain examples analogous to the
weather/pizza tickets in the 9-ticket adversarial set. The set can measure
"does the top-1 retrieval have the right category" but not "does the gate
correctly reject a genuinely irrelevant ticket," which is the actual job
this threshold exists to do. The mechanical cliff-edge it *did* find, T =
0.85, was a low-N artifact (only 2/175 tickets proceed there — precision
= 1.0 is trivially easy to hit when almost nothing is being tested) and
was explicitly **not** adopted. This is a citable finding in its own
right: an in-domain-only calibration set can produce an unrepresentative
threshold cliff-edge when the underlying gate's actual job is detecting
out-of-domain inputs.

**Attempt 2 — build a genuine 45-ticket out-of-domain (OOD) calibration
set.** `src/classification/generate_ood_calibration_set.py` hand-writes 15
seed scenarios across 4 buckets (weather/personal errands, food, non-IT
departmental questions like HR/Finance/Facilities, and vague non-actionable
"it's broken" messages — deliberately broader than the original 9
adversarial tickets, to avoid near-duplicate clustering that would
overstate how cleanly separated the resulting threshold looks), then
Gemini-paraphrases each into 3 independent variants (temperature 0.9, JSON
schema pinned where the installed SDK supports it) for 45 total tickets,
written to `data/ood_calibration_tickets.json`. `calibrate_rag_similarity_
threshold.py` was extended to fold these in as real negative-class signal:
an OOD ticket proceeding past the gate counts as a false positive; an OOD
ticket correctly escalating counts as a true negative — combined with the
existing in-domain confusion matrix to drive one combined precision/
recall/F1 sweep, while reporting the OOD-only "leakage rate" separately at
every threshold as its own diagnostic.

**A second citable finding: adding OOD data did not, by itself, fix the
low-N artifact.** The mechanical cliff-edge on the *combined* sweep still
landed at T = 0.85 — because the 45 OOD tickets' top-1 similarities range
only 0.5051–0.7249, never reaching 0.80 or 0.85. At those thresholds the
OOD set contributes zero information; the artifact from Attempt 1 survived
untouched. A negative class only helps a cliff-edge search where its own
similarity distribution actually overlaps the range being swept.

**Two hardening checks added alongside this extension, given the
project's history of stale-constant and silent-mismatch bugs:**
- **Self-retrieval contamination check** — since the 175 in-domain
  calibration tickets are paraphrases of tickets that are themselves in
  the FAISS index, each one's top-1 retrieval could simply be its own
  source ticket, inflating the in-domain precision curve. Measured
  directly: **5.7% (10/175)** of in-domain calibration tickets retrieve
  their own source ticket as the top-1 match. Not dominant, but real —
  the in-domain precision numbers are somewhat inflated, and this
  threshold's final derivation deliberately did not rely on that curve.
- **Exact-source-match diagnostic** — a strictly stronger relevance
  signal (`retrieved.id == calibration_ticket.id`) computed and reported
  side by side with the original category-agreement proxy, for
  comparison only; the category-agreement proxy remains the metric
  actually feeding the cliff-edge, per the original script's documented
  approach, since switching the primary metric is a methodology decision
  that shouldn't happen silently inside a script.

**The final threshold was derived by intersecting two independent, real
constraints rather than trusting either mechanical sweep alone:**

1. The 9-ticket adversarial set fixes a hard safe range. Real similarity
   values from a live pipeline run: tickets that must escalate top out at
   0.6196; tickets that must proceed bottom out at 0.6704. Any threshold
   in **(0.6196, 0.6704]** passes all 9 adversarial tickets by
   construction.
2. Within that range, a finer-grained sweep (0.61–0.69 in 0.01 steps)
   shows OOD leakage rate increasing monotonically as the threshold
   drops, with in-domain recall (category proxy) flat at 100% throughout
   — so there's no real tradeoff inside the safe range, only a clear
   minimum: **T = 0.67**, at 13.3% OOD leakage vs. 37.8% at the prior
   provisional 0.65.

**T = 0.67 was confirmed 9/9 on a live re-run of the adversarial
escalation test** before being adopted into production, replacing the
provisional 0.65. Full per-threshold sweep data (both the combined metric
and the exact-source diagnostic) is in
[`rag_similarity_calibration_combined.csv`](data/rag_similarity_calibration_combined.csv);
the original in-domain-only sweep (Attempt 1, kept for reference, not
overwritten) is in
[`rag_similarity_calibration.csv`](data/rag_similarity_calibration.csv).

### Cascade Confidence Calibration

Reliability diagrams (predicted confidence vs. observed accuracy) for both
cascade tiers, evaluated against the real 500-ticket production batch.
Both tiers score 100% accuracy in-distribution but never reach full
confidence (Tier-1 tops out ~0.92, Tier-2 ~0.94) — the models are
**underconfident** rather than overconfident, a safer failure mode for a
system designed around escalation thresholds.

| Tier-1 (TF-IDF) — ECE = 0.1122 | Tier-2 (embedding-based) — ECE = 0.0992 |
|---|---|
| ![Tier-1 Calibration](data/calibration_tier1_reliability_diagram.png) | ![Tier-2 Calibration](data/calibration_tier2_reliability_diagram.png) |

Bin-level data: [`calibration_reliability_data.csv`](data/calibration_reliability_data.csv)

### Ablation Study — What the Safety Nets Are Actually Worth

Both cascade and RAG escalation thresholds were disabled independently
to measure their real cost/benefit, rather than assuming they help:

| Mode | 45-Ticket Accuracy | Adversarial Escalation |
|---|---:|---:|
| **Baseline** (both thresholds active) | 71.11% (32/45) | 9/9 correctly escalated |
| **No cascade** (Tier-1 only, threshold=0) | 35.56% (16/45) | — |
| **Tier-2 only** (BGE answers everything) | 73.33% (33/45) | — |
| **No RAG gate** (pretend threshold=0) | — | 9/9 would attempt a resolution; 6/9 should have escalated |

**Cascade threshold (0.50) — read this row carefully.** Baseline minus
no-cascade is 35.6 points, and that number is real, but it is **not the
value of cascading**. `no-cascade` is TF-IDF answering *every* ticket, so
the comparison is BGE-versus-TF-IDF: it measures the **representation
gap**, and it would be almost as large with no cascade logic in the
system at all.

The comparison that isolates the cascade is baseline against **Tier-2
only**, and it goes the other way: **the cascade is one ticket *worse*
than simply letting BGE answer everything** (32/45 vs 33/45), a
difference an exact McNemar test cannot distinguish from zero
(p = 1.0). What the cascade actually buys is latency, not accuracy. The
full measurement, on this benchmark and on a second 175-ticket set, is
in **"Phase 5B — the honest ablation"** below.

> **Correction (Phase 5B, 2026-09-20).** This paragraph previously read
> "removing it drops classification accuracy by 35.6 percentage points —
> the cascade isn't a marginal tweak, it roughly doubles real-world
> classification accuracy versus running the cheap Tier-1 model alone."
> The arithmetic was right and the claim attached to it was wrong: it
> credited the cascade with a gap produced by the embedding model. The
> `tier2-only` row above is the control that was missing.

> **Correction (re-measured under BGE).** The numbers above were
> originally 68.89% (31/45) baseline and a 33.3-point gain. Those were
> measured on **MiniLM** and never re-run after the BGE swap:
> `run_ablation_study.py` had kept its own hardcoded `ticket_index.faiss`,
> `ticket_metadata.json`, `ticket_classifier.joblib` and
> `all-MiniLM-L6-v2`. Because those four artifacts were mutually
> consistent, the script ran without error and silently measured the old
> pipeline. The results CSV was written 2026-08-15; the BGE swap landed
> 2026-08-26, eleven days later. This was the **fourth** occurrence of
> this project's recurring stale-artifact bug class, and the first to
> reach published results — caught by the golden-parity discipline
> introduced during the `src/agent/` consolidation. The script now reads
> every one of those four values from `src/agent/config.py`, so it cannot
> drift from production independently again.
>
> Note that **no-cascade is unchanged at 16/45**, which is the expected
> result and a useful internal check: Tier-1 is TF-IDF, so a Tier-2
> embedding swap cannot affect Tier-1-only accuracy.

**RAG similarity threshold:** removing it means every one of the 9
adversarial tickets — including an off-topic weather question and a
pizza recommendation request — would now trigger a real Gemini call
attempting a resolution instead of correctly escalating. 6 of those 9
tickets genuinely needed human escalation, meaning the gate is
preventing 6 concrete instances of confidently-fabricated, wrong output
per this test set alone.

Scripts: `src/experiments/run_ablation_study.py`. Results:
[`ablation_baseline_results.csv`](data/ablation_baseline_results.csv),
[`ablation_no-cascade_results.csv`](data/ablation_no-cascade_results.csv),
[`ablation_no-rag_results.csv`](data/ablation_no-rag_results.csv).

### Conformal Prediction — from *calibrated* to *provably risk-controlled*

Every threshold above is calibrated: swept against real data and chosen for a
documented reason. None of them carries a *guarantee*. Split conformal
prediction does — for a chosen error rate α, the true category lies in the
predicted set with probability ≥ 1−α, distribution-free and finite-sample,
with no assumption that the model is well calibrated (which matters here,
since both tiers are measurably underconfident).

The catch is that the guarantee holds **only under exchangeability** between
calibration and deployment data. This project turns out to be an unusually
clean setting in which to measure what happens when that assumption fails.

Implementation is hand-rolled in `src/agent/conformal.py` (numpy only, ~150
lines). The quantile is computed as an exact order statistic — the
⌈(n+1)(1−α)⌉-th smallest calibration score — rather than via `np.quantile`,
whose interpolation silently voids the finite-sample guarantee. This is
**measurement only**: `settings.conformal.enabled` is `False`, production still
gates on 0.50 / 0.67, and Phase 0's golden parity is untouched.

#### Finding 1: coverage transfer is a property of the *representation*, not just the data

Both tiers were calibrated on the same 175-ticket in-domain set, at the same α,
and evaluated on the same 45-ticket plain-English benchmark. They behave
completely differently.

| Tier | α | Nominal | Coverage on 175 | Coverage on 45 | Gap | ±2 s.d. | Mean set size | Singleton rate |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Tier-1 (TF-IDF) | 0.20 | 0.800 | 0.806 | 0.578 | **−0.222** | 0.060 | 2.67 | 8.9% |
| Tier-1 (TF-IDF) | 0.10 | 0.900 | 0.909 | 0.667 | **−0.233** | 0.045 | 3.44 | 8.9% |
| Tier-1 (TF-IDF) | 0.05 | 0.950 | 0.960 | 0.844 | **−0.106** | 0.033 | 5.00 | 6.7% |
| Tier-2 (BGE) | 0.20 | 0.800 | 0.806 | 0.756 | −0.044 | 0.060 | 1.11 | 84.4% |
| Tier-2 (BGE) | 0.10 | 0.900 | 0.909 | **0.889** | −0.011 | 0.045 | 1.84 | 35.6% |
| Tier-2 (BGE) | 0.05 | 0.950 | 0.960 | 0.978 | +0.028 | 0.033 | 2.56 | 17.8% |

*(LAC score, marginal, all 175 labels, de-contaminated fit. The ±2 s.d. column
is the noise band from a single calibration draw — the conformal guarantee is
marginal over the calibration draw, not conditional on it, so a gap inside that
band is not evidence of anything.)*

At α = 0.10 the **lexical model loses 23.3 points of coverage** — more than
five times its noise band — while the **semantic model loses 1.1 points**,
comfortably inside it. Conformal's promise holds for BGE and collapses for
TF-IDF, on identical data.

The mechanism is straightforward once stated: TF-IDF's nonconformity scores are
built from surface vocabulary, and a paraphrase shift changes the vocabulary
entirely, so the calibration score distribution does not transfer. BGE encodes
meaning, which survives the paraphrase. **Exchangeability is not purely a
property of how the data was sampled — it is also a property of the
representation the scores are computed in.** That is a testable claim, and it
is the sharpest result in this project: it converts "BGE generalizes better,"
already established on accuracy, into a statement about whether a statistical
guarantee survives deployment at all.

It also reframes the cascade. Tier-1 at α=0.10 emits a singleton for only 8.9%
of benchmark tickets and averages 3.4 of 7 categories per set — it is, by its
own admission, unable to commit. Tier-2 at α=0.20 emits singletons for 84.4%
with a mean set size of 1.11, and at α=0.10 still manages 35.6%. The accuracy/autonomy trade-off the cascade
encodes as a hand-tuned 0.50 falls out of conformal as a consequence of the
chosen risk level.

#### Finding 2: this calibration set cannot be de-contaminated

`train_cascade.py` scores the 175 calibration tickets with models fit on all
4,000 rows — and those tickets are Gemini paraphrases of rows from that same
CSV. The model has seen the original of every ticket it is calibrated on.
Predicted consequence: memorised sources depress nonconformity scores, shrink
the quantile, shrink the sets, and push coverage below nominal.

**The prediction was wrong in practice, and the reason is more interesting than
the hypothesis.** Refitting both tiers with all 175 source rows removed moves
benchmark coverage by a mean of 0.005 across all 64 paired configurations
(maximum 0.067, in a low-sample Mondrian cell), and by nothing at three decimal
places in most. The
measurement of *why* is reproducible in the script's Step 1b:

| | |
|---|---:|
| Scenario templates in the dataset | 66 |
| Templates touched by the 175 calibration tickets | 62 (93.9%) |
| Median rows per template | 62 |
| Sibling rows left after removing one source row | 61 |
| Rows surviving template-level exclusion | **210 / 4000** |

Memorisation here is at *template* level, not row level. Deleting one row
removes roughly 1.6% of its template's evidence, so the fitted model is
effectively unchanged. The stronger move — excluding every row sharing a
template with any calibration ticket — would leave 210 rows to train on, which
is not a de-contamination but a destruction.

> **Correction (Phase 5A, 2026-09-20).** This table previously read 12
> templates, 11 touched (91.7%), a median of 430 rows per template and
> **40 / 4000** rows surviving, with the prose claiming ~0.2% of a template's
> evidence per removed row. Those numbers came from grouping rows by
> `scenario_id` alone — but `scenario_id` is an index *within* a category
> (`data/generate_dataset.py:634` says so in its own comment), so the
> diagnostic merged seven categories' templates into one. Grouped correctly by
> `(category, scenario_id)` the dataset has **66** templates of median **62**
> rows, of which the calibration set touches **62**, leaving **210 / 4000**.
>
> **Finding 2's conclusion is unchanged**, and so is every coverage number in
> this section: the de-contamination measurement excludes rows by **id**, never
> by template, so the refit results and all 128 configurations in
> `conformal_calibration_results.csv` are byte-for-byte unaffected — verified by
> re-running the script and diffing (both result CSVs identical, and the
> artifact's entire 90 KB `fits` block identical). Only this diagnostic table
> was wrong. The corrected artifact differs from the original in exactly two
> keys: `contamination_structure`, and the recorded `config_fingerprint`
> (`05f391baf27c` → `9c9a5cbcb53f`), which moved because `settings.drift` and
> its Signal B reference name were added in Phase 4 — a useful demonstration
> that the fingerprint is deliberately conservative and fires on config edits
> that cannot affect the measurement. The
> corrected run is committed beside the original as
> `conformal_calibration_corrected_bge-base-en-v1-5.json` rather than
> overwriting it. The re-run's two result CSVs came out byte-identical to the
> published ones and are therefore not committed as duplicates —
> re-`--out-suffix` the script to reproduce them. And
> `tests/test_contamination_structure.py` now pins both the corrected counts
> and the fact that a `scenario_id`-only grouping disagrees with them.
>
> This is the **sixth** instance of this project's recurring bug class — a key
> that is internally consistent, plausible, and wrong for its context — and the
> second to reach published results. It was found while auditing the project's
> own documentation, not by any test.

So: **an in-domain calibration set built by paraphrasing training rows cannot
be made exchangeable with a model trained on that data, at any amount of
cleaning, when the training data is template-redundant.** The remedy is not to
filter the calibration set; it is to calibrate on data drawn from the
deployment distribution. That is the motivation for the next step, and it is a
much stronger one than "the numbers are slightly off."

#### Finding 3: the RAG gate, reframed as conformal novelty detection

The RAG similarity threshold's actual job is detecting out-of-domain input. An
earlier finding in this project was that an in-domain-only calibration set
cannot calibrate such a gate. Conformal novelty detection answers that
directly: it **calibrates on inliers only by construction**, producing a
p-value for "is this ticket exchangeable with the indexed corpus?" and
escalating when p ≤ α. The 45-ticket OOD set is then freed to be pure
evaluation data rather than the thing that sets the threshold it is judged
against.

| α | In-domain false-escalation rate | OOD detection (15 seeds) | OOD detection (45 variants) | Adversarial |
|---:|---:|---:|---:|---:|
| 0.20 | 0.194 | 100% | 100% | 9/9 |
| 0.10 | 0.091 | 100% | 97.8% | 9/9 |
| 0.05 | 0.040 | 100% | 95.6% | 9/9 |
| 0.01 | 0.000 | 93.3% | 84.4% | 6/9 |

The false-escalation rate tracks α almost exactly, which is the guarantee doing
its job: choosing α *is* choosing how often a legitimate ticket may be sent to
a human unnecessarily — a quantity the hand-tuned 0.67 threshold never had.
And at any α ≥ 0.05 the method matches production's **9/9** on the adversarial
set. The degradation at α = 0.01 is a resolution limit, not a failure: with 175
calibration points the smallest attainable p-value is 1/176 ≈ 0.0057, so α =
0.01 sits near the floor of what this sample size can certify.

OOD detection is reported at **seed level** as the headline. The 45 OOD tickets
are 15 hand-written seeds × 3 Gemini paraphrases, and the generator carries a
near-duplicate warning for exactly that reason — the variants are not
independent, so the honest denominator is 15.

#### Finding 4: matching the calibration distribution is necessary but not sufficient

> **Which measurement stands.** Every figure in this section is from the
> **deduplicated** 175-ticket set (0 near-duplicate pairs, max off-diagonal
> similarity 0.9494). An earlier run of the same comparison used a set that
> contained 21 near-duplicate pairs; those results were discarded, never
> published here, and are **superseded** by the table below. The defect and
> its cause are documented at the end of this section rather than omitted,
> because on the defective set Tier-2 appeared to *lose* coverage at α = 0.10
> — a false negative result that would have been reported had the duplicates
> gone unnoticed. The pre-dedup set is committed as
> `deployment_calibration_tickets.predup.json` so the difference is auditable.

Finding 2 established that the in-domain calibration set cannot be repaired by
filtering. The remedy had to be different data, so a **deployment-distribution
calibration set** was generated: 175 tickets, 25 per category — exactly the
size and class balance of the in-domain set, so any difference is attributable
to the distribution alone and not to sample size or quantile granularity.

It is written in the same plain, non-technical register as the 45-ticket
benchmark, with per-category scope anchors (generalising the fix from the
Infrastructure labelling bug), and disjointness from the benchmark is
*enforced* rather than assumed: every candidate is embedded with the production
model and rejected above 0.90 cosine against any benchmark ticket. Observed
maximum after generation: 0.872. Self-consistency disagreements are discarded,
not kept — benchmark policy, not the old calibration policy.
Script: `generate_deployment_calibration_set.py`.

| Tier | α | Gap (in-domain) | Gap (deployment) | ±2 s.d. | Set size in-dom → dep | Singleton in-dom → dep |
|---|---:|---:|---:|---:|---:|---:|
| Tier-1 | 0.20 | −0.222 | −0.244 | 0.060 | 2.67 → 2.58 | 8.9% → 8.9% |
| Tier-1 | 0.10 | −0.233 | **−0.144** | 0.045 | 3.44 → 4.04 | 8.9% → 6.7% |
| Tier-1 | 0.05 | −0.106 | −0.083 | 0.033 | 5.00 → 5.13 | 6.7% → 6.7% |
| Tier-2 | 0.20 | −0.044 | **0.000** | 0.060 | 1.11 → 1.20 | 84.4% → 80.0% |
| Tier-2 | 0.10 | −0.011 | −0.011 | 0.045 | 1.84 → **1.67** | 35.6% → **46.7%** |
| Tier-2 | 0.05 | +0.028 | −0.039 | 0.033 | 2.56 → **1.89** | 17.8% → **33.3%** |
| Tier-2 | 0.01 | +0.010 | −0.012 | 0.015 | 4.76 → **2.47** | 2.2% → **20.0%** |

Two results, and the second is the more useful one.

**The guarantee is not restored for Tier-1.** At α = 0.10 the shortfall closes
from −0.233 to −0.144 — about 38% of the gap recovered — but −0.144 is still
more than six times the noise band. Calibrating on realistic deployment data
helps a lexical model materially and does not fix it. **Distribution matching
is necessary but not sufficient; where the representation itself fails to
transfer, no amount of calibration-set realism repairs the guarantee.** That
strengthens Finding 1 rather than competing with it: the representation is the
binding constraint, and the calibration set is the looser one.

**For Tier-2 the win is set size, not coverage.** Coverage was already inside
the noise band, so there was nothing to repair — but at identical coverage the
deployment-calibrated predictor emits markedly tighter sets: at α = 0.10, mean
size 1.67 against 1.84 and singletons on 46.7% of benchmark tickets against
35.6%. At α = 0.01 mean set size nearly halves, 4.76 → 2.47. Since a singleton
is precisely the case the system can route without a human, that is **~11
percentage points more autonomy at the same risk level** — the practical payoff
of an exchangeable calibration set, independent of whether coverage was broken.

##### A data-quality defect found after the first run, and what it cost

The first generated set contained **21 near-duplicate pairs** at ≥0.95 cosine,
3 of them byte-identical. Duplicated calibration points are not exchangeable
draws: they inflate *n* without adding information and drag the empirical
quantile toward whatever score region they cluster in. This is the same defect
already documented for the OOD set (15 seeds × 3 near-identical variants), and
it matters more here, because a calibration set *determines* the quantile
rather than merely being scored against it.

It had a specific and slightly embarrassing cause. Tightening the
Infrastructure scope anchor to demand an explicit machine-level cue cut
self-consistency rejections from 36% to 3% — and narrowed the scenario space
enough that the generator kept re-writing the same "ran out of memory" ticket.
**Scope anchors buy label accuracy at the cost of diversity**, and that
trade-off is worth stating because the same prompt pattern is used in three
generators in this project.

The fix is enforced at generation time rather than cleaned up afterwards: a
within-category near-duplicate cap, set at **0.95** because the measured
within-category similarity distribution runs p50 = 0.729, p95 = 0.892,
p99 = 0.949 — legitimate same-category tickets genuinely reach 0.89, so 0.95
removes the anomalous top ~1% without suppressing real variation. Dedup kept
159 of 175; 16 regenerated under the guard. Final set: **0 near-duplicate
pairs, maximum off-diagonal similarity 0.9494.**

The numbers in the table above are from the clean set. The pre-dedup set is
preserved as `deployment_calibration_tickets.predup.json` so the effect is
auditable — and it was not cosmetic: on the defective set Tier-2 appeared to
*lose* coverage at α = 0.10 (−0.056), which would have been reported as a
finding had the duplicates gone unnoticed.

Scripts: `src/experiments/calibrate_conformal.py`. Results:
[`conformal_calibration_results.csv`](data/conformal_calibration_results.csv)
(128 configurations: 2 tiers × 2 contamination variants × 2 label filters × 2
score functions × marginal/Mondrian × 4 α),
[`conformal_novelty_results.csv`](data/conformal_novelty_results.csv),
[`deployment_calibration_tickets.json`](data/deployment_calibration_tickets.json)
(the 175-ticket deployment-distribution calibration set) and
[`deployment_calibration_rejected.json`](data/deployment_calibration_rejected.json)
(every candidate the self-consistency, benchmark-overlap and near-duplicate
guards rejected, with the reason).

### Streamlit demo

`src/app/streamlit_app.py` ties the cascade classifier and RAG layer into
one clickable demo: predicted category, which tier resolved it and why
(plain-language explanation of the escalation decision), a retrieved
similar-tickets table, and either a Gemini-grounded suggestion or a
visually distinct human-escalation alert.

The demo was verified live, end to end, across every decision path it can
take:

- **Tier-1 (fast path) resolution** — confirmed on direct, keyword-heavy
  tickets (e.g. a storage-quota request resolved at 91.3% Tier-1
  confidence; an account-lockout ticket at 67.8%). Correct category, good
  retrieval, correct suggestion.
- **Tier-2 (escalated) resolution** — confirmed on more conversational
  phrasing (e.g. a VPN timeout ticket, a password-reset ticket, and a
  "reports are timing out" ticket), where Tier-1 confidence fell below the
  0.50 threshold and the ticket correctly escalated to the stronger
  Tier-2 model, which then classified correctly. In one case, Gemini's
  suggestion correctly flagged that the new ticket didn't specify which
  underlying database was affected, unlike the retrieved examples, and
  recommended a human double-check that detail.
- **RAG-similarity escalation to a human** — confirmed on an intentionally
  unrelated ticket ("what's the weather today"), a vague multi-category
  ticket with no distinctive vocabulary, and a maximally uninformative
  ticket ("something is wrong, please fix it"). In every case, retrieval
  similarity fell below the threshold, the Gemini call was correctly
  skipped, and the ticket was escalated to a human instead of a fabricated,
  confident-sounding suggestion. This is a meaningful result on its own: a
  forced-choice classifier can never simply "refuse" to output a category,
  but the retrieval-confidence gate catches exactly this failure mode
  before it reaches an end user.

### Batch-intake simulation (production traffic skew)

Motivated by a mentor question: what happens when real ticket traffic
arrives unevenly across categories? `simulate_ticket_intake.py` draws a
500-ticket batch using a Zipf-style skew, and `process_ticket_batch.py`
runs the full pipeline against it. Result: 100% accuracy, zero escalations
— expected, since the batch is sampled from the same in-distribution
held-out pool the classifier already performs perfectly on. This validated
the *plumbing* (routing, per-category storage, rate-limit handling) under
uneven volume, not classification robustness.

Operationally, this run hit Gemini's free-tier rate limits (15
requests/minute, 500/day) — 15 of 500 resolution calls failed after
retries. Fixed by increasing the per-call delay from 1.5s to 4.5s to stay
under the per-minute cap; the 15 failed calls were later retried
successfully once the daily cap reset.

### Class-imbalance experiment (training-data skew)

The complementary experiment: instead of testing traffic-volume skew
against an already-trained model, this tests what happens when the model
itself is trained on progressively less minority-class data.
`generate_skewed_datasets.py` samples "Access Management" down from its
full 571 tickets to 500 / 200 / 100 / 50, holding the other 6 categories
fixed. Access Management was chosen because it scored perfectly on the
14-ticket benchmark at full data (so any degradation is attributable to
the skew itself) and is semantically adjacent to Security, giving a
plausible, checkable failure mode.

| AM training size | MiniLM 14-ticket benchmark | AM tickets specifically |
|---:|---:|---|
| 571 (baseline) | 11/14 | both correct |
| 500 | 11/14 | both correct |
| 200 | 10/14 | one misclassified → Storage |
| 100 | 9/14 | both wrong → Security, → Storage |
| 50 | 9/14 | both still wrong → Security, → Storage |

Real degradation appears as Access Management shrinks, and one failure
mode is exactly the predicted one (misclassification as Security).
In-distribution accuracy stayed ~99.6–100% at every level, confirming
(again) that in-distribution numbers are uninformative here — only the
14-ticket benchmark exposed the effect. Every wrong Access Management
prediction across all skew levels fell below the 0.50 confidence threshold
and was correctly escalated to a human rather than confidently accepted —
a promising signal, though the sample (1–3 wrong predictions per level) is
small enough that this should be read as directional, not conclusive.

*(Methodology note: this sweep trains and evaluates on an 80% train split,
whereas the cascade's originally-reported 10/14 MiniLM score used a model
retrained on the full dataset — the two numbers aren't directly
comparable; the meaningful result is the degradation trend across skew
levels, not the absolute score at the 571 baseline.)*

### Expanding the benchmark: 14 → 45 tickets

The 14-ticket benchmark is informative but noisy — each ticket is worth
7.1 percentage points. A larger, still-rigorous benchmark was built using
the same Gemini paraphrase-and-verify methodology already proven for the
175-ticket calibration set, but with one deliberate policy difference: for
this benchmark, self-consistency mismatches are **never** auto-kept — they
are flagged for manual human review and excluded by default, since this is
the primary accuracy metric rather than a calibration set.

**Stage 1:** generated 35 candidates (5/category), 32 passed clean, 3 were
flagged and, after manual review, judged genuinely ambiguous boundary
cases — legitimately excluded. Merged with the original 14 = a 46-ticket
benchmark.

**Stage 2 — the Infrastructure bug:** after running the 3-way embedding
comparison against the 46-ticket set, every model got every one of 5
specific "Infrastructure" tickets wrong. Investigation traced the root
cause: those 5 tickets were all physical-facilities issues (flickering
lights, a stuck door, a burst pipe, a broken elevator, a broken heater) —
but this project's actual `Infrastructure` category is tightly and
consistently scoped to compute/server issues (CPU, memory, Kubernetes,
NTP drift, cron jobs, load balancers, disk I/O, autoscaling) with zero
facilities-adjacent scenarios anywhere in the training data. The
generation prompt had only said "plain, non-technical, everyday language"
without anchoring what "Infrastructure" means *for this project*, so
Gemini used the everyday-English sense of the word instead. This was a
benchmark-generation artifact, not a flaw in the dataset's own category
design.

**Fix:** the 5 mislabeled tickets were stripped (46 → 41), 5 replacement
tickets were generated with an explicit scope anchor (positive:
compute/server/K8s/CPU/memory/disk/clock-sync/autoscaling; negative:
explicitly excludes lighting/doors/elevators/plumbing/HVAC), and after
manual review of flagged candidates across two generation attempts, 41 +
2 auto-clean + 2 manually-approved were merged into a final,
fully-trusted **45-ticket benchmark** (`data/novel_tickets_expanded.json`).
A round number (46/49) was deliberately not chased — 45 with full
confidence in every ticket was preferred over a round number with any
doubt.

**Validation that the fix worked:** every model's score improved once the
mislabeled tickets were removed (MiniLM 29/46 → 32/45, BGE 31/46 → 33/45,
E5 27/46 → 27/45). The **BGE-beats-MiniLM finding (73.3% vs 71.1% at
n=45)** is now trustworthy and citable, and was the basis for the
subsequent production swap to BGE (see "Embedding Model: MiniLM → BGE"
above). One genuinely hard case survived the fix and is worth discussing
on its own: a "status page shows green but the service is actually down"
ticket, which all three models still get wrong — a realistic
infrastructure-monitoring ambiguity, not a labeling error.

### Resolution-clustering calibration

A separate calibration effort, aimed at a different question: can
*resolved* tickets be automatically clustered by their resolution text to
find recurring issues worth flagging for self-service automation?

Resolution text (not the ticket symptom) was embedded with MiniLM (this
calibration was deliberately left un-migrated in the BGE swap — scoped as
its own separate exercise — the BGE clustering re-run below), then grouped via connected
components at cosine-similarity thresholds swept from 0.99 down to 0.60.
Each threshold's clustering was evaluated with pairwise precision/recall/F1
against real `scenario_id` ground truth (recovered from the dataset
generator's own internal `random.choice()` selection, verified
byte-for-byte reproducible against the pre-change data).

**Finding: a cliff-edge, not a gradual slope.** Precision holds perfectly
(1.0000) all the way down to threshold = 0.80, then collapses sharply at
0.75 and keeps collapsing at looser thresholds. **Threshold = 0.80 was
chosen** — the last point with zero false "these tickets share a fix"
claims — deliberately preferred over the recall-better 0.75, because a
false positive here (wrongly telling ops two different problems share a
fix) is costlier than a false negative (a missed automation opportunity,
which just means the status quo continues).

**BGE clustering re-run:** the same pooled clustering + pairwise evaluation
was re-run under BGE embeddings (`explore_resolution_clustering.py` and
`calibrate_resolution_clustering.py`, both re-pointed at BGE-suffixed
output files so the original MiniLM artifacts are preserved for
comparison, not overwritten). `calibrate_resolution_clustering.py` prints
the full per-threshold sweep but does not itself compute a cliff-edge, so
this was derived by hand from the real aggregated output using the
project's standard rule (lowest threshold with precision exactly 1.0,
scanning tight → loose): **BGE's pooled cliff-edge is 0.90**, not 0.85 —
precision is 1.0000 at 0.90 but drops to 0.9814 at 0.85, a real (if small)
break in the streak. This is measurement-only; production automation-
flagging remains on MiniLM at the calibrated 0.80 (see "Pending" below for
why the production threshold was not swapped).

### Category-specific resolution-clustering calibration

The pooled 0.80 threshold above was derived by combining all 7 categories
into one calibration run. Given the category-level differences later
found in the automation-flagging feature (Infrastructure highly
standardized, Database more bespoke), the same clustering and pairwise
evaluation methodology was re-run independently per category, to check
whether each category's own precision cliff-edge lands at or near 0.80,
or whether some categories would need a different cutoff on their own
data.

The clustering method (connected components via union-find on the cosine
similarity matrix) and the pairwise evaluation logic (precision/recall/F1
against `scenario_id` ground truth) were reused unchanged from the pooled
calibration script — only the population was filtered to one category at
a time before running the same threshold sweep (0.99 down to 0.60). Two
real bugs were found and fixed while building this script: an overly
strict small-N confidence bar (100 → 60) that mis-flagged categories as
low-confidence, and a cliff-edge-detection bug that broke on undefined
(0/0) precision instead of skipping it, which had caused 3 categories to
falsely report "no cliff-edge found."

| Category | n | Cliff-Edge | Precision at Cliff | Recall at Cliff | vs. Pooled 0.80 |
|---|---:|---:|---:|---:|---|
| Infrastructure | 115 | 0.80 | 1.0000 | 0.5425 | Matches |
| Application | 100 | 0.80 | 1.0000 | 0.4991 | Matches |
| Security | 61 | 0.80 | 1.0000 | 0.8708 | Matches |
| Access Management | 48 | 0.80 | 1.0000 | 0.3966 | Matches (low-N) |
| Storage | 53 | 0.75 | 1.0000 | 0.8312 | Differs, -0.05 (low-N) |
| Database | 49 | 0.70 | 1.0000 | 0.7882 | Differs, -0.10 (low-N) |
| Network | 74 | 0.75 | 1.0000 | 0.5737 | Differs, -0.05 |

Four of seven categories match the pooled threshold exactly, which
supports the pooled 0.80 choice as a reasonable default. The most notable
individual result is **Network**: at n=74 it is above the low-confidence
sample-size bar (60) used elsewhere in this analysis, yet it still needs a
threshold 0.05 looser than pooled before precision holds — a genuine,
statistically credible divergence rather than small-sample noise.
Database shows the largest deviation (-0.10) but carries a low-N caveat,
so it is read as indicative rather than conclusive on its own. This
corrected `find_cliff_edge()` implementation (with the None-precision fix
above) is the reference version later reused and adapted for the RAG
similarity threshold recalibration script.

**BGE clustering re-run:** the same per-category sweep was re-run under
BGE (`calibrate_resolution_clustering_percategory.py`, `MODEL_NAME`
swapped to `BAAI/bge-base-en-v1.5`, output re-pointed at BGE-suffixed
files). Its `POOLED_THRESHOLD = 0.80` constant reflects the live
*production* (MiniLM) value and was deliberately left unchanged rather
than hardcoding a second stale constant — so the script's own printed
"vs. Pooled 0.80" comparison column is not the right one to read for a
BGE-vs-BGE comparison; the correct baseline is BGE's own pooled cliff of
0.90 (above), applied by hand:

| Category | n | BGE Cliff-Edge | vs. BGE Pooled 0.90 |
|---|---:|---:|---|
| Infrastructure | 115 | 0.85 | −0.05 |
| Application | 100 | 0.85 | −0.05 |
| Security | 61 | 0.85 | −0.05 |
| Database | 49 | 0.85 | −0.05 (low-N) |
| Storage | 53 | 0.85 | −0.05 (low-N) |
| Network | 74 | 0.85 | −0.05 |
| Access Management | 48 | 0.90 | matches exactly (low-N) |

A genuinely citable finding falls out of comparing the two embedding
models' spreads directly: under MiniLM, per-category cliffs ranged across
a full 0.10 band (0.70–0.80), with 4/7 categories matching pooled exactly
and the other 3 diverging by up to −0.10. Under BGE, every category lands
within a single 0.05 step of its own pooled cliff. **BGE produces more
cross-category-consistent clustering behavior than MiniLM did**, on top of
needing a systematically higher absolute threshold — consistent with the
same higher-similarity-score pattern already established during the RAG
threshold recalibration (see "RAG Similarity Threshold Recalibration"
above). This BGE comparison is measurement-only, same as the MiniLM
result it's benchmarked against; see "Pending" for why it hasn't been
promoted to production.

This is a measurement-only diagnostic: it does not change the production
threshold. It establishes that the pooled 0.80 threshold is well-supported
overall, while identifying Network (and, with lower confidence, Database
and Storage) as categories where a category-specific threshold could
recover additional recall without sacrificing precision — a candidate for
a future, explicit production decision rather than an automatic change.

Scripts: `src/experiments/calibrate_resolution_clustering_percategory.py`.
Results:
[`resolution_clustering_calibration_percategory.csv`](data/resolution_clustering_calibration_percategory.csv)
(full per-threshold detail),
[`resolution_clustering_calibration_percategory_summary.csv`](data/resolution_clustering_calibration_percategory_summary.csv)
(one row per category).

### Phase 2A — automation-flag validation (the BGE promotion question)

The two sections above leave one question open, and the "Pending" list
named it as the prerequisite for any production swap: BGE's clustering is
*internally* tidier than MiniLM's, but is it actually *better* at flagging
automation candidates? Cliff-edge math alone cannot answer that. The RAG
similarity threshold was only adopted because a 45-ticket accuracy
benchmark and a 9-ticket adversarial set could re-confirm the specific
chosen value against real cases. Resolution clustering had no equivalent.

Phase 2 built that equivalent. The answer it produced is a negative one,
and the reason is more interesting than the verdict.

#### The structural diagnostic

Before spending any human labelling budget, the two configurations were
compared pair-by-pair. For each category, every pair of resolved tickets
co-clustered by MiniLM @ 0.80 was compared against every pair co-clustered
by BGE @ 0.90 — each model at its **own** calibrated cliff-edge, since
comparing both at 0.80 would handicap BGE at a threshold its own precision
curve does not endorse.

| Category | MiniLM@0.80 | BGE@0.90 | Both | MiniLM-only | BGE-only |
|---|---:|---:|---:|---:|---:|
| Infrastructure | 402 | 449 | 370 | 32 | 79 |
| Application | 277 | 372 | 254 | 23 | 118 |
| Security | 182 | 160 | 160 | 22 | 0 |
| Database | 34 | 34 | 23 | 11 | 11 |
| Storage | 101 | 119 | 99 | 2 | 20 |
| Network | 125 | 201 | 121 | 4 | 80 |
| Access Management | 46 | 47 | 42 | 4 | 5 |
| **Total** | **1,167** | **1,382** | **1,069** | **98** | **313** |

Then the decisive check, against `scenario_id` ground truth:

**Not one of the 1,382 merges either configuration makes is a
cross-template merge.** All 1,069 pairs both models merge, and all 411
pairs they disagree about, join two tickets from the same dataset
template.

That single fact reframes the whole question. Both configurations are
perfectly template-precise at their own cliff-edges — which is, of course,
exactly how those cliff-edges were chosen. **The entire measured
difference between MiniLM @ 0.80 and BGE @ 0.90 is recall**: which
within-template pairs each one manages to find. BGE finds 313 that MiniLM
misses; MiniLM finds 98 that BGE misses.

#### A pre-registered decision rule that had to be thrown out

The validation set was originally built with a rule fixed in advance: each
sampled pair is co-clustered by exactly one configuration, so a human label
awards the point to exactly one of them, and the winner is decided by an
exact two-sided binomial (McNemar) test on the split.

Statistically that rule is sound. For *this* project it was wrong, and the
diagnostic above is why. Since the sample is drawn proportionally from a
disagreement region that is 313 BGE-only against 98 MiniLM-only, labelling
every pair `same_fix` hands BGE a 42–18 win at p = 0.0027. A dry run
against synthetic labels confirmed it: the rule printed **PROMOTE** on zero
evidence about flag correctness. It was rewarding whichever model merges
more.

That directly inverts the cost asymmetry this feature is built on — a false
"these two share a fix" claim misleads operators, while a missed automation
opportunity merely preserves the status quo. A precision gate that promotes
on recall is not a gate.

The rule was replaced before any label was collected:

> **Primary:** the false-merge rate on each configuration's *extra* merges
> — the pairs it uniquely co-clusters. A `different_fix` label there is a
> false merge by that configuration, and a false merge is the costly error.
> Promote BGE only if it makes **zero** observed false merges **and**
> MiniLM makes at least one. If neither makes one, the verdict is **no
> precision signal**, not promotion.
>
> **Secondary:** the binomial win-rate, still reported but explicitly
> demoted and labelled as the recall comparison it is.

#### The pilot

Rather than spend the full 60-judgement budget confirming something the
diagnostic already made likely, a 12-pair pilot was drawn from the **most
textually divergent end** of the disagreement region — ranked by
`1 − token Jaccard` of the two resolution texts, since a genuinely
different fix, if one exists anywhere in the region, would surface where
the wording diverges most. The selected pairs span divergence **0.53–0.68**
against a region median of ~0.42, and the near-duplicate guard rejected 7
candidates along the way.

The pilot is balanced 6/6 across directions on purpose. That balance would
bias a win-rate comparison, so the scorer refuses to compute the
head-to-head on a pilot at all; the probe asks only whether a false merge
is observable *at all*. Labelling was blind: the file shown to the labeller
omits `scenario_id` and does not reveal which configuration co-clustered
each pair, and entries are shuffled so ordering cannot leak direction
either.

#### Result

| Configuration | Extra merges judged | `same_fix` | False merges | Rate |
|---|---:|---:|---:|---:|
| MiniLM @ 0.80 | 6 | 6 | **0** | 0.0% |
| BGE @ 0.90 | 6 | 6 | **0** | 0.0% |

**Zero false merges by either configuration, in the 12 pairs where a false
merge was most likely to appear.**

The one pair that came close is worth recording, because it is the only
place in the pilot where the judgement was genuinely contested. **PL005**
(Infrastructure) pairs two `fstab`-blocked boot failures: ticket A corrects
the mount entry, ticket B corrects it *and* explicitly restarts/resumes the
boot. It was labelled `same_fix` but flagged low-confidence at labelling
time — the note reads, verbatim, that it "could be read as an implied
trivial follow-on to an otherwise identical fix, OR as a materially
different remediation SOP (one action vs two)." For contrast, **PL012**
pairs two tickets that *both* state the reboot step, and is unambiguous.
The remaining judgement-worthy pairs resolved cleanly: **PL009/PL010** name
the SMTP change as a cause on one side only, but prescribe word-for-word
equivalent remediation; **PL011** pairs two tickets from the same branch,
where whether it is one recurring incident or two is a separate question
from whether the fix is shared.

The secondary `scenario_id` measurement agreed with the human label 12/12
(100%) — exactly the degeneracy predicted by the diagnostic, reported so
the degeneracy is visible rather than assumed.

#### Zero observed is not zero

The honest bound on this result is wide, and the scorer prints it rather
than letting the table above speak unqualified. By the rule of three, zero
events in *n* trials supports a one-sided 95% upper bound of 3/*n*:

| Claim | Observed | 95% upper bound on the true rate |
|---|---|---|
| MiniLM @ 0.80 makes no false merges | 0 / 6 | **50.0%** |
| BGE @ 0.90 makes no false merges | 0 / 6 | **50.0%** |
| Neither makes a false merge | 0 / 12 | **25.0%** |

So this finding does **not** establish that no false-merge case exists in
the disagreement region. It establishes that none was found among the 12
most divergent pairs in it — which is a much weaker claim, and the correct
one to make. A false-merge rate as high as one in four would be entirely
consistent with observing zero here.

What makes the result still worth acting on is not the bound but the
diagnostic underneath it: the region contains no cross-template merges at
all, so there is no structural mechanism by which either model's extra
merges *could* be systematically wrong on this data. The pilot is
consistent with the diagnostic rather than carrying the conclusion alone.

#### Conclusion: this is a product decision, not an evidence one

**The promotion from MiniLM @ 0.80 to BGE @ 0.90 cannot be settled by
flag-correctness evidence on this dataset.** The two configurations are
indistinguishable on precision — the axis the production threshold exists
to protect — and differ only in recall, which the cluster counts already
report for free and which no amount of labelling will convert into a
precision argument.

Swapping to BGE would surface more automation candidates (1,382
co-clustered pairs against 1,167). Whether that is an improvement depends
on how many candidates the review queue should carry, which is a product
judgement about human review capacity, not a calibration result. It should
be argued and recorded as such.

**Production therefore stays on MiniLM @ 0.80**, and the remaining 48
judgements were not spent — the pilot's purpose was to determine whether
they would measure anything, and they would not.

The deeper limitation is the dataset, not the method. Template-generated
data cannot produce two tickets that look alike but need different fixes,
because the templates *are* the fix classes. Distinguishing these two
configurations on precision needs deployment-distribution resolved tickets
— the same conclusion the conformal calibration work reached from an
entirely different direction, which is itself some evidence that the
limitation is real rather than an artifact of one experiment's design.

Scripts: `src/experiments/build_flag_validation_set.py` (add `--pilot` for
the probe), `src/experiments/score_flag_validation_set.py` (likewise).
Both are offline, deterministic at seed 42, load no model and spend no
Gemini quota. Data:
[`automation_flag_validation_pilot.json`](data/automation_flag_validation_pilot.json)
(the 12 labelled pairs),
[`automation_flag_validation_pilot_key.json`](data/automation_flag_validation_pilot_key.json)
(the withheld key),
[`automation_flag_validation_pilot_results.csv`](data/automation_flag_validation_pilot_results.csv)
(scored output). The unlabelled 60-pair set is retained for the record,
since building it is what produced the diagnostic.

### Phase 2B — resolution groundedness, and the limits of an LLM judge

Phase 2A ended by establishing that the in-distribution corpus could not
answer the question put to it. Phase 2B asked a different question —
**does the resolver say only what its retrieved context supports?** — and
this time the data could answer, because a pre-flight diagnostic was run
before any quota was spent to check that it could.

#### The pre-flight diagnostic

A judge can only find an unsupported claim if the retrieved context leaves
room for one. If all five retrieved resolutions prescribe the same fix, any
faithful draft is grounded by construction and the judge returns 1.0 on every
item with no variance — the exact 2A failure mode. So before designing the
harness, the retrieved context was measured directly: production BGE
retrieval, with the calibrated production resolution-clustering instrument
(MiniLM @ 0.80, union-find) used to ask how many *distinct fixes* appear in
each top-5.

| Query set | All 5 → one fix | 2+ distinct fixes | Mean pairwise sim | Spans >1 category |
|---|---:|---:|---:|---:|
| In-distribution (n=200) | **85.0%** | 15.0% | 0.945 | 0.5% |
| Novel-45 | 28.9% | **71.1%** | 0.720 | 40.0% |
| Adversarial-9 | 11.1% | **88.9%** | 0.663 | 44.4% |

The in-distribution corpus is structurally degenerate, exactly as 2A found
for clustering. The benchmarks are not. **The degeneracy is a property of the
template-generated corpus, not of groundedness as a construct** — which is
why 2B could proceed where 2A could not.

That settled the scope: the two fixed benchmarks, and not the 500-ticket
in-distribution batch. The batch was disqualified twice over — degenerate
context, and drafts generated under MiniLM retrieval, so the context they saw
is not the context they would be judged against. A groundedness audit must
judge a draft against its real context.

Of 54 benchmark tickets, 21 escalate at the RAG gate and never reach Gemini,
so no draft exists to judge. That leaves **33** (novel-45: 30,
adversarial-9: 3) — the entire eligible population, not a sample. The
14-ticket `NOVEL_TICKETS` set was checked and is a strict subset of the 45
(14/14 verbatim); it adds nothing.

#### Design

One rubric, defined once and shared **verbatim** by the human labeller and the
LLM judge — agreement is only interpretable if both answered the same
question, so the judge imports the rubric string rather than restating it.
Three levels (`grounded` / `partially_grounded` / `ungrounded`) plus a
`hedge_appropriate` boolean.

At n = 33 the human labels everything, so **the judge is not a labour-saving
device — it is the object of study**, and judge–human agreement is reported as
a result in its own right. Its verdicts are written to a separate file so they
cannot leak into the labelling pass, which is itself blind: the labelling file
carries only the ticket, the draft and the retrieved resolutions, shuffled at
seed 42, with no provenance, similarities or source benchmark.

One rubric clause was added after a 3-draft dry run and **before any
labelling**, because a dry-run case exposed a gap: a draft that declines to
apply the retrieved fix and recommends human investigation. Read literally,
its "investigate" step is unsupported, which would have penalised a correct
refusal. The clause makes declining explicitly `grounded`. It was fixed before
labelling deliberately — changing a rubric afterwards would silently
invalidate the judge–human comparison.

#### Primary result: groundedness

| Label | n | Rate | Wilson 95% CI |
|---|---:|---:|---|
| `grounded` | 31 / 33 | **93.9%** | 80.4% – 98.3% |
| `partially_grounded` | 0 / 33 | 0.0% | 0.0% – 10.4% |
| `ungrounded` | 2 / 33 | 6.1% | 1.7% – 19.6% |

The two ungrounded cases share a precise and repeatable shape, and it is not
the shape the rubric clause was written for.

**G021** (laptop disk full) and **G024** (ransomware pop-ups) both retrieved
irrelevant context — network-share disk performance and NTFS permissions
respectively. Both drafts correctly *recognised* the mismatch and said so
explicitly. Then both went on to prescribe a substantive fix anyway, drawn
from general knowledge rather than from anything retrieved: inspecting local
disk usage and clearing old files in G021; disconnecting the machine from the
network and escalating to incident response in G024.

The contrast is what makes this a finding rather than an anecdote. **Ten of
the 33 drafts contain explicit mismatch language. Eight of them declined
cleanly and are grounded. Only these two acknowledged the mismatch and then
prescribed unsupported content regardless.** So the failure is not "the model
cannot tell when retrieval is irrelevant" — it demonstrably can, 10 times out
of 10. The failure is that recognising irrelevance does not reliably stop it
from answering anyway.

**G024 is the cleanest illustration of why groundedness is not quality.**
"Disconnect from the network and escalate to the security team" is excellent
ransomware advice — very likely better than anything the retrieved tickets
could have supported. It is also entirely ungrounded. The rubric anticipated
this ("a draft can be excellent advice and still be ungrounded"), and the
labelling held the line. A groundedness metric that rewarded good advice
would have measured something else. (Note also that a ransomware ticket was
routed to **Storage** — a separate classification concern, not a groundedness
one.)

#### Secondary result: hedge appropriateness

The resolver prompt instructs the drafter to close with a note saying either
that the retrieved examples closely match or that a human should verify.
Judged against how well the retrieved resolutions actually agree with each
other and with the ticket:

**32 / 33 appropriate (97.0%, Wilson 95% CI 84.7–99.5%)** — 10/10 where the
context collapsed to a single fix, 22/23 where it carried two or more. The
single inappropriate hedge was G033.

This costs nothing extra to measure: the note is already in every draft, and
the context heterogeneity is the same quantity the pre-flight diagnostic
computed offline.

#### Methodological result: the LLM judge fails on this rubric, and the negative kappa says how

| Dimension | Raw agreement | Cohen's κ |
|---|---:|---:|
| Groundedness | 30 / 33 = **90.9%** | **−0.042** |
| `hedge_appropriate` | 32 / 33 = 97.0% | 0.000 |

A κ below zero means the two raters agreed *less* than their marginal
distributions alone would predict. With 90.9% raw agreement that looks
paradoxical, and part of it genuinely is the well-known prevalence effect:
when 31 of 33 items sit in one category, chance agreement is already ≈91%, so
there is almost no headroom above it and κ becomes hypersensitive. At n = 33
with three disagreements, the point estimate −0.042 carries enormous
uncertainty and **should not be quoted as a magnitude**.

What survives that caveat is the *mechanism*, because all three disagreements
fall on one axis — and in both directions:

- **G021, G024** — judge `grounded`, human `ungrounded`. The judge credited
  the drafts for recognising that retrieved context was irrelevant, and did
  not penalise the unsupported fix each then prescribed. Its stated reasons
  say so outright: *"correctly recognizes that the retrieved examples are
  irrelevant... and appropriately declines to apply them."* The draft did not
  decline. It said the examples did not match and then answered anyway. The
  judge **over-applied the declining-is-not-unsupported clause**, stopping at
  the acknowledgement without checking what followed it.

- **G012** — judge `ungrounded`, human `grounded`. Here the draft prescribed
  the retrieved session-cookie fix, which the context does support, and hedged
  that the new employee may instead need an account provisioned. The judge
  marked it ungrounded because the fix *"completely contradicts its own
  correct observation"* — that is, because the advice was inappropriate to the
  ticket. The rubric explicitly forbids this: *"Do NOT judge whether the draft
  is a good fix."*

Both error types are the same confusion with opposite signs: **the judge
substituted appropriateness for support.** In G021/G024 it rewarded a draft
for noticing that its context was inappropriate; in G012 it punished a draft
for being inappropriate despite being supported. That is why the errors are
anti-aligned with the human axis rather than scattered, and it is why κ lands
at or below zero instead of merely low.

So the negative κ is a real result and not a null one — but the load-bearing
evidence is the case-by-case mechanism, identifiable because n = 33 was small
enough to read every disagreement. **An LLM judge should not be trusted
unaudited on a rubric whose central distinction is support-versus-quality,**
which is precisely the distinction groundedness rests on. Had the judge been
used as a scaling tool over hundreds of items — the design this harness
started as — it would have reported ~97% grounded, missed both real failures,
and invented a third. The agreement number would have looked reassuring.

That is the methodological contribution here: not that LLM judges are
unreliable in general, but that raw agreement of 90.9% concealed a
systematically wrong judge, and only labelling the full population and reading
every disagreement exposed it.

#### Honest limits

n = 33 gives a proportion a ±2 s.d. band of roughly ±17 points at p = 0.5,
tightening to about ±10 near p = 0.9 — hence the Wilson intervals above rather
than bare percentages. The 93.9% groundedness rate is compatible with anything
from ~80% to ~98%. Enough to say gross ungroundedness is not endemic; not
enough to claim a precise rate.

The adversarial-9 set contributes only 3 drafts (the other 6 escalate, as they
are designed to). All three are grounded, but nothing can be claimed about
that subset on its own.

The κ figures inherit both the small n and the prevalence effect, as above.
The hedge κ of 0.000 is degenerate for a specific reason worth recording: the
**judge** returned `hedge_appropriate: true` on all 33 items, so it had no
variance at all, while the human returned one `false`. A κ computed against a
constant rater is uninformative by construction; the 97% raw agreement is the
only readable number there, and it should never be cited as a κ.

Finally, the same dataset limit that closed 2A applies here in a narrower
form: these 33 tickets are the only out-of-template text the project has, and
they are a fixed benchmark that has now been used for classification
accuracy, conformal coverage, and groundedness. Reusing one small benchmark
across several questions accumulates selection risk that no single experiment
can see.

Scripts: `src/experiments/build_groundedness_set.py` (`--limit N` to dry-run),
`src/experiments/run_groundedness_judge.py`,
`src/experiments/score_groundedness_set.py` (offline). Data:
[`groundedness_set.json`](data/groundedness_set.json) (33 labelled drafts with
their retrieved context),
[`groundedness_key.json`](data/groundedness_key.json) (provenance),
[`groundedness_judge.json`](data/groundedness_judge.json) (judge verdicts),
[`groundedness_results.csv`](data/groundedness_results.csv) (scored). Every
draft carries config fingerprint `05f391baf27c`; the builder aborts if a run
produces more than one.

### Phase 3 — the agent architecture: orchestrator and service

The one phase whose deliverable is architecture rather than a measurement, and
therefore the one most at risk of quietly changing a published number while
claiming to be a refactor. So every sub-phase was gated on the same thing:
**the goldens must not move.** They didn't — `tests/goldens/*.json` reproduce
exactly (45/45 and 9/9), the adversarial gate stayed 9/9, the 45-ticket
benchmark stayed 32/45, and `data/adversarial_escalation_results.csv`
regenerates byte-identical at every step. The test suite grew 69 → 104.

#### 3A — Tier-1 stopped being refitted at startup

Tier-1 (TF-IDF + LogReg) was refitted from `synthetic_tickets.csv` on every
process start. Fine for a script, wrong for a service: a per-request API must
not derive a model from raw training data at boot. It is now a persisted
artifact built by `src/classification/train_tier1.py`.

The honest accounting: fitting costs **1.56s**, loading the persisted bundle
**0.014s** — a real saving, but small beside BGE's ~60s. Speed is not the
justification; determinism is. The model became a pinned, fingerprinted object
instead of something re-derived at each startup from whatever the CSV happened
to contain.

Persisting a model creates a new place for a stale artifact to hide, which is
this project's recurring bug class, so the guard shipped in the same change.
The bundle carries a manifest — dataset sha256, rows fitted, sklearn version,
vectorizer config — and `artifacts.load_tier1()` refuses a bundle whose
manifest no longer matches reality. There is deliberately **no refit-on-miss
fallback**: quietly refitting would paper over a missing or stale artifact and
leave the cascade running on a model nobody asked for.

The specific trap the manifest guards: **Tier-1 is fitted on the full 4,000
rows, not the 80/20 split every other script in `src/classification/` uses**,
because that is what the live demo did and what the goldens were captured
under. A split fit would shift every Tier-1 confidence and every cascade
routing decision with it, invisibly. The manifest records rows-fitted against
rows-available and the loader rejects a mismatch.

3A also found and closed a live instance of the duplication problem:
`streamlit_app.py` and `test_adversarial_escalation.py` each still fitted
their *own* Tier-1 and passed it into `pipeline.run()`, so both would have
bypassed the persisted artifact entirely — three separately-fitted models free
to diverge, the same shape as the four divergent pipeline copies Phase 0
removed. Both now load through `artifacts.load_tier1()`.
`calibrate_conformal.py` and `plot_calibration_curves.py` keep their own fits
on purpose: they fit on deliberate leave-out subsets and must not use the
production artifact.

#### 3B — agents with declared dependencies, and an orchestrator that owns routing

Before this, the three stages were free functions with three different
signatures, each handed the whole `Artifacts` blob and reaching into whatever
it needed. Nothing declared what a stage actually depended on, so nothing could
be moved, replaced, or served independently without reading its body.

Three things changed, and the third is the one that mattered later:

1. **Declared dependencies.** Each agent names the `Artifacts` fields it
   requires, validated at construction — a missing *or misspelled* dependency
   fails before any routing happens. That declaration is the agent boundary
   written down, and it is what made 3C's HTTP split mechanical rather than
   exploratory.
2. **The orchestrator owns every routing decision.** No agent reads a threshold
   or knows what runs after it. Both gates became pure functions
   (`filing_gate`, `rag_gate`) testable without loading a model — an escalation
   policy that needs BGE and FAISS to test is one nobody tests, and this
   project's entire claim rests on that policy.
3. **Per-agent traces.** `PipelineResult.steps` records every agent including
   the ones that deliberately did not run. A resolution step marked `skipped`
   is *positive* evidence that the RAG gate held and no LLM call was made; an
   absent step would be ambiguous. The decision log gained `agent_status` and
   `agent_latency_ms` alongside it — the per-agent history the planned
   drift-detection phase will read.

The stage implementations (`classifier.py`, `retriever.py`, `resolver.py`) were
deliberately **not** touched. They hold the parity-critical logic — the
L2-normalise before search, the calibrated prompt, the retry ladder — and the
agents wrap them rather than absorbing them. `pipeline.run()` stayed as a
façade over `orchestrator.run()`, so all seven call sites were untouched.

#### 3C — the HTTP service, and where the failure boundary belongs

`src/service/api.py` exposes each agent individually, plus `/health`,
`/policy/rag-gate` and `/triage`:

| Endpoint | Purpose |
|---|---|
| `GET /health` | `config_fingerprint()` over the wire, plus index size and the Tier-1 manifest — a deployment's stale-artifact detector |
| `POST /agents/classify` | the classification agent alone |
| `POST /agents/retrieve` | the retrieval agent alone |
| `POST /agents/resolve` | the resolution agent alone — the only endpoint that spends Gemini quota |
| `POST /policy/rag-gate` | the escalation decision for a retrieval result |
| `POST /triage` | the whole pipeline |

**`/triage` orchestrates in-process.** It calls `pipeline.run()`, not its own
endpoints. Golden parity must not depend on a running server, and network hops
would add failure modes to the measured path — the path every number in this
README was measured on. So the agents are independently *addressable*, not
independently *running*, and the honest claim is a sequential pipeline with
agent boundaries and an HTTP surface, not a distributed system.

`/triage` defaults to `generate_resolution=false`. The free tier is 500
calls/day and a looping workflow would drain it; spending quota should be
something you asked for.

**The failure boundary ended up in the service, not the library — a reversal
of the plan, recorded here rather than silently swapped.** The intention was a
new `EscalationReason` so that any agent failure produced an escalated
`PipelineResult`. That turned out to be the wrong design:
`PipelineResult.classification` is a **required** field, so a classification
failure cannot produce a `PipelineResult` at all without inventing a category —
and fabricating a routing decision is precisely the thing this system exists
not to do. Making the field optional would have pushed `None`-safety onto ~20
dereference sites across the logger, the demo and the experiment scripts,
trading a caught error for an `AttributeError` in front of an audience.

So the boundary lives in the API's own response envelope, which needs no
classification, and **the library still raises**. That is also the better
research answer: in a calibration run, a `RetrievalError` quietly becoming an
escalation row would corrupt the result with no error — this project's
recurring bug class wearing a new hat. A service must not crash on one bad
ticket; an experiment must not continue past one.

The matching rule: **only known `AgentError`s become escalations** (HTTP 200 —
the ticket needs a human, and a 5xx would wrongly invite a retry). Anything
else returns 500, because laundering an unexpected bug into a plausible
"needs a human" response is that same failure shape again. Both branches are
proven by fault injection in `tests/test_service.py`, and HTTP parity is tested
directly: `/triage` reproduces the library's decision for all 9 adversarial
tickets and 15 of the 45. Checked against a real uvicorn process as well as
`TestClient`, `adv_08` returns tier 2, `tier1_conf` 0.3183, similarity 0.6124,
escalated — the adversarial gate's numbers exactly.

#### Why the Python orchestrator supersedes the n8n proposal

The "Pending" list below originally scoped this phase as *"likely via n8n
(wrapping the existing Python pieces as small local API endpoints, then
building a real n8n workflow with visual conditional routing, e.g. IF
confidence < threshold → escalate)"*. That plan was changed deliberately, and
the reasoning is the point:

- **It would have moved the escalation gate out of the test suite.** The RAG
  similarity threshold (0.67) is a measured, evidence-backed constant with its
  own calibration history in this README. An `IF confidence < threshold` node
  in a workflow tool is a *second copy* of that comparison, in a place where
  neither `pytest`, the goldens, nor the adversarial gate can reach it. This
  project's recurring bug class is a value that is wrong for its context but
  internally consistent; creating an untested duplicate of the one gate the
  whole thesis rests on is the most expensive possible instance of it.
- **A workflow JSON cannot be regression-tested against the goldens.** Every
  other routing change in this project was proven safe by reproducing
  `tests/goldens/*.json` exactly. Orchestration living in n8n would be the
  first routing logic in the system with no parity net under it.
- **The gap with Paper 1 closes on the orchestrator existing and being
  tested, not on which tool draws it.** Paper 1 proposed an
  escalation/orchestration design and never implemented or empirically tested
  it; `src/agent/orchestrator.py` is implemented, and its routing is covered by
  104 tests plus two fixed benchmark sets. Rendering the same logic in a
  workflow canvas adds presentation, not evidence.

What n8n was genuinely for — a visual, legible routing diagram — survives
without giving it any authority. **`POST /policy/rag-gate` returns the
escalation decision computed by the tested Python**, so a workflow tool can
branch on a boolean it did not compute and holds no threshold of its own. That
remains an open presentation step; it is explicitly not on the critical path,
and nothing about the system's behaviour depends on it.

### Phase 4 — drift detection, and the cost of a fixed reference

The recurring bug class in this project is a value or artifact that is wrong
for its context, internally consistent, and therefore silent. `artifacts.py`'s
three guards catch that at *load* time. Drift detection is the population-level
detector for the same class: a world that has moved away from what the gates
were calibrated on shows up as a shift **in the gates' own inputs** before it
shows up as a visibly wrong answer.

Nothing here gates production. `settings.drift.enabled` is `False`, no routing
decision reads any value in `settings.drift`, and — deliberately — **no window
size and no alarm threshold exist in config at all**, with a test pinning their
absence. Naming either before measuring the null false-alarm rate would be a
hand-tuned threshold wearing a lab coat, which is the thing this project has
rejected twice already (the in-distribution split; the 35-ticket calibration
set).

#### 4A — the history that did not exist, and the detector library

Three modules (`logging_setup.py`, `orchestrator.py`, `schemas.py`) described
decision records as "the raw input for drift detection". **They were never
persisted** — `configure_logging()` attached only a stderr handler, and every
evaluation path passed `emit_log=False`. Doc ahead of code, the same gap the BGE
clustering re-run closed. So 4A added an **opt-in** JSONL sink (default off,
nothing in the project enables it, and it refuses a logger level above INFO
because a starved sink records an empty history that reads as a quiet period)
plus `src/agent/drift.py`: pure functions over records, no I/O and no models, so
the detector is testable without loading BGE.

Three signals, kept apart on purpose:

- **Signal A (primary) — retrieval novelty.** Each record's top-1 similarity
  becomes a conformal p-value against the 175-ticket in-domain reference,
  reusing `conformal.conformal_p_values()` verbatim. Under exchangeability the
  p-values are super-uniform, so drift piles mass at low p.
- **Signal B — the rates the thesis rests on:** escalation rate, Tier-1 share
  (a *published* number — if it moves in deployment, that is itself a finding),
  and category mix.
- **Signal C — the config fingerprint**, reported in its own field. A window
  carrying an unexpected fingerprint is a **deployment fault, not a distribution
  shift**, and averaging the two together would bury it.

Building it produced four findings, each of which changed what 4B had to
measure. Two are worth restating here because they are the reason this phase has
a real result: **"the false-alarm rate is α by construction" was overstated** —
it holds marginally over calibration draws, not for one fixed n=175 reference,
where the per-ticket rate is Beta(17,159)-distributed (exact marginal 17/176 =
0.0966; calibration-conditional bound at δ=0.10 **0.126**) — and **the in-domain
reference escalates 0/175**, which makes any escalation-rate test against it
degenerate. The second forced a *second* reference for Signal B, built from the
deployment-distribution calibration set, which escalates **39/175 (22.3%)**.

#### 4B — audit, null, then power

The evaluation (`src/experiments/evaluate_drift_detection.py`, offline, no
Gemini quota, seed 42) was audited before it was run, against this project's own
rules. Worth recording from that audit:

- Records are collected by running each ticket through `pipeline.run()`, writing
  them through the **real** sink and reading them back through the **real**
  parser — so the detector is evaluated on exactly what a deployment would
  persist, not on a re-derivation of it.
- **Escalation is counted from `decision.escalated` at every layer** (the sink
  writes `result.decision.escalated`; `DecisionRecord.escalated` reads it with
  no default). No status enum appears anywhere in the drift code. This is the
  fifth bug instance's exact shape, and it is structurally absent.
- Every load-bearing count has a second derivation, and all of them agree:
  the Signal B reference's 39 escalations equal an independent count of
  similarities below 0.67; the sink-recorded similarities reproduce **all 24**
  published Phase 1 detection values (both score variants, four α); and
  benchmark 15/45 + adversarial 6/9 = **21 of 54** escalating reproduces the
  independently published Phase 2B eligibility split.
- The 175 in-domain reference scores are **all distinct**, which is what makes
  4A's "the planned split check could not fail" claim literally true rather than
  merely plausible.

**Three deviations from the approved plan, each because a planned null was
wrong**, all recorded in the output's own `deviations_from_plan` field: the
1,000-random-splits check could not fail (for distinct scores a split's flag
count is exactly Beta-binomial whatever the data); the first Signal B null
bootstrapped windows from the leftover half of each split, which
anti-correlates pool and reference and inflated every test's false-alarm rate;
and Signal A's power arm was moved to leave-one-out p-values against 174
reference tickets for the same reason.

##### The null false-alarm rates (Signal A, 20,000 windows per cell, alarm at p ≤ 0.05)

| W | α | marginal binomial (CI hi) | conditional binomial (CI hi) | KS (CI hi) |
|---:|---:|---|---|---|
| 25 | 0.01 | 0.0146 (0.0164) ✔ | **0.0146 (0.0164) ✔** | 0.0773 (0.0811) |
| 25 | 0.10 | 0.0406 (0.0434) ✔ | **0.0118 (0.0133) ✔** | 0.0794 (0.0832) |
| 50 | 0.05 | 0.0437 (0.0466) ✔ | **0.0169 (0.0188) ✔** | 0.1090 (0.1134) |
| 50 | 0.10 | 0.0690 (0.0726) | **0.0173 (0.0192) ✔** | 0.1035 (0.1078) |
| 100 | 0.01 | 0.0534 (0.0566) | **0.0209 (0.0229) ✔** | 0.1878 (0.1932) |
| 100 | 0.05 | 0.0795 (0.0834) | **0.0127 (0.0144) ✔** | 0.1804 (0.1858) |
| 200 | 0.05 | 0.1029 (0.1072) | **0.0170 (0.0189) ✔** | 0.3334 (0.3400) |
| 200 | 0.10 | 0.1253 (0.1299) | **0.0182 (0.0202) ✔** | 0.3250 (0.3315) |

*(✔ = eligible under the pre-registered rule: the upper end of the 95%
Clopper–Pearson interval on the measured rate must be ≤ 0.05. Full grid, all
four α × four window sizes, in* [`drift_evaluation_null.csv`](data/drift_evaluation_null.csv)*.)*

**The marginal test's false-alarm rate is not α, and the gap grows with the
window.** It is eligible only at W=25 (α ≤ 0.10) and W=50 (α ≤ 0.05); by W=200 it
runs 0.077–0.125, **two to two-and-a-half times nominal**. This is 4A's second
finding converted from an argument into a measurement: a long window is
precisely what resolves the fixed reference's Beta-distributed offset, so the
marginal null is the one thing a long window cannot be trusted with. The
calibration-conditional test is eligible in all 16 cells (0.0092–0.0226), which
is what testing against a 1−δ upper bound should look like.

**The KS test is never eligible, and it too degrades with window size**
(0.077 at W=25 → 0.33 at W=200). With 175 calibration points the p-values are
discrete on a 1/176 grid and super-uniform by construction, so a growing window
lets KS detect the *discreteness* rather than any drift. It stays in the report
as a descriptive read and must never be used as an alarm.

The F2 arm confirms the machinery underneath, on 100,000 synthetic draws:
closed-form marginal 0.0966 vs realised mean 0.0965, closed-form bound 0.1259 vs
realised 90th percentile 0.1257, and 0.099 of references above the bound against
a nominal δ = 0.10. The bound's conservatism is strongly α-dependent —
bound/marginal is 2.30 at α=0.01 but 1.20 at α=0.20. Three independent Signal A
implementations (scalar reference, `drift.novelty_signal`, vectorised matrix)
agree to under 1e-9 across 300 windows.

##### Signal B: the one-sample binomial is unusable, and Fisher is what rescues it

| W | escalation binomial | escalation Fisher | Tier-1 binomial | Tier-1 Fisher | category χ² |
|---:|---:|---:|---:|---:|---:|
| 25 | 0.0470 | **0.0335 ✔** | 0.0498 | **0.0293 ✔** | 0.1187 |
| 50 | 0.0823 | 0.0432 | 0.0785 | **0.0387 ✔** | 0.1673 |
| 100 | 0.1165 | 0.0500 | 0.1140 | **0.0403 ✔** | 0.2780 |
| 200 | 0.1690 | 0.0462 | 0.1698 | 0.0432 | 0.4597 |

The binomial treats the reference rate as truth when it is a 175-ticket
estimate, and pays for it: **0.047 at W=25 rising to 0.169 at W=200**, three
times nominal. The two-sample Fisher test, which carries the reference's
estimation noise, stays near or under nominal at every window and is eligible up
to W=100 for Tier-1 share. The category χ² test is never eligible (0.12 → 0.46):
seven categories against a 175-ticket reference violates the expected-count
conditions outright, so category mix is descriptive only.

**25 of 68 operating points are eligible.**

##### Power, among eligible points only, and provisional

Signal A's conditional binomial detects out-of-template contamination well.
Representative cells (detection rate, 2,000 windows each):

| W | α | 5% OOD | 10% OOD | 10% adversarial | 25% benchmark |
|---:|---:|---:|---:|---:|---:|
| 50 | 0.05 | 0.033 | 0.579 | 0.649 | 1.000 |
| 100 | 0.01 | **0.898** | 1.000 | 0.986 | 0.989 |
| 100 | 0.05 | 0.048 | 0.921 | 0.980 | 0.998 |
| 200 | 0.05 | 0.297 | 0.999 | 1.000 | 1.000 |

**Signal B has essentially no power on this data**, eligible though parts of it
are: `escalation_fisher` at W=25 reaches 0.913 only at 50% OOD contamination,
`tier1_fisher` never reaches the 80% target at any window or contamination, and
benchmark contamination is indistinguishable from its own null. That is a fair
result rather than a disappointing one — the rates move only when the *gate*
outcome changes, and benchmark tickets escalate at close to the deployment
reference's own rate.

##### Two findings that temper the verdict, and belong beside it

**The best-powered operating point is governed by a single reference ticket.**
At α=0.01 the order statistic is l=1, so the decision threshold *is* the
smallest calibration similarity. F3's leave-one-out sweep shows exactly one drop
of 175 moving every detection metric, and moving it a long way: adversarial
detection 0.667 → 0.889, OOD 0.844 → 0.911, benchmark 0.333 → 0.467. At α ≥ 0.05
(l = 8 or more) the metrics are stable — the only movement anywhere is
deployment flag rate at α=0.10, 0.514 → 0.520. **So the defensible candidate is
α=0.05 at W=100–200, not the strongest-power cell**: measured null 0.0127–0.0170,
≥80% detection at 10% OOD or adversarial contamination and 25% benchmark, and a
threshold resting on eight reference tickets rather than one.

**Realistic traffic reads as drift.** The deployment-register tickets — plain
English, disjoint from the benchmark, not drift by any construction — flag
against the in-domain reference at:

| α | flagged | rate | marginal null | conditional bound |
|---:|---:|---:|---:|---:|
| 0.01 | 38/175 | 0.217 | 0.0057 | 0.0131 |
| 0.05 | 75/175 | 0.429 | 0.0455 | 0.0663 |
| 0.10 | 90/175 | 0.514 | 0.0966 | 0.1259 |
| 0.20 | 113/175 | 0.646 | 0.1989 | 0.2381 |

Four to seven times the null at every level. **A Signal A monitor on this
reference would alarm continuously on legitimate traffic**, because the
reference is built from in-domain paraphrases of training rows and the register
shift alone saturates it. This is the same wall Phase 1's Finding 4 hit from the
coverage side, reached independently from the monitoring side: the binding
constraint is what the reference is made of, not the test applied to it.

##### Verdict: measured, not shipped

An eligible operating point exists and is now measured with its false-alarm rate
beside it — but **nothing was promoted.** `settings.drift.enabled` stays `False`,
no window size or alarm threshold was added to config, and the α=0.05 / W=100–200
candidate is recorded as a measurement, not a default. The realistic-traffic
result is why: a monitor whose false-alarm rate is 0.017 on exchangeable data and
~0.43 on the actual deployment register is not deployable, and the fix is a
reference drawn from deployment traffic, not a tuned threshold.

**Limitations, stated with the result rather than after it:**

- **Signal A's power numbers are provisional.** Their in-domain portion reuses
  the same 175 reference tickets (leave-one-out against 174); a held-out
  in-domain set is required before any power figure is quoted as final.
- **The Signal A null is a property of the procedure**, measured on synthetic
  exchangeable scores at n=175 — not of this project's data. Where the one fixed
  reference actually sits within that Beta law needs held-out tickets.
- **Signal B's nulls are parametric**, drawn from the reference rates, so they
  calibrate the test procedure rather than the data.
- **Signal B's 22.3% escalation rate is that of Gemini-generated
  benchmark-register tickets, not a measured production rate**, and its category
  mix is uneven (Storage 45, Database 41, Infrastructure 13) despite 25/category
  true labels — the same prediction skew 4A recorded for the in-domain reference.
- **Contamination is abrupt and out-of-template**, so this measures power against
  novel text, not the gradual shift a deployed monitor would actually face.
- **Selection risk is accumulating on the 45-ticket benchmark**, which now
  carries classification accuracy, conformal coverage, groundedness and drift.
- **Template-generated data makes the null unusually clean**, so every measured
  false-alarm rate here is optimistic.
- The escalation-rate test remains unavailable against the in-domain reference
  (0/175, degenerate), and the realistic-traffic and power arms both resample
  from fixed sets, so their windows are not independent draws.
- One RNG stream (seed 42) is threaded through every step, so these numbers
  reproduce only while the step order is unchanged.

Gate: nothing that already existed moved — `pytest` 150 passed, adversarial 9/9
with `data/adversarial_escalation_results.csv` byte-identical, goldens 45/45 and
9/9 exact, ablation baseline 32/45 = 71.11% with its CSV byte-identical, and no
`logs/`, `.jsonl` or `.npy` anywhere after the full run. Scripts:
`src/experiments/build_drift_reference.py` (add `--source deployment` for Signal
B's reference), `src/experiments/evaluate_drift_detection.py` (`--smoke` first).
Results: [`drift_evaluation_null.csv`](data/drift_evaluation_null.csv),
[`drift_evaluation_power.csv`](data/drift_evaluation_power.csv),
[`drift_evaluation_summary.json`](data/drift_evaluation_summary.json).

### Phase 5B — the honest ablation: what the cascade is actually worth

**Measurement only. No production threshold, artifact, benchmark or golden
changed.** Cascade stays 0.50, RAG 0.67, clustering 0.80;
`settings.conformal.enabled` and `settings.drift.enabled` stay `False`.

#### The question, and why the old answer was the wrong one

The published ablation credited the cascade threshold with **+35.6 accuracy
points**. That figure is `baseline (32/45)` minus `no-cascade (16/45)`, and
`no-cascade` is **Tier-1 (TF-IDF) answering every ticket**. So it compares a
BGE-backed system against a TF-IDF-only system and calls the difference a
cascade effect. It is a representation-gap number wearing a cascade label —
the control it needed, *the strong model answering everything*, had never been
run.

Phase 5B added `--mode tier2-only`, ran it on two sets, tested the difference
as the **paired** comparison it actually is, and replaced the cascade's
latency proxy with real warm inference timing.

#### First: were the two published numbers even comparable?

Before changing anything, the prerequisite check. The README quotes **33/45**
for BGE alone and **32/45** for the cascade. If the 33/45 came from
`train_embeddings_comparison.py`'s in-memory 80/20-split model rather than the
production Tier-2 artifact, the one-ticket gap would not be a cascade effect
at all and the comparison would not be like-for-like — a seventh instance of
this project's recurring bug class. Derived four ways:

| Derivation | Result |
|---|---|
| Production `ticket_classifier_bge-base-en-v1-5.joblib`, Tier-2 alone on the 45 | **33/45** |
| Refit under `train_embeddings_comparison.py`'s own recipe, from its cached `embeddings_bge.npy` | **33/45** |
| The two classifiers compared directly | agree **45/45** ticket-by-ticket; `max abs coef difference = 0.0`; identical `classes_` |
| Cascade @0.50 recomputed, vs the published `ablation_baseline_results.csv` | **32/45**, matching **45/45** on `predicted` |

**They are the same classifier, bit-identical.** Both scripts fit
`LogisticRegression(max_iter=1000)` on
`train_test_split(test_size=0.2, random_state=42, stratify=y)` over the same
BGE encoding of `title + " " + description`. **No seventh bug**; the
comparison was sound. The problem was never the numbers, it was the claim
attached to them.

#### Result 1 — the cascade costs one ticket, on both sets

| Set | Cascade (0.50) | Tier-2 only | No cascade (Tier-1 only) | Difference | Exact McNemar |
|---|---:|---:|---:|---:|---:|
| 45-ticket benchmark | 32/45 (71.11%) | **33/45 (73.33%)** | 16/45 (35.56%) | −1 ticket (−2.22 pts) | b=0, c=1, **p = 1.000** |
| 175-ticket deployment set | 131/175 (74.86%) | **132/175 (75.43%)** | 91/175 (52.00%) | −1 ticket (−0.57 pts) | b=1, c=2, **p = 1.000** |

The cascade is **not** an accuracy improvement over the strong model alone. It
is one ticket worse on both sets, and on both sets that difference is **well
inside what the sample can resolve** — with 1 and 3 discordant pairs, an exact
McNemar returns p = 1.000 either way. The honest statement is *indistinguishable
on accuracy*, not *worse*; the −1 is noise, and so was any reading of the
+35.6.

**Limitation, beside the number:** at n=45 with one discordant pair, this test
has essentially no power — it could not detect a real effect of this size if
one existed. That cuts both ways, and it is precisely why the paired test
matters: the raw one-ticket gap was never evidence of anything. The
175-ticket set adds resolution but only reaches 3 discordant pairs.

**Limitation on the second set:** `deployment_calibration_tickets.json` is
Gemini-generated deployment-register text, **not production traffic**, and the
same 175 tickets already carry Phase 1 Finding 4's conformal calibration.
Using them here adds another use of an already multiply-used set — the same
selection-risk accumulation already recorded for the 45-ticket benchmark.

#### Result 2 — where the cascade actually acts

| Set | Tier-1 answered | Tier-1 correct there | Tier-2 would have been correct there |
|---|---:|---:|---:|
| 45-ticket benchmark | 4 / 45 (8.9%) | 2 | 3 |
| 175-ticket deployment set | 33 / 175 (18.9%) | 27 | 28 |

At 0.50, Tier-1 keeps between 9% and 19% of tickets — and on the benchmark it
is **right on only half of the ones it is most confident about**. Every
discordant ticket is, by construction, one Tier-1 kept; the comparison script
asserts this rather than assuming it, since a discordant pair on an escalated
ticket would mean both runs used the same Tier-2 prediction and therefore that
one of the CSVs is stale.

The single benchmark discordance is instructive: benchmark `#38`
(`tier1_conf = 0.5481`) is a VPN-won't-connect ticket. Tier-1 answers
**Database**; Tier-2 answers **Network**, correctly. Two of the three
deployment-set discordances have the same shape — Tier-1 confidently answering
**Database** on a Security ticket and on another VPN ticket. Tier-1's
confident errors are not scattered; they collapse toward one class.

**This is Finding 1's mechanism showing up as a routing error.** Three of the
four discordant tickets across both sets are Tier-1 assigning **Database** to a
Network or Security ticket at 0.51–0.57 confidence — lexical overconfidence on
out-of-template phrasing. That is the same failure Finding 1 measures as
coverage collapse: TF-IDF scores are built from surface vocabulary, so text
phrased outside the training templates lands in a region where its confidence
is no longer calibrated. Finding 1 sees it as a 23.3-point conformal coverage
loss for Tier-1 against BGE's 1.1; here it is four tickets the cascade kept and
should not have. Same cause, two instruments — which is why raising the cascade
threshold would not fix it, and why the gate that matters is the one measured
against a calibration set rather than tuned by hand.

#### Result 3 — latency, measured warm rather than inferred from a fit

The cascade's efficiency claim previously rested on Phase 3A's **1.56 s to fit
Tier-1 versus 0.014 s to load it**. That is a startup number and says nothing
about per-ticket cost. Measured properly — models loaded once, 20 warmup
iterations discarded, 200 timed single-ticket runs per tier, batch size 1:

| | median | p95 | mean |
|---|---:|---:|---:|
| Tier-1 (TF-IDF + LogReg) | **1.04 ms** | 1.74 ms | 1.13 ms |
| Tier-2 (BGE + LogReg) | **156.40 ms** | 227.05 ms | 152.82 ms |

**Tier-2 costs 151× Tier-1 per ticket.** But the cascade does not pay Tier-1's
price — it *always* runs Tier-1 and *then* runs Tier-2 whenever Tier-1 is
below 0.50, so its expected cost is `tier1 + (1 − share) × tier2`:

| Set | Tier-1 share | Cascade expected | Tier-2 alone | Saving |
|---|---:|---:|---:|---:|
| 45-ticket benchmark | 8.9% | 143.54 ms | 156.40 ms | **−12.87 ms (8.2%)** |
| 175-ticket deployment set | 18.9% | 127.95 ms | 156.40 ms | **−28.46 ms (18.2%)** |

So the cascade **does** buy something real: 8–18% of median per-ticket
latency. It is a cost trade, not an accuracy gain.

**Limitation, beside the number:** one machine, one process, batch size 1, CPU
only (13th Gen Intel Core i5-1334U, 12 logical CPUs, Python 3.14.3, Windows
11), no competing load controlled for. This bounds per-ticket inference cost
on this hardware; it is not a throughput or served-latency measurement. The
Tier-1 share is also a property of these two sets, not a deployment rate.

#### What this means for the paper's framing

The cascade is a **latency optimisation that costs a statistically
undetectable amount of accuracy** — roughly 8–18% less compute per ticket for
somewhere between −1 ticket and nothing. That is a perfectly respectable
engineering result, and it is *not* the contribution this project has.

The contribution is the **calibrated escalation gates**: the RAG similarity
gate at 0.67 (which the ablation shows prevents 6 concrete
confidently-wrong Gemini resolutions on the adversarial set alone), conformal
novelty detection with a *measured* false-escalation rate the hand-tuned
threshold never had, and the discipline of refusing to ship a detector without
its null. Those are the results that survive contact with a control. The
cascade should be described as what it is — a cheap-tier cost saving — and the
paper's accuracy story should rest on the gates.

This is also the fifth time a control changed the reading of a result in this
project: the ablation's own BGE correction, the Phase 2A pre-registered rule
that would have promoted whichever model merged more, the 2B judge whose 90.9%
agreement concealed κ = −0.042, 4B-1's two unusable drift tests, and now this.
**An uncontrolled comparison here has never once survived being controlled.**

#### One thing fixed along the way

`run_ablation_study.py` did not load through `artifacts.load_artifacts()`. It
carried four loaders of its own, and one of them **refitted Tier-1 from
`synthetic_tickets.csv` on every run** — a locally-derived model that would
stay internally consistent while silently diverging from the artifact
production serves, and the precise shape of the bug class that has now hit
this project six times. It is now migrated, which also gives the script the
three hard guards it never had (index/metadata alignment, encoder dim ==
index.d == configured dim, Tier-1 manifest vs dataset).

The migration was verified parity-preserving **before** it landed: persisted
Tier-1 and the old local refit agreed to `max |Δ tier1_conf| = 0.0` across the
45, both rounding to the published CSVs' 6 decimals with zero mismatches.
After it landed, all three published ablation CSVs regenerate **byte-identical**.

#### Gate

Re-run, not quoted: `pytest` **169 passed** (153 + 16 new), adversarial
**9/9** with `data/adversarial_escalation_results.csv` byte-identical, goldens
**45/45 and 9/9** exact, ablation baseline **32/45 = 71.11%** and no-cascade
**16/45** with both CSVs byte-identical, and no `.npy` anywhere after the full
run.

Scripts: `src/experiments/run_ablation_study.py` (`--mode tier2-only`,
`--set deployment175`), `src/experiments/compare_cascade_vs_tier2.py`,
`src/experiments/measure_inference_latency.py`. Results:
[`ablation_tier2-only_results.csv`](data/ablation_tier2-only_results.csv),
[`cascade_vs_tier2_mcnemar_benchmark45.csv`](data/cascade_vs_tier2_mcnemar_benchmark45.csv),
[`cascade_vs_tier2_mcnemar_deployment175.csv`](data/cascade_vs_tier2_mcnemar_deployment175.csv),
[`inference_latency_13th-gen-intel-r-core-tm-i5-1334u.csv`](data/inference_latency_13th-gen-intel-r-core-tm-i5-1334u.csv),
plus the three `*_deployment175.csv` ablation files.

### Phase 5C — zero-shot LLM baselines: what the training corpus is actually worth

**Measurement only. No production threshold, artifact, benchmark or golden
changed.** Cascade stays 0.50, RAG 0.67, clustering 0.80;
`settings.conformal.enabled` and `settings.drift.enabled` stay `False`.

#### The question

Phase 5B left the project's accuracy claim resting on Tier-2 alone (BGE +
LogReg, 33/45). That number had never been compared against the baseline any
reviewer raises first: **just ask an LLM.** Phase 5C runs that comparison on
both fixed benchmarks, with two independently-vendored models so the answer
cannot be an artifact of one provider.

**One prompt, two backends.** `build_prompt()` in
`run_zeroshot_baselines.py` is backend-agnostic, so both arms receive
byte-identical text by construction. That is not taken on trust: the
`prompt_sha256` recorded in each arm's cached raw response was compared ticket
by ticket, and **all 59 matched in both directions**. A script per vendor is
how two baselines quietly stop being comparable — the same failure 5B found in
the ablation.

**The parser never coerces.** A near-miss like "Networking" is recorded as
unparseable rather than mapped onto "Network", so an unparseable rate is a real
measurement and not an artefact of being strict.

#### Result 1 — both zero-shot LLMs match or beat the trained classifier

| Model | benchmark45 | 95% Wilson | benchmark14 | 95% Wilson | Unparseable |
|---|---:|---|---:|---|---:|
| Zero-shot Gemini (`gemini-flash-lite-latest`) | **40/45 (88.89%)** | [76.50%, 95.16%] | **14/14 (100%)** | [78.47%, 100%] | 0/59 |
| Zero-shot Qwen2.5-3B-Instruct (local, CPU) | **34/45 (75.56%)** | [61.33%, 85.76%] | **12/14 (85.71%)** | [60.06%, 95.99%] | 0/59 |
| Trained Tier-2 (BGE + LogReg, 3,200 rows) | 33/45 (73.33%) | — | 10/14 (71.43%) | — | n/a |

Paired, as exact McNemar on the discordant pairs:

| Comparison | Set | b / c | Δ | Exact McNemar |
|---|---|---|---:|---|
| Gemini vs Tier-2 | benchmark45 | 8 / 1 | +7 | **p = 0.0391 — distinguishable** |
| Gemini vs Tier-2 | benchmark14 | 4 / 0 | +4 | p = 0.125 |
| Qwen-3B vs Tier-2 | benchmark45 | 7 / 6 | +1 | p = 1.000 |
| Qwen-3B vs Tier-2 | benchmark14 | 4 / 2 | +2 | p = 0.688 |
| Qwen-3B vs Gemini | benchmark45 | 1 / 7 | −6 | p = 0.0703 |
| Qwen-3B vs Gemini | benchmark14 | 0 / 2 | −2 | p = 0.500 |

**The honest readings, which differ by pair.** Gemini is distinguishably better
than the trained classifier on the 45. Qwen-3B is **indistinguishable from** it
— +1 and +2 tickets at p = 1.000 and p = 0.688 — which is *not* the same as
"better", and must not be written as if it were. Qwen-3B is 6 tickets below
Gemini on the 45 at p = 0.070: suggestive, not significant at α = 0.05.

**What survives all of it:** a 3B model running on a laptop CPU, with no
training on this corpus at all, is not beaten by a classifier fitted on 3,200
of its rows. **That direction holds for both vendors**, which is what part 2
was for.

#### Result 2 — Qwen-3B's deficit is one category, not a diffuse weakness

Per-category recall on the 45, from the confusion matrices:

| Category | Support | Gemini recall | Qwen-3B recall |
|---|---:|---:|---:|
| Database | 5 | 60% | **0%** |
| Infrastructure | 6 | 50% | 50% |
| Security | 6 | 100% | 83% |
| Storage | 7 | 100% | 86% |
| Access Management | 7 | 100% | 86% |
| Application | 7 | 100% | 100% |
| Network | 7 | 100% | 100% |

**Qwen-3B never emits "Database" once in 59 tickets.** All five Database
tickets on the 45 go to Application (4) or Network (1), and both on the 14 go
elsewhere. That single category accounts for **3 of the 6 tickets** separating
it from Gemini. Its Application precision is correspondingly 47% — Application
is where the misroutes land.

**Both LLMs fail Infrastructure at exactly 50%**, independently. That is the
symptom-vs-cause family the README already names, including the "status page
green but the service is down" ticket every trained model also misses. A
failure both vendors and the trained classifier share is a property of the
*tickets*, not of any model.

#### Result 3 — latency, and what it costs to buy those tickets

| System | Median per ticket | vs Tier-2 |
|---|---:|---:|
| Trained Tier-2 (BGE + LogReg), warm, batch 1 | **0.156 s** (Phase 5B) | 1× |
| Zero-shot Gemini, hosted | 0.81 s | 5.2× |
| Zero-shot Qwen2.5-3B, local CPU | 6.98 s | 44.7× |

The accuracy the LLMs buy is not free, and the local arm is not free of a
network round trip by being free of a vendor.

#### Limitations, beside the numbers

- **3B is a floor for the non-Gemini family, not a fair ceiling.** Qwen-3B was
  chosen because it fits the available RAM, not because it is the best
  open-weight model. Its 6-ticket gap to Gemini on the 45 **cannot be
  attributed** — this run cannot separate *"the model is too small"* from
  *"Gemini benefits from having authored the evaluation data"*. Both are live,
  and nothing here distinguishes them. Do not write the gap as evidence of
  either.
- **The benchmark is Gemini-generated.** `novel_tickets_expanded.json` was
  produced by Gemini and human-label-reviewed, so the Gemini arm is being scored
  on text from its own model family. This is precisely why part 2 exists: **the
  Qwen arm is a partial control for that confound**, and the
  "zero-shot ≥ trained classifier" direction survives removing the authorship
  advantage. Gemini's *additional* margin over Qwen does not.
- **Both sets are small.** At n=45 a single ticket is 2.2 points, and the
  Wilson intervals above overlap heavily. n=14 resolves almost nothing — its
  100% carries a lower bound of 78.5%.
- **Zero-shot vs trained is not like-for-like**, in exactly the way 5B found
  baseline-vs-Tier-1-only was not. Few-shot prompting was deliberately out of
  scope, as was any fine-tuning of either LLM.
- **The latency comparison is not hardware-matched.** Qwen ran CPU-only on an
  i5-1334U with integrated graphics; Gemini ran on hosted accelerators. Both
  arms pinned temperature 0 and constrained decoding to JSON, but Ollama
  additionally fixed `seed=42` and capped output at 48 tokens.
- **Licensing.** Qwen2.5-3B ships under the *Qwen Research* licence, not
  Apache-2.0 (the 7B is). If the paper describes this baseline as "freely
  available", that wording needs checking against the licence first.

#### What this measures — the framing that governs the whole phase

**5C measures what a 3,200-row, template-generated training corpus is worth on
out-of-template phrasing. It is not "LLMs beat the pipeline".** A zero-shot LLM
against a classifier trained on 3,200 rows is not a method comparison; the
quantity actually being estimated is how much that training data buys. The
answer here is: **on out-of-template text, not more than a 3B model already
knows without it.**

This is the **third independent view of the named finding** below, and the
first from the *training* side rather than the calibration/reference side. The
mechanism is the same register mismatch: the corpus is template-generated, the
benchmarks are not, and the advantage the corpus confers does not survive the
crossing. Phase 1 found it for a coverage *guarantee*, Phase 4B-1 for a
*monitor*, and 5C now for *accuracy itself*. (Phase 2A remains a **related but
distinct** corpus limitation — training-data redundancy limiting what can be
evaluated — and is still not an instance of this mechanism.)

#### Gate — re-run, not quoted

`pytest` **246 passed** (214 + 32 new), adversarial escalation **9/9 PASS**
with `adversarial_escalation_results.csv` byte-identical, golden parity
**45/45 and 9/9** exact, ablation baseline **32/45 = 71.11%** with 9/9
escalations, no published result CSV modified, and no `.npy` anywhere after
the full run.

**Parity check on the edited comparison script:** `compare_zeroshot_vs_tier2.py`
gained an `--against zeroshot` mode, and both already-committed Gemini
comparison CSVs were regenerated with `--force` and verified **byte-identical**
before any local-model time was spent. One trap caught doing it: part 1 wrote
`tier2` in the `direction` column but `tier2only` in the field names, so
deriving one from the other would have silently rewritten both files.

**Quota:** part 1 spent 59 Gemini calls (2026-09-21), reconciled two ways.
**Part 2 spent zero** — Ollama is local, and part 1's 59 responses were re-read
from cache.

Scripts: `src/experiments/run_zeroshot_baselines.py` (`--backend ollama`,
`--allow-low-ram`), `src/experiments/compare_zeroshot_vs_tier2.py`
(`--against zeroshot`), `src/experiments/summarize_zeroshot_baselines.py`.
Results:
[`zeroshot_ollama_qwen2-5-3b-instruct_benchmark45.csv`](data/zeroshot_ollama_qwen2-5-3b-instruct_benchmark45.csv),
[`zeroshot_ollama_qwen2-5-3b-instruct_benchmark14.csv`](data/zeroshot_ollama_qwen2-5-3b-instruct_benchmark14.csv),
the four `zeroshot_vs_*_mcnemar_*.csv` reports and the four
`zeroshot_summary_*.csv` files.

### Phase 6A - conformal deferral vs a confidence threshold

**Measurement only. `settings.conformal.enabled` stays `False`** - and the result
below is not an argument to change that. Cascade 0.50, RAG 0.67, clustering 0.80
unchanged; no artifact, benchmark or golden touched.

#### The question Phase 1 could not answer

Phase 1 measured whether conformal's *coverage guarantee* holds. That is not the
operational question. The live cascade defers on a raw confidence threshold, and
the open question was whether conformal should replace it - which coverage
cannot settle, because coverage says whether the guarantee holds, not whether
the rule **defers on the right tickets**.

#### The rules reduce to rankings, and the reduction is proven, not asserted

A deferral rule is a scalar score plus a threshold, so a risk-coverage curve
depends only on the *ordering* the score induces. Working the sets out against
`src/agent/conformal.py`:

| Rule | Prediction set | Singleton while | Ranks by |
|---|---|---|---|
| Confidence (incumbent) | - | `p1 >= tau` | **p1** |
| LAC conformal | `{y : p(y) >= 1-q}` | `p2 < 1-q <= p1` | **p2** |
| APS conformal | `{y : cum(y) <= q}` | `p1 <= q < p1+p2` | **p1+p2** |

**LAC-conformal deferral ranks by the second-largest probability where
confidence ranks by the largest** - genuinely different orderings, so this is a
real comparison and not a re-parameterisation. Two consequences: the ranking
comparison is **calibration-free** (q cancels, so it is not exposed to Phase 1's
contamination), and **"defer unless singleton" is not monotone in alpha**,
because both scores admit an *empty* set, which is also a deferral.

`tests/test_deferral_rules.py` does not check this algebra against itself: it
brute-forces real sets through `conformal.predict_sets()` over a grid of
quantiles and asserts the scalar rule says accept **iff** the real set is a
singleton. If the derivation were wrong, every number below would be wrong while
staying perfectly self-consistent - this project's recurring bug class.

**The control that would otherwise be missing:** **margin (p1 - p2)**, the
standard selective-prediction baseline. Confidence alone is a weak opponent, and
conformal beating only that would repeat Phase 5B's mistake.

#### Result 1 - on the axis that matters, there is no signal anywhere

The live 0.50 gate's operating coverage was **measured, not assumed**: Tier-1
answers **8.9% (4/45)** of the benchmark and **18.9% (34/175)** of the deployment
set. Risk at that coverage is the pre-registered primary metric.

**All 12 comparisons (3 challengers x 2 tiers x 2 sets) return "no signal on the
gated axis".** Not one paired-bootstrap interval excludes zero.

But the stronger statement is about *why*, and it is not "the rules are equal":

| Tier / set | Risk at the live gate, all four rules | Gated axis measurable? |
|---|---|---|
| tier1 / benchmark45 | 0.500 (2 errors in 4 accepted) | **No - degenerate** |
| tier2 / benchmark45 | 0.000 | **No - degenerate** |
| tier2 / deployment175 | 0.000 | **No - degenerate** |
| tier1 / deployment175 | 0.147-0.206 (n=34) | Yes |

**In three of four configurations every rule accepts essentially the same
tickets at the live gate's coverage, so the test has no resolution at all.** The
script detects and records this rather than leaving it to a reader - the same
treatment Phase 4B-1 gave its degenerate escalation-rate test, which reports
`None` rather than p = 0. Only `tier1/deployment175` provides a real test, and
there the differences are inside noise.

#### Result 2 - AURC manufactures findings the gated axis does not support

AURC shows a significant effect in **3 of 12** comparisons - and they contradict
each other:

| Comparison | dAURC vs confidence | 95% bootstrap CI | Reading |
|---|---:|---|---|
| margin, tier2 / benchmark45 | **-0.0207** | [-0.0477, -0.0006] | margin *better* |
| margin, tier1 / deployment175 | **+0.0121** | [0.0003, 0.0249] | margin *worse* |
| lac, tier1 / deployment175 | **+0.0556** | [0.0242, 0.0899] | conformal *worse* |

Margin "wins" on one set and "loses" on the other. **This is the cost-asymmetry
failure caught in the act**: an ungated average produces three
publishable-looking effects, in inconsistent directions, where the axis the
project actually gates on shows nothing. It is the same shape as the
resolution-clustering promotion rule that would have rewarded whichever
configuration merged more.

Full AURC table (oracle in brackets - lower is better):

| Tier / set | confidence | margin | lac | aps | (oracle) |
|---|---:|---:|---:|---:|---:|
| tier1 / benchmark45 | 0.5339 | 0.5452 | 0.5935 | 0.5355 | (0.2830) |
| tier2 / benchmark45 | 0.1625 | 0.1421 | **0.1209** | 0.1903 | (0.0401) |
| tier1 / deployment175 | 0.2919 | 0.3039 | 0.3473 | **0.2900** | (0.1408) |
| tier2 / deployment175 | 0.1026 | **0.1006** | 0.1059 | 0.1157 | (0.0332) |

LAC has the best AURC on tier2/benchmark45 (0.1209 vs 0.1625) but its interval
[-0.1003, 0.0031] includes zero, and it is the **worst** rule on both Tier-1
configurations. No rule wins consistently across both tiers and both sets.

#### Result 3 - where conformal can actually be operated

The curves above are calibration-free rankings. These are the points the real
procedure lands on, using the published Phase 1 calibration (clean, marginal).
The figure is the singleton rate, i.e. the fraction auto-routed:

| Tier / set | Score | a=0.01 | a=0.05 | a=0.10 | a=0.20 |
|---|---|---:|---:|---:|---:|
| tier2 / deployment175 | LAC | 4.0% | 18.3% | 34.3% | 78.9% |
| tier2 / deployment175 | APS | 0.0% | 4.6% | 8.6% | 16.6% |
| tier1 / deployment175 | LAC | 1.1% | 6.9% | 14.3% | 20.6% |

Two things follow. **alpha=0.01 auto-routes essentially nothing** (0-4%), so the
tightest guarantee is operationally useless here. And **APS is far more
conservative than LAC at every alpha**, which matters because APS is the variant
Phase 1 preferred for adaptivity. At the one place conformal lands near the live
gate's coverage - Tier-2 LAC at alpha=0.05, 18.3% - its risk among singletons is
0.000, but so is the incumbent's at that coverage, which is precisely the
degeneracy above.

**The non-monotonicity is real and measured:** APS at alpha=0.20 returns **13.7%
empty sets** on deployment175 and 13.3% on the benchmark. Empty sets are
deferrals, so the singleton rate is *not* monotone in alpha, and any future
promotion must treat "defer unless singleton" as a two-sided condition rather
than a threshold.

#### Verdict

**Do not promote conformal to the live deferral gate**, and note that this is
*not* a finding that conformal is worse. The honest statement is: **there is no
evidence it defers better on the axis this project gates on, and in three of four
configurations the data cannot answer the question at that operating point at
all.** The only statistically significant effects belong to a simpler baseline
(margin), point in opposite directions on the two sets, and are on a metric the
project has decided not to gate on.

#### Limitations, beside the numbers

- **The primary metric is degenerate in 3 of 4 configurations.** At 8.9% coverage
  on n=45 the gate accepts **4 tickets**; no comparison can resolve anything
  there. That is a statement about the evaluation sets' size and the gate's very
  low operating coverage, not about the rules.
- **n=45 resolves almost nothing.** A single ticket is 2.2 points.
- **deployment175 is Gemini-generated deployment-register text, not production
  traffic**, and it is now *multiply* used - Phase 1 Finding 4's conformal
  calibration, Phase 5B's ablation, and now this. Each additional use weakens it
  further as independent evidence.
- **The operating-point overlay uses Phase 1's calibration**, which was fitted on
  the contaminated in-domain 175. The config fingerprint is checked before the
  overlay is drawn, but contamination is a property of that set and is not
  repaired here.
- **The secondary coverage grid was extended downward after the first run**, once
  the live gate was measured at 8.9%/18.9% and the original (50/70/80/90%) grid
  turned out not to span the operating point at all. The **primary** metric -
  risk at the measured operating coverage - was pre-registered and is unchanged,
  and it is what the verdict rule reads.
- **Bootstrap intervals at n=45 are wide**, and a paired bootstrap over 45 points
  resamples a small number of distinct tickets many times.

#### Gate - re-run, not quoted

`pytest` **261 passed** (246 + 15 from 6A), adversarial escalation **9/9 PASS**
with its CSV byte-identical, golden parity **45/45 and 9/9**, ablation baseline
**32/45 = 71.11%** with 9/9 escalations, no published result CSV modified, no
stray `.npy`.

**Guard that fired correctly:** every curve's coverage-1.0 endpoint is asserted
against the published accuracy (16/45, 33/45, 91/175, 132/175) and is **fatal**
on mismatch - the independent recount that would catch probabilities which are
not the ones the published results came from.

Script: `src/experiments/compare_deferral_rules.py`. Results:
[`deferral_rule_summary.csv`](data/deferral_rule_summary.csv),
[`deferral_risk_coverage.csv`](data/deferral_risk_coverage.csv),
[`deferral_conformal_operating_points.csv`](data/deferral_conformal_operating_points.csv),
plus four `deferral_risk_coverage_{tier}_{set}.png` figures.

### Phase 6B - weighted conformal under shift: a partial repair, not a correction

**Measurement only. `settings.conformal.enabled` stays `False`.** Cascade 0.50,
RAG 0.67, clustering 0.80 unchanged; no artifact, benchmark or golden touched.
Offline, zero Gemini calls.

#### The question

Finding 1 measured that a conformal predictor calibrated on the in-domain 175
loses **23.3 coverage points** on the 45-ticket benchmark for Tier-1 - 10.3 s.d.
below nominal, against a +-2 s.d. band of 0.045. Phase 1 Finding 4 then showed
that **distribution matching** - rebuilding the calibration set at deployment
register, matched on size and class balance - recovers only ~38% of that
shortfall (-0.233 -> -0.144) and leaves it six s.d. outside nominal.

6B asks the successor question: can **covariate reweighting** do what
distribution matching could not? Weighted split conformal (Tibshirani et al.
2019; Barber et al. 2022, *Conformal prediction beyond exchangeability*)
reweights calibration scores by the covariate likelihood ratio between the test
and calibration distributions. If the shift is a covariate shift, reweighting is
the principled repair.

#### Pre-registered, before any result was seen

Written into the script's docstring before the first run, because 6A's primary
metric turned out to have no resolution in 3 of 4 configurations and that was
discovered *after* the run.

- **Primary metric:** Tier-1 benchmark-45 **marginal coverage** gap at
  alpha = 0.10, weighted vs unweighted, judged against the +-2 s.d. band.
- **Answerability, decided in advance:** 6B is *not* on 6A's wall. 6A's gated
  axis was selective risk at low coverage, where the benchmark admitted 4
  tickets. 6B's axis is marginal coverage over **all 45** - no coverage
  restriction - and Finding 1's gap already resolved there.
- **Tier-2 declared unanswerable in advance.** Its unweighted gap (-0.011) is
  already inside the band, so there is no headroom for an improvement to show.
  Tier-2 is a **sanity check**, never a finding.
- **Degeneracy rule:** report "no resolution", naming the condition, if
  `n_eff < 50`, or cross-fitted domain AUC `>= 0.95`, or weighted and unweighted
  sets are identical. No secondary or averaged statistic is promoted in place of
  a primary that cannot resolve - 6A's AURC lesson, written into the rule.

#### Result 1 - reweighting improves coverage measurably but does not repair it

Density ratio from a **cross-fitted** (5-fold, seed 42) logistic-regression
domain classifier separating the in-domain 175 from the deployment 175, the
latter used **unlabeled as the target proxy and never scored**. Fitted in BGE
space and TF-IDF space separately. Clip variants: none, p95, p99.

| Space | Cross-fitted AUC | n_eff | Gap at alpha=0.10 | Change | Residual |
|---|---|---|---|---|---|
| (unweighted) | - | 175 | **-0.2333** | - | 10.3 s.d. out |
| BGE | 0.9908 | 145.0-153.9 | -0.1667 | +0.0667 | **blocked by pre-registration** |
| TF-IDF | 0.9295 | 124.4-129.1 | **-0.1222** | **+0.1111** | **5.4 s.d. out** |

All three clip variants give the same gap in each space; clipping barely matters
because `n_eff` never collapsed.

Both pre-registered readings are reported, because **the pre-registered wording
turned out to be ambiguous** and the two readings disagree here:

- **(a) the change exceeds the band: 3/3 resolving configurations.**
- **(b) the residual gap falls inside the band: 0/3.**

Reading (a) alone would license "the shift is correctable by covariate
reweighting". It is not. Coverage is still **5.4 s.d. below nominal** after
reweighting. The ambiguity is disclosed rather than resolved in whichever
direction flatters the result - picking one after seeing the numbers is
precisely what the pre-registration exists to prevent.

**The honest claim: a partial recovery, not a correction.** Post-hoc and
descriptive, the recovery fraction is **47.6%**, against distribution matching's
~38%. Two independent repair strategies, reached by different mechanisms, both
partial, both leaving coverage 5-6 s.d. outside nominal. **This sharpens the
named finding rather than overturning it:** if neither matching the calibration
distribution nor reweighting it closes the gap, the constraint is a property of
the corpus, not of the method - which is exactly what the named finding claims.

> **Do NOT write that reweighting recovers more than distribution matching.**
> The two recovery percentages (47.6% vs ~38%) are **post-hoc**, and the
> quantity that matters is the **residual gap**: **-0.122 vs -0.144**, a
> difference of **0.022 - inside the +-2 s.d. band of 0.045**. The two
> strategies are **statistically indistinguishable** on this evidence. The
> permitted claim is that **both are partial and both leave the gap 5-6 s.d.
> outside nominal**; an ordered comparison between them is not supported.
> Percentages of a shortfall exaggerate here, because dividing by a small base
> inflates an apparent margin between residuals that are within noise of each
> other.

#### A second finding: separability does not imply score shift

**BGE separates the calibration set from the deployment set MORE easily than
TF-IDF does - cross-fitted domain AUC 0.9908 vs 0.9295 - yet BGE is the space
whose conformal coverage transfers** (Tier-2's unweighted gap is -0.011, inside
the band, where Tier-1/TF-IDF's is -0.233). **Separability of two distributions
in a representation therefore does not imply that the scores computed in that
representation shift.** A domain classifier can tell the two corpora apart
almost perfectly while the classifier's own nonconformity scores remain
exchangeable enough for the guarantee to hold.

This is worth stating because the intuitive inference runs the other way -
"the embedding can tell them apart, so the calibration will not transfer" - and
it is wrong here. Cross-reference **Finding 1** (conformal coverage transfers
for Tier-2 but not Tier-1: BGE loses 1.1 coverage points on the 45-ticket
benchmark where TF-IDF loses 23.3, against a noise band of 4.5). The two
measurements are about different things: domain AUC measures whether the
*inputs* are distinguishable, coverage transfer measures whether the *scores*
are exchangeable, and 6B shows the first does not predict the second.

#### Result 2 - the production representation is the degenerate one

The BGE arm triggered a pre-registered degeneracy condition: **cross-fitted
domain AUC 0.9908**. The in-domain and deployment sets are very nearly
**perfectly separable in the embedding space Tier-2 actually scores in**. A
density ratio is ill-posed when the two domains barely overlap, which is why
that condition was registered in advance.

This matters twice over. First, it blocks the BGE numbers from being read as a
result - and they are the *flattering* ones: BGE at alpha = 0.05 moves the gap
from -0.1056 to **+0.0056**, essentially exactly nominal. That is in the CSV and
is **not** a finding. Quoting it would be the 6A AURC error repeated.

Second, an AUC of 0.9908 is itself a direct, quantitative statement of the named
finding: the two distributions are near-disjoint in the production
representation. The separability *is* the constraint.

#### Result 3 - the Tier-2 sanity check is NOT clean

Declared in advance as a check that weighting must not break, and it did not
come back clean. Of 18 Tier-2 configurations: **7 made |gap| worse, 5 were
pushed outside the band, and 3 changed by more than the band.** Worst cases are
all TF-IDF - alpha=0.20 moves -0.0444 to -0.0667, and alpha=0.10 moves -0.0111
to -0.0556.

**Reweighting has a cost, and it is charged to the tier that did not need
repairing.** That belongs beside any Tier-1 gain, not in a footnote.

#### Limitations, recorded beside the result

- **The target proxy is not the test set.** Weights are built toward the
  deployment 175, but coverage is measured on the 45-ticket benchmark. Weighted
  conformal assumes test points are drawn from the target the weights describe.
  So this measures "reweighting toward *this* target did not repair it", **not**
  "no reweighting could". Stated before the run, not after.
- **The weights are estimated, not known.** Barber et al. (2022) give the
  coverage cost of estimated weights; none of the residual gap is attributed
  here to any particular cause.
- **n = 45.** Each benchmark ticket is worth 2.2 coverage points, so the
  +-2 s.d. band is ~4.5 points. Differences smaller than that are not read.
- **`recovery_fraction` is post-hoc** - added after the results were seen,
  flagged as such in the CSV and here. It describes; it does not decide.
- **Tier-2 carries no finding by construction**, as declared in advance.

#### Verification

- The unweighted rows **reproduce Finding 1 exactly** - Tier-1 -0.233333,
  Tier-2 -0.011111 - checked **twice**: against a constant in the script and
  independently against the published `conformal_calibration_results.csv`, so a
  typo in the constant cannot become the thing the run validates against. Fatal
  on mismatch.
- **Uniform weights reproduce the unweighted quantile exactly** (`==`, not
  `approx`), across four alphas and six calibration sizes. Both sides reduce to
  rank `ceil((n+1)(1-alpha))` from the same float expression.
- **The batched quantile equals the scalar one exactly** - two internally
  consistent implementations of one quantity being allowed to drift is this
  project's recurring bug class.
- **Coverage and `n_eff` each recomputed by a second, independent expression**
  (`n_eff` via `n/(1+CV^2)`), fatal on disagreement.
- **A synthetic covariate-shift test** where `P(Y|X)` is held identical and only
  `P(X)` moves, with the true density ratio supplied: unweighted coverage 0.829,
  weighted 0.889 at nominal 0.90. Without it, a null on the real data could not
  be told apart from a broken implementation. (The first version of this test
  did not bite - a flatter test distribution *over*-covered at 0.893 - and would
  have passed a broken implementation; it was rebuilt.)
- **The prior correction `n_cal/n_target` is verified to be exactly 1.0** rather
  than silently dropped.

Script: `src/experiments/run_weighted_conformal.py`. Results:
[`weighted_conformal_results.csv`](data/weighted_conformal_results.csv) (42
rows). Library: `weighted_conformal_quantile`, `weighted_predict_sets`,
`effective_sample_size`, `true_label_scores` in `src/agent/conformal.py`, all
additive; tests in `tests/test_weighted_conformal.py`.

### Phase 7A - external-validity feasibility: the corpus is usable, and our own is worse

**FEASIBILITY ONLY. No replication, no experiment, no finding about this
project's methods.** 7A decides whether `Tobi-Bueck/customer-support-tickets`
can carry Phase 7B and at what n. Offline, zero Gemini calls. Production
untouched; an explicit isolation check confirms our dataset and artifacts are
byte-identical before and after.

#### Provenance, and what this dataset is NOT

- Repo `Tobi-Bueck/customer-support-tickets`, revision
  **`ddf1c81a5475992c4fa6752bf1e8b4e31f07bbeb`** (pinned; a mismatch is fatal),
  licence **CC-BY-NC-4.0**.
- **It is INDEPENDENTLY GENERATED DATA, NOT REAL PRODUCTION DATA.** The dataset
  card advertises a synthetic ticket generator from the same author. So Phase 7
  tests whether our findings survive **a different generator** - not whether
  they survive reality. Any write-up must say so in those words.
- **Why the real alternative was rejected:** the Endava/Microsoft
  `all_tickets.csv` is genuinely real, but its text is anonymized/encrypted, so
  a pretrained encoder such as BGE cannot read it. Real but unreadable is worse
  here than synthetic but readable.

#### The feasibility numbers

| Item | Result |
|---|---|
| Rows | **61,765** total -> **28,261 English** (the German-normalised file contributes 0) |
| vs our 45-ticket benchmark | **628x** raw, **278x** after de-duplication |
| Queues >= 300 English rows | **10 / 10** (Technical Support 8,149 -> General Inquiry 404) |
| Text length (median chars) | subject 42, body 365, answer 372 |
| Missing fields | **6 empty answers**, 1 empty body - but **3,639 empty subjects (12.9%)** |
| Exact duplicates | **4,514 redundant rows (15.97%)**; 9,028 rows (**31.95%**) sit in a duplicate group; largest group 2 |
| Near-duplicates (BGE >= 0.95) | **79.39%** of rows; **12,500 distinct components** (44.2% survive) |
| `version` | NaN 11,923 / 400.0 10,441 / 52.0 5,346 / 51.0 551 |

**Two duplicate rates are reported deliberately.** 15.97% counts only the
redundant copies; 31.95% counts every row that shares its text with another.
They describe the same corpus and differ by a factor of two, so quoting one as
the other would misstate redundancy outright.

#### The methodological point: a near-duplicate rate is meaningless without a control

79.39% looks disqualifying. It is not, and the two controls are the reason -
both now computed inside the script rather than asserted in prose.

**Control 1 - is 0.95 a duplicate threshold for this encoder at all?** BGE has a
high similarity floor for same-domain text, so a raw rate could be reading the
embedding's scale rather than duplication. Measured: **random pairs sit at
median 0.5912**, and only **0.0100%** of random pairs reach 0.95. The threshold
discriminates.

**Control 2 - high relative to what?** The only fair reference is the corpus
this project already publishes on:

| | External | **Ours** |
|---|---|---|
| Near-duplicate rate (BGE >= 0.95) | **79.39%** | **85.20%** |
| Nearest-neighbour p05 | 0.8867 | 0.9270 |
| Distinct components / rows | 12,500 / 28,261 (**44.2%**) | 1,213 / 4,000 (**30.3%**) |

**Our own corpus is MORE redundant than the external one, on every measure.**
So 79.39% is not a defect that distinguishes this dataset - **high
near-duplication is a property of template-generated corpora in general**, which
is the same mechanism Finding 2 and Phase 2A already document from the inside.
De-duplication becomes a mandatory step in any 7B design, but it is not an
argument for preferring our corpus. Reporting the external rate without this
control would have been a plausible, internally consistent, and wrong
conclusion - this project's recurring bug class, in its statistics form.

#### Item 7 - the Phase 2A question IS answerable here

Phase 2A could not measure clustering precision because on our corpus **the
templates ARE the fix classes**: no configuration ever makes a cross-template
merge, so a false merge cannot occur and precision is undefined. Measured here
with the production configuration - **MiniLM at 0.80**, read from config, using
`group_by_threshold` from `flag_automation_candidates.py` rather than a second
implementation:

| Queue | n scored | Clusters | Singletons | Largest | Distinct rate |
|---|---|---|---|---|---|
| Technical Support | 1,500 | 819 | 646 | 209 | 0.546 |
| Product Support | 1,500 | 698 | 489 | 190 | 0.465 |
| IT Support | 1,500 | 652 | 391 | 173 | 0.435 |
| Customer Service | 1,500 | 631 | 410 | 186 | 0.421 |
| Billing and Payments | 1,500 | 398 | 225 | 464 | 0.265 |
| General Inquiry | 404 | 134 | 47 | 36 | 0.332 |

**Yes.** Every queue shows both genuine singletons *and* substantial clusters -
43% singletons alongside a 209-member cluster in Technical Support. That mix is
exactly what our corpus lacks: there are merge candidates *and* items that must
not merge, so a false merge is possible and precision is therefore measurable.
Pairs at or above the 0.80 threshold run 0.16%-0.83% per queue.

#### Limitations, recorded beside the numbers

- **Independently generated, not real.** Stated above; it bounds every
  conclusion Phase 7 can draw.
- **CC-BY-NC-4.0** - non-commercial. Fine for an academic project; the term and
  attribution are recorded in `PROVENANCE.json`.
- **Item 7 is sampled at 1,500 answers per queue**, seeded (42), because
  `group_by_threshold` is production's O(n^2) Python loop. The `sampled` and
  `sample_cap` columns record this per row so a sample can never be mistaken
  for a full queue. Four queues were scored in full.
- **`version` and source-file are confounded.** All 11,923 version-NaN rows come
  from `dataset-tickets-multi-lang-4-20k.csv` and every versioned row from
  `aa_dataset-tickets-multi-lang-5-2-50-version.csv`. They are one split, not
  two independent shift axes.
- **12.9% of rows have no subject.** Text is built as subject + body, so those
  rows are body-only rather than empty - but any per-field analysis must
  account for it.
- **Encoding cost is real:** 28,261 texts took **5,541 s (~92 min at 5.1/s)** on
  CPU. Throughput varied 2.9-5.1/s with machine load, so a single-number
  estimate for this box would be false precision. The embeddings are cached, and
  `--force` re-renders the report **without** re-encoding (`--reencode` is a
  separate flag) - deliberately, so fixing a report line never costs 90 minutes.

#### Verification

- **Revision pinned and verified against the Hub**; fatal on mismatch.
- **Rule 6, two independent derivations:** row counts from pandas *and* a raw
  newline count (treated as an upper bound because of quoted fields); the
  English subset from a boolean mask *and* from `value_counts`; exact duplicates
  from `n_rows - n_unique` *and* from summing `(count - 1)`; near-duplicate
  neighbours from FAISS *and* brute force on 200 probed rows (**0 mismatches**);
  the test count as 306 + 16 = **322**.
- **Isolation check:** `synthetic_tickets.csv`, `novel_tickets_expanded.json`
  and `calibration_tickets_paraphrased.json` hashed before and after - all
  byte-identical. Nothing written outside `data/external_tobibueck/`.
- **16 unit tests** on the profiling helpers, checking duplicate and cluster
  rates against *constructed* ground truth rather than eyeballed output,
  including the two-rate distinction and that `--force` does not imply
  `--reencode`.

#### The shift designs, and why the largest one was rejected

Candidate shifts were not asserted to be real -- each was **measured**, with a
cross-fitted domain classifier in BGE space, reusing Phase 6B's machinery. The
headline: **every candidate sits below the 0.95 degeneracy threshold that
blocked 6B's BGE arm at 0.9908.** This corpus offers shifts that are real *and*
operable, which is exactly what 6B lacked.

| Design | Split | Domain AUC | Post-dedup n | Status |
|---|---|---|---|---|
| **A** | version 51+52 -> 400, **within the `aa` file** | ~~0.8584~~ **0.8727** | ~~~2,600~~ **3,305** / 10,441 | **PRIMARY** |
| **B** | queue held out of calibration only | ~~0.6706-0.9316~~ **0.8312-0.9693** | ~180-1,280 | pre-registered secondary |

> **Corrected by Phase 7B (2026-09-22).** The struck-through figures were
> computed ad hoc at this gate and their derivation was never committed; they
> **do not reproduce**. The replacements come from 7B's fully specified
> recomputation - see "RECORD CORRECTION" in the Phase 7B entry below. The
> post-de-duplication estimate was likewise an extrapolation from the
> corpus-wide 44.2% distinct rate; measured within the pool the rate is 56.05%.
> **The design conclusion is unchanged**: every value sits below the 0.95
> degeneracy line, which is what these numbers were recorded to establish.
| C | language EN -> DE | not measured | ~12,500 / ~14,800 | **REJECTED** |

**Design A's confound is removed by construction.** Across the whole corpus
`version` is confounded with source file -- all 11,923 version-NaN rows are
`dataset-tickets-multi-lang-4-20k.csv`. Inside the `aa` file, versions 51, 52
and 400 coexist, so the split is free of it; version is also spread
proportionally across queues. At 10% operating coverage the test arm holds ~460
tickets against the **4** that left Phase 6A's gated axis unmeasurable.

**Design B is the only design that varies shift magnitude deliberately**, so it
can ask *how much* shift Finding 1's effect requires rather than only whether it
reappears. Its confound is reportable but not removable: queue correlates with
`type` (Technical Support is 52% Incident; General Inquiry is 27% Change), so a
queue holdout also shifts the type mix.

**Design C is rejected despite having the largest n**, and the reason is worth
stating plainly: **a coverage drop on German would measure encoder competence,
not distribution shift.** The production encoder is `bge-base-en-v1.5`, an
English-only model, so "the encoder cannot read the input" would be
indistinguishable from "the distribution moved". That answers a different
question than Finding 1 asks, and no amount of extra n repairs a measurement
pointed at the wrong quantity.

Scripts: `src/experiments/fetch_external_dataset.py`,
`src/experiments/profile_external_dataset.py`. Outputs:
[`PROVENANCE.json`](data/external_tobibueck/PROVENANCE.json),
[`profile_summary.json`](data/external_tobibueck/profile_summary.json),
[`queue_distribution.csv`](data/external_tobibueck/queue_distribution.csv),
[`answer_diversity_by_queue.csv`](data/external_tobibueck/answer_diversity_by_queue.csv).
The raw CSVs and the embedding cache are gitignored and reproducible from the
pinned revision.

### Phase 7B - Finding 1 does NOT replicate on a different generator's corpus

**The headline, stated plainly because it undercuts a published claim.** On
`Tobi-Bueck/customer-support-tickets`, calibrated and tested under a measured
covariate shift, **TF-IDF and BGE transfer coverage equally well**. There is no
trace of the 23.3-vs-1.1 point contrast that Finding 1 rests on. Of twelve
pre-registered readings, exactly one gap falls outside its band - TF-IDF at
alpha=0.20, at -0.0220 against a band of 0.0220 - and that one returns inside
the band on the de-contaminated test arm. **Finding 1's scope narrows: on the
evidence now available it is a property of our corpus, not a general property
of lexical versus dense representations.** What that means for the paper is in
"What this does to Finding 1" below.

This corpus is **independently generated data, not real production data**. 7B
tests whether Finding 1 survives a *different generator*, not whether it
survives reality.

#### The design, pre-registered before any result was seen

Design A, chosen at the 7A gate: calibrate on the `aa` file's English rows at
`version` in {51, 52}, test on `version` == 400. The file restriction is what
removes the confound - corpus-wide, `version` is confounded with source file,
but inside the `aa` file all three versions coexist.

| arm | raw n | after de-duplication |
|---|---|---|
| calibration pool (v51+v52) | 5,897 | **3,305 components** (56.05% distinct) |
| -> training split (60%) | - | **1,978** |
| -> calibration split (40%) | - | **1,327** |
| test (v400) | **10,441** | used in full |

De-duplication runs on the pool at BGE cosine >= 0.95 **before** the split, so
no near-duplicate spans training and calibration and no duplicated point enters
the conformal quantile. The test arm is deliberately **not** de-duplicated:
removing rows there would change the very coverage being measured. Split is
seed 42, stratified by queue; both tiers are fitted **fresh on the external
training split only** and never load `models/`.

At n_cal = 1,327 the +-2 s.d. band is **0.0120 / 0.0165 / 0.0220** at
alpha = 0.05 / 0.10 / 0.20 - against **0.0454** at our n_cal = 175, so this is a
2.8x sharper instrument than the one that measured Finding 1.

#### The primary result

| test arm | tier | alpha | coverage | gap | band | outside? | mean set |
|---|---|---|---|---|---|---|---|
| full | tier1 | 0.05 | 0.9480 | -0.0020 | 0.0120 | in | 6.55 |
| full | tier1 | 0.10 | 0.9009 | +0.0009 | 0.0165 | in | 5.04 |
| full | tier1 | 0.20 | 0.7780 | **-0.0220** | 0.0220 | **OUT** | 3.49 |
| full | tier2 | 0.05 | 0.9545 | +0.0045 | 0.0120 | in | 6.68 |
| full | tier2 | 0.10 | 0.8928 | -0.0072 | 0.0165 | in | 4.88 |
| full | tier2 | 0.20 | 0.7907 | -0.0093 | 0.0220 | in | 3.53 |
| no-neighbour | tier1 | 0.20 | 0.7791 | -0.0209 | 0.0220 | in | 3.48 |
| no-neighbour | tier2 | 0.20 | 0.7917 | -0.0083 | 0.0220 | in | 3.52 |

On our corpus Tier-1's gap at alpha=0.10 is **-0.2333**. Here it is **+0.0009**
- not smaller, but on the *other side* of nominal and three orders of magnitude
closer to it. The pre-registered headline question was "does TF-IDF lose far
more coverage than BGE, as on our data?" The measured answer is **no**.

#### Boundary contamination: measured, and not the explanation

**152 of 10,441 version-400 test tickets (1.46%)** have a BGE >= 0.95
near-duplicate in training or calibration; median top-1 similarity is 0.8795.
The full and no-neighbour readings agree everywhere to within 0.0011, so
memorisation is not doing the work here. The rate is far below 7A's 79.39%
corpus-wide figure for the simple reason that the reference set is 3,305 rows
rather than 28,261 - a near-duplicate rate is a statement about a *pair of
sets*, not about a corpus alone, which is 7A's control lesson applied again.

#### Limitations, recorded beside the result, not after it

- **The external queue labels are generator-assigned and have not been
  audited.** In-distribution accuracy is **0.3476 (Tier-1)** and **0.3732
  (Tier-2)** over ten queues. Label noise depresses accuracy and inflates
  prediction-set sizes for both tiers - mean sets here run 3.5-6.7 labels out
  of 10 - so a low ceiling bounds every downstream reading, and set sizes are
  not comparable in absolute terms with our seven-class corpus.
- **The most important caveat is that this corpus has no representation gap to
  find.** Tier-1 and Tier-2 score 34.8% and 37.3%: 2.6 points apart. On our
  corpus the same contrast is 35.6 points. Finding 1's mechanism needs a
  representation that is *better* to be the one that transfers; where neither
  representation works, the test has little to detect. **This weakens 7B as
  evidence against Finding 1 - and it is a limitation, not a rescue.** The
  honest statement is that Finding 1 has not been shown to hold anywhere its
  originating corpus's structure is absent.
- **The shift is real but moderate**: cross-fitted domain AUC **0.8472** on the
  arms actually operated (0.8727 on the raw arms), well below the 0.95 line
  that blocked 6B's BGE arm at 0.9908.
- Independently generated data, not real production data.

#### RECORD CORRECTION: the 7A gate's domain AUC does not reproduce

`PROJECT_STATUS.md` and this README recorded Design A's cross-fitted domain AUC
as **0.8584**. That figure was computed ad hoc at the 7A gate and its
derivation was never committed. **It does not reproduce.** 7B's fully specified
recomputation - 5-fold cross-fitted logistic regression in BGE space, seed 42,
reusing 6B's `cross_fitted_domain_probabilities` - gives **0.8727** on the raw
arms and **0.8472** on the arms actually operated on. Five variants were tried
while chasing the recorded number (balanced subsampling two ways, first-n,
L2-normalised embeddings, post-de-duplication) and they span 0.8637-0.8727;
none lands on 0.8584.

**The design conclusion is unchanged** - every variant sits far below the 0.95
degeneracy line, so the shift is real and operable, which is what the number
was recorded to establish. The corrected values are the ones above, and the
same defect appears again in Design B's recorded AUC range.

#### Design B: reported, and it resolves nothing - twice, for two reasons

Design B holds one queue out of **calibration only**, keeping it in training so
the label space is preserved. The recorded design did not specify the test arm,
so **both readings were run rather than one being chosen after the fact**:

| variant | test arm | AUC range | why it does not resolve |
|---|---|---|---|
| `full_test` | all of version 400 | 0.8312-0.8566 (span **0.025**) | the magnitude knob does not move: a queue holdout barely changes a shift the version split already dominates |
| `held_out_queue` | version-400 rows *of* queue Q | 0.8499-0.9693 (span **0.120**) | the test arm carries **one class**, so coverage becomes class-conditional for that class rather than marginal - a different quantity |

The pre-registered degeneracy rule applies to both: **name the condition,
report "no resolution", never substitute a metric that happens to resolve.**
The rule also fired for real - Billing and Payments reaches AUC **0.9693** in
the second variant and is **BLOCKED**, the same condition that blocked 6B's BGE
arm. Its rows are in the CSV marked blocked, not as results.

Note also that Design B's recorded 7A range (0.6706-0.9316) does not reproduce
either, though the *ordering* of queues by AUC does: Billing highest, IT
Support lowest. Second instance of the same documentation defect.

Design B's confound remains reportable and not removable: queue correlates with
`type`, so a queue holdout also shifts the type mix.

#### Pre-registered secondary: the 6A deferral question, where it can resolve

Phase 6A asked whether deferring on conformal set size beats deferring on a
confidence threshold, and could not answer: the live gate's operating coverage
held **four** tickets on benchmark45 and 34 on deployment175, leaving 3 of 4
configurations degenerate. Here the transplanted 0.50 gate covers **6.15% =
642 tickets**, and the comparison resolves.

| tier | rule | risk at gate | delta vs confidence | 95% interval | signal |
|---|---|---|---|---|---|
| tier1 | confidence | 0.1791 | (incumbent) | | |
| tier1 | margin | 0.1745 | -0.0074 | [-0.0234, +0.0081] | no |
| tier1 | **lac** | 0.2290 | **+0.0466** | [+0.0156, +0.0779] | **yes** |
| tier1 | **aps** | 0.2321 | **+0.0532** | [+0.0236, +0.0836] | **yes** |
| tier2 | confidence | 0.0966 | (incumbent) | | |
| tier2 | margin | 0.0872 | -0.0085 | [-0.0202, +0.0031] | no |
| tier2 | lac | 0.0888 | -0.0071 | [-0.0296, +0.0156] | no |
| tier2 | **aps** | 0.1324 | **+0.0330** | [+0.0126, +0.0530] | **yes** |

**Three of six comparisons resolve, and all three favour the incumbent.** Risk
is error among accepted tickets, so a positive delta means the conformal rule
accepts a worse set of tickets than the plain confidence threshold. **No
comparison favours conformal.**

Two things this does and does not license. It **does** say that on this corpus,
at this operating coverage, conformal deferral is worse rather than equal -
which is a stronger statement than 6A could make. It **does not** retro-license
"conformal is worse" as a claim about *our* corpus: 6A's fixed wording, "no
evidence either way on the gated axis", still stands for the data 6A measured.
Different corpus, transplanted gate, ~35% label accuracy.

And conformal still cannot be **operated** at the gate: the achievable
alpha-indexed operating points sit at **0.39% / 0.75% / 2.43%** coverage for
Tier-1 and **0.34% / 1.82% / 3.90%** for Tier-2, against a 6.15% gate. The
obstacle 6A identified is present here too, on 10,441 tickets.

**AURC is reported and is not promoted**, per the standing rule. For the record
it agrees with the gated axis this time - every rule's AURC is worse than
confidence's, 7 of 8 with a signal - which is a coherence observation, not a
reason to promote anything. In 6A the two axes contradicted; the rule that
ignores AURC is the same either way.

#### What this does to Finding 1

Finding 1 is **scope-narrowed, not withdrawn.** Its numbers on our corpus are
unchanged and were re-verified by 6B.

**The paper's claim becomes:** *coverage transfer depends on the
representation **under paraphrase shift**, on our corpus; untested elsewhere
until Phase 7C.* The shift mechanism belongs **in the claim**, not only the
corpus - narrowing to "a property of our corpus" alone would silently assert
that 7B's shift and Finding 1's shift were equivalent tests, and they may not
be.

**Two candidate explanations for the null. Both are stated; neither is allowed
to rescue the finding.**

1. **7B may not be a like-for-like test.** Design A is a **version shift from
   the same generator**. Finding 1 was measured under a **paraphrase / register
   shift** - benchmark tickets rewritten out of the training templates' voice.
   TF-IDF's failure mechanism is **surface-vocabulary change**: when the words
   move, a bag-of-n-grams model's scores move with them, which is why its
   conformal quantile stops transferring. A version shift within one generator
   need not change surface vocabulary at all, so it may simply never exercise
   the mechanism. **Phase 7C tests exactly this** by paraphrasing version-400
   tickets into plain register and re-running the same table.
2. **The corpus has no representation contrast to find.** Tier-1 34.8% vs
   Tier-2 37.3% - 2.6 points, against 35.6 on ours. Finding 1's mechanism needs
   a representation that is *better* to be the one that transfers.

Neither explanation makes Finding 1 true beyond its measured scope. Until 7C
reports, the honest position is that the finding holds under the shift it was
measured under, on the corpus it was measured on, and **has not been shown to
hold anywhere else** - and that 7B's null is weaker evidence against it than
the bare numbers suggest.

The cross-phase named finding is *sharpened*, not damaged. "The
calibration/reference distribution, not the test or method, is the binding
constraint" predicts exactly this: change the corpus and the effect changes,
because the corpus was doing the work all along. 7B is the fourth phase to
reach that wall, and the first to reach it by *removing* an effect rather than
by failing to repair one.

#### Verification

- **Embedding-cache alignment proved, not assumed**: 64 seeded rows re-encoded
  through the production encoder, worst cosine **0.99999994** against the 7A
  cache. Row-count agreement alone is exactly the check a misaligned cache
  passes.
- **Rule 6, two independent derivations**: arm sizes from a boolean mask *and*
  `value_counts`; coverage from `cp.coverage` *and* an independent loop over
  the prediction sets (fatal beyond 1e-12); top-1 similarities from FAISS *and*
  brute force on 200 probed rows (**0 mismatches**); each deferral curve's
  coverage-1.0 endpoint checked against the accuracy the primary script
  measured separately.
- **Split integrity**: disjoint by row index **and** by de-duplication
  component, with the union covering all 3,305 components - a split can be
  row-disjoint and still leak a near-duplicate.
- **Isolation check, widened from 7A's three files to 22**: our dataset, both
  benchmarks, the deployment calibration set, `models/`, `tests/goldens/`, and
  every published conformal, deferral and ablation CSV - all hashed before and
  after, all byte-identical.
- **18 unit tests** on the new helpers against constructed ground truth,
  including the two bugs found while building 7B: a CSV writer that dropped
  rows whose key sets differed (which would have deleted every BLOCKED row from
  the Design B file) and a split that was row-disjoint while sharing
  components.
- **Gates re-run, not quoted**: `pytest` **340 passed**; adversarial escalation
  **9/9** with its CSV byte-identical; goldens **45/45** and **9/9**; ablation
  baseline **32/45**.

Scripts: `src/experiments/run_external_conformal_shift.py`,
`src/experiments/compare_deferral_rules_external.py`. Outputs:
[`external_conformal_designA.csv`](data/external_tobibueck/external_conformal_designA.csv),
[`external_conformal_designB.csv`](data/external_tobibueck/external_conformal_designB.csv),
[`external_deferral_results.csv`](data/external_tobibueck/external_deferral_results.csv),
[`external_contamination.json`](data/external_tobibueck/external_contamination.json),
[`external_label_noise.json`](data/external_tobibueck/external_label_noise.json),
[`external_shift_splits.json`](data/external_tobibueck/external_shift_splits.json).

### Phase 7C - Finding 1 under a paraphrase shift: BLOCKED, and the question stays open

**The headline is a non-answer, and it is reported as one.** 7C set out to test
Finding 1 under the shift it was actually measured under, because 7B's version
shift may never have exercised TF-IDF's failure mechanism. **The
pre-registered degeneracy rule fired: the TF-IDF-space cross-fitted domain AUC
between originals and paraphrases is 0.9972, above the 0.95 line, so the arm is
BLOCKED and no verdict is drawn on the primary.** The blocked numbers are in
the CSV marked blocked, as 6B's BGE arm is - not as a result.

**So after 7B and 7C, Finding 1's status is unchanged: it holds under the shift
it was measured under, on the corpus it was measured on, and has not been shown
to hold or to fail anywhere else.**

> **The paper's framing for Finding 1, fixed at the 7C gate for Phase 9A:**
> measured on our corpus; **external replication inconclusive** - 7B applied the
> wrong shift type, 7C was blocked by its own pre-registered rule; with
> **post-hoc evidence of the mechanism in accuracy** on the external corpus
> (difference-in-differences −0.0804, 95% CI [−0.1364, −0.0210]). Those three
> clauses travel together; none of them is quoted alone.

#### The manipulation check, reported before the result

| space | mean cosine | median | cross-fitted domain AUC |
|---|---|---|---|
| **BGE** (meaning preserved?) | 0.8462 | 0.8522 | 0.9981 |
| **TF-IDF** (surface vocabulary moved?) | **0.2489** | 0.2183 | **0.9972** |

The rewrite did exactly what it was asked to do: surface vocabulary is almost
entirely replaced (TF-IDF cosine 0.2489) while meaning is preserved (BGE cosine
0.8462). For scale, 7B's version shift sits at a BGE-space domain AUC of 0.8472
- this manipulation is far stronger.

**Length ratio, pre-registered secondary** (paraphrase / original):

| unit | mean | median | aggregate | mean length |
|---|---|---|---|---|
| BGE tokens | 1.0279 | 0.9543 | **0.9417** | 71.7 -> 67.5 |
| words | 1.0390 | 0.9595 | **0.9344** | 59.7 -> 55.8 |

Length is essentially preserved, about 6% shorter in aggregate, so a length
confound is not doing the work. The mean sitting above 1 while the median sits
below says a few pairs grew and pulled the mean; the aggregate is the robust
read.

#### The pre-registered verdict

**BLOCKED (`blocked_auc_ge_0.95`).** The rule was fixed in advance, by analogy
with 6B, where a domain AUC of 0.9908 blocked the BGE arm. It fired, so the
primary is not interpreted and no secondary is promoted to fill the gap.

#### SPEC ERROR, OWNED: the blocking rule was copied across designs

**Decided at the 7C gate (2026-09-22): the arm is NOT unblocked.** The rule was
pre-registered, and changing it after seeing results is exactly what this
project refuses to do, however good the reason. **The pre-registered verdict
stands: BLOCKED.**

*The critique below was raised after seeing the result and is recorded as a
specification error, not as grounds for a re-run.*

In **6B** the >= 0.95 rule gated a **density-ratio estimate**. Weighted
conformal divides by `p(cal|x)`, so near-perfect separability makes the ratio
ill-posed and its weights meaningless. The rule measured the thing it gated.

**7C estimates no density ratio.** It runs plain split conformal and uses the
domain AUC only as a *manipulation check*. In that role a near-1.0 AUC means
the manipulation was **strong**, which is what the design wanted. Carrying the
threshold across may therefore be gating the wrong quantity - the same error
the project has already named once, when a promotion rule for resolution
clustering measured recall while claiming to gate precision.

**The counter-reading is real and is recorded too:** a rewrite that a classifier
can identify with near-certainty is arguably a *different corpus* rather than a
shifted one, which is a coherent reason to refuse to call it a shift. The two
readings are not resolved here.

> **The lesson, for every future pre-registration: a degeneracy rule must be
> justified by what the specific design estimates, not copied across designs.**
> Ask what quantity becomes ill-posed at the threshold, and confirm that
> quantity exists in *this* experiment. A threshold guarding a density ratio
> means nothing where no density ratio is computed.

**The unblocked coverage numbers remain in
[`external_paraphrase_conformal.csv`](data/external_tobibueck/external_paraphrase_conformal.csv),
labelled post-hoc. They are never quoted as the verdict.**

#### POST-HOC: the mechanism IS visible - in accuracy, not in coverage

*Also added after seeing the result, and the most interesting thing in 7C.*

| arm | Tier-1 accuracy | Tier-2 accuracy |
|---|---|---|
| original | 0.3776 | 0.3776 |
| paraphrased | **0.2727** | **0.3531** |
| change | **-10.5 points** | **-2.5 points** |

**The paraphrase shift hit the lexical model's accuracy roughly four times
harder than the dense model's.** That is Finding 1's mechanism, confirmed
directly: surface-vocabulary change is what TF-IDF cannot absorb. 7B never
produced this contrast, which supports the diagnosis that its version shift was
the wrong kind of shift.

**And it survives its own interval.** Because this project does not write down a
difference without one, the accuracy difference-in-differences was bootstrapped
before it was written up - 10,000 paired draws at seed 42, the same
configuration as 6A and the blocked coverage arm:

| statistic | point | 95% CI |
|---|---|---|
| paraphrased, Tier-1 − Tier-2 | −0.0804 | [−0.1259, −0.0350] |
| original, Tier-1 − Tier-2 | +0.0000 | [−0.0455, +0.0455] |
| **difference-in-differences** | **−0.0804** | **[−0.1364, −0.0210]** |

The interval **excludes zero**: the paraphrase shift cost Tier-1 more than
Tier-2, over and above any difference the tiers had on the originals. In counts,
Tier-1 fell from **108 to 78** of 286 correct while Tier-2 fell from **108 to
101**.

**This remains POST-HOC and exploratory.** It does not replace the blocked
primary, it was not pre-registered, and the external labels are still
generator-assigned and unaudited. It is evidence about the *mechanism*, on the
external corpus, and that is all it is.

**And yet conformal coverage barely moved for either tier** (blocked numbers, in
the CSV, post-hoc).

#### FUTURE WORK, not tested here: are large prediction sets buffering coverage?

The obvious candidate explanation for accuracy moving while coverage did not is
structural: with ten classes and ~35% base accuracy, prediction sets here run
**4.9 to 6.9 labels out of 10**. Sets that large would **buffer coverage against
a score shift** - the true label stays inside them even when the argmax moves.
Our corpus's seven classes and far higher accuracy give small sets, where the
same score shift pushes labels out and coverage drops.

If it holds, Finding 1 would need **both** a representation contrast **and** an
operating regime where sets are small enough for coverage to be sensitive -
sharper than "it is a property of our corpus", and testable. **It was not tested
in 7C and is recorded as future work, not as a finding or a hypothesis this
programme goes on to check.**

The identical original-arm accuracies are a genuine coincidence, checked rather
than assumed: the two tiers disagree on **87 of 286** tickets and the
discordant-correct split is exactly **23 / 23**, with different probability
matrices.

#### Limitations, recorded beside the result

- **The external labels are generator-assigned and unaudited, and base accuracy
  is ~35%.** A null here is weaker evidence than a null on a well-learned task
  - and, as the post-hoc note above argues, the low accuracy may not be
  incidental to the null but the direct cause of it.
- **The paraphraser is a 3B local model.** Its rewrites carry a recognisable
  style, which is part of why the domain classifier separates the arms so
  easily. A stronger or more varied paraphraser would be a different
  manipulation.
- Independently generated data, not real production data - both the corpus and
  now the paraphrases.

#### Run provenance, recorded in full

The generation run was **killed twice by the Claude Code harness for low system
memory**, and **resumed from cache both times** with no regeneration and no data
loss:

1. First kill at **168 / 299** responses. The cache is keyed by prompt hash, so
   the resumed run skipped those and continued.
2. Second kill after **all 299 generations had completed**, during the
   post-generation guard step. **The cause was the BGE/Ollama overlap**: Ollama
   holds its model for five minutes after the last call, so at the exact moment
   the script loads BGE to compute source-paraphrase similarities, both models
   are resident. That overlap is the memory peak of the whole phase.

The fix, applied before the final pass: **unload the model first**
(`ollama stop qwen2.5:3b-instruct`, confirmed against `/api/ps`), then run with
**`--cached-only`**, which refuses every live call and turns a cache miss or a
prompt-hash mismatch into a fatal error rather than a silent regeneration. That
pass verified **299/299 prompt hashes against the rebuilt prompt** and made
**zero** calls.

#### DEVIATION: the RAM preflight was corrected, not bypassed

5C's `require_free_ram` compares *available* RAM against a floor sized for
**loading** the weights. Once Ollama already holds the model, that floor
double-counts - it demands headroom to load something already loaded. Measured
here: **2,275 MB available with 2,064 MB already resident, refused against a
3,277 MB floor**, with 4,339 MB genuinely in play.

`--allow-low-ram` would have proceeded, but it stamps every response
`low_ram_override`, permanently marking these paraphrases as produced under an
untrustworthy configuration - which would be **false**, and unfixable without
regenerating all 299. So the **check was made correct instead**: a 7C-local
preflight reads `/api/ps`, adds the already-resident CPU-side bytes back, and
reports all three numbers. `run_zeroshot_baselines.py` is untouched and 5C's
published behaviour is unchanged.

**The conservative direction is preserved**, and that is what the tests pin:
an unreadable residency reading counts as **zero resident**, so the check falls
back to 5C's strict behaviour rather than passing on an assumption. Four tests
cover the ways a relaxed guard could quietly stop refusing - residency
unreadable, available unreadable, resident-but-still-short, and the genuine
false-refusal case - plus tests that the reader returns `None` rather than
`0.0` on failure and counts only the non-VRAM part.

#### Verification

- **7B's models proved unchanged, not assumed**: refit from the 7B manifest
  reproduces its cached test probabilities at **max |delta| = 0.0** for both
  tiers, classes identical. A non-zero delta is fatal.
- **Pairing checked before any statistic**: 286 pairs, identical ticket ids in
  both arms, labels and source text re-verified against the corpus.
- **Coverage derived twice** (`cp.coverage` and an independent loop), fatal
  beyond 1e-12.
- **Guard counts two ways**: discarded-by-similarity from the mask and from
  `n_sampled - n_surviving - n_unparseable`.
- **The band is COMBINED, not just the calibration term.** At n = 286 the
  test-sampling sd (0.0173 at alpha=0.10) dominates the calibration-draw sd
  (0.0082); quoting the calibration term alone, as Finding 1 and 7B do on much
  larger test arms, would understate uncertainty by about a factor of two here.
  Both are in the CSV.
- **Isolation check: 27 files** - our dataset, both benchmarks, `models/`,
  `tests/goldens/`, every published conformal/deferral/ablation CSV **and 7B's
  own outputs** - byte-identical before and after.
- **32 unit tests** on the new helpers against constructed ground truth,
  including the fail-closed RAM tests and a DiD test asserting the statistic
  stays at zero when paraphrasing hurts both tiers equally.
- **Gates re-run, not quoted**: `pytest` **372 passed**; adversarial escalation
  **9/9** with its CSV byte-identical; goldens **45/45** and **9/9**; ablation
  baseline **32/45**.

Scripts: `src/experiments/paraphrase_external_tickets.py`,
`src/experiments/run_paraphrase_shift_conformal.py`. Outputs:
[`external_paraphrase_set.json`](data/external_tobibueck/external_paraphrase_set.json),
[`external_paraphrase_conformal.csv`](data/external_tobibueck/external_paraphrase_conformal.csv),
[`external_paraphrase_summary.json`](data/external_tobibueck/external_paraphrase_summary.json),
and the 299 raw responses under `data/external_paraphrase_raw/`.

### Phase 6C - a retrieval-sufficiency gate: catches both, but unusable as a gate

**The question.** The production RAG gate is one scalar: top-1 BGE cosine
>= 0.67, else escalate. Phase 2B found two tickets that cleared it where the
retrieval was useless - **G021** (N26, top-sim 0.7010) and **G024** (N31,
0.6747). Both produced a draft the human labelled **ungrounded**, and 2B's LLM
judge, which *saw the draft*, called both `grounded`. 6C asks whether a rater
that sees **only the ticket and its retrieved context, never a draft** flags
them. This is the failure mode Joren et al. (2024) name: a strong model answers
instead of abstaining when context is insufficient.

**Verdict: it catches both - and flags 26 of the 31 it should not.** A
sufficiency autorater is **not usable as a second gate on this corpus.** The
headline is the false-flag count, not the catch.

#### The primary, as counts (2 positives - no rate is estimable, by design)

|  | human `ungrounded` (n=2) | human `grounded` (n=31) |
|---|---|---|
| rater `INSUFFICIENT` | **2** | **26** |
| rater `SUFFICIENT` | 0 | 5 |

**Caught 2 of 2. Flagged 26 of 31.** The false-flag proportion on the
31-ticket axis is **0.839, Wilson 95% [0.674, 0.929]** - reported because that
axis *can* carry one; **no rate, interval, kappa or significance test is
reported on the 2-positive axis**, which was pre-registered as a case study by
construction. 26 of 33 eligible tickets are disagreements, every one quoted in
the summary JSON.

A gate that escalates 28 of 33 tickets that currently reach the resolver would
suppress ~85% of auto-resolution to recover two bad drafts. That is not a
tuning problem to be fixed by a threshold: the rater has no threshold, and its
verdicts are **stable** (see below), so there is nothing to move.

#### What the disagreements actually are

The rater is not malfunctioning - its reasons are specific and factually
correct about the retrieval. On **S052** (N02/G032, top-sim **0.8112**, the
highest in the eligible arm) it wrote: *"the retrieved tickets deal with users
who can see the folder but get 'access denied' when opening files inside it,
whereas the new ticket describes a user who cannot access the folder at all."*
That is a real distinction. The 2B human still labelled the resulting draft
`grounded`, because the draft's prescribed steps were traceable to the
retrieved resolutions. **Both readings are defensible**, which is the finding:
the two measures are not the same question, and a high cosine does not make
them agree.

#### POST-HOC: the "different questions" hypothesis is NOT supported

2B's rubric states that **declining is not an unsupported claim** - a draft
that says the retrieved examples do not fit and recommends a human is
`grounded`. So the false flags might concentrate on declining drafts, which
would mean the rater and the label were simply answering different questions.
Split (post-hoc, transparent phrase detector, every match recorded):

| 2B draft | n | flagged `INSUFFICIENT` |
|---|---|---|
| declines / hedges mismatch | 3 | 3 |
| does not | 28 | 23 |

**No concentration.** At n=3 the declining group cannot be compared with the
other - and the non-declining group is flagged at 23 of 28 regardless, so the
false flags are **everywhere**, not localised to a rubric artifact. The
hypothesis is recorded as **not supported**; it does not explain the result and
does not soften it.

#### Secondaries, all pre-registered

- **Agreement with the live gate on the 21 it already escalates: 20 of 21**
  (0.952, Wilson [0.773, 0.992]). The one disagreement is **S051** (N45,
  top-sim 0.6397): the rater found an NTP/chrony resolution that squarely
  addresses the ticket, so the **similarity gate escalated a ticket whose
  context was adequate**. A single case, but it is the mirror image of the two
  2B misses - the scalar errs in both directions.
- **Cross-family, Qwen2.5-3B, zero quota:** 54/54 parseable, overall agreement
  with Gemini **39/54 = 0.722**, and prompt identity **54/54 byte-identical by
  sha256 in both directions**, so the two arms provably answered the same
  question. Qwen's own 2x2: caught **1 of 2**, flagged **20 of 31**. It is less
  aggressive (35 vs 48 `INSUFFICIENT` of 54) and it **misses one of the two
  positives** - so the result is not an artifact of one vendor's prior, and the
  weaker model is not a usable gate either.
- **Stability:** both positives' items re-rated at 3 repeats, temperature 0.0 -
  `INSUFFICIENT` **3/3 on both**. **No determinism finding.** The primary is
  rep 1 by pre-registration regardless; repeats never revise it and are never
  majority-voted.

#### A rejected hypothesis, recorded so it is not re-proposed

The aggressiveness is **not** explained by near-duplicate retrieved neighbours
leaving nothing to distinguish. That contradicts 2B's own diagnostic on these
same tickets: **71.1%** of benchmark-45 and **88.9%** of adversarial-9
retrievals contain **2+ distinct fixes**. The context is heterogeneous.
**6C is therefore NOT claimed as an instance of the project's named finding**
about the calibration/reference distribution - the mechanism here is a
measurement-target mismatch between sufficiency and groundedness, not a
register mismatch in the corpus.

#### Limitations, beside the result

- **The 2B labels are an outcome proxy.** They record whether the *draft* was
  supported, not whether a human would call the *context* sufficient. The 26
  "false" flags are false only against that proxy; no human has labelled
  sufficiency directly. **That, not the rater, is the binding limitation** -
  and producing those labels is human work, not something to be substituted.
- **2 positives.** The catch is a case study, not an estimate.
- Gemini rates its own vendor's family but **never its own draft** (the draft
  is absent from the prompt), which is why the Qwen arm is the control for
  vendor prior rather than for self-preference.

#### Provenance, stated in full

The `--limit 3` dry run happened to include **one** of the two positives (G024
at S002; G021 landed at S029), so **its primary verdict was seen before the
full pass was launched.** Nothing was revised on seeing it: the rubric, output
schema, population and primary rule were all fixed at the plan gate before any
call, and the prompt hash is in the cache. P(at least one of two positives in
the first three of 54) = **0.109, about 1 in 9**. The seed-42 ordering was
audited: the permutation was independently re-derived from the benchmark order
plus 2B's own `escalated_ids` and matched **all 54** assignments, with the two
arms interleaved - a genuine shuffle, uncorrelated with the labels.

**No degeneracy or blocking rule was pre-registered, deliberately.** 6C
estimates no density ratio and fits no model, so there is no ill-posedness
threshold to import - the 7C lesson applied rather than repeated.

**Gates re-run, not quoted:** `pytest` **406 passed** (372 + 34 new);
adversarial escalation **9/9** with its CSV byte-identical
(`d53e40c6...` before and after); goldens **45** and **9** rows exact; ablation
baseline **32/45 (71.11%)**. Production unchanged: cascade **0.50**, RAG
**0.67**, clustering **0.80**; `conformal.enabled` and `drift.enabled` both
`False`. **58 Gemini calls** (54 + 4 repeats), in-code cap 70.

Scripts: `src/experiments/build_sufficiency_context.py`,
`src/experiments/run_sufficiency_autorater.py`,
`src/experiments/score_sufficiency_gate.py`. Outputs:
[`sufficiency_gate_results.csv`](data/sufficiency_gate_results.csv),
[`sufficiency_gate_summary.json`](data/sufficiency_gate_summary.json),
the context bundle and key, and the raw responses under
`data/sufficiency_raw/`.

### Automation-flagging feature

The production payoff of the calibration above: `flag_automation_candidates.py`
applies the fixed, calibrated 0.80 threshold to the real resolved-ticket
pipeline output (`data/category_stores/*.csv`) — with no dependency on
ground truth, since production tickets don't have any at run time — and
outputs flagged clusters of tickets that likely share the same underlying
fix, for human review and potential self-service automation.

Run against the full 500-ticket dataset:

| Category | Tickets | Flagged Clusters | Flagged Tickets | Singletons |
|---|---:|---:|---:|---:|
| Infrastructure | 115 | 17 | 107 | 8 |
| Application | 100 | 17 | 84 | 16 |
| Security | 61 | 9 | 56 | 5 |
| Database | 49 | 7 | 24 | 25 |
| Storage | 53 | 10 | 45 | 8 |
| Network | 74 | 14 | 51 | 23 |
| Access Management | 48 | 9 | 29 | 19 |
| **Total** | **500** | **83** | **396** | **104** |

A real, citable pattern emerges in the category-level automation rate:
**Infrastructure's fixes are highly standardized** (93% of tickets fall
into a repeatable cluster — the same handful of runbook-style scenarios
recur constantly), while **Database's fixes are comparatively bespoke**
(49% flagged) — a genuine finding about which categories are naturally
more automatable, not an artifact of the method. Output is written to
`data/automation_candidates.json` (full detail) and
`data/automation_candidates_summary.csv` (one row per flagged cluster, for
citation). As with the confidence-based escalation layer, this flagging is
explicitly designed as a suggestion for human review, not an autonomous
action — the calibration's 1.0000 precision was measured against known
ground truth on the development set, and production tickets carry no such
guarantee at run time.

---

## Final Classification Comparison

| Method | In-Distribution Accuracy | 14-Ticket Generalization | 45-Ticket Generalization |
|---|---|---|---|
| TF-IDF + Logistic Regression | 100.0% | 7/14 (50.0%) | — |
| Frozen MiniLM embeddings + Logistic Regression | 100.0% | 10/14 (71.4%) | 32/45 (71.1%) |
| **Frozen BGE embeddings + Logistic Regression** | — | — | **33/45 (73.3%) — production choice** |
| Frozen E5 embeddings + Logistic Regression | — | — | 27/45 (60.0%) |
| Fine-tuned DistilBERT (best epoch) | 100.0% | 7/14 (50.0%) | — |
| Cascade (TF-IDF → embeddings, 70–80% target) | — | 10/14 (71.4%), ~21% resolved by cheap tier | — |

BGE is now the production embedding model for classification, RAG
retrieval, and cascade Tier-2 (swapped from MiniLM on the strength of this
result — see "Embedding Model: MiniLM → BGE" above). Resolution-text
clustering for automation-flagging remains on MiniLM, deliberately scoped
as a separate BGE clustering re-run rather than migrated alongside the rest of
the pipeline.

---

## Research / Novelty

Two real academic papers were reviewed to identify a genuine gap:

- **Paper 1** (multi-agent CX architecture) proposed an escalation/
  orchestration design but never implemented or empirically tested it.
- **Paper 2** (IT-ticket classification) rigorously tested classification
  on real enterprise data, but never touched resolution generation, RAG,
  or confidence-based escalation. It did handle real-world class
  imbalance, which directly inspired the class-imbalance experiment above.

**This project's contribution:** an actual end-to-end pipeline covering
classification, retrieval-grounded resolution, confidence-based
escalation, and automation-flagging — empirically measured at every layer,
including honest reporting of calibration methods that *didn't* work and
honest reporting of how the system behaves under both traffic-volume skew
and training-data skew. Confidence-based cascading itself is not novel
(cascade classifiers date to Viola & Jones, 2001; FrugalGPT applies the
same idea to LLM cost) — the honest novelty claim is applying and
rigorously measuring this exact pattern, calibrated against real data
independently at three separate layers, specifically for IT ticket triage.
The RAG similarity threshold's recalibration story is itself a small,
citable methodological contribution: an in-domain-only calibration set can
produce a misleading cliff-edge for a gate whose actual job is detecting
out-of-domain inputs, and adding negative-class data does not
automatically fix that if the negative class's own similarity
distribution doesn't reach into the artifact's threshold range.

### Named finding — the calibration/reference distribution, not the test or method, is the binding constraint

This is the project's strongest cross-phase result, and it is named here because
**three** phases reached it independently, from different directions, through
the same mechanism: **a register mismatch between the data the system was built
from and the data it actually sees.** None of the three was designed to test it.

*(Scope widened after Phase 5C. This finding was first stated over the
**calibration/reference** distribution alone. 5C reached the same wall from the
**training** distribution, so the mechanism is broader than the original wording
allowed — the heading is kept for continuity, but the claim the paper should
make is the wider one. **Phase 9A must decide the final wording**; the two
original instances are unchanged and are not weakened by the addition.)*

- **The coverage side — Phase 1, Finding 4.** A deployment-distribution
  calibration set, built to the same size and class balance as the in-domain one
  so only the distribution differed, recovered about 38% of Tier-1's conformal
  coverage shortfall (−0.233 → −0.144 at α=0.10) and left it **six times outside
  the noise band**. Matching the distribution is *necessary but not sufficient*:
  where the representation itself fails to transfer, no amount of
  calibration-set realism repairs the guarantee.
- **The monitoring side — Phase 4B, the realistic-traffic arm.** Scored against
  the in-domain reference, deployment-register tickets — legitimate traffic, not
  drift by any construction — flag at **0.217 / 0.429 / 0.514 / 0.646** for
  α = 0.01 / 0.05 / 0.10 / 0.20 against nulls of 0.006 / 0.046 / 0.097 / 0.199.
  Four to seven times the null. A Signal A monitor on that reference would alarm
  continuously, and **no choice of test, α or window size fixes it** — the
  conditional binomial's own null false-alarm rate is a clean 0.009–0.023 on
  exchangeable data at the very same operating points.
- **The accuracy side — Phase 5C, the zero-shot baselines.** A classifier
  trained on 3,200 rows of the corpus is **indistinguishable from a 3B
  open-weight model that never saw it** (34/45 vs 33/45, exact McNemar
  p = 1.000) and distinguishably *worse* than zero-shot Gemini (40/45,
  p = 0.0391), on out-of-template benchmark phrasing. The training data's
  advantage does not survive the crossing either.

The three results are not a repeat of one measurement. One is about whether a
finite-sample *guarantee* survives deployment; the second about whether a
*monitor* can run without false alarms; the third about whether *accuracy*
bought with training data transfers at all. All three fail for the same reason
and are fixed by the same thing — data drawn from the deployment distribution —
which is what makes the constraint a property of the corpus rather than of any
method. The practical form of the claim: **in this system, every attempt to
improve a test, a threshold or a statistic hit a ceiling set by what the
calibration data was made of.**

**What is *not* part of this finding.** Phase 2A's negative result is a
**related but distinct dataset limitation**: template-generated resolutions make
the templates *be* the fix classes, so no pair of tickets can look alike and
need different fixes, and clustering precision is unmeasurable on this corpus at
all. That is redundancy in the *training* data limiting what can be evaluated —
not a calibration/reference register mismatch. It belongs beside this finding as
a second, independent reason the synthetic corpus constrains the conclusions, and
it should not be presented as an instance of the same mechanism. Phase 5C, by
contrast, **is** one: it is a register mismatch, not a redundancy problem.

*(Recorded for the write-up phase. No new experiment was run for the finding
itself — it is a reframing of three results already measured and published
above.)*

**Important distinction, updated after Phase 3.** This previously read that
the system was *"not yet a true multi-agent system"* and that the restructure
"remains a scoped future extension, not something built yet". That is no
longer accurate, and the accurate version is narrower than the phrase
"multi-agent system" usually implies:

- **What exists.** Independent Classification, Retrieval and Resolution agents
  with declared dependencies, coordinated by an orchestrator that owns every
  routing decision (`src/agent/agents.py`, `src/agent/orchestrator.py`), each
  agent individually addressable over HTTP (`src/service/api.py`). This is
  what closes the gap with Paper 1's proposed-but-never-implemented
  orchestration design — implemented, and covered by 104 tests plus two fixed
  benchmark sets.
- **What does not.** The agents are independently *addressable*, not
  independently *running*: `/triage` orchestrates in-process, one ticket at a
  time, because golden parity must not depend on a running server. There is no
  distributed execution, no message bus, no concurrent or autonomous agents,
  and no agent that decides anything — the gates belong to the orchestrator
  alone.

So the honest claim is **a sequential pipeline with real agent boundaries, an
orchestrator, and an HTTP surface** — not a distributed multi-agent system. The
restructure is described in full under "Phase 3 — the agent architecture"
above, including why the original n8n plan was superseded.

---

## What's Done vs. What's Pending

### Done

- 4,000-ticket dataset with a fixed 14-ticket generalization benchmark
- 3-way classifier comparison (TF-IDF / MiniLM / DistilBERT), re-verified
  at both 1,000 and 4,000 tickets
- Cascade classifier with a fully validated 3-attempt calibration
  methodology and accuracy/efficiency tradeoff analysis
- RAG layer (FAISS retrieval + Gemini-grounded resolution) with a
  similarity-based human-escalation guard
- Expanded 45-ticket generalization benchmark, including finding and
  correctly fixing a benchmark-generation labeling bug (not a dataset
  flaw), and a validated BGE > MiniLM finding at the larger sample size
- **Embedding model swap (MiniLM → BGE)** for classification, RAG
  retrieval, and cascade Tier-2, on the strength of the 45-ticket
  benchmark result; model-aware, filename-suffixed artifacts throughout
  to prevent silent stale-cache mismatches
- **RAG similarity threshold fully recalibrated** (provisional 0.65 →
  final **0.67**), via a new 45-ticket Gemini-paraphrased out-of-domain
  calibration set, a combined in-domain + OOD confusion matrix, a
  measured 5.7% in-domain self-retrieval contamination rate, and a
  final value derived from the intersection of the 9-ticket adversarial
  set's safe range and minimum OOD leakage rate — confirmed 9/9 on the
  adversarial set. See "RAG Similarity Threshold Recalibration" above.
- Working Streamlit demo tying classification + RAG together — verified
  live across all three decision paths (Tier-1 resolution, Tier-2
  escalation, RAG-similarity escalation to a human)
- Batch-intake simulation validating pipeline plumbing under skewed
  production-style traffic volume
- Class-imbalance experiment confirming real accuracy degradation on a
  shrinking minority category, with the escalation mechanism catching
  every resulting error in this run (directional, not conclusive)
- Resolution-clustering calibration, finding a genuine precision
  cliff-edge and selecting a calibrated threshold (0.80)
- Production automation-flagging feature built and run against the full
  500-ticket dataset, with a citable category-level automation-rate
  finding
- Cascade confidence-calibration reliability diagrams (both tiers),
  evaluated against the real 500-ticket production batch — found both
  tiers to be genuinely underconfident rather than overconfident, a
  safer failure mode for an escalation-gated system
- **Agent architecture (Phase 3)** — Tier-1 persisted behind a manifest
  guard instead of refitted at startup; the three stages separated into
  agents with declared dependencies under an orchestrator that owns every
  routing decision; a FastAPI service exposing each agent, a `/health`
  endpoint reporting the config fingerprint, `/policy/rag-gate`, and a
  `/triage` failure boundary that turns a known agent error into an
  escalation while still returning 500 for an unknown one. All three
  sub-phases gated on the goldens reproducing exactly; test suite 69 → 104.
  The original n8n orchestration plan was superseded — see "Phase 3 — the
  agent architecture" for the reasoning
- **Conformal prediction (measurement-only)** — hand-rolled split conformal
  over both cascade tiers plus conformal novelty detection for the RAG gate.
  Three results: coverage transfer is a property of the *representation*
  (TF-IDF loses 23.3 coverage points under a paraphrase shift where BGE
  loses 1.1, on identical data); the in-domain calibration set is
  structurally impossible to de-contaminate because memorisation is
  template-level, not row-level; and conformal novelty detection matches
  production's 9/9 on the adversarial set while adding a calibrated
  false-escalation rate the 0.67 threshold never had. Production gating is
  unchanged (`settings.conformal.enabled = False`).
- **Deployment-distribution calibration set** (175 tickets, 25/category)
  built to test whether matching the calibration distribution repairs the
  conformal guarantee. It does not, for a lexical model: Tier-1's coverage
  shortfall closes from −0.233 to −0.144 at α=0.10, still six times the
  noise band. For Tier-2 the payoff is set size rather than coverage —
  identical coverage with singletons on 46.7% of benchmark tickets versus
  35.6%, i.e. more autonomy at the same risk. Includes a generation-time
  near-duplicate guard added after the first run produced 21 near-duplicate
  pairs.
- Literature review identifying a genuine research gap
- Permanent 9-ticket adversarial escalation test set with a live-pipeline
  regression script, which itself caught a real bug (stale, un-migrated
  MiniLM constants surviving the BGE swap) and is now the anchor
  constraint for the RAG similarity threshold's derivation
- Ablation study quantifying the real measured value of both safety-net
  thresholds, **including the control that was missing until Phase 5B**:
  the cascade beats Tier-1-only by 35.6 points, but that gap is the
  BGE-vs-TF-IDF representation difference, not the value of cascading.
  Against Tier-2 alone the cascade is one ticket *worse* on both
  evaluation sets and statistically indistinguishable from it (exact
  McNemar p = 1.000 on each); what it actually buys is 8–18% of median
  per-ticket latency. The RAG gate prevents 6/9 adversarial tickets from
  receiving a fabricated resolution instead of correctly escalating.
  (Re-measured under BGE — see the correction note in "Ablation Study"
  for why the original 33.3-point figure was a MiniLM-era number — and
  re-framed in "Phase 5B — the honest ablation".)
- Category-specific resolution-clustering threshold check: re-ran the
  pooled calibration independently per category. Four of seven categories
  (Infrastructure, Application, Security, Access Management) match the
  pooled 0.80 cliff-edge exactly. Storage and Database diverge, though
  both are low-N categories where the deviation is indicative rather than
  conclusive. Network (n=74, above the low-N bar) also diverges to 0.75 —
  a statistically credible deviation from the pooled threshold, not
  sample-size noise. Measurement-only; production still uses the pooled
  0.80 threshold pending a future decision on category-specific
  thresholds.
- Resolution-clustering automation-flagging threshold: BGE vs.
  production decision.** The BGE clustering re-run (re-running pooled and per-category
  resolution-clustering calibration under BGE) is done — see "Resolution-
  clustering calibration" and "Category-specific resolution-clustering
  calibration" above. BGE's clustering is measurably tighter across
  categories than MiniLM's, but the production automation-flagging
  threshold was deliberately **not** swapped to BGE (still 0.80 on
  MiniLM), because — unlike the RAG similarity threshold, which had both
  a validated accuracy benchmark (the 45-ticket set) and an adversarial
  test to re-confirm the specific chosen value before adoption — there is
  currently no equivalent ground-truth check for whether BGE clustering
  produces *better* automation flags, only that its own precision/recall
  curve looks internally consistent. Swapping production on cliff-edge
  math alone, with no way to validate the specific value against real
  cases, is exactly the category of unchecked assumption this project has
  caught and rejected before (the in-distribution split, the 35-ticket
  calibration set, the low-N artifact at RAG threshold 0.85). That
  validation method has since been built and run — see "Phase 2 —
  automation-flag validation" above. It returned a negative result: at
  their own cliff-edges neither configuration makes a single
  cross-template merge, so the two are indistinguishable on precision and
  differ only in recall. **Production stays on MiniLM @ 0.80**, and any
  future swap is a product decision about review-queue capacity rather
  than something flag-correctness evidence can settle on this dataset.
  

### Pending

1. **Re-run both Phase 2 harnesses on deployment-distribution data** — 2A
   and 2B hit the same dataset wall from opposite directions, and neither is
   blocked by its method.

   *2A* could not measure precision at all: at their own cliff-edges neither
   MiniLM@0.80 nor BGE@0.90 makes a single cross-template merge, because
   template-generated tickets cannot produce two cases that look alike yet
   need different fixes — the templates *are* the fix classes. The 12-pair
   pilot found zero false merges with a 25% rule-of-three upper bound, which
   is consistent with that diagnostic but cannot rule a false merge out.
   Deciding MiniLM@0.80 vs BGE@0.90 on precision needs real resolved
   tickets — the same blocker the conformal work hit independently. Point the
   harness at a new `category_stores` and re-run.

   *2B* could measure groundedness, but only on the 33 out-of-template
   benchmark drafts, since 85% of in-distribution retrievals collapse to a
   single fix. That same 45-ticket benchmark has now carried classification
   accuracy, conformal coverage and groundedness, accumulating selection risk
   that no single experiment can see. It also leaves the judge finding resting
   on three disagreements.

   Both harnesses are built, guarded and need no code changes. ("Phase 2A" and
   "Phase 2B" refer to these harnesses; the earlier BGE measurement is the
   "BGE clustering re-run" throughout.)

2. ~~**Genuine multi-agent restructure**~~ — **done in Phase 3, and the n8n
   half was superseded.** Independent Classification, Retrieval and Resolution
   agents coordinated by a real orchestrator now exist in Python
   (`src/agent/agents.py`, `src/agent/orchestrator.py`), each addressable over
   HTTP (`src/service/api.py`). See "Phase 3 — the agent architecture" above.

   The original scope put the orchestration itself in n8n. That was changed
   deliberately: an `IF confidence < threshold` node would be a second,
   untested copy of the calibrated 0.67 gate, living where neither `pytest`,
   the goldens, nor the adversarial gate can reach it — the most expensive
   possible instance of this project's recurring bug class. A workflow JSON
   also cannot be regression-tested against `tests/goldens/*.json`, so it
   would be the first routing logic here with no parity net under it. The full
   reasoning is in "Why the Python orchestrator supersedes the n8n proposal".

   **What remains optional:** an n8n workflow purely as a visual figure,
   branching on `POST /policy/rag-gate`, which returns the decision computed
   by the tested Python so the workflow holds no threshold of its own. This is
   presentation, not evidence, and nothing depends on it.


---

## Project Structure

```
ticket-routing-agent/
├── data/
│   ├── generate_dataset.py
│   ├── synthetic_tickets.csv
│   ├── ticket_embeddings.npy                          (gitignored, regenerable cache)
│   ├── ticket_index_bge-base-en-v1-5.faiss
│   ├── ticket_metadata_bge-base-en-v1-5.json
│   ├── calibration_tickets_paraphrased.json
│   ├── ood_calibration_tickets.json                    (45-ticket OOD calibration set)
│   ├── rag_similarity_calibration.csv                  (Attempt 1: in-domain-only sweep)
│   ├── rag_similarity_calibration_combined.csv         (Attempt 2: combined in-domain + OOD sweep)
│   ├── novel_tickets_expanded.json                     (final 45-ticket benchmark)
│   ├── resolution_clustering_calibration_results.csv
│   ├── resolution_clustering_calibration_percategory.csv
│   ├── resolution_clustering_calibration_percategory_summary.csv
│   ├── exploratory_clustering_results.json
│   ├── automation_candidates.json
│   ├── automation_candidates_summary.csv
│   ├── category_stores_with_scenario_id.csv
│   ├── calibration_tier1_reliability_diagram.png
│   ├── calibration_tier2_reliability_diagram.png
│   ├── calibration_reliability_data.csv
│   ├── adversarial_escalation_tickets.json
│   ├── adversarial_escalation_results.csv
│   ├── ablation_baseline_results.csv
│   ├── ablation_no-cascade_results.csv
│   ├── ablation_no-rag_results.csv
│   ├── batch_intake/
│   │   ├── incoming_tickets_batch.csv
│   │   └── batch_summary.csv
│   ├── category_stores/
│   │   ├── {Category}.csv                             (one per category)
│   │   └── escalation_needs_review.csv
│   ├── embedding_comparison/
│   │   └── embedding_model_comparison.csv
│   └── skewed/
│       ├── synthetic_tickets_am{571,500,200,100,50}.csv
│       ├── embeddings_am{571,500,200,100,50}.npy
│       └── imbalance_sweep_results.csv
├── src/
│   ├── agent/                                          (THE library — all inference)
│   │   ├── config.py                                   (frozen typed config + provenance)
│   │   ├── schemas.py                                  (pydantic models for every stage)
│   │   ├── errors.py                                   (typed exceptions; one Gemini ladder)
│   │   ├── logging_setup.py                            (structured JSON decision logs)
│   │   ├── artifacts.py                                (THE loader, three hard guards)
│   │   ├── classifier.py                               (cascade Tier-1 → Tier-2)
│   │   ├── retriever.py                                (FAISS retrieval)
│   │   ├── resolver.py                                 (prompt + Gemini call with retry)
│   │   ├── conformal.py                                (split conformal, measurement-only)
│   │   ├── agents.py                                   (the three agents: name + requires + run)
│   │   ├── orchestrator.py                             (the sequence and BOTH gates)
│   │   └── pipeline.py                                 (public façade over orchestrator.run)
│   ├── service/
│   │   └── api.py                                      (FastAPI: agents, /policy, /triage)
│   ├── classification/
│   │   ├── train_baseline_tfidf.py
│   │   ├── generalization_test.py                     (source of truth for NOVEL_TICKETS)
│   │   ├── train_tier1.py                              (persisted Tier-1 + manifest guard)
│   │   ├── train_embeddings.py
│   │   ├── train_embeddings_comparison.py
│   │   ├── generalization_test_embeddings.py
│   │   ├── train_distilbert.py
│   │   ├── train_cascade.py
│   │   ├── generate_calibration_set.py                (175-ticket in-domain calibration set)
│   │   └── generate_ood_calibration_set.py             (45-ticket OOD calibration set)
│   ├── rag/
│   │   ├── build_vector_index.py
│   │   └── suggest_resolution.py
│   ├── app/
│   │   └── streamlit_app.py
│   └── experiments/
│       ├── simulate_ticket_intake.py
│       ├── process_ticket_batch.py
│       ├── generate_skewed_datasets.py
│       ├── run_imbalance_sweep.py
│       ├── join_scenario_ground_truth.py
│       ├── explore_resolution_clustering.py
│       ├── calibrate_resolution_clustering.py
│       ├── calibrate_resolution_clustering_percategory.py
│       ├── calibrate_rag_similarity_threshold.py       (combined in-domain + OOD sweep)
│       ├── flag_automation_candidates.py
│       ├── plot_calibration_curves.py
│       ├── test_adversarial_escalation.py
│       └── run_ablation_study.py
├── tests/                                              (pytest suite + golden parity)
│   ├── conftest.py                                     (read-only benchmark fixtures)
│   ├── capture_goldens.py                              (regenerate ONLY on purpose)
│   ├── goldens/*.json                                  (the refactor safety net)
│   └── test_*.py
├── models/                                             (gitignored — regenerate, never commit)
│   ├── ticket_classifier_bge-base-en-v1-5.joblib
│   ├── tier1_tfidf_logreg.joblib                       (persisted Tier-1 + manifest)
│   └── distilbert_ticket_classifier/
├── .env                                                (gitignored — GEMINI_API_KEY)
└── README.md                                           (this file)
```

## How to Run

```powershell
# Activate the venv
.\venv\Scripts\Activate.ps1

# 1. Generate the dataset (if not already present)
python data/generate_dataset.py

# 2. Train Tier-1 (persisted; the pipeline refuses to load a stale one)
python src/classification/train_tier1.py

# 3. Train the production classifier (Tier-2)
python src/classification/train_embeddings.py

# 4. Build the RAG index
python src/rag/build_vector_index.py

# 5a. Launch the live demo
streamlit run src/app/streamlit_app.py

# 5b. ...or the HTTP service (from the project root)
uvicorn src.service.api:app --port 8000
```

`GEMINI_API_KEY` must be set in a `.env` file at the project root (get a
free key from https://aistudio.google.com/apikey). Classification, retrieval
and the escalation gate all work without it — only resolution drafting
needs it.

Run the test suite with `pytest` (104 tests, offline, no quota). Any change to
a live threshold, embedding model, or retrieval path must also re-confirm 9/9
on `python src/experiments/test_adversarial_escalation.py` before being
committed.

### Talking to the service

```powershell
# Config fingerprint, index size, Tier-1 provenance
Invoke-RestMethod http://localhost:8000/health

Invoke-RestMethod -Method Post http://localhost:8000/triage `
  -ContentType 'application/json' `
  -Body '{"title":"vpn keeps dropping every ten minutes"}'
```

(`Invoke-RestMethod` rather than `curl`, which on Windows PowerShell is an
alias for `Invoke-WebRequest` and does not accept `-X` or `-d`.)

`POST /triage` defaults to `generate_resolution=false`, so it spends no Gemini
quota unless asked. The interactive API docs are at `/docs`.

### Running the imbalance experiments

```powershell
# Traffic-volume skew (batch simulation)
python src/experiments/simulate_ticket_intake.py
python src/experiments/process_ticket_batch.py

# Training-data skew (the main class-imbalance experiment)
python src/experiments/generate_skewed_datasets.py
python src/experiments/run_imbalance_sweep.py
```

### Running the resolution-clustering / automation-flagging pipeline

```powershell
# (Requires data/category_stores/*.csv to already exist from batch processing)
python src/experiments/join_scenario_ground_truth.py
python src/experiments/explore_resolution_clustering.py
python src/experiments/calibrate_resolution_clustering.py

# Category-specific calibration check (diagnostic only, does not change
# the production threshold):
python src/experiments/calibrate_resolution_clustering_percategory.py

# The production feature itself (fixed, calibrated threshold=0.80):
python src/experiments/flag_automation_candidates.py
```

### Running the RAG similarity threshold calibration

```powershell
# (Re-)generate the 45-ticket OOD calibration set (~3.5 min, ~45 Gemini calls):
python src\classification\generate_ood_calibration_set.py

# Run the full combined in-domain + OOD calibration sweep (measurement
# only — does not modify production SIMILARITY_THRESHOLD):
python -m src.experiments.calibrate_rag_similarity_threshold

# Confirm any candidate threshold against the adversarial set before
# adopting it into production (SIMILARITY_THRESHOLD in suggest_resolution.py):
python src/experiments/test_adversarial_escalation.py
```

---

## Known Open Items

- ~~Stray `RESULTS.md` references in `streamlit_app.py`~~ — fixed; they
  now point at this README.
- Resolution-clustering calibration has been re-verified under BGE
  (pooled and per-category), but production automation-flagging still
  runs on the original MiniLM-calibrated threshold (0.80) pending a
  validation method for the BGE alternative — see "Pending" above.
- The `no-rag` and `no-cascade` ablation CSVs have been regenerated under
  BGE along with the baseline. `process_ticket_batch.py`'s MiniLM/BGE
  dimension mismatch is fixed in code, but the script has deliberately
  **not** been re-run: `data/category_stores/*.csv` were produced under
  MiniLM and feed the resolution-clustering calibration behind the
  production 0.80 threshold. Regenerating them under BGE would silently
  invalidate that calibration, so re-running is its own deliberate
  decision with its own re-derivation — not a side effect of a refactor.

---

## Working Conventions (for future contributors / future me)

- Fixed random seed (`42`) everywhere, for reproducibility.
- Path resolution: project root is always two directories up from a
  script's own location, using `os.path.*` so it works cross-platform.
- Every classification script uses the same 80/20 stratified split
  (`test_size=0.2, random_state=42`) so results are directly comparable.
- The 14-ticket generalization benchmark (`NOVEL_TICKETS`, defined in
  `generalization_test.py`) is never altered — it's one of two fixed
  reference points (alongside the 45-ticket expanded benchmark) used
  across every method comparison in this project.
- Every script fails with clear, actionable error messages instead of raw
  tracebacks — especially important for the Streamlit demo, which may run
  live in front of an audience.
- Embedding caches are always scoped to the exact dataset (and, since the
  BGE swap, the exact embedding model) they were computed from — never
  reuse a cache keyed to a different row count or model, since that
  silently misaligns embeddings to the wrong tickets. Every loading
  function checks `index.ntotal == len(metadata)` and fails loudly on
  mismatch, after this exact failure mode was caught twice: once as a
  general silent-stale-cache risk during the BGE swap, and once
  specifically in `test_adversarial_escalation.py`, which had kept its
  own independent hardcoded MiniLM constants un-migrated.
- **All inference goes through `src/agent/`, and new consumers call
  `pipeline.run()`** — never a re-implementation of the orchestration. This
  project once had four independent copies of the classify → retrieve →
  escalate pipeline and three disagreeing model loaders; one of them was
  encoding queries with MiniLM against a BGE index and had been for weeks.
  Artifacts load only through `artifacts.load_artifacts()`.
- **No agent decides routing.** Agents (`src/agent/agents.py`) declare the
  artifacts they need and do one thing; every gate lives in
  `src/agent/orchestrator.py`, where it can be read and tested in one place
  without loading a model. An escalation policy scattered across the things
  being orchestrated is one nobody can audit.
- Calibrated thresholds (RAG similarity **0.67**, cascade confidence 0.50,
  resolution-clustering 0.80) are fixed, evidence-backed constants defined
  once in `src/agent/config.py` — never re-derived casually or exposed as
  trivially-overridable CLI flags, since that would undermine the point
  that these values are measured, not guessed. `config.py` is frozen, and
  each value carries its evidence in `CALIBRATION_PROVENANCE`. Any change to a live
  threshold must be re-confirmed against the 9-ticket adversarial
  escalation set before being committed.
- Gemini free-tier rate limits: 15 requests/minute, 500 requests/day
  (as of this writing) — keep `GEMINI_CALL_DELAY_SEC` comfortably above
  the per-minute floor (4s) in any batch-calling script.
- Before every commit: run `git status | Select-String "\.npy"` to catch
  any regenerable embedding cache that might have been accidentally
  staged.
