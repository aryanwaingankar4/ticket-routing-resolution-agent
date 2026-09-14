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

**Tech stack:** Python, scikit-learn, sentence-transformers, FAISS, Gemini
API (`gemini-flash-lite-latest`, via the `google-genai` SDK), Streamlit,
pandas/numpy.

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
| **No RAG gate** (pretend threshold=0) | — | 9/9 would attempt a resolution; 6/9 should have escalated |

**Cascade threshold (0.50):** removing it drops classification accuracy
by 35.6 percentage points on the 45-ticket benchmark — the cascade
isn't a marginal tweak, it roughly doubles real-world classification
accuracy versus running the cheap Tier-1 model alone.

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
| Scenario templates in the dataset | 12 |
| Templates touched by the 175 calibration tickets | 11 (91.7%) |
| Median rows per template | 430 |
| Sibling rows left after removing one source row | 429 |
| Rows surviving template-level exclusion | **40 / 4000** |

Memorisation here is at *template* level, not row level. Deleting one row
removes roughly 0.2% of its template's evidence, so the fitted model is
effectively unchanged. The stronger move — excluding every row sharing a
template with any calibration ticket — would leave 40 rows to train on, which
is not a de-contamination but a destruction.

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

**Important distinction:** the current system is a *sequential pipeline*
with confidence-based decision points — it is **not** yet a true
multi-agent system (independent agents coordinating via an orchestrator).
That remains a scoped future extension, not something built yet.

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
  thresholds: the cascade threshold contributes a 35.6-point accuracy
  gain over Tier-1-only; the RAG gate prevents 6/9 adversarial tickets
  from receiving a fabricated resolution instead of correctly escalating
  (re-measured under BGE — see the correction note in "Ablation Study"
  above for why the original 33.3-point figure was a MiniLM-era number)
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

2. **Genuine multi-agent restructure** — independent Classification,
   Retrieval, and Resolution agents coordinated by a real Orchestrator,
   likely via n8n (wrapping the existing Python pieces as small local API
   endpoints, then building a real n8n workflow with visual conditional
   routing, e.g. IF confidence < threshold → escalate). Directly closes
   the gap with Paper 1's design, which was proposed but never
   implemented there either. Deliberately done **last**, once everything
   above is measured and stable — it's an architecture/presentation step,
   not a new experiment.


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
│   ├── classification/
│   │   ├── train_baseline_tfidf.py
│   │   ├── generalization_test.py                     (source of truth for NOVEL_TICKETS)
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
├── models/                                             (gitignored, except joblib artifact)
│   ├── ticket_classifier_bge-base-en-v1-5.joblib
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

# 2. Train the production classifier
python src/classification/train_embeddings.py

# 3. Build the RAG index
python src/rag/build_vector_index.py

# 4. Launch the live demo
streamlit run src/app/streamlit_app.py
```

`GEMINI_API_KEY` must be set in a `.env` file at the project root (get a
free key from https://aistudio.google.com/apikey).

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
- Calibrated thresholds (RAG similarity **0.67**, cascade confidence 0.50,
  resolution-clustering 0.80) are fixed, evidence-backed constants in
  their respective scripts — never re-derived casually or exposed as
  trivially-overridable CLI flags, since that would undermine the point
  that these values are measured, not guessed. Any change to a live
  threshold must be re-confirmed against the 9-ticket adversarial
  escalation set before being committed.
- Gemini free-tier rate limits: 15 requests/minute, 500 requests/day
  (as of this writing) — keep `GEMINI_CALL_DELAY_SEC` comfortably above
  the per-minute floor (4s) in any batch-calling script.
- Before every commit: run `git status | Select-String "\.npy"` to catch
  any regenerable embedding cache that might have been accidentally
  staged.
