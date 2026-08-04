# AI-Powered Intelligent Ticket Routing & Resolution Agent

A three-layer AI system for IT support ticket triage: **classify** the ticket
into the right team, **retrieve** similar past tickets and suggest a grounded
resolution, and **escalate to a human** whenever the system isn't confident
enough to act on its own. Built as a B.Tech final-year AI/ML project, based on
a nasscom hackathon use-case brief, with an ongoing extension toward an
IEEE-style research paper.

---

## Architecture

1. **Classification Layer** — "which team should this ticket go to?"
   Categories: Infrastructure, Application, Security, Database, Storage,
   Network, Access Management (7 total).
2. **RAG Layer** — "how was this solved before?" Retrieves similar past
   tickets via FAISS and asks Gemini to draft a grounded resolution.
3. **Agentic Layer** — escalates to a human whenever confidence is too low,
   at two separate decision points (classification and retrieval), rather
   than letting the system guess.

**Tech stack:** Python, scikit-learn, sentence-transformers, FAISS, Gemini
API (`gemini-flash-lite-latest`, via the `google-genai` SDK), Streamlit,
pandas/numpy.

**Repo:** `aryanwaingankar4/ticket-routing-resolution-agent`

---

## Project Journey — Day by Day

### Day 1 — Dataset generation
- Built `data/generate_dataset.py`: generates synthetic IT support tickets
  (columns: `id, title, description, category, resolution, priority`).
- Each "scenario" is a linked `(title_template, symptom_phrase,
  resolution_text)` tuple so fields stay semantically consistent — no
  mismatched title/resolution pairs.
- One shared per-ticket "context" dict resolves all placeholders
  (`{app}`, `{srv}`, `{db}`, etc.) exactly once, so entities never disagree
  across fields.
- Fixed random seed (`42`) throughout for reproducibility.
- Validation gates before writing the CSV: category balance, no empty
  fields, duplicate threshold.
- Started at **1,000 tickets** (~142-143/category, 0 duplicate pairs).

### Day 2 — TF-IDF baseline classifier
- `train_baseline_tfidf.py`: TF-IDF (`max_features=5000, ngram_range=(1,2)`)
  + Logistic Regression.
- **100% in-distribution accuracy — flagged as a red flag, not a success**,
  since it's template-generated data prone to lexical memorization.
- Built the **14-ticket generalization benchmark**: 2 hand-written tickets
  per category, phrased in plain, non-technical, everyday language that
  never appears in the training templates. This became *the* metric used
  across the entire project — reused verbatim in every later comparison.
  Defined in `src/classification/generalization_test.py` as `NOVEL_TICKETS`
  — this is the single source of truth; other files copy it verbatim and
  must never diverge from it.
- TF-IDF generalization score: **7/14 (50.0%)** — proved it was purely
  keyword-matching, with no real understanding of meaning.

### Day 3 — Embeddings-based classifier
- `train_embeddings.py`: frozen `all-MiniLM-L6-v2` (384-dim) embeddings +
  Logistic Regression.
- Generalization score: **10/14 (71.4%)** — a real +21.4 point improvement
  over TF-IDF, because embeddings capture meaning, not just keywords.
- Tried a bigger model (`all-mpnet-base-v2`) — performed worse (9/14, 64.3%),
  reverted to MiniLM.
- Added `joblib.dump()` to save the trained classifier to
  `models/ticket_classifier.joblib`, avoiding retraining on every run.

### Day 4 — DistilBERT fine-tuning experiment
- `train_distilbert.py`: fine-tuned `distilbert-base-uncased` on CPU, 4
  epochs, `batch_size=16`, `max_length=128`, `lr=2e-5`.
- **Classic overfitting signature**: in-distribution accuracy hit 100% by
  epoch 2, while generalization *declined* from epoch 1 onward
  (42.9% → 35.7% → 35.7% → 35.7%). Best checkpoint (epoch 1) still lost to
  both TF-IDF and MiniLM.
- **Conclusion:** a bigger, more capable model does not automatically help
  on a small, template-heavy dataset — it just memorizes faster.

### Day 5 — RAG layer (retrieval + Gemini-grounded suggestions)
- `build_vector_index.py`: FAISS `IndexFlatIP` over L2-normalized MiniLM
  embeddings (cosine similarity via inner product). Saves an *aligned*
  metadata JSON (`ticket_metadata.json`), with an explicit assertion that
  `len(metadata) == index.ntotal` to catch any misalignment immediately.
  Built-in sanity check: queries the index with a ticket's own embedding,
  confirms it retrieves itself at similarity ~1.0.
- `suggest_resolution.py`: `retrieve_similar_tickets(top_k=5)`,
  `SIMILARITY_THRESHOLD = 0.35` — below this, the LLM call is **skipped
  entirely** and the ticket is escalated to a human instead. This became
  the design principle later mirrored one layer down in the cascade
  classifier.
- Switched from the deprecated `google-generativeai` package to the current
  unified SDK, `google-genai` (`from google import genai`), and from
  `gemini-2.5-flash` (retired/404s for new users) to
  `gemini-flash-lite-latest`.
- **Gemini's role is strictly resolution generation** — it never decides
  the ticket category. Classification is handled entirely by the trained
  models above.
- Demo results: VPN ticket (retrieved well, Gemini correctly flagged a
  scope mismatch), password reset (confident grounded match), and a
  deliberately unrelated "weather" question (similarity dropped to
  ~0.06-0.12, correctly skipped the LLM call and escalated).

### Day 6 — Scaling the dataset (1,000 → 4,000 tickets)
- Extended `generate_dataset.py`: `TOTAL_TICKETS = 4000`, 3 new scenario
  templates added per category (9 total/category), balanced ~571-572/category.
- Re-ran the MiniLM generalization test on the scaled dataset —
  **score stayed exactly the same, 10/14 (71.4%), same 4 tickets wrong in
  the same way.** This proved the ceiling was *representation-limited*
  (the frozen embedding model's own resolution), not data-volume-limited.

### Cascade Classifier — confidence-based tier routing
- Built `train_cascade.py`: Tier-1 (cheap, TF-IDF) resolves a ticket if
  confident enough; otherwise escalates to Tier-2 (expensive, MiniLM).
  Mirrors the RAG layer's "escalate when unsure" philosophy, applied one
  level down at model-selection time.
- **Calibration took three attempts**, each one a genuine finding:
  1. **In-distribution held-out split** — rejected. Showed 100% accuracy in
     every confidence bucket, producing a threshold (0.70) that looked
     trustworthy but missed a confidently-wrong (0.76) real-world
     prediction entirely.
  2. **35 hand-written calibration tickets** — rejected. Too sparse
     (34/35 collapsed into one bucket), threshold driven by statistical
     noise.
  3. **175 Gemini-paraphrased calibration tickets** (25/category,
     independent of the 14-ticket benchmark, with a self-consistency label
     check — 21/175 flagged for review, kept anyway since ground truth was
     preserved) — adopted. Dense enough to correctly reveal Tier-1's real
     overconfidence and catch the previously-missed wrong prediction.
- Also fixed a **threshold-derivation logic bug**: the scan was breaking on
  the first small/noisy bucket it failed, before ever checking better
  buckets further down. Fixed by removing the early `break`.
- **Final accuracy/efficiency tradeoff** (swept target reliability bars of
  90%/80%/70% against the 175-ticket calibration set):

  | Target Acc | Threshold | Held-out Tier-1% | Held-out Acc | Novel Tier-1% | Novel Acc |
  |-----------:|----------:|------------------:|-------------:|---------------:|----------:|
  | 90%        | 1.00      | 0.0%              | 100.0%       | 0.0%           | 71.4%     |
  | 80%        | 0.50      | 100.0%            | 100.0%       | 21.4%          | 71.4%     |
  | 70%        | 0.50      | 100.0%            | 100.0%       | 21.4%          | 71.4%     |

  At a strict 90% bar, no benefit exists — the cascade collapses to pure
  Tier-2. At a relaxed 70-80% bar, a real threshold (0.50) emerges, routing
  ~21% of real-world tickets through the cheap tier at **zero accuracy
  cost**. The real contribution isn't a big accuracy jump — it's a
  validated calibration methodology that catches what naive calibration
  would silently miss. Full write-up in `RESULTS.md`.

### Streamlit Demo
- Built `src/app/streamlit_app.py`, tying the cascade classifier + RAG
  layer into one clickable demo.
- Shows: predicted category, which tier resolved it and why (with a plain-
  language explanation of the escalation decision), retrieved similar
  tickets table, and either Gemini's grounded suggestion or a visually
  distinct human-escalation alert.
- 3 one-click example tickets built in (VPN, password reset, and a
  deliberately unrelated question to demonstrate the escalation guard).
- Debugged several signature mismatches between the app and the actual
  `suggest_resolution.py` / `train_cascade.py` functions along the way —
  a good reminder to always verify real function signatures
  (`inspect.signature(...)`) rather than assuming them.

### DistilBERT re-test at 4,000 tickets
- Re-ran `train_distilbert.py` after the dataset scale-up (previously only
  tested at 1,000 tickets — a gap in the evidence, since MiniLM *was*
  re-tested at the larger scale but DistilBERT wasn't, until now).
- **Result:** best-epoch generalization improved modestly, 42.9% → **50.0%**
  (now tying TF-IDF). The post-epoch-1 *decline* seen at 1,000 tickets did
  not recur — scores plateaued instead of getting worse. However, the same
  fast-memorization signature persists: 100% in-distribution accuracy from
  epoch 1 onward, `train_loss` still shrinking toward zero.
- **Conclusion unchanged**: DistilBERT still doesn't justify its extra
  size/training cost — it now ties the simplest method but remains 21.4
  points behind MiniLM embeddings.

### Day 7 — Batch-intake simulation (production traffic skew)
- Motivated by a mentor question: what happens when real ticket traffic
  arrives unevenly across categories?
- Built `simulate_ticket_intake.py`: draws a 500-ticket batch from the
  held-out data pool using a Zipf-style skew (some categories dominate,
  others are sparse), with pool-size-aware redistribution when a
  category's ideal Zipf share exceeds its available held-out rows.
- Built `process_ticket_batch.py`: runs the full existing pipeline
  (cascade classification → RAG retrieval → Gemini resolution or
  escalation) against the simulated batch, writing results to per-category
  CSV stores plus an escalation queue.
- **Result:** 100% accuracy across every category, zero escalations. This
  is expected, not a meaningful finding on its own — the batch is sampled
  from the same in-distribution held-out pool the classifier already
  performs perfectly on (a pattern seen everywhere else in this project).
  This run validates the *plumbing* (routing, storage, rate-limit
  handling) under uneven volume, not classification robustness.
- **Operational finding:** hit Gemini free-tier rate limits (15
  requests/minute, 500/day) mid-run — 15 of 500 resolution calls failed
  after retries. Root cause confirmed via the Google AI Studio dashboard.
  Fix: increased `GEMINI_CALL_DELAY_SEC` from 1.5s to 4.5s to stay under
  the per-minute cap; daily cap requires waiting for reset or a temporary
  model swap for any same-day retry.

### Day 8 — Class-imbalance experiment (training-data skew)
- The complementary experiment to Day 7: instead of testing traffic
  volume skew against an already-trained model, this tests what happens
  when the model itself is trained on less and less minority-class data.
- Built `generate_skewed_datasets.py`: samples "Access Management" down
  from its full 571 tickets to 500 / 200 / 100 / 50, holding all other 6
  categories fixed. Access Management was chosen as the skew target
  because it scored perfectly (0 errors) on the 14-ticket benchmark at
  full data — any degradation after skewing is attributable to the skew
  itself, not pre-existing confusion. It's also semantically adjacent to
  Security (both revolve around credentials/permissions), giving a
  plausible failure mode to watch for.
- Built `run_imbalance_sweep.py`: retrains fresh TF-IDF and MiniLM
  classifiers from scratch at each skew level (never reuses the
  full-data model), evaluates on the held-out split, the 14-ticket
  benchmark, and the cascade's existing 0.50 confidence threshold
  (reused as-is, not re-derived per level). Embeddings are cached
  separately per skew level to avoid silently misaligning them against
  the wrong row counts.

**Result — real degradation on the 14-ticket benchmark:**

| AM training size | MiniLM benchmark | AM tickets specifically |
|---:|---:|---|
| 571 (baseline) | 11/14 | ✅ ✅ both correct |
| 500 | 11/14 | ✅ ✅ both correct |
| 200 | 10/14 | ✅ ❌ one misclassified → Storage |
| 100 | 9/14 | ❌ ❌ both wrong → Security, → Storage |
| 50 | 9/14 | ❌ ❌ both still wrong → Security, → Storage |

As Access Management shrinks, the model's real-world (novel-phrasing)
accuracy on that category degrades — and one failure mode is exactly the
predicted one: misclassification as Security. In-distribution held-out
accuracy stayed at ~99.6-100% across all levels throughout, confirming
(again) that in-distribution numbers are uninformative here; only the
14-ticket benchmark showed the effect.

**Escalation safety net:** every wrong Access Management prediction
across all skew levels (1 at the 100-ticket level, 3 at the 50-ticket
level) fell below the 0.50 confidence threshold and was correctly
escalated to a human rather than confidently accepted —
`cascade_safety_net_failures = 0` at every level. This is a promising
early signal, but the sample is small (1 and 3 wrong predictions
respectively) — it should be read as directional evidence that the
escalation mechanism *can* catch minority-class degradation, not as
proof that it always will.

**Methodology note:** this sweep trains and evaluates the benchmark model
on the 80% train split only, whereas the cascade's originally-reported
10/14 (71.4%) MiniLM score used a model retrained on the full dataset.
The two numbers aren't directly comparable for that reason (11/14 here
vs. 10/14 elsewhere) — the meaningful result is the *degradation trend*
across skew levels, not the absolute score at the 571 baseline.

Full per-level results saved to `data/skewed/imbalance_sweep_results.csv`.

---

## Final Classification Comparison (14-ticket generalization benchmark)

| Method | In-Distribution Accuracy | Generalization Score |
|---|---|---|
| TF-IDF + Logistic Regression | 100.0% | 7/14 (50.0%) |
| **Frozen MiniLM embeddings + Logistic Regression** | 100.0% | **10/14 (71.4%) — production choice** |
| Fine-tuned DistilBERT (4,000 tickets, best epoch 1) | 100.0% | 7/14 (50.0%) |
| Cascade (TF-IDF → MiniLM, 70-80% target) | — | 10/14 (71.4%), ~21% resolved by cheap tier |

**Winner for production use:** frozen MiniLM embeddings + Logistic
Regression, selected on measured generalization performance rather than
in-distribution accuracy alone.

---

## Research / Novelty

Reviewed two real academic papers to identify a genuine gap:

- **Paper 1** (multi-agent CX architecture) proposed an escalation/
  orchestration design but never implemented or empirically tested it.
- **Paper 2** (IT-ticket classification) rigorously tested classification
  on real enterprise data, but never touched resolution generation, RAG,
  or confidence-based escalation. It did handle real-world class
  imbalance, which is the direct inspiration for Days 7-8 above.

**This project's contribution:** an actual end-to-end pipeline covering
classification, retrieval-grounded resolution, and confidence-based
escalation — empirically measured at every layer, including honest
reporting of calibration methods that *didn't* work, and honest reporting
of how the system behaves under both traffic-volume skew and
training-data skew. Confidence-based cascading itself is not novel
(FrugalGPT, cascade classifiers date to Viola & Jones, 2001) — the honest
novelty claim is applying and rigorously measuring this pattern
specifically for IT ticket triage, using the same underlying philosophy
at two different layers of one system.

**Important distinction:** the current system is a *sequential pipeline*
with confidence-based decision points — it is **not** a true multi-agent
system (independent agents coordinating via an orchestrator). That remains
a scoped future extension (see below), not something built yet.

---

## What's Done vs. What's Pending

### ✅ Done
- 4,000-ticket dataset with a fixed 14-ticket generalization benchmark
- 3-way classifier comparison (TF-IDF / MiniLM / DistilBERT), re-verified
  at both 1,000 and 4,000 tickets
- Cascade classifier with a fully validated 3-attempt calibration
  methodology and accuracy/efficiency tradeoff analysis
- RAG layer (FAISS retrieval + Gemini-grounded resolution) with a
  similarity-based human-escalation guard
- Working Streamlit demo tying classification + RAG together
- Literature review identifying a genuine research gap
- **Batch-intake simulation**: validated pipeline plumbing under skewed
  production-style traffic volume (500 tickets, Zipf skew); confirmed
  in-distribution accuracy stays perfect, so this tests routing/storage
  behavior rather than classification robustness
- **Class-imbalance experiment**: confirmed real accuracy degradation on
  a shrinking minority category (Access Management), including the
  predicted Access Management → Security confusion pattern, with the
  cascade's confidence-based escalation catching every resulting error
  in this run (small sample — directional, not conclusive)

### ⏳ Pending
1. **Automation-flagging for recurring issues**: detect ticket "bursts"
   (e.g. 40 password resets in a week) and flag for self-service
   automation. This is the second, unbuilt half of the Agentic Layer —
   confidence-based escalation is already done.
2. **Larger generalization benchmark**: expand beyond the current 14
   hand-written tickets (each currently worth ~7 percentage points) using
   the same paraphrase-and-verify methodology proven for the 175-ticket
   calibration set, while keeping a hand-written core for credibility.
3. **Accuracy-improvement experiments** (ranked by cost/plausibility):
   (a) swap the frozen embedding model for a newer family (BGE, GTE, E5)
   — cheapest to test; (b) paraphrase-augment the *training* set using the
   same Gemini pipeline proven for calibration data, without touching the
   14-ticket benchmark or the 175-ticket calibration set; (c) ensemble
   with an orthogonal method (Gemini zero-shot classification) only after
   checking error-overlap with existing models, since correlated-error
   ensembles can underperform the best single model.
4. **Genuine multi-agent extension**: restructure the current linear
   pipeline into independent agents (Classification, Retrieval,
   Resolution) coordinated by an Orchestrator agent with real conditional
   routing — directly closing the gap with Paper 1's design, which was
   never implemented or tested there either.
5. **Formal paper write-up**, once the above experiments are complete.

---

## Project Structure

```
ticket-routing-agent/
├── data/
│   ├── generate_dataset.py
│   ├── synthetic_tickets.csv
│   ├── ticket_embeddings.npy         (gitignored, regenerable cache)
│   ├── ticket_index.faiss
│   ├── ticket_metadata.json
│   ├── calibration_tickets_paraphrased.json
│   ├── batch_intake/
│   │   ├── incoming_tickets_batch.csv
│   │   └── batch_summary.csv
│   ├── category_stores/
│   │   ├── {Category}.csv            (one per category)
│   │   └── escalation_needs_review.csv
│   └── skewed/
│       ├── synthetic_tickets_am{571,500,200,100,50}.csv
│       ├── embeddings_am{571,500,200,100,50}.npy
│       └── imbalance_sweep_results.csv
├── src/
│   ├── classification/
│   │   ├── train_baseline_tfidf.py
│   │   ├── generalization_test.py    (source of truth for NOVEL_TICKETS)
│   │   ├── train_embeddings.py
│   │   ├── generalization_test_embeddings.py
│   │   ├── train_distilbert.py
│   │   ├── train_cascade.py
│   │   └── generate_calibration_set.py
│   ├── rag/
│   │   ├── build_vector_index.py
│   │   └── suggest_resolution.py
│   ├── app/
│   │   └── streamlit_app.py
│   └── experiments/
│       ├── simulate_ticket_intake.py
│       ├── process_ticket_batch.py
│       ├── generate_skewed_datasets.py
│       └── run_imbalance_sweep.py
├── models/                            (gitignored, except joblib artifact)
│   ├── ticket_classifier.joblib
│   └── distilbert_ticket_classifier/
├── .env                                (gitignored — GEMINI_API_KEY)
├── RESULTS.md                          (full experimental write-up)
└── README.md                           (this file)
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

---

## Working Conventions (for future contributors / future me)

- Fixed random seed (`42`) everywhere, for reproducibility.
- Path resolution: project root is always two directories up from a
  script's own location, using `os.path.*` so it works cross-platform.
- Every classification script uses the same 80/20 stratified split
  (`test_size=0.2, random_state=42`) so results are directly comparable.
- The 14-ticket generalization benchmark (`NOVEL_TICKETS`, defined in
  `generalization_test.py`) is never altered — it's the one fixed
  reference point across every method comparison in this project. Other
  files copy it verbatim; check for drift if it's ever edited.
- Every script fails with clear, actionable error messages instead of raw
  tracebacks — especially important for the Streamlit demo, which may run
  live in front of an audience.
- Embedding caches are always scoped to the exact dataset they were
  computed from (e.g. per skew level) — never reuse a cache keyed to a
  different row count, since that silently misaligns embeddings to the
  wrong tickets.
- Gemini free-tier rate limits: 15 requests/minute, 500 requests/day
  (as of this writing) — keep `GEMINI_CALL_DELAY_SEC` comfortably above
  the per-minute floor (4s) in any batch-calling script.