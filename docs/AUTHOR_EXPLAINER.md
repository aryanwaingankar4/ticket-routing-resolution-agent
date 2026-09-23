# Author explainer: every result, in plain language

*Phase 9C. Written for Aryan, the author, so you can explain and defend every
result without re-reading the lab notebook. It is not the paper, and it is not
written for a reviewer.*

## How to read this

- **Every number comes from `paper/NUMBERS.md`**, and the id sits beside it in
  backticks, e.g. **32/45** (`T1.cascade.benchmark45`). To see where a number
  comes from, search NUMBERS.md for that id: it gives the result file and the
  command that regenerates it.
- Numbers are sometimes rounded or shown as a percentage or in points. No
  number here is newly computed.
- **This is a snapshot.** It matches NUMBERS.md built at config fingerprint
  `9c9a5cbcb53f`, commit `8c9ac59`. If the two ever disagree, NUMBERS.md is
  right and this file is stale.
- Some facts have no entry in NUMBERS.md, such as how many times the recurring
  bug has appeared. Those are described in words, with a pointer to where the
  count is kept.
- Each result has four parts: **In plain words**, **The number that matters**,
  **Where it lives** and **Examiner might ask**.

---

## The system in one picture

```
ticket text
   |
   v
Tier-1: TF-IDF + logistic regression --- confident enough? ---> category
   |  not confident (below the cascade gate)
   v
Tier-2: BGE embeddings + logistic regression ------------------> category
   |
   v
Retrieve the most similar past tickets (FAISS)
   |  top match similar enough? (RAG gate)
   |--- yes --> Gemini drafts a fix, grounded in those tickets
   |--- no  --> ESCALATE to a human (Gemini is never called)

offline: cluster resolved tickets by fix text (clustering gate) -> flag
         recurring fixes as automation candidates for a human to review
```

The system has three calibrated, frozen gates:
- **Cascade gate: 0.5** (`F1.gate.cascade`)
- **RAG gate: 0.67** (`F1.gate.rag`)
- **Clustering gate: 0.8** (`F1.gate.clustering`)

Gemini never chooses the category. It only writes the draft fix.

The code for all of this is in `src/agent/`, and the two gates live in
`src/agent/orchestrator.py`:
- `filing_gate()` handles escalation when a ticket can't be filed.
- `rag_gate()` handles the similarity gate.
- `run()` runs the whole sequence.

---

## Mini-glossary

- **Coverage.** Conformal prediction outputs a *set* of possible labels.
  Coverage is how often that set contains the right one.
- **Split conformal.** You score a held-out calibration set and take a
  quantile of the scores. Sets built from that quantile then reach the target
  coverage on average, *if new tickets look like the calibration tickets*.
- **α (alpha).** The error rate you allow. At α = 0.1 (`T7.alpha`) the target
  coverage is 90%.
- **Noise band (±2 s.d.).** How much a coverage number would wobble just from
  which calibration tickets you happened to draw. A gap inside the band is not
  evidence of anything.
- **Wilson CI.** A 95% confidence interval for a proportion that behaves well
  on small samples.
- **Exact McNemar test.** Compares two classifiers *on the same tickets*,
  counting only the tickets where exactly one of the two is right. A high
  p-value means you can't tell the two apart.
- **AURC.** The area under the risk–coverage curve, which averages risk over
  every possible coverage. The project gates at one coverage, not the average.
- **ECE.** Expected calibration error, the average gap between the confidence
  a model states and the accuracy it achieves.
- **Domain AUC.** Train a classifier to tell set A from set B. Near 0.5 means
  the two look the same; near 1.0 means they are trivially separable.
- **Pre-registration.** Writing down the decision rule *before* seeing the
  result, so the result can't bend the rule.
- **Post-hoc.** Found after looking at the data. It is worth reporting, but it
  can't replace a pre-registered result.

---

# The results

## R1. The corpus: where it came from, and why it's studied as an object

**In plain words.** The project follows a nasscom hackathon brief, which *asked
for* a synthetic, LLM-generated ticket dataset with a fixed set of categories.
So the synthetic data is the specification being followed, not a shortcut. We
generated **4000** tickets (`T19.dataset_rows`) across **7** categories
(`T19.categories`), from a fixed set of scenario templates. Because every
ticket comes from a template, the corpus itself became something we measure.
Several results below are really statements about what this corpus is worth.

**The number that matters.** **66** templates (`T8.templates_total`), with a
median of **62** rows each (`T8.median_rows_per_template`). The whole dataset
is a few dozen patterns, each repeated many times.

**Where it lives.**
- `data/generate_dataset.py`: `generate_tickets()`, with the templates in
  `SCENARIOS`.
- The template structure is pinned by
  `tests/test_contamination_structure.py`.
- The eval-set sizes are in `build_paper_artifacts.py: build_t19()`.

**Examiner might ask.**
- *Q: Why synthetic data at all?*
  A: The brief specified it. We followed the specification, then measured
  what that choice costs instead of hiding it.
- *Q: How big are the evaluation sets?*
  A: There are **14** (`T19.benchmark14.n`) and **45**
  (`T19.benchmark45.n`) hand-checked out-of-template tickets, **9**
  adversarial tickets (`T19.adversarial.n`), and **175**
  deployment-register tickets (`T19.deployment175.n`). All of them are
  read-only.
- *Q: What exactly is a "template"?*
  A: A (category, scenario_id) pair. scenario_id alone repeats across
  categories. Grouping by it alone was one of our bugs, and it has been
  corrected (see R9).

---

## R2. Classification accuracy, and what the baselines show

**In plain words.** In-distribution accuracy is useless here. Almost every
model scores about 100% (`T1.tfidf_baseline.in_distribution_accuracy`) because
the test rows come from the same templates as the training rows. Only the
out-of-template benchmarks measure anything. On the 45-ticket benchmark, BGE
embeddings plus logistic regression (Tier-2) get **33/45**
(`T1.tier2only.benchmark45`). TF-IDF answering everything gets only **16/45**
(`T1.tier1only.benchmark45`). Fine-tuned DistilBERT did *not* beat frozen
embeddings. The gap is about the representation, not the classifier on top of
it.

**The number that matters.** **33/45** for Tier-2 alone, Wilson
[0.590, 0.840] (`T1.tier2only.benchmark45`). That is the classifier's real
out-of-template accuracy.

**Where it lives.**
- `src/experiments/run_ablation_study.py`: `run_tier2_only()` and
  `run_no_cascade()`.
- The TF-IDF baseline is in `src/classification/generalization_test.py`:
  `fit_tfidf_logreg()`.
- DistilBERT is in `src/classification/train_distilbert.py`:
  `score_benchmark()`.

**Examiner might ask.**
- *Q: Why not fine-tune a transformer?*
  A: We did. DistilBERT scored **7/14** (`T1.distilbert.benchmark14`)
  against the TF-IDF baseline's **6/14** (`T1.tfidf_baseline.benchmark14`).
  On the 45-ticket benchmark, two runs of the *identical* configuration gave
  **18/45** (`T1.distilbert.benchmark45`) and **21/45**
  (`T1.distilbert.reference.benchmark45`). CPU fine-tuning isn't even
  reproducible there, and neither run beats frozen BGE.
- *Q: Why BGE rather than another embedding model?*
  A: It was measured against the others: MiniLM **32/45**
  (`T1.minilm.benchmark45`) and E5 **27/45** (`T1.e5.benchmark45`). BGE's
  margin over MiniLM is small, and we don't claim more than that.
- *Q: What about class imbalance?*
  A: We starved one category down to **50** rows (`T6.skew.min_am_rows`),
  from **571** (`T6.skew.max_am_rows`). In-distribution accuracy stayed at
  **0.9957** (`T6.skew.heldout_acc_at_min_rows`), which just shows the
  templates are easy. The cascade's safety net failed **0** times
  (`T6.skew.cascade_safety_net_failures_total`).

---

## R3. The cascade saves time, not accuracy (Phase 5B)

**In plain words.** The obvious claim would be that the ablation's big
accuracy jump is what the cascade adds. That is wrong: the jump is the gap
between BGE and TF-IDF, not the value of cascading. The honest comparison is the cascade against Tier-2 on its own.
The cascade is **-1** ticket (`T3.cascade_delta.benchmark45`) on the benchmark
and **-1** (`T3.cascade_delta.deployment175`) on the deployment set. The exact
McNemar test gives p = **1** (`T3.mcnemar.benchmark45`) on both, so the two
are indistinguishable. What the cascade buys is speed: Tier-2 costs about
**151×** (`T3.latency.tier2_over_tier1`) as much as Tier-1 per ticket, so
skipping Tier-2 when Tier-1 is confident saves time.

**The number that matters.** It saves **8.2%** (`T3.latency.saving.benchmark45`)
to **18.2%** (`T3.latency.saving.deployment175`) of median per-ticket latency,
at no measurable accuracy cost.

**Where it lives.**
- `src/experiments/compare_cascade_vs_tier2.py`: `mcnemar_from_pairs()`.
- `src/experiments/measure_inference_latency.py`: `time_calls()`, with
  medians **1.0356 ms** (`T3.latency.tier1_median_ms`) and **156.4024 ms**
  (`T3.latency.tier2_median_ms`).
- The paper table is built by `build_paper_artifacts.py: build_t3()`.

**Examiner might ask.**
- *Q: So the cascade is pointless?*
  A: It is pointless for accuracy and useful for latency. We say exactly
  that.
- *Q: Where does the big ablation figure come from then?*
  A: The cascade minus Tier-1-only, **35.5556** points
  (`T3.repr_gap_points`). That is the representation gap between BGE and
  TF-IDF, and it is never quoted as the value of the cascade.
- *Q: Could a bigger benchmark show a difference?*
  A: Possibly. Our sets can't detect a one-ticket difference, and we claim
  "indistinguishable", not "equal".

---

## R4. How the cascade threshold was chosen: three attempts, two rejected

**In plain words.** A threshold has to be calibrated on data where confidence
actually means something. We made **3** attempts
(`T4.calibration.attempts_total`):
- **Attempt 1** used an in-distribution held-out split and gave **0.7**
  (`T4.calibration.attempt1.threshold`). It was rejected because every bucket
  there is about 100% accurate, so the data had nothing to say about the
  threshold.
- **Attempt 2** used hand-written tickets that were never committed, so it
  can't be re-run and its numbers aren't cited.
- **Attempt 3** used the paraphrased calibration tickets. At a 90% accuracy target it
  returns **1.0001** (`T4.calibration.attempt3.threshold`), which is the
  "escalate everything" sentinel. At the 70–80% targets it returns **0.5**
  (`T4.sweep.threshold.target80`), and that became the live gate.

**The number that matters.** **0.5** (`T4.cascade.threshold`), where Tier-1
answers only **8.9%** (`T4.sweep.benchmark45_tier1_share.target80`) of the
45-ticket benchmark.

**Where it lives.**
- `src/classification/train_cascade.py`: `derive_threshold_for_target()`
  and `write_sweep_csv()`.
- The attempts are recorded by `write_calibration_attempts_csv()`, in
  `data/cascade_calibration_attempts.csv`.

**Examiner might ask.**
- *Q: Why not the 90% target?*
  A: At 90%, no confidence band on the paraphrased set is trustworthy, so the
  rule says escalate everything to Tier-2. That is a valid answer; it just
  makes Tier-1 pointless.
- *Q: What happened to attempt 2?*
  A: Its ticket set was never committed and isn't in the repo's history. It
  is recorded as having no artifact, and nothing was invented to replace it.
- *Q: Is 0.5 a round number you picked?*
  A: No. The sweep returns it at both the 70% and 80% targets.

---

## R5. The reliability (ECE) numbers show a ceiling, not good calibration

**In plain words.** The ECEs are **0.112214** for Tier-1 (`T4.tier1.ece`) and
**0.099248** for Tier-2 (`T4.tier2.ece`), measured on **500**
(`T4.reliability.n_tickets`) in-distribution tickets. They look like a normal
calibration error, but the observed accuracy is **1** in every bin
(`T4.observed_accuracy_every_bin`). So the error is just the distance from
each model's confidence up to a 100% ceiling. Both tiers are
*under-confident* here, and this tells us nothing about real traffic.

**The number that matters.** Minimum observed accuracy across all bins =
**1** (`T4.observed_accuracy_every_bin`).

**Where it lives.**
- `src/experiments/plot_calibration_curves.py`: `build_reliability_bins()`.
- The data is in `data/calibration_reliability_data.csv`.

**Examiner might ask.**
- *Q: So your models are well calibrated?*
  A: We can't tell from this data. It is a ceiling effect.
- *Q: Over- or under-confident?*
  A: Under-confident on this batch, because they're always right. Never the
  other way round (see the never-use table).
- *Q: Why report it then?*
  A: To show why in-distribution metrics can't be trusted on template data,
  and to head off a reader who would take the ECE at face value.

---

## R6. The RAG gate at 0.67, which errs in both directions

**In plain words.** The RAG gate decides whether the retrieved past tickets
are similar enough to draft a fix from. At **0.67** (`T5.gate`), all in-domain
calibration tickets proceed, **175/175** (`T5.in_domain_proceed_at_gate`), and
**13.3%** (`T5.ood_leakage_at_gate`) of out-of-domain tickets leak through.
Dropping the gate two steps nearly triples that leakage to **37.8%**
(`T5.ood_leakage_below_gate`).

The gate sits in a narrow safe band, **0.619577** (`T5.safe_range_low`) to
**0.670427** (`T5.safe_range_high`), so it is fragile. A single similarity
number makes mistakes both ways:
- **N26** (`T17.G021.ticket`) at **0.701027**
  (`T17.G021.top_similarity`) and **N31** (`T17.G024.ticket`) at
  **0.674742** (`T17.G024.top_similarity`) passed the gate and got
  ungrounded drafts.
- **N45** (`T17.escalated_adequate.0.ticket`) at **0.639674**
  (`T17.escalated_adequate.0.top_similarity`) was escalated although its
  context was adequate.

**The number that matters.** **0.67** (`T5.gate`), sitting inside a narrow
safe band rather than on a clear cliff.

**Where it lives.**
- The gate itself: `src/agent/orchestrator.py`: `rag_gate()`.
- Calibration: `src/experiments/calibrate_rag_similarity_threshold.py`:
  `find_cliff_edge()` and `run()`.
- Contamination check: **10/175** (`T5.self_retrieval_rate`) calibration
  tickets retrieve their own source row, recorded in
  `data/rag_self_retrieval_check.csv`.

**Examiner might ask.**
- *Q: Why not set it a little lower?*
  A: Two steps lower, out-of-domain leakage nearly triples, from **13.3%**
  to **37.8%**.
- *Q: Is the gate perfect?*
  A: No, and we show it isn't. N26 and N31 passed and were ungrounded, and
  N45 was escalated with adequate context. N31 passed by only **0.004742**
  (`T17.G024.margin_to_gate`).
- *Q: Is it recalibrated if the embedding model changes?*
  A: Yes. It depends on the model. The cascade gate doesn't, because it reads
  only Tier-1 confidence.

---

## R7. Adversarial escalation: all 9 decided correctly, against 3 without the gate

**In plain words.** The 9 adversarial tickets are hand-written traps. Some
*must* be escalated (out-of-scope requests, disguised non-IT issues) and some
*must* proceed. With the full pipeline, all of them are decided correctly:
**9/9** (`T3.baseline.escalation_correct`). With the RAG gate removed, only
**3/9** (`T3.norag.escalation_correct`) are right, namely the ones that should
have proceeded anyway. This set is also the project's regression gate: any
change to a threshold, model or retrieval path must still give 9/9.

**The number that matters.** **9/9** against **3/9**. The RAG gate is what
catches the traps.

**Where it lives.**
- `src/experiments/test_adversarial_escalation.py`:
  `run_ticket_through_pipeline()` and `run_all_tickets()`.
- The no-RAG arm is `run_ablation_study.py: run_no_rag()`.

**Examiner might ask.**
- *Q: Nine tickets is tiny. Is this evidence?*
  A: It is a regression gate and a case study, not a statistical claim. We
  never put a confidence interval on it.
- *Q: Wasn't this number published differently before?*
  A: Yes. The paper table briefly counted the tickets that *escalated*
  instead of the tickets *decided correctly*. That was a labelling bug, found
  in Phase 9A and corrected (see R20).
- *Q: Who wrote the adversarial tickets?*
  A: I did, by hand. That is a limitation: they test the traps I thought of.

---

## R8. Finding 1: whether coverage transfers depends on the representation

**In plain words.** We calibrated conformal prediction on the same 175
tickets for both tiers and tested both on the 45-ticket benchmark. TF-IDF's
coverage fell by **23.3 points** (`T7.tier1.coverage_gap`), which is
**10.3 s.d.** (`T7.tier1.gap_in_sd`) outside the noise band. BGE's fell by
only **1.1 points** (`T7.tier2.coverage_gap`), well inside a band of
**±4.5 points** (`T7.noise_band_2sd`). So the conformal guarantee survived the
shift for the embedding model and broke for the lexical one.

This finding has **three parts that always travel together**:
1. **It was measured on our corpus.** That is the result above.
2. **External replication is inconclusive.** Phase 7B used a *version* shift,
   which is the wrong kind of shift for a vocabulary-based failure. Tier-1's
   gap there was **+0.0009** (`T11.7b.tier1.coverage_gap`), so TF-IDF
   transferred fine. Phase 7C used the right kind of shift (paraphrase), but
   its own pre-registered rule blocked it: the domain AUC was **0.9972**
   (`T11.7c.tfidf_domain_auc`). So, outside our corpus, Finding 1 has been
   shown neither to hold nor to fail.
3. **There is post-hoc evidence of the mechanism in accuracy.** Under
   paraphrase, TF-IDF lost clearly more accuracy than BGE. The
   difference-in-differences is **−0.0804, 95% CI [−0.1364, −0.0210]**
   (`T11.7c.did_point`).

**The number that matters.** **23.3 points** lost by TF-IDF against **1.1**
lost by BGE, with a noise band of **4.5**.

**Where it lives.**
- `src/experiments/calibrate_conformal.py`: `evaluate()` and
  `coverage_sd()`.
- `src/agent/conformal.py`: `calibrate()`, `predict_sets()` and
  `coverage()`.
- 7B is in `run_external_conformal_shift.py: run()`, and 7C in
  `run_paraphrase_shift_conformal.py: paired_bootstrap_did()`.

**Examiner might ask.**
- *Q: Didn't 7B refute it?*
  A: No. 7B applied the wrong kind of shift, and that corpus has almost no
  representation gap to find: **0.3476** (`T11.7b.label_ceiling_tier1`)
  against **0.3732** (`T11.7b.label_ceiling_tier2`) accuracy. It narrows the
  scope of the finding. It does not refute it.
- *Q: Why not just unblock 7C?*
  A: Because the rule was pre-registered. Changing a rule after seeing the
  result is exactly what pre-registration exists to stop. We recorded the
  rule as a specification error instead of acting on it.
- *Q: Is the accuracy evidence enough?*
  A: It is post-hoc, so it supports the mechanism but doesn't replace the
  blocked primary test.

---

## R9. Finding 2: the calibration set can't be de-contaminated

**In plain words.** The 175 calibration tickets are paraphrases of training
rows, so it seems natural to remove the rows they came from. But the model
memorises *templates*, not individual rows. The calibration set touches
**62** (`T8.templates_touched`) of the **66** (`T8.templates_total`)
templates, so removing the source rows changes nothing. Removing whole
templates would leave only **210** (`T8.rows_surviving_template_exclusion`) of
**4000** (`T8.dataset_rows`) rows. The only real fix is data from the
deployment distribution. Filtering can't do it.

**The number that matters.** **210 of 4000** rows would survive. The set
can't be cleaned by filtering.

**Where it lives.**
- `src/experiments/calibrate_conformal.py`:
  `report_contamination_structure()`.
- `tests/test_contamination_structure.py`:
  `test_templates_are_keyed_by_category_and_scenario_id()`.

**Examiner might ask.**
- *Q: Didn't you publish different numbers for this before?*
  A: Yes. The original diagnostic grouped by scenario_id alone, which merges
  every category's templates together. Phase 5A corrected it. The conclusion was
  unchanged.
- *Q: Did the correction move the coverage numbers?*
  A: No. That measurement excludes by row id, not by template. The largest
  movement anywhere is **0.022** (`T8.max_gap_movement_clean_vs_contaminated`).
- *Q: So your calibration set is contaminated?*
  A: Yes, it is, and we say so. That is part of why we built a
  deployment-register set (R11).

---

## R10. Finding 3: conformal novelty detection matches the hand-tuned gate

**In plain words.** Instead of a hand-picked similarity threshold, a conformal
p-value can say how unusual a ticket is compared with the calibration set. It
flags all **9/9** adversarial tickets (`T9.adversarial_flagged`) and detects
out-of-domain tickets at **1** (`T9.ood_detection_seed_level`) at seed level.
Its in-domain false-escalation rate is **0.0914**
(`T9.false_escalation_in_domain`), close to α as designed. The hand-tuned 0.67
never had a calibrated false-escalation rate. This is measured only; it wasn't
promoted.

**The number that matters.** **9/9** adversarial flagged, with a
false-escalation rate that tracks α (**0.0914**).

**Where it lives.**
- `src/agent/conformal.py`: `conformal_p_values()`.
- `src/experiments/calibrate_conformal.py`: `run()`.
- The results are in `data/conformal_novelty_results.csv`.

**Examiner might ask.**
- *Q: Why not replace the 0.67 gate with it?*
  A: Promotion needs its own evidence and its own gate decision. The whole of
  Phases 5–9 is measurement only, so production is frozen.
- *Q: Is the α guarantee exact?*
  A: It holds on average over calibration draws, not conditionally on our
  particular draw.
- *Q: Is there a floor on the p-values?*
  A: Yes. The smallest possible p is **0.005682** (`T9.min_attainable_p`),
  which is 1/(n+1), where n is the number of calibration tickets.

---

## R11. Finding 4: matching the distribution is necessary but not sufficient

**In plain words.** If contamination and register mismatch cause the coverage
loss, calibrating on deployment-style tickets should fix it. We built **175**
(`T10.deployment.n_calibration`) deployment-register tickets, the same size as
the in-domain set, and recalibrated. Tier-1's gap shrank from **−0.2333**
(`T10.tier1.gap_in_domain`) to **−0.1444** (`T10.tier1.gap_deployment`). That
helps, but it is still far outside the noise band. The method, α and
benchmark were all the same; only the calibration *distribution* changed. That
is why this result is the first instance of the named finding (R21).

**The number that matters.** **−0.2333 → −0.1444**. It is better, but not
fixed.

**Where it lives.**
- `src/experiments/calibrate_conformal.py`.
- The deployment set comes from
  `src/classification/generate_deployment_calibration_set.py`.
- The results are in `data/conformal_calibration_results.csv`.

**Examiner might ask.**
- *Q: Why didn't it fully fix it?*
  A: Gemini-written "deployment-style" tickets are still not real deployment
  traffic, and 175 is small. The data is what limits it.
- *Q: Does it help Tier-2?*
  A: Tier-2 was already inside the band. For Tier-2 the benefit shows up in
  set size, not coverage.
- *Q: Is the recovery percentage in the paper?*
  A: No. We report the two gaps, not a recovery ratio.

---

## R12. Phase 2A: automation flags, a negative result

**In plain words.** The offline feature clusters resolved tickets whose fix
text is similar and flags them as automation candidates. We wanted to know
whether BGE at **0.9** (`T16.bge.pooled_cliff`) flags better than the
production MiniLM at **0.8** (`T16.minilm.pooled_cliff`). The finding: neither
ever merges tickets across templates, because in this corpus the templates
*are* the fix classes. So the two can't differ on precision, only on recall.
A blind pilot found **0/12** (`T16.pilot.false_merges`) false merges. This is
a limit of the dataset, not of the method.

**The number that matters.** **0/12** false merges. With zero events, the
honest 95% upper bound is **0.25** (`T16.pilot.rule_of_three_upper`) by the
rule of three.

**Where it lives.**
- `src/experiments/build_flag_validation_set.py`: `co_clustered_pairs()`.
- `src/experiments/score_flag_validation_set.py`:
  `rule_of_three_upper_bound()`.
- The production feature is `flag_automation_candidates.py:
  group_by_threshold()`.

**Examiner might ask.**
- *Q: Zero false merges means it's perfect?*
  A: No. Zero observed in 12 still allows a rate up to 25%.
- *Q: Why keep MiniLM?*
  A: The two are indistinguishable on precision. Switching would be a
  product decision about review-queue size, not a calibration one.
- *Q: Didn't you change a pre-registered rule here?*
  A: Yes, *before* any label was collected. A dry run showed the original
  rule would promote whichever model merged more, which inverts our
  precision-first stance. That is a design fix, not a post-hoc edit.

---

## R13. Phase 2B: groundedness, and why the LLM judge failed

**In plain words.** Of the **33** (`T15.population.eligible`) tickets that pass the RAG gate and get a Gemini
draft, **31/33** (`T14.grounded`) drafts are grounded in the retrieved
tickets, Wilson [0.804, 0.983]. We also tried an LLM judge. It agreed with my
labels on **30/33** (`T14.judge.raw_agreement`), which sounds great, but
Cohen's κ was **−0.042** (`T14.judge.cohens_kappa`), which is worse than
chance. In all **3** (`T14.judge.n_disagreements`) disagreements the judge
graded whether the advice was *appropriate* instead of whether it was
*supported by the context*.

**The number that matters.** κ = **−0.042**. A high raw-agreement score hid a
judge that was systematically answering the wrong question.

**Where it lives.**
- `src/experiments/score_groundedness_set.py`: `cohens_kappa()` and
  `wilson_interval()`.
- The judge is `run_groundedness_judge.py: build_judge_prompt()`.

**Examiner might ask.**
- *Q: How can 30/33 agreement give a negative κ?*
  A: Almost everything is "grounded", so agreeing on the majority class is
  easy. κ corrects for that, and the judge got the rare cases wrong.
- *Q: What were the two ungrounded drafts?*
  A: N26 and N31. The retrieved context was irrelevant, the draft *said so*,
  and then it gave a general-knowledge fix anyway.
- *Q: Who labelled them?*
  A: I did, alone. That is a limitation, and there is no second annotator.

---

## R14. Phase 4B-1: drift detection, measured and not shipped

**In plain words.** A drift monitor should alarm rarely when nothing changes
and reliably when something does. We tested several statistical tests at many
window sizes and α values. **25/68** (`T13.eligible_operating_points`)
operating points were eligible. The conditional binomial test held its
false-alarm rate at **0.009–0.023** (`T13.conditional_binomial.null_rate_min`,
`T13.conditional_binomial.null_rate_max`).

Then we fed it realistic, *legitimate* deployment-style tickets. Against the
in-domain reference it flagged them at **3.2×** to **38.2×** the null rate
(`T13.realistic_traffic.ratio_to_null.min`,
`T13.realistic_traffic.ratio_to_null.max`). It would alarm continuously on
normal traffic.

**The number that matters.** **3.2× to 38.2×** the null rate on legitimate
traffic. The reference distribution is the problem, not the test.

**Where it lives.**
- `src/agent/drift.py`: `detect()` and `conditional_null_rate_bound()`.
- `src/experiments/evaluate_drift_detection.py`: `realistic_traffic()` and
  `_null_rates()`.

**Examiner might ask.**
- *Q: Why not ship it?*
  A: Against the only reference we have, it would alarm constantly. It needs
  a reference built from real deployment traffic first.
- *Q: Which tests are unusable?*
  A: The KS test, for one. Its null rate grows from **0.0747**
  (`T13.ks.null_rate_at_w25`) to **0.3392** (`T13.ks.null_rate_at_w200`)
  because it reads the p-values' discreteness, not drift. We keep it as a
  descriptive read only.
- *Q: Earlier docs quoted a narrower range. Which is right?*
  A: 3.2× to 38.2×, recomputed in 9A. The old range didn't hold across all
  four α values (see the never-use table).

---

## R15. Phase 5C: zero-shot LLMs against the trained classifier

**In plain words.** Two LLMs got the same prompt with no training on our data:
- Gemini got **40/45** (`T1.zeroshot_gemini.benchmark45`), against Tier-2's
  **33/45** (`T1.tier2only.benchmark45`), with p = **0.039**
  (`T1.mcnemar.gemini_vs_tier2.benchmark45`). That difference is
  distinguishable.
- A small local model, Qwen2.5-3B, got **34/45**
  (`T1.zeroshot_qwen.benchmark45`), with p = **1**
  (`T1.mcnemar.qwen_vs_tier2.benchmark45`). That is *indistinguishable* from
  the trained classifier, which is not the same as better.

The honest headline: a 3B model on a laptop CPU, never trained on this corpus,
is not beaten by a classifier fitted on **3200** (`T19.train_rows`) of its
rows. This is zero-shot against trained, which is not a like-for-like
comparison. What it measures is how little our template corpus is worth on
out-of-template phrasing.

**The number that matters.** Qwen **34/45** against Tier-2 **33/45**, p = 1:
the training data bought nothing measurable.

**Where it lives.**
- `src/experiments/run_zeroshot_baselines.py`: `build_prompt()`, which is
  shared by both backends.
- `compare_zeroshot_vs_tier2.py: pair_by_index()`.
- `summarize_zeroshot_baselines.py: check_prompt_identity()`: all **45**
  (`T2.prompt_identity.benchmark45`) prompts are byte-identical across the
  two arms.

**Examiner might ask.**
- *Q: The benchmark was written by Gemini. Isn't Gemini's win just
  authorship?*
  A: It might be partly that. Qwen is the partial control. We don't
  attribute Gemini's extra margin to capability.
- *Q: Is Gemini better than Qwen?*
  A: Not significantly: p = **0.070**
  (`T1.mcnemar.qwen_vs_gemini.benchmark45`).
- *Q: Which categories are hard?*
  A: Infrastructure is at **0.5** recall for Tier-2
  (`T2.tier2.infrastructure.recall.benchmark45`), Gemini
  (`T2.gemini.infrastructure.recall.benchmark45`) and Qwen
  (`T2.qwen.infrastructure.recall.benchmark45`). That points to the tickets,
  not any one method. Qwen also never predicted Database: **0/5**
  (`T2.qwen.database.recall.benchmark45`).

---

## R16. Phase 6A: conformal deferral against the confidence gate

**In plain words.** Could "defer when the conformal set has more than one
label" beat "defer when confidence is below **0.5**" (`T4.cascade.threshold`)? We compared them at the
coverage the live gate actually runs at. Only **8.9%**
(`T12.6a.gate_coverage.benchmark45`) of the benchmark is answered there, so
there are very few tickets to compare on. **0** of **12** comparisons showed a
signal (`T12.6a.comparisons_with_signal`, `T12.6a.comparisons_total`). In
**3/4** (`T12.6a.degenerate_configurations`) configurations every rule accepts
the same tickets, so the test has no resolution at all. The honest verdict:
*no evidence either way on the gated axis.*

**The number that matters.** **0** signals in **12** comparisons, with
**3/4** configurations degenerate.

**Where it lives.**
- `src/experiments/compare_deferral_rules.py`: `risk_at_coverage()`,
  `operating_coverage_of_live_gate()` and `paired_bootstrap()`.

**Examiner might ask.**
- *Q: So conformal deferral is worse?*
  A: No. We have no evidence either way on our data.
- *Q: AURC showed differences. Why not use it?*
  A: AURC averages over coverages the system never runs at. Its "wins"
  contradicted each other across the two sets, and they favoured a simpler
  baseline, not conformal.
- *Q: Did anything resolve it?*
  A: On the external corpus (7B), **642** (`T12.7b.n_at_gate`) tickets sit at
  the gate, and **3/6** (`T12.7b.risk_signals`) comparisons signal, all
  favouring the confidence gate. That is a different corpus, and the result
  is never merged with ours.

---

## R17. Phase 6B: weighted conformal, a partial repair

**In plain words.** Weighted conformal re-weights calibration tickets to look
more like the test tickets, which is meant to correct for a shift. For
Tier-1, the coverage gap improved by **0.111**
(`T7.weighted.tier1.delta_vs_unweighted`), which is more than the noise band.
But what remains is still **5.4 s.d.** (`T7.weighted.tier1.residual_in_sd`)
below target, so both statements are true at once. It also costs Tier-2: in
**6/9** (`T7.weighted.tier2.tfidf.worse`) configurations Tier-2 moved further
from its target. The BGE arm was blocked, because the two sets were almost
perfectly separable in that space, with a domain AUC of **0.9908**
(`T7.weighted.bge.domain_auc`). Its numbers are never quoted.

**The number that matters.** The change was **+0.111** and the residual is
still **5.4 s.d.** below nominal. It is a partial repair, not a correction.

**Where it lives.**
- `src/experiments/run_weighted_conformal.py`:
  `cross_fitted_domain_probabilities()` and `density_ratio()`.
- `src/agent/conformal.py`: `weighted_conformal_quantile()`.

**Examiner might ask.**
- *Q: So reweighting corrects the shift?*
  A: No. It moved the gap, and the gap is still far out.
- *Q: Is reweighting better than distribution matching (R11)?*
  A: We can't order them. Their residuals differ by less than the noise
  band.
- *Q: Why is the BGE arm blocked if it looked good?*
  A: The density ratio is ill-posed when the sets are almost perfectly
  separable. Its flattering numbers are exactly the kind we refuse to quote.

---

## R18. Phase 6C: a retrieval-sufficiency check catches both, and still isn't usable

**In plain words.** Could an LLM, shown only the ticket and its retrieved
context, catch the two bad drafts from R13 before any draft is written? It
caught both, **2/2** (`T15.caught_of_ungrounded`). But it also flagged
**26/31** (`T15.flagged_of_grounded`) of the *good* ones. As a gate it would
escalate **28/33** (`T15.escalated_of_eligible`) of the tickets that currently
get a draft, and there is no threshold to tune. It also found a counter-case,
**N45** (`T17.escalated_adequate.0.ticket`): the 0.67 gate escalated it
although its context was adequate. So both checks make errors.

**The number that matters.** **26/31** false flags. The catch is worthless
without that number beside it.

**Where it lives.**
- `src/experiments/run_sufficiency_autorater.py`: `build_prompt()`.
- `src/experiments/score_sufficiency_gate.py`: `two_by_two()`.
- The summary is `data/sufficiency_gate_summary.json`.

**Examiner might ask.**
- *Q: It caught both failures. Why not use it?*
  A: Recovering two bad drafts would cost nearly all auto-resolution.
- *Q: Maybe the "false" flags are right?*
  A: Possibly. The labels measure whether the *draft* was supported, not
  whether the *context* was sufficient. That label mismatch is the binding
  limitation.
- *Q: Is this the same finding as the rest (R21)?*
  A: No. It is a mismatch between what was measured and what was labelled,
  not a distribution problem.

---

## R19. Phase 7A: the external corpus, and the control lesson

**In plain words.** To test outside our own data, we used a public dataset,
`Tobi-Bueck/customer-support-tickets`: **28261** English rows
(`T11.7a.rows_english`). It is **not real production data**
(`T11.7a.is_real_production_data`); it comes from a different generator, so it
tests "a different generator", not "reality". Its near-duplicate rate,
**79.4%** (`T11.7a.near_duplicate_rate_external`), sounds disqualifying until
you compare it with ours, which is **85.2%**
(`T11.7a.near_duplicate_rate_our_corpus`).

**The number that matters.** **79.4%** against our own **85.2%**. A
redundancy rate means nothing until you say what it is high *relative to*.

**Where it lives.**
- `src/experiments/profile_external_dataset.py`:
  `near_duplicate_stats()` and `own_corpus_comparison()`.

**Examiner might ask.**
- *Q: Why not use a real ticket dataset?*
  A: The real one we found is anonymised and encrypted, so an embedding
  model can't read it. Readable synthetic data was the better test here.
- *Q: Did the external data leak into ours?*
  A: No. An isolation check hashes our dataset and benchmarks before and
  after every run.
- *Q: Earlier domain AUCs for 7A were different?*
  A: Yes. They were computed interactively, never committed, and don't
  reproduce. The committed replacement is **0.8472**
  (`T11.7b.domain_auc_operating`).

---

## R20. Reproducibility: decisions reproduce exactly, floats don't, and the recurring bug

**In plain words.** Across Windows, a Linux container and a GitHub runner,
every routing *decision* on the **54** (`T18.golden_tickets`) golden tickets
is identical. The float values underneath differ in their last digits:
- The container's similarity differs by **1.19e-07**
  (`T18.container_8b.adv_08.rag_similarity.abs_delta`).
- The runner's differs by **7.15e-07**
  (`T18.runner_91a5b38.adv_08.rag_similarity.abs_delta`).

That is normal float32 behaviour, bounded by a derived tolerance of
**4.578e-05** (`T18.gamma_n`). We also found that **760** of the **5000**
TF-IDF features (`T18.ties.full4000.slots_for_tied`,
`T18.ties.full4000.max_features`) were picked by an unstable sort's
tie-breaking, not by the data. That is fixed by committing the vocabulary. The
project's recurring bug is "a value that is wrong for its context but looks
fine". The count and the full list are in CLAUDE.md, "The recurring bug
class".

**The number that matters.** Decisions are identical on all **54**
golden tickets. The closest one sits **3.18×** (`T18.headroom.similarity_ratio`)
the tolerance away from the 0.67 gate.

**Where it lives.**
- `tests/test_pipeline_parity.py`: `test_benchmark_parity()` and
  `_check_gate_headroom()`.
- `src/service/verify_deployment.py`: `check_adv_08()`.
- `src/experiments/measure_tier1_vocabulary_ties.py`: `tie_profile()`.
- `src/classification/train_tier1.py`: `load_vocabulary()`.

**Examiner might ask.**
- *Q: Does the model give exactly the same numbers on every machine?*
  A: No, and we withdrew an earlier claim that it did. Decisions are exact;
  the floats agree to their own precision.
- *Q: How big was the vocabulary bug's effect?*
  A: On one ticket, Tier-1 confidence moved by **1.18e-04**
  (`T18.runner_91a5b38.adv_08.tier1_confidence.abs_delta`), from a model
  nobody had changed. The decision didn't flip.
- *Q: How do you know a paper number isn't stale?*
  A: Every number is emitted from a committed result file into NUMBERS.md,
  and a parity test fails if one moves.

---

## R21. The named finding: the data a component is fitted or calibrated on is the binding constraint

**In plain words.** Three separate phases hit the same wall. The
method or the test was never the limit; the data distribution the component
was fitted or calibrated on was.
- **Coverage (R11):** calibrating on better-matched data helped, **−0.2333**
  (`T10.tier1.gap_in_domain`) to **−0.1444** (`T10.tier1.gap_deployment`),
  but didn't fix it.
- **Drift (R14):** every test alarms on legitimate traffic, **3.2×–38.2×**
  (`T13.realistic_traffic.ratio_to_null.min`,
  `T13.realistic_traffic.ratio_to_null.max`), against an in-domain reference.
- **Classification (R15):** the training data bought no measurable accuracy
  over an untrained 3B model.

Phase 2A (R12) is a *related* corpus limitation, not an instance, and 6C
(R18) is not an instance.

**The number that matters.** The same method, α and benchmark give
**−0.2333** against **−0.1444**; only the calibration data changed.

**Where it lives.**
- The wording is fixed in `paper/NUMBERS.md`, clause group `named-finding`.
- The evidence is in R11, R14 and R15.

**Examiner might ask.**
- *Q: Isn't this just "garbage in, garbage out"?*
  A: It is more specific than that. The same pattern appears in three
  different components (calibration, monitoring, training), each with a
  measured size, and in each one swapping the method didn't help.
- *Q: What would fix it?*
  A: Real deployment traffic. We say so, and it is future work.
- *Q: Why aren't 2A and 6C instances?*
  A: 2A is template redundancy (the templates are the fix classes), not a
  register mismatch. 6C is a label mismatch.

---

# The 10 hardest questions

### Q1. Your data is synthetic. Why should anyone believe any of this?

The brief specified synthetic data, so it is the specification being
followed. The honest move was to treat the corpus as something we measure,
and we did (R1, R9, R12, R15). **Our claims are about this corpus and this
kind of corpus. They are not claims about real enterprise tickets. That's a
limitation**, and the paper says so up front.

### Q2. A zero-shot LLM is more accurate. Why not just use Gemini?

The contribution is the escalation machinery, not the accuracy:
- calibrated gates;
- a system that refuses to answer when it shouldn't (R7);
- measured costs.

A zero-shot LLM gives a label with no calibrated abstention and no guarantee.
Gemini's **40/45** (`T1.zeroshot_gemini.benchmark45`) beat Tier-2's **33/45**
at p = **0.039** (`T1.mcnemar.gemini_vs_tier2.benchmark45`). Qwen2.5-3B's
**34/45** (`T1.zeroshot_qwen.benchmark45`) is indistinguishable from Tier-2,
p = **1** (`T1.mcnemar.qwen_vs_tier2.benchmark45`). We put this baseline in
the paper prominently, because that contrast *is* the argument.

### Q3. 45 tickets is tiny. What can it actually support?

Only large effects. The noise band on coverage is **±4.5 points**
(`T7.noise_band_2sd`). Finding 1's coverage gap for TF-IDF clears it easily
(R8, where all three of its clauses are given together), while a one-ticket
accuracy difference never does. So we
claim "indistinguishable", never "equal" or "better", whenever p is high.
**The small benchmark is a limitation.**

### Q4. The benchmark was written by Gemini, and Gemini wins on it. Isn't that authorship bias?

It could be partly that. Qwen is a partial control, and Qwen is *not* better
than Tier-2. We never attribute Gemini's extra margin to capability. **Not
fully controlled, so that's a limitation.**

### Q5. Finding 1 didn't replicate on external data. Isn't it just an artefact of your corpus?

Maybe. We say outright that outside our corpus it has been shown neither to
hold nor to fail:
- On our corpus the TF-IDF gap is **−0.2333** (`T7.tier1.coverage_gap`)
  against BGE's **−0.0111** (`T7.tier2.coverage_gap`), with a band of
  **0.0454** (`T7.noise_band_2sd`).
- 7B used the wrong kind of shift and got a TF-IDF gap of **+0.0009**
  (`T11.7b.tier1.coverage_gap`).
- 7C was blocked at AUC **0.9972** (`T11.7c.tfidf_domain_auc`).
- The post-hoc accuracy difference-in-differences is **−0.0804, 95% CI
  [−0.1364, −0.0210]** (`T11.7c.did_point`).

**Its scope is our corpus. That's a limitation.**

### Q6. The 0.67 threshold was tuned by hand. Where is the guarantee?

There isn't a statistical guarantee. It is an empirically calibrated
threshold, and it errs in both directions (R6). The conformal alternative
*does* come with a calibrated false-escalation rate (R10), but it wasn't
promoted, because promotion needs its own evidence. **The live gate has no
finite-sample guarantee. That's a limitation.**

### Q7. You changed a pre-registered rule in 2A, but kept one you admit was wrong in 7C. Which is it?

The difference is *when*. In 2A the rule was replaced before any label
existed, because a dry run showed it would reward whichever model merged
more. In 7C the results were already in, and changing the rule then would be
exactly the post-hoc move pre-registration forbids. So we kept it and
recorded it as a specification error. The lesson we took: a degeneracy rule
must fit the specific design, not be copied from another one.

### Q8. The same bug class kept recurring, and some of it reached published results. Why trust any number?

Because each one was found and corrected in the open, and the structure now
catches them:
- goldens that pin every routing decision;
- an adversarial gate that must stay at **9/9** (`T3.baseline.escalation_correct`);
- NUMBERS.md with a parity test;
- a lint that stops a hand-typed number reaching the paper.

The occurrences are listed in CLAUDE.md. **We can't prove there isn't
another; that's a limitation.** Every count we rely on is checked against a
second, independent derivation.

### Q9. The human labels are all yours. What about annotator bias?

That's a real limitation. The groundedness labels (R13), the 2A pilot labels
(R12) and the adversarial tickets (R7) all come from one person. There is no
inter-annotator agreement. The LLM judge was tested as a possible second rater
and failed, κ = **−0.042** (`T14.judge.cohens_kappa`), so it can't stand in
for one either.

### Q10. Nothing was promoted to production. So what is the contribution?

The contribution is the measurements: which gate is worth what, where
conformal guarantees survive and where they don't, and why. The
headline is that the data distribution is the binding constraint. Keeping
production frozen was deliberate, so that every number stays comparable.
Promotion decisions (conformal, drift, the BGE clustering swap) each need
their own evidence and are listed as open questions.

---

# Numbers and words never to use

This is the one section where these phrasings appear, and each is followed by
the right version.

| Don't say | Say instead |
|---|---|
| "The cascade adds +35.6 points" | That is BGE minus TF-IDF. The cascade is −1 ticket, indistinguishable, and saves latency (R3) |
| "The pipeline scores 6/9 on adversarial escalation" | **9/9** decided correctly, against **3/9** without the gate (R7) |
| "TF-IDF scores 7/14" | **6/14** (`T1.tfidf_baseline.benchmark14`). 7/14 didn't reproduce |
| Drift "4–7×" | **3.2×–38.2×** (R14) |
| Headroom "1.458096e-04 / 3.19×" | **1.4575e-04** (`T18.headroom.similarity_distance`) / **3.18×** |
| 7A's domain AUC "0.8584" | **0.8472** (`T11.7b.domain_auc_operating`) |
| "12 templates of ~430 rows" | **66** templates of median **62** (R9) |
| "Tier-1 is bit-identical across platforms" | Decisions are identical; the floats agree to their own precision (R20) |
| "The models are over-confident" | It is a ceiling effect; they are under-confident on that batch (R5) |
| "LLMs beat the pipeline" | Zero-shot against trained; Qwen is indistinguishable (R15) |
| "Conformal is worse" (on our data) | No evidence either way on the gated axis (R16) |
| "The shift is correctable by covariate reweighting" | A partial repair, not a correction (R17) |
| "7B refutes Finding 1" | Replication is inconclusive (R8) |
| "The sufficiency check caught the two cases the gate missed" (alone) | It caught both and flagged 26/31 good ones, so it isn't usable (R18) |
| "Adversarial tickets G021/G024" | They are 45-ticket benchmark items **N26** and **N31** (R6) |
| Any number from the adv_03 or adv_05 ticket *notes* | Read the result columns instead. Those notes are stale |
| Cascade calibration attempt 2's bucket counts | It has no committed source, so it is not quoted (R4) |
| "novel", "state-of-the-art", "the first to" | Not claimed |
