# FRAMING.md -- how the results must be written up

**GENERATED. Do not edit by hand.** Every number below is interpolated from `NUMBERS.md`, so this file cannot drift away from the measurements it describes.

This file exists for Phase 9A. Each item was agreed at a phase gate and is not reopened here.

---

## 1. The corpus is an object of study, not a given

The nasscom brief recommended a synthetic, LLM-generated dataset. We followed it, at 4000 rows, and the experimental-setup section opens by saying so. The corpus-as-object-of-study framing follows immediately: Phase 2A, Finding 2 and Phase 5C are measurements of what that corpus is worth, not incidental limitations.

In-distribution accuracy is uninformative on template-generated data -- every model scores about 100%. Only the 14- and 45-ticket benchmarks measure anything real, and a new 100% in-distribution number is a red flag rather than a success.

## 2. The named cross-phase finding

> THE NAMED CROSS-PHASE FINDING -- the calibration/reference distribution, not the test or the method, is the binding constraint. Instances: Phase 1 Finding 4 (coverage), Phase 4B's realistic-traffic arm (drift), Phase 5C (classification, which widened it to the TRAINING distribution as well). Phase 2A is a RELATED corpus limitation, not an instance. Phase 6C is NOT an instance. Phase 9A settles the final wording.

Its sharpest single instance is Finding 4: the same method at the same alpha against the same benchmark moves from a coverage gap of -0.233333 to -0.144444 when only the calibration DISTRIBUTION changes.

Phase 9A settles the final wording. Phase 5C widened the mechanism from the calibration/reference distribution to the training distribution as well; the two original instances are unchanged and are not weakened by the addition.

## 3. The contribution is the calibrated escalation machinery, not accuracy

> PHASE 5C -- the correct sentence is: a 3B model on a laptop CPU with no training on this corpus is not beaten by a classifier fitted on 3,200 of its rows, across two vendors. NEVER 'LLMs beat the pipeline'. It is zero-shot LLM vs trained classifier, which is not like-for-like, and what it measures is what a template-generated corpus is worth on out-of-template phrasing. The readings differ by pair and must not be merged: Gemini is distinguishably better than Tier-2 on the 45; Qwen2.5-3B is INDISTINGUISHABLE from it, which is not 'better'. The benchmark is Gemini-generated, so the Qwen arm is the partial control for authorship.

The contrast is the argument, so the zero-shot baseline goes in the paper prominently and is never buried. Zero-shot Gemini scores 40/45 [95% CI 0.765009, 0.951595] against the trained Tier-2's 33/45 [95% CI 0.589609, 0.840353] (exact McNemar p = 0.039062) while having no abstention guarantee, no calibrated gate and no cost model. Qwen2.5-3B, running locally on a laptop CPU, scores 34/45 [95% CI 0.613266, 0.857645] and is indistinguishable from Tier-2 (p = 1).

## 4. Methodological findings

### Ungated and averaged metrics manufacture significance

> PHASE 6A -- the honest claim is 'no evidence either way on the gated axis', NEVER 'conformal is worse' on our data. All twelve comparisons returned no signal at the live gate's measured operating coverage, and three of four configurations are degenerate there. An AURC win is never a reason to promote: this project gates on risk at the operating coverage, and AURC averages over coverages the system never runs at.

At the live gate, 3/4 configurations are degenerate and 0 of 12 comparisons show a signal. Phase 2A's decision-rule inversion is the second instance of the same shape.

### In-distribution calibration is a ceiling effect, not calibration

On the 500-ticket in-distribution batch the observed accuracy is 1 in every confidence bin of both tiers. The reported ECE -- Tier-1 0.112214, Tier-2 0.099248 -- is therefore entirely the distance between the model's confidence and a ceiling, and both tiers are systematically UNDER-confident there. It is not evidence that either is well calibrated on real traffic. This is the same ceiling effect as the ~100% in-distribution accuracy, and it is why both escalation gates were calibrated empirically against benchmark behaviour rather than read off a predicted probability.

### An LLM judge must never be used unaudited

Raw agreement with the human labels was 30/33 and Cohen's kappa was -0.042105. Every disagreement was the judge substituting APPROPRIATENESS for SUPPORT, in both directions. An aggregate agreement score would not have revealed this; only labelling the full population and reading each disagreement did. A judge is an object of study in this project, never a scaling tool.

### A degeneracy rule must be justified per design

Phase 7C's blocking rule was copied from Phase 6B, where it guarded a DENSITY-RATIO estimate. 7C computes no density ratio and uses the AUC only as a manipulation check, where a near-1.0 value means the manipulation was STRONG. The critique is sound and the gate still DECLINED to unblock, because revising a pre-registered rule after seeing the results is precisely what the discipline exists to prevent. It is recorded as a specification error, not acted on. The counter-reading is recorded too: a rewrite a classifier identifies with near-certainty may be a DIFFERENT corpus rather than a shifted one.

### A number quoted at a gate must come from a committed script

Phase 7A's two design domain AUCs were computed interactively, quoted at a gate, never committed, and do not reproduce. The conclusions were unaffected because every value cleared the same threshold, but the figures were wrong in print for a day. This is the motivation for Phase 8A and belongs in the reproducibility section.

### What reproduces across machines, and what does not (Phase 8B)

Running the pipeline on a second platform separated three kinds of number, and the distinction belongs in the reproducibility section rather than in a footnote.

DECISIONS reproduce exactly. Category, tier, escalated and the number of retrieved neighbours were identical on all 54 golden tickets between Windows and a Linux container. No routing decision has ever differed.

EMBEDDING SIMILARITIES DO NOT, and cannot. They are float32 inner products of 768-dimensional normalised vectors, accumulated in an order set by the machine's BLAS kernel and SIMD width: 2.384e-07 between this machine and a Linux container, about 7e-07 against a GitHub runner. The published figures are quoted to six decimals and are unaffected. Parity is therefore asserted with a DERIVED bound -- gamma_n = n*u/(1-n*u) = 4.578e-05 for n=768 at float32 unit roundoff u = 2^-24 (Higham, section 3.1) -- and never with a tolerance fitted to one machine's measurement. That bound covers the inner product only; it does not model the encoder's own cross-platform difference.

TIER-1's VOCABULARY WAS NOT DETERMINED BY THE DATA, and this is the finding worth reporting. TfidfVectorizer(max_features=5000) keeps the most frequent terms using an UNSTABLE quicksort. On the 4,000-row corpus 4,240 terms sit strictly above the cut and 11,834 terms tie at a count of 1 for the remaining 760 slots -- so 760 of the 5,000 features, 15.2%, were selected by the sort's tie-break rather than by the corpus. numpy dispatches SIMD sorts by CPU, so a GitHub runner kept a different vocabulary and produced a Tier-1 confidence of 0.31818 where this machine produces 0.31830 -- a difference of 1.18e-04, about 1e11 times float64 noise, from a model nobody had changed. Phase 8B.3 commits the vocabulary as an artifact and fits against it, which moves the published probabilities by 8.9e-16 (float64 construction order, 1.25 ulp) and no decision at all.

THE SAME EXPOSURE REMAINS, UNFIXED, IN FOUR OTHER FITS: generalization_test.py (both arms), train_baseline_tfidf.py and run_imbalance_sweep.py all use max_features=5000. On the 80/20 training split the tie is larger still -- 950 of 5,000 features. They were deliberately NOT changed, because refitting them would move published numbers. The limitation to state is therefore: those figures reproduce on this platform and may differ slightly on another.

## 5. The escalation gate errs in both directions

The RAG similarity gate sits at 0.67, inside the adversarial safe range [0.619577, 0.670427]. It is not a perfect separator and the paper says so in both directions: adversarial tickets G021 and G024 pass the gate when they should not, and ticket N45 (top similarity 0.6397) is escalated although Phase 6C's rater found its retrieved context adequate. A scalar threshold on a single similarity is the simplest thing that works, not a claim that it is sufficient.

## 6. Separability does not imply score shift

Phase 6B's cross-fitted domain classifier separates the calibration set from the deployment set MORE easily in BGE space (0.990792) than in TF-IDF space -- and BGE is nevertheless the space whose coverage transfers. The BGE arm's own numbers are BLOCKED, not a result; the AUC itself is a quantitative statement of the named finding. Cross-reference Finding 1.

> PHASE 6B -- weighted conformal is a PARTIAL REPAIR, NOT A CORRECTION. Never write 'the shift is correctable by covariate reweighting'. Both readings are reported together: the CHANGE cleared the +/-2 s.d. band, and the RESIDUAL is still 5-6 s.d. below nominal. No ordered comparison between reweighting and distribution matching -- the residuals differ by less than the band. The BGE arm is BLOCKED, not a result.

## 7. Finding 1, in full

> FINDING 1 -- the three clauses travel together, and none is quoted alone:
>   (1) MEASURED ON OUR CORPUS: split-conformal coverage transfers for the BGE representation but not the lexical one.
>   (2) EXTERNAL REPLICATION IS INCONCLUSIVE: Phase 7B applied a VERSION shift -- the wrong shift type for a mechanism that is surface-vocabulary change -- and Phase 7C applied the right type but was BLOCKED by its own pre-registered degeneracy rule. After 7B and 7C, Finding 1 has been shown neither to hold nor to fail outside its original corpus. Never write '7B refutes Finding 1'.
>   (3) POST-HOC EVIDENCE OF THE MECHANISM IN ACCURACY on the external corpus: difference-in-differences -0.0804, 95% CI [-0.1364, -0.0210]. Post-hoc, and it does not replace the blocked primary.

On our corpus, at alpha = 0.1: Tier-1's coverage gap is -0.233333 against Tier-2's -0.011111, with a +/-2 s.d. band of 0.045356. On the external corpus under a version shift the two are indistinguishable (0.000872 against -0.007174, band 0.016471) -- but that corpus has almost no representation gap to find (Tier-1 0.347572 vs Tier-2 0.37324), which weakens 7B as evidence and is a limitation, not a rescue.

A test-arm coverage band must include the test-sampling term. At Phase 7C's n = 286 the calibration-only band is 0.016471 while the combined band is 0.039116; quoting the calibration term alone on a small test arm understates uncertainty by roughly 2x.

## 8. Phase 6C

> PHASE 6C -- the headline is 'catches both, and is still not usable as a gate', never 'the sufficiency check caught the two cases the gate missed', which is true and misleading alone. 6C is NOT an instance of the named finding; the near-duplicate explanation is a REJECTED hypothesis, refuted by 2B's own distinct-fix diagnostic, and the mechanism is a MEASUREMENT-TARGET MISMATCH between sufficiency and groundedness. The binding limitation is the LABEL, not the rater. The N45 counter-case travels with the result: the scalar errs in both directions.

The rater caught 2/2 of the human-labelled ungrounded drafts and flagged 26/31 [95% CI 0.673653, 0.929075] of the grounded ones. A gate escalating 28/33 of the tickets that currently reach the resolver is not usable, and there is nothing to tune.

## 9. Phase 2A and the clustering question

Phase 2A is a RELATED corpus limitation, not an instance of the named finding. Its pilot found 0/12 false merges, which bounds the false-merge rate above by 0.25 by the rule of three -- the honest reading of a zero count. Promotion of BGE clustering is now a product decision about review-queue capacity, not a calibration one, and must not be reopened as a calibration question.

## 10. What the system is, described honestly

A sequential pipeline with two independently calibrated confidence gates, agent boundaries and an HTTP surface. The agents are independently ADDRESSABLE, not independently running: POST /triage orchestrates in process. It is never described as a distributed or autonomous multi-agent system. Gemini never decides the category; classification is entirely the trained models', and Gemini's only job is resolution generation grounded in retrieved tickets.

> PHASE 7A's CONTROL LESSON -- never quote a redundancy or similarity rate without stating what it is high RELATIVE TO. Our own corpus is more near-duplicated than the external one. High near-duplication is a property of template-generated corpora generally, not a flaw of that dataset.

The external dataset is INDEPENDENTLY GENERATED DATA, NOT REAL PRODUCTION DATA -- its card advertises a synthetic generator from the same author. It tests whether findings survive a DIFFERENT GENERATOR, not whether they survive reality. Any write-up must say so in those words.
