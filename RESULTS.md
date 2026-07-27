# Model Comparison Results — IT Ticket Classification

## Dataset
1000 synthetic tickets, 7 categories (balanced ~142-143 each), 0 duplicate
title/description pairs. Generated via `data/generate_dataset.py`.

## Evaluation methodology
Two metrics reported for every method:
- **In-distribution accuracy**: held-out 20% split of the same synthetic,
  template-generated data (200 tickets).
- **Generalization score**: 14 hand-written tickets phrased in everyday,
  non-technical language that never appears in the training templates
  (2 per category). This is the metric that actually matters — it measures
  whether the model learned real signal or just memorized template vocabulary.

## Results

| Method | In-Distribution Accuracy | Generalization Score |
|---|---|---|
| TF-IDF + Logistic Regression | 100.0% | 7/14 (50.0%) |
| Frozen embeddings (all-MiniLM-L6-v2) + Logistic Regression | 100.0% | **10/14 (71.4%)** ✅ Best |
| Fine-tuned DistilBERT (best epoch: 1) | 95.0% | 6/14 (42.9%) |
| Fine-tuned DistilBERT (epoch 4, fully converged) | 100.0% | 5/14 (35.7%) |

**Winner: frozen MiniLM sentence embeddings.**

## Key finding: in-distribution accuracy is a misleading metric here

All three methods hit ~95-100% in-distribution accuracy. If we had only
looked at that number, every method would look equally excellent — and we'd
have picked the wrong one. The generalization test is what actually
differentiates them.

## Key finding: fine-tuning overfit rather than improved

DistilBERT's per-epoch trend during training:

| Epoch | Train Loss | In-Dist Val Accuracy | Generalization Score |
|---|---|---|---|
| 1 | 1.628 | 95.00% | 6/14 (42.9%) |
| 2 | 0.511 | 100.00% | 5/14 (35.7%) |
| 3 | 0.074 | 100.00% | 5/14 (35.7%) |
| 4 | 0.027 | 100.00% | 5/14 (35.7%) |

In-distribution accuracy saturated at 100% by epoch 2, while generalization
*declined* from epoch 1 onward. This is a textbook overfitting signature:
past epoch 1, the model was memorizing the specific synthetic templates
rather than learning generalizable patterns. With only 800 training
examples built from a limited set of scenario templates, DistilBERT's high
parameter count let it memorize the training set almost perfectly — at the
cost of real-world generalization.

This is a legitimate, evidence-backed finding, not a failed experiment:
more powerful models are not automatically better on small, templated
datasets. Model capacity needs to be matched to dataset size and diversity.

## Why frozen embeddings won

Sentence-transformer embeddings (all-MiniLM-L6-v2) are pretrained on a huge,
diverse corpus and never see the ticket templates during training — only the
small Logistic Regression classifier on top is trained on our data. This
gave the best of both worlds: genuine semantic understanding (unlike TF-IDF's
keyword matching) without the capacity to memorize/overfit our limited
template set (unlike fine-tuned DistilBERT).

## Conclusion for the classification layer

We selected **frozen MiniLM embeddings + Logistic Regression** as the
production classifier for the ticket routing system, based on measured
generalization performance rather than in-distribution accuracy alone.