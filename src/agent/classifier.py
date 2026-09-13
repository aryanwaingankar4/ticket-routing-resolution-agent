"""
Cascade classification: Tier-1 (TF-IDF) -> Tier-2 (BGE embeddings).

Ported verbatim from streamlit_app.classify_ticket_cascade() (line 336),
which is the live demo path and the behaviour the adversarial regression test
compares against. The signature-identical copy that used to live in
test_adversarial_escalation.py:297-342 is removed by this consolidation.

The routing rule is unchanged: if Tier-1 confidence clears the threshold,
Tier-1's answer stands; otherwise the ticket escalates to the stronger,
more expensive Tier-2 model. The threshold depends only on Tier-1 (TF-IDF)
confidence, which is why swapping the Tier-2 embedding model does not
require recalibrating it.
"""

from __future__ import annotations

import numpy as np

from src.agent.artifacts import Artifacts
from src.agent.config import settings
from src.agent.errors import ClassificationError
from src.agent.schemas import ClassificationResult, Tier


def classify(text: str, artifacts: Artifacts) -> ClassificationResult:
    """Classify one ticket through the cascade."""
    from src.classification.train_cascade import get_tier1_confidence

    try:
        tier1_preds, tier1_confs = get_tier1_confidence(
            artifacts.tier1_vectorizer, artifacts.tier1_classifier, [text]
        )
        tier1_pred = str(np.asarray(tier1_preds).ravel()[0])
        tier1_conf = float(np.asarray(tier1_confs).ravel()[0])
    except Exception as exc:
        raise ClassificationError(
            f"Tier-1 classification failed "
            f"({type(exc).__name__}: {exc})."
        ) from exc

    if tier1_conf >= settings.cascade.confidence_threshold:
        return ClassificationResult(
            category=tier1_pred,
            confidence=tier1_conf,
            tier=Tier.TIER1_TFIDF,
            tier1_pred=tier1_pred,
            tier1_conf=tier1_conf,
        )

    # ---- Escalate to Tier-2 ------------------------------------------------
    try:
        embedding = artifacts.embedder.encode([text])
        embedding = np.asarray(embedding, dtype=np.float32)
        tier2_pred = artifacts.tier2_classifier.predict(embedding)[0]
    except Exception as exc:
        raise ClassificationError(
            f"Tier-2 classification failed "
            f"({type(exc).__name__}: {exc})."
        ) from exc

    # Confidence of the predicted class specifically -- falling back to the
    # max probability if the predicted label is somehow absent from classes_.
    tier2_conf = 0.0
    clf = artifacts.tier2_classifier
    if hasattr(clf, "predict_proba"):
        proba = clf.predict_proba(embedding)[0]
        classes = list(clf.classes_)
        try:
            tier2_conf = float(proba[classes.index(tier2_pred)])
        except ValueError:
            tier2_conf = float(np.max(proba))

    return ClassificationResult(
        category=str(tier2_pred),
        confidence=tier2_conf,
        tier=Tier.TIER2_EMBEDDING,
        tier1_pred=tier1_pred,
        tier1_conf=tier1_conf,
    )
