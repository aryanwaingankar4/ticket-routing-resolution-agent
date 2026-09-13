"""
FAISS retrieval over past tickets.

Ported from suggest_resolution.retrieve_similar_tickets() (line 315), keeping
its two load-bearing details intact:

  * The query is L2-normalized before searching. The index is IndexFlatIP
    over normalized vectors, so inner product equals cosine similarity ONLY
    if the query is normalized the same way. Skipping this yields
    plausible-looking but wrongly-ranked results.

  * k is clamped to index.ntotal, and negative FAISS indices (empty slots)
    are skipped defensively.

The returned ordering is descending by similarity. IndexFlatIP already
returns results in that order, but the explicit sort is kept rather than
relying on it -- all three former copies of this pipeline re-sorted
defensively, and that consensus is worth preserving.
"""

from __future__ import annotations

import numpy as np

from src.agent.artifacts import Artifacts
from src.agent.config import settings
from src.agent.errors import RetrievalError
from src.agent.schemas import RetrievalResult, RetrievedTicket


def retrieve(text: str, artifacts: Artifacts,
             top_k: int | None = None) -> RetrievalResult:
    """Retrieve the most similar past tickets for a query string."""
    k_requested = settings.rag.top_k if top_k is None else top_k
    faiss = artifacts.faiss
    index = artifacts.index

    try:
        query_emb = artifacts.embedder.encode([text], convert_to_numpy=True)
        query_emb = np.asarray(query_emb, dtype=np.float32)
        query_emb = np.ascontiguousarray(query_emb)
        faiss.normalize_L2(query_emb)

        k = min(k_requested, index.ntotal)
        scores, indices = index.search(query_emb, k)
    except Exception as exc:
        raise RetrievalError(
            f"FAISS retrieval failed ({type(exc).__name__}: {exc})."
        ) from exc

    retrieved: list[RetrievedTicket] = []
    for score, idx in zip(scores[0], indices[0]):
        if idx < 0:
            continue
        entry = artifacts.metadata[idx]
        retrieved.append(
            RetrievedTicket(
                id=entry.get("id"),
                title=entry.get("title", "") or "",
                description=entry.get("description", "") or "",
                category=entry.get("category", "") or "",
                resolution=entry.get("resolution", "") or "",
                priority=str(entry.get("priority") or ""),
                similarity=float(score),
            )
        )

    retrieved.sort(key=lambda r: r.similarity, reverse=True)
    return RetrievalResult(retrieved=retrieved)
