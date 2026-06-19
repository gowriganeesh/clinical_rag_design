"""Two-bucket semantic cache.

General-knowledge answers are shared-cached by question meaning. Patient-specific
queries are never shared-cached because false hits can cross patient boundaries.
"""
from __future__ import annotations

import math

from src.config import config
from src.factory import get_embeddings

_shared_store: list[tuple[list[float], str]] = []


def _cosine(a: list[float], b: list[float]) -> float:
    dot = sum(x * y for x, y in zip(a, b))
    na = math.sqrt(sum(x * x for x in a))
    nb = math.sqrt(sum(y * y for y in b))
    return dot / (na * nb) if na and nb else 0.0


def lookup(query: str, query_type: str) -> str | None:
    """Return a cached answer for a general query if similarity is tight enough."""

    if query_type != "general" or not _shared_store:
        return None
    qv = get_embeddings().embed(query)
    best_answer, best_sim = None, 0.0
    for emb, answer in _shared_store:
        sim = _cosine(qv, emb)
        if sim > best_sim:
            best_sim, best_answer = sim, answer
    return best_answer if best_sim >= config.cache_similarity_threshold else None


def store(query: str, query_type: str, answer: str) -> None:
    """Cache only general-query answers."""

    if query_type != "general":
        return
    _shared_store.append((get_embeddings().embed(query), answer))


def clear() -> None:
    _shared_store.clear()
