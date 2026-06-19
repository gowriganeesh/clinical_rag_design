"""Orchestrator for the multi-agent answering pipeline.

scrub PHI -> route -> invalid: fallback
                   -> valid: cache lookup -> miss: retrieve + answer
"""
from __future__ import annotations

from src.cache import semantic_cache
from src.factory import get_fallback_agent, get_retriever_agent, get_router
from src.tools.access_control import answer_roster_query, is_roster_query
from src.tools.deid import scrub_phi


def respond(raw_query: str, user_id: str) -> dict:
    """Return {"answer": str, "meta": {...}} for a query from a user_id."""

    query = scrub_phi(raw_query)
    if is_roster_query(query):
        return {
            "answer": answer_roster_query(user_id),
            "meta": {"path": "access_control", "valid": True, "type": "access"},
        }

    decision = get_router().route(query)
    if not decision["valid"]:
        return {
            "answer": get_fallback_agent().handle(query),
            "meta": {"path": "fallback", "valid": False, "type": decision["type"]},
        }

    qtype = decision["type"]
    cached = semantic_cache.lookup(query, qtype)
    if cached is not None:
        return {"answer": cached, "meta": {"path": "cache_hit", "type": qtype}}

    answer = get_retriever_agent().answer(query, user_id=user_id)
    if qtype == "general":
        semantic_cache.store(query, qtype, answer)

    return {"answer": answer, "meta": {"path": "retrieve", "type": qtype}}
