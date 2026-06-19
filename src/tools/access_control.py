"""Helpers for access-control questions.

Questions such as "Which patients are under me?" are answered from the Azure
Search index using the acting user's ACL filter. No patient roster is displayed
until the user asks for it.
"""
from __future__ import annotations

import json
import re
from functools import lru_cache

from src.config import config

_ROSTER_PATTERNS = [
    re.compile(r"\b(who|which|what)\b.*\b(patient|patients)\b.*\b(under|assigned|mine|my|me|for me|cover)\b", re.I),
    re.compile(r"\b(my|assigned)\b.*\b(patient|patients|caseload|panel)\b", re.I),
    re.compile(r"\b(patient|patients)\b.*\b(under me|assigned to me|in my care|on my panel)\b", re.I),
    re.compile(r"\b(caseload|patient panel|care team)\b", re.I),
]


def is_roster_query(query: str) -> bool:
    return any(pattern.search(query) for pattern in _ROSTER_PATTERNS)


def answer_roster_query(user_id: str) -> str:
    """Answer a roster question from indexed ACL-filtered documents."""

    try:
        rows = _authorized_documents_from_index(user_id)
    except _resource_not_found_error():
        return "The clinical index is not ready yet. Use Data / Ingestion first."

    if not rows:
        return "No patients were found for this user in the clinical index."

    patient_rows = [
        f"- {row['patient_id']} ({row['source_doc']})"
        for row in sorted(rows, key=lambda item: item["patient_id"])
    ]
    if user_id.startswith("EXAMINER"):
        heading = f"{user_id} is authorized for these patients:"
    else:
        heading = f"{user_id} is authorized for this record:"
    return "\n".join([heading, *patient_rows])


def all_users() -> list[str]:
    """Return users for the prototype selector.

    Prefer indexed ACL values once ingestion has run. Fall back to the manifest
    only so the selector is still usable before the index exists; production
    should replace this dropdown with the authenticated Entra principal.
    """

    try:
        users = _all_users_from_index()
        if users:
            return users
    except Exception:
        pass
    return _all_users_from_manifest()


def _authorized_documents_from_index(user_id: str) -> list[dict]:
    client = _search_client()
    acl_filter = f"acl/any(u: u eq '{_odata_escape(user_id)}')"
    results = client.search(
        search_text="*",
        filter=acl_filter,
        select=["patient_id", "source_doc"],
        top=1000,
    )
    by_patient: dict[str, dict] = {}
    for row in results:
        patient_id = row.get("patient_id")
        source_doc = row.get("source_doc")
        if patient_id and source_doc and patient_id not in by_patient:
            by_patient[patient_id] = {"patient_id": patient_id, "source_doc": source_doc}
    return list(by_patient.values())


def _all_users_from_index() -> list[str]:
    client = _search_client()
    results = client.search(search_text="*", select=["acl"], top=1000)
    users = []
    for row in results:
        for user in row.get("acl", []) or []:
            if user not in users:
                users.append(user)
    return _order_users(users)


@lru_cache(maxsize=1)
def _manifest_documents() -> list[dict]:
    manifest_path = config.data_dir / "access_control.json"
    payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    return payload["documents"]


def _all_users_from_manifest() -> list[str]:
    users = []
    for entry in _manifest_documents():
        for user in entry.get("acl", []):
            if user not in users:
                users.append(user)
    return _order_users(users)


def _search_client():
    config.validate_production()
    from azure.core.credentials import AzureKeyCredential
    from azure.search.documents import SearchClient

    return SearchClient(
        endpoint=config.search_endpoint,
        index_name=config.index_name,
        credential=AzureKeyCredential(config.search_key),
    )


def _resource_not_found_error():
    from azure.core.exceptions import ResourceNotFoundError

    return ResourceNotFoundError


def _odata_escape(value: str) -> str:
    return value.replace("'", "''")


def _order_users(users: list[str]) -> list[str]:
    examiners = sorted([user for user in users if user.startswith("EXAMINER")])
    patients = sorted([user for user in users if user.startswith("PAT_")])
    others = sorted([user for user in users if user not in examiners and user not in patients])
    return examiners + patients + others
