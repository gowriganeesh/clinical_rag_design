"""Clinical records retrieval tool backed by Azure AI Search."""
from __future__ import annotations

import re

from src.config import config

SEMANTIC_CONFIG_NAME = "clinical-semantic"
VECTOR_FIELD = "contentVector"
_PATIENT_RE = re.compile(r"\b(?:PAT[_\-\s]?)?([A-Z]\d{2})\b", re.IGNORECASE)


class AzureSearchTool:
    """ACL-filtered hybrid + semantic search over Azure AI Search."""

    def __init__(self) -> None:
        config.validate_production()
        from azure.core.credentials import AzureKeyCredential
        from azure.search.documents import SearchClient
        from openai import AzureOpenAI

        self._aoai = AzureOpenAI(
            azure_endpoint=config.aoai_endpoint,
            api_key=config.aoai_key,
            api_version=config.aoai_api_version,
        )
        self._search = SearchClient(
            endpoint=config.search_endpoint,
            index_name=config.index_name,
            credential=AzureKeyCredential(config.search_key),
        )

    def retrieve(self, query: str, user_id: str) -> dict:
        """Run ACL-filtered hybrid + semantic search.

        Returns {"confident": bool, "chunks": [{content, source_doc,
        patient_id, reranker_score}]}.
        """

        from azure.search.documents.models import VectorizedQuery

        vector_query = VectorizedQuery(
            vector=self._embed(query),
            k_nearest_neighbors=config.retrieval_top_k,
            fields=VECTOR_FIELD,
        )
        acl_filter = f"acl/any(u: u eq '{_odata_escape(user_id)}')"
        patient_ids = _mentioned_patient_ids(query)
        if len(patient_ids) == 1:
            acl_filter = f"{acl_filter} and patient_id eq '{_odata_escape(next(iter(patient_ids)))}'"
        elif len(patient_ids) > 1:
            patient_filter = " or ".join(
                f"patient_id eq '{_odata_escape(patient_id)}'"
                for patient_id in sorted(patient_ids)
            )
            acl_filter = f"{acl_filter} and ({patient_filter})"
        results = self._search.search(
            search_text=query,
            vector_queries=[vector_query],
            filter=acl_filter,
            query_type="semantic",
            semantic_configuration_name=SEMANTIC_CONFIG_NAME,
            top=config.rerank_top_n,
            select=["content", "source_doc", "patient_id"],
        )

        chunks = []
        for row in results:
            chunks.append(
                {
                    "content": row["content"],
                    "source_doc": row["source_doc"],
                    "patient_id": row.get("patient_id"),
                    "reranker_score": row.get("@search.reranker_score"),
                }
            )

        top_score = (chunks[0]["reranker_score"] if chunks else 0.0) or 0.0
        confident = bool(chunks) and top_score >= config.reranker_score_threshold
        return {"confident": confident, "chunks": chunks}

    def _embed(self, text: str) -> list[float]:
        response = self._aoai.embeddings.create(
            model=config.embed_deployment,
            input=text,
            dimensions=config.embed_dimensions,
        )
        return response.data[0].embedding


def retrieve(query: str, user_id: str) -> dict:
    from src.factory import get_search

    return get_search().retrieve(query, user_id=user_id)


def _odata_escape(value: str) -> str:
    return value.replace("'", "''")


def _mentioned_patient_ids(query: str) -> set[str]:
    return {f"PAT_{match.group(1).upper()}" for match in _PATIENT_RE.finditer(query)}
