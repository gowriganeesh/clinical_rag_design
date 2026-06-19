"""Ingestion pipeline.

Flow: read documents -> join ACL -> sentence-aware chunk -> embed -> create the
index if needed -> upload only missing chunks.
"""
from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable

from src.config import config
from src.factory import get_embeddings
from src.ingest.chunking import chunk_text

SEMANTIC_CONFIG_NAME = "clinical-semantic"
VECTOR_PROFILE_NAME = "clinical-hnsw-profile"
VECTOR_ALGO_NAME = "clinical-hnsw"
VECTOR_COMPRESSION_NAME = "clinical-sq-compression"

ProgressCallback = Callable[[str, dict], None]


@dataclass
class DocumentIngestStats:
    source_doc: str
    patient_id: str
    chunks_created: int
    uploaded: int
    skipped: int


@dataclass
class IngestStats:
    documents_found: int = 0
    chunks_created: int = 0
    uploaded: int = 0
    skipped: int = 0
    total_index_chunks: int = 0
    per_document: list[DocumentIngestStats] = field(default_factory=list)


def load_documents() -> list[dict]:
    manifest = json.loads((config.data_dir / "access_control.json").read_text(encoding="utf-8"))
    docs = []
    for entry in manifest["documents"]:
        text = (config.data_dir / "documents" / entry["source_doc"]).read_text(encoding="utf-8")
        docs.append(
            {
                "source_doc": entry["source_doc"],
                "patient_id": entry["patient_id"],
                "acl": entry["acl"],
                "text": text,
            }
        )
    return docs


def index_has_data() -> bool:
    try:
        return index_count() > 0
    except _resource_not_found_error():
        return False


def index_count() -> int:
    _, search_client = _get_real_index_clients()
    try:
        return search_client.get_document_count()
    except _resource_not_found_error():
        return 0


def ingest(progress: ProgressCallback | None = None) -> IngestStats:
    """Run idempotent ingestion and return structured counts."""

    return _ingest_real(progress)


def _ingest_real(progress: ProgressCallback | None) -> IngestStats:
    index_client, search_client = _get_real_index_clients()
    ensure_index(index_client)
    embeddings = get_embeddings()
    stats = IngestStats()
    docs = load_documents()
    stats.documents_found = len(docs)
    _emit(progress, "start", {"documents_found": stats.documents_found})

    batch: list[dict] = []
    for doc in docs:
        chunks = chunk_text(
            doc["text"],
            max_tokens=config.chunk_size_tokens,
            overlap_sentences=config.chunk_overlap_sentences,
        )
        doc_stats = DocumentIngestStats(
            source_doc=doc["source_doc"],
            patient_id=doc["patient_id"],
            chunks_created=len(chunks),
            uploaded=0,
            skipped=0,
        )

        for index, chunk in enumerate(chunks):
            doc_id = _chunk_id(doc["source_doc"], index)
            if _real_doc_exists(search_client, doc_id):
                doc_stats.skipped += 1
                continue
            batch.append(
                {
                    "id": doc_id,
                    "content": chunk,
                    "contentVector": embeddings.embed(chunk),
                    "patient_id": doc["patient_id"],
                    "acl": doc["acl"],
                    "source_doc": doc["source_doc"],
                    "chunk_index": index,
                }
            )
            doc_stats.uploaded += 1
            if len(batch) == 100:
                search_client.upload_documents(documents=batch)
                batch.clear()

        _merge_doc_stats(stats, doc_stats)
        _emit(progress, "document", doc_stats.__dict__)

    if batch:
        search_client.upload_documents(documents=batch)

    stats.total_index_chunks = _wait_for_document_count(
        search_client,
        minimum=stats.uploaded + stats.skipped,
    )
    _emit(progress, "complete", {"total_index_chunks": stats.total_index_chunks})
    return stats


def build_index_definition():
    from azure.search.documents.indexes.models import (
        HnswAlgorithmConfiguration,
        SearchField,
        SearchFieldDataType,
        SearchIndex,
        SearchableField,
        SemanticConfiguration,
        SemanticField,
        SemanticPrioritizedFields,
        SemanticSearch,
        SimpleField,
        VectorSearch,
        VectorSearchProfile,
    )

    fields = [
        SimpleField(name="id", type=SearchFieldDataType.String, key=True),
        SearchableField(name="content", type=SearchFieldDataType.String),
        SearchField(
            name="contentVector",
            type=SearchFieldDataType.Collection(SearchFieldDataType.Single),
            searchable=True,
            hidden=True,
            vector_search_dimensions=config.embed_dimensions,
            vector_search_profile_name=VECTOR_PROFILE_NAME,
        ),
        SimpleField(name="patient_id", type=SearchFieldDataType.String, filterable=True),
        SimpleField(
            name="acl",
            type=SearchFieldDataType.Collection(SearchFieldDataType.String),
            filterable=True,
        ),
        SimpleField(name="source_doc", type=SearchFieldDataType.String, filterable=True),
        SimpleField(name="chunk_index", type=SearchFieldDataType.Int32, filterable=True),
    ]

    vector_kwargs = {
        "algorithms": [HnswAlgorithmConfiguration(name=VECTOR_ALGO_NAME)],
        "profiles": [
            VectorSearchProfile(
                name=VECTOR_PROFILE_NAME,
                algorithm_configuration_name=VECTOR_ALGO_NAME,
                compression_name=VECTOR_COMPRESSION_NAME,
            )
        ],
    }
    try:
        from azure.search.documents.indexes.models import ScalarQuantizationCompression

        vector_kwargs["compressions"] = [
            ScalarQuantizationCompression(
                compression_name=VECTOR_COMPRESSION_NAME,
                rerank_with_original_vectors=True,
                default_oversampling=10,
            )
        ]
    except Exception:
        vector_kwargs["profiles"] = [
            VectorSearchProfile(
                name=VECTOR_PROFILE_NAME,
                algorithm_configuration_name=VECTOR_ALGO_NAME,
            )
        ]

    vector_search = VectorSearch(**vector_kwargs)
    semantic_search = SemanticSearch(
        configurations=[
            SemanticConfiguration(
                name=SEMANTIC_CONFIG_NAME,
                prioritized_fields=SemanticPrioritizedFields(
                    content_fields=[SemanticField(field_name="content")]
                ),
            )
        ]
    )
    return SearchIndex(
        name=config.index_name,
        fields=fields,
        vector_search=vector_search,
        semantic_search=semantic_search,
    )


def ensure_index(index_client) -> None:
    if config.index_name in list(index_client.list_index_names()):
        return
    index_client.create_index(build_index_definition())


def _get_real_index_clients():
    config.validate_production()
    from azure.core.credentials import AzureKeyCredential
    from azure.search.documents import SearchClient
    from azure.search.documents.indexes import SearchIndexClient

    credential = AzureKeyCredential(config.search_key)
    index_client = SearchIndexClient(endpoint=config.search_endpoint, credential=credential)
    search_client = SearchClient(
        endpoint=config.search_endpoint,
        index_name=config.index_name,
        credential=credential,
    )
    return index_client, search_client


def _real_doc_exists(search_client, key: str) -> bool:
    from azure.core.exceptions import ResourceNotFoundError

    try:
        search_client.get_document(key=key)
        return True
    except ResourceNotFoundError:
        return False


def _resource_not_found_error():
    from azure.core.exceptions import ResourceNotFoundError

    return ResourceNotFoundError


def _wait_for_document_count(search_client, minimum: int, timeout_seconds: int = 30) -> int:
    deadline = time.monotonic() + timeout_seconds
    count = search_client.get_document_count()
    while count < minimum and time.monotonic() < deadline:
        time.sleep(2)
        count = search_client.get_document_count()
    return count


def _chunk_id(source_doc: str, chunk_index: int) -> str:
    return f"{Path(source_doc).stem}__{chunk_index:03d}"


def _merge_doc_stats(stats: IngestStats, doc_stats: DocumentIngestStats) -> None:
    stats.per_document.append(doc_stats)
    stats.chunks_created += doc_stats.chunks_created
    stats.uploaded += doc_stats.uploaded
    stats.skipped += doc_stats.skipped


def _emit(progress: ProgressCallback | None, event: str, payload: dict) -> None:
    if progress is not None:
        progress(event, payload)


def main() -> None:
    stats = ingest()
    print(f"Documents found: {stats.documents_found}")
    print(f"Chunks created: {stats.chunks_created}")
    print(f"Uploaded: {stats.uploaded}")
    print(f"Skipped: {stats.skipped}")
    print(f"Index ready: {stats.total_index_chunks} chunks")


if __name__ == "__main__":
    main()
