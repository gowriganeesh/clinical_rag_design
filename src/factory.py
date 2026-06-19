"""Central constructors for production Azure-backed services."""
from __future__ import annotations

from functools import lru_cache

from src.config import config


class AzureEmbeddings:
    """Azure OpenAI embedding client used by ingestion and semantic cache."""

    def __init__(self) -> None:
        config.validate_production()
        from openai import AzureOpenAI

        self._client = AzureOpenAI(
            azure_endpoint=config.aoai_endpoint,
            api_key=config.aoai_key,
            api_version=config.aoai_api_version,
        )

    def embed(self, text: str) -> list[float]:
        response = self._client.embeddings.create(
            model=config.embed_deployment,
            input=text,
            dimensions=config.embed_dimensions,
        )
        return response.data[0].embedding


@lru_cache(maxsize=1)
def get_embeddings() -> AzureEmbeddings:
    return AzureEmbeddings()


@lru_cache(maxsize=1)
def get_search():
    from src.tools.search_tool import AzureSearchTool

    return AzureSearchTool()


@lru_cache(maxsize=1)
def get_router():
    from src.agents.router import AzureRouter

    return AzureRouter()


@lru_cache(maxsize=1)
def get_retriever_agent():
    from src.agents.retriever import AzureRetrieverAgent

    return AzureRetrieverAgent()


@lru_cache(maxsize=1)
def get_fallback_agent():
    from src.agents.fallback import AzureFallback

    return AzureFallback()


def reset_factory_cache() -> None:
    """Clear cached clients, mostly useful for tests and Streamlit reruns."""

    get_embeddings.cache_clear()
    get_search.cache_clear()
    get_router.cache_clear()
    get_retriever_agent.cache_clear()
    get_fallback_agent.cache_clear()
