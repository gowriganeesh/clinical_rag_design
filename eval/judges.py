"""Azure OpenAI judge configuration for RAGAS and DeepEval."""
from __future__ import annotations

from src.config import config

JUDGE_DEPLOYMENT = config.answer_deployment


def ragas_judge():
    """Return (llm, embeddings) wrapped for RAGAS, backed by Azure OpenAI."""

    from langchain_openai import AzureChatOpenAI, AzureOpenAIEmbeddings
    from ragas.embeddings import LangchainEmbeddingsWrapper
    from ragas.llms import LangchainLLMWrapper

    llm = LangchainLLMWrapper(
        AzureChatOpenAI(
            azure_endpoint=config.aoai_endpoint,
            api_key=config.aoai_key,
            api_version=config.aoai_chat_api_version,
            azure_deployment=JUDGE_DEPLOYMENT,
            temperature=0,
        )
    )
    embeddings = LangchainEmbeddingsWrapper(
        AzureOpenAIEmbeddings(
            azure_endpoint=config.aoai_endpoint,
            api_key=config.aoai_key,
            api_version=config.aoai_api_version,
            azure_deployment=config.embed_deployment,
        )
    )
    return llm, embeddings


def deepeval_judge():
    """Return an Azure OpenAI judge model for DeepEval metrics."""

    from deepeval.models import AzureOpenAIModel

    return AzureOpenAIModel(
        model=JUDGE_DEPLOYMENT,
        deployment_name=JUDGE_DEPLOYMENT,
        api_key=config.aoai_key,
        api_version=config.aoai_chat_api_version,
        base_url=config.aoai_endpoint,
        temperature=0,
    )
