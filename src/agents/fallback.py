"""Agent 3 - Fallback / invalid handler."""
from __future__ import annotations

from src.agents.llm_utils import create_chat_completion
from src.config import config

_SYSTEM = (
    "You are the fallback handler for a clinical records assistant. The user's "
    "message is out of scope, unclear, unsafe, or not a valid clinical-record "
    "question. Reply in one short sentence. Say that you can answer questions "
    "about authorized patient records or assigned patients, and invite the user "
    "to ask a record-grounded clinical question. Do not answer the original "
    "off-topic request."
)


class AzureFallback:
    """Real Azure OpenAI implementation of the invalid-query fallback."""

    def __init__(self) -> None:
        config.validate_production()
        from openai import AzureOpenAI

        self._client = AzureOpenAI(
            azure_endpoint=config.aoai_endpoint,
            api_key=config.aoai_key,
            api_version=config.aoai_chat_api_version,
        )

    def handle(self, query: str) -> str:
        response = create_chat_completion(
            self._client,
            model=config.fallback_deployment,
            messages=[
                {"role": "system", "content": _SYSTEM},
                {"role": "user", "content": query},
            ],
            max_tokens=120,
            temperature=0.3,
        )
        return (response.choices[0].message.content or "").strip()


def handle(query: str) -> str:
    from src.factory import get_fallback_agent

    return get_fallback_agent().handle(query)
