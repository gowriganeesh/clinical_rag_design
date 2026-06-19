"""Agent 2 - Retrieval + answer with server-side tool execution."""
from __future__ import annotations

import json

from src.agents.llm_utils import create_chat_completion
from src.config import config

ABSTAIN = "I don't have enough information in the records to answer that."

_TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "search_clinical_records",
            "description": (
                "Search authorized clinical records for evidence relevant to the "
                "question. The server injects the acting user_id and applies ACL "
                "filtering; the model must only provide the search query."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {"type": "string", "description": "The search query."}
                },
                "required": ["query"],
            },
        },
    }
]

_SYSTEM = (
    "You are a clinical records assistant for de-identified patient histories. "
    "Use the search_clinical_records tool before answering. Answer only from the "
    "tool results; never use outside medical knowledge or infer missing facts. "
    "Be direct and concise: answer the user's question first, then include the "
    "supporting record citation. Cite the source_doc for every clinical fact in "
    "parentheses, for example (source_doc). If multiple facts come "
    "from different records, use short bullets. If the tool returns no confident "
    "records, the requested patient is not authorized, or the record does not "
    f"contain the answer, reply exactly: '{ABSTAIN}' Do not provide diagnoses, "
    "treatment recommendations, or safety advice beyond what is explicitly in "
    "the returned records."
)


class AzureRetrieverAgent:
    """Real Azure OpenAI answer agent using tool-calling."""

    def __init__(self) -> None:
        config.validate_production()
        from openai import AzureOpenAI

        self._client = AzureOpenAI(
            azure_endpoint=config.aoai_endpoint,
            api_key=config.aoai_key,
            api_version=config.aoai_chat_api_version,
        )

    def answer(self, query: str, user_id: str) -> str:
        messages = [
            {"role": "system", "content": _SYSTEM},
            {"role": "user", "content": query},
        ]

        first = create_chat_completion(
            self._client,
            model=config.answer_deployment,
            messages=messages,
            tools=_TOOLS,
            tool_choice={
                "type": "function",
                "function": {"name": "search_clinical_records"},
            },
            max_tokens=config.max_output_tokens,
            temperature=0,
        )
        msg = first.choices[0].message
        if not msg.tool_calls:
            return msg.content or ABSTAIN

        messages.append(msg)

        from src.factory import get_search

        search = get_search()
        for call in msg.tool_calls:
            try:
                args = json.loads(call.function.arguments or "{}")
            except json.JSONDecodeError:
                args = {}
            result = search.retrieve(args.get("query", query), user_id=user_id)
            payload = (
                {"chunks": result["chunks"]}
                if result["confident"]
                else {"chunks": [], "note": "no confident match"}
            )
            messages.append(
                {
                    "role": "tool",
                    "tool_call_id": call.id,
                    "content": json.dumps(payload),
                }
            )

        final = create_chat_completion(
            self._client,
            model=config.answer_deployment,
            messages=messages,
            max_tokens=config.max_output_tokens,
            temperature=0,
        )
        return final.choices[0].message.content or ABSTAIN


def answer(query: str, user_id: str) -> str:
    from src.factory import get_retriever_agent

    return get_retriever_agent().answer(query, user_id=user_id)
