"""Small OpenAI chat helper for deployment differences."""
from __future__ import annotations

from src.config import config


def create_chat_completion(client, *, model: str, messages: list[dict], max_tokens: int, temperature=None, **kwargs):
    """Create a chat completion while handling token-parameter differences.

    Some reasoning deployments reject temperature and require
    max_completion_tokens. In auto mode, try the common shape first, then retry
    with max_completion_tokens and without temperature if the service rejects the
    request shape.
    """

    token_param = config.chat_token_param.strip().lower()
    if token_param not in {"auto", "max_tokens", "max_completion_tokens"}:
        token_param = "auto"

    primary_name = "max_tokens" if token_param in {"auto", "max_tokens"} else "max_completion_tokens"
    params = {
        "model": model,
        "messages": messages,
        primary_name: max_tokens,
        **kwargs,
    }
    if temperature is not None:
        params["temperature"] = temperature

    try:
        return client.chat.completions.create(**params)
    except Exception as exc:
        if token_param != "auto" or not _looks_like_shape_error(exc):
            raise
        retry = {
            "model": model,
            "messages": messages,
            "max_completion_tokens": max_tokens,
            **kwargs,
        }
        return client.chat.completions.create(**retry)


def _looks_like_shape_error(exc: Exception) -> bool:
    text = str(exc).lower()
    return any(
        marker in text
        for marker in [
            "max_tokens",
            "max_completion_tokens",
            "temperature",
            "unsupported",
            "not support",
            "unrecognized",
        ]
    )
