import asyncio
import logging
import time
from typing import Any

import anthropic
from anthropic import AsyncAnthropic
import httpx

from app.config import settings

logger = logging.getLogger(__name__)

MODEL_NAME = "claude-haiku-4-5"
DEFAULT_MAX_TOKENS = 4096


def _normalize_error(exc: Exception) -> str:
    """Normalize provider-specific exceptions into a human-readable description."""
    if isinstance(exc, anthropic.RateLimitError):
        msg = getattr(exc, "message", None) or str(exc)
        return f"Rate limit exceeded (HTTP 429): {msg}"

    if isinstance(exc, anthropic.AuthenticationError):
        msg = getattr(exc, "message", None) or str(exc)
        return f"Authentication error: {msg}"

    if isinstance(exc, anthropic.APITimeoutError):
        msg = getattr(exc, "message", None) or str(exc)
        return f"Request timed out: {msg}"

    if isinstance(exc, anthropic.APIConnectionError):
        msg = getattr(exc, "message", None) or str(exc)
        return f"Network error: {msg}"

    if isinstance(exc, anthropic.APIStatusError):
        status_code = getattr(exc, "status_code", None)
        msg = getattr(exc, "message", None) or str(exc)
        if status_code in (408, 504):
            return f"Request timed out: {msg}"
        if status_code:
            return f"Anthropic API error ({status_code}): {msg}"
        return f"Anthropic API error: {msg}"

    if isinstance(exc, (anthropic.APIError, anthropic.AnthropicError)):
        msg = getattr(exc, "message", None) or str(exc)
        return f"Anthropic API error: {msg}"

    if isinstance(exc, (httpx.TimeoutException, asyncio.TimeoutError, TimeoutError)):
        return f"Request timed out: {exc}"

    if isinstance(exc, (httpx.RequestError, ConnectionError, OSError)):
        return f"Network error: {exc}"

    if isinstance(exc, ValueError):
        msg = str(exc)
        if any(term in msg.lower() for term in ("api key", "api_key", "credential", "auth")):
            return f"Authentication error: {msg}"
        return f"Configuration error: {msg}"

    return f"Unexpected error: {exc}"


def get_claude_client(
    api_key: str | None = None,
    workspace_id: str | None = None,
) -> AsyncAnthropic:
    """Initialize and return an AsyncAnthropic client instance.

    Args:
        api_key: Optional Anthropic API key. Defaults to settings.anthropic_api_key.
        workspace_id: Optional Anthropic workspace ID. Defaults to settings.anthropic_workspace_id.

    Returns:
        A configured AsyncAnthropic instance.
    """
    key = api_key if api_key is not None else settings.anthropic_api_key
    if not key or not str(key).strip():
        raise ValueError("Anthropic API key is missing or empty.")

    ws_id = (
        workspace_id
        if workspace_id is not None
        else settings.anthropic_workspace_id
    )
    default_headers: dict[str, str] = {}
    if ws_id and str(ws_id).strip():
        default_headers["anthropic-workspace-id"] = str(ws_id).strip()

    return AsyncAnthropic(
        api_key=key,
        default_headers=default_headers if default_headers else None,
    )


async def call_claude(
    prompt: str,
    system_prompt: str,
    timeout: int = 12,
    client: AsyncAnthropic | None = None,
) -> dict[str, Any]:
    """Execute an asynchronous prompt call to the Anthropic Claude model.

    Args:
        prompt: The user or task prompt content.
        system_prompt: System instruction setting persona/context.
        timeout: SDK request timeout in seconds (defaults to 12).
        client: Optional pre-configured AsyncAnthropic for testing or reuse.

    Returns:
        A normalized dictionary conforming to the ProviderResult schema:
        - On success: status, model, duration_seconds, tokens, response_text
        - On failure: status, model, duration_seconds, tokens, response_text, error_message
    """
    start_time = time.perf_counter()
    try:
        active_client = client if client is not None else get_claude_client()

        create_kwargs: dict[str, Any] = {
            "model": MODEL_NAME,
            "max_tokens": DEFAULT_MAX_TOKENS,
            "messages": [{"role": "user", "content": prompt}],
            "timeout": float(timeout),
        }
        if system_prompt:
            create_kwargs["system"] = system_prompt

        response = await active_client.messages.create(**create_kwargs)

        duration_seconds = round(time.perf_counter() - start_time, 4)

        input_tokens = 0
        output_tokens = 0
        if getattr(response, "usage", None):
            input_tokens = getattr(response.usage, "input_tokens", 0) or 0
            output_tokens = getattr(response.usage, "output_tokens", 0) or 0

        response_text = ""
        if getattr(response, "content", None) and len(response.content) > 0:
            first_block = response.content[0]
            raw_text = getattr(first_block, "text", None)
            if raw_text is not None:
                response_text = raw_text

        return {
            "status": "success",
            "model": MODEL_NAME,
            "duration_seconds": duration_seconds,
            "tokens": {
                "input": input_tokens,
                "output": output_tokens,
            },
            "response_text": response_text,
        }
    except Exception as exc:
        duration_seconds = round(time.perf_counter() - start_time, 4)
        error_message = _normalize_error(exc)
        logger.error("Claude provider call failed: %s", error_message)
        return {
            "status": "error",
            "model": MODEL_NAME,
            "duration_seconds": duration_seconds,
            "tokens": {"input": 0, "output": 0},
            "response_text": None,
            "error_message": error_message,
        }


__all__ = ["DEFAULT_MAX_TOKENS", "MODEL_NAME", "call_claude", "get_claude_client"]
