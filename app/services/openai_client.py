import asyncio
import logging
import time
from typing import Any

import httpx
import openai
from openai import AsyncOpenAI

from app.config import settings

logger = logging.getLogger(__name__)

MODEL_NAME = "gpt-4o-mini"


def _normalize_error(exc: Exception) -> str:
    """Normalize provider-specific exceptions into a human-readable description."""
    if isinstance(exc, openai.RateLimitError):
        msg = getattr(exc, "message", None) or str(exc)
        return f"Rate limit exceeded (HTTP 429): {msg}"

    if isinstance(exc, openai.AuthenticationError):
        msg = getattr(exc, "message", None) or str(exc)
        return f"Authentication error: {msg}"

    if isinstance(exc, openai.APITimeoutError):
        msg = getattr(exc, "message", None) or str(exc)
        return f"Request timed out: {msg}"

    if isinstance(exc, openai.APIConnectionError):
        msg = getattr(exc, "message", None) or str(exc)
        return f"Network error: {msg}"

    if isinstance(exc, openai.APIStatusError):
        status_code = getattr(exc, "status_code", None)
        msg = getattr(exc, "message", None) or str(exc)
        if status_code in (408, 504):
            return f"Request timed out: {msg}"
        if status_code:
            return f"OpenAI API error ({status_code}): {msg}"
        return f"OpenAI API error: {msg}"

    if isinstance(exc, openai.OpenAIError):
        msg = getattr(exc, "message", None) or str(exc)
        return f"OpenAI API error: {msg}"

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


def get_openai_client(api_key: str | None = None) -> AsyncOpenAI:
    """Initialize and return an AsyncOpenAI client instance.

    Args:
        api_key: Optional OpenAI API key. Defaults to settings.openai_api_key.

    Returns:
        A configured AsyncOpenAI instance.
    """
    key = api_key if api_key is not None else settings.openai_api_key
    if not key or not str(key).strip():
        raise ValueError("OpenAI API key is missing or empty.")
    return AsyncOpenAI(api_key=key)


async def call_openai(
    prompt: str,
    system_prompt: str,
    timeout: int = 12,
    client: AsyncOpenAI | None = None,
) -> dict[str, Any]:
    """Execute an asynchronous prompt call to the OpenAI model.

    Args:
        prompt: The user or task prompt content.
        system_prompt: System instruction setting persona/context.
        timeout: SDK request timeout in seconds (defaults to 12).
        client: Optional pre-configured AsyncOpenAI for testing or reuse.

    Returns:
        A normalized dictionary conforming to the ProviderResult schema:
        - On success: status, model, duration_seconds, tokens, response_text
        - On failure: status, model, duration_seconds, tokens, response_text, error_message
    """
    start_time = time.perf_counter()
    try:
        active_client = client if client is not None else get_openai_client()

        messages: list[dict[str, str]] = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": prompt})

        response = await active_client.chat.completions.create(
            model=MODEL_NAME,
            messages=messages,
            timeout=float(timeout),
        )

        duration_seconds = round(time.perf_counter() - start_time, 4)

        input_tokens = 0
        output_tokens = 0
        if getattr(response, "usage", None):
            input_tokens = getattr(response.usage, "prompt_tokens", 0) or 0
            output_tokens = getattr(response.usage, "completion_tokens", 0) or 0

        response_text = ""
        if getattr(response, "choices", None) and len(response.choices) > 0:
            raw_text = getattr(response.choices[0].message, "content", None)
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
        logger.error("OpenAI provider call failed: %s", error_message)
        return {
            "status": "error",
            "model": MODEL_NAME,
            "duration_seconds": duration_seconds,
            "tokens": {"input": 0, "output": 0},
            "response_text": None,
            "error_message": error_message,
        }


__all__ = ["MODEL_NAME", "call_openai", "get_openai_client"]
