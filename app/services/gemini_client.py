import asyncio
import logging
import time
from typing import Any

from google import genai
from google.genai import errors, types
import httpx

from app.config import settings

logger = logging.getLogger(__name__)

MODEL_NAME = "gemini-3-flash-preview"


def _normalize_error(exc: Exception) -> str:
    """Normalize provider-specific exceptions into a human-readable description."""
    if isinstance(exc, errors.APIError):
        code = getattr(exc, "code", None)
        message = getattr(exc, "message", None) or str(exc)
        status = getattr(exc, "status", None)

        if code == 429 or status == "RESOURCE_EXHAUSTED" or "RESOURCE_EXHAUSTED" in str(exc):
            return f"Rate limit exceeded (HTTP 429): {message}"
        if code in (400, 401, 403) and any(
            term in str(exc).upper()
            for term in (
                "API_KEY",
                "API KEY",
                "AUTHENTICATION",
                "UNAUTHENTICATED",
                "PERMISSION_DENIED",
                "CREDENTIAL",
                "INVALID_ARGUMENT",
            )
        ):
            return f"Authentication error: {message}"
        if code in (408, 504):
            return f"Request timed out: {message}"
        if code and 500 <= code < 600:
            return f"Gemini server error ({code}): {message}"
        if code:
            return f"Gemini API error ({code}): {message}"
        return f"Gemini API error: {message}"

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


def get_gemini_client(api_key: str | None = None) -> genai.Client:
    """Initialize and return a Google GenAI Client instance.

    Args:
        api_key: Optional Gemini API key. Defaults to settings.gemini_api_key.

    Returns:
        A configured genai.Client instance.
    """
    key = api_key if api_key is not None else settings.gemini_api_key
    if not key or not str(key).strip():
        raise ValueError("Google Gemini API key is missing or empty.")
    return genai.Client(api_key=key)


async def call_gemini(
    prompt: str,
    system_prompt: str,
    timeout: int = 12,
    client: genai.Client | None = None,
) -> dict[str, Any]:
    """Execute an asynchronous prompt call to the Google Gemini model.

    Args:
        prompt: The user or task prompt content.
        system_prompt: System instruction setting persona/context.
        timeout: SDK request timeout in seconds (defaults to 12).
        client: Optional pre-configured genai.Client for testing or reuse.

    Returns:
        A normalized dictionary conforming to the ProviderResult schema:
        - On success: status, model, duration_seconds, tokens, response_text
        - On failure: status, model, duration_seconds, tokens, response_text, error_message
    """
    start_time = time.perf_counter()
    try:
        active_client = client if client is not None else get_gemini_client()

        config = types.GenerateContentConfig(
            system_instruction=system_prompt if system_prompt else None,
            http_options=types.HttpOptions(timeout=int(timeout * 1000)),
            automatic_function_calling=types.AutomaticFunctionCallingConfig(disable=True),
        )

        response = await active_client.aio.models.generate_content(
            model=MODEL_NAME,
            contents=prompt,
            config=config,
        )

        duration_seconds = round(time.perf_counter() - start_time, 4)

        input_tokens = 0
        output_tokens = 0
        if getattr(response, "usage_metadata", None):
            input_tokens = response.usage_metadata.prompt_token_count or 0
            output_tokens = response.usage_metadata.candidates_token_count or 0

        raw_text = getattr(response, "text", None)
        response_text = raw_text if raw_text is not None else ""

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
        logger.error("Gemini provider call failed: %s", error_message)
        return {
            "status": "error",
            "model": MODEL_NAME,
            "duration_seconds": duration_seconds,
            "tokens": {"input": 0, "output": 0},
            "response_text": None,
            "error_message": error_message,
        }


__all__ = ["MODEL_NAME", "call_gemini", "get_gemini_client"]
