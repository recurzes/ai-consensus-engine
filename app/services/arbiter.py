"""Arbiter synthesis service for generating authoritative consensus responses.

Reconciles surviving model responses from worker LLMs into a single consensus
answer using a configured high-capability model (Gemini 2.5 Pro or GPT-4o).
"""

import logging
from typing import Any

from google import genai
from google.genai import types
from openai import AsyncOpenAI

from app.config import settings
from app.core.prompts import build_arbiter_prompt
from app.schemas.models import Context
from app.services.gemini_client import get_gemini_client
from app.services.openai_client import get_openai_client

logger = logging.getLogger(__name__)

ARBITER_MODELS: dict[str, str] = {
    "gemini": "gemini-2.5-pro",
    "openai": "gpt-4o",
}


class ArbiterError(Exception):
    """Raised when arbiter synthesis fails."""


def build_arbiter_user_message(
    original_prompt: str,
    successful_results: list[dict[str, Any]],
) -> str:
    """Build the user message payload sent to the Arbiter model.

    Formats the original prompt followed by each surviving model's response
    labeled anonymously (e.g. 'Model 1 Response:', 'Model 2 Response:') to avoid
    biasing the synthesis toward any specific provider.

    Args:
        original_prompt: The user's original query.
        successful_results: List of successful worker result dicts containing 'response_text'.

    Returns:
        Formatted user prompt string for the Arbiter.
    """
    blocks = [f"Original Query: {original_prompt.strip()}"]
    for idx, result in enumerate(successful_results, start=1):
        response_text = ""
        if isinstance(result, dict):
            response_text = result.get("response_text") or ""
        blocks.append(f"Model {idx} Response:\n{response_text.strip()}")

    blocks.append("Please synthesize these responses into a single authoritative answer.")
    return "\n\n".join(blocks)


async def _call_gemini_arbiter(
    user_message: str,
    system_prompt: str,
    timeout: int | float,
    client: genai.Client | None = None,
) -> str:
    """Execute arbiter synthesis call against Gemini 2.5 Pro."""
    active_client = client if client is not None else get_gemini_client()
    model = ARBITER_MODELS["gemini"]
    config = types.GenerateContentConfig(
        system_instruction=system_prompt if system_prompt else None,
        http_options=types.HttpOptions(timeout=int(timeout * 1000)),
        automatic_function_calling=types.AutomaticFunctionCallingConfig(disable=True),
    )
    try:
        response = await active_client.aio.models.generate_content(
            model=model,
            contents=user_message,
            config=config,
        )
        text = getattr(response, "text", None)
        return text if text is not None else ""
    except Exception as exc:
        logger.error("Gemini arbiter synthesis call failed: %s", exc)
        raise ArbiterError(f"Gemini arbiter synthesis failed: {exc}") from exc


async def _call_openai_arbiter(
    user_message: str,
    system_prompt: str,
    timeout: int | float,
    client: AsyncOpenAI | None = None,
) -> str:
    """Execute arbiter synthesis call against OpenAI GPT-4o."""
    active_client = client if client is not None else get_openai_client()
    model = ARBITER_MODELS["openai"]
    messages: list[dict[str, str]] = []
    if system_prompt:
        messages.append({"role": "system", "content": system_prompt})
    messages.append({"role": "user", "content": user_message})

    try:
        response = await active_client.chat.completions.create(
            model=model,
            messages=messages,
            timeout=float(timeout),
        )
        if response.choices and len(response.choices) > 0:
            content = getattr(response.choices[0].message, "content", None)
            return content if content is not None else ""
        return ""
    except Exception as exc:
        logger.error("OpenAI arbiter synthesis call failed: %s", exc)
        raise ArbiterError(f"OpenAI arbiter synthesis failed: {exc}") from exc


async def synthesize(
    original_prompt: str,
    context: Context,
    successful_results: list[dict[str, Any]],
    *,
    client: Any | None = None,
    provider: str | None = None,
    timeout: int | float | None = None,
) -> str:
    """Synthesize multiple AI provider responses into a single authoritative consensus answer.

    Happy-path arbiter synthesis reconciling 2 or 3 surviving provider responses.
    Applies the active insurance persona via build_arbiter_prompt and dispatches
    to the configured arbiter model provider (Gemini 2.5 Pro or GPT-4o).

    Args:
        original_prompt: The initial insurance query submitted by the user.
        context: Context object specifying role, line of business, and jurisdiction.
        successful_results: List of successful worker result dictionaries (2 or 3 items).
        client: Optional pre-configured client for testing or reuse (genai.Client or AsyncOpenAI).
        provider: Optional provider override ('gemini' or 'openai'). Defaults to settings.arbiter_model_provider.
        timeout: Optional request timeout in seconds. Defaults to settings.request_timeout_seconds.

    Returns:
        The raw synthesized consensus answer string.

    Raises:
        ValueError: If fewer than 2 successful results are provided, or if the provider is unsupported.
        ArbiterError: If the downstream arbiter LLM call fails.
    """
    if not isinstance(successful_results, list) or len(successful_results) < 2:
        count = len(successful_results) if isinstance(successful_results, list) else 0
        raise ValueError(
            f"Arbiter happy path requires at least 2 successful provider results, got {count}."
        )

    resolved_provider = (
        provider
        if provider is not None
        else settings.arbiter_model_provider
    )
    if isinstance(resolved_provider, str):
        resolved_provider = resolved_provider.strip().lower()

    if resolved_provider not in ARBITER_MODELS:
        valid_providers = ", ".join(repr(k) for k in ARBITER_MODELS.keys())
        raise ValueError(
            f"Unsupported arbiter model provider: '{resolved_provider}'. Supported providers are: {valid_providers}."
        )

    effective_timeout = (
        timeout if timeout is not None else settings.request_timeout_seconds
    )

    role = (
        getattr(context, "role", None)
        or (context.get("role") if isinstance(context, dict) else None)
    )
    line_of_business = (
        getattr(context, "line_of_business", None)
        or (context.get("line_of_business") if isinstance(context, dict) else None)
    )
    state = (
        getattr(context, "state", None)
        or (context.get("state") if isinstance(context, dict) else "MT")
    )

    arbiter_system_prompt = build_arbiter_prompt(
        role=role,
        line_of_business=line_of_business,
        state=state,
    )

    arbiter_user_message = build_arbiter_user_message(
        original_prompt=original_prompt,
        successful_results=successful_results,
    )

    if resolved_provider == "gemini":
        return await _call_gemini_arbiter(
            user_message=arbiter_user_message,
            system_prompt=arbiter_system_prompt,
            timeout=effective_timeout,
            client=client,
        )
    else:  # "openai"
        return await _call_openai_arbiter(
            user_message=arbiter_user_message,
            system_prompt=arbiter_system_prompt,
            timeout=effective_timeout,
            client=client,
        )


__all__ = [
    "ARBITER_MODELS",
    "ArbiterError",
    "build_arbiter_user_message",
    "synthesize",
]
