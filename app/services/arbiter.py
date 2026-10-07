"""Arbiter synthesis service for generating authoritative consensus responses.

Reconciles surviving model responses from worker LLMs into a single consensus
answer using a configured high-capability model (Claude Sonnet 5.5, Gemini, or
GPT-4o). When Claude is the arbiter, GPT-4o is used as the fallback.
"""

import logging
from typing import Any

from anthropic import AsyncAnthropic
from google import genai
from google.genai import types
from openai import AsyncOpenAI

from app.config import settings
from app.core.prompts import build_arbiter_prompt
from app.schemas.models import Context
from app.services.claude_client import get_claude_client
from app.services.gemini_client import get_gemini_client
from app.services.openai_client import get_openai_client

logger = logging.getLogger(__name__)

ARBITER_MODELS: dict[str, str] = {
    "claude": "claude-sonnet-5-5",
    "gemini": "gemini-3-flash-preview",
    "openai": "gpt-4o",
}

# Provider to retry with when the primary arbiter call fails.
ARBITER_FALLBACKS: dict[str, str] = {
    "claude": "openai",
}

CLAUDE_ARBITER_MAX_TOKENS = 16000


class ArbiterError(Exception):
    """Raised when arbiter synthesis fails."""


class AllProvidersFailedError(ArbiterError):
    """Raised when synthesize is called with zero surviving provider results."""


def build_arbiter_user_message(
    original_prompt: str,
    successful_results: list[dict[str, Any]],
) -> str:
    """Build the user message payload sent to the Arbiter model.

    For multiple surviving responses (2 or 3), formats the original query followed
    by anonymous numbered model responses to reconcile into consensus.
    For a single surviving response (degraded path), formats the original query
    and the single response for persona validation and formatting.

    Args:
        original_prompt: The user's original query.
        successful_results: List of successful worker result dicts containing 'response_text'.

    Returns:
        Formatted user prompt string for the Arbiter.

    Raises:
        AllProvidersFailedError: If successful_results is empty or invalid.
    """
    if not isinstance(successful_results, list) or len(successful_results) == 0:
        raise AllProvidersFailedError(
            "Cannot build arbiter user message: no successful provider results."
        )

    if len(successful_results) == 1:
        result = successful_results[0]
        response_text = ""
        if isinstance(result, dict):
            response_text = result.get("response_text") or ""
        blocks = [
            f"Original Query: {original_prompt.strip()}",
            "One model response was received (other providers were unavailable):",
            f"Model Response:\n{response_text.strip()}",
            "Please validate, complete, and format this response through the lens of the active professional persona.",
        ]
        return "\n\n".join(blocks)

    blocks = [f"Original Query: {original_prompt.strip()}"]
    for idx, result in enumerate(successful_results, start=1):
        response_text = ""
        if isinstance(result, dict):
            response_text = result.get("response_text") or ""
        blocks.append(f"Model {idx} Response:\n{response_text.strip()}")

    blocks.append("Please synthesize these responses into a single authoritative answer.")
    return "\n\n".join(blocks)


async def _call_claude_arbiter(
    user_message: str,
    system_prompt: str,
    timeout: int | float,
    client: AsyncAnthropic | None = None,
) -> str:
    """Execute arbiter synthesis call against Claude Sonnet 5.5."""
    model = ARBITER_MODELS["claude"]
    try:
        active_client = client if client is not None else get_claude_client()
        create_kwargs: dict[str, Any] = {
            "model": model,
            "max_tokens": CLAUDE_ARBITER_MAX_TOKENS,
            "output_config": {"effort": "medium"},
            "messages": [{"role": "user", "content": user_message}],
            "timeout": float(timeout),
        }
        if system_prompt:
            create_kwargs["system"] = system_prompt

        response = await active_client.messages.create(**create_kwargs)

        if getattr(response, "stop_reason", None) == "refusal":
            raise ArbiterError("Claude arbiter declined to synthesize a response.")

        # Thinking blocks precede the answer, so collect only the text blocks.
        return "".join(
            block.text
            for block in (getattr(response, "content", None) or [])
            if getattr(block, "type", None) == "text"
        )
    except ArbiterError:
        raise
    except Exception as exc:
        logger.error("Claude arbiter synthesis call failed: %s", exc)
        raise ArbiterError(f"Claude arbiter synthesis failed: {exc}") from exc


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


_ARBITER_CALLERS: dict[str, Any] = {
    "claude": _call_claude_arbiter,
    "gemini": _call_gemini_arbiter,
    "openai": _call_openai_arbiter,
}


async def synthesize(
    original_prompt: str,
    context: Context,
    successful_results: list[dict[str, Any]],
    *,
    client: Any | None = None,
    provider: str | None = None,
    timeout: int | float | None = None,
) -> str:
    """Synthesize AI provider responses into a single authoritative consensus answer.

    Reconciles 2 or 3 surviving provider responses into an authoritative consensus,
    or validates/formats a single surviving response through the active insurance
    persona in degraded scenarios. Short-circuits with AllProvidersFailedError if
    zero providers succeeded.

    Args:
        original_prompt: The initial insurance query submitted by the user.
        context: Context object specifying role, line of business, and jurisdiction.
        successful_results: List of successful worker result dictionaries (1 to 3 items).
        client: Optional pre-configured client for the primary provider, for testing or reuse
            (AsyncAnthropic, genai.Client, or AsyncOpenAI).
        provider: Optional provider override ('claude', 'gemini', or 'openai'). Defaults to
            settings.arbiter_model_provider. A failed 'claude' call falls back to 'openai'.
        timeout: Optional request timeout in seconds. Defaults to settings.request_timeout_seconds.

    Returns:
        The raw synthesized consensus answer string.

    Raises:
        AllProvidersFailedError: If zero successful provider results are provided.
        ValueError: If the requested provider is unsupported.
        ArbiterError: If the downstream arbiter LLM call (and its fallback, if any) fails.
    """
    if not isinstance(successful_results, list) or len(successful_results) == 0:
        raise AllProvidersFailedError(
            "All AI providers failed. No consensus could be generated."
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
        timeout
        if timeout is not None
        else max(30.0, float(settings.request_timeout_seconds * 2))
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

    try:
        return await _ARBITER_CALLERS[resolved_provider](
            user_message=arbiter_user_message,
            system_prompt=arbiter_system_prompt,
            timeout=effective_timeout,
            client=client,
        )
    except ArbiterError as exc:
        fallback_provider = ARBITER_FALLBACKS.get(resolved_provider)
        if fallback_provider is None:
            raise
        logger.warning(
            "Arbiter %s (%s) failed: %s. Falling back to %s (%s).",
            resolved_provider,
            ARBITER_MODELS[resolved_provider],
            exc,
            fallback_provider,
            ARBITER_MODELS[fallback_provider],
        )
        # The injected client belongs to the primary provider; the fallback builds its own.
        return await _ARBITER_CALLERS[fallback_provider](
            user_message=arbiter_user_message,
            system_prompt=arbiter_system_prompt,
            timeout=effective_timeout,
        )


async def synthesize_single(
    original_prompt: str,
    context: Context,
    single_result: dict[str, Any],
    *,
    client: Any | None = None,
    provider: str | None = None,
    timeout: int | float | None = None,
) -> str:
    """Synthesize and format a single surviving provider response through the active persona.

    Convenience wrapper around synthesize() for single-survivor degraded scenarios.

    Args:
        original_prompt: The initial insurance query submitted by the user.
        context: Context object specifying role, line of business, and jurisdiction.
        single_result: Successful worker result dictionary for the surviving provider.
        client: Optional pre-configured client for testing or reuse.
        provider: Optional provider override ('claude', 'gemini', or 'openai').
        timeout: Optional request timeout in seconds.

    Returns:
        The refined consensus answer string.
    """
    return await synthesize(
        original_prompt=original_prompt,
        context=context,
        successful_results=[single_result],
        client=client,
        provider=provider,
        timeout=timeout,
    )


__all__ = [
    "ARBITER_FALLBACKS",
    "ARBITER_MODELS",
    "AllProvidersFailedError",
    "ArbiterError",
    "build_arbiter_user_message",
    "synthesize",
    "synthesize_single",
]
