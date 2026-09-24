import asyncio
import logging
from typing import Any

from app.config import settings
from app.services.claude_client import call_claude
from app.services.gemini_client import call_gemini
from app.services.openai_client import call_openai

logger = logging.getLogger(__name__)


async def run_workers(
    prompt: str,
    system_prompt: str,
    timeout: int | None = None,
    *,
    gemini_client: Any | None = None,
    openai_client: Any | None = None,
    claude_client: Any | None = None,
) -> list[dict[str, Any] | Any]:
    """Execute asynchronous prompt calls to all three provider models concurrently.

    Fires Gemini, OpenAI, and Claude concurrently via asyncio.gather and collects
    their results. The total worker-phase duration approximates the latency of the
    slowest individual provider call rather than their sum.

    Args:
        prompt: The user or task prompt content.
        system_prompt: System instruction setting persona/context.
        timeout: Optional request timeout in seconds. Defaults to
            settings.request_timeout_seconds if not specified.
        gemini_client: Optional pre-configured Gemini client for testing or reuse.
        openai_client: Optional pre-configured AsyncOpenAI client for testing or reuse.
        claude_client: Optional pre-configured AsyncAnthropic client for testing or reuse.

    Returns:
        A list of exactly 3 items in deterministic order:
        [gemini_result, openai_result, claude_result].
        Each result conforms to the ProviderResult schema shape
        (status, model, duration_seconds, tokens, response_text, [error_message]),
        or an Exception instance if an unhandled exception occurred within the task.
    """
    effective_timeout = (
        timeout if timeout is not None else settings.request_timeout_seconds
    )

    gemini_kwargs: dict[str, Any] = {"client": gemini_client} if gemini_client is not None else {}
    openai_kwargs: dict[str, Any] = {"client": openai_client} if openai_client is not None else {}
    claude_kwargs: dict[str, Any] = {"client": claude_client} if claude_client is not None else {}

    results = await asyncio.gather(
        call_gemini(prompt, system_prompt, effective_timeout, **gemini_kwargs),
        call_openai(prompt, system_prompt, effective_timeout, **openai_kwargs),
        call_claude(prompt, system_prompt, effective_timeout, **claude_kwargs),
        return_exceptions=True,
    )

    return list(results)


__all__ = ["run_workers"]
