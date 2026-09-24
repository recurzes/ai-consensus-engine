import asyncio
import logging
from typing import Any

from app.config import settings
from app.services.claude_client import MODEL_NAME as CLAUDE_MODEL_NAME, call_claude
from app.services.gemini_client import MODEL_NAME as GEMINI_MODEL_NAME, call_gemini
from app.services.openai_client import MODEL_NAME as OPENAI_MODEL_NAME, call_openai

logger = logging.getLogger(__name__)


async def run_workers(
    prompt: str,
    system_prompt: str,
    timeout: int | float | None = None,
    *,
    gemini_client: Any | None = None,
    openai_client: Any | None = None,
    claude_client: Any | None = None,
) -> list[dict[str, Any] | Any]:
    """Execute asynchronous prompt calls to all three provider models concurrently.

    Fires Gemini, OpenAI, and Claude concurrently via asyncio.gather and collects
    their results. Each worker call is guarded by asyncio.wait_for with a per-worker timeout.
    If a worker times out, a normalized failure dict is returned and other workers continue
    unaffected.

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

    async def guarded_call(coro: Any, model_name: str) -> dict[str, Any]:
        try:
            return await asyncio.wait_for(coro, timeout=effective_timeout)
        except (asyncio.TimeoutError, TimeoutError):
            logger.warning(
                "Worker call for model '%s' timed out after %s seconds",
                model_name,
                effective_timeout,
            )
            return {
                "status": "error",
                "model": model_name,
                "duration_seconds": float(effective_timeout),
                "tokens": {"input": 0, "output": 0},
                "response_text": None,
                "error_message": f"Request timed out after {effective_timeout} seconds",
            }

    gemini_kwargs: dict[str, Any] = {"client": gemini_client} if gemini_client is not None else {}
    openai_kwargs: dict[str, Any] = {"client": openai_client} if openai_client is not None else {}
    claude_kwargs: dict[str, Any] = {"client": claude_client} if claude_client is not None else {}

    results = await asyncio.gather(
        guarded_call(
            call_gemini(prompt, system_prompt, effective_timeout, **gemini_kwargs),
            GEMINI_MODEL_NAME,
        ),
        guarded_call(
            call_openai(prompt, system_prompt, effective_timeout, **openai_kwargs),
            OPENAI_MODEL_NAME,
        ),
        guarded_call(
            call_claude(prompt, system_prompt, effective_timeout, **claude_kwargs),
            CLAUDE_MODEL_NAME,
        ),
        return_exceptions=True,
    )

    return list(results)


MODEL_TO_PROVIDER: dict[str, str] = {
    GEMINI_MODEL_NAME: "gemini",
    OPENAI_MODEL_NAME: "openai",
    CLAUDE_MODEL_NAME: "claude",
    "gemini-3-flash-preview": "gemini",
    "gemini-2.5-flash": "gemini",
    "gpt-4o-mini": "openai",
    "claude-3-5-haiku": "claude",
}

ORDERED_PROVIDERS: tuple[str, ...] = ("gemini", "openai", "claude")


def partition_results(
    raw_results: list[dict[str, Any] | Exception],
) -> tuple[list[dict[str, Any]], list[dict[str, Any] | Exception], list[str], list[str]]:
    """Partition raw worker results into successful and failed lists with provider names.

    Classifies worker results returned from run_workers into successful and failed
    categories, and extracts the corresponding provider names ("gemini", "openai", "claude").

    A result is considered successful if:
    - It is a dictionary,
    - result["status"] == "success", and
    - result["response_text"] is not None.

    A result is considered failed if:
    - It is a bare Python Exception (e.g. from return_exceptions=True),
    - result["status"] == "error", or
    - It does not meet the success criteria.

    Args:
        raw_results: List of raw worker results (dicts or Exception instances).

    Returns:
        A 4-tuple of:
        (successful_results, failed_results, successful_providers, failed_providers)
    """
    successful_results: list[dict[str, Any]] = []
    failed_results: list[dict[str, Any] | Exception] = []
    successful_providers: list[str] = []
    failed_providers: list[str] = []

    for idx, item in enumerate(raw_results):
        provider_name: str | None = None

        if isinstance(item, dict):
            model = item.get("model")
            if model in MODEL_TO_PROVIDER:
                provider_name = MODEL_TO_PROVIDER[model]
            elif isinstance(item.get("provider"), str):
                provider_name = item["provider"]
            elif isinstance(model, str):
                model_lower = model.lower()
                if "gemini" in model_lower:
                    provider_name = "gemini"
                elif "openai" in model_lower or "gpt" in model_lower:
                    provider_name = "openai"
                elif "claude" in model_lower or "anthropic" in model_lower:
                    provider_name = "claude"
        elif isinstance(item, Exception):
            provider_attr = getattr(item, "provider", None)
            if isinstance(provider_attr, str):
                provider_name = provider_attr

        # Fallback to positional mapping for deterministic 3-provider order
        if provider_name is None and 0 <= idx < len(ORDERED_PROVIDERS):
            provider_name = ORDERED_PROVIDERS[idx]

        if provider_name is None:
            provider_name = "unknown"

        if (
            isinstance(item, dict)
            and item.get("status") == "success"
            and item.get("response_text") is not None
        ):
            successful_results.append(item)
            successful_providers.append(provider_name)
        else:
            failed_results.append(item)
            failed_providers.append(provider_name)

    return successful_results, failed_results, successful_providers, failed_providers


__all__ = [
    "CLAUDE_MODEL_NAME",
    "GEMINI_MODEL_NAME",
    "OPENAI_MODEL_NAME",
    "MODEL_TO_PROVIDER",
    "ORDERED_PROVIDERS",
    "partition_results",
    "run_workers",
]

