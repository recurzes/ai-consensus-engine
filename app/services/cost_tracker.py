"""Cost tracker service for estimating per-request LLM API token costs.

Pricing data changes over time; keeping it as an isolated constant makes future
updates straightforward.
"""

from typing import Any, Final

# Pricing last verified: September 2026
# Provider pricing reference documentation:
# - Google Gemini (gemini-2.5-flash, gemini-2.5-pro): https://ai.google.dev/pricing
# - OpenAI (gpt-4o-mini, gpt-4o): https://openai.com/pricing
# - Anthropic (claude-3-5-haiku): https://www.anthropic.com/pricing

PRICING: Final[dict[str, dict[str, float]]] = {
    "gemini-2.5-flash": {
        "input_per_million": 0.30,   # USD per 1M input tokens (paid tier)
        "output_per_million": 2.50,  # USD per 1M output tokens (paid tier)
    },
    "gpt-4o-mini": {
        "input_per_million": 0.15,
        "output_per_million": 0.60,
    },
    "claude-3-5-haiku": {
        "input_per_million": 0.80,
        "output_per_million": 4.00,
    },
    "gemini-2.5-pro": {
        "input_per_million": 1.25,
        "output_per_million": 10.00,
    },
    "gpt-4o": {
        "input_per_million": 2.50,
        "output_per_million": 10.00,
    },
}


def calculate_cost(model: str, input_tokens: int, output_tokens: int) -> float:
    """Calculate the estimated USD cost for a single model call based on token usage.

    Formula:
        (input_tokens / 1_000_000 * input_rate) + (output_tokens / 1_000_000 * output_rate)

    Args:
        model: Model identifier key in the PRICING table.
        input_tokens: Number of prompt/input tokens.
        output_tokens: Number of completion/output tokens.

    Returns:
        The total USD cost as a float rounded to 6 decimal places.

    Raises:
        ValueError: If model is not found in PRICING or if token counts are negative.
    """
    if model not in PRICING:
        available_models = ", ".join(sorted(PRICING.keys()))
        raise ValueError(
            f"Unknown model '{model}'. Available models: {available_models}"
        )

    if input_tokens < 0 or output_tokens < 0:
        raise ValueError(
            f"Token counts cannot be negative (got input_tokens={input_tokens}, "
            f"output_tokens={output_tokens})"
        )

    if input_tokens == 0 and output_tokens == 0:
        return 0.0

    rates = PRICING[model]
    input_cost = (input_tokens / 1_000_000.0) * rates["input_per_million"]
    output_cost = (output_tokens / 1_000_000.0) * rates["output_per_million"]
    total_cost = input_cost + output_cost

    return round(total_cost, 6)


def calculate_all_costs(
    provider_results: list[dict[str, Any]],
) -> tuple[list[dict[str, Any]], float]:
    """Calculate and inject estimated USD costs for a collection of provider results.

    Iterates over each provider result dict, extracts token metrics, calculates the cost,
    injects 'estimated_cost_usd' into each dict, and sums the total estimated cost.

    Failed providers with zero tokens automatically contribute $0.00 to the total.

    Args:
        provider_results: List of provider result dicts conforming to ProviderResult shape.

    Returns:
        A tuple of:
        (updated_provider_results, total_estimated_cost_usd)
    """
    total_estimated_cost_usd = 0.0

    for result in provider_results:
        model = result.get("model")
        tokens = result.get("tokens") or {}
        input_tokens = tokens.get("input", 0) if isinstance(tokens, dict) else 0
        output_tokens = tokens.get("output", 0) if isinstance(tokens, dict) else 0

        if input_tokens == 0 and output_tokens == 0:
            cost = 0.0
        else:
            if not isinstance(model, str) or model not in PRICING:
                available_models = ", ".join(sorted(PRICING.keys()))
                raise ValueError(
                    f"Unknown model '{model}'. Available models: {available_models}"
                )
            cost = calculate_cost(model, input_tokens, output_tokens)

        result["estimated_cost_usd"] = cost
        total_estimated_cost_usd += cost

    return provider_results, round(total_estimated_cost_usd, 6)


__all__ = ["PRICING", "calculate_all_costs", "calculate_cost"]

