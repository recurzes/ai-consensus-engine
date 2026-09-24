"""Cost tracker service for estimating per-request LLM API token costs.

Pricing data changes over time; keeping it as an isolated constant makes future
updates straightforward.
"""

from typing import Final

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

__all__ = ["PRICING"]
