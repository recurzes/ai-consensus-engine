"""Unit tests for the cost tracker pricing table."""

import inspect
import unittest

from app.schemas.models import ProviderResult, Telemetry
from app.services import (
    PRICING as SERVICES_PRICING,
    calculate_all_costs as services_calculate_all_costs,
    calculate_cost as services_calculate_cost,
)
from app.services.cost_tracker import PRICING, calculate_all_costs, calculate_cost
import app.services.cost_tracker as cost_tracker_module


class TestCostTrackerPricingTable(unittest.TestCase):
    """Test suite verifying PRICING constant table structure and values."""

    EXPECTED_MODELS = {
        "gemini-2.5-flash",
        "gpt-4o-mini",
        "claude-haiku-4-5",
        "gemini-2.5-pro",
        "gpt-4o",
    }

    EXPECTED_PRICING = {
        "gemini-2.5-flash": {
            "input_per_million": 0.30,
            "output_per_million": 2.50,
        },
        "gpt-4o-mini": {
            "input_per_million": 0.15,
            "output_per_million": 0.60,
        },
        "claude-haiku-4-5": {
            "input_per_million": 1.00,
            "output_per_million": 5.00,
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

    def test_pricing_importable(self):
        """Verify PRICING is a dictionary and can be imported from both modules."""
        self.assertIsInstance(PRICING, dict)
        self.assertIs(PRICING, SERVICES_PRICING)

    def test_all_expected_models_present(self):
        """Verify all three worker models and both arbiter models exist in PRICING."""
        self.assertTrue(self.EXPECTED_MODELS.issubset(set(PRICING.keys())))

    def test_pricing_entries_have_required_keys(self):
        """Verify all entries contain input_per_million and output_per_million."""
        for model_name in self.EXPECTED_MODELS:
            with self.subTest(model=model_name):
                self.assertIn(model_name, PRICING)
                entry = PRICING[model_name]
                self.assertIsInstance(entry, dict)
                self.assertIn("input_per_million", entry)
                self.assertIn("output_per_million", entry)

    def test_pricing_values_are_floats_and_positive(self):
        """Verify that all pricing values are floats and strictly positive."""
        for model_name, rates in PRICING.items():
            for key in ("input_per_million", "output_per_million"):
                with self.subTest(model=model_name, key=key):
                    rate = rates[key]
                    self.assertIsInstance(
                        rate,
                        float,
                        f"Expected float for {model_name}['{key}'], got {type(rate).__name__}",
                    )
                    self.assertNotIsInstance(
                        rate,
                        bool,
                        f"Boolean value not allowed for {model_name}['{key}']",
                    )
                    self.assertGreater(
                        rate,
                        0.0,
                        f"Expected positive rate for {model_name}['{key}'], got {rate}",
                    )

    def test_exact_pricing_rates_match_spec(self):
        """Verify the exact pricing values match the Phase 5 pricing table specification."""
        for model_name, expected_rates in self.EXPECTED_PRICING.items():
            with self.subTest(model=model_name):
                self.assertEqual(
                    PRICING[model_name]["input_per_million"],
                    expected_rates["input_per_million"],
                )
                self.assertEqual(
                    PRICING[model_name]["output_per_million"],
                    expected_rates["output_per_million"],
                )

    def test_source_contains_verification_comment_and_links(self):
        """Verify cost_tracker.py documents when pricing was last verified and provider URLs."""
        source = inspect.getsource(cost_tracker_module)
        self.assertIn("verified", source.lower())
        self.assertIn("https://ai.google.dev/pricing", source)
        self.assertIn("https://openai.com/pricing", source)
        self.assertIn("https://www.anthropic.com/pricing", source)


class TestCostTrackerCalculation(unittest.TestCase):
    """Test suite verifying calculate_cost and calculate_all_costs logic."""

    def test_functions_importable_from_services_and_cost_tracker(self):
        """Verify calculation functions are importable from app.services and cost_tracker."""
        self.assertIs(calculate_cost, services_calculate_cost)
        self.assertIs(calculate_all_costs, services_calculate_all_costs)
        self.assertTrue(callable(calculate_cost))
        self.assertTrue(callable(calculate_all_costs))

    def test_calculate_cost_gemini_2_5_flash_positive(self):
        """Verify calculate_cost for gemini-2.5-flash with tokens returns positive float.

        Rates for gemini-2.5-flash: input=$0.30/1M, output=$2.50/1M
        85 input tokens: 85 / 1_000_000 * 0.30 = 0.0000255
        210 output tokens: 210 / 1_000_000 * 2.50 = 0.000525
        Total = 0.0005505 -> rounded to 6 decimals = 0.000551
        """
        cost = calculate_cost("gemini-2.5-flash", 85, 210)
        self.assertIsInstance(cost, float)
        self.assertGreater(cost, 0.0)
        self.assertEqual(cost, 0.000551)

    def test_calculate_cost_zero_tokens_returns_zero(self):
        """Verify calculate_cost with 0 input and 0 output tokens returns 0.0."""
        for model in PRICING:
            with self.subTest(model=model):
                cost = calculate_cost(model, 0, 0)
                self.assertIsInstance(cost, float)
                self.assertEqual(cost, 0.0)

    def test_calculate_cost_all_pricing_models(self):
        """Verify calculate_cost computes expected USD costs for all configured models."""
        test_cases = [
            # gpt-4o-mini: input 0.15, output 0.60 per 1M
            ("gpt-4o-mini", 1000, 2000, round((1000 / 1e6 * 0.15) + (2000 / 1e6 * 0.60), 6)),
            # claude-haiku-4-5: input 1.00, output 5.00 per 1M
            ("claude-haiku-4-5", 500, 500, round((500 / 1e6 * 1.00) + (500 / 1e6 * 5.00), 6)),
            # gemini-2.5-pro: input 1.25, output 10.00 per 1M
            ("gemini-2.5-pro", 1000, 1000, round((1000 / 1e6 * 1.25) + (1000 / 1e6 * 10.00), 6)),
            # gpt-4o: input 2.50, output 10.00 per 1M
            ("gpt-4o", 1000, 1000, round((1000 / 1e6 * 2.50) + (1000 / 1e6 * 10.00), 6)),
        ]
        for model, inp, out, expected in test_cases:
            with self.subTest(model=model):
                cost = calculate_cost(model, inp, out)
                self.assertEqual(cost, expected)

    def test_calculate_cost_unknown_model_raises_value_error(self):
        """Verify calculate_cost raises ValueError with clear message for unknown models."""
        with self.assertRaises(ValueError) as ctx:
            calculate_cost("unknown_model", 100, 100)
        error_msg = str(ctx.exception)
        self.assertIn("unknown_model", error_msg)
        self.assertIn("Available models", error_msg)

    def test_calculate_cost_negative_tokens_raises_value_error(self):
        """Verify calculate_cost raises ValueError when negative token counts are passed."""
        with self.assertRaises(ValueError) as ctx1:
            calculate_cost("gpt-4o-mini", -10, 100)
        self.assertIn("negative", str(ctx1.exception).lower())

        with self.assertRaises(ValueError) as ctx2:
            calculate_cost("gpt-4o-mini", 100, -5)
        self.assertIn("negative", str(ctx2.exception).lower())

    def test_calculate_cost_is_pure_function(self):
        """Verify calculate_cost produces idempotent results without side effects."""
        first_call = calculate_cost("gemini-2.5-flash", 1234, 5678)
        second_call = calculate_cost("gemini-2.5-flash", 1234, 5678)
        third_call = calculate_cost("gemini-2.5-flash", 1234, 5678)
        self.assertEqual(first_call, second_call)
        self.assertEqual(second_call, third_call)

    def test_calculate_cost_rounds_to_six_decimal_places(self):
        """Verify calculate_cost rounds the resulting USD value to 6 decimal places."""
        # 1 input token of gpt-4o-mini: 1 / 1e6 * 0.15 = 0.00000015 -> rounds to 0.000000
        cost_sub_micro = calculate_cost("gpt-4o-mini", 1, 0)
        self.assertEqual(cost_sub_micro, 0.0)

        # 7 input tokens of gpt-4o-mini: 7 / 1e6 * 0.15 = 0.00000105 -> rounds to 0.000001
        cost_micro = calculate_cost("gpt-4o-mini", 7, 0)
        self.assertEqual(cost_micro, 0.000001)

    def test_calculate_all_costs_all_success_three_providers(self):
        """Verify calculate_all_costs updates each dict with estimated_cost_usd and sums total."""
        results = [
            {
                "status": "success",
                "model": "gemini-2.5-flash",
                "duration_seconds": 1.15,
                "tokens": {"input": 85, "output": 210},
                "response_text": "Gemini response text",
            },
            {
                "status": "success",
                "model": "gpt-4o-mini",
                "duration_seconds": 1.48,
                "tokens": {"input": 85, "output": 230},
                "response_text": "OpenAI response text",
            },
            {
                "status": "success",
                "model": "claude-haiku-4-5",
                "duration_seconds": 1.82,
                "tokens": {"input": 85, "output": 245},
                "response_text": "Claude response text",
            },
        ]

        expected_gemini_cost = calculate_cost("gemini-2.5-flash", 85, 210)
        expected_openai_cost = calculate_cost("gpt-4o-mini", 85, 230)
        expected_claude_cost = calculate_cost("claude-haiku-4-5", 85, 245)
        expected_total = round(
            expected_gemini_cost + expected_openai_cost + expected_claude_cost, 6
        )

        updated_results, total_cost = calculate_all_costs(results)

        self.assertIs(updated_results, results)
        self.assertEqual(len(updated_results), 3)

        self.assertIn("estimated_cost_usd", updated_results[0])
        self.assertEqual(updated_results[0]["estimated_cost_usd"], expected_gemini_cost)

        self.assertIn("estimated_cost_usd", updated_results[1])
        self.assertEqual(updated_results[1]["estimated_cost_usd"], expected_openai_cost)

        self.assertIn("estimated_cost_usd", updated_results[2])
        self.assertEqual(updated_results[2]["estimated_cost_usd"], expected_claude_cost)

        self.assertEqual(total_cost, expected_total)
        self.assertGreater(total_cost, 0.0)

    def test_calculate_all_costs_partial_failure(self):
        """Verify failed provider with zero tokens contributes 0.0 to estimated cost and total."""
        results = [
            {
                "status": "success",
                "model": "gemini-2.5-flash",
                "duration_seconds": 1.15,
                "tokens": {"input": 85, "output": 210},
                "response_text": "Gemini response text",
            },
            {
                "status": "success",
                "model": "gpt-4o-mini",
                "duration_seconds": 1.48,
                "tokens": {"input": 85, "output": 230},
                "response_text": "OpenAI response text",
            },
            {
                "status": "error",
                "model": "claude-haiku-4-5",
                "duration_seconds": 0.05,
                "tokens": {"input": 0, "output": 0},
                "response_text": None,
                "error_message": "Invalid API Key",
            },
        ]

        expected_gemini_cost = calculate_cost("gemini-2.5-flash", 85, 210)
        expected_openai_cost = calculate_cost("gpt-4o-mini", 85, 230)
        expected_total = round(expected_gemini_cost + expected_openai_cost, 6)

        updated_results, total_cost = calculate_all_costs(results)

        self.assertEqual(updated_results[2]["estimated_cost_usd"], 0.0)
        self.assertEqual(total_cost, expected_total)

    def test_calculate_all_costs_all_failures(self):
        """Verify all failed providers contribute 0.0 and total is 0.0."""
        results = [
            {
                "status": "error",
                "model": "gemini-2.5-flash",
                "duration_seconds": 0.05,
                "tokens": {"input": 0, "output": 0},
                "response_text": None,
                "error_message": "Timeout",
            },
            {
                "status": "error",
                "model": "gpt-4o-mini",
                "duration_seconds": 0.05,
                "tokens": {"input": 0, "output": 0},
                "response_text": None,
                "error_message": "Network error",
            },
            {
                "status": "error",
                "model": "claude-haiku-4-5",
                "duration_seconds": 0.05,
                "tokens": {"input": 0, "output": 0},
                "response_text": None,
                "error_message": "Rate limit",
            },
        ]

        updated_results, total_cost = calculate_all_costs(results)

        for res in updated_results:
            self.assertEqual(res["estimated_cost_usd"], 0.0)
        self.assertEqual(total_cost, 0.0)

    def test_calculate_all_costs_empty_list(self):
        """Verify calculate_all_costs with empty list returns empty list and 0.0 total."""
        updated_results, total_cost = calculate_all_costs([])
        self.assertEqual(updated_results, [])
        self.assertEqual(total_cost, 0.0)

    def test_calculate_all_costs_schema_compatibility(self):
        """Verify updated results and total cost populate ProviderResult and Telemetry schemas."""
        results = [
            {
                "status": "success",
                "model": "gemini-2.5-flash",
                "duration_seconds": 1.15,
                "tokens": {"input": 85, "output": 210},
                "response_text": "The garage door is covered under Section I...",
            },
            {
                "status": "error",
                "model": "claude-haiku-4-5",
                "duration_seconds": 1.82,
                "tokens": {"input": 0, "output": 0},
                "response_text": None,
                "error_message": "Rate limit exceeded (HTTP 429)",
            },
        ]

        updated_results, total_cost = calculate_all_costs(results)

        # Confirm ProviderResult model_validate succeeds and preserves estimated_cost_usd
        gemini_pr = ProviderResult.model_validate(updated_results[0])
        self.assertEqual(gemini_pr.status, "success")
        self.assertIsInstance(gemini_pr.estimated_cost_usd, float)
        self.assertEqual(gemini_pr.estimated_cost_usd, updated_results[0]["estimated_cost_usd"])
        self.assertIn("estimated_cost_usd", gemini_pr.model_dump())

        claude_pr = ProviderResult.model_validate(updated_results[1])
        self.assertEqual(claude_pr.status, "error")
        self.assertEqual(claude_pr.estimated_cost_usd, 0.0)
        self.assertIn("estimated_cost_usd", claude_pr.model_dump())

        # Confirm Telemetry model accommodates total_estimated_cost_usd
        telemetry = Telemetry(
            total_duration_seconds=3.12,
            total_estimated_cost_usd=total_cost,
            successful_providers=["gemini"],
            failed_providers=["claude"],
        )
        self.assertEqual(telemetry.total_estimated_cost_usd, total_cost)
        self.assertEqual(telemetry.model_dump()["total_estimated_cost_usd"], total_cost)

    def test_calculate_all_costs_unknown_model_with_tokens_raises(self):
        """Verify calculate_all_costs raises ValueError if a result has unknown model and non-zero tokens."""
        results = [
            {
                "status": "success",
                "model": "non-existent-llm",
                "duration_seconds": 1.0,
                "tokens": {"input": 100, "output": 100},
                "response_text": "Some text",
            }
        ]
        with self.assertRaises(ValueError) as ctx:
            calculate_all_costs(results)
        self.assertIn("non-existent-llm", str(ctx.exception))


if __name__ == "__main__":
    unittest.main()

