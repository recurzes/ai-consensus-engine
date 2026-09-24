"""Unit tests for the cost tracker pricing table."""

import inspect
import unittest

from app.services import PRICING as SERVICES_PRICING
from app.services.cost_tracker import PRICING
import app.services.cost_tracker as cost_tracker_module


class TestCostTrackerPricingTable(unittest.TestCase):
    """Test suite verifying PRICING constant table structure and values."""

    EXPECTED_MODELS = {
        "gemini-2.5-flash",
        "gpt-4o-mini",
        "claude-3-5-haiku",
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


if __name__ == "__main__":
    unittest.main()
