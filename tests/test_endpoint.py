import os
import unittest
from unittest.mock import AsyncMock, patch

# Ensure test env defaults exist before loading settings or app
_TEST_ENV_DEFAULTS = {
    "GEMINI_API_KEY": "test-gemini-key",
    "OPENAI_API_KEY": "test-openai-key",
    "ANTHROPIC_API_KEY": "test-anthropic-key",
    "ARBITER_MODEL_PROVIDER": "gemini",
}
for k, v in _TEST_ENV_DEFAULTS.items():
    os.environ.setdefault(k, v)

from fastapi.testclient import TestClient

from app.main import app
from app.schemas.models import ConsensusResponse
from app.services.arbiter import ArbiterError


class TestConsensusEndpointHappyPath(unittest.TestCase):
    """Integration and route tests for POST /api/v1/consensus happy path."""

    def setUp(self):
        self.client = TestClient(app)
        self.valid_payload = {
            "prompt": "An insured backed a trailer into their garage door. How is this covered?",
            "context": {
                "role": "claims_adjuster",
                "line_of_business": "homeowners",
                "state": "MT",
            },
        }

        self.mock_gemini_success = {
            "status": "success",
            "model": "gemini-2.5-flash",
            "duration_seconds": 0.85,
            "tokens": {"input": 120, "output": 250},
            "response_text": "The garage door is covered under Section I Dwelling.",
        }
        self.mock_openai_success = {
            "status": "success",
            "model": "gpt-4o-mini",
            "duration_seconds": 1.10,
            "tokens": {"input": 120, "output": 280},
            "response_text": "Coverage applies to the attached garage door under Dwelling Coverage.",
        }
        self.mock_claude_success = {
            "status": "success",
            "model": "claude-3-5-haiku",
            "duration_seconds": 1.35,
            "tokens": {"input": 120, "output": 310},
            "response_text": "First-party property coverage applies under homeowners terms.",
        }

    def test_root_endpoint_health(self):
        """Verify GET / returns HTTP 200 and status ok."""
        response = self.client.get("/")
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json(), {"status": "ok"})

    @patch("app.main.synthesize", new_callable=AsyncMock)
    @patch("app.main.run_workers", new_callable=AsyncMock)
    def test_consensus_endpoint_all_success_happy_path(
        self, mock_run_workers, mock_synthesize
    ):
        """Verify full happy path pipeline when all 3 providers and arbiter succeed."""
        mock_run_workers.return_value = [
            self.mock_gemini_success,
            self.mock_openai_success,
            self.mock_claude_success,
        ]
        mock_synthesize.return_value = (
            "Consensus: Damage to the attached garage door is covered under Section I (Dwelling Coverage A) "
            "subject to the policy deductible."
        )

        response = self.client.post("/api/v1/consensus", json=self.valid_payload)

        self.assertEqual(response.status_code, 200)
        data = response.json()

        # Validate with Pydantic model
        validated = ConsensusResponse(**data)
        self.assertEqual(validated.status, "success")
        self.assertEqual(validated.prompt, self.valid_payload["prompt"])
        self.assertEqual(validated.applied_context.role, "claims_adjuster")
        self.assertEqual(validated.applied_context.line_of_business, "homeowners")
        self.assertEqual(validated.applied_context.state, "MT")
        self.assertEqual(validated.consensus_answer, mock_synthesize.return_value)

        # Telemetry assertions
        telemetry = data["telemetry"]
        self.assertEqual(
            telemetry["successful_providers"], ["gemini", "openai", "claude"]
        )
        self.assertEqual(telemetry["failed_providers"], [])
        self.assertGreater(telemetry["total_duration_seconds"], 0.0)
        self.assertGreater(telemetry["total_estimated_cost_usd"], 0.0)

        # Provider breakdown assertions
        breakdown = data["provider_breakdown"]
        self.assertIn("gemini", breakdown)
        self.assertIn("openai", breakdown)
        self.assertIn("claude", breakdown)

        for p_name in ("gemini", "openai", "claude"):
            provider_data = breakdown[p_name]
            self.assertEqual(provider_data["status"], "success")
            self.assertIsNotNone(provider_data["model"])
            self.assertGreater(provider_data["duration_seconds"], 0.0)
            self.assertIn("input", provider_data["tokens"])
            self.assertIn("output", provider_data["tokens"])
            self.assertIsNotNone(provider_data["response_text"])
            self.assertGreater(provider_data["estimated_cost_usd"], 0.0)

        # Confirm synthesize received the original prompt, context, and successful results
        mock_synthesize.assert_awaited_once()
        synth_args, synth_kwargs = mock_synthesize.call_args
        self.assertEqual(synth_kwargs["original_prompt"], self.valid_payload["prompt"])
        self.assertEqual(synth_kwargs["context"].role, "claims_adjuster")
        self.assertEqual(len(synth_kwargs["successful_results"]), 3)

    @patch("app.main.synthesize", new_callable=AsyncMock)
    @patch("app.main.run_workers", new_callable=AsyncMock)
    def test_consensus_endpoint_arbiter_error_returns_502(
        self, mock_run_workers, mock_synthesize
    ):
        """Verify ArbiterError is caught by exception handler and returns HTTP 502."""
        mock_run_workers.return_value = [
            self.mock_gemini_success,
            self.mock_openai_success,
            self.mock_claude_success,
        ]
        mock_synthesize.side_effect = ArbiterError("Arbiter LLM connection dropped")

        response = self.client.post("/api/v1/consensus", json=self.valid_payload)

        self.assertEqual(response.status_code, 502)
        data = response.json()
        self.assertEqual(data["status"], "error")
        self.assertIn("Arbiter synthesis failed", data["message"])
        self.assertIn("Arbiter LLM connection dropped", data["message"])

    def test_validation_error_missing_prompt(self):
        """Verify HTTP 422 is returned when prompt is missing."""
        invalid_payload = {
            "context": {
                "role": "claims_adjuster",
                "line_of_business": "homeowners",
            }
        }
        response = self.client.post("/api/v1/consensus", json=invalid_payload)
        self.assertEqual(response.status_code, 422)

    def test_validation_error_missing_context(self):
        """Verify HTTP 422 is returned when context is missing."""
        invalid_payload = {"prompt": "What is the deductible?"}
        response = self.client.post("/api/v1/consensus", json=invalid_payload)
        self.assertEqual(response.status_code, 422)

    def test_validation_error_invalid_role_enum(self):
        """Verify HTTP 422 is returned when an invalid role persona is submitted."""
        invalid_payload = {
            "prompt": "Coverage question",
            "context": {
                "role": "non_existent_role",
                "line_of_business": "homeowners",
            },
        }
        response = self.client.post("/api/v1/consensus", json=invalid_payload)
        self.assertEqual(response.status_code, 422)

    def test_validation_error_invalid_lob_enum(self):
        """Verify HTTP 422 is returned when an invalid line_of_business is submitted."""
        invalid_payload = {
            "prompt": "Coverage question",
            "context": {
                "role": "claims_adjuster",
                "line_of_business": "space_travel_insurance",
            },
        }
        response = self.client.post("/api/v1/consensus", json=invalid_payload)
        self.assertEqual(response.status_code, 422)


if __name__ == "__main__":
    unittest.main()
