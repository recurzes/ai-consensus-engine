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



class TestConsensusEndpointGracefulDegradation(unittest.TestCase):
    """Integration and route tests for POST /api/v1/consensus graceful degradation paths."""

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
        self.mock_claude_error = {
            "status": "error",
            "model": "claude-3-5-haiku",
            "duration_seconds": 0.25,
            "tokens": {"input": 0, "output": 0},
            "response_text": None,
            "error_message": "Authentication error: invalid API key",
        }
        self.mock_openai_error = {
            "status": "error",
            "model": "gpt-4o-mini",
            "duration_seconds": 0.30,
            "tokens": {"input": 0, "output": 0},
            "response_text": None,
            "error_message": "Rate limit exceeded (HTTP 429)",
        }
        self.mock_gemini_timeout = {
            "status": "error",
            "model": "gemini-2.5-flash",
            "duration_seconds": 12.0,
            "tokens": {"input": 0, "output": 0},
            "response_text": None,
            "error_message": "Request timed out after 12 seconds",
        }

    @patch("app.main.synthesize", new_callable=AsyncMock)
    @patch("app.main.run_workers", new_callable=AsyncMock)
    def test_one_provider_failure_returns_200_and_consensus(
        self, mock_run_workers, mock_synthesize
    ):
        """Verify 1-failure scenario (Claude fails, Gemini & OpenAI succeed) returns HTTP 200."""
        mock_run_workers.return_value = [
            self.mock_gemini_success,
            self.mock_openai_success,
            self.mock_claude_error,
        ]
        mock_synthesize.return_value = (
            "Consensus: Damage to the garage door is covered under Section I Dwelling. "
            "Damage to the trailer requires collision coverage under the auto policy."
        )

        response = self.client.post("/api/v1/consensus", json=self.valid_payload)

        self.assertEqual(response.status_code, 200)
        data = response.json()

        # Validate with Pydantic model
        validated = ConsensusResponse(**data)
        self.assertEqual(validated.status, "success")
        self.assertEqual(validated.prompt, self.valid_payload["prompt"])
        self.assertEqual(validated.consensus_answer, mock_synthesize.return_value)

        # Telemetry assertions
        telemetry = data["telemetry"]
        self.assertEqual(
            telemetry["successful_providers"], ["gemini", "openai"]
        )
        self.assertEqual(telemetry["failed_providers"], ["claude"])
        self.assertGreater(telemetry["total_duration_seconds"], 0.0)
        self.assertGreater(telemetry["total_estimated_cost_usd"], 0.0)

        # Provider breakdown assertions
        breakdown = data["provider_breakdown"]
        self.assertEqual(breakdown["gemini"]["status"], "success")
        self.assertIsNotNone(breakdown["gemini"]["response_text"])
        self.assertEqual(breakdown["openai"]["status"], "success")
        self.assertIsNotNone(breakdown["openai"]["response_text"])

        self.assertEqual(breakdown["claude"]["status"], "error")
        self.assertIsNone(breakdown["claude"]["response_text"])
        self.assertEqual(
            breakdown["claude"]["error_message"],
            "Authentication error: invalid API key",
        )

        # Verify synthesize was called with exactly 2 successful results
        mock_synthesize.assert_awaited_once()
        synth_kwargs = mock_synthesize.call_args[1]
        self.assertEqual(len(synth_kwargs["successful_results"]), 2)

    @patch("app.main.synthesize_single", new_callable=AsyncMock)
    @patch("app.main.run_workers", new_callable=AsyncMock)
    def test_two_providers_failure_returns_200_and_single_survivor_consensus(
        self, mock_run_workers, mock_synthesize_single
    ):
        """Verify 2-failure scenario (OpenAI & Claude fail, Gemini succeeds) calls synthesize_single and returns HTTP 200."""
        mock_run_workers.return_value = [
            self.mock_gemini_success,
            self.mock_openai_error,
            self.mock_claude_error,
        ]
        mock_synthesize_single.return_value = (
            "Refined: Damage to the garage door is covered under Section I Dwelling."
        )

        response = self.client.post("/api/v1/consensus", json=self.valid_payload)

        self.assertEqual(response.status_code, 200)
        data = response.json()

        validated = ConsensusResponse(**data)
        self.assertEqual(validated.status, "success")
        self.assertEqual(validated.consensus_answer, mock_synthesize_single.return_value)

        # Telemetry assertions
        telemetry = data["telemetry"]
        self.assertEqual(telemetry["successful_providers"], ["gemini"])
        self.assertEqual(telemetry["failed_providers"], ["openai", "claude"])

        # Provider breakdown assertions
        breakdown = data["provider_breakdown"]
        self.assertEqual(breakdown["gemini"]["status"], "success")
        self.assertEqual(breakdown["openai"]["status"], "error")
        self.assertIsNone(breakdown["openai"]["response_text"])
        self.assertEqual(breakdown["claude"]["status"], "error")
        self.assertIsNone(breakdown["claude"]["response_text"])

        # Verify synthesize_single was called with the single survivor
        mock_synthesize_single.assert_awaited_once()
        synth_kwargs = mock_synthesize_single.call_args[1]
        self.assertEqual(
            synth_kwargs["single_result"]["model"], "gemini-2.5-flash"
        )

    @patch("app.main.synthesize_single", new_callable=AsyncMock)
    @patch("app.main.run_workers", new_callable=AsyncMock)
    def test_timeout_and_exception_failures_in_workers(
        self, mock_run_workers, mock_synthesize_single
    ):
        """Verify pipeline handles timeout dicts and unexpected Exceptions without unhandled error."""
        mock_run_workers.return_value = [
            self.mock_gemini_timeout,
            self.mock_openai_success,
            RuntimeError("Unrecoverable network transport error"),
        ]
        mock_synthesize_single.return_value = (
            "Consensus: Garage door coverage validated under homeowners policy."
        )

        response = self.client.post("/api/v1/consensus", json=self.valid_payload)

        self.assertEqual(response.status_code, 200)
        data = response.json()

        telemetry = data["telemetry"]
        self.assertEqual(telemetry["successful_providers"], ["openai"])
        self.assertEqual(telemetry["failed_providers"], ["gemini", "claude"])

        breakdown = data["provider_breakdown"]
        self.assertEqual(breakdown["gemini"]["status"], "error")
        self.assertIn("timed out", breakdown["gemini"]["error_message"])
        self.assertEqual(breakdown["claude"]["status"], "error")
        self.assertIn(
            "Unrecoverable network transport error",
            breakdown["claude"]["error_message"],
        )
        self.assertEqual(breakdown["openai"]["status"], "success")

    @patch("app.main.synthesize", new_callable=AsyncMock)
    @patch("app.main.run_workers", new_callable=AsyncMock)
    def test_degraded_arbiter_error_returns_502(
        self, mock_run_workers, mock_synthesize
    ):
        """Verify ArbiterError during degraded synthesis returns HTTP 502."""
        mock_run_workers.return_value = [
            self.mock_gemini_success,
            self.mock_openai_success,
            self.mock_claude_error,
        ]
        mock_synthesize.side_effect = ArbiterError("Arbiter service unavailable")

        response = self.client.post("/api/v1/consensus", json=self.valid_payload)

        self.assertEqual(response.status_code, 502)
        data = response.json()
        self.assertEqual(data["status"], "error")
        self.assertIn("Arbiter synthesis failed", data["message"])


if __name__ == "__main__":
    unittest.main()

