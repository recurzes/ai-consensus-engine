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

    @patch("app.main.synthesize", new_callable=AsyncMock)
    @patch("app.main.run_workers", new_callable=AsyncMock)
    def test_custom_anthropic_api_key_header_passed_to_worker_orchestrator(
        self, mock_run_workers, mock_synthesize
    ):
        """Verify custom x-anthropic-api-key header creates custom claude_client for run_workers."""
        mock_run_workers.return_value = [
            self.mock_gemini_success,
            self.mock_openai_success,
            self.mock_claude_error,
        ]
        mock_synthesize.return_value = "Consensus answer"

        headers = {"x-anthropic-api-key": "sk-intentionally-invalid-for-failover-test"}
        response = self.client.post(
            "/api/v1/consensus",
            json=self.valid_payload,
            headers=headers,
        )

        self.assertEqual(response.status_code, 200)
        mock_run_workers.assert_awaited_once()
        _, kwargs = mock_run_workers.call_args
        self.assertIn("claude_client", kwargs)
        self.assertIsNotNone(kwargs["claude_client"])
        self.assertEqual(
            kwargs["claude_client"].api_key,
            "sk-intentionally-invalid-for-failover-test",
        )

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


class TestConsensusEndpointTotalFailure(unittest.TestCase):
    """Integration and route tests for total failure and global exception handling."""

    def setUp(self):
        # Disable re-raising server exceptions so TestClient returns HTTP 500 responses
        self.client = TestClient(app, raise_server_exceptions=False)
        self.valid_payload = {
            "prompt": "An insured backed a trailer into their garage door. How is this covered?",
            "context": {
                "role": "claims_adjuster",
                "line_of_business": "homeowners",
                "state": "MT",
            },
        }

        self.mock_gemini_error = {
            "status": "error",
            "model": "gemini-2.5-flash",
            "duration_seconds": 12.0,
            "tokens": {"input": 0, "output": 0},
            "response_text": None,
            "error_message": "Request timed out after 12 seconds",
        }
        self.mock_openai_error = {
            "status": "error",
            "model": "gpt-4o-mini",
            "duration_seconds": 0.30,
            "tokens": {"input": 0, "output": 0},
            "response_text": None,
            "error_message": "Authentication error: invalid API key",
        }
        self.mock_claude_error = {
            "status": "error",
            "model": "claude-3-5-haiku",
            "duration_seconds": 0.25,
            "tokens": {"input": 0, "output": 0},
            "response_text": None,
            "error_message": "Rate limit exceeded",
        }

    @patch("app.main.synthesize_single", new_callable=AsyncMock)
    @patch("app.main.synthesize", new_callable=AsyncMock)
    @patch("app.main.run_workers", new_callable=AsyncMock)
    def test_all_providers_failure_returns_502_and_structured_error_payload(
        self, mock_run_workers, mock_synthesize, mock_synthesize_single
    ):
        """Verify that when all 3 providers fail, HTTP 502 is returned and arbiter is not called."""
        mock_run_workers.return_value = [
            self.mock_gemini_error,
            self.mock_openai_error,
            self.mock_claude_error,
        ]

        response = self.client.post("/api/v1/consensus", json=self.valid_payload)

        self.assertEqual(response.status_code, 502)
        data = response.json()
        self.assertEqual(data["status"], "error")
        self.assertNotIn("consensus_answer", data)
        self.assertIsNone(data.get("consensus_answer"))
        self.assertEqual(
            data["message"],
            "All AI providers failed. No consensus could be generated.",
        )
        self.assertIn("failed_providers", data)
        self.assertEqual(
            data["failed_providers"],
            {
                "gemini": "Request timed out after 12 seconds",
                "openai": "Authentication error: invalid API key",
                "claude": "Rate limit exceeded",
            },
        )

        # Arbiter must NEVER be called
        mock_synthesize.assert_not_called()
        mock_synthesize_single.assert_not_called()

    @patch("app.main.synthesize_single", new_callable=AsyncMock)
    @patch("app.main.synthesize", new_callable=AsyncMock)
    @patch("app.main.run_workers", new_callable=AsyncMock)
    def test_all_providers_exceptions_returns_502(
        self, mock_run_workers, mock_synthesize, mock_synthesize_single
    ):
        """Verify raw exceptions from all 3 workers return HTTP 502 with individual error messages."""
        mock_run_workers.return_value = [
            TimeoutError("Gemini transport timed out"),
            RuntimeError("OpenAI authentication failure"),
            ConnectionError("Claude rate limit reached"),
        ]

        response = self.client.post("/api/v1/consensus", json=self.valid_payload)

        self.assertEqual(response.status_code, 502)
        data = response.json()
        self.assertEqual(data["status"], "error")
        self.assertEqual(
            data["message"],
            "All AI providers failed. No consensus could be generated.",
        )
        self.assertEqual(
            data["failed_providers"],
            {
                "gemini": "Gemini transport timed out",
                "openai": "OpenAI authentication failure",
                "claude": "Claude rate limit reached",
            },
        )

        mock_synthesize.assert_not_called()
        mock_synthesize_single.assert_not_called()

    @patch("app.main.run_workers", new_callable=AsyncMock)
    def test_global_exception_handler_returns_500_on_unhandled_route_exception(
        self, mock_run_workers
    ):
        """Verify unexpected unhandled exceptions in route return HTTP 500 with clean JSON body."""
        mock_run_workers.side_effect = RuntimeError("Fatal internal route bug")

        response = self.client.post("/api/v1/consensus", json=self.valid_payload)

        self.assertEqual(response.status_code, 500)
        data = response.json()
        self.assertEqual(
            data,
            {
                "status": "error",
                "message": "An unexpected error occurred.",
            },
        )
        # Verify no traceback in the response body
        self.assertNotIn("Traceback", response.text)
        self.assertNotIn("Fatal internal route bug", response.text)

    @patch("app.main.synthesize", new_callable=AsyncMock)
    @patch("app.main.run_workers", new_callable=AsyncMock)
    def test_app_recovers_after_total_failure_without_restart(
        self, mock_run_workers, mock_synthesize
    ):
        """Verify app does not crash or corrupt state, successfully serving requests after total failure."""
        # 1. Total failure request
        mock_run_workers.return_value = [
            self.mock_gemini_error,
            self.mock_openai_error,
            self.mock_claude_error,
        ]
        fail_response = self.client.post("/api/v1/consensus", json=self.valid_payload)
        self.assertEqual(fail_response.status_code, 502)

        # 2. Subsequent happy path request
        mock_run_workers.return_value = [
            {
                "status": "success",
                "model": "gemini-2.5-flash",
                "duration_seconds": 0.8,
                "tokens": {"input": 100, "output": 200},
                "response_text": "Covered under Dwelling A.",
            },
            {
                "status": "success",
                "model": "gpt-4o-mini",
                "duration_seconds": 0.9,
                "tokens": {"input": 100, "output": 200},
                "response_text": "Covered under Dwelling A.",
            },
            {
                "status": "success",
                "model": "claude-3-5-haiku",
                "duration_seconds": 1.0,
                "tokens": {"input": 100, "output": 200},
                "response_text": "Covered under Dwelling A.",
            },
        ]
        mock_synthesize.return_value = "Consensus: Covered under Dwelling A."
        success_response = self.client.post("/api/v1/consensus", json=self.valid_payload)
        self.assertEqual(success_response.status_code, 200)
        self.assertEqual(success_response.json()["status"], "success")


SPEC_VALID_PAYLOAD = {
    "prompt": "Explain coverage triggers for a homeowners claim in Montana.",
    "context": {
        "role": "claims_adjuster",
        "line_of_business": "homeowners",
        "state": "MT",
    },
}


class TestEndpointIntegrationSpec(unittest.TestCase):
    """Integration tests for POST /api/v1/consensus corresponding to Phase 9.

    Specification: notes/backend-spec/29-test-endpoint-integration-tests.md
    Tests the full HTTP layer (status codes, response shape, success, degradation, and failure).
    All provider calls and arbiter synthesis are mocked at the service layer.
    """

    def setUp(self):
        self.client = TestClient(app, raise_server_exceptions=False)
        self.valid_payload = SPEC_VALID_PAYLOAD

        self.mock_gemini_success = {
            "status": "success",
            "model": "gemini-2.5-flash",
            "duration_seconds": 0.85,
            "tokens": {"input": 120, "output": 250},
            "response_text": "Montana homeowners coverage triggers upon direct physical loss to covered dwelling.",
        }
        self.mock_openai_success = {
            "status": "success",
            "model": "gpt-4o-mini",
            "duration_seconds": 1.10,
            "tokens": {"input": 120, "output": 280},
            "response_text": "Coverage applies under Section I Dwelling for accidental physical damage in MT.",
        }
        self.mock_claude_success = {
            "status": "success",
            "model": "claude-3-5-haiku",
            "duration_seconds": 1.35,
            "tokens": {"input": 120, "output": 310},
            "response_text": "First-party property coverage triggers require fortuitous direct physical loss.",
        }
        self.mock_gemini_failure = {
            "status": "error",
            "model": "gemini-2.5-flash",
            "duration_seconds": 0.05,
            "tokens": {"input": 0, "output": 0},
            "response_text": None,
            "error_message": "Gemini quota exceeded (HTTP 429)",
        }
        self.mock_openai_failure = {
            "status": "error",
            "model": "gpt-4o-mini",
            "duration_seconds": 0.08,
            "tokens": {"input": 0, "output": 0},
            "response_text": None,
            "error_message": "OpenAI authentication failed: invalid API key",
        }
        self.mock_claude_failure = {
            "status": "error",
            "model": "claude-3-5-haiku",
            "duration_seconds": 0.10,
            "tokens": {"input": 0, "output": 0},
            "response_text": None,
            "error_message": "Claude connection timed out after 12 seconds",
        }

    # ------------------------------------------------------------------------
    # Scenario 1: All-success path
    # ------------------------------------------------------------------------
    @patch("app.main.synthesize", new_callable=AsyncMock)
    @patch("app.main.run_workers", new_callable=AsyncMock)
    def test_scenario_1_all_success_path(self, mock_run_workers, mock_synthesize):
        """Scenario 1: All 3 providers succeed -> HTTP 200, status success, non-empty answer, 3 successful, 0 failed."""
        mock_run_workers.return_value = [
            self.mock_gemini_success,
            self.mock_openai_success,
            self.mock_claude_success,
        ]
        canned_consensus = (
            "Consensus: Homeowners coverage in Montana triggers upon accidental direct physical loss "
            "to the dwelling under Section I, subject to policy exclusions and deductible."
        )
        mock_synthesize.return_value = canned_consensus

        response = self.client.post("/api/v1/consensus", json=self.valid_payload)

        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertEqual(data["status"], "success")
        self.assertIsInstance(data["consensus_answer"], str)
        self.assertGreater(len(data["consensus_answer"]), 0)
        self.assertEqual(data["consensus_answer"], canned_consensus)

        telemetry = data["telemetry"]
        self.assertEqual(len(telemetry["successful_providers"]), 3)
        self.assertEqual(telemetry["successful_providers"], ["gemini", "openai", "claude"])
        self.assertEqual(telemetry["failed_providers"], [])
        self.assertEqual(len(telemetry["failed_providers"]), 0)

        # Validate Arbiter was invoked with all 3 successful results
        mock_synthesize.assert_awaited_once()
        synth_kwargs = mock_synthesize.call_args[1]
        self.assertEqual(len(synth_kwargs["successful_results"]), 3)

    # ------------------------------------------------------------------------
    # Scenario 2: 1-failure path
    # ------------------------------------------------------------------------
    @patch("app.main.synthesize", new_callable=AsyncMock)
    @patch("app.main.run_workers", new_callable=AsyncMock)
    def test_scenario_2_one_failure_path_claude_fails(
        self, mock_run_workers, mock_synthesize
    ):
        """Scenario 2: 2 succeed, 1 fails (Claude fails) -> HTTP 200, consensus populated, 1 failed provider."""
        mock_run_workers.return_value = [
            self.mock_gemini_success,
            self.mock_openai_success,
            self.mock_claude_failure,
        ]
        canned_consensus = "Consensus from Gemini & OpenAI: Coverage triggers under Dwelling Section I."
        mock_synthesize.return_value = canned_consensus

        response = self.client.post("/api/v1/consensus", json=self.valid_payload)

        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertEqual(data["status"], "success")
        self.assertIsInstance(data["consensus_answer"], str)
        self.assertGreater(len(data["consensus_answer"]), 0)
        self.assertEqual(data["consensus_answer"], canned_consensus)

        telemetry = data["telemetry"]
        self.assertEqual(len(telemetry["successful_providers"]), 2)
        self.assertEqual(telemetry["successful_providers"], ["gemini", "openai"])
        self.assertEqual(len(telemetry["failed_providers"]), 1)
        self.assertEqual(telemetry["failed_providers"], ["claude"])

        mock_synthesize.assert_awaited_once()
        self.assertEqual(len(mock_synthesize.call_args[1]["successful_results"]), 2)

    @patch("app.main.synthesize", new_callable=AsyncMock)
    @patch("app.main.run_workers", new_callable=AsyncMock)
    def test_scenario_2_one_failure_path_openai_fails(
        self, mock_run_workers, mock_synthesize
    ):
        """Scenario 2 permutation: OpenAI fails, Gemini & Claude succeed -> HTTP 200, failed has 1 entry."""
        mock_run_workers.return_value = [
            self.mock_gemini_success,
            self.mock_openai_failure,
            self.mock_claude_success,
        ]
        mock_synthesize.return_value = "Consensus from Gemini & Claude."

        response = self.client.post("/api/v1/consensus", json=self.valid_payload)

        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertEqual(data["status"], "success")
        self.assertTrue(bool(data["consensus_answer"]))
        self.assertEqual(len(data["telemetry"]["successful_providers"]), 2)
        self.assertEqual(len(data["telemetry"]["failed_providers"]), 1)
        self.assertEqual(data["telemetry"]["failed_providers"], ["openai"])

    @patch("app.main.synthesize", new_callable=AsyncMock)
    @patch("app.main.run_workers", new_callable=AsyncMock)
    def test_scenario_2_one_failure_path_gemini_fails(
        self, mock_run_workers, mock_synthesize
    ):
        """Scenario 2 permutation: Gemini fails, OpenAI & Claude succeed -> HTTP 200, failed has 1 entry."""
        mock_run_workers.return_value = [
            self.mock_gemini_failure,
            self.mock_openai_success,
            self.mock_claude_success,
        ]
        mock_synthesize.return_value = "Consensus from OpenAI & Claude."

        response = self.client.post("/api/v1/consensus", json=self.valid_payload)

        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertEqual(data["status"], "success")
        self.assertTrue(bool(data["consensus_answer"]))
        self.assertEqual(len(data["telemetry"]["successful_providers"]), 2)
        self.assertEqual(len(data["telemetry"]["failed_providers"]), 1)
        self.assertEqual(data["telemetry"]["failed_providers"], ["gemini"])

    # ------------------------------------------------------------------------
    # Scenario 3: 2-failure path
    # ------------------------------------------------------------------------
    @patch("app.main.synthesize_single", new_callable=AsyncMock)
    @patch("app.main.run_workers", new_callable=AsyncMock)
    def test_scenario_3_two_failure_path_gemini_survives(
        self, mock_run_workers, mock_synthesize_single
    ):
        """Scenario 3: 1 succeeds, 2 fail (Gemini survives) -> HTTP 200, consensus populated, 2 failed providers."""
        mock_run_workers.return_value = [
            self.mock_gemini_success,
            self.mock_openai_failure,
            self.mock_claude_failure,
        ]
        single_survivor_consensus = (
            "Refined: Montana homeowners coverage triggers upon direct physical loss to dwelling."
        )
        mock_synthesize_single.return_value = single_survivor_consensus

        response = self.client.post("/api/v1/consensus", json=self.valid_payload)

        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertEqual(data["status"], "success")
        self.assertIsInstance(data["consensus_answer"], str)
        self.assertGreater(len(data["consensus_answer"]), 0)
        self.assertEqual(data["consensus_answer"], single_survivor_consensus)

        telemetry = data["telemetry"]
        self.assertEqual(len(telemetry["successful_providers"]), 1)
        self.assertEqual(telemetry["successful_providers"], ["gemini"])
        self.assertEqual(len(telemetry["failed_providers"]), 2)
        self.assertEqual(telemetry["failed_providers"], ["openai", "claude"])

        mock_synthesize_single.assert_awaited_once()
        synth_kwargs = mock_synthesize_single.call_args[1]
        self.assertEqual(synth_kwargs["single_result"]["model"], "gemini-2.5-flash")

    @patch("app.main.synthesize_single", new_callable=AsyncMock)
    @patch("app.main.run_workers", new_callable=AsyncMock)
    def test_scenario_3_two_failure_path_openai_survives(
        self, mock_run_workers, mock_synthesize_single
    ):
        """Scenario 3 permutation: Only OpenAI succeeds -> HTTP 200, consensus populated, 2 failed providers."""
        mock_run_workers.return_value = [
            self.mock_gemini_failure,
            self.mock_openai_success,
            self.mock_claude_failure,
        ]
        mock_synthesize_single.return_value = "Refined single response from OpenAI."

        response = self.client.post("/api/v1/consensus", json=self.valid_payload)

        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertEqual(data["status"], "success")
        self.assertTrue(bool(data["consensus_answer"]))
        self.assertEqual(len(data["telemetry"]["successful_providers"]), 1)
        self.assertEqual(telemetry := data["telemetry"]["failed_providers"], ["gemini", "claude"])
        self.assertEqual(len(telemetry), 2)

    @patch("app.main.synthesize_single", new_callable=AsyncMock)
    @patch("app.main.run_workers", new_callable=AsyncMock)
    def test_scenario_3_two_failure_path_claude_survives(
        self, mock_run_workers, mock_synthesize_single
    ):
        """Scenario 3 permutation: Only Claude succeeds -> HTTP 200, consensus populated, 2 failed providers."""
        mock_run_workers.return_value = [
            self.mock_gemini_failure,
            self.mock_openai_failure,
            self.mock_claude_success,
        ]
        mock_synthesize_single.return_value = "Refined single response from Claude."

        response = self.client.post("/api/v1/consensus", json=self.valid_payload)

        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertEqual(data["status"], "success")
        self.assertTrue(bool(data["consensus_answer"]))
        self.assertEqual(len(data["telemetry"]["successful_providers"]), 1)
        self.assertEqual(len(data["telemetry"]["failed_providers"]), 2)
        self.assertEqual(data["telemetry"]["failed_providers"], ["gemini", "openai"])

    # ------------------------------------------------------------------------
    # Scenario 4: All-failure path
    # ------------------------------------------------------------------------
    @patch("app.main.synthesize_single", new_callable=AsyncMock)
    @patch("app.main.synthesize", new_callable=AsyncMock)
    @patch("app.main.run_workers", new_callable=AsyncMock)
    def test_scenario_4_all_failure_path(
        self, mock_run_workers, mock_synthesize, mock_synthesize_single
    ):
        """Scenario 4: All 3 fail -> HTTP 502/503, status error, consensus_answer absent/null."""
        mock_run_workers.return_value = [
            self.mock_gemini_failure,
            self.mock_openai_failure,
            self.mock_claude_failure,
        ]

        response = self.client.post("/api/v1/consensus", json=self.valid_payload)

        self.assertIn(response.status_code, (502, 503))
        self.assertEqual(response.status_code, 502)

        data = response.json()
        self.assertEqual(data["status"], "error")
        self.assertNotIn("consensus_answer", data)
        self.assertIsNone(data.get("consensus_answer"))
        self.assertEqual(
            data["message"],
            "All AI providers failed. No consensus could be generated.",
        )
        self.assertIn("failed_providers", data)
        self.assertEqual(len(data["failed_providers"]), 3)
        self.assertIn("gemini", data["failed_providers"])
        self.assertIn("openai", data["failed_providers"])
        self.assertIn("claude", data["failed_providers"])

        # Arbiter must not be called when all workers fail
        mock_synthesize.assert_not_called()
        mock_synthesize_single.assert_not_called()

    # ------------------------------------------------------------------------
    # Scenario 5: Invalid request body
    # ------------------------------------------------------------------------
    def test_scenario_5_invalid_request_body_missing_prompt(self):
        """Scenario 5: Missing prompt field -> HTTP 422 Unprocessable Entity."""
        invalid_payload = {
            "context": {
                "role": "claims_adjuster",
                "line_of_business": "homeowners",
                "state": "MT",
            }
        }
        response = self.client.post("/api/v1/consensus", json=invalid_payload)
        self.assertEqual(response.status_code, 422)

    def test_scenario_5_invalid_request_body_invalid_role_enum(self):
        """Scenario 5: Invalid role enum value -> HTTP 422 Unprocessable Entity."""
        invalid_payload = {
            "prompt": "Explain coverage triggers for homeowners in Montana.",
            "context": {
                "role": "not_a_valid_insurance_role",
                "line_of_business": "homeowners",
                "state": "MT",
            },
        }
        response = self.client.post("/api/v1/consensus", json=invalid_payload)
        self.assertEqual(response.status_code, 422)

    def test_scenario_5_invalid_request_body_empty_payload(self):
        """Scenario 5: Completely empty payload -> HTTP 422 Unprocessable Entity."""
        response = self.client.post("/api/v1/consensus", json={})
        self.assertEqual(response.status_code, 422)

    # ------------------------------------------------------------------------
    # Status Code Matrix Verification
    # ------------------------------------------------------------------------
    @patch("app.main.synthesize_single", new_callable=AsyncMock)
    @patch("app.main.synthesize", new_callable=AsyncMock)
    @patch("app.main.run_workers", new_callable=AsyncMock)
    def test_status_code_matrix(
        self, mock_run_workers, mock_synthesize, mock_synthesize_single
    ):
        """Verifies the complete status code matrix from the specification:
        | Scenario               | Expected HTTP Status |
        | All 3 providers succeed| 200                  |
        | 2 of 3 succeed         | 200                  |
        | 1 of 3 succeeds        | 200                  |
        | All 3 fail             | 502                  |
        | Invalid request body   | 422                  |
        """
        mock_synthesize.return_value = "Consensus answer"
        mock_synthesize_single.return_value = "Single survivor answer"

        # 1. All 3 succeed -> 200
        mock_run_workers.return_value = [
            self.mock_gemini_success,
            self.mock_openai_success,
            self.mock_claude_success,
        ]
        r1 = self.client.post("/api/v1/consensus", json=self.valid_payload)
        self.assertEqual(r1.status_code, 200, "All 3 succeed must return HTTP 200")

        # 2. 2 of 3 succeed -> 200
        mock_run_workers.return_value = [
            self.mock_gemini_success,
            self.mock_openai_success,
            self.mock_claude_failure,
        ]
        r2 = self.client.post("/api/v1/consensus", json=self.valid_payload)
        self.assertEqual(r2.status_code, 200, "2 of 3 succeed must return HTTP 200")

        # 3. 1 of 3 succeeds -> 200
        mock_run_workers.return_value = [
            self.mock_gemini_success,
            self.mock_openai_failure,
            self.mock_claude_failure,
        ]
        r3 = self.client.post("/api/v1/consensus", json=self.valid_payload)
        self.assertEqual(r3.status_code, 200, "1 of 3 succeeds must return HTTP 200")

        # 4. All 3 fail -> 502
        mock_run_workers.return_value = [
            self.mock_gemini_failure,
            self.mock_openai_failure,
            self.mock_claude_failure,
        ]
        r4 = self.client.post("/api/v1/consensus", json=self.valid_payload)
        self.assertEqual(r4.status_code, 502, "All 3 fail must return HTTP 502")

        # 5. Invalid request body -> 422
        r5 = self.client.post("/api/v1/consensus", json={"context": {"role": "claims_adjuster"}})
        self.assertEqual(r5.status_code, 422, "Invalid request body must return HTTP 422")


if __name__ == "__main__":
    unittest.main()



