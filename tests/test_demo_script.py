import io
import os
import sys
import unittest
from unittest.mock import MagicMock, patch

import httpx

import run_demo


class TestDemoScript(unittest.TestCase):
    """Test suite for run_demo.py CLI script and utilities."""

    def setUp(self):
        self.valid_env = {
            "GEMINI_API_KEY": "dummy-gemini-key",
            "OPENAI_API_KEY": "dummy-openai-key",
            "ANTHROPIC_API_KEY": "dummy-anthropic-key",
            "ARBITER_MODEL_PROVIDER": "gemini",
        }

    def test_format_telemetry_populated(self):
        """Test formatting telemetry with populated fields."""
        telemetry = {
            "total_duration_seconds": 3.1234,
            "total_estimated_cost_usd": 0.00142,
            "successful_providers": ["gemini", "openai", "claude"],
            "failed_providers": [],
        }
        output = run_demo.format_telemetry(telemetry)
        expected = (
            "Telemetry:\n"
            "  Total Duration:  3.12s\n"
            "  Estimated Cost:  $0.00142\n"
            "  Successful:      [gemini, openai, claude]\n"
            "  Failed:          []"
        )
        self.assertEqual(output, expected)

    def test_format_telemetry_with_failed_providers(self):
        """Test formatting telemetry when some providers failed."""
        telemetry = {
            "total_duration_seconds": 2.50,
            "total_estimated_cost_usd": 0.00085,
            "successful_providers": ["gemini", "openai"],
            "failed_providers": ["claude"],
        }
        output = run_demo.format_telemetry(telemetry)
        expected = (
            "Telemetry:\n"
            "  Total Duration:  2.50s\n"
            "  Estimated Cost:  $0.00085\n"
            "  Successful:      [gemini, openai]\n"
            "  Failed:          [claude]"
        )
        self.assertEqual(output, expected)

    def test_format_telemetry_empty(self):
        """Test formatting telemetry with default/empty values."""
        output = run_demo.format_telemetry({})
        expected = (
            "Telemetry:\n"
            "  Total Duration:  0.00s\n"
            "  Estimated Cost:  $0.00000\n"
            "  Successful:      []\n"
            "  Failed:          []"
        )
        self.assertEqual(output, expected)

    def test_validate_environment_success(self):
        """Test validate_environment passes with valid configuration."""
        with patch.dict(os.environ, self.valid_env, clear=True):
            self.assertTrue(run_demo.validate_environment(exit_on_error=False, load_env=False))

    def test_validate_environment_case_insensitive_arbiter(self):
        """Test validate_environment accepts uppercase or mixed case arbiter provider."""
        env = dict(self.valid_env)
        env["ARBITER_MODEL_PROVIDER"] = "OPENAI"
        with patch.dict(os.environ, env, clear=True):
            self.assertTrue(run_demo.validate_environment(exit_on_error=False, load_env=False))

    def test_validate_environment_missing_key_non_exit(self):
        """Test validate_environment returns False when exit_on_error=False on missing key."""
        env = dict(self.valid_env)
        del env["GEMINI_API_KEY"]
        with patch.dict(os.environ, env, clear=True), patch("sys.stderr", new_callable=io.StringIO):
            self.assertFalse(run_demo.validate_environment(exit_on_error=False, load_env=False))

    def test_validate_environment_empty_key_sys_exit(self):
        """Test validate_environment calls sys.exit(1) on empty key when exit_on_error=True."""
        env = dict(self.valid_env)
        env["OPENAI_API_KEY"] = "   "
        with patch.dict(os.environ, env, clear=True), patch("sys.stderr", new_callable=io.StringIO):
            with self.assertRaises(SystemExit) as ctx:
                run_demo.validate_environment(exit_on_error=True, load_env=False)
            self.assertEqual(ctx.exception.code, 1)

    def test_validate_environment_invalid_arbiter_sys_exit(self):
        """Test validate_environment calls sys.exit(1) on invalid arbiter provider."""
        env = dict(self.valid_env)
        env["ARBITER_MODEL_PROVIDER"] = "unsupported_provider"
        with patch.dict(os.environ, env, clear=True), patch("sys.stderr", new_callable=io.StringIO):
            with self.assertRaises(SystemExit) as ctx:
                run_demo.validate_environment(exit_on_error=True, load_env=False)
            self.assertEqual(ctx.exception.code, 1)

    @patch("run_demo.load_dotenv")
    def test_validate_environment_calls_load_dotenv(self, mock_load_dotenv):
        """Test validate_environment calls load_dotenv when load_env=True."""
        with patch.dict(os.environ, self.valid_env, clear=True):
            run_demo.validate_environment(exit_on_error=False, load_env=True)
            mock_load_dotenv.assert_called_once_with(override=False)

    @patch("run_demo.httpx.Client")
    def test_run_scenario_success(self, mock_client_cls):
        """Test run_scenario with a successful HTTP 200 response."""
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "status": "success",
            "consensus_answer": "This is the synthesized answer.",
            "telemetry": {
                "total_duration_seconds": 1.23,
                "total_estimated_cost_usd": 0.0005,
                "successful_providers": ["gemini", "openai"],
                "failed_providers": ["claude"],
            },
        }

        mock_client = MagicMock()
        mock_client.post.return_value = mock_response
        mock_client_cls.return_value.__enter__.return_value = mock_client

        payload = {
            "prompt": "Test query",
            "context": {
                "role": "layman_linguist",
                "line_of_business": "homeowners",
                "state": "MT",
            },
        }

        with patch("sys.stdout", new_callable=io.StringIO) as mock_stdout:
            result = run_demo.run_scenario("Scenario 1: Test", payload)

        self.assertIsNotNone(result)
        self.assertEqual(result["consensus_answer"], "This is the synthesized answer.")
        output = mock_stdout.getvalue()
        self.assertIn("Scenario 1: Test", output)
        self.assertIn("Prompt: Test query", output)
        self.assertIn("Consensus Answer:", output)
        self.assertIn("This is the synthesized answer.", output)
        self.assertIn("Total Duration:  1.23s", output)
        self.assertIn("Estimated Cost:  $0.00050", output)

    @patch("run_demo.httpx.Client")
    def test_run_scenario_connection_error(self, mock_client_cls):
        """Test run_scenario cleanly catches httpx.ConnectError without raising."""
        mock_client = MagicMock()
        mock_client.post.side_effect = httpx.ConnectError("Connection refused")
        mock_client_cls.return_value.__enter__.return_value = mock_client

        payload = {"prompt": "Hello"}
        with patch("sys.stdout", new_callable=io.StringIO), patch("sys.stderr", new_callable=io.StringIO) as mock_stderr:
            result = run_demo.run_scenario("Scenario 1: Test", payload)

        self.assertIsNone(result)
        output = mock_stderr.getvalue()
        self.assertIn("Could not connect to API server", output)
        self.assertIn("uvicorn app.main:app --reload", output)

    @patch("run_demo.httpx.Client")
    def test_run_scenario_timeout_error(self, mock_client_cls):
        """Test run_scenario cleanly catches httpx.TimeoutException without raising."""
        mock_client = MagicMock()
        mock_client.post.side_effect = httpx.TimeoutException("Timed out")
        mock_client_cls.return_value.__enter__.return_value = mock_client

        payload = {"prompt": "Hello"}
        with patch("sys.stdout", new_callable=io.StringIO), patch("sys.stderr", new_callable=io.StringIO) as mock_stderr:
            result = run_demo.run_scenario("Scenario 1: Test", payload)

        self.assertIsNone(result)
        output = mock_stderr.getvalue()
        self.assertIn("Request timed out", output)

    @patch("run_demo.httpx.Client")
    def test_run_scenario_non_200_status(self, mock_client_cls):
        """Test run_scenario handles non-200 responses cleanly."""
        mock_response = MagicMock()
        mock_response.status_code = 502
        mock_response.text = '{"status": "error", "message": "All providers failed"}'

        mock_client = MagicMock()
        mock_client.post.return_value = mock_response
        mock_client_cls.return_value.__enter__.return_value = mock_client

        payload = {"prompt": "Hello"}
        with patch("sys.stdout", new_callable=io.StringIO), patch("sys.stderr", new_callable=io.StringIO) as mock_stderr:
            result = run_demo.run_scenario("Scenario 1: Test", payload)

        self.assertIsNone(result)
        output = mock_stderr.getvalue()
        self.assertIn("API request failed with status HTTP 502", output)

    @patch("run_demo.validate_environment")
    def test_main_with_no_scenarios(self, mock_validate):
        """Test main runs validation and displays scaffold message when no scenarios."""
        with patch("run_demo.SCENARIOS", []):
            with patch("sys.stdout", new_callable=io.StringIO) as mock_stdout:
                run_demo.main()

            mock_validate.assert_called_once_with(exit_on_error=True)
            output = mock_stdout.getvalue()
            self.assertIn("AI Consensus Engine - Demo Runner", output)
            self.assertIn("No demo scenarios registered yet", output)

    @patch("run_demo.validate_environment")
    @patch("run_demo.run_scenario")
    def test_main_with_registered_scenarios(self, mock_run_scenario, mock_validate):
        """Test main iterates over registered scenarios."""
        sample_scenarios = [
            ("Scenario 1", {"prompt": "q1"}),
            ("Scenario 2", {"prompt": "q2"}),
        ]
        with patch("run_demo.SCENARIOS", sample_scenarios):
            run_demo.main()

            mock_validate.assert_called_once_with(exit_on_error=True)
            self.assertEqual(mock_run_scenario.call_count, 2)
            mock_run_scenario.assert_any_call("Scenario 1", {"prompt": "q1"})
            mock_run_scenario.assert_any_call("Scenario 2", {"prompt": "q2"})

    def test_scenario_1_structure_and_schema_validation(self):
        """Test SCENARIO_1 payload strictly conforms to ConsensusRequest schema."""
        from app.schemas.models import ConsensusRequest

        req = ConsensusRequest(**run_demo.SCENARIO_1)
        self.assertIn("Ordinance or Law", req.prompt)
        self.assertEqual(req.context.role, "layman_linguist")
        self.assertEqual(req.context.line_of_business, "homeowners")
        self.assertEqual(req.context.state, "MT")

    def test_scenario_1_registered_in_scenarios(self):
        """Test SCENARIO_1 is registered in the SCENARIOS registry list."""
        self.assertGreaterEqual(len(run_demo.SCENARIOS), 1)
        name, payload = run_demo.SCENARIOS[0]
        self.assertEqual(name, "Scenario 1: Consumer Translation (layman_linguist)")
        self.assertEqual(payload, run_demo.SCENARIO_1)

    @patch("run_demo.validate_environment")
    @patch("run_demo.run_scenario")
    def test_main_executes_scenario_1_by_default(self, mock_run_scenario, mock_validate):
        """Test main executes SCENARIO_1 by default."""
        run_demo.main()
        mock_validate.assert_called_once_with(exit_on_error=True)
        self.assertGreaterEqual(mock_run_scenario.call_count, 1)
        mock_run_scenario.assert_any_call(
            "Scenario 1: Consumer Translation (layman_linguist)",
            run_demo.SCENARIO_1,
        )

    def test_scenario_2_structure_and_schema_validation(self):
        """Test SCENARIO_2 payload strictly conforms to ConsensusRequest schema."""
        from app.schemas.models import ConsensusRequest

        req = ConsensusRequest(**run_demo.SCENARIO_2)
        self.assertIn("trailer", req.prompt)
        self.assertIn("garage door", req.prompt)
        self.assertEqual(req.context.role, "claims_adjuster")
        self.assertEqual(req.context.line_of_business, "homeowners")
        self.assertEqual(req.context.state, "MT")

    def test_scenario_2_registered_in_scenarios(self):
        """Test SCENARIO_2 is registered in the SCENARIOS registry list at index 1."""
        self.assertGreaterEqual(len(run_demo.SCENARIOS), 2)
        name, payload = run_demo.SCENARIOS[1]
        self.assertEqual(
            name, "Scenario 2: Underwriting / Coverage Analysis (claims_adjuster, MT)"
        )
        self.assertEqual(payload, run_demo.SCENARIO_2)

    @patch("run_demo.validate_environment")
    @patch("run_demo.run_scenario")
    def test_main_executes_scenarios_in_order(self, mock_run_scenario, mock_validate):
        """Test main executes all registered scenarios in sequential order."""
        run_demo.main()
        mock_validate.assert_called_once_with(exit_on_error=True)
        self.assertEqual(mock_run_scenario.call_count, len(run_demo.SCENARIOS))

        expected_calls = [
            (
                (name, payload),
            )
            for name, payload in run_demo.SCENARIOS
        ]
        self.assertEqual(
            [call.args for call in mock_run_scenario.call_args_list],
            [args[0] for args in expected_calls],
        )


    def test_scenario_3_structure_and_schema_validation(self):
        """Test SCENARIO_3 payload strictly conforms to ConsensusRequest schema."""
        from app.schemas.models import ConsensusRequest

        req = ConsensusRequest(**run_demo.SCENARIO_3)
        self.assertIn("BOP", req.prompt)
        self.assertIn("Montana", req.prompt)
        self.assertEqual(req.context.role, "underwriter")
        self.assertEqual(req.context.line_of_business, "commercial_pnc")
        self.assertEqual(req.context.state, "MT")

    def test_scenario_3_registered_in_scenarios(self):
        """Test SCENARIO_3 is registered in the SCENARIOS registry list at index 2."""
        self.assertGreaterEqual(len(run_demo.SCENARIOS), 3)
        name, payload = run_demo.SCENARIOS[2]
        self.assertEqual(
            name, "Scenario 3: Failover Verification (Claude API key corrupted)"
        )
        self.assertEqual(payload, run_demo.SCENARIO_3)

    @patch("run_demo.httpx.Client")
    def test_run_scenario_failover_banner_and_verification(self, mock_client_cls):
        """Test run_scenario with is_failover=True prints corruption banner and verification block."""
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "status": "success",
            "consensus_answer": "BOP provides bundled coverage for Montana small businesses.",
            "telemetry": {
                "total_duration_seconds": 2.87,
                "total_estimated_cost_usd": 0.00098,
                "successful_providers": ["gemini", "openai"],
                "failed_providers": ["claude"],
            },
        }

        mock_client = MagicMock()
        mock_client.post.return_value = mock_response
        mock_client_cls.return_value.__enter__.return_value = mock_client

        with patch("sys.stdout", new_callable=io.StringIO) as mock_stdout:
            result = run_demo.run_scenario(
                "Scenario 3: Failover Verification (Claude API key corrupted)",
                run_demo.SCENARIO_3,
                headers={"x-anthropic-api-key": "sk-intentionally-invalid-for-failover-test"},
                is_failover=True,
            )

        self.assertIsNotNone(result)
        output = mock_stdout.getvalue()
        self.assertIn(
            "[!] Intentionally corrupting ANTHROPIC_API_KEY to simulate provider failure...",
            output,
        )
        self.assertIn("BOP provides bundled coverage", output)
        self.assertIn("Successful:      [gemini, openai]", output)
        self.assertIn("Failed:          [claude]", output)
        self.assertIn(
            "✓ FAILOVER VERIFIED: App returned a consensus answer despite one provider failure.",
            output,
        )
        self.assertIn("✓ Failed provider: claude", output)
        self.assertIn("✓ Successful providers: [gemini, openai]", output)
        self.assertIn(
            "✓ The application did NOT crash or raise an unhandled exception.", output
        )

    @patch("run_demo.validate_environment")
    @patch("run_demo.run_scenario")
    def test_main_corrupts_and_restores_anthropic_api_key(
        self, mock_run_scenario, mock_validate
    ):
        """Test main corrupts ANTHROPIC_API_KEY during Scenario 3 and restores original key afterwards."""
        captured_keys: list[str | None] = []

        def side_effect(name, payload, headers=None, is_failover=False):
            if is_failover:
                captured_keys.append(os.environ.get("ANTHROPIC_API_KEY"))
                self.assertEqual(
                    headers,
                    {"x-anthropic-api-key": "sk-intentionally-invalid-for-failover-test"},
                )
            return {"status": "success"}

        mock_run_scenario.side_effect = side_effect

        with patch.dict(os.environ, {"ANTHROPIC_API_KEY": "original-real-key"}):
            run_demo.main()
            self.assertEqual(os.environ.get("ANTHROPIC_API_KEY"), "original-real-key")

        self.assertEqual(len(captured_keys), 1)
        self.assertEqual(
            captured_keys[0], "sk-intentionally-invalid-for-failover-test"
        )


if __name__ == "__main__":
    unittest.main()
