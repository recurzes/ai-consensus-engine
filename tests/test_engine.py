import asyncio
import os
import time
import unittest
from unittest.mock import AsyncMock, patch

# Ensure test env defaults exist before loading settings
_TEST_ENV_DEFAULTS = {
    "GEMINI_API_KEY": "test-gemini-key",
    "OPENAI_API_KEY": "test-openai-key",
    "ANTHROPIC_API_KEY": "test-anthropic-key",
    "ARBITER_MODEL_PROVIDER": "gemini",
}
for k, v in _TEST_ENV_DEFAULTS.items():
    os.environ.setdefault(k, v)

from app.config import settings
from app.schemas.models import ProviderResult
from app.services.orchestrator import run_workers


class TestWorkerOrchestration(unittest.IsolatedAsyncioTestCase):
    """Unit and concurrency tests for run_workers orchestration."""

    def setUp(self):
        self.prompt = "Summarize insurance policy coverage."
        self.system_prompt = "You are an expert insurance analyst."

        self.mock_gemini_success = {
            "status": "success",
            "model": "gemini-2.5-flash",
            "duration_seconds": 0.12,
            "tokens": {"input": 15, "output": 25},
            "response_text": "Gemini response text",
        }
        self.mock_openai_success = {
            "status": "success",
            "model": "gpt-4o-mini",
            "duration_seconds": 0.18,
            "tokens": {"input": 18, "output": 30},
            "response_text": "OpenAI response text",
        }
        self.mock_claude_success = {
            "status": "success",
            "model": "claude-3-5-haiku",
            "duration_seconds": 0.22,
            "tokens": {"input": 14, "output": 28},
            "response_text": "Claude response text",
        }

    @patch("app.services.orchestrator.call_claude")
    @patch("app.services.orchestrator.call_openai")
    @patch("app.services.orchestrator.call_gemini")
    async def test_run_workers_all_success_returns_deterministic_order(
        self, mock_gemini, mock_openai, mock_claude
    ):
        """Verify run_workers returns exactly 3 results in [gemini, openai, claude] order."""
        mock_gemini.return_value = self.mock_gemini_success
        mock_openai.return_value = self.mock_openai_success
        mock_claude.return_value = self.mock_claude_success

        results = await run_workers(self.prompt, self.system_prompt)

        self.assertIsInstance(results, list)
        self.assertEqual(len(results), 3)

        # Ordering guarantee: [gemini, openai, claude]
        self.assertEqual(results[0]["model"], "gemini-2.5-flash")
        self.assertEqual(results[1]["model"], "gpt-4o-mini")
        self.assertEqual(results[2]["model"], "claude-3-5-haiku")

        # Conforms to ProviderResult schema
        for res in results:
            validated = ProviderResult(**res)
            self.assertEqual(validated.status, "success")

    @patch("app.services.orchestrator.call_claude")
    @patch("app.services.orchestrator.call_openai")
    @patch("app.services.orchestrator.call_gemini")
    async def test_run_workers_latency_approximates_slowest_provider(
        self, mock_gemini, mock_openai, mock_claude
    ):
        """Verify total latency approximates slowest provider, not the cumulative sum."""
        async def slow_gemini(*args, **kwargs):
            await asyncio.sleep(0.1)
            return self.mock_gemini_success

        async def slow_openai(*args, **kwargs):
            await asyncio.sleep(0.2)
            return self.mock_openai_success

        async def slow_claude(*args, **kwargs):
            await asyncio.sleep(0.4)  # slowest
            return self.mock_claude_success

        mock_gemini.side_effect = slow_gemini
        mock_openai.side_effect = slow_openai
        mock_claude.side_effect = slow_claude

        start_time = time.perf_counter()
        results = await run_workers(self.prompt, self.system_prompt)
        elapsed = time.perf_counter() - start_time

        self.assertEqual(len(results), 3)
        # Sum would be ~0.7s (0.1 + 0.2 + 0.4).
        # Concurrent execution should be ~0.4s (well below 0.6s).
        self.assertGreaterEqual(elapsed, 0.35, f"Expected elapsed >= 0.35s, got {elapsed:.4f}s")
        self.assertLess(elapsed, 0.60, f"Expected elapsed < 0.60s (not sum ~0.7s), got {elapsed:.4f}s")

    @patch("app.services.orchestrator.call_claude")
    @patch("app.services.orchestrator.call_openai")
    @patch("app.services.orchestrator.call_gemini")
    async def test_run_workers_exception_isolation_with_return_exceptions(
        self, mock_gemini, mock_openai, mock_claude
    ):
        """Verify return_exceptions=True isolates bare exceptions so other tasks still complete."""
        mock_gemini.return_value = self.mock_gemini_success
        mock_openai.side_effect = RuntimeError("OpenAI unexpected crash")
        mock_claude.return_value = self.mock_claude_success

        results = await run_workers(self.prompt, self.system_prompt)

        self.assertEqual(len(results), 3)
        # Gemini succeeded
        self.assertEqual(results[0]["status"], "success")
        self.assertEqual(results[0]["model"], "gemini-2.5-flash")

        # OpenAI exception captured in list without raising
        self.assertIsInstance(results[1], RuntimeError)
        self.assertEqual(str(results[1]), "OpenAI unexpected crash")

        # Claude succeeded
        self.assertEqual(results[2]["status"], "success")
        self.assertEqual(results[2]["model"], "claude-3-5-haiku")

    @patch("app.services.orchestrator.call_claude")
    @patch("app.services.orchestrator.call_openai")
    @patch("app.services.orchestrator.call_gemini")
    async def test_run_workers_normalized_error_result_handled(
        self, mock_gemini, mock_openai, mock_claude
    ):
        """Verify standard error dict from provider client is returned cleanly."""
        mock_gemini.return_value = self.mock_gemini_success
        mock_openai.return_value = self.mock_openai_success
        mock_claude.return_value = {
            "status": "error",
            "model": "claude-3-5-haiku",
            "duration_seconds": 0.05,
            "tokens": {"input": 0, "output": 0},
            "response_text": None,
            "error_message": "Rate limit exceeded (HTTP 429)",
        }

        results = await run_workers(self.prompt, self.system_prompt)

        self.assertEqual(len(results), 3)
        self.assertEqual(results[0]["status"], "success")
        self.assertEqual(results[1]["status"], "success")
        self.assertEqual(results[2]["status"], "error")
        self.assertEqual(results[2]["error_message"], "Rate limit exceeded (HTTP 429)")

    @patch("app.services.orchestrator.call_claude")
    @patch("app.services.orchestrator.call_openai")
    @patch("app.services.orchestrator.call_gemini")
    async def test_run_workers_timeout_forwarding(
        self, mock_gemini, mock_openai, mock_claude
    ):
        """Verify timeout defaults to settings.request_timeout_seconds and respects override."""
        mock_gemini.return_value = self.mock_gemini_success
        mock_openai.return_value = self.mock_openai_success
        mock_claude.return_value = self.mock_claude_success

        # Default timeout
        await run_workers(self.prompt, self.system_prompt)
        mock_gemini.assert_called_with(self.prompt, self.system_prompt, settings.request_timeout_seconds)
        mock_openai.assert_called_with(self.prompt, self.system_prompt, settings.request_timeout_seconds)
        mock_claude.assert_called_with(self.prompt, self.system_prompt, settings.request_timeout_seconds)

        # Explicit timeout override
        mock_gemini.reset_mock()
        mock_openai.reset_mock()
        mock_claude.reset_mock()

        await run_workers(self.prompt, self.system_prompt, timeout=25)
        mock_gemini.assert_called_with(self.prompt, self.system_prompt, 25)
        mock_openai.assert_called_with(self.prompt, self.system_prompt, 25)
        mock_claude.assert_called_with(self.prompt, self.system_prompt, 25)

    @patch("app.services.orchestrator.call_claude")
    @patch("app.services.orchestrator.call_openai")
    @patch("app.services.orchestrator.call_gemini")
    async def test_run_workers_client_kwargs_passthrough(
        self, mock_gemini, mock_openai, mock_claude
    ):
        """Verify optional client instances are forwarded to respective call functions."""
        mock_gemini.return_value = self.mock_gemini_success
        mock_openai.return_value = self.mock_openai_success
        mock_claude.return_value = self.mock_claude_success

        fake_gemini_client = object()
        fake_openai_client = object()
        fake_claude_client = object()

        await run_workers(
            self.prompt,
            self.system_prompt,
            gemini_client=fake_gemini_client,
            openai_client=fake_openai_client,
            claude_client=fake_claude_client,
        )

        mock_gemini.assert_called_with(
            self.prompt,
            self.system_prompt,
            settings.request_timeout_seconds,
            client=fake_gemini_client,
        )
        mock_openai.assert_called_with(
            self.prompt,
            self.system_prompt,
            settings.request_timeout_seconds,
            client=fake_openai_client,
        )
        mock_claude.assert_called_with(
            self.prompt,
            self.system_prompt,
            settings.request_timeout_seconds,
            client=fake_claude_client,
        )

    @patch("app.services.orchestrator.call_claude")
    @patch("app.services.orchestrator.call_openai")
    @patch("app.services.orchestrator.call_gemini")
    async def test_run_workers_single_provider_timeout_returns_normalized_error_and_preserves_others(
        self, mock_gemini, mock_openai, mock_claude
    ):
        """Verify slow provider times out into normalized error dict without impacting others."""
        async def slow_openai(*args, **kwargs):
            await asyncio.sleep(1.0)
            return self.mock_openai_success

        mock_gemini.return_value = self.mock_gemini_success
        mock_openai.side_effect = slow_openai
        mock_claude.return_value = self.mock_claude_success

        results = await run_workers(self.prompt, self.system_prompt, timeout=0.05)

        self.assertEqual(len(results), 3)

        # Gemini succeeded
        self.assertEqual(results[0]["status"], "success")
        self.assertEqual(results[0]["model"], "gemini-2.5-flash")

        # OpenAI timed out
        openai_res = results[1]
        self.assertEqual(openai_res["status"], "error")
        self.assertEqual(openai_res["model"], "gpt-4o-mini")
        self.assertEqual(openai_res["duration_seconds"], 0.05)
        self.assertEqual(openai_res["tokens"], {"input": 0, "output": 0})
        self.assertIsNone(openai_res["response_text"])
        self.assertIn("timed out", openai_res["error_message"])
        self.assertEqual(openai_res["error_message"], "Request timed out after 0.05 seconds")

        # Claude succeeded
        self.assertEqual(results[2]["status"], "success")
        self.assertEqual(results[2]["model"], "claude-3-5-haiku")

        # Conforms to ProviderResult schema
        for res in results:
            validated = ProviderResult(**res)
            self.assertIn(validated.status, ("success", "error"))

    @patch("app.services.orchestrator.call_claude")
    @patch("app.services.orchestrator.call_openai")
    @patch("app.services.orchestrator.call_gemini")
    async def test_run_workers_all_providers_timeout(
        self, mock_gemini, mock_openai, mock_claude
    ):
        """Verify gather still completes and returns 3 error results when all providers time out."""
        async def hang(*args, **kwargs):
            await asyncio.sleep(1.0)

        mock_gemini.side_effect = hang
        mock_openai.side_effect = hang
        mock_claude.side_effect = hang

        results = await run_workers(self.prompt, self.system_prompt, timeout=0.05)

        self.assertEqual(len(results), 3)
        expected_models = ["gemini-2.5-flash", "gpt-4o-mini", "claude-3-5-haiku"]
        for res, expected_model in zip(results, expected_models):
            self.assertEqual(res["status"], "error")
            self.assertEqual(res["model"], expected_model)
            self.assertEqual(res["duration_seconds"], 0.05)
            self.assertEqual(res["tokens"], {"input": 0, "output": 0})
            self.assertIsNone(res["response_text"])
            self.assertIn("timed out", res["error_message"])
            self.assertEqual(res["error_message"], "Request timed out after 0.05 seconds")
            # Conforms to ProviderResult schema
            validated = ProviderResult(**res)
            self.assertEqual(validated.status, "error")

    @patch("app.services.orchestrator.call_claude")
    @patch("app.services.orchestrator.call_openai")
    @patch("app.services.orchestrator.call_gemini")
    async def test_run_workers_timeout_cancels_hanging_task(
        self, mock_gemini, mock_openai, mock_claude
    ):
        """Verify that asyncio.wait_for cancels the hanging coroutine when timing out."""
        task_was_cancelled = False

        async def hanging_gemini(*args, **kwargs):
            nonlocal task_was_cancelled
            try:
                await asyncio.sleep(1.0)
            except asyncio.CancelledError:
                task_was_cancelled = True
                raise

        mock_gemini.side_effect = hanging_gemini
        mock_openai.return_value = self.mock_openai_success
        mock_claude.return_value = self.mock_claude_success

        results = await run_workers(self.prompt, self.system_prompt, timeout=0.05)

        self.assertTrue(task_was_cancelled)
        self.assertEqual(results[0]["status"], "error")
        self.assertEqual(results[0]["model"], "gemini-2.5-flash")

    @patch("app.services.orchestrator.settings")
    @patch("app.services.orchestrator.call_claude")
    @patch("app.services.orchestrator.call_openai")
    @patch("app.services.orchestrator.call_gemini")
    async def test_run_workers_timeout_defaults_to_settings(
        self, mock_gemini, mock_openai, mock_claude, mock_settings
    ):
        """Verify timeout defaults to settings.request_timeout_seconds."""
        mock_settings.request_timeout_seconds = 0.05

        async def hang(*args, **kwargs):
            await asyncio.sleep(1.0)

        mock_gemini.side_effect = hang
        mock_openai.return_value = self.mock_openai_success
        mock_claude.return_value = self.mock_claude_success

        results = await run_workers(self.prompt, self.system_prompt)

        self.assertEqual(results[0]["status"], "error")
        self.assertEqual(results[0]["duration_seconds"], 0.05)
        self.assertEqual(
            results[0]["error_message"],
            "Request timed out after 0.05 seconds",
        )

    @patch("app.services.orchestrator.call_claude")
    @patch("app.services.orchestrator.call_openai")
    @patch("app.services.orchestrator.call_gemini")
    async def test_run_workers_internal_timeout_error_caught(
        self, mock_gemini, mock_openai, mock_claude
    ):
        """Verify explicit asyncio.TimeoutError raised from inside provider call is caught."""
        mock_gemini.side_effect = asyncio.TimeoutError()
        mock_openai.return_value = self.mock_openai_success
        mock_claude.return_value = self.mock_claude_success

        results = await run_workers(self.prompt, self.system_prompt, timeout=12)

        self.assertEqual(len(results), 3)
        self.assertEqual(results[0]["status"], "error")
        self.assertEqual(results[0]["model"], "gemini-2.5-flash")
        self.assertEqual(results[0]["duration_seconds"], 12.0)
        self.assertEqual(results[0]["tokens"], {"input": 0, "output": 0})
        self.assertIsNone(results[0]["response_text"])
        self.assertEqual(results[0]["error_message"], "Request timed out after 12 seconds")


if __name__ == "__main__":
    unittest.main()

