import asyncio
import os
import unittest
from unittest.mock import AsyncMock, MagicMock, patch

# Ensure test env defaults exist before loading settings
_TEST_ENV_DEFAULTS = {
    "GEMINI_API_KEY": "test-gemini-key",
    "OPENAI_API_KEY": "test-openai-key",
    "ANTHROPIC_API_KEY": "test-anthropic-key",
    "ARBITER_MODEL_PROVIDER": "gemini",
}
for k, v in _TEST_ENV_DEFAULTS.items():
    os.environ.setdefault(k, v)

from google.genai.errors import ClientError, ServerError
import httpx

from app.config import settings
from app.schemas.models import ProviderResult
from app.services.gemini_client import MODEL_NAME, call_gemini, get_gemini_client


class TestGeminiClient(unittest.IsolatedAsyncioTestCase):
    """Unit tests for the Google Gemini async provider client."""

    def setUp(self):
        self.mock_client = MagicMock()
        self.mock_client.aio = MagicMock()
        self.mock_client.aio.models = MagicMock()
        self.mock_client.aio.models.generate_content = AsyncMock()

    async def test_call_gemini_success(self):
        """Verify successful API call returns normalized dict matching ProviderResult schema."""
        mock_response = MagicMock()
        mock_response.text = "This is a detailed analysis of personal auto insurance."
        mock_response.usage_metadata = MagicMock(
            prompt_token_count=65,
            candidates_token_count=180,
        )
        self.mock_client.aio.models.generate_content.return_value = mock_response

        prompt = "Explain liability limits in Montana."
        system_prompt = "You are an expert insurance underwriter."
        result = await call_gemini(
            prompt=prompt,
            system_prompt=system_prompt,
            timeout=12,
            client=self.mock_client,
        )

        self.assertEqual(result["status"], "success")
        self.assertEqual(result["model"], MODEL_NAME)
        self.assertEqual(result["model"], "gemini-3-flash-preview")
        self.assertEqual(
            result["response_text"],
            "This is a detailed analysis of personal auto insurance.",
        )
        self.assertEqual(result["tokens"], {"input": 65, "output": 180})
        self.assertIsInstance(result["duration_seconds"], float)
        self.assertGreater(result["duration_seconds"], 0.0)
        self.assertNotIn("error_message", result)

        # Validate that ProviderResult validates the returned dict
        provider_result = ProviderResult.model_validate(result)
        self.assertEqual(provider_result.status, "success")
        self.assertEqual(provider_result.tokens["input"], 65)
        self.assertEqual(provider_result.tokens["output"], 180)

    async def test_call_gemini_passes_config_correctly(self):
        """Verify system prompt and timeout in milliseconds are forwarded to SDK config."""
        mock_response = MagicMock()
        mock_response.text = "Response text"
        mock_response.usage_metadata = MagicMock(prompt_token_count=10, candidates_token_count=20)
        self.mock_client.aio.models.generate_content.return_value = mock_response

        await call_gemini(
            prompt="Test prompt",
            system_prompt="Test system instruction",
            timeout=15,
            client=self.mock_client,
        )

        self.mock_client.aio.models.generate_content.assert_awaited_once()
        _, kwargs = self.mock_client.aio.models.generate_content.call_args
        self.assertEqual(kwargs["model"], MODEL_NAME)
        self.assertEqual(kwargs["contents"], "Test prompt")
        config = kwargs["config"]
        self.assertEqual(config.system_instruction, "Test system instruction")
        self.assertEqual(config.http_options.timeout, 15000)

    async def test_call_gemini_auth_error_invalid_key(self):
        """Verify invalid API key raises no unhandled exception and returns status='error'."""
        error_json = {
            "error": {
                "code": 400,
                "message": "API key not valid. Please pass a valid API key.",
                "status": "INVALID_ARGUMENT",
                "details": [{"reason": "API_KEY_INVALID"}],
            }
        }
        self.mock_client.aio.models.generate_content.side_effect = ClientError(
            400, error_json, None
        )

        result = await call_gemini(
            prompt="Prompt with invalid key",
            system_prompt="System prompt",
            timeout=12,
            client=self.mock_client,
        )

        self.assertEqual(result["status"], "error")
        self.assertEqual(result["model"], MODEL_NAME)
        self.assertIsNone(result["response_text"])
        self.assertEqual(result["tokens"], {"input": 0, "output": 0})
        self.assertIsInstance(result["duration_seconds"], float)
        self.assertIn("error_message", result)
        self.assertIn("Authentication error", result["error_message"])

        provider_result = ProviderResult.model_validate(result)
        self.assertEqual(provider_result.status, "error")
        self.assertIsNotNone(provider_result.error_message)

    async def test_call_gemini_rate_limit_429(self):
        """Verify HTTP 429 rate limit is captured and formatted clearly."""
        error_json = {
            "error": {
                "code": 429,
                "message": "Resource exhausted: quota exceeded for quota metric.",
                "status": "RESOURCE_EXHAUSTED",
            }
        }
        self.mock_client.aio.models.generate_content.side_effect = ClientError(
            429, error_json, None
        )

        result = await call_gemini(
            prompt="Prompt exceeding quota",
            system_prompt="System prompt",
            timeout=12,
            client=self.mock_client,
        )

        self.assertEqual(result["status"], "error")
        self.assertEqual(result["tokens"], {"input": 0, "output": 0})
        self.assertIsNone(result["response_text"])
        self.assertIn("Rate limit exceeded (HTTP 429)", result["error_message"])

        provider_result = ProviderResult.model_validate(result)
        self.assertEqual(provider_result.status, "error")

    async def test_call_gemini_server_error_500(self):
        """Verify 5xx server error is handled cleanly."""
        error_json = {
            "error": {
                "code": 503,
                "message": "The service is temporarily unavailable.",
                "status": "UNAVAILABLE",
            }
        }
        self.mock_client.aio.models.generate_content.side_effect = ServerError(
            503, error_json, None
        )

        result = await call_gemini(
            prompt="Test prompt",
            system_prompt="System prompt",
            timeout=12,
            client=self.mock_client,
        )

        self.assertEqual(result["status"], "error")
        self.assertIn("Gemini server error (503)", result["error_message"])

    async def test_call_gemini_httpx_timeout(self):
        """Verify httpx.TimeoutException is captured as a request timeout."""
        self.mock_client.aio.models.generate_content.side_effect = httpx.ReadTimeout(
            "Connection timed out while reading response"
        )

        result = await call_gemini(
            prompt="Prompt timing out",
            system_prompt="System prompt",
            timeout=12,
            client=self.mock_client,
        )

        self.assertEqual(result["status"], "error")
        self.assertIn("Request timed out", result["error_message"])

    async def test_call_gemini_asyncio_timeout(self):
        """Verify asyncio.TimeoutError is normalized to a request timeout."""
        self.mock_client.aio.models.generate_content.side_effect = asyncio.TimeoutError()

        result = await call_gemini(
            prompt="Prompt timing out",
            system_prompt="System prompt",
            timeout=12,
            client=self.mock_client,
        )

        self.assertEqual(result["status"], "error")
        self.assertIn("Request timed out", result["error_message"])

    async def test_call_gemini_network_error(self):
        """Verify connection and network errors are normalized."""
        self.mock_client.aio.models.generate_content.side_effect = httpx.ConnectError(
            "Failed to resolve host"
        )

        result = await call_gemini(
            prompt="Prompt with connection fail",
            system_prompt="System prompt",
            timeout=12,
            client=self.mock_client,
        )

        self.assertEqual(result["status"], "error")
        self.assertIn("Network error", result["error_message"])

    async def test_call_gemini_missing_api_key(self):
        """Verify calling without configured API key returns an error dict without crashing."""
        with patch.object(settings, "gemini_api_key", ""):
            result = await call_gemini(
                prompt="Prompt without key",
                system_prompt="System prompt",
                timeout=12,
            )

            self.assertEqual(result["status"], "error")
            self.assertEqual(result["tokens"], {"input": 0, "output": 0})
            self.assertIsNone(result["response_text"])
            self.assertIn("Authentication error", result["error_message"])

    async def test_call_gemini_empty_usage_metadata(self):
        """Verify tokens default to 0 if usage_metadata is missing."""
        mock_response = MagicMock()
        mock_response.text = "Answer without usage metadata"
        mock_response.usage_metadata = None
        self.mock_client.aio.models.generate_content.return_value = mock_response

        result = await call_gemini(
            prompt="Prompt",
            system_prompt="System",
            timeout=12,
            client=self.mock_client,
        )

        self.assertEqual(result["status"], "success")
        self.assertEqual(result["tokens"], {"input": 0, "output": 0})
        self.assertEqual(result["response_text"], "Answer without usage metadata")

    async def test_call_gemini_none_response_text(self):
        """Verify response_text defaults to empty string if response.text is None."""
        mock_response = MagicMock()
        mock_response.text = None
        mock_response.usage_metadata = MagicMock(prompt_token_count=10, candidates_token_count=0)
        self.mock_client.aio.models.generate_content.return_value = mock_response

        result = await call_gemini(
            prompt="Prompt",
            system_prompt="System",
            timeout=12,
            client=self.mock_client,
        )

        self.assertEqual(result["status"], "success")
        self.assertEqual(result["response_text"], "")

    def test_get_gemini_client_custom_key(self):
        """Verify get_gemini_client accepts a custom API key."""
        client = get_gemini_client(api_key="custom-valid-key")
        self.assertIsNotNone(client)

    def test_get_gemini_client_empty_raises(self):
        """Verify get_gemini_client raises ValueError when key is empty."""
        with self.assertRaises(ValueError):
            get_gemini_client(api_key="")


if __name__ == "__main__":
    unittest.main()
