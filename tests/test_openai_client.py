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

import httpx
import openai

from app.config import settings
from app.schemas.models import ProviderResult
from app.services.gemini_client import call_gemini
from app.services.openai_client import MODEL_NAME, call_openai, get_openai_client


class TestOpenAIClient(unittest.IsolatedAsyncioTestCase):
    """Unit tests for the OpenAI async provider client."""

    def setUp(self):
        self.mock_client = MagicMock()
        self.mock_client.chat = MagicMock()
        self.mock_client.chat.completions = MagicMock()
        self.mock_client.chat.completions.create = AsyncMock()

    async def test_call_openai_success(self):
        """Verify successful API call returns normalized dict matching ProviderResult schema."""
        mock_response = MagicMock()
        mock_choice = MagicMock()
        mock_choice.message = MagicMock(content="This is a comprehensive OpenAI response.")
        mock_response.choices = [mock_choice]
        mock_response.usage = MagicMock(
            prompt_tokens=55,
            completion_tokens=140,
        )
        self.mock_client.chat.completions.create.return_value = mock_response

        prompt = "Explain liability coverage under personal auto policies."
        system_prompt = "You are an expert insurance underwriter."
        result = await call_openai(
            prompt=prompt,
            system_prompt=system_prompt,
            timeout=12,
            client=self.mock_client,
        )

        self.assertEqual(result["status"], "success")
        self.assertEqual(result["model"], MODEL_NAME)
        self.assertEqual(result["model"], "gpt-4o-mini")
        self.assertEqual(
            result["response_text"],
            "This is a comprehensive OpenAI response.",
        )
        self.assertEqual(result["tokens"], {"input": 55, "output": 140})
        self.assertIsInstance(result["duration_seconds"], float)
        self.assertGreaterEqual(result["duration_seconds"], 0.0)
        self.assertNotIn("error_message", result)

        # Validate that ProviderResult validates the returned dict
        provider_result = ProviderResult.model_validate(result)
        self.assertEqual(provider_result.status, "success")
        self.assertEqual(provider_result.model, "gpt-4o-mini")
        self.assertEqual(provider_result.tokens["input"], 55)
        self.assertEqual(provider_result.tokens["output"], 140)
        self.assertEqual(
            provider_result.response_text,
            "This is a comprehensive OpenAI response.",
        )
        self.assertIsNone(provider_result.error_message)

    async def test_call_openai_passes_parameters_correctly(self):
        """Verify model, messages format (system + user), and timeout are forwarded to SDK."""
        mock_response = MagicMock()
        mock_response.choices = [MagicMock(message=MagicMock(content="Response"))]
        mock_response.usage = MagicMock(prompt_tokens=10, completion_tokens=20)
        self.mock_client.chat.completions.create.return_value = mock_response

        await call_openai(
            prompt="Test prompt",
            system_prompt="Test system instruction",
            timeout=15,
            client=self.mock_client,
        )

        self.mock_client.chat.completions.create.assert_awaited_once()
        _, kwargs = self.mock_client.chat.completions.create.call_args
        self.assertEqual(kwargs["model"], "gpt-4o-mini")
        self.assertEqual(kwargs["timeout"], 15.0)
        self.assertEqual(
            kwargs["messages"],
            [
                {"role": "system", "content": "Test system instruction"},
                {"role": "user", "content": "Test prompt"},
            ],
        )

    async def test_call_openai_without_system_prompt(self):
        """Verify when system prompt is empty, only user message is sent."""
        mock_response = MagicMock()
        mock_response.choices = [MagicMock(message=MagicMock(content="Response"))]
        mock_response.usage = MagicMock(prompt_tokens=8, completion_tokens=12)
        self.mock_client.chat.completions.create.return_value = mock_response

        await call_openai(
            prompt="Prompt only",
            system_prompt="",
            timeout=12,
            client=self.mock_client,
        )

        _, kwargs = self.mock_client.chat.completions.create.call_args
        self.assertEqual(
            kwargs["messages"],
            [{"role": "user", "content": "Prompt only"}],
        )

    async def test_call_openai_auth_error_invalid_key(self):
        """Verify invalid API key raises no unhandled exception and returns status='error'."""
        req = httpx.Request("POST", "https://api.openai.com/v1/chat/completions")
        resp = httpx.Response(401, request=req)
        self.mock_client.chat.completions.create.side_effect = openai.AuthenticationError(
            message="Incorrect API key provided: sk-invalid.",
            response=resp,
            body={"error": {"message": "Incorrect API key provided: sk-invalid."}},
        )

        result = await call_openai(
            prompt="Prompt with invalid key",
            system_prompt="System prompt",
            timeout=12,
            client=self.mock_client,
        )

        self.assertEqual(result["status"], "error")
        self.assertEqual(result["model"], "gpt-4o-mini")
        self.assertIsNone(result["response_text"])
        self.assertEqual(result["tokens"], {"input": 0, "output": 0})
        self.assertIsInstance(result["duration_seconds"], float)
        self.assertIn("error_message", result)
        self.assertIn("Authentication error", result["error_message"])

        provider_result = ProviderResult.model_validate(result)
        self.assertEqual(provider_result.status, "error")
        self.assertIsNotNone(provider_result.error_message)

    async def test_call_openai_rate_limit_429(self):
        """Verify HTTP 429 rate limit is captured and formatted clearly."""
        req = httpx.Request("POST", "https://api.openai.com/v1/chat/completions")
        resp = httpx.Response(429, request=req)
        self.mock_client.chat.completions.create.side_effect = openai.RateLimitError(
            message="You exceeded your current quota, please check your plan and billing details.",
            response=resp,
            body={"error": {"message": "quota exceeded"}},
        )

        result = await call_openai(
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

    async def test_call_openai_server_error_500(self):
        """Verify 5xx server error is handled cleanly."""
        req = httpx.Request("POST", "https://api.openai.com/v1/chat/completions")
        resp = httpx.Response(500, request=req)
        self.mock_client.chat.completions.create.side_effect = openai.InternalServerError(
            message="The server had an error while processing your request.",
            response=resp,
            body=None,
        )

        result = await call_openai(
            prompt="Test prompt",
            system_prompt="System prompt",
            timeout=12,
            client=self.mock_client,
        )

        self.assertEqual(result["status"], "error")
        self.assertIn("OpenAI API error (500)", result["error_message"])

    async def test_call_openai_api_timeout_error(self):
        """Verify openai.APITimeoutError is captured as a request timeout."""
        req = httpx.Request("POST", "https://api.openai.com/v1/chat/completions")
        self.mock_client.chat.completions.create.side_effect = openai.APITimeoutError(request=req)

        result = await call_openai(
            prompt="Prompt timing out",
            system_prompt="System prompt",
            timeout=12,
            client=self.mock_client,
        )

        self.assertEqual(result["status"], "error")
        self.assertIn("Request timed out", result["error_message"])

    async def test_call_openai_api_connection_error(self):
        """Verify openai.APIConnectionError is normalized to a network error."""
        req = httpx.Request("POST", "https://api.openai.com/v1/chat/completions")
        self.mock_client.chat.completions.create.side_effect = openai.APIConnectionError(
            request=req, message="Connection refused"
        )

        result = await call_openai(
            prompt="Prompt failing connection",
            system_prompt="System prompt",
            timeout=12,
            client=self.mock_client,
        )

        self.assertEqual(result["status"], "error")
        self.assertIn("Network error", result["error_message"])

    async def test_call_openai_generic_openai_error(self):
        """Verify generic openai.OpenAIError catch-all."""
        self.mock_client.chat.completions.create.side_effect = openai.OpenAIError(
            "Generic SDK issue"
        )

        result = await call_openai(
            prompt="Test prompt",
            system_prompt="System prompt",
            timeout=12,
            client=self.mock_client,
        )

        self.assertEqual(result["status"], "error")
        self.assertIn("OpenAI API error: Generic SDK issue", result["error_message"])

    async def test_call_openai_httpx_timeout(self):
        """Verify httpx.TimeoutException is captured as a request timeout."""
        self.mock_client.chat.completions.create.side_effect = httpx.ReadTimeout(
            "Connection timed out while reading response"
        )

        result = await call_openai(
            prompt="Prompt timing out",
            system_prompt="System prompt",
            timeout=12,
            client=self.mock_client,
        )

        self.assertEqual(result["status"], "error")
        self.assertIn("Request timed out", result["error_message"])

    async def test_call_openai_asyncio_timeout(self):
        """Verify asyncio.TimeoutError is normalized to a request timeout."""
        self.mock_client.chat.completions.create.side_effect = asyncio.TimeoutError()

        result = await call_openai(
            prompt="Prompt timing out",
            system_prompt="System prompt",
            timeout=12,
            client=self.mock_client,
        )

        self.assertEqual(result["status"], "error")
        self.assertIn("Request timed out", result["error_message"])

    async def test_call_openai_httpx_network_error(self):
        """Verify httpx network errors are normalized."""
        self.mock_client.chat.completions.create.side_effect = httpx.ConnectError(
            "Failed to resolve host api.openai.com"
        )

        result = await call_openai(
            prompt="Prompt with connection fail",
            system_prompt="System prompt",
            timeout=12,
            client=self.mock_client,
        )

        self.assertEqual(result["status"], "error")
        self.assertIn("Network error", result["error_message"])

    async def test_call_openai_missing_api_key(self):
        """Verify calling without configured API key returns an error dict without crashing."""
        with patch.object(settings, "openai_api_key", ""):
            result = await call_openai(
                prompt="Prompt without key",
                system_prompt="System prompt",
                timeout=12,
            )

            self.assertEqual(result["status"], "error")
            self.assertEqual(result["tokens"], {"input": 0, "output": 0})
            self.assertIsNone(result["response_text"])
            self.assertIn("Authentication error", result["error_message"])

    async def test_call_openai_empty_usage(self):
        """Verify tokens default to 0 if usage is missing or None."""
        mock_response = MagicMock()
        mock_response.choices = [MagicMock(message=MagicMock(content="Answer"))]
        mock_response.usage = None
        self.mock_client.chat.completions.create.return_value = mock_response

        result = await call_openai(
            prompt="Prompt",
            system_prompt="System",
            timeout=12,
            client=self.mock_client,
        )

        self.assertEqual(result["status"], "success")
        self.assertEqual(result["tokens"], {"input": 0, "output": 0})
        self.assertEqual(result["response_text"], "Answer")

    async def test_call_openai_none_response_text(self):
        """Verify response_text defaults to empty string if content is None."""
        mock_response = MagicMock()
        mock_choice = MagicMock()
        mock_choice.message = MagicMock(content=None)
        mock_response.choices = [mock_choice]
        mock_response.usage = MagicMock(prompt_tokens=10, completion_tokens=0)
        self.mock_client.chat.completions.create.return_value = mock_response

        result = await call_openai(
            prompt="Prompt",
            system_prompt="System",
            timeout=12,
            client=self.mock_client,
        )

        self.assertEqual(result["status"], "success")
        self.assertEqual(result["response_text"], "")

    async def test_call_openai_empty_choices(self):
        """Verify response_text defaults to empty string if choices list is empty."""
        mock_response = MagicMock()
        mock_response.choices = []
        mock_response.usage = MagicMock(prompt_tokens=5, completion_tokens=0)
        self.mock_client.chat.completions.create.return_value = mock_response

        result = await call_openai(
            prompt="Prompt",
            system_prompt="System",
            timeout=12,
            client=self.mock_client,
        )

        self.assertEqual(result["status"], "success")
        self.assertEqual(result["response_text"], "")

    def test_get_openai_client_custom_key(self):
        """Verify get_openai_client accepts a custom API key."""
        client = get_openai_client(api_key="sk-custom-valid-key")
        self.assertIsNotNone(client)
        self.assertEqual(client.api_key, "sk-custom-valid-key")

    def test_get_openai_client_empty_raises(self):
        """Verify get_openai_client raises ValueError when key is empty."""
        with self.assertRaises(ValueError):
            get_openai_client(api_key="")

    async def test_contract_parity_with_gemini(self):
        """Verify return shape is strictly identical between call_openai and call_gemini."""
        # Success shapes
        mock_gemini_client = MagicMock()
        mock_gemini_resp = MagicMock(text="Gemini text", usage_metadata=MagicMock(prompt_token_count=10, candidates_token_count=20))
        mock_gemini_client.aio.models.generate_content = AsyncMock(return_value=mock_gemini_resp)

        mock_openai_resp = MagicMock(
            choices=[MagicMock(message=MagicMock(content="OpenAI text"))],
            usage=MagicMock(prompt_tokens=10, completion_tokens=20),
        )
        self.mock_client.chat.completions.create.return_value = mock_openai_resp

        gemini_success = await call_gemini("Prompt", "System", client=mock_gemini_client)
        openai_success = await call_openai("Prompt", "System", client=self.mock_client)

        self.assertEqual(set(gemini_success.keys()), set(openai_success.keys()))
        for key in ("status", "tokens", "response_text"):
            self.assertEqual(type(gemini_success[key]), type(openai_success[key]))
        self.assertEqual(set(gemini_success["tokens"].keys()), set(openai_success["tokens"].keys()))

        # Error shapes
        self.mock_client.chat.completions.create.side_effect = Exception("OpenAI failure")
        mock_gemini_client.aio.models.generate_content.side_effect = Exception("Gemini failure")

        gemini_error = await call_gemini("Prompt", "System", client=mock_gemini_client)
        openai_error = await call_openai("Prompt", "System", client=self.mock_client)

        self.assertEqual(set(gemini_error.keys()), set(openai_error.keys()))
        for key in ("status", "tokens", "response_text", "error_message"):
            self.assertEqual(type(gemini_error[key]), type(openai_error[key]))


if __name__ == "__main__":
    unittest.main()
