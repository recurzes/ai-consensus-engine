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

import anthropic
import httpx

from app.config import settings
from app.schemas.models import ProviderResult
from app.services.claude_client import (
    DEFAULT_MAX_TOKENS,
    MODEL_NAME,
    call_claude,
    get_claude_client,
)
from app.services.gemini_client import call_gemini
from app.services.openai_client import call_openai


class TestClaudeClient(unittest.IsolatedAsyncioTestCase):
    """Unit tests for the Anthropic Claude async provider client."""

    def setUp(self):
        self.mock_client = MagicMock()
        self.mock_client.messages = MagicMock()
        self.mock_client.messages.create = AsyncMock()

    async def test_call_claude_success(self):
        """Verify successful API call returns normalized dict matching ProviderResult schema."""
        mock_response = MagicMock()
        mock_content_block = MagicMock()
        mock_content_block.text = "This is a comprehensive Claude response."
        mock_response.content = [mock_content_block]
        mock_response.usage = MagicMock(
            input_tokens=48,
            output_tokens=135,
        )
        self.mock_client.messages.create.return_value = mock_response

        prompt = "Explain liability coverage under personal auto policies."
        system_prompt = "You are an expert insurance underwriter."
        result = await call_claude(
            prompt=prompt,
            system_prompt=system_prompt,
            timeout=12,
            client=self.mock_client,
        )

        self.assertEqual(result["status"], "success")
        self.assertEqual(result["model"], MODEL_NAME)
        self.assertEqual(result["model"], "claude-3-5-haiku")
        self.assertEqual(
            result["response_text"],
            "This is a comprehensive Claude response.",
        )
        self.assertEqual(result["tokens"], {"input": 48, "output": 135})
        self.assertIsInstance(result["duration_seconds"], float)
        self.assertGreaterEqual(result["duration_seconds"], 0.0)
        self.assertNotIn("error_message", result)

        # Validate that ProviderResult validates the returned dict
        provider_result = ProviderResult.model_validate(result)
        self.assertEqual(provider_result.status, "success")
        self.assertEqual(provider_result.model, "claude-3-5-haiku")
        self.assertEqual(provider_result.tokens["input"], 48)
        self.assertEqual(provider_result.tokens["output"], 135)
        self.assertEqual(
            provider_result.response_text,
            "This is a comprehensive Claude response.",
        )
        self.assertIsNone(provider_result.error_message)

    async def test_call_claude_passes_parameters_correctly(self):
        """Verify model, system prompt, max_tokens, messages format, and timeout are forwarded to SDK."""
        mock_response = MagicMock()
        mock_response.content = [MagicMock(text="Response")]
        mock_response.usage = MagicMock(input_tokens=10, output_tokens=20)
        self.mock_client.messages.create.return_value = mock_response

        await call_claude(
            prompt="Test prompt",
            system_prompt="Test system instruction",
            timeout=15,
            client=self.mock_client,
        )

        self.mock_client.messages.create.assert_awaited_once()
        _, kwargs = self.mock_client.messages.create.call_args
        self.assertEqual(kwargs["model"], "claude-3-5-haiku")
        self.assertEqual(kwargs["max_tokens"], DEFAULT_MAX_TOKENS)
        self.assertEqual(kwargs["max_tokens"], 4096)
        self.assertEqual(kwargs["timeout"], 15.0)
        self.assertEqual(kwargs["system"], "Test system instruction")
        self.assertEqual(
            kwargs["messages"],
            [{"role": "user", "content": "Test prompt"}],
        )

    async def test_call_claude_without_system_prompt(self):
        """Verify when system prompt is empty, system parameter is omitted."""
        mock_response = MagicMock()
        mock_response.content = [MagicMock(text="Response")]
        mock_response.usage = MagicMock(input_tokens=8, output_tokens=12)
        self.mock_client.messages.create.return_value = mock_response

        await call_claude(
            prompt="Prompt only",
            system_prompt="",
            timeout=12,
            client=self.mock_client,
        )

        _, kwargs = self.mock_client.messages.create.call_args
        self.assertNotIn("system", kwargs)
        self.assertEqual(
            kwargs["messages"],
            [{"role": "user", "content": "Prompt only"}],
        )

    async def test_call_claude_auth_error_invalid_key(self):
        """Verify invalid API key raises no unhandled exception and returns status='error'."""
        req = httpx.Request("POST", "https://api.anthropic.com/v1/messages")
        resp = httpx.Response(401, request=req)
        self.mock_client.messages.create.side_effect = anthropic.AuthenticationError(
            message="Invalid API Key provided: sk-ant-invalid.",
            response=resp,
            body={"error": {"type": "authentication_error", "message": "Invalid API Key provided: sk-ant-invalid."}},
        )

        result = await call_claude(
            prompt="Prompt with invalid key",
            system_prompt="System prompt",
            timeout=12,
            client=self.mock_client,
        )

        self.assertEqual(result["status"], "error")
        self.assertEqual(result["model"], "claude-3-5-haiku")
        self.assertIsNone(result["response_text"])
        self.assertEqual(result["tokens"], {"input": 0, "output": 0})
        self.assertIsInstance(result["duration_seconds"], float)
        self.assertIn("error_message", result)
        self.assertIn("Authentication error", result["error_message"])

        provider_result = ProviderResult.model_validate(result)
        self.assertEqual(provider_result.status, "error")
        self.assertIsNotNone(provider_result.error_message)

    async def test_call_claude_rate_limit_429(self):
        """Verify HTTP 429 rate limit is captured and formatted clearly."""
        req = httpx.Request("POST", "https://api.anthropic.com/v1/messages")
        resp = httpx.Response(429, request=req)
        self.mock_client.messages.create.side_effect = anthropic.RateLimitError(
            message="Number of request tokens has exceeded your daily rate limit.",
            response=resp,
            body={"error": {"type": "rate_limit_error", "message": "Rate limit exceeded"}},
        )

        result = await call_claude(
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

    async def test_call_claude_server_error_500(self):
        """Verify 5xx server error is handled cleanly."""
        req = httpx.Request("POST", "https://api.anthropic.com/v1/messages")
        resp = httpx.Response(500, request=req)
        self.mock_client.messages.create.side_effect = anthropic.InternalServerError(
            message="Internal server error occurred.",
            response=resp,
            body=None,
        )

        result = await call_claude(
            prompt="Test prompt",
            system_prompt="System prompt",
            timeout=12,
            client=self.mock_client,
        )

        self.assertEqual(result["status"], "error")
        self.assertIn("Anthropic API error (500)", result["error_message"])

    async def test_call_claude_api_timeout_error(self):
        """Verify anthropic.APITimeoutError is captured as a request timeout."""
        req = httpx.Request("POST", "https://api.anthropic.com/v1/messages")
        self.mock_client.messages.create.side_effect = anthropic.APITimeoutError(request=req)

        result = await call_claude(
            prompt="Prompt timing out",
            system_prompt="System prompt",
            timeout=12,
            client=self.mock_client,
        )

        self.assertEqual(result["status"], "error")
        self.assertIn("Request timed out", result["error_message"])

    async def test_call_claude_api_connection_error(self):
        """Verify anthropic.APIConnectionError is normalized to a network error."""
        req = httpx.Request("POST", "https://api.anthropic.com/v1/messages")
        self.mock_client.messages.create.side_effect = anthropic.APIConnectionError(
            request=req, message="Connection refused"
        )

        result = await call_claude(
            prompt="Prompt failing connection",
            system_prompt="System prompt",
            timeout=12,
            client=self.mock_client,
        )

        self.assertEqual(result["status"], "error")
        self.assertIn("Network error", result["error_message"])

    async def test_call_claude_generic_api_error(self):
        """Verify generic anthropic.APIError catch-all."""
        req = httpx.Request("POST", "https://api.anthropic.com/v1/messages")
        self.mock_client.messages.create.side_effect = anthropic.APIError(
            "Generic Anthropic SDK issue", request=req, body=None
        )

        result = await call_claude(
            prompt="Test prompt",
            system_prompt="System prompt",
            timeout=12,
            client=self.mock_client,
        )

        self.assertEqual(result["status"], "error")
        self.assertIn("Anthropic API error: Generic Anthropic SDK issue", result["error_message"])

    async def test_call_claude_httpx_timeout(self):
        """Verify httpx.TimeoutException is captured as a request timeout."""
        self.mock_client.messages.create.side_effect = httpx.ReadTimeout(
            "Connection timed out while reading response"
        )

        result = await call_claude(
            prompt="Prompt timing out",
            system_prompt="System prompt",
            timeout=12,
            client=self.mock_client,
        )

        self.assertEqual(result["status"], "error")
        self.assertIn("Request timed out", result["error_message"])

    async def test_call_claude_asyncio_timeout(self):
        """Verify asyncio.TimeoutError is normalized to a request timeout."""
        self.mock_client.messages.create.side_effect = asyncio.TimeoutError()

        result = await call_claude(
            prompt="Prompt timing out",
            system_prompt="System prompt",
            timeout=12,
            client=self.mock_client,
        )

        self.assertEqual(result["status"], "error")
        self.assertIn("Request timed out", result["error_message"])

    async def test_call_claude_httpx_network_error(self):
        """Verify httpx network errors are normalized."""
        self.mock_client.messages.create.side_effect = httpx.ConnectError(
            "Failed to resolve host api.anthropic.com"
        )

        result = await call_claude(
            prompt="Prompt with connection fail",
            system_prompt="System prompt",
            timeout=12,
            client=self.mock_client,
        )

        self.assertEqual(result["status"], "error")
        self.assertIn("Network error", result["error_message"])

    async def test_call_claude_missing_api_key(self):
        """Verify calling without configured API key returns an error dict without crashing."""
        with patch.object(settings, "anthropic_api_key", ""):
            result = await call_claude(
                prompt="Prompt without key",
                system_prompt="System prompt",
                timeout=12,
            )

            self.assertEqual(result["status"], "error")
            self.assertEqual(result["tokens"], {"input": 0, "output": 0})
            self.assertIsNone(result["response_text"])
            self.assertIn("Authentication error", result["error_message"])

    async def test_call_claude_empty_usage(self):
        """Verify tokens default to 0 if usage is missing or None."""
        mock_response = MagicMock()
        mock_response.content = [MagicMock(text="Answer")]
        mock_response.usage = None
        self.mock_client.messages.create.return_value = mock_response

        result = await call_claude(
            prompt="Prompt",
            system_prompt="System",
            timeout=12,
            client=self.mock_client,
        )

        self.assertEqual(result["status"], "success")
        self.assertEqual(result["tokens"], {"input": 0, "output": 0})
        self.assertEqual(result["response_text"], "Answer")

    async def test_call_claude_none_response_text(self):
        """Verify response_text defaults to empty string if content block text is None."""
        mock_response = MagicMock()
        mock_block = MagicMock()
        mock_block.text = None
        mock_response.content = [mock_block]
        mock_response.usage = MagicMock(input_tokens=10, output_tokens=0)
        self.mock_client.messages.create.return_value = mock_response

        result = await call_claude(
            prompt="Prompt",
            system_prompt="System",
            timeout=12,
            client=self.mock_client,
        )

        self.assertEqual(result["status"], "success")
        self.assertEqual(result["response_text"], "")

    async def test_call_claude_empty_content(self):
        """Verify response_text defaults to empty string if content list is empty."""
        mock_response = MagicMock()
        mock_response.content = []
        mock_response.usage = MagicMock(input_tokens=5, output_tokens=0)
        self.mock_client.messages.create.return_value = mock_response

        result = await call_claude(
            prompt="Prompt",
            system_prompt="System",
            timeout=12,
            client=self.mock_client,
        )

        self.assertEqual(result["status"], "success")
        self.assertEqual(result["response_text"], "")

    def test_get_claude_client_custom_key(self):
        """Verify get_claude_client accepts a custom API key."""
        client = get_claude_client(api_key="sk-ant-custom-valid-key")
        self.assertIsNotNone(client)
        self.assertEqual(client.api_key, "sk-ant-custom-valid-key")

    def test_get_claude_client_empty_raises(self):
        """Verify get_claude_client raises ValueError when key is empty."""
        with self.assertRaises(ValueError):
            get_claude_client(api_key="")

    def test_get_claude_client_with_workspace_id(self):
        """Verify get_claude_client passes anthropic-workspace-id header when provided."""
        client = get_claude_client(
            api_key="sk-ant-custom-valid-key", workspace_id="wrkspc_test_123"
        )
        self.assertIsNotNone(client)
        self.assertEqual(
            client.default_headers.get("anthropic-workspace-id"), "wrkspc_test_123"
        )

    def test_get_claude_client_from_settings_workspace_id(self):
        """Verify get_claude_client uses settings.anthropic_workspace_id by default."""
        with patch.object(settings, "anthropic_workspace_id", "wrkspc_settings_456"):
            client = get_claude_client(api_key="sk-ant-custom-valid-key")
            self.assertEqual(
                client.default_headers.get("anthropic-workspace-id"),
                "wrkspc_settings_456",
            )

    async def test_contract_parity_with_gemini_and_openai(self):
        """Verify return shape is strictly identical across call_claude, call_openai, and call_gemini."""
        # Success shapes
        mock_gemini_client = MagicMock()
        mock_gemini_resp = MagicMock(
            text="Gemini text",
            usage_metadata=MagicMock(prompt_token_count=10, candidates_token_count=20),
        )
        mock_gemini_client.aio.models.generate_content = AsyncMock(return_value=mock_gemini_resp)

        mock_openai_client = MagicMock()
        mock_openai_resp = MagicMock(
            choices=[MagicMock(message=MagicMock(content="OpenAI text"))],
            usage=MagicMock(prompt_tokens=10, completion_tokens=20),
        )
        mock_openai_client.chat.completions.create = AsyncMock(return_value=mock_openai_resp)

        mock_claude_resp = MagicMock(
            content=[MagicMock(text="Claude text")],
            usage=MagicMock(input_tokens=10, output_tokens=20),
        )
        self.mock_client.messages.create.return_value = mock_claude_resp

        gemini_success = await call_gemini("Prompt", "System", client=mock_gemini_client)
        openai_success = await call_openai("Prompt", "System", client=mock_openai_client)
        claude_success = await call_claude("Prompt", "System", client=self.mock_client)

        self.assertEqual(set(claude_success.keys()), set(openai_success.keys()))
        self.assertEqual(set(claude_success.keys()), set(gemini_success.keys()))
        for key in ("status", "tokens", "response_text"):
            self.assertEqual(type(claude_success[key]), type(openai_success[key]))
            self.assertEqual(type(claude_success[key]), type(gemini_success[key]))
        self.assertEqual(set(claude_success["tokens"].keys()), set(openai_success["tokens"].keys()))
        self.assertEqual(set(claude_success["tokens"].keys()), set(gemini_success["tokens"].keys()))

        # Error shapes
        self.mock_client.messages.create.side_effect = Exception("Claude failure")
        mock_openai_client.chat.completions.create.side_effect = Exception("OpenAI failure")
        mock_gemini_client.aio.models.generate_content.side_effect = Exception("Gemini failure")

        gemini_error = await call_gemini("Prompt", "System", client=mock_gemini_client)
        openai_error = await call_openai("Prompt", "System", client=mock_openai_client)
        claude_error = await call_claude("Prompt", "System", client=self.mock_client)

        self.assertEqual(set(claude_error.keys()), set(openai_error.keys()))
        self.assertEqual(set(claude_error.keys()), set(gemini_error.keys()))
        for key in ("status", "tokens", "response_text", "error_message"):
            self.assertEqual(type(claude_error[key]), type(openai_error[key]))
            self.assertEqual(type(claude_error[key]), type(gemini_error[key]))


if __name__ == "__main__":
    unittest.main()
