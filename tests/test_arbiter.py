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

from google.genai import types

from app.config import settings
from app.core.prompts import build_arbiter_prompt
from app.schemas.models import Context, LOBEnum, RoleEnum
from app.services.arbiter import (
    ARBITER_MODELS,
    ArbiterError,
    build_arbiter_user_message,
    synthesize,
)


class TestArbiterUserMessage(unittest.TestCase):
    """Unit tests for building user message payloads sent to the Arbiter model."""

    def test_build_user_message_two_results(self):
        """Verify user message format when 2 providers successfully returned responses."""
        prompt = "An insured backed a trailer into their garage door. How are these covered?"
        results = [
            {"response_text": "The garage door is covered under Section I Dwelling."},
            {"response_text": "First-party coverage applies to the attached structure."},
        ]

        message = build_arbiter_user_message(prompt, results)

        expected = (
            "Original Query: An insured backed a trailer into their garage door. How are these covered?\n\n"
            "Model 1 Response:\n"
            "The garage door is covered under Section I Dwelling.\n\n"
            "Model 2 Response:\n"
            "First-party coverage applies to the attached structure.\n\n"
            "Please synthesize these responses into a single authoritative answer."
        )
        self.assertEqual(message, expected)

    def test_build_user_message_three_results(self):
        """Verify user message format when all 3 providers successfully returned responses."""
        prompt = "Explain liability limits."
        results = [
            {"response_text": "Response from worker A."},
            {"response_text": "Response from worker B."},
            {"response_text": "Response from worker C."},
        ]

        message = build_arbiter_user_message(prompt, results)

        self.assertIn("Original Query: Explain liability limits.", message)
        self.assertIn("Model 1 Response:\nResponse from worker A.", message)
        self.assertIn("Model 2 Response:\nResponse from worker B.", message)
        self.assertIn("Model 3 Response:\nResponse from worker C.", message)
        self.assertTrue(
            message.endswith(
                "Please synthesize these responses into a single authoritative answer."
            )
        )

    def test_build_user_message_anonymity_no_provider_names_in_labels(self):
        """Verify provider and model names are completely omitted from input labels."""
        prompt = "Coverage query"
        results = [
            {
                "provider": "gemini",
                "model": "gemini-2.5-flash",
                "response_text": "Gemini text",
            },
            {
                "provider": "openai",
                "model": "gpt-4o-mini",
                "response_text": "OpenAI text",
            },
            {
                "provider": "claude",
                "model": "claude-3-5-haiku",
                "response_text": "Claude text",
            },
        ]

        message = build_arbiter_user_message(prompt, results)

        # Ensure headers do NOT reveal provider names
        self.assertNotIn("Gemini Response:", message)
        self.assertNotIn("OpenAI Response:", message)
        self.assertNotIn("Claude Response:", message)
        self.assertNotIn("gemini-2.5-flash:", message)
        self.assertNotIn("gpt-4o-mini:", message)
        self.assertNotIn("claude-3-5-haiku:", message)

        self.assertIn("Model 1 Response:\nGemini text", message)
        self.assertIn("Model 2 Response:\nOpenAI text", message)
        self.assertIn("Model 3 Response:\nClaude text", message)

    def test_build_user_message_whitespace_handling(self):
        """Verify leading/trailing whitespace in prompts and responses is trimmed cleanly."""
        prompt = "   Query with whitespace   \n"
        results = [
            {"response_text": "   Answer 1 with spaces   \n"},
            {"response_text": "\nAnswer 2\n"},
        ]

        message = build_arbiter_user_message(prompt, results)
        self.assertIn("Original Query: Query with whitespace", message)
        self.assertIn("Model 1 Response:\nAnswer 1 with spaces", message)
        self.assertIn("Model 2 Response:\nAnswer 2", message)

    def test_build_user_message_missing_or_none_text(self):
        """Verify missing or None response_text defaults safely to empty string."""
        prompt = "Test prompt"
        results = [
            {"response_text": None},
            {},
        ]

        message = build_arbiter_user_message(prompt, results)
        self.assertIn("Model 1 Response:\n", message)
        self.assertIn("Model 2 Response:\n", message)


class TestSynthesizeValidation(unittest.IsolatedAsyncioTestCase):
    """Unit tests for parameter validation in the synthesize function."""

    def setUp(self):
        self.context = Context(
            role=RoleEnum.claims_adjuster,
            line_of_business=LOBEnum.homeowners,
            state="MT",
        )

    async def test_empty_results_raises_value_error(self):
        """Verify ValueError is raised when 0 successful results are provided."""
        with self.assertRaises(ValueError) as ctx:
            await synthesize("Prompt", self.context, [])
        self.assertIn("requires at least 2 successful provider results", str(ctx.exception))

    async def test_single_result_raises_value_error_in_happy_path(self):
        """Verify ValueError is raised when only 1 result is provided (degraded path deferred)."""
        with self.assertRaises(ValueError) as ctx:
            await synthesize("Prompt", self.context, [{"response_text": "Solo response"}])
        self.assertIn("requires at least 2 successful provider results", str(ctx.exception))

    async def test_invalid_provider_raises_value_error(self):
        """Verify ValueError is raised when an unsupported provider is requested."""
        results = [{"response_text": "A"}, {"response_text": "B"}]
        with self.assertRaises(ValueError) as ctx:
            await synthesize("Prompt", self.context, results, provider="unsupported_ai")
        self.assertIn("Unsupported arbiter model provider", str(ctx.exception))
        self.assertIn("'gemini'", str(ctx.exception))
        self.assertIn("'openai'", str(ctx.exception))


class TestSynthesizeGemini(unittest.IsolatedAsyncioTestCase):
    """Unit tests for Arbiter synthesis using Google Gemini 2.5 Pro."""

    def setUp(self):
        self.mock_client = MagicMock()
        self.mock_client.aio = MagicMock()
        self.mock_client.aio.models = MagicMock()
        self.mock_client.aio.models.generate_content = AsyncMock()

        self.context = Context(
            role=RoleEnum.claims_adjuster,
            line_of_business=LOBEnum.homeowners,
            state="MT",
        )
        self.results = [
            {"response_text": "Garage door falls under Dwelling Coverage A."},
            {"response_text": "Coverage A applies subject to the $500 deductible."},
        ]

    async def test_gemini_synthesis_success(self):
        """Verify successful Gemini 2.5 Pro call returns raw authoritative consensus text."""
        mock_response = MagicMock()
        mock_response.text = "Authoritative consensus: The attached garage door is covered under Coverage A."
        self.mock_client.aio.models.generate_content.return_value = mock_response

        prompt = "An insured backed a trailer into their garage door. How are these covered?"
        consensus = await synthesize(
            original_prompt=prompt,
            context=self.context,
            successful_results=self.results,
            provider="gemini",
            client=self.mock_client,
        )

        self.assertEqual(
            consensus,
            "Authoritative consensus: The attached garage door is covered under Coverage A.",
        )
        self.mock_client.aio.models.generate_content.assert_awaited_once()

        call_args = self.mock_client.aio.models.generate_content.call_args
        self.assertEqual(call_args.kwargs["model"], ARBITER_MODELS["gemini"])
        self.assertEqual(call_args.kwargs["model"], "gemini-2.5-pro")

        # Verify system prompt interpolation
        expected_system_prompt = build_arbiter_prompt(
            role=self.context.role,
            line_of_business=self.context.line_of_business,
            state=self.context.state,
        )
        config = call_args.kwargs["config"]
        self.assertEqual(config.system_instruction, expected_system_prompt)

        # Verify user message format
        expected_user_message = build_arbiter_user_message(prompt, self.results)
        self.assertEqual(call_args.kwargs["contents"], expected_user_message)

    async def test_gemini_synthesis_failure_raises_arbiter_error(self):
        """Verify underlying Gemini API exceptions are raised as ArbiterError."""
        self.mock_client.aio.models.generate_content.side_effect = RuntimeError(
            "Gemini quota exhausted"
        )

        with self.assertRaises(ArbiterError) as ctx:
            await synthesize(
                original_prompt="Query",
                context=self.context,
                successful_results=self.results,
                provider="gemini",
                client=self.mock_client,
            )
        self.assertIn("Gemini arbiter synthesis failed", str(ctx.exception))


class TestSynthesizeOpenAI(unittest.IsolatedAsyncioTestCase):
    """Unit tests for Arbiter synthesis using OpenAI GPT-4o."""

    def setUp(self):
        self.mock_client = MagicMock()
        self.mock_client.chat = MagicMock()
        self.mock_client.chat.completions = MagicMock()
        self.mock_client.chat.completions.create = AsyncMock()

        self.context = Context(
            role=RoleEnum.underwriter,
            line_of_business=LOBEnum.personal_auto,
            state="MT",
        )
        self.results = [
            {"response_text": "Model 1 says bodily injury limit is 25/50 in MT."},
            {"response_text": "Model 2 confirms statutory minimums 25/50/20 in MT."},
        ]

    async def test_openai_synthesis_success(self):
        """Verify successful OpenAI GPT-4o call returns raw authoritative consensus text."""
        mock_response = MagicMock()
        mock_choice = MagicMock()
        mock_choice.message = MagicMock(
            content="Consensus: In Montana, statutory minimum auto limits are 25/50/20."
        )
        mock_response.choices = [mock_choice]
        self.mock_client.chat.completions.create.return_value = mock_response

        prompt = "What are the required auto liability limits in Montana?"
        consensus = await synthesize(
            original_prompt=prompt,
            context=self.context,
            successful_results=self.results,
            provider="openai",
            client=self.mock_client,
        )

        self.assertEqual(
            consensus,
            "Consensus: In Montana, statutory minimum auto limits are 25/50/20.",
        )
        self.mock_client.chat.completions.create.assert_awaited_once()

        call_args = self.mock_client.chat.completions.create.call_args
        self.assertEqual(call_args.kwargs["model"], ARBITER_MODELS["openai"])
        self.assertEqual(call_args.kwargs["model"], "gpt-4o")

        # Verify messages: system instruction and user message
        expected_system_prompt = build_arbiter_prompt(
            role=self.context.role,
            line_of_business=self.context.line_of_business,
            state=self.context.state,
        )
        expected_user_message = build_arbiter_user_message(prompt, self.results)

        messages = call_args.kwargs["messages"]
        self.assertEqual(len(messages), 2)
        self.assertEqual(messages[0], {"role": "system", "content": expected_system_prompt})
        self.assertEqual(messages[1], {"role": "user", "content": expected_user_message})

    async def test_openai_synthesis_failure_raises_arbiter_error(self):
        """Verify underlying OpenAI API exceptions are raised as ArbiterError."""
        self.mock_client.chat.completions.create.side_effect = RuntimeError(
            "Rate limit exceeded (HTTP 429)"
        )

        with self.assertRaises(ArbiterError) as ctx:
            await synthesize(
                original_prompt="Query",
                context=self.context,
                successful_results=self.results,
                provider="openai",
                client=self.mock_client,
            )
        self.assertIn("OpenAI arbiter synthesis failed", str(ctx.exception))


class TestSynthesizeProviderResolution(unittest.IsolatedAsyncioTestCase):
    """Unit tests verifying arbiter model provider selection via settings."""

    def setUp(self):
        self.context = Context(
            role=RoleEnum.layman_linguist,
            line_of_business=LOBEnum.homeowners,
            state="MT",
        )
        self.results = [
            {"response_text": "Answer 1"},
            {"response_text": "Answer 2"},
        ]

    async def test_default_to_settings_gemini(self):
        """Verify defaults to Gemini 2.5 Pro when settings.arbiter_model_provider is 'gemini'."""
        mock_gemini_client = MagicMock()
        mock_gemini_client.aio = MagicMock()
        mock_gemini_client.aio.models = MagicMock()
        mock_gemini_client.aio.models.generate_content = AsyncMock()
        mock_response = MagicMock(text="Gemini Arbiter Response")
        mock_gemini_client.aio.models.generate_content.return_value = mock_response

        with patch.object(settings, "arbiter_model_provider", "gemini"):
            consensus = await synthesize(
                original_prompt="Prompt",
                context=self.context,
                successful_results=self.results,
                client=mock_gemini_client,
            )

        self.assertEqual(consensus, "Gemini Arbiter Response")
        mock_gemini_client.aio.models.generate_content.assert_awaited_once()
        self.assertEqual(
            mock_gemini_client.aio.models.generate_content.call_args.kwargs["model"],
            "gemini-2.5-pro",
        )

    async def test_default_to_settings_openai(self):
        """Verify defaults to GPT-4o when settings.arbiter_model_provider is 'openai'."""
        mock_openai_client = MagicMock()
        mock_openai_client.chat = MagicMock()
        mock_openai_client.chat.completions = MagicMock()
        mock_openai_client.chat.completions.create = AsyncMock()
        mock_response = MagicMock(
            choices=[MagicMock(message=MagicMock(content="OpenAI Arbiter Response"))]
        )
        mock_openai_client.chat.completions.create.return_value = mock_response

        with patch.object(settings, "arbiter_model_provider", "openai"):
            consensus = await synthesize(
                original_prompt="Prompt",
                context=self.context,
                successful_results=self.results,
                client=mock_openai_client,
            )

        self.assertEqual(consensus, "OpenAI Arbiter Response")
        mock_openai_client.chat.completions.create.assert_awaited_once()
        self.assertEqual(
            mock_openai_client.chat.completions.create.call_args.kwargs["model"],
            "gpt-4o",
        )

    async def test_provider_case_insensitivity(self):
        """Verify provider strings with mixed case or whitespace are normalized."""
        mock_gemini_client = MagicMock()
        mock_gemini_client.aio = MagicMock()
        mock_gemini_client.aio.models = MagicMock()
        mock_gemini_client.aio.models.generate_content = AsyncMock()
        mock_gemini_client.aio.models.generate_content.return_value = MagicMock(text="Normalized")

        consensus = await synthesize(
            original_prompt="Prompt",
            context=self.context,
            successful_results=self.results,
            provider="  GEMINI  ",
            client=mock_gemini_client,
        )
        self.assertEqual(consensus, "Normalized")


class TestSynthesizeContextHandling(unittest.IsolatedAsyncioTestCase):
    """Unit tests verifying Context object and dictionary context compatibility."""

    async def test_dict_context_compatibility(self):
        """Verify synthesize supports context provided as a dictionary."""
        mock_gemini_client = MagicMock()
        mock_gemini_client.aio = MagicMock()
        mock_gemini_client.aio.models = MagicMock()
        mock_gemini_client.aio.models.generate_content = AsyncMock()
        mock_gemini_client.aio.models.generate_content.return_value = MagicMock(
            text="Dict Context Output"
        )

        dict_context = {
            "role": "claims_adjuster",
            "line_of_business": "homeowners",
            "state": "MT",
        }
        results = [
            {"response_text": "Answer 1"},
            {"response_text": "Answer 2"},
        ]

        consensus = await synthesize(
            original_prompt="Prompt",
            context=dict_context,
            successful_results=results,
            provider="gemini",
            client=mock_gemini_client,
        )

        self.assertEqual(consensus, "Dict Context Output")
        config = mock_gemini_client.aio.models.generate_content.call_args.kwargs["config"]
        self.assertIn("claims_adjuster", config.system_instruction)
        self.assertIn("homeowners", config.system_instruction)
        self.assertIn("MT", config.system_instruction)


if __name__ == "__main__":
    unittest.main()
