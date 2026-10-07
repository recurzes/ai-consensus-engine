"""Unit tests for the three worker client modules using mocked API calls.

Covers success and failure paths for:
- call_gemini()
- call_openai()
- call_claude()

Verifies that the normalization contract (ProviderResult schema) holds in all cases.
"""

import asyncio
import os
from unittest.mock import AsyncMock, MagicMock, patch

# Ensure test env defaults exist before loading settings or clients
_TEST_ENV_DEFAULTS = {
    "GEMINI_API_KEY": "test-gemini-key",
    "OPENAI_API_KEY": "test-openai-key",
    "ANTHROPIC_API_KEY": "test-anthropic-key",
    "ARBITER_MODEL_PROVIDER": "gemini",
}
for k, v in _TEST_ENV_DEFAULTS.items():
    os.environ.setdefault(k, v)

import anthropic
from google.genai.errors import APIError, ClientError
import httpx
import openai
import pytest

from app.schemas.models import ProviderResult
from app.services.claude_client import MODEL_NAME as CLAUDE_MODEL_NAME, call_claude
from app.services.gemini_client import MODEL_NAME as GEMINI_MODEL_NAME, call_gemini
from app.services.openai_client import MODEL_NAME as OPENAI_MODEL_NAME, call_openai


# ============================================================================
# 1. Gemini Client Unit Tests
# ============================================================================

@pytest.mark.asyncio
async def test_gemini_client_success():
    """Verify Gemini client success path returns normalized dict matching ProviderResult."""
    mock_response = MagicMock()
    mock_response.text = "The garage door is covered under Coverage B - Other Structures."
    mock_response.usage_metadata = MagicMock(
        prompt_token_count=85,
        candidates_token_count=210,
    )

    mock_client = MagicMock()
    mock_client.aio.models.generate_content = AsyncMock(return_value=mock_response)

    with patch("app.services.gemini_client.get_gemini_client", return_value=mock_client):
        result = await call_gemini(
            prompt="Is the garage door covered?",
            system_prompt="You are an insurance underwriter.",
            timeout=12,
        )

    assert result["status"] == "success"
    assert result["model"] == GEMINI_MODEL_NAME
    assert result["response_text"] == "The garage door is covered under Coverage B - Other Structures."
    assert result["tokens"]["input"] == 85
    assert result["tokens"]["output"] == 210
    assert isinstance(result["duration_seconds"], float)
    assert result["duration_seconds"] >= 0.0
    assert "error_message" not in result

    # Validate schema conformity
    validated = ProviderResult.model_validate(result)
    assert validated.status == "success"
    assert validated.tokens["input"] == 85
    assert validated.tokens["output"] == 210


@pytest.mark.asyncio
async def test_gemini_client_rate_limit():
    """Verify Gemini rate limit exception (HTTP 429) returns status='error' with error_message."""
    error_json = {
        "error": {
            "code": 429,
            "message": "Resource exhausted: quota exceeded for quota metric.",
            "status": "RESOURCE_EXHAUSTED",
        }
    }
    mock_client = MagicMock()
    mock_client.aio.models.generate_content = AsyncMock(
        side_effect=ClientError(429, error_json, None)
    )

    with patch("app.services.gemini_client.get_gemini_client", return_value=mock_client):
        result = await call_gemini(
            prompt="Is this covered?",
            system_prompt="Insurance analyst.",
            timeout=12,
        )

    assert result["status"] == "error"
    assert result["model"] == GEMINI_MODEL_NAME
    assert result["response_text"] is None
    assert result["tokens"] == {"input": 0, "output": 0}
    assert isinstance(result["duration_seconds"], float)
    assert "error_message" in result
    assert "Rate limit exceeded (HTTP 429)" in result["error_message"]

    validated = ProviderResult.model_validate(result)
    assert validated.status == "error"
    assert validated.error_message is not None


@pytest.mark.asyncio
async def test_gemini_client_auth_error():
    """Verify Gemini auth exception returns status='error' with clear error_message."""
    error_json = {
        "error": {
            "code": 400,
            "message": "API key not valid. Please pass a valid API key.",
            "status": "INVALID_ARGUMENT",
            "details": [{"reason": "API_KEY_INVALID"}],
        }
    }
    mock_client = MagicMock()
    mock_client.aio.models.generate_content = AsyncMock(
        side_effect=ClientError(400, error_json, None)
    )

    with patch("app.services.gemini_client.get_gemini_client", return_value=mock_client):
        result = await call_gemini(
            prompt="Is this covered?",
            system_prompt="Insurance analyst.",
            timeout=12,
        )

    assert result["status"] == "error"
    assert result["model"] == GEMINI_MODEL_NAME
    assert result["response_text"] is None
    assert result["tokens"] == {"input": 0, "output": 0}
    assert "error_message" in result
    assert "Authentication error" in result["error_message"]

    validated = ProviderResult.model_validate(result)
    assert validated.status == "error"


@pytest.mark.asyncio
async def test_gemini_client_generic_error():
    """Verify Gemini base/unexpected exception does not escape and returns status='error'."""
    mock_client = MagicMock()
    mock_client.aio.models.generate_content = AsyncMock(
        side_effect=APIError(500, {"error": {"code": 500, "message": "Backend service unavailable"}}, None)
    )

    with patch("app.services.gemini_client.get_gemini_client", return_value=mock_client):
        result = await call_gemini(
            prompt="Is this covered?",
            system_prompt="Insurance analyst.",
            timeout=12,
        )

    assert result["status"] == "error"
    assert result["response_text"] is None
    assert result["tokens"] == {"input": 0, "output": 0}
    assert "error_message" in result
    assert "Gemini server error (500)" in result["error_message"]

    validated = ProviderResult.model_validate(result)
    assert validated.status == "error"


@pytest.mark.asyncio
async def test_gemini_client_timeout_error():
    """Verify Gemini timeout exception is captured and normalized."""
    mock_client = MagicMock()
    mock_client.aio.models.generate_content = AsyncMock(
        side_effect=asyncio.TimeoutError()
    )

    with patch("app.services.gemini_client.get_gemini_client", return_value=mock_client):
        result = await call_gemini(
            prompt="Timing out prompt",
            system_prompt="Insurance analyst.",
            timeout=12,
        )

    assert result["status"] == "error"
    assert result["response_text"] is None
    assert "Request timed out" in result["error_message"]


# ============================================================================
# 2. OpenAI Client Unit Tests
# ============================================================================

@pytest.mark.asyncio
async def test_openai_client_success():
    """Verify OpenAI client success path returns normalized dict matching ProviderResult."""
    mock_response = MagicMock()
    mock_choice = MagicMock()
    mock_choice.message = MagicMock(content="OpenAI coverage evaluation: Detached garage is covered.")
    mock_response.choices = [mock_choice]
    mock_response.usage = MagicMock(
        prompt_tokens=60,
        completion_tokens=150,
    )

    mock_client = MagicMock()
    mock_client.chat.completions.create = AsyncMock(return_value=mock_response)

    with patch("app.services.openai_client.get_openai_client", return_value=mock_client):
        result = await call_openai(
            prompt="Is the garage door covered?",
            system_prompt="You are an insurance underwriter.",
            timeout=12,
        )

    assert result["status"] == "success"
    assert result["model"] == OPENAI_MODEL_NAME
    assert result["model"] == "gpt-4o-mini"
    assert result["response_text"] == "OpenAI coverage evaluation: Detached garage is covered."
    assert result["tokens"]["input"] == 60
    assert result["tokens"]["output"] == 150
    assert isinstance(result["duration_seconds"], float)
    assert result["duration_seconds"] >= 0.0
    assert "error_message" not in result

    # Validate schema conformity
    validated = ProviderResult.model_validate(result)
    assert validated.status == "success"
    assert validated.tokens["input"] == 60
    assert validated.tokens["output"] == 150


@pytest.mark.asyncio
async def test_openai_client_rate_limit():
    """Verify OpenAI rate limit exception (HTTP 429) returns status='error' with error_message."""
    req = httpx.Request("POST", "https://api.openai.com/v1/chat/completions")
    resp = httpx.Response(429, request=req)
    mock_client = MagicMock()
    mock_client.chat.completions.create = AsyncMock(
        side_effect=openai.RateLimitError(
            message="You exceeded your current quota, please check your plan and billing details.",
            response=resp,
            body={"error": {"message": "quota exceeded"}},
        )
    )

    with patch("app.services.openai_client.get_openai_client", return_value=mock_client):
        result = await call_openai(
            prompt="Is this covered?",
            system_prompt="Insurance analyst.",
            timeout=12,
        )

    assert result["status"] == "error"
    assert result["model"] == OPENAI_MODEL_NAME
    assert result["response_text"] is None
    assert result["tokens"] == {"input": 0, "output": 0}
    assert isinstance(result["duration_seconds"], float)
    assert "error_message" in result
    assert "Rate limit exceeded (HTTP 429)" in result["error_message"]

    validated = ProviderResult.model_validate(result)
    assert validated.status == "error"
    assert validated.error_message is not None


@pytest.mark.asyncio
async def test_openai_client_auth_error():
    """Verify OpenAI auth exception returns status='error' with clear error_message."""
    req = httpx.Request("POST", "https://api.openai.com/v1/chat/completions")
    resp = httpx.Response(401, request=req)
    mock_client = MagicMock()
    mock_client.chat.completions.create = AsyncMock(
        side_effect=openai.AuthenticationError(
            message="Incorrect API key provided: sk-invalid.",
            response=resp,
            body={"error": {"message": "Incorrect API key"}},
        )
    )

    with patch("app.services.openai_client.get_openai_client", return_value=mock_client):
        result = await call_openai(
            prompt="Is this covered?",
            system_prompt="Insurance analyst.",
            timeout=12,
        )

    assert result["status"] == "error"
    assert result["model"] == OPENAI_MODEL_NAME
    assert result["response_text"] is None
    assert result["tokens"] == {"input": 0, "output": 0}
    assert "error_message" in result
    assert "Authentication error" in result["error_message"]

    validated = ProviderResult.model_validate(result)
    assert validated.status == "error"


@pytest.mark.asyncio
async def test_openai_client_generic_error():
    """Verify OpenAI base/unexpected exception does not escape and returns status='error'."""
    mock_client = MagicMock()
    mock_client.chat.completions.create = AsyncMock(
        side_effect=openai.OpenAIError("OpenAI generic internal client issue")
    )

    with patch("app.services.openai_client.get_openai_client", return_value=mock_client):
        result = await call_openai(
            prompt="Is this covered?",
            system_prompt="Insurance analyst.",
            timeout=12,
        )

    assert result["status"] == "error"
    assert result["response_text"] is None
    assert result["tokens"] == {"input": 0, "output": 0}
    assert "error_message" in result
    assert "OpenAI API error: OpenAI generic internal client issue" in result["error_message"]

    validated = ProviderResult.model_validate(result)
    assert validated.status == "error"


@pytest.mark.asyncio
async def test_openai_client_timeout_error():
    """Verify OpenAI timeout exception is captured and normalized."""
    req = httpx.Request("POST", "https://api.openai.com/v1/chat/completions")
    mock_client = MagicMock()
    mock_client.chat.completions.create = AsyncMock(
        side_effect=openai.APITimeoutError(request=req)
    )

    with patch("app.services.openai_client.get_openai_client", return_value=mock_client):
        result = await call_openai(
            prompt="Timing out prompt",
            system_prompt="Insurance analyst.",
            timeout=12,
        )

    assert result["status"] == "error"
    assert result["response_text"] is None
    assert "Request timed out" in result["error_message"]


# ============================================================================
# 3. Claude Client Unit Tests
# ============================================================================

@pytest.mark.asyncio
async def test_claude_client_success():
    """Verify Claude client success path returns normalized dict matching ProviderResult."""
    mock_response = MagicMock()
    mock_content_block = MagicMock()
    mock_content_block.text = "Claude coverage analysis: Replacement cost applies up to limit."
    mock_response.content = [mock_content_block]
    mock_response.usage = MagicMock(
        input_tokens=52,
        output_tokens=142,
    )

    mock_client = MagicMock()
    mock_client.messages.create = AsyncMock(return_value=mock_response)

    with patch("app.services.claude_client.get_claude_client", return_value=mock_client):
        result = await call_claude(
            prompt="Is the garage door covered?",
            system_prompt="You are an insurance underwriter.",
            timeout=12,
        )

    assert result["status"] == "success"
    assert result["model"] == CLAUDE_MODEL_NAME
    assert result["model"] == "claude-haiku-4-5"
    assert result["response_text"] == "Claude coverage analysis: Replacement cost applies up to limit."
    assert result["tokens"]["input"] == 52
    assert result["tokens"]["output"] == 142
    assert isinstance(result["duration_seconds"], float)
    assert result["duration_seconds"] >= 0.0
    assert "error_message" not in result

    # Validate schema conformity
    validated = ProviderResult.model_validate(result)
    assert validated.status == "success"
    assert validated.tokens["input"] == 52
    assert validated.tokens["output"] == 142


@pytest.mark.asyncio
async def test_claude_client_rate_limit():
    """Verify Claude rate limit exception (HTTP 429) returns status='error' with error_message."""
    req = httpx.Request("POST", "https://api.anthropic.com/v1/messages")
    resp = httpx.Response(429, request=req)
    mock_client = MagicMock()
    mock_client.messages.create = AsyncMock(
        side_effect=anthropic.RateLimitError(
            message="Number of request tokens has exceeded your daily rate limit.",
            response=resp,
            body={"error": {"type": "rate_limit_error", "message": "Rate limit exceeded"}},
        )
    )

    with patch("app.services.claude_client.get_claude_client", return_value=mock_client):
        result = await call_claude(
            prompt="Is this covered?",
            system_prompt="Insurance analyst.",
            timeout=12,
        )

    assert result["status"] == "error"
    assert result["model"] == CLAUDE_MODEL_NAME
    assert result["response_text"] is None
    assert result["tokens"] == {"input": 0, "output": 0}
    assert isinstance(result["duration_seconds"], float)
    assert "error_message" in result
    assert "Rate limit exceeded (HTTP 429)" in result["error_message"]

    validated = ProviderResult.model_validate(result)
    assert validated.status == "error"
    assert validated.error_message is not None


@pytest.mark.asyncio
async def test_claude_client_auth_error():
    """Verify Claude auth exception returns status='error' with clear error_message."""
    req = httpx.Request("POST", "https://api.anthropic.com/v1/messages")
    resp = httpx.Response(401, request=req)
    mock_client = MagicMock()
    mock_client.messages.create = AsyncMock(
        side_effect=anthropic.AuthenticationError(
            message="Invalid API Key provided: sk-ant-invalid.",
            response=resp,
            body={"error": {"type": "authentication_error", "message": "Invalid API Key"}},
        )
    )

    with patch("app.services.claude_client.get_claude_client", return_value=mock_client):
        result = await call_claude(
            prompt="Is this covered?",
            system_prompt="Insurance analyst.",
            timeout=12,
        )

    assert result["status"] == "error"
    assert result["model"] == CLAUDE_MODEL_NAME
    assert result["response_text"] is None
    assert result["tokens"] == {"input": 0, "output": 0}
    assert "error_message" in result
    assert "Authentication error" in result["error_message"]

    validated = ProviderResult.model_validate(result)
    assert validated.status == "error"


@pytest.mark.asyncio
async def test_claude_client_generic_error():
    """Verify Claude base/unexpected exception does not escape and returns status='error'."""
    req = httpx.Request("POST", "https://api.anthropic.com/v1/messages")
    mock_client = MagicMock()
    mock_client.messages.create = AsyncMock(
        side_effect=anthropic.APIError(
            "Generic Anthropic backend error",
            request=req,
            body=None,
        )
    )

    with patch("app.services.claude_client.get_claude_client", return_value=mock_client):
        result = await call_claude(
            prompt="Is this covered?",
            system_prompt="Insurance analyst.",
            timeout=12,
        )

    assert result["status"] == "error"
    assert result["response_text"] is None
    assert result["tokens"] == {"input": 0, "output": 0}
    assert "error_message" in result
    assert "Anthropic API error: Generic Anthropic backend error" in result["error_message"]

    validated = ProviderResult.model_validate(result)
    assert validated.status == "error"


@pytest.mark.asyncio
async def test_claude_client_timeout_error():
    """Verify Claude timeout exception is captured and normalized."""
    req = httpx.Request("POST", "https://api.anthropic.com/v1/messages")
    mock_client = MagicMock()
    mock_client.messages.create = AsyncMock(
        side_effect=anthropic.APITimeoutError(request=req)
    )

    with patch("app.services.claude_client.get_claude_client", return_value=mock_client):
        result = await call_claude(
            prompt="Timing out prompt",
            system_prompt="Insurance analyst.",
            timeout=12,
        )

    assert result["status"] == "error"
    assert result["response_text"] is None
    assert "Request timed out" in result["error_message"]


# ============================================================================
# 4. Cross-Client Parity & Contract Verification Tests
# ============================================================================

@pytest.mark.asyncio
async def test_all_clients_success_shape_parity():
    """Verify success return shapes are strictly identical in keys and value types across all providers."""
    # Mock Gemini
    mock_gemini_resp = MagicMock(
        text="Gemini text",
        usage_metadata=MagicMock(prompt_token_count=10, candidates_token_count=20),
    )
    mock_gemini_client = MagicMock()
    mock_gemini_client.aio.models.generate_content = AsyncMock(return_value=mock_gemini_resp)

    # Mock OpenAI
    mock_openai_resp = MagicMock(
        choices=[MagicMock(message=MagicMock(content="OpenAI text"))],
        usage=MagicMock(prompt_tokens=15, completion_tokens=25),
    )
    mock_openai_client = MagicMock()
    mock_openai_client.chat.completions.create = AsyncMock(return_value=mock_openai_resp)

    # Mock Claude
    mock_claude_resp = MagicMock(
        content=[MagicMock(text="Claude text")],
        usage=MagicMock(input_tokens=12, output_tokens=22),
    )
    mock_claude_client = MagicMock()
    mock_claude_client.messages.create = AsyncMock(return_value=mock_claude_resp)

    with (
        patch("app.services.gemini_client.get_gemini_client", return_value=mock_gemini_client),
        patch("app.services.openai_client.get_openai_client", return_value=mock_openai_client),
        patch("app.services.claude_client.get_claude_client", return_value=mock_claude_client),
    ):
        gemini_res = await call_gemini("Prompt", "System")
        openai_res = await call_openai("Prompt", "System")
        claude_res = await call_claude("Prompt", "System")

    expected_keys = {"status", "model", "duration_seconds", "tokens", "response_text"}
    assert set(gemini_res.keys()) == expected_keys
    assert set(openai_res.keys()) == expected_keys
    assert set(claude_res.keys()) == expected_keys

    for res in (gemini_res, openai_res, claude_res):
        assert res["status"] == "success"
        assert isinstance(res["model"], str)
        assert isinstance(res["duration_seconds"], float)
        assert isinstance(res["tokens"], dict)
        assert set(res["tokens"].keys()) == {"input", "output"}
        assert isinstance(res["tokens"]["input"], int)
        assert isinstance(res["tokens"]["output"], int)
        assert isinstance(res["response_text"], str)


@pytest.mark.asyncio
async def test_all_clients_error_shape_parity():
    """Verify error return shapes are strictly identical in keys and value types across all providers."""
    mock_gemini_client = MagicMock()
    mock_gemini_client.aio.models.generate_content = AsyncMock(side_effect=Exception("Gemini fail"))

    mock_openai_client = MagicMock()
    mock_openai_client.chat.completions.create = AsyncMock(side_effect=Exception("OpenAI fail"))

    mock_claude_client = MagicMock()
    mock_claude_client.messages.create = AsyncMock(side_effect=Exception("Claude fail"))

    with (
        patch("app.services.gemini_client.get_gemini_client", return_value=mock_gemini_client),
        patch("app.services.openai_client.get_openai_client", return_value=mock_openai_client),
        patch("app.services.claude_client.get_claude_client", return_value=mock_claude_client),
    ):
        gemini_res = await call_gemini("Prompt", "System")
        openai_res = await call_openai("Prompt", "System")
        claude_res = await call_claude("Prompt", "System")

    expected_keys = {"status", "model", "duration_seconds", "tokens", "response_text", "error_message"}
    assert set(gemini_res.keys()) == expected_keys
    assert set(openai_res.keys()) == expected_keys
    assert set(claude_res.keys()) == expected_keys

    for res in (gemini_res, openai_res, claude_res):
        assert res["status"] == "error"
        assert isinstance(res["model"], str)
        assert isinstance(res["duration_seconds"], float)
        assert res["tokens"] == {"input": 0, "output": 0}
        assert res["response_text"] is None
        assert isinstance(res["error_message"], str)
