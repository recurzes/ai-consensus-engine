"""Unit and concurrency tests for worker orchestration and fault tolerance.

Phase 9: Automated Tests
Specification: notes/backend-spec/28-test-orchestration-and-fault-tolerance-tests.md

Tests verify:
1. partition_results correctly partitions worker results into successful and failed
   providers across all scenarios (all-success, 1-fail, 2-fail, all-fail, bare exceptions).
2. run_workers executes provider calls concurrently such that total latency
   approximates the slowest provider (not the sum).
3. Fault-tolerance mechanisms isolate errors, exceptions, and timeouts.
"""

import asyncio
import os
import time
from unittest.mock import patch

# Ensure test env defaults exist before loading settings or orchestrator
_TEST_ENV_DEFAULTS = {
    "GEMINI_API_KEY": "test-gemini-key",
    "OPENAI_API_KEY": "test-openai-key",
    "ANTHROPIC_API_KEY": "test-anthropic-key",
    "ARBITER_MODEL_PROVIDER": "gemini",
}
for k, v in _TEST_ENV_DEFAULTS.items():
    os.environ.setdefault(k, v)

import pytest

from app.schemas.models import ProviderResult
from app.services.orchestrator import (
    CLAUDE_MODEL_NAME,
    GEMINI_MODEL_NAME,
    OPENAI_MODEL_NAME,
    partition_results,
    run_workers,
)


# ============================================================================
# Fixtures & Helpers
# ============================================================================

@pytest.fixture
def mock_gemini_success() -> dict:
    return {
        "status": "success",
        "model": GEMINI_MODEL_NAME,
        "duration_seconds": 0.12,
        "tokens": {"input": 15, "output": 25},
        "response_text": "Gemini response text for auto coverage.",
    }


@pytest.fixture
def mock_openai_success() -> dict:
    return {
        "status": "success",
        "model": OPENAI_MODEL_NAME,
        "duration_seconds": 0.18,
        "tokens": {"input": 18, "output": 30},
        "response_text": "OpenAI response text for auto coverage.",
    }


@pytest.fixture
def mock_claude_success() -> dict:
    return {
        "status": "success",
        "model": CLAUDE_MODEL_NAME,
        "duration_seconds": 0.22,
        "tokens": {"input": 14, "output": 28},
        "response_text": "Claude response text for auto coverage.",
    }


@pytest.fixture
def mock_gemini_error() -> dict:
    return {
        "status": "error",
        "model": GEMINI_MODEL_NAME,
        "duration_seconds": 0.05,
        "tokens": {"input": 0, "output": 0},
        "response_text": None,
        "error_message": "Gemini rate limit exceeded (HTTP 429)",
    }


@pytest.fixture
def mock_openai_error() -> dict:
    return {
        "status": "error",
        "model": OPENAI_MODEL_NAME,
        "duration_seconds": 0.05,
        "tokens": {"input": 0, "output": 0},
        "response_text": None,
        "error_message": "OpenAI authentication failed",
    }


@pytest.fixture
def mock_claude_error() -> dict:
    return {
        "status": "error",
        "model": CLAUDE_MODEL_NAME,
        "duration_seconds": 0.05,
        "tokens": {"input": 0, "output": 0},
        "response_text": None,
        "error_message": "Claude service unavailable (HTTP 500)",
    }


# ============================================================================
# 1. Partition Results Tests (Pure Logic)
# ============================================================================

def test_partition_results_all_success_path(
    mock_gemini_success, mock_openai_success, mock_claude_success
):
    """Scenario 1: All 3 succeed -> successful_providers has 3, failed_providers is empty."""
    raw_results = [mock_gemini_success, mock_openai_success, mock_claude_success]
    successful_results, failed_results, successful_providers, failed_providers = (
        partition_results(raw_results)
    )

    assert successful_providers == ["gemini", "openai", "claude"]
    assert failed_providers == []
    assert len(successful_results) == 3
    assert len(failed_results) == 0
    assert successful_results == raw_results


def test_partition_results_one_failure_path_claude(
    mock_gemini_success, mock_openai_success, mock_claude_error
):
    """Scenario 2a: 2 succeed + 1 failure (Claude fails) -> successful has 2, failed has 1."""
    raw_results = [mock_gemini_success, mock_openai_success, mock_claude_error]
    successful_results, failed_results, successful_providers, failed_providers = (
        partition_results(raw_results)
    )

    assert successful_providers == ["gemini", "openai"]
    assert failed_providers == ["claude"]
    assert len(successful_results) == 2
    assert len(failed_results) == 1
    assert successful_results == [mock_gemini_success, mock_openai_success]
    assert failed_results == [mock_claude_error]


def test_partition_results_one_failure_path_openai(
    mock_gemini_success, mock_openai_error, mock_claude_success
):
    """Scenario 2b: 2 succeed + 1 failure (OpenAI fails) -> successful has 2, failed has 1."""
    raw_results = [mock_gemini_success, mock_openai_error, mock_claude_success]
    successful_results, failed_results, successful_providers, failed_providers = (
        partition_results(raw_results)
    )

    assert successful_providers == ["gemini", "claude"]
    assert failed_providers == ["openai"]
    assert len(successful_results) == 2
    assert len(failed_results) == 1
    assert successful_results == [mock_gemini_success, mock_claude_success]
    assert failed_results == [mock_openai_error]


def test_partition_results_one_failure_path_gemini(
    mock_gemini_error, mock_openai_success, mock_claude_success
):
    """Scenario 2c: 2 succeed + 1 failure (Gemini fails) -> successful has 2, failed has 1."""
    raw_results = [mock_gemini_error, mock_openai_success, mock_claude_success]
    successful_results, failed_results, successful_providers, failed_providers = (
        partition_results(raw_results)
    )

    assert successful_providers == ["openai", "claude"]
    assert failed_providers == ["gemini"]
    assert len(successful_results) == 2
    assert len(failed_results) == 1
    assert successful_results == [mock_openai_success, mock_claude_success]
    assert failed_results == [mock_gemini_error]


def test_partition_results_two_failure_path_gemini_succeeds(
    mock_gemini_success, mock_openai_error, mock_claude_error
):
    """Scenario 3a: 1 succeeds (Gemini) + 2 failures -> successful has 1, failed has 2."""
    raw_results = [mock_gemini_success, mock_openai_error, mock_claude_error]
    successful_results, failed_results, successful_providers, failed_providers = (
        partition_results(raw_results)
    )

    assert successful_providers == ["gemini"]
    assert failed_providers == ["openai", "claude"]
    assert len(successful_results) == 1
    assert len(failed_results) == 2
    assert successful_results == [mock_gemini_success]
    assert failed_results == [mock_openai_error, mock_claude_error]


def test_partition_results_two_failure_path_openai_succeeds(
    mock_gemini_error, mock_openai_success, mock_claude_error
):
    """Scenario 3b: 1 succeeds (OpenAI) + 2 failures -> successful has 1, failed has 2."""
    raw_results = [mock_gemini_error, mock_openai_success, mock_claude_error]
    successful_results, failed_results, successful_providers, failed_providers = (
        partition_results(raw_results)
    )

    assert successful_providers == ["openai"]
    assert failed_providers == ["gemini", "claude"]
    assert len(successful_results) == 1
    assert len(failed_results) == 2
    assert successful_results == [mock_openai_success]
    assert failed_results == [mock_gemini_error, mock_claude_error]


def test_partition_results_two_failure_path_claude_succeeds(
    mock_gemini_error, mock_openai_error, mock_claude_success
):
    """Scenario 3c: 1 succeeds (Claude) + 2 failures -> successful has 1, failed has 2."""
    raw_results = [mock_gemini_error, mock_openai_error, mock_claude_success]
    successful_results, failed_results, successful_providers, failed_providers = (
        partition_results(raw_results)
    )

    assert successful_providers == ["claude"]
    assert failed_providers == ["gemini", "openai"]
    assert len(successful_results) == 1
    assert len(failed_results) == 2
    assert successful_results == [mock_claude_success]
    assert failed_results == [mock_gemini_error, mock_openai_error]


def test_partition_results_all_failure_path(
    mock_gemini_error, mock_openai_error, mock_claude_error
):
    """Scenario 4: All 3 fail -> successful_providers == [], failed_providers == ["gemini", "openai", "claude"]."""
    raw_results = [mock_gemini_error, mock_openai_error, mock_claude_error]
    successful_results, failed_results, successful_providers, failed_providers = (
        partition_results(raw_results)
    )

    assert successful_providers == []
    assert failed_providers == ["gemini", "openai", "claude"]
    assert len(successful_results) == 0
    assert len(failed_results) == 3
    assert failed_results == raw_results


def test_partition_results_single_bare_exception_passthrough(
    mock_gemini_success, mock_claude_success
):
    """Scenario 5a: A bare Python exception in the results list is treated as a failure."""
    exc = RuntimeError("Worker network failure")
    raw_results = [mock_gemini_success, exc, mock_claude_success]
    successful_results, failed_results, successful_providers, failed_providers = (
        partition_results(raw_results)
    )

    assert successful_providers == ["gemini", "claude"]
    assert failed_providers == ["openai"]
    assert len(successful_results) == 2
    assert len(failed_results) == 1
    assert failed_results[0] is exc


def test_partition_results_all_bare_exceptions_passthrough():
    """Scenario 5b: Multiple bare exceptions are all categorized as failures with ordered providers."""
    exc_gemini = TimeoutError("Gemini timed out")
    exc_openai = ConnectionResetError("OpenAI connection reset")
    exc_claude = RuntimeError("Claude internal failure")

    raw_results = [exc_gemini, exc_openai, exc_claude]
    successful_results, failed_results, successful_providers, failed_providers = (
        partition_results(raw_results)
    )

    assert successful_providers == []
    assert failed_providers == ["gemini", "openai", "claude"]
    assert len(successful_results) == 0
    assert len(failed_results) == 3
    assert failed_results == [exc_gemini, exc_openai, exc_claude]


def test_partition_results_mixed_exceptions_and_error_dicts(
    mock_gemini_error, mock_claude_success
):
    """Scenario 5c: Mixed error dicts and bare exceptions handled correctly."""
    exc_openai = Exception("Unhandled OpenAI thread error")
    raw_results = [mock_gemini_error, exc_openai, mock_claude_success]
    successful_results, failed_results, successful_providers, failed_providers = (
        partition_results(raw_results)
    )

    assert successful_providers == ["claude"]
    assert failed_providers == ["gemini", "openai"]
    assert len(successful_results) == 1
    assert len(failed_results) == 2
    assert failed_results == [mock_gemini_error, exc_openai]


def test_partition_results_success_status_with_none_response_text_fails(
    mock_openai_success, mock_claude_success
):
    """Invalid result with status='success' but response_text=None is classified as failed."""
    invalid_gemini = {
        "status": "success",
        "model": GEMINI_MODEL_NAME,
        "duration_seconds": 0.1,
        "tokens": {"input": 10, "output": 0},
        "response_text": None,
    }
    raw_results = [invalid_gemini, mock_openai_success, mock_claude_success]
    successful_results, failed_results, successful_providers, failed_providers = (
        partition_results(raw_results)
    )

    assert successful_providers == ["openai", "claude"]
    assert failed_providers == ["gemini"]
    assert len(successful_results) == 2
    assert len(failed_results) == 1
    assert failed_results[0] == invalid_gemini


def test_partition_results_empty_list():
    """Empty results list returns 4 empty lists."""
    successful_results, failed_results, successful_providers, failed_providers = (
        partition_results([])
    )
    assert successful_results == []
    assert failed_results == []
    assert successful_providers == []
    assert failed_providers == []


# ============================================================================
# 2. Concurrency & Latency Tests (run_workers)
# ============================================================================

@pytest.mark.asyncio
async def test_worker_latency_is_not_sum(
    mock_gemini_success, mock_openai_success, mock_claude_success
):
    """Scenario 6: Total duration should approximate the slowest provider, not the sum.

    Controlled mock timing:
    - Gemini: 0.1s
    - OpenAI: 0.2s
    - Claude: 0.5s (slowest)

    Sum would be ~0.8s (0.1 + 0.2 + 0.5).
    Concurrent execution completes in ~0.5s (slowest), well below 0.70s.
    """
    async def slow_gemini(*args, **kwargs):
        await asyncio.sleep(0.1)
        return mock_gemini_success

    async def slow_openai(*args, **kwargs):
        await asyncio.sleep(0.2)
        return mock_openai_success

    async def slow_claude(*args, **kwargs):
        await asyncio.sleep(0.5)  # slowest
        return mock_claude_success

    with (
        patch("app.services.orchestrator.call_gemini", side_effect=slow_gemini),
        patch("app.services.orchestrator.call_openai", side_effect=slow_openai),
        patch("app.services.orchestrator.call_claude", side_effect=slow_claude),
    ):
        start = time.perf_counter()
        results = await run_workers("prompt", "system_prompt")
        duration = time.perf_counter() - start

    # Duration must be ~0.5s (slowest), not ~0.8s (sum)
    assert duration >= 0.45, f"Expected duration >= 0.45s, got {duration:.4f}s"
    assert duration < 0.70, f"Expected duration < 0.70s (not sum ~0.8s), got {duration:.4f}s"
    assert len(results) == 3

    # Ensure result order and content
    assert results[0]["model"] == GEMINI_MODEL_NAME
    assert results[1]["model"] == OPENAI_MODEL_NAME
    assert results[2]["model"] == CLAUDE_MODEL_NAME

    for res in results:
        validated = ProviderResult.model_validate(res)
        assert validated.status == "success"


@pytest.mark.asyncio
async def test_worker_latency_gemini_slowest(
    mock_gemini_success, mock_openai_success, mock_claude_success
):
    """Verify concurrency holds when Gemini (index 0) is the slowest provider.

    Controlled mock timing:
    - Gemini: 0.4s (slowest)
    - OpenAI: 0.1s
    - Claude: 0.2s

    Sum would be ~0.7s. Concurrent execution is ~0.4s (< 0.60s).
    """
    async def slow_gemini(*args, **kwargs):
        await asyncio.sleep(0.4)
        return mock_gemini_success

    async def slow_openai(*args, **kwargs):
        await asyncio.sleep(0.1)
        return mock_openai_success

    async def slow_claude(*args, **kwargs):
        await asyncio.sleep(0.2)
        return mock_claude_success

    with (
        patch("app.services.orchestrator.call_gemini", side_effect=slow_gemini),
        patch("app.services.orchestrator.call_openai", side_effect=slow_openai),
        patch("app.services.orchestrator.call_claude", side_effect=slow_claude),
    ):
        start = time.perf_counter()
        results = await run_workers("prompt", "system_prompt")
        duration = time.perf_counter() - start

    assert duration >= 0.35, f"Expected duration >= 0.35s, got {duration:.4f}s"
    assert duration < 0.60, f"Expected duration < 0.60s (not sum ~0.7s), got {duration:.4f}s"
    assert len(results) == 3


@pytest.mark.asyncio
async def test_worker_latency_openai_slowest(
    mock_gemini_success, mock_openai_success, mock_claude_success
):
    """Verify concurrency holds when OpenAI (index 1) is the slowest provider.

    Controlled mock timing:
    - Gemini: 0.1s
    - OpenAI: 0.4s (slowest)
    - Claude: 0.2s

    Sum would be ~0.7s. Concurrent execution is ~0.4s (< 0.60s).
    """
    async def slow_gemini(*args, **kwargs):
        await asyncio.sleep(0.1)
        return mock_gemini_success

    async def slow_openai(*args, **kwargs):
        await asyncio.sleep(0.4)
        return mock_openai_success

    async def slow_claude(*args, **kwargs):
        await asyncio.sleep(0.2)
        return mock_claude_success

    with (
        patch("app.services.orchestrator.call_gemini", side_effect=slow_gemini),
        patch("app.services.orchestrator.call_openai", side_effect=slow_openai),
        patch("app.services.orchestrator.call_claude", side_effect=slow_claude),
    ):
        start = time.perf_counter()
        results = await run_workers("prompt", "system_prompt")
        duration = time.perf_counter() - start

    assert duration >= 0.35, f"Expected duration >= 0.35s, got {duration:.4f}s"
    assert duration < 0.60, f"Expected duration < 0.60s (not sum ~0.7s), got {duration:.4f}s"
    assert len(results) == 3


@pytest.mark.asyncio
async def test_run_workers_fault_tolerance_with_mixed_outcomes(
    mock_gemini_success, mock_claude_error
):
    """Verify concurrent execution completes properly when tasks fail or raise exceptions.

    - Gemini succeeds after 0.1s.
    - OpenAI raises bare RuntimeError after 0.2s.
    - Claude returns error dict after 0.4s (slowest).

    Total time ~0.4s (slowest). All 3 results captured. Partition isolates the single success.
    """
    async def slow_gemini(*args, **kwargs):
        await asyncio.sleep(0.1)
        return mock_gemini_success

    async def crash_openai(*args, **kwargs):
        await asyncio.sleep(0.2)
        raise RuntimeError("Unexpected OpenAI crash during streaming")

    async def slow_claude_error(*args, **kwargs):
        await asyncio.sleep(0.4)
        return mock_claude_error

    with (
        patch("app.services.orchestrator.call_gemini", side_effect=slow_gemini),
        patch("app.services.orchestrator.call_openai", side_effect=crash_openai),
        patch("app.services.orchestrator.call_claude", side_effect=slow_claude_error),
    ):
        start = time.perf_counter()
        results = await run_workers("prompt", "system_prompt")
        duration = time.perf_counter() - start

    assert duration >= 0.35
    assert duration < 0.60
    assert len(results) == 3

    # Gemini success
    assert results[0]["status"] == "success"
    # OpenAI exception caught by return_exceptions=True
    assert isinstance(results[1], RuntimeError)
    assert str(results[1]) == "Unexpected OpenAI crash during streaming"
    # Claude error dict
    assert results[2]["status"] == "error"

    # Verify partition_results seamlessly classifies the combined output
    successful_results, failed_results, successful_providers, failed_providers = (
        partition_results(results)
    )
    assert successful_providers == ["gemini"]
    assert failed_providers == ["openai", "claude"]
    assert len(successful_results) == 1
    assert len(failed_results) == 2


@pytest.mark.asyncio
async def test_run_workers_timeout_isolation_under_concurrency(
    mock_gemini_success, mock_claude_success
):
    """Verify timeout on one slow provider does not block or fail faster providers.

    - Gemini: 0.05s (success)
    - OpenAI: 1.0s (hangs past timeout=0.15s)
    - Claude: 0.08s (success)

    Total execution duration ≈ 0.15s (timeout), not 1.0s.
    Gemini & Claude succeed, OpenAI returns normalized timeout error dict.
    """
    async def fast_gemini(*args, **kwargs):
        await asyncio.sleep(0.05)
        return mock_gemini_success

    async def hanging_openai(*args, **kwargs):
        await asyncio.sleep(1.0)
        return {"status": "success", "model": OPENAI_MODEL_NAME}

    async def fast_claude(*args, **kwargs):
        await asyncio.sleep(0.08)
        return mock_claude_success

    with (
        patch("app.services.orchestrator.call_gemini", side_effect=fast_gemini),
        patch("app.services.orchestrator.call_openai", side_effect=hanging_openai),
        patch("app.services.orchestrator.call_claude", side_effect=fast_claude),
    ):
        start = time.perf_counter()
        results = await run_workers("prompt", "system_prompt", timeout=0.15)
        duration = time.perf_counter() - start

    assert duration >= 0.12, f"Expected duration >= 0.12s, got {duration:.4f}s"
    assert duration < 0.35, f"Expected duration < 0.35s (timeout at 0.15s, not 1.0s), got {duration:.4f}s"
    assert len(results) == 3

    assert results[0]["status"] == "success"
    assert results[1]["status"] == "error"
    assert "timed out after 0.15 seconds" in results[1]["error_message"]
    assert results[2]["status"] == "success"

    # Schema validation
    for res in results:
        validated = ProviderResult.model_validate(res)
        assert validated.status in ("success", "error")
