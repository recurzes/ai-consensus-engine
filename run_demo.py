"""AI Consensus Engine - Standalone Demo Script.

Architecture Note:
------------------
This script communicates with the AI Consensus Engine by issuing HTTP POST
requests to the running API server (default: http://localhost:8000/api/v1/consensus)
using `httpx`.

We chose the HTTP client approach over direct internal function calls for three reasons:
1. End-to-End Validation: Exercises the complete production path including FastAPI
   routing, Pydantic request/response schema parsing, and global exception handlers.
2. Decoupling: Demonstrates how external consumers and clients interact with the service.
3. Resilience & Graceful Errors: Catches connection issues (`httpx.ConnectError`, timeouts)
   and provides clear, actionable terminal instructions instead of raw tracebacks.
"""

from __future__ import annotations

import os
import sys
from typing import Any

from dotenv import load_dotenv
import httpx

# Optional rich styling support for enhanced terminal output
try:
    from rich.console import Console

    HAS_RICH = True
except ImportError:
    HAS_RICH = False


def get_console(stderr: bool = False) -> Console | None:
    """Return a Rich Console instance bound to current stdout/stderr."""
    if not HAS_RICH:
        return None
    target_stream = sys.stderr if stderr else sys.stdout
    return Console(file=target_stream, highlight=False)


# Constants
DEFAULT_API_URL = os.getenv("CONSENSUS_API_URL", "http://localhost:8000/api/v1/consensus")
DIVIDER_LENGTH = 60
DIVIDER_CHAR = "="
DIVIDER = DIVIDER_CHAR * DIVIDER_LENGTH

REQUIRED_ENV_VARS = (
    "GEMINI_API_KEY",
    "OPENAI_API_KEY",
    "ANTHROPIC_API_KEY",
    "ARBITER_MODEL_PROVIDER",
)

VALID_ARBITER_PROVIDERS = ("gemini", "openai")

# Extensible Scenario Registry:
# - Scenario 1: Consumer Translation (layman_linguist)
# - Scenario 2: Underwriting Analysis (claims_adjuster, MT)
# - Scenario 3: Failover Verification
SCENARIO_1: dict[str, Any] = {
    "prompt": (
        "My homeowners policy has an 'Ordinance or Law' endorsement. "
        "What does that mean and when would it actually pay out?"
    ),
    "context": {
        "role": "layman_linguist",
        "line_of_business": "homeowners",
        "state": "MT",
    },
}

SCENARIO_2: dict[str, Any] = {
    "prompt": (
        "An insured backed a trailer into their garage door, damaging both the door "
        "and the trailer. How are these damages covered under the homeowners policy?"
    ),
    "context": {
        "role": "claims_adjuster",
        "line_of_business": "homeowners",
        "state": "MT",
    },
}

SCENARIOS: list[tuple[str, dict[str, Any]]] = [
    ("Scenario 1: Consumer Translation (layman_linguist)", SCENARIO_1),
    ("Scenario 2: Underwriting / Coverage Analysis (claims_adjuster, MT)", SCENARIO_2),
]


def print_divider(char: str = DIVIDER_CHAR, length: int = DIVIDER_LENGTH, stderr: bool = False) -> None:
    """Print a horizontal divider rule."""
    line = char * length
    console = get_console(stderr=stderr)
    if console is not None:
        console.print(f"[bold cyan]{line}[/bold cyan]")
    else:
        stream = sys.stderr if stderr else sys.stdout
        stream.write(f"{line}\n")


def print_header(title: str, length: int = DIVIDER_LENGTH) -> None:
    """Print a formatted header enclosed by horizontal dividers."""
    print_divider(DIVIDER_CHAR, length)
    console = get_console()
    if console is not None:
        console.print(f"[bold white]{title}[/bold white]")
    else:
        print(title)
    print_divider(DIVIDER_CHAR, length)


def print_error(message: str) -> None:
    """Print a formatted error message to stderr."""
    console = get_console(stderr=True)
    if console is not None:
        console.print(f"[bold red]{message}[/bold red]")
    else:
        sys.stderr.write(f"{message}\n")


def format_telemetry(telemetry: dict[str, Any]) -> str:
    """Format the telemetry response into a human-readable summary string."""
    duration = telemetry.get("total_duration_seconds", 0.0)
    cost = telemetry.get("total_estimated_cost_usd", 0.0)
    successful = telemetry.get("successful_providers") or []
    failed = telemetry.get("failed_providers") or []

    success_str = ", ".join(successful)
    failed_str = ", ".join(failed)

    return (
        f"Telemetry:\n"
        f"  Total Duration:  {duration:.2f}s\n"
        f"  Estimated Cost:  ${cost:.5f}\n"
        f"  Successful:      [{success_str}]\n"
        f"  Failed:          [{failed_str}]"
    )


def validate_environment(
    exit_on_error: bool = True,
    env_file: str | None = None,
    load_env: bool = True,
) -> bool:
    """Validate that required environment variables and API keys are present.

    Loads `.env` via `python-dotenv` if not already loaded. If any required keys
    are missing or empty, prints a human-readable message and exits gracefully.

    Args:
        exit_on_error: If True, calls `sys.exit(1)` on validation failure.
        env_file: Optional explicit path to .env file to load.
        load_env: If True, attempts to load variables from .env file.

    Returns:
        bool: True if environment is valid, False otherwise.
    """
    if load_env:
        if env_file:
            load_dotenv(dotenv_path=env_file, override=False)
        else:
            load_dotenv(override=False)

    missing_keys: list[str] = []
    for var in REQUIRED_ENV_VARS:
        value = os.getenv(var)
        if not value or not value.strip():
            missing_keys.append(var)

    arbiter_provider = os.getenv("ARBITER_MODEL_PROVIDER", "").strip().lower()
    invalid_arbiter = (
        bool(arbiter_provider) and arbiter_provider not in VALID_ARBITER_PROVIDERS
    )

    if missing_keys or invalid_arbiter:
        print_divider("-", DIVIDER_LENGTH, stderr=True)
        print_error("ENVIRONMENT VALIDATION ERROR:")
        if missing_keys:
            for key in missing_keys:
                print_error(f"  - Missing or empty required configuration: {key}")
        if invalid_arbiter:
            print_error(
                f"  - Invalid ARBITER_MODEL_PROVIDER: '{arbiter_provider}'. "
                f"Must be one of: {list(VALID_ARBITER_PROVIDERS)}"
            )
        print_divider("-", DIVIDER_LENGTH, stderr=True)
        print_error(
            "\nPlease ensure your '.env' file is created and properly configured.\n"
            "You can copy '.env.example' to '.env' and fill in the required API keys:\n"
            "  cp .env.example .env\n"
        )
        if exit_on_error:
            sys.exit(1)
        return False

    return True


def run_scenario(
    scenario_name: str,
    payload: dict[str, Any],
    endpoint_url: str = DEFAULT_API_URL,
    timeout: float = 60.0,
) -> dict[str, Any] | None:
    """Execute a single scenario by posting to the consensus endpoint.

    Prints formatted headers, consensus answer, and telemetry data.

    Args:
        scenario_name: Human-readable name for the scenario.
        payload: JSON request payload conforming to ConsensusRequest schema.
        endpoint_url: URL to the consensus endpoint.
        timeout: Request timeout in seconds.

    Returns:
        dict[str, Any] | None: Response dictionary on success, or None on failure.
    """
    print_header(scenario_name)

    prompt = payload.get("prompt", "")
    if prompt:
        console = get_console()
        if console is not None:
            console.print(f"Prompt: {prompt}\n")
        else:
            print(f"Prompt: {prompt}\n")

    try:
        with httpx.Client(timeout=timeout) as client:
            response = client.post(endpoint_url, json=payload)
    except httpx.ConnectError:
        print_error(
            f"\n[ERROR] Could not connect to API server at:\n  {endpoint_url}\n\n"
            "Please ensure the FastAPI server is running locally. You can start it with:\n"
            "  uvicorn app.main:app --reload\n"
        )
        print_divider(stderr=True)
        return None
    except httpx.TimeoutException:
        print_error(
            f"\n[ERROR] Request timed out after {timeout} seconds waiting for {endpoint_url}.\n"
        )
        print_divider(stderr=True)
        return None
    except httpx.RequestError as exc:
        print_error(f"\n[ERROR] Network error while calling {endpoint_url}: {exc}\n")
        print_divider(stderr=True)
        return None

    if response.status_code != 200:
        print_error(
            f"\n[ERROR] API request failed with status HTTP {response.status_code}:\n"
            f"{response.text}\n"
        )
        print_divider(stderr=True)
        return None

    try:
        data: dict[str, Any] = response.json()
    except Exception as exc:
        print_error(f"\n[ERROR] Failed to parse API JSON response: {exc}\n")
        print_divider(stderr=True)
        return None

    consensus_answer = data.get("consensus_answer", "(No consensus answer returned)")
    telemetry = data.get("telemetry", {})

    print("Consensus Answer:")
    console = get_console()
    if console is not None:
        console.print(f"[white]{consensus_answer}[/white]")
    else:
        print(consensus_answer)
    print()

    print(format_telemetry(telemetry))
    print_divider()

    return data


def main() -> None:
    """Entrypoint orchestrating environment validation and scenario execution."""
    validate_environment(exit_on_error=True)

    if not SCENARIOS:
        print_header("AI Consensus Engine - Demo Runner")
        print("Environment validated successfully.")
        print(
            "No demo scenarios registered yet. Scenarios 1-3 will be added in subsequent branches:\n"
            "  - Scenario 1: Consumer Translation (layman_linguist)\n"
            "  - Scenario 2: Underwriting Analysis (claims_adjuster, MT)\n"
            "  - Scenario 3: Failover Verification\n"
        )
        print_divider()
        return

    for idx, (scenario_name, payload) in enumerate(SCENARIOS):
        if idx > 0:
            print()
        run_scenario(scenario_name, payload)


if __name__ == "__main__":
    main()
