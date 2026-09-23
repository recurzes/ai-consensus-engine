# Architecture Decision Records (ADR)

This document records key architectural decisions made for the AI Consensus Engine.

---

## ADR 001: Provider Integration Strategy (Native SDKs vs litellm)

- **Date:** 2026-09-23
- **Status:** Accepted
- **Phase:** 0 — Project Foundation (`chore/decide-provider-integration-strategy`)
- **Deciders:** Engineering Team

---

### Context & Problem Statement

The AI Consensus Engine orchestrates parallel queries across multiple Large Language Model (LLM) providers—specifically **Google Gemini**, **OpenAI**, and **Anthropic Claude**—and synthesizes their answers through an arbiter.

To build the provider clients (Phase 3) and worker orchestration (Phase 4), we must establish whether to interact with each provider using their **official native SDKs** (`google-genai`, `openai`, `anthropic`) or through a **unified abstraction proxy** (`litellm`).

Key architectural requirements include:
1. **Custom System Prompts:** Passing distinct system prompts alongside user queries across differing provider conventions.
2. **Deterministic Token & Duration Telemetry:** Capturing exact input and output token counts and execution duration for pricing calculation (Phase 4).
3. **Robust Error Normalization:** Identifying and normalizing rate limits (HTTP 429), authentication errors, client timeouts, and network failures into a uniform failure shape without crashing workers.
4. **Mockability & Unit Testing:** Isolating each provider client with reliable unit test fixtures in Phase 5.
5. **Support for Target Models:** First-class support for `gemini-2.5-flash`, `gpt-4o-mini`, `claude-3-5-haiku`, `gemini-2.5-pro`, and `gpt-4o`.

---

### Options Evaluated

#### Option A: Native SDKs per Provider
Use `google-genai`, `openai`, and `anthropic` directly within dedicated modules (`app/services/gemini_client.py`, `app/services/openai_client.py`, `app/services/claude_client.py`).

- **Pros:**
  - **Direct API & Parameter Control:** Each provider handles system instructions differently (Gemini uses `GenerateContentConfig(system_instruction=...)`, OpenAI uses a `{"role": "system", ...}` message, Anthropic uses the top-level `system` parameter). Native SDKs handle these cleanly without translation friction.
  - **Accurate Telemetry:** Direct access to native token usage metadata (`response.usage_metadata.prompt_token_count` / `candidates_token_count` for Gemini; `response.usage.prompt_tokens` / `completion_tokens` for OpenAI; `response.usage.input_tokens` / `output_tokens` for Anthropic).
  - **Fine-Grained Typed Exceptions:** Allows catching exact SDK error classes (e.g., `openai.RateLimitError`, `openai.AuthenticationError`, `anthropic.RateLimitError`, `anthropic.AuthenticationError`, `google.genai.errors.APIError`) to normalize status codes and error messages deterministically.
  - **No Upstream Abstraction Lag:** Zero waiting period for third-party library releases when providers publish model updates, new features, or deprecations.
  - **Lean Dependency Footprint:** Keeps the virtual environment minimal without the large transitive dependency footprint brought in by proxy libraries.
  - **Isolated Unit Testing:** Standard async clients (`AsyncOpenAI`, `AsyncAnthropic`, `google.genai.Client`) can be mocked independently without intercepting global proxy calls.

- **Cons:**
  - Small amount of boilerplate across the three client modules to normalize their return shapes to the engine's internal contract.

#### Option B: Unified Abstraction Proxy (`litellm`)
Use `litellm` as a single gateway for calling all providers through an OpenAI-compatible function interface.

- **Pros:**
  - Single API surface for all provider calls, reducing boilerplate across client modules.

- **Cons:**
  - **Abstraction Leakage & Version Drift:** Error mapping, usage schemas, and parameter translation depend on `litellm`'s internal mapping logic, which can introduce subtle bugs or mismatches when upstream provider APIs change.
  - **Dependency Bloat:** Pulls in dozens of transitive dependencies that increase image size and heighten risk of version conflicts with FastAPI and Pydantic v2.
  - **Model Identifier Mapping Overhead:** Requires verifying that newer model identifiers (e.g. `gemini-2.5-flash`, `claude-3-5-haiku`) and their pricing/usage objects are fully supported and mapped identically in litellm.
  - **Complex Error Normalization:** Catching and distinguishing genuine upstream provider HTTP 429s, credential errors, and timeouts becomes coupled to litellm's custom exception hierarchy.

---

### Decision

**Adopt Option A: Native SDKs per provider (`google-genai`, `openai`, `anthropic`).**

---

### Rationale

1. **Precision in Telemetry and Cost Calculation:** Accurately calculating cost in Phase 4 requires exact, trusted token metrics directly from provider responses. Native SDKs guarantee the usage fields are authoritative.
2. **Fault Tolerance and Error Normalization:** Phase 4 orchestration and graceful degradation depend on reliable detection of transient vs. terminal failures (such as the Anthropic failure scenario in Phase 5). Native exception classes allow strict, explicit `try/except` blocks per provider.
3. **Architectural Clarity:** Maintaining three focused, decoupled client files (`gemini_client.py`, `openai_client.py`, `claude_client.py`) that adhere to a single common output shape (`ProviderResult`) provides cleaner separation of concerns than a centralized multi-provider wrapper.
4. **Current Dependency Alignment:** The repository already contains pinned versions of `google-genai==2.25.0`, `openai==3.19.0`, and `anthropic==1.8.0` in `requirements.txt`.

---

### Consequences & Next Steps

1. **Dependencies:**
   - Retain `google-genai`, `openai`, and `anthropic` in `requirements.txt`.
   - Do **not** install or add `litellm`.
2. **Phase 3 Provider Clients:**
   - Implement `app/services/gemini_client.py` using `google-genai`.
   - Implement `app/services/openai_client.py` using `openai.AsyncOpenAI`.
   - Implement `app/services/claude_client.py` using `anthropic.AsyncAnthropic`.
   - Each client will implement the uniform signature:
     ```python
     async def call_<provider>(prompt: str, system_prompt: str, timeout: int) -> dict:
         ...
     ```
   - Each client returns a normalized dictionary matching the engine's internal `ProviderResult` shape:
     ```python
     {
         "status": "success" | "error",
         "model": "<model_name>",
         "duration_seconds": <float>,
         "tokens": {"input": <int>, "output": <int>},
         "response_text": <str | None>,
         "error_message": <str | None>  # only present on error
     }
     ```
3. **Phase 0 Scope Constraint:**
   - No code is added to `app/services/` in this branch. Service implementation begins in Phase 3.
