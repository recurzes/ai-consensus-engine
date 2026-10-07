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
5. **Support for Target Models:** First-class support for `gemini-2.5-flash`, `gpt-4o-mini`, `claude-haiku-4-5`, `gemini-2.5-pro`, and `gpt-4o`.

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
  - **Model Identifier Mapping Overhead:** Requires verifying that newer model identifiers (e.g. `gemini-2.5-flash`, `claude-haiku-4-5`) and their pricing/usage objects are fully supported and mapped identically in litellm.
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

---

## ADR 002: Line of Business (LOB) Context Handling in System Prompts

- **Date:** 2026-09-24
- **Status:** Accepted
- **Phase:** 2 — Prompt Engine (`feat/prompts-lob-and-state-handling`)
- **Deciders:** Engineering Team

---

### Context & Problem Statement

Spec section 4.1 outlines five core insurance personas (`layman_linguist`, `underwriter`, `claims_adjuster`, `client_communications`, `aca_expert`) and six lines of business (`personal_auto`, `homeowners`, `umbrella`, `commercial_auto`, `commercial_pnc`, `aca_health`).

While the persona templates in spec section 4.1 explicitly define `{state}` placeholders for jurisdiction-dependent roles, they do not include a `{line_of_business}` token in their verbatim strings. Meanwhile, the Arbiter system prompt (Phase 2, spec section 4.4) explicitly receives `{line_of_business}` in its header directive.

We must resolve how the active line of business participates in worker system prompts (the "LOB ambiguity") and how jurisdiction defaults are guarded defensively.

---

### Options Evaluated

#### Option A: `LOB_CONTEXT` Dictionary with Context Appending / Interpolation (Chosen)
Maintain a `LOB_CONTEXT: dict[str, str]` dictionary in `app/core/prompts.py` mapping each supported LOB to a descriptive phrase (e.g. `"homeowners": "homeowners insurance"`).
- If a role prompt template contains a `{line_of_business}` placeholder, interpolate the LOB descriptive phrase.
- If the template does not contain `{line_of_business}` (verbatim spec templates) and an LOB is provided, append `\n\nLine of business: <lob_phrase>.` to provide worker models with domain boundary constraints.
- If `line_of_business` is omitted or `None`, no LOB clause is appended.

- **Pros:**
  - Workers receive explicit domain grounding, preventing an underwriter or claims adjuster from assuming commercial terms when evaluating a homeowners query.
  - Directly satisfies checklist item 2 of Phase 10 (`31-chore-final-acceptance-review.md`): *"Prompt dictionary (`app/core/prompts.py`): All 5 roles, all 6 LOBs, state handling present"*.
  - Safe fallback when LOB is not supplied.

- **Cons:**
  - Appends a single sentence to worker system prompts when an LOB is present.

#### Option B: Data-Only LOB for Arbiter Only
Treat LOB as purely metadata passed to the Arbiter, leaving worker system prompts entirely unaware of the line of business unless mentioned in the user query.

- **Pros:**
  - Keeps worker role prompts strictly identical to the verbatim strings in spec section 4.1.

- **Cons:**
  - Worker LLMs risk interpreting domain-specific queries under the wrong LOB if the user prompt lacks explicit framing.
  - Does not satisfy the presence of all 6 LOBs in `app/core/prompts.py` required by the final acceptance checklist.

---

### Decision

**Adopt Option A.**

In addition:
1. **Defensive State Handling:** `build_system_prompt()` enforces a fallback to `"MT"` (Montana) if `state` is `None`, empty, or whitespace-only. This serves as a secondary line of defense behind Pydantic schema validation.
2. **Function Signature:** Standardize `build_system_prompt(role, line_of_business=None, state="MT") -> str` as a pure, deterministic function. Maintain `get_role_prompt(role, state="MT", line_of_business=None)` for backwards compatibility.

---

### Consequences

- `app/core/prompts.py` exposes `ROLE_PROMPTS`, `LOB_CONTEXT`, `build_system_prompt`, and `get_role_prompt`.
- Calling `build_system_prompt("claims_adjuster", "homeowners", "MT")` yields a non-empty system prompt with `"MT"` and `"homeowners insurance"`.
- Calling `build_system_prompt("claims_adjuster", "homeowners", None)` falls back to `"MT"` cleanly.

---

## ADR 003: Arbiter Prompt Template and Synthesis System Prompt Design

- **Date:** 2026-09-24
- **Status:** Accepted
- **Phase:** 2 — Prompt Engine (`feat/prompts-arbiter-template`)
- **Deciders:** Engineering Team

---

### Context & Problem Statement

The synthesis Arbiter model (executed by Gemini 2.5 Pro or GPT-4o in Phase 6) serves a distinct operational purpose from worker models:
1. Worker models answer an end-user query under an individual insurance persona.
2. The Arbiter synthesizes answers from multiple independent worker models into a unified consensus, detecting hallucinations, reconciling conflicts, and matching the requested persona.

The Arbiter system directive requires injecting `{role}`, `{line_of_business}`, and `{state}` into a fixed 4-step reconciliation prompt (spec section 4.4). We need a clear structural abstraction for this prompt.

---

### Decision

1. **Module Constant & Builder Separation:** Define `ARBITER_SYSTEM_PROMPT` as a raw template string constant in `app/core/prompts.py` and provide `build_arbiter_prompt(role, line_of_business, state="MT") -> str` as the formatting entrypoint.
2. **Distinct from Worker Prompt Engine:** Maintain `build_arbiter_prompt()` strictly separate from `build_system_prompt()`. Worker prompts append domain boundary clauses; the Arbiter prompt interpolates persona variables directly into its header directive.
3. **Type Flexibility & Validation:** `build_arbiter_prompt()` supports both string literals and `RoleEnum` / `LOBEnum` members, validating against `ROLE_PROMPTS` and `LOB_CONTEXT` keys with clear `ValueError` feedback.
4. **Defensive Jurisdiction Fallback:** If `state` is `None`, empty, or whitespace, default to `"MT"`, consistent with ADR 002.

---

### Consequences

- `app/core/prompts.py` and `app/core` export `ARBITER_SYSTEM_PROMPT` and `build_arbiter_prompt`.
- Downstream synthesis service (`app/services/arbiter.py` in Phase 6) can directly call `build_arbiter_prompt(context.role, context.line_of_business, context.state)`.
- Pure, deterministic, side-effect-free implementation that is easily testable and inspectable.


