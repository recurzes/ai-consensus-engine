# AI Consensus Engine

An asynchronous, insurance-tailored backend engine that aggregates consensus across leading commercial Large Language Models (LLMs)—**Google Gemini**, **OpenAI ChatGPT**, and **Anthropic Claude**. When queried under a specific insurance persona (such as an underwriter or claims adjuster) and regulatory jurisdiction, the engine executes parallel worker queries, synthesizes the results through an intelligent Arbiter model, filters hallucinations and conflicting terms, and returns an authoritative consensus answer along with complete latency, token, and cost telemetry.

For in-depth architectural and technical background, see the [Architecture Decision Records (docs/decisions.md)](docs/decisions.md) and the [Technical Specification (notes/Consensus Engine Plan.md)](notes/Consensus%20Engine%20Plan.md).

---

## Architecture

The system implements a **Worker Fan-Out $\to$ Arbiter Synthesis** pattern:

1. **Client Request**: A client submits a prompt accompanied by insurance context (`role`, `line_of_business`, `state`).
2. **Worker Fan-Out**: FastAPI builds role-tailored system prompts and dispatches requests concurrently across three worker models using `asyncio.gather(..., return_exceptions=True)`:
   - **Google Gemini Worker**: `gemini-2.5-flash` via `google-genai`
   - **OpenAI Worker**: `gpt-4o-mini` via `openai`
   - **Anthropic Claude Worker**: `claude-3-5-haiku` via `anthropic`
3. **Timeout & Failure Isolation**: Each worker call is guarded by an individual timeout (`REQUEST_TIMEOUT_SECONDS`). If any provider times out, encounters a rate limit (HTTP 429), or fails authentication, the failure is captured in telemetry without failing the overall request.
4. **Arbiter Synthesis**: The surviving provider answers are fed to the Arbiter model (configurable: `gemini-2.5-pro` or `gpt-4o`). The Arbiter validates coverage triggers, reconciles disagreements, eliminates hallucinations, and generates a unified consensus answer.
5. **Telemetry & Cost Tracking**: Accurate duration, token counts (input/output), and USD costs are calculated per provider and returned in the response payload.

```text
User Request
    │
    ▼
┌─────────────────────────────────┐
│      FastAPI /api/v1/consensus  │
└────────────┬────────────────────┘
             │ asyncio.gather (concurrent)
     ┌───────┼───────┐
     ▼       ▼       ▼
 Gemini   OpenAI   Claude
 Worker   Worker   Worker
     └───────┼───────┘
             │ surviving responses
             ▼
        ┌─────────┐
        │ Arbiter │ (Gemini 2.5 Pro or GPT-4o)
        └────┬────┘
             ▼
      ConsensusResponse
```

---

## Setup

### Prerequisites
- **Python 3.11+**
- Active API keys for:
  - **Google Gemini** (for worker and default arbiter)
  - **OpenAI** (for worker and alternative arbiter)
  - **Anthropic** (for worker)

### Installation

Clone the repository and install dependencies in a virtual environment:

```bash
git clone <repo-url>
cd ai-consensus-engine
python -m venv venv

# Activate virtual environment:
# On macOS / Linux:
source venv/bin/activate
# On Windows (PowerShell):
venv\Scripts\Activate.ps1
# On Windows (cmd):
venv\Scripts\activate.bat

pip install -r requirements.txt
```

### Environment Configuration

Create a `.env` file from the provided template:

```bash
cp .env.example .env
```

Open `.env` and configure your credentials:

```dotenv
# Provider API Keys
GEMINI_API_KEY=your_gemini_api_key_here
OPENAI_API_KEY=your_openai_api_key_here
ANTHROPIC_API_KEY=your_anthropic_api_key_here

# Optional: Set only if your Anthropic key requires an organization/workspace header
# ANTHROPIC_WORKSPACE_ID=wrkspc_your_workspace_id_here

# Arbiter Model Provider ('gemini' for gemini-2.5-pro or 'openai' for gpt-4o)
ARBITER_MODEL_PROVIDER=gemini

# Per-worker request timeout budget in seconds (default: 12)
REQUEST_TIMEOUT_SECONDS=12
```

---

## Running the Server

Start the local development server with hot reloading enabled via Uvicorn:

```bash
uvicorn app.main:app --reload
```

Once running:
- **Consensus Endpoint**: `POST http://localhost:8000/api/v1/consensus`
- **Interactive Swagger UI**: `http://localhost:8000/docs`
- **ReDoc Documentation**: `http://localhost:8000/redoc`

### Example Request

```bash
curl -X POST http://localhost:8000/api/v1/consensus \
  -H "Content-Type: application/json" \
  -d '{
    "prompt": "My homeowners policy has an Ordinance or Law endorsement. What does that mean and when would it actually pay out?",
    "context": {
      "role": "layman_linguist",
      "line_of_business": "homeowners",
      "state": "MT"
    }
  }'
```

### Example Response Structure

```json
{
  "consensus_answer": "An Ordinance or Law endorsement helps cover the extra costs required to rebuild or repair your home to meet current building codes...",
  "provider_responses": [
    {
      "provider": "gemini",
      "model": "gemini-2.5-flash",
      "status": "success",
      "duration_seconds": 1.42,
      "tokens": { "input": 128, "output": 215 },
      "response_text": "..."
    },
    {
      "provider": "openai",
      "model": "gpt-4o-mini",
      "status": "success",
      "duration_seconds": 1.15,
      "tokens": { "input": 140, "output": 198 },
      "response_text": "..."
    },
    {
      "provider": "claude",
      "model": "claude-3-5-haiku",
      "status": "success",
      "duration_seconds": 1.68,
      "tokens": { "input": 135, "output": 204 },
      "response_text": "..."
    }
  ],
  "telemetry": {
    "total_duration_seconds": 3.85,
    "total_estimated_cost_usd": 0.00412,
    "successful_providers": ["gemini", "openai", "claude"],
    "failed_providers": []
  }
}
```

---

## Running the Demo Script

The repository includes an automated end-to-end demo script that exercises the running API server across three realistic scenarios:

```bash
python run_demo.py
```

### Scenarios Tested

1. **Scenario 1: Consumer Translation (`layman_linguist`)**
   - **Role**: `layman_linguist`, Line of Business: `homeowners`, Jurisdiction: `MT`.
   - **Prompt**: Explaining an "Ordinance or Law" endorsement.
   - **Demonstrates**: Translates technical policy clauses into plain, empathetic language suitable for policyholders.
2. **Scenario 2: Underwriting / Coverage Analysis (`claims_adjuster`)**
   - **Role**: `claims_adjuster`, Line of Business: `homeowners`, Jurisdiction: `MT`.
   - **Prompt**: Coverage analysis for backing an insured trailer into a garage door.
   - **Demonstrates**: Evaluates property damage vs. vehicle damage, applicable deductibles, and jurisdiction-specific claim workflow in Montana.
3. **Scenario 3: Failover Verification (`underwriter`, degraded path)**
   - **Role**: `underwriter`, Line of Business: `commercial_pnc`, Jurisdiction: `MT`.
   - **Prompt**: Coverage differences between a BOP (Business Owner's Policy) and standalone Commercial Property policy.
   - **Demonstrates**: The script intentionally corrupts the Anthropic API key (`sk-intentionally-invalid-for-failover-test`) to prove graceful degradation. The application **does not crash**; Gemini and OpenAI succeed, the Arbiter synthesizes consensus from surviving workers, and Claude is reported in `failed_providers`.

---

## Running Tests

The test suite covers provider client isolation, prompt templates, concurrent orchestration, timeout enforcement, graceful degradation, and API integration:

```bash
pytest tests/ -v
```

On Windows environments or when using a specific Python interpreter:

```bash
python -m pytest tests/ -v
# or
.venv\Scripts\pytest tests/ -v
```

---

## Key Design Decisions

The engine's architectural choices are formally documented in [docs/decisions.md](docs/decisions.md):

- **Native SDKs over `litellm` ([ADR 001](docs/decisions.md#adr-001-provider-integration-strategy-native-sdks-vs-litellm))**: The engine integrates directly with official provider libraries (`google-genai`, `openai`, `anthropic`). This ensures authoritative token metadata for pricing, fine-grained typed exception handling (e.g. distinguishing HTTP 429 rate limits from authentication errors), and eliminates third-party proxy abstraction drift.
- **Montana Default State & Context Appending ([ADR 002](docs/decisions.md#adr-002-line-of-business-lob-context-handling-in-system-prompts))**: If a client request omits the regulatory jurisdiction or provides empty whitespace, the engine defaults to `"MT"` (Montana) as a defensive boundary. Line of Business (LOB) context is injected into system prompts to ensure models evaluate terms within the appropriate insurance domain.
- **Configurable Arbiter Synthesis ([ADR 003](docs/decisions.md#adr-003-arbiter-prompt-template-and-synthesis-system-prompt-design))**: The synthesis Arbiter operates under a dedicated 4-step reconciliation prompt. The active model provider can be toggled via `ARBITER_MODEL_PROVIDER` between Google Gemini (`gemini-2.5-pro`) and OpenAI (`gpt-4o`).
- **Resilience & Total Failure Behavior**: If 1 or 2 workers fail, the engine degrades gracefully and synthesizes surviving results. In the event all three workers fail or time out, the server returns an `HTTP 502 Bad Gateway` with structured failure telemetry rather than an unhandled 500 error or crash.
