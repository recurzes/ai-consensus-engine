import logging
import time
from typing import Any

from fastapi import FastAPI, Request, status
from fastapi.responses import JSONResponse

from app.core.prompts import build_system_prompt
from app.schemas.models import (
    ConsensusRequest,
    ConsensusResponse,
    ProviderResult,
    Telemetry,
)
from app.services.arbiter import ArbiterError, synthesize, synthesize_single
from app.services.cost_tracker import calculate_all_costs
from app.services.orchestrator import (
    MODEL_TO_PROVIDER,
    ORDERED_PROVIDERS,
    partition_results,
    run_workers,
)

logger = logging.getLogger(__name__)

app = FastAPI(
    title="AI Consensus Engine",
    description="Multi-LLM consensus engine with professional insurance personas and arbiter synthesis.",
    version="1.0.0",
)


@app.exception_handler(ArbiterError)
async def arbiter_error_handler(request: Request, exc: ArbiterError) -> JSONResponse:
    """Handle arbiter synthesis errors and return a structured 502 error payload."""
    logger.error("Arbiter synthesis failed: %s", exc)
    return JSONResponse(
        status_code=status.HTTP_502_BAD_GATEWAY,
        content={
            "status": "error",
            "message": f"Arbiter synthesis failed: {exc}",
        },
    )


@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception) -> JSONResponse:
    """Catch-all safety net for unhandled exceptions returning clean JSON 500."""
    logger.exception("Unhandled exception during request processing: %s", exc)
    return JSONResponse(
        status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
        content={
            "status": "error",
            "message": "An unexpected error occurred.",
        },
    )


@app.get("/")
def index() -> dict[str, str]:
    return {"status": "ok"}


@app.post(
    "/api/v1/consensus",
    response_model=ConsensusResponse,
    status_code=status.HTTP_200_OK,
)
async def create_consensus(
    request: ConsensusRequest,
) -> ConsensusResponse | JSONResponse:
    """Execute end-to-end consensus pipeline across multiple LLM providers.

    Pipeline stages:
      1. Validate incoming request (handled automatically by Pydantic/FastAPI).
      2. Start pipeline latency timer and build role-specific system prompt.
      3. Dispatch concurrent worker queries to Gemini, OpenAI, and Claude.
      4. Partition results into successful and failed provider categories.
      5. Calculate and inject estimated token costs for all providers.
      6. Run Arbiter model to synthesize surviving outputs into consensus:
         - 3 survivors: reconcile 3 responses (normal happy path)
         - 2 survivors: reconcile 2 responses (graceful degraded path)
         - 1 survivor: validate/format single response (single-survivor degraded path)
      7. Stop timer and assemble fully populated ConsensusResponse.
    """
    start_time = time.perf_counter()

    # Step 2: Build domain system prompt
    system_prompt = build_system_prompt(
        role=request.context.role,
        line_of_business=request.context.line_of_business,
        state=request.context.state,
    )

    # Step 3: Dispatch workers concurrently
    raw_results = await run_workers(
        prompt=request.prompt,
        system_prompt=system_prompt,
    )

    # Step 4: Partition results into success and failure categories
    (
        successful_results,
        failed_results,
        successful_providers,
        failed_providers,
    ) = partition_results(raw_results)

    # Step 5: Normalize results, calculate costs, and assemble provider breakdown
    dict_results: list[dict[str, Any]] = []
    for idx, item in enumerate(raw_results):
        if isinstance(item, dict):
            dict_results.append(item)
        else:
            fallback_provider = (
                ORDERED_PROVIDERS[idx]
                if idx < len(ORDERED_PROVIDERS)
                else f"provider_{idx}"
            )
            dict_results.append(
                {
                    "status": "error",
                    "model": fallback_provider,
                    "duration_seconds": 0.0,
                    "tokens": {"input": 0, "output": 0},
                    "response_text": None,
                    "error_message": str(item),
                }
            )

    costed_results, total_estimated_cost = calculate_all_costs(dict_results)

    provider_breakdown: dict[str, ProviderResult] = {}
    for idx, res in enumerate(costed_results):
        model = res.get("model")
        provider_name = MODEL_TO_PROVIDER.get(
            model,
            ORDERED_PROVIDERS[idx]
            if idx < len(ORDERED_PROVIDERS)
            else f"provider_{idx}",
        )
        provider_breakdown[provider_name] = ProviderResult(**res)

    # Step 6: Branch on surviving providers and synthesize consensus
    num_successful = len(successful_results)
    if num_successful >= 2:
        logger.info(
            "Arbiter synthesizing from %d successful providers (%s)",
            num_successful,
            successful_providers,
        )
        consensus_answer = await synthesize(
            original_prompt=request.prompt,
            context=request.context,
            successful_results=successful_results,
        )
    elif num_successful == 1:
        logger.warning(
            "Degraded consensus: 1 surviving provider (%s). Validating through active persona.",
            successful_providers[0] if successful_providers else "unknown",
        )
        consensus_answer = await synthesize_single(
            original_prompt=request.prompt,
            context=request.context,
            single_result=successful_results[0],
        )
    else:
        logger.error(
            "All providers failed (%s). No consensus could be generated.",
            failed_providers,
        )
        failed_providers_detail = {
            provider_name: (res.error_message or "Unknown provider error")
            for provider_name, res in provider_breakdown.items()
        }
        return JSONResponse(
            status_code=status.HTTP_502_BAD_GATEWAY,
            content={
                "status": "error",
                "message": "All AI providers failed. No consensus could be generated.",
                "failed_providers": failed_providers_detail,
            },
        )

    # Step 7: Record elapsed time and assemble final response
    total_duration_seconds = round(time.perf_counter() - start_time, 4)

    telemetry = Telemetry(
        total_duration_seconds=total_duration_seconds,
        total_estimated_cost_usd=total_estimated_cost,
        successful_providers=successful_providers,
        failed_providers=failed_providers,
    )

    return ConsensusResponse(
        status="success",
        prompt=request.prompt,
        applied_context=request.context,
        consensus_answer=consensus_answer,
        telemetry=telemetry,
        provider_breakdown=provider_breakdown,
    )