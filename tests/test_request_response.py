import json
import unittest
from pydantic import ValidationError

from app.schemas import (
    ConsensusRequest,
    ConsensusResponse,
    Context,
    LOBEnum,
    ProviderResult,
    RoleEnum,
    Telemetry,
)
from app.schemas.models import (
    ConsensusRequest as DirectConsensusRequest,
    ConsensusResponse as DirectConsensusResponse,
)


class TestRequestResponseModels(unittest.TestCase):
    """Unit tests for ConsensusRequest, ProviderResult, Telemetry, and ConsensusResponse."""

    def test_direct_imports_from_models_module(self):
        """Verify imports directly from app.schemas.models function without error."""
        self.assertIs(ConsensusRequest, DirectConsensusRequest)
        self.assertIs(ConsensusResponse, DirectConsensusResponse)

    def test_consensus_request_valid_with_context_instance(self):
        """Verify ConsensusRequest instantiates when passing a Context model instance."""
        context = Context(
            role=RoleEnum.claims_adjuster,
            line_of_business=LOBEnum.homeowners,
            state="MT",
        )
        req = ConsensusRequest(
            prompt="An insured backed a trailer into their garage door...",
            context=context,
        )
        self.assertEqual(
            req.prompt, "An insured backed a trailer into their garage door..."
        )
        self.assertEqual(req.context.role, RoleEnum.claims_adjuster)
        self.assertEqual(req.context.line_of_business, LOBEnum.homeowners)
        self.assertEqual(req.context.state, "MT")

    def test_consensus_request_valid_with_context_dict(self):
        """Verify ConsensusRequest instantiates when passing a dict for context."""
        req_data = {
            "prompt": "What is the deductible for wind damage?",
            "context": {
                "role": "underwriter",
                "line_of_business": "personal_auto",
                "state": "CA",
            },
        }
        req = ConsensusRequest(**req_data)
        self.assertEqual(req.prompt, "What is the deductible for wind damage?")
        self.assertEqual(req.context.role, "underwriter")
        self.assertEqual(req.context.line_of_business, "personal_auto")
        self.assertEqual(req.context.state, "CA")

    def test_consensus_request_missing_prompt_raises(self):
        """Verify ValidationError is raised when prompt is omitted."""
        context = Context(
            role=RoleEnum.underwriter,
            line_of_business=LOBEnum.commercial_pnc,
        )
        with self.assertRaises(ValidationError) as cm:
            ConsensusRequest(context=context)  # type: ignore[call-arg]
        self.assertIn("prompt", str(cm.exception))

    def test_consensus_request_missing_context_raises(self):
        """Verify ValidationError is raised when context is omitted."""
        with self.assertRaises(ValidationError) as cm:
            ConsensusRequest(prompt="Does this policy cover hail?")  # type: ignore[call-arg]
        self.assertIn("context", str(cm.exception))

    def test_provider_result_success_shape(self):
        """Verify ProviderResult handles successful execution shape."""
        provider_dict = {
            "status": "success",
            "model": "gemini-2.5-flash",
            "duration_seconds": 1.15,
            "tokens": {"input": 85, "output": 210},
            "response_text": "The garage door is covered under Section I...",
        }
        result = ProviderResult(**provider_dict)
        self.assertEqual(result.status, "success")
        self.assertEqual(result.model, "gemini-2.5-flash")
        self.assertEqual(result.duration_seconds, 1.15)
        self.assertEqual(result.tokens, {"input": 85, "output": 210})
        self.assertEqual(
            result.response_text, "The garage door is covered under Section I..."
        )
        self.assertIsNone(result.error_message)

        # Serialized form should omit error_message when None
        dumped = result.model_dump()
        self.assertEqual(dumped, provider_dict)

    def test_provider_result_failure_shape(self):
        """Verify ProviderResult handles failure shape with null response_text and error_message."""
        error_dict = {
            "status": "error",
            "model": "claude-3-5-haiku",
            "duration_seconds": 1.82,
            "tokens": {"input": 0, "output": 0},
            "response_text": None,
            "error_message": "Rate limit exceeded (HTTP 429)",
        }
        result = ProviderResult(**error_dict)
        self.assertEqual(result.status, "error")
        self.assertEqual(result.model, "claude-3-5-haiku")
        self.assertEqual(result.duration_seconds, 1.82)
        self.assertEqual(result.tokens, {"input": 0, "output": 0})
        self.assertIsNone(result.response_text)
        self.assertEqual(result.error_message, "Rate limit exceeded (HTTP 429)")

        # Serialized form should retain error_message and response_text: None
        dumped = result.model_dump()
        self.assertEqual(dumped, error_dict)

    def test_telemetry_instantiation_and_defaults(self):
        """Verify Telemetry instantiates correctly and supports default list factories."""
        # Full instantiation
        t = Telemetry(
            total_duration_seconds=3.12,
            total_estimated_cost_usd=0.00142,
            successful_providers=["gemini", "openai"],
            failed_providers=["claude"],
        )
        self.assertEqual(t.total_duration_seconds, 3.12)
        self.assertEqual(t.total_estimated_cost_usd, 0.00142)
        self.assertEqual(t.successful_providers, ["gemini", "openai"])
        self.assertEqual(t.failed_providers, ["claude"])

        # Default provider lists
        t_default = Telemetry(
            total_duration_seconds=1.5,
            total_estimated_cost_usd=0.0,
        )
        self.assertEqual(t_default.successful_providers, [])
        self.assertEqual(t_default.failed_providers, [])

    def test_consensus_response_matches_spec_sample_json_field_for_field(self):
        """Verify ConsensusResponse matches sample JSON from spec section 5 field-for-field."""
        sample_json_dict = {
            "status": "success",
            "prompt": "An insured backed a trailer into their garage door...",
            "applied_context": {
                "role": "claims_adjuster",
                "line_of_business": "homeowners",
                "state": "MT",
            },
            "consensus_answer": (
                "Damage to the garage door is typically covered under Coverage A..."
            ),
            "telemetry": {
                "total_duration_seconds": 3.12,
                "total_estimated_cost_usd": 0.00142,
                "successful_providers": ["gemini", "openai", "claude"],
                "failed_providers": [],
            },
            "provider_breakdown": {
                "gemini": {
                    "status": "success",
                    "model": "gemini-2.5-flash",
                    "duration_seconds": 1.15,
                    "tokens": {"input": 85, "output": 210},
                    "response_text": "The garage door is covered under Section I...",
                }
            },
        }

        response = ConsensusResponse(**sample_json_dict)

        # Field-for-field dictionary equality
        self.assertEqual(response.model_dump(), sample_json_dict)

        # JSON round-trip equality
        serialized_json = response.model_dump_json()
        parsed_json = json.loads(serialized_json)
        self.assertEqual(parsed_json, sample_json_dict)

    def test_consensus_response_multi_provider_sample(self):
        """Verify ConsensusResponse with multiple providers from the full spec reference."""
        full_sample_dict = {
            "status": "success",
            "prompt": "An insured backed a trailer into their garage door...",
            "applied_context": {
                "role": "claims_adjuster",
                "line_of_business": "homeowners",
                "state": "MT",
            },
            "consensus_answer": "Coverage applies under standard Dwelling terms.",
            "telemetry": {
                "total_duration_seconds": 3.12,
                "total_estimated_cost_usd": 0.00142,
                "successful_providers": ["gemini", "openai", "claude"],
                "failed_providers": [],
            },
            "provider_breakdown": {
                "gemini": {
                    "status": "success",
                    "model": "gemini-2.5-flash",
                    "duration_seconds": 1.15,
                    "tokens": {"input": 85, "output": 210},
                    "response_text": "The garage door is covered under Section I...",
                },
                "openai": {
                    "status": "success",
                    "model": "gpt-4o-mini",
                    "duration_seconds": 1.48,
                    "tokens": {"input": 85, "output": 230},
                    "response_text": "Under standard ISO forms, the door falls under Dwelling...",
                },
                "claude": {
                    "status": "success",
                    "model": "claude-3-5-haiku",
                    "duration_seconds": 1.82,
                    "tokens": {"input": 85, "output": 245},
                    "response_text": "First-party property coverage applies...",
                },
            },
        }

        response = ConsensusResponse(**full_sample_dict)
        self.assertEqual(response.model_dump(), full_sample_dict)

    def test_consensus_response_partial_failure_scenario(self):
        """Verify ConsensusResponse when one provider failed as in degradation spec."""
        degraded_dict = {
            "status": "success",
            "prompt": "Coverage inquiry",
            "applied_context": {
                "role": "claims_adjuster",
                "line_of_business": "homeowners",
                "state": "MT",
            },
            "consensus_answer": "Synthesized from 2 providers...",
            "telemetry": {
                "total_duration_seconds": 2.87,
                "total_estimated_cost_usd": 0.00098,
                "successful_providers": ["gemini", "openai"],
                "failed_providers": ["claude"],
            },
            "provider_breakdown": {
                "gemini": {
                    "status": "success",
                    "model": "gemini-2.5-flash",
                    "duration_seconds": 1.15,
                    "tokens": {"input": 85, "output": 210},
                    "response_text": "Gemini response text",
                },
                "openai": {
                    "status": "success",
                    "model": "gpt-4o-mini",
                    "duration_seconds": 1.48,
                    "tokens": {"input": 85, "output": 230},
                    "response_text": "OpenAI response text",
                },
                "claude": {
                    "status": "error",
                    "model": "claude-3-5-haiku",
                    "duration_seconds": 0.05,
                    "tokens": {"input": 0, "output": 0},
                    "response_text": None,
                    "error_message": "Invalid API key",
                },
            },
        }

        response = ConsensusResponse(**degraded_dict)
        self.assertEqual(response.model_dump(), degraded_dict)
        parsed = json.loads(response.model_dump_json())
        self.assertEqual(parsed, degraded_dict)

    def test_consensus_response_total_failure_scenario(self):
        """Verify ConsensusResponse when all providers failed and consensus_answer is None."""
        failed_dict = {
            "status": "error",
            "prompt": "Coverage inquiry",
            "applied_context": {
                "role": "claims_adjuster",
                "line_of_business": "homeowners",
                "state": "MT",
            },
            "consensus_answer": None,
            "telemetry": {
                "total_duration_seconds": 1.0,
                "total_estimated_cost_usd": 0.0,
                "successful_providers": [],
                "failed_providers": ["gemini", "openai", "claude"],
            },
            "provider_breakdown": {},
        }
        response = ConsensusResponse(**failed_dict)
        self.assertEqual(response.status, "error")
        self.assertIsNone(response.consensus_answer)
        self.assertEqual(response.model_dump(), failed_dict)


if __name__ == "__main__":
    unittest.main()
