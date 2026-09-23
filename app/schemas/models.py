from enum import Enum
from pydantic import BaseModel, ConfigDict, Field, model_serializer


class RoleEnum(str, Enum):
    """Supported insurance professional personas."""

    layman_linguist = "layman_linguist"
    underwriter = "underwriter"
    claims_adjuster = "claims_adjuster"
    client_communications = "client_communications"
    aca_expert = "aca_expert"


class LOBEnum(str, Enum):
    """Supported lines of business."""

    personal_auto = "personal_auto"
    homeowners = "homeowners"
    umbrella = "umbrella"
    commercial_auto = "commercial_auto"
    commercial_pnc = "commercial_pnc"
    aca_health = "aca_health"


class Context(BaseModel):
    """Context object carrying persona, line of business, and jurisdiction."""

    model_config = ConfigDict(use_enum_values=True)

    role: RoleEnum
    line_of_business: LOBEnum
    state: str = "MT"


class ConsensusRequest(BaseModel):
    """Request payload containing the user prompt and domain context."""

    model_config = ConfigDict(use_enum_values=True)

    prompt: str
    context: Context


class ProviderResult(BaseModel):
    """Standardized response from an individual AI model provider."""

    model_config = ConfigDict(use_enum_values=True)

    status: str
    model: str
    duration_seconds: float
    tokens: dict[str, int]
    response_text: str | None = None
    error_message: str | None = None

    @model_serializer(mode="wrap")
    def _serialize(self, handler, info):
        data = handler(self)
        if isinstance(data, dict) and data.get("error_message") is None:
            data.pop("error_message", None)
        return data


class Telemetry(BaseModel):
    """End-to-end performance and cost telemetry for the consensus pipeline."""

    model_config = ConfigDict(use_enum_values=True)

    total_duration_seconds: float
    total_estimated_cost_usd: float
    successful_providers: list[str] = Field(default_factory=list)
    failed_providers: list[str] = Field(default_factory=list)


class ConsensusResponse(BaseModel):
    """Unified consensus API response matching spec section 5."""

    model_config = ConfigDict(use_enum_values=True)

    status: str
    prompt: str
    applied_context: Context
    consensus_answer: str | None = None
    telemetry: Telemetry
    provider_breakdown: dict[str, ProviderResult] = Field(default_factory=dict)


__all__ = [
    "ConsensusRequest",
    "ConsensusResponse",
    "Context",
    "LOBEnum",
    "ProviderResult",
    "RoleEnum",
    "Telemetry",
]
