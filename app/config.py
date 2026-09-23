from typing import Any, Literal
from pydantic import Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Centralized application settings validated via Pydantic."""

    gemini_api_key: str = Field(..., min_length=1, description="Google Gemini API key")
    openai_api_key: str = Field(..., min_length=1, description="OpenAI API key")
    anthropic_api_key: str = Field(..., min_length=1, description="Anthropic API key")
    arbiter_model_provider: Literal["gemini", "openai"] = Field(
        ..., description="Arbiter model provider ('gemini' or 'openai')"
    )
    request_timeout_seconds: int = Field(
        default=12, gt=0, description="Per-worker request timeout in seconds"
    )
    default_state: str = Field(
        default="MT", min_length=1, description="Default state code for context"
    )

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )

    @field_validator("arbiter_model_provider", mode="before")
    @classmethod
    def normalize_arbiter_provider(cls, v: Any) -> Any:
        if isinstance(v, str):
            return v.strip().lower()
        return v


settings = Settings()
