from typing import Any

from app.schemas.models import LOBEnum, RoleEnum

ROLE_PROMPTS: dict[str, str] = {
    "layman_linguist": (
        "You are an expert insurance communicator skilled at translating complex policy language, "
        "coverage terms, endorsements, and statutory definitions into plain, empathetic, everyday "
        "English for consumers."
    ),
    "underwriter": (
        "You are a senior insurance underwriter evaluating risk acceptability, hazard exposures, "
        "limits, deductibles, and guidelines according to insurance company standards and statutes in {state}."
    ),
    "claims_adjuster": (
        "You are a Property & Casualty claims adjuster analyzing coverage triggers, exclusions, "
        "reservation of rights considerations, proof of loss, and claim workflows in {state}."
    ),
    "client_communications": (
        "You are an Agency Client Relations Specialist drafting clear, highly professional, polite, "
        "and legally sound email and letter correspondence to policyholders."
    ),
    "aca_expert": (
        "You are a health insurance compliance specialist with deep expertise in ACA regulations, "
        "federal poverty levels (FPL), subsidies, open/special enrollment triggers, and marketplace "
        "guidelines in {state}."
    ),
}


class _SafeFormatDict(dict[str, Any]):
    """Dictionary that preserves placeholder syntax for any missing format keys."""

    def __missing__(self, key: str) -> str:
        return f"{{{key}}}"


def get_role_prompt(
    role: str | RoleEnum,
    state: str = "MT",
    line_of_business: str | LOBEnum | None = None,
) -> str:
    """Format and return the system prompt for a specified insurance role.

    Args:
        role: The role persona key (as str or RoleEnum).
        state: US state jurisdiction for state-specific templates (defaults to 'MT').
        line_of_business: Optional line of business (as str or LOBEnum).

    Returns:
        The formatted role system prompt string.

    Raises:
        ValueError: If the role is invalid or unsupported.
    """
    role_key = role.value if hasattr(role, "value") else str(role)

    if role_key not in ROLE_PROMPTS:
        valid_roles = ", ".join(repr(r) for r in ROLE_PROMPTS.keys())
        raise ValueError(
            f"Invalid role '{role}'. Supported roles are: {valid_roles}."
        )

    template = ROLE_PROMPTS[role_key]

    resolved_state = state if state is not None else "MT"
    resolved_lob = (
        line_of_business.value
        if hasattr(line_of_business, "value")
        else (line_of_business or "")
    )

    format_vars = _SafeFormatDict(
        state=resolved_state,
        line_of_business=resolved_lob,
    )

    return template.format_map(format_vars)


__all__ = [
    "ROLE_PROMPTS",
    "get_role_prompt",
]
