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

LOB_CONTEXT: dict[str, str] = {
    "personal_auto": "personal auto insurance",
    "homeowners": "homeowners insurance",
    "umbrella": "umbrella insurance",
    "commercial_auto": "commercial auto insurance",
    "commercial_pnc": "commercial property and general liability insurance",
    "aca_health": "ACA and health insurance",
}


class _SafeFormatDict(dict[str, Any]):
    """Dictionary that preserves placeholder syntax for any missing format keys."""

    def __missing__(self, key: str) -> str:
        return f"{{{key}}}"


def build_system_prompt(
    role: str | RoleEnum,
    line_of_business: str | LOBEnum | None = None,
    state: str | None = "MT",
) -> str:
    """Build and format the final system prompt for a worker AI model.

    This function represents the prompt-building layer for worker LLMs.
    It resolves:
      1. Role persona template selection and validation against ROLE_PROMPTS.
      2. Jurisdiction fallback: Defensively defaults to 'MT' (Montana) if state
         is omitted, None, or empty, ensuring jurisdiction-dependent prompts
         always render a valid state.
      3. Line of business (LOB) context handling: Resolves the LOB against
         LOB_CONTEXT. If the role template contains a '{line_of_business}' placeholder,
         it interpolates the LOB descriptive phrase. If not, and line_of_business
         is provided, it appends '\\n\\nLine of business: <lob_phrase>.' to provide
         explicit domain context to worker models (ADR 002 Option A). If line_of_business
         is None, no LOB clause is appended.

    Args:
        role: The role persona key (as str or RoleEnum).
        line_of_business: Optional line of business (as str, LOBEnum, or None).
        state: US state jurisdiction for state-specific templates (defaults to 'MT').

    Returns:
        The fully formatted and resolved system prompt string.

    Raises:
        ValueError: If role is invalid, or if an unsupported line_of_business is provided.
    """
    role_key = role.value if hasattr(role, "value") else str(role)
    if role_key not in ROLE_PROMPTS:
        valid_roles = ", ".join(repr(r) for r in ROLE_PROMPTS.keys())
        raise ValueError(
            f"Invalid role '{role}'. Supported roles are: {valid_roles}."
        )

    # State defensive fallback: None, empty string, or whitespace defaults to 'MT'
    if state is None or not str(state).strip():
        resolved_state = "MT"
    else:
        resolved_state = str(state).strip()

    template = ROLE_PROMPTS[role_key]

    # Resolve and validate line of business
    lob_phrase: str | None = None
    if line_of_business is not None:
        lob_key = (
            line_of_business.value
            if hasattr(line_of_business, "value")
            else str(line_of_business).strip()
        )
        if lob_key:
            if lob_key not in LOB_CONTEXT:
                valid_lobs = ", ".join(repr(k) for k in LOB_CONTEXT.keys())
                raise ValueError(
                    f"Invalid line of business '{line_of_business}'. Supported lines of business are: {valid_lobs}."
                )
            lob_phrase = LOB_CONTEXT[lob_key]

    format_vars = _SafeFormatDict(
        state=resolved_state,
        line_of_business=lob_phrase or "",
    )

    rendered = template.format_map(format_vars)

    # If the template did not have a {line_of_business} placeholder and an LOB was provided,
    # append the LOB context as explicit domain context for the worker model.
    if lob_phrase and "{line_of_business}" not in template:
        rendered = f"{rendered}\n\nLine of business: {lob_phrase}."

    return rendered


def get_role_prompt(
    role: str | RoleEnum,
    state: str | None = "MT",
    line_of_business: str | LOBEnum | None = None,
) -> str:
    """Format and return the system prompt for a specified insurance role.

    This function is maintained for backwards compatibility with the
    (role, state, line_of_business) signature and delegates to build_system_prompt.

    Args:
        role: The role persona key (as str or RoleEnum).
        state: US state jurisdiction for state-specific templates (defaults to 'MT').
        line_of_business: Optional line of business (as str or LOBEnum).

    Returns:
        The formatted role system prompt string.
    """
    return build_system_prompt(
        role=role,
        line_of_business=line_of_business,
        state=state,
    )


__all__ = [
    "LOB_CONTEXT",
    "ROLE_PROMPTS",
    "build_system_prompt",
    "get_role_prompt",
]
