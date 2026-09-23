from enum import Enum
from pydantic import BaseModel


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

    role: RoleEnum
    line_of_business: LOBEnum
    state: str = "MT"
