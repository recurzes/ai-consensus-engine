import json
import unittest
from pydantic import ValidationError

from app.schemas import Context, LOBEnum, RoleEnum


class TestContextModel(unittest.TestCase):
    """Unit tests for Context schema and associated enums."""

    def test_valid_context_instantiation_with_enums(self):
        """Verify Context instantiates properly when using enum members."""
        ctx = Context(
            role=RoleEnum.claims_adjuster,
            line_of_business=LOBEnum.homeowners,
            state="MT",
        )
        self.assertEqual(ctx.role, RoleEnum.claims_adjuster)
        self.assertEqual(ctx.line_of_business, LOBEnum.homeowners)
        self.assertEqual(ctx.state, "MT")

    def test_valid_context_instantiation_with_strings(self):
        """Verify Context instantiates properly when passing raw string values."""
        ctx = Context(
            role="claims_adjuster",  # type: ignore[arg-type]
            line_of_business="homeowners",  # type: ignore[arg-type]
            state="MT",
        )
        self.assertEqual(ctx.role, RoleEnum.claims_adjuster)
        self.assertEqual(ctx.line_of_business, LOBEnum.homeowners)
        self.assertEqual(ctx.state, "MT")

    def test_default_state_value(self):
        """Verify default state is 'MT' when omitted."""
        ctx = Context(
            role=RoleEnum.underwriter,
            line_of_business=LOBEnum.personal_auto,
        )
        self.assertEqual(ctx.state, "MT")

    def test_custom_state_value(self):
        """Verify custom state can be specified."""
        ctx = Context(
            role=RoleEnum.underwriter,
            line_of_business=LOBEnum.commercial_pnc,
            state="CA",
        )
        self.assertEqual(ctx.state, "CA")

    def test_all_supported_roles(self):
        """Verify all 5 supported roles can be used."""
        expected_roles = [
            "layman_linguist",
            "underwriter",
            "claims_adjuster",
            "client_communications",
            "aca_expert",
        ]
        self.assertEqual([r.value for r in RoleEnum], expected_roles)
        for role in expected_roles:
            ctx = Context(role=role, line_of_business="personal_auto")  # type: ignore[arg-type]
            self.assertEqual(ctx.role, role)

    def test_all_supported_lines_of_business(self):
        """Verify all 6 supported lines of business can be used."""
        expected_lobs = [
            "personal_auto",
            "homeowners",
            "umbrella",
            "commercial_auto",
            "commercial_pnc",
            "aca_health",
        ]
        self.assertEqual([lob.value for lob in LOBEnum], expected_lobs)
        for lob in expected_lobs:
            ctx = Context(role="underwriter", line_of_business=lob)  # type: ignore[arg-type]
            self.assertEqual(ctx.line_of_business, lob)

    def test_invalid_role_raises_validation_error(self):
        """Verify ValidationError is raised when an unsupported role is provided."""
        with self.assertRaises(ValidationError) as cm:
            Context(
                role="invalid_role",  # type: ignore[arg-type]
                line_of_business="homeowners",  # type: ignore[arg-type]
            )
        self.assertIn("role", str(cm.exception))

    def test_invalid_line_of_business_raises_validation_error(self):
        """Verify ValidationError is raised when an unsupported line of business is provided."""
        with self.assertRaises(ValidationError) as cm:
            Context(
                role="underwriter",  # type: ignore[arg-type]
                line_of_business="invalid_lob",  # type: ignore[arg-type]
            )
        self.assertIn("line_of_business", str(cm.exception))

    def test_missing_required_fields(self):
        """Verify ValidationError is raised if role or line_of_business is missing."""
        with self.assertRaises(ValidationError):
            Context(line_of_business=LOBEnum.homeowners)  # type: ignore[call-arg]

        with self.assertRaises(ValidationError):
            Context(role=RoleEnum.underwriter)  # type: ignore[call-arg]

    def test_string_enum_subclass_and_json_serialization(self):
        """Verify enums are str subclasses and serialize cleanly without custom encoders."""
        self.assertTrue(issubclass(RoleEnum, str))
        self.assertTrue(issubclass(LOBEnum, str))
        self.assertIsInstance(RoleEnum.layman_linguist, str)
        self.assertIsInstance(LOBEnum.homeowners, str)

        ctx = Context(
            role=RoleEnum.layman_linguist,
            line_of_business=LOBEnum.aca_health,
            state="TX",
        )

        dumped_dict = ctx.model_dump()
        self.assertEqual(
            dumped_dict,
            {
                "role": "layman_linguist",
                "line_of_business": "aca_health",
                "state": "TX",
            },
        )

        json_str = ctx.model_dump_json()
        parsed = json.loads(json_str)
        self.assertEqual(
            parsed,
            {
                "role": "layman_linguist",
                "line_of_business": "aca_health",
                "state": "TX",
            },
        )


if __name__ == "__main__":
    unittest.main()
