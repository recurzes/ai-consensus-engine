import unittest

from app.core.prompts import ROLE_PROMPTS, get_role_prompt
from app.schemas.models import LOBEnum, RoleEnum


class TestRolePrompts(unittest.TestCase):
    """Unit tests for insurance persona role prompt templates and helper."""

    def test_role_prompts_contains_all_expected_roles(self):
        """Verify ROLE_PROMPTS defines all 5 roles specified in spec section 4.1."""
        expected_keys = {
            "layman_linguist",
            "underwriter",
            "claims_adjuster",
            "client_communications",
            "aca_expert",
        }
        self.assertEqual(set(ROLE_PROMPTS.keys()), expected_keys)

    def test_role_prompts_verbatim_content(self):
        """Verify templates match spec section 4.1 character-for-character."""
        self.assertEqual(
            ROLE_PROMPTS["layman_linguist"],
            (
                "You are an expert insurance communicator skilled at translating complex policy language, "
                "coverage terms, endorsements, and statutory definitions into plain, empathetic, everyday "
                "English for consumers."
            ),
        )
        self.assertEqual(
            ROLE_PROMPTS["underwriter"],
            (
                "You are a senior insurance underwriter evaluating risk acceptability, hazard exposures, "
                "limits, deductibles, and guidelines according to insurance company standards and statutes in {state}."
            ),
        )
        self.assertEqual(
            ROLE_PROMPTS["claims_adjuster"],
            (
                "You are a Property & Casualty claims adjuster analyzing coverage triggers, exclusions, "
                "reservation of rights considerations, proof of loss, and claim workflows in {state}."
            ),
        )
        self.assertEqual(
            ROLE_PROMPTS["client_communications"],
            (
                "You are an Agency Client Relations Specialist drafting clear, highly professional, polite, "
                "and legally sound email and letter correspondence to policyholders."
            ),
        )
        self.assertEqual(
            ROLE_PROMPTS["aca_expert"],
            (
                "You are a health insurance compliance specialist with deep expertise in ACA regulations, "
                "federal poverty levels (FPL), subsidies, open/special enrollment triggers, and marketplace "
                "guidelines in {state}."
            ),
        )

    def test_claims_adjuster_state_substitution(self):
        """Verify get_role_prompt substitutes {state} in claims_adjuster prompt."""
        result = get_role_prompt(
            "claims_adjuster", state="MT", line_of_business="homeowners"
        )
        self.assertIn("in MT.", result)
        self.assertNotIn("{state}", result)
        expected = (
            "You are a Property & Casualty claims adjuster analyzing coverage triggers, exclusions, "
            "reservation of rights considerations, proof of loss, and claim workflows in MT."
        )
        self.assertEqual(result, expected)

    def test_layman_linguist_unchanged(self):
        """Verify get_role_prompt returns layman_linguist prompt unchanged."""
        result = get_role_prompt(
            "layman_linguist", state="MT", line_of_business="homeowners"
        )
        self.assertEqual(result, ROLE_PROMPTS["layman_linguist"])
        self.assertNotIn("{state}", result)

    def test_client_communications_unchanged(self):
        """Verify get_role_prompt returns client_communications prompt unchanged."""
        result = get_role_prompt(
            "client_communications", state="FL", line_of_business="commercial_pnc"
        )
        self.assertEqual(result, ROLE_PROMPTS["client_communications"])
        self.assertNotIn("{state}", result)

    def test_underwriter_custom_state_substitution(self):
        """Verify get_role_prompt substitutes custom state in underwriter prompt."""
        result = get_role_prompt(
            "underwriter", state="CA", line_of_business="commercial_pnc"
        )
        self.assertIn("in CA.", result)
        self.assertNotIn("{state}", result)

    def test_aca_expert_custom_state_substitution(self):
        """Verify get_role_prompt substitutes custom state in aca_expert prompt."""
        result = get_role_prompt(
            "aca_expert", state="TX", line_of_business="aca_health"
        )
        self.assertIn("in TX.", result)
        self.assertNotIn("{state}", result)

    def test_default_state_is_mt(self):
        """Verify default state fallback is 'MT' when omitted or None."""
        result_default = get_role_prompt("underwriter")
        self.assertIn("in MT.", result_default)

        result_none = get_role_prompt("underwriter", state=None)
        self.assertIn("in MT.", result_none)

    def test_invoking_with_enum_members(self):
        """Verify RoleEnum and LOBEnum can be passed directly."""
        result = get_role_prompt(
            role=RoleEnum.claims_adjuster,
            state="MT",
            line_of_business=LOBEnum.homeowners,
        )
        self.assertIn("in MT.", result)
        self.assertNotIn("{state}", result)

    def test_all_supported_roles_render_cleanly(self):
        """Verify every supported role renders without leftover placeholders."""
        for role in RoleEnum:
            rendered = get_role_prompt(role=role, state="MT", line_of_business="homeowners")
            self.assertIsInstance(rendered, str)
            self.assertGreater(len(rendered), 0)
            self.assertNotIn("{state}", rendered)

    def test_invalid_role_raises_value_error(self):
        """Verify calling with an invalid role raises ValueError with descriptive message."""
        with self.assertRaises(ValueError) as cm:
            get_role_prompt("invalid_role", state="MT", line_of_business="homeowners")
        error_msg = str(cm.exception)
        self.assertIn("Invalid role 'invalid_role'", error_msg)
        self.assertIn("claims_adjuster", error_msg)

    def test_purity_and_idempotency(self):
        """Verify get_role_prompt has no side effects and produces identical results on repeated calls."""
        res1 = get_role_prompt("claims_adjuster", state="NY")
        res2 = get_role_prompt("claims_adjuster", state="NY")
        self.assertEqual(res1, res2)
        # Verify the underlying template was not permanently mutated
        self.assertIn("{state}", ROLE_PROMPTS["claims_adjuster"])


if __name__ == "__main__":
    unittest.main()
