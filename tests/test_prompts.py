import unittest

from app.core.prompts import (
    ARBITER_SYSTEM_PROMPT,
    LOB_CONTEXT,
    ROLE_PROMPTS,
    build_arbiter_prompt,
    build_system_prompt,
    get_role_prompt,
)
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

    def test_lob_context_contains_all_expected_lobs(self):
        """Verify LOB_CONTEXT defines all 6 lines of business specified in spec section 4.1."""
        expected_lobs = {
            "personal_auto": "personal auto insurance",
            "homeowners": "homeowners insurance",
            "umbrella": "umbrella insurance",
            "commercial_auto": "commercial auto insurance",
            "commercial_pnc": "commercial property and general liability insurance",
            "aca_health": "ACA and health insurance",
        }
        self.assertEqual(set(LOB_CONTEXT.keys()), set(expected_lobs.keys()))
        for lob, expected_phrase in expected_lobs.items():
            self.assertEqual(LOB_CONTEXT[lob], expected_phrase)

    def test_claims_adjuster_state_substitution(self):
        """Verify get_role_prompt substitutes {state} in claims_adjuster prompt."""
        result = get_role_prompt(
            "claims_adjuster", state="MT", line_of_business="homeowners"
        )
        self.assertIn("in MT.", result)
        self.assertNotIn("{state}", result)
        self.assertIn("Line of business: homeowners insurance.", result)

    def test_claims_adjuster_without_lob(self):
        """Verify get_role_prompt without LOB returns the verbatim template with state substituted."""
        result = get_role_prompt("claims_adjuster", state="MT")
        expected = (
            "You are a Property & Casualty claims adjuster analyzing coverage triggers, exclusions, "
            "reservation of rights considerations, proof of loss, and claim workflows in MT."
        )
        self.assertEqual(result, expected)

    def test_layman_linguist_unchanged_when_no_lob(self):
        """Verify get_role_prompt returns layman_linguist prompt unchanged when no LOB is supplied."""
        result = get_role_prompt("layman_linguist", state="MT")
        self.assertEqual(result, ROLE_PROMPTS["layman_linguist"])
        self.assertNotIn("{state}", result)

    def test_layman_linguist_with_lob(self):
        """Verify layman_linguist includes LOB context when supplied."""
        result = get_role_prompt(
            "layman_linguist", state="MT", line_of_business="homeowners"
        )
        self.assertIn(ROLE_PROMPTS["layman_linguist"], result)
        self.assertIn("Line of business: homeowners insurance.", result)
        self.assertNotIn("{state}", result)

    def test_client_communications_unchanged_when_no_lob(self):
        """Verify get_role_prompt returns client_communications prompt unchanged without LOB."""
        result = get_role_prompt("client_communications", state="FL")
        self.assertEqual(result, ROLE_PROMPTS["client_communications"])
        self.assertNotIn("{state}", result)

    def test_underwriter_custom_state_substitution(self):
        """Verify get_role_prompt substitutes custom state in underwriter prompt."""
        result = get_role_prompt(
            "underwriter", state="CA", line_of_business="commercial_pnc"
        )
        self.assertIn("in CA.", result)
        self.assertNotIn("{state}", result)
        self.assertIn("Line of business: commercial property and general liability insurance.", result)

    def test_aca_expert_custom_state_substitution(self):
        """Verify get_role_prompt substitutes custom state in aca_expert prompt."""
        result = get_role_prompt(
            "aca_expert", state="TX", line_of_business="aca_health"
        )
        self.assertIn("in TX.", result)
        self.assertNotIn("{state}", result)
        self.assertIn("Line of business: ACA and health insurance.", result)

    def test_default_state_is_mt(self):
        """Verify default state fallback is 'MT' when omitted or None."""
        result_default = get_role_prompt("underwriter")
        self.assertIn("in MT.", result_default)

        result_none = get_role_prompt("underwriter", state=None)
        self.assertIn("in MT.", result_none)

        result_whitespace = get_role_prompt("underwriter", state="   ")
        self.assertIn("in MT.", result_whitespace)

    def test_invoking_with_enum_members(self):
        """Verify RoleEnum and LOBEnum can be passed directly."""
        result = get_role_prompt(
            role=RoleEnum.claims_adjuster,
            state="MT",
            line_of_business=LOBEnum.homeowners,
        )
        self.assertIn("in MT.", result)
        self.assertNotIn("{state}", result)
        self.assertIn("Line of business: homeowners insurance.", result)

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

    def test_invalid_lob_raises_value_error(self):
        """Verify calling with an invalid line_of_business raises ValueError."""
        with self.assertRaises(ValueError) as cm:
            get_role_prompt("underwriter", state="MT", line_of_business="pet_insurance")
        error_msg = str(cm.exception)
        self.assertIn("Invalid line of business 'pet_insurance'", error_msg)
        self.assertIn("homeowners", error_msg)

    def test_purity_and_idempotency(self):
        """Verify get_role_prompt has no side effects and produces identical results on repeated calls."""
        res1 = get_role_prompt("claims_adjuster", state="NY")
        res2 = get_role_prompt("claims_adjuster", state="NY")
        self.assertEqual(res1, res2)
        # Verify the underlying template was not permanently mutated
        self.assertIn("{state}", ROLE_PROMPTS["claims_adjuster"])


class TestBuildSystemPrompt(unittest.TestCase):
    """Unit tests specifically covering the build_system_prompt API and acceptance criteria."""

    def test_acceptance_criteria_claims_adjuster_mt(self):
        """Acceptance Criteria: build_system_prompt('claims_adjuster', 'homeowners', 'MT')

        Returns non-empty string with 'MT' present.
        """
        output = build_system_prompt("claims_adjuster", "homeowners", "MT")
        self.assertIsInstance(output, str)
        self.assertGreater(len(output), 0)
        self.assertIn("MT", output)
        self.assertIn("homeowners insurance", output)

    def test_acceptance_criteria_claims_adjuster_none_state_fallback(self):
        """Acceptance Criteria: build_system_prompt('claims_adjuster', 'homeowners', None)

        Falls back to 'MT' without raising an error.
        """
        output = build_system_prompt("claims_adjuster", "homeowners", None)
        self.assertIsInstance(output, str)
        self.assertIn("in MT.", output)
        self.assertNotIn("{state}", output)
        self.assertIn("Line of business: homeowners insurance.", output)

    def test_acceptance_criteria_layman_linguist_tx(self):
        """Acceptance Criteria: build_system_prompt('layman_linguist', 'personal_auto', 'TX')

        Returns a valid string.
        """
        output = build_system_prompt("layman_linguist", "personal_auto", "TX")
        self.assertIsInstance(output, str)
        self.assertIn(ROLE_PROMPTS["layman_linguist"], output)
        self.assertIn("Line of business: personal auto insurance.", output)

    def test_build_system_prompt_pure_and_deterministic(self):
        """Acceptance Criteria: Function is pure - no I/O, no side effects, deterministic output."""
        res1 = build_system_prompt("underwriter", "commercial_pnc", "IL")
        res2 = build_system_prompt("underwriter", "commercial_pnc", "IL")
        self.assertEqual(res1, res2)
        self.assertIn("in IL.", res1)
        self.assertIn("commercial property and general liability insurance", res1)

    def test_build_system_prompt_with_enum_arguments(self):
        """Verify build_system_prompt works identically with RoleEnum and LOBEnum."""
        res_str = build_system_prompt("underwriter", "commercial_auto", "FL")
        res_enum = build_system_prompt(RoleEnum.underwriter, LOBEnum.commercial_auto, "FL")
        self.assertEqual(res_str, res_enum)

    def test_build_system_prompt_without_lob_or_state(self):
        """Verify default arguments: LOB is optional, state defaults to MT."""
        output = build_system_prompt("underwriter")
        self.assertIn("in MT.", output)
        self.assertNotIn("Line of business:", output)

    def test_build_system_prompt_empty_state_fallback(self):
        """Verify empty and whitespace strings fallback to MT."""
        output_empty = build_system_prompt("claims_adjuster", "umbrella", "")
        self.assertIn("in MT.", output_empty)
        output_ws = build_system_prompt("claims_adjuster", "umbrella", "   ")
        self.assertIn("in MT.", output_ws)


class TestArbiterPrompt(unittest.TestCase):
    """Unit tests for the Arbiter system prompt template and build_arbiter_prompt."""

    def test_arbiter_system_prompt_constant_and_placeholders(self):
        """Verify ARBITER_SYSTEM_PROMPT is defined and contains required placeholders."""
        self.assertIsInstance(ARBITER_SYSTEM_PROMPT, str)
        self.assertIn("{role}", ARBITER_SYSTEM_PROMPT)
        self.assertIn("{line_of_business}", ARBITER_SYSTEM_PROMPT)
        self.assertIn("{state}", ARBITER_SYSTEM_PROMPT)

    def test_arbiter_system_prompt_verbatim_content(self):
        """Verify ARBITER_SYSTEM_PROMPT verbatim content matches spec section 4.4."""
        expected = (
            "You are an expert executive AI insurance arbiter and fact-checker. You are given an original insurance query, "
            "the active professional persona ({role}, {line_of_business}, {state}), and answers from multiple independent AI models. "
            "Your task is to:\n"
            "1. Identify common ground and core facts agreed upon by the models.\n"
            "2. Detect and reconcile conflicting statements, eliminating obvious hallucinations, incorrect policy terms, or statutory inaccuracies.\n"
            "3. Synthesize the most accurate, structured, and clear response possible matching the requested persona.\n"
            "4. Maintain a professional, definitive tone. Do NOT mention 'Model A said' or cite specific AI names in your final output unless there is an unresolvable contradiction that the user must be alerted to."
        )
        self.assertEqual(ARBITER_SYSTEM_PROMPT, expected)

    def test_build_arbiter_prompt_acceptance_criteria(self):
        """Acceptance Criteria: build_arbiter_prompt('claims_adjuster', 'homeowners', 'MT')

        Returns the full arbiter prompt with 'claims_adjuster', 'homeowners', and 'MT'
        substituted in for {role}, {line_of_business}, and {state} respectively.
        """
        result = build_arbiter_prompt("claims_adjuster", "homeowners", "MT")
        expected = (
            "You are an expert executive AI insurance arbiter and fact-checker. You are given an original insurance query, "
            "the active professional persona (claims_adjuster, homeowners, MT), and answers from multiple independent AI models. "
            "Your task is to:\n"
            "1. Identify common ground and core facts agreed upon by the models.\n"
            "2. Detect and reconcile conflicting statements, eliminating obvious hallucinations, incorrect policy terms, or statutory inaccuracies.\n"
            "3. Synthesize the most accurate, structured, and clear response possible matching the requested persona.\n"
            "4. Maintain a professional, definitive tone. Do NOT mention 'Model A said' or cite specific AI names in your final output unless there is an unresolvable contradiction that the user must be alerted to."
        )
        self.assertEqual(result, expected)
        self.assertNotIn("{role}", result)
        self.assertNotIn("{line_of_business}", result)
        self.assertNotIn("{state}", result)

    def test_build_arbiter_prompt_pure_and_deterministic(self):
        """Acceptance Criteria: Function is pure - deterministic, no I/O, no side effects."""
        res1 = build_arbiter_prompt("underwriter", "commercial_pnc", "CA")
        res2 = build_arbiter_prompt("underwriter", "commercial_pnc", "CA")
        self.assertEqual(res1, res2)
        # Ensure underlying constant template is unchanged
        self.assertIn("{role}", ARBITER_SYSTEM_PROMPT)
        self.assertIn("{line_of_business}", ARBITER_SYSTEM_PROMPT)
        self.assertIn("{state}", ARBITER_SYSTEM_PROMPT)

    def test_build_arbiter_prompt_with_enum_arguments(self):
        """Verify build_arbiter_prompt works identically with RoleEnum and LOBEnum."""
        res_str = build_arbiter_prompt("claims_adjuster", "homeowners", "MT")
        res_enum = build_arbiter_prompt(RoleEnum.claims_adjuster, LOBEnum.homeowners, "MT")
        self.assertEqual(res_str, res_enum)

    def test_build_arbiter_prompt_state_fallback(self):
        """Verify defensive state fallback to 'MT' when omitted, None, or whitespace."""
        res_default = build_arbiter_prompt("underwriter", "personal_auto")
        self.assertIn("(underwriter, personal_auto, MT)", res_default)

        res_none = build_arbiter_prompt("underwriter", "personal_auto", None)
        self.assertIn("(underwriter, personal_auto, MT)", res_none)

        res_ws = build_arbiter_prompt("underwriter", "personal_auto", "   ")
        self.assertIn("(underwriter, personal_auto, MT)", res_ws)

    def test_build_arbiter_prompt_custom_state(self):
        """Verify custom state is correctly interpolated."""
        res_custom = build_arbiter_prompt("aca_expert", "aca_health", "FL")
        self.assertIn("(aca_expert, aca_health, FL)", res_custom)

    def test_build_arbiter_prompt_invalid_role(self):
        """Verify invalid role raises ValueError with descriptive message."""
        with self.assertRaises(ValueError) as cm:
            build_arbiter_prompt("invalid_role", "homeowners", "MT")
        self.assertIn("Invalid role 'invalid_role'", str(cm.exception))

    def test_build_arbiter_prompt_invalid_lob(self):
        """Verify invalid line_of_business raises ValueError with descriptive message."""
        with self.assertRaises(ValueError) as cm:
            build_arbiter_prompt("underwriter", "marine_cargo", "MT")
        self.assertIn("Invalid line of business 'marine_cargo'", str(cm.exception))

    def test_import_from_app_core(self):
        """Acceptance Criteria: from app.core.prompts import build_arbiter_prompt works without error."""
        from app.core import ARBITER_SYSTEM_PROMPT as RE_EXPORTED_PROMPT
        from app.core import build_arbiter_prompt as re_exported_builder
        from app.core.prompts import ARBITER_SYSTEM_PROMPT as CORE_PROMPT
        from app.core.prompts import build_arbiter_prompt as core_builder

        self.assertIs(RE_EXPORTED_PROMPT, CORE_PROMPT)
        self.assertIs(re_exported_builder, core_builder)


if __name__ == "__main__":
    unittest.main()


