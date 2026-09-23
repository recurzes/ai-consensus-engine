import importlib
import os
import unittest
from unittest.mock import patch
from pydantic import ValidationError

# Set dummy env vars during module load so test discovery succeeds without a local .env
_TEST_ENV_DEFAULTS = {
    "GEMINI_API_KEY": "test-gemini-key",
    "OPENAI_API_KEY": "test-openai-key",
    "ANTHROPIC_API_KEY": "test-anthropic-key",
    "ARBITER_MODEL_PROVIDER": "gemini",
}
for k, v in _TEST_ENV_DEFAULTS.items():
    os.environ.setdefault(k, v)

import app.config
from app.config import Settings


class TestConfig(unittest.TestCase):
    """Unit tests for centralized configuration and settings validation."""

    def tearDown(self):
        # Always restore app.config to a valid state after each test
        with patch.dict(os.environ, _TEST_ENV_DEFAULTS, clear=True):
            importlib.reload(app.config)

    def test_default_values(self):
        """Verify default timeout is 12 and default state is 'MT'."""
        s = Settings(
            _env_file=None,
            gemini_api_key="gemini-key",
            openai_api_key="openai-key",
            anthropic_api_key="anthropic-key",
            arbiter_model_provider="gemini",
        )
        self.assertEqual(s.request_timeout_seconds, 12)
        self.assertEqual(s.default_state, "MT")
        self.assertEqual(s.arbiter_model_provider, "gemini")

    def test_custom_values(self):
        """Verify custom timeout and state override defaults."""
        s = Settings(
            _env_file=None,
            gemini_api_key="gemini-key",
            openai_api_key="openai-key",
            anthropic_api_key="anthropic-key",
            arbiter_model_provider="openai",
            request_timeout_seconds=30,
            default_state="CA",
        )
        self.assertEqual(s.request_timeout_seconds, 30)
        self.assertEqual(s.default_state, "CA")
        self.assertEqual(s.arbiter_model_provider, "openai")

    def test_missing_required_keys(self):
        """Verify ValidationError is raised when required keys are missing."""
        base_args = {
            "gemini_api_key": "gemini-key",
            "openai_api_key": "openai-key",
            "anthropic_api_key": "anthropic-key",
            "arbiter_model_provider": "gemini",
        }

        for key in base_args:
            args = {k: v for k, v in base_args.items() if k != key}
            with self.subTest(missing_key=key):
                with patch.dict(os.environ, {}, clear=True):
                    with self.assertRaises(ValidationError) as ctx:
                        Settings(_env_file=None, **args)
                    errors = ctx.exception.errors()
                    self.assertTrue(any(e["loc"] == (key,) for e in errors))

    def test_empty_api_key_fails_validation(self):
        """Verify empty API key strings are rejected."""
        with patch.dict(os.environ, {}, clear=True):
            with self.assertRaises(ValidationError):
                Settings(
                    _env_file=None,
                    gemini_api_key="",
                    openai_api_key="openai-key",
                    anthropic_api_key="anthropic-key",
                    arbiter_model_provider="gemini",
                )

    def test_arbiter_provider_normalization_and_validation(self):
        """Verify case normalization and rejection of invalid arbiter providers."""
        # Uppercase and padded whitespace should normalize
        s_gemini = Settings(
            _env_file=None,
            gemini_api_key="g",
            openai_api_key="o",
            anthropic_api_key="a",
            arbiter_model_provider="  GEMINI  ",
        )
        self.assertEqual(s_gemini.arbiter_model_provider, "gemini")

        s_openai = Settings(
            _env_file=None,
            gemini_api_key="g",
            openai_api_key="o",
            anthropic_api_key="a",
            arbiter_model_provider="OpenAI",
        )
        self.assertEqual(s_openai.arbiter_model_provider, "openai")

        # Invalid provider values should fail loudly
        for invalid_provider in ["claude", "mistral", "llama", ""]:
            with self.subTest(provider=invalid_provider):
                with patch.dict(os.environ, {}, clear=True):
                    with self.assertRaises(ValidationError):
                        Settings(
                            _env_file=None,
                            gemini_api_key="g",
                            openai_api_key="o",
                            anthropic_api_key="a",
                            arbiter_model_provider=invalid_provider,
                        )

    def test_request_timeout_must_be_positive(self):
        """Verify request_timeout_seconds must be greater than 0."""
        for invalid_timeout in [0, -1, -10]:
            with self.subTest(timeout=invalid_timeout):
                with patch.dict(os.environ, {}, clear=True):
                    with self.assertRaises(ValidationError):
                        Settings(
                            _env_file=None,
                            gemini_api_key="g",
                            openai_api_key="o",
                            anthropic_api_key="a",
                            arbiter_model_provider="gemini",
                            request_timeout_seconds=invalid_timeout,
                        )

    def test_loading_from_environment_variables(self):
        """Verify environment variables are read and mapped properly."""
        env_vars = {
            "GEMINI_API_KEY": "env-gemini",
            "OPENAI_API_KEY": "env-openai",
            "ANTHROPIC_API_KEY": "env-anthropic",
            "ARBITER_MODEL_PROVIDER": "OPENAI",
            "REQUEST_TIMEOUT_SECONDS": "25",
            "DEFAULT_STATE": "WA",
        }
        with patch.dict(os.environ, env_vars, clear=True):
            s = Settings(_env_file=None)
            self.assertEqual(s.gemini_api_key, "env-gemini")
            self.assertEqual(s.openai_api_key, "env-openai")
            self.assertEqual(s.anthropic_api_key, "env-anthropic")
            self.assertEqual(s.arbiter_model_provider, "openai")
            self.assertEqual(s.request_timeout_seconds, 25)
            self.assertEqual(s.default_state, "WA")

    def test_module_singleton_import_success(self):
        """Verify from app.config import settings works when env is valid."""
        env_vars = {
            "GEMINI_API_KEY": "singleton-gemini",
            "OPENAI_API_KEY": "singleton-openai",
            "ANTHROPIC_API_KEY": "singleton-anthropic",
            "ARBITER_MODEL_PROVIDER": "gemini",
        }
        with patch.dict(os.environ, env_vars, clear=True):
            importlib.reload(app.config)
            self.assertIsInstance(app.config.settings, app.config.Settings)
            self.assertEqual(app.config.settings.gemini_api_key, "singleton-gemini")
            self.assertEqual(app.config.settings.request_timeout_seconds, 12)
            self.assertEqual(app.config.settings.default_state, "MT")

    def test_module_singleton_import_fails_on_missing_required_keys(self):
        """Verify importing/reloading app.config fails fast when required key is missing."""
        # Clean environment with no keys set
        with patch.dict(os.environ, {}, clear=True):
            with self.assertRaises(ValidationError) as ctx:
                importlib.reload(app.config)
            error_fields = [e["loc"][0] for e in ctx.exception.errors()]
            self.assertIn("gemini_api_key", error_fields)
            self.assertIn("openai_api_key", error_fields)
            self.assertIn("anthropic_api_key", error_fields)
            self.assertIn("arbiter_model_provider", error_fields)


if __name__ == "__main__":
    unittest.main()
