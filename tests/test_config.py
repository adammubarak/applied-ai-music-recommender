"""
Tests for src.config — safe environment/config handling.

These verify that configuration loads without an API key, reads environment
values, reports key availability, and never exposes the key via str()/repr().
They also check that .env.example ships placeholders rather than a real secret.
No real API key is used and no API call is made.
"""

import importlib
import os

from src.config import Settings, load_settings

EXAMPLE_PATH = os.path.join(os.path.dirname(__file__), "..", ".env.example")


def test_import_without_api_key(monkeypatch):
    """Importing/reloading config with no API key must not raise."""
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    monkeypatch.delenv("ANTHROPIC_MODEL", raising=False)

    import src.config as config
    reloaded = importlib.reload(config)

    assert reloaded.settings.has_api_key is False
    assert reloaded.settings.model == reloaded.DEFAULT_MODEL


def test_load_settings_reads_environment(monkeypatch):
    """load_settings should pick up API key and model from the environment."""
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-test-do-not-use")
    monkeypatch.setenv("ANTHROPIC_MODEL", "claude-test-model")

    settings = load_settings()
    assert settings.has_api_key is True
    assert settings.api_key == "sk-test-do-not-use"
    assert settings.model == "claude-test-model"


def test_model_defaults_when_absent(monkeypatch):
    """A missing ANTHROPIC_MODEL falls back to DEFAULT_MODEL."""
    monkeypatch.delenv("ANTHROPIC_MODEL", raising=False)
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-test-do-not-use")

    from src.config import DEFAULT_MODEL

    settings = load_settings()
    assert settings.model == DEFAULT_MODEL


def test_has_api_key_reports_availability():
    """has_api_key reflects presence of a non-blank key."""
    assert Settings(api_key="sk-something").has_api_key is True
    assert Settings(api_key=None).has_api_key is False
    assert Settings(api_key="   ").has_api_key is False  # blank counts as absent


def test_api_key_hidden_from_str_and_repr():
    """The API key must never appear in str() or repr()."""
    secret = "sk-super-secret-value-12345"
    settings = Settings(api_key=secret, model="claude-test-model")

    assert secret not in repr(settings)
    assert secret not in str(settings)
    # The representation still communicates availability and model.
    assert "api_key_available=True" in repr(settings)
    assert "claude-test-model" in repr(settings)


def test_env_example_contains_placeholders_only():
    """.env.example must ship placeholders, not a real secret."""
    with open(EXAMPLE_PATH, encoding="utf-8") as f:
        content = f.read()

    assert "ANTHROPIC_API_KEY=your_api_key_here" in content
    assert "ANTHROPIC_MODEL=your_model_name_here" in content
    # Guard against a real Anthropic key being committed by mistake.
    assert "sk-ant" not in content
