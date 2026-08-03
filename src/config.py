"""
Configuration and safe secret handling for the Music Discovery Assistant.

This module is the single place that reads environment variables for the
future Anthropic Claude integration. It is deliberately import-safe: importing
it (or constructing `Settings`) never requires an API key and never raises when
one is absent, so unit tests and offline runs work without secrets.

Security
--------
The API key is never printed, logged, or included in `str()` / `repr()` output.
`Settings` masks it: its representation reports only whether a key is available,
never the value. Read the key through `settings.api_key` when actually calling
the API; do not log that value.

Usage
-----
    from src.config import settings

    if settings.has_api_key:
        client = anthropic.Anthropic(api_key=settings.api_key)
        model = settings.model
    else:
        # fall back to the deterministic recommender (added in a later phase)
        ...
"""

import os
from dataclasses import dataclass, field
from typing import Optional

from dotenv import load_dotenv

# Load variables from a local .env file if present. This is a no-op when the
# file is absent, so imports never fail for lack of an .env.
load_dotenv()

# Fallback model used only when ANTHROPIC_MODEL is not set. Overridable via the
# environment; see .env.example.
DEFAULT_MODEL = "claude-sonnet-5"


@dataclass(frozen=True, repr=False)
class Settings:
    """
    Application settings for the AI integration.

    Attributes:
        api_key: The Anthropic API key, or None if not configured. Excluded
            from the generated representation; read it only when calling the API.
        model: The Anthropic model id to use (defaults to DEFAULT_MODEL).
    """

    api_key: Optional[str] = field(default=None, repr=False)
    model: str = DEFAULT_MODEL

    @property
    def has_api_key(self) -> bool:
        """Return True if a non-empty API key is configured."""
        return bool(self.api_key and self.api_key.strip())

    def __repr__(self) -> str:
        # Never expose the key — report only its availability.
        return f"Settings(model={self.model!r}, api_key_available={self.has_api_key})"

    # Same masked output for str() as for repr().
    __str__ = __repr__


def load_settings() -> Settings:
    """
    Build a Settings object from the current environment.

    Reads ANTHROPIC_API_KEY and ANTHROPIC_MODEL. A missing or blank API key is
    tolerated (returns None) so imports and tests never crash without secrets.
    """
    api_key = os.environ.get("ANTHROPIC_API_KEY") or None
    model = os.environ.get("ANTHROPIC_MODEL") or DEFAULT_MODEL
    return Settings(api_key=api_key, model=model)


# Module-level singleton for convenient import by future modules.
settings = load_settings()
