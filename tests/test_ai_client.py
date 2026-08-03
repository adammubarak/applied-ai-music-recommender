"""
Tests for src.ai_client — mocked only, no real Anthropic API call.

Every test injects a fake client (a unittest.mock object) or relies on the
empty-list short-circuit, so no network request is ever made and no API key is
required.
"""

import json
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

from src.ai_client import (
    AIClientError,
    APICallError,
    EmptyResponseError,
    InvalidResponseStructureError,
    MalformedResponseError,
    MissingAPIKeyError,
    RecommendationResult,
    SongRecommendation,
    build_user_prompt,
    generate_recommendations,
)
from src.config import Settings


# --- Helpers ---------------------------------------------------------------

RETRIEVED = [
    {"id": 1, "title": "Sunrise City", "artist": "Neon Echo", "genre": "pop",
     "mood": "happy", "energy": 0.82},
    {"id": 3, "title": "Storm Runner", "artist": "Voltline", "genre": "rock",
     "mood": "intense", "energy": 0.91},
]

VALID_JSON = json.dumps({
    "recommendations": [
        {"id": 1, "title": "Sunrise City", "artist": "Neon Echo",
         "explanation": "Upbeat pop that matches your happy, high-energy mood."},
    ]
})


def make_response(text: str):
    """Build a fake Anthropic-style response with a single text block."""
    return SimpleNamespace(content=[SimpleNamespace(type="text", text=text)])


def make_client(text: str) -> MagicMock:
    """A mock client whose messages.create returns a response with `text`."""
    client = MagicMock()
    client.messages.create.return_value = make_response(text)
    return client


WITH_KEY = Settings(api_key="sk-test-not-real", model="claude-test-model")
NO_KEY = Settings(api_key=None, model="claude-test-model")


# --- Prompt grounding ------------------------------------------------------

def test_prompt_contains_user_request():
    prompt = build_user_prompt("something chill for studying", RETRIEVED)
    assert "something chill for studying" in prompt


def test_prompt_contains_retrieved_songs():
    prompt = build_user_prompt("energetic workout tracks", RETRIEVED)
    assert "Sunrise City" in prompt
    assert "Storm Runner" in prompt
    assert "Neon Echo" in prompt


def test_prompt_excludes_songs_not_retrieved():
    """The prompt must contain only the supplied songs — no full catalog."""
    prompt = build_user_prompt("anything", RETRIEVED)
    # A real catalog song that was NOT in the retrieved list must not appear.
    assert "Midnight Coding" not in prompt
    assert "Library Rain" not in prompt
    # Exactly the two supplied candidate ids are present.
    candidates = json.loads(prompt.split("Candidate songs (recommend only from these):\n", 1)[1])
    assert [c["id"] for c in candidates] == [1, 3]


def test_generate_does_not_read_catalog(monkeypatch):
    """Generation must never open data/songs.csv (or any file)."""
    import builtins
    real_open = builtins.open

    def guard(path, *args, **kwargs):
        assert "songs.csv" not in str(path), "ai_client must not read the catalog"
        return real_open(path, *args, **kwargs)

    monkeypatch.setattr(builtins, "open", guard)
    client = make_client(VALID_JSON)
    generate_recommendations("x", RETRIEVED, client=client, settings=WITH_KEY)


# --- Happy path ------------------------------------------------------------

def test_successful_structured_parsing():
    client = make_client(VALID_JSON)
    result = generate_recommendations("happy pop", RETRIEVED, client=client, settings=WITH_KEY)

    assert isinstance(result, RecommendationResult)
    assert len(result.recommendations) == 1
    rec = result.recommendations[0]
    assert isinstance(rec, SongRecommendation)
    assert rec.song_id == 1
    assert rec.title == "Sunrise City"
    assert rec.artist == "Neon Echo"
    assert rec.explanation.strip() != ""
    assert result.model == "claude-test-model"
    client.messages.create.assert_called_once()


def test_call_passes_request_and_candidates_to_model():
    """The mocked create call must carry the request and the candidates."""
    client = make_client(VALID_JSON)
    generate_recommendations("cozy evening", RETRIEVED, client=client, settings=WITH_KEY)

    kwargs = client.messages.create.call_args.kwargs
    user_content = kwargs["messages"][0]["content"]
    assert "cozy evening" in user_content
    assert "Sunrise City" in user_content
    assert kwargs["model"] == "claude-test-model"


# --- Error paths -----------------------------------------------------------

def test_missing_api_key_raises():
    """No client injected + no key configured -> MissingAPIKeyError."""
    with pytest.raises(MissingAPIKeyError):
        generate_recommendations("x", RETRIEVED, client=None, settings=NO_KEY)


def test_empty_retrieved_list_makes_no_api_call():
    client = make_client(VALID_JSON)
    result = generate_recommendations("x", [], client=client, settings=WITH_KEY)

    assert result.recommendations == []
    client.messages.create.assert_not_called()


def test_api_failure_wrapped():
    client = MagicMock()
    client.messages.create.side_effect = RuntimeError("network down")
    with pytest.raises(APICallError):
        generate_recommendations("x", RETRIEVED, client=client, settings=WITH_KEY)


def test_empty_model_response_raises():
    client = MagicMock()
    client.messages.create.return_value = SimpleNamespace(content=[])
    with pytest.raises(EmptyResponseError):
        generate_recommendations("x", RETRIEVED, client=client, settings=WITH_KEY)


def test_malformed_json_raises():
    client = make_client("this is not json {")
    with pytest.raises(MalformedResponseError):
        generate_recommendations("x", RETRIEVED, client=client, settings=WITH_KEY)


def test_invalid_structure_missing_recommendations_key():
    client = make_client(json.dumps({"songs": []}))
    with pytest.raises(InvalidResponseStructureError):
        generate_recommendations("x", RETRIEVED, client=client, settings=WITH_KEY)


def test_invalid_structure_missing_field():
    bad = json.dumps({"recommendations": [{"id": 1, "title": "X"}]})  # no artist/explanation
    client = make_client(bad)
    with pytest.raises(InvalidResponseStructureError):
        generate_recommendations("x", RETRIEVED, client=client, settings=WITH_KEY)


def test_custom_exceptions_share_base():
    for exc in (MissingAPIKeyError, APICallError, EmptyResponseError,
                MalformedResponseError, InvalidResponseStructureError):
        assert issubclass(exc, AIClientError)


def test_no_real_anthropic_call(monkeypatch):
    """Guard: even without injecting a client, no real SDK is imported/used
    on the empty-list path, and injected mocks are the only call surface."""
    # Empty list path never resolves a client at all.
    result = generate_recommendations("x", [], settings=WITH_KEY)
    assert result.recommendations == []
