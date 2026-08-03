"""
Claude AI client for grounded music-recommendation explanations.

Responsibility (Phase 2 only)
-----------------------------
Given a user's music request and a list of *already-retrieved* song
dictionaries, ask Claude to select and explain songs **only** from that
supplied list, and return a typed result.

This module is deliberately narrow:
- It never loads data/songs.csv and never performs retrieval — the caller
  supplies the candidate songs.
- It does NOT validate that recommendations actually exist in the candidate
  list (hallucination checking), compute confidence/reliability, or fall back
  to the deterministic recommender. Those belong to later phases.

Grounding
---------
The candidate songs are the only recommendation context handed to Claude. The
system prompt instructs Claude to recommend strictly from that list and to
return JSON; the user prompt embeds the request plus the candidate list.

Testability
-----------
The Anthropic client is injected (the `client` argument). When omitted, a real
client is constructed lazily from `src.config.settings` — so importing this
module never requires the `anthropic` package, and mocked tests never touch it.
The API key is never printed, logged, or exposed.
"""

import json
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from .config import Settings
from .config import settings as _default_settings


# --- Custom exceptions -----------------------------------------------------

class AIClientError(Exception):
    """Base class for all AI client errors."""


class MissingAPIKeyError(AIClientError):
    """Raised when no API key is configured and a real client is required."""


class APICallError(AIClientError):
    """Raised when the Anthropic API call itself fails."""


class EmptyResponseError(AIClientError):
    """Raised when the model returns no usable text content."""


class MalformedResponseError(AIClientError):
    """Raised when the model response is not valid JSON."""


class InvalidResponseStructureError(AIClientError):
    """Raised when the parsed JSON does not match the expected structure."""


# --- Typed results ---------------------------------------------------------

@dataclass(frozen=True)
class SongRecommendation:
    """A single recommended song and its explanation, as returned by Claude."""
    song_id: Any          # int or str, as supplied by the model
    title: str
    artist: str
    explanation: str


@dataclass(frozen=True)
class RecommendationResult:
    """The typed result of an AI generation call."""
    recommendations: List[SongRecommendation] = field(default_factory=list)
    model: Optional[str] = None


# --- Prompt construction ---------------------------------------------------

SYSTEM_PROMPT = (
    "You are a music discovery assistant. You will be given a user's request "
    "and a list of candidate songs (the ONLY songs you may recommend). "
    "Select the most relevant songs strictly from the candidate list and write "
    "a short, personalized explanation for each. Never invent songs, artists, "
    "or IDs that are not in the candidate list.\n\n"
    "Respond with ONLY a JSON object of the form:\n"
    '{"recommendations": [{"id": <candidate id>, "title": <string>, '
    '"artist": <string>, "explanation": <string>}]}\n'
    "Use the exact id, title, and artist from the candidate list. Output no "
    "text outside the JSON object."
)

# Fields from each retrieved song dict that are safe/useful to show Claude.
_CANDIDATE_FIELDS = ("id", "title", "artist", "genre", "mood", "energy")


def _candidate_view(song: Dict[str, Any]) -> Dict[str, Any]:
    """Trim a retrieved song dict to the fields shown to the model."""
    return {k: song[k] for k in _CANDIDATE_FIELDS if k in song}


def build_user_prompt(
    user_request: str, retrieved_songs: List[Dict[str, Any]]
) -> str:
    """
    Build the user-turn prompt from the request and the retrieved songs.

    The retrieved songs are the entire recommendation context; no catalog is
    read or embedded here.
    """
    candidates = [_candidate_view(song) for song in retrieved_songs]
    candidates_json = json.dumps(candidates, ensure_ascii=False, indent=2)
    return (
        f"User request:\n{user_request}\n\n"
        f"Candidate songs (recommend only from these):\n{candidates_json}"
    )


# --- Response parsing ------------------------------------------------------

def _extract_text(response: Any) -> str:
    """
    Pull the concatenated text from an Anthropic-style response.

    Accepts content blocks that are either objects with a `.text` attribute or
    plain dicts with a "text" key. Raises EmptyResponseError if nothing usable
    is present.
    """
    content = getattr(response, "content", None)
    if not content:
        raise EmptyResponseError("Model response contained no content.")

    parts: List[str] = []
    for block in content:
        text = getattr(block, "text", None)
        if text is None and isinstance(block, dict):
            text = block.get("text")
        if text:
            parts.append(text)

    combined = "".join(parts).strip()
    if not combined:
        raise EmptyResponseError("Model response contained no text.")
    return combined


def _parse_recommendations(text: str) -> List[SongRecommendation]:
    """Parse and validate the model's JSON into SongRecommendation objects."""
    try:
        data = json.loads(text)
    except json.JSONDecodeError as exc:
        raise MalformedResponseError("Model response was not valid JSON.") from exc

    if not isinstance(data, dict) or not isinstance(data.get("recommendations"), list):
        raise InvalidResponseStructureError(
            "Response must be an object with a 'recommendations' list."
        )

    recommendations: List[SongRecommendation] = []
    for item in data["recommendations"]:
        if not isinstance(item, dict):
            raise InvalidResponseStructureError("Each recommendation must be an object.")
        try:
            song_id = item["id"]
            title = item["title"]
            artist = item["artist"]
            explanation = item["explanation"]
        except KeyError as exc:
            raise InvalidResponseStructureError(
                f"Recommendation missing required field: {exc}"
            ) from exc

        if not isinstance(title, str) or not isinstance(artist, str) or not isinstance(explanation, str):
            raise InvalidResponseStructureError(
                "Recommendation fields 'title', 'artist', and 'explanation' must be strings."
            )

        recommendations.append(
            SongRecommendation(
                song_id=song_id,
                title=title,
                artist=artist,
                explanation=explanation,
            )
        )
    return recommendations


# --- Client resolution -----------------------------------------------------

def _resolve_client(client: Optional[Any], settings: Settings) -> Any:
    """
    Return the injected client, or lazily construct a real Anthropic client.

    The `anthropic` package is imported only here, so importing this module
    (and running mocked tests) never requires it. Raises MissingAPIKeyError
    when no key is configured.
    """
    if client is not None:
        return client
    if not settings.has_api_key:
        raise MissingAPIKeyError("ANTHROPIC_API_KEY is not configured.")

    import anthropic  # lazy import — only when a real client is needed

    return anthropic.Anthropic(api_key=settings.api_key)


# --- Public interface ------------------------------------------------------

def generate_recommendations(
    user_request: str,
    retrieved_songs: List[Dict[str, Any]],
    *,
    client: Optional[Any] = None,
    settings: Optional[Settings] = None,
    max_tokens: int = 1024,
) -> RecommendationResult:
    """
    Ask Claude to recommend and explain songs from the retrieved candidates.

    Args:
        user_request: The user's natural-language music request.
        retrieved_songs: Already-retrieved candidate song dicts (the only
            recommendation context). This function does not perform retrieval.
        client: An Anthropic-style client to use. If None, a real client is
            built from configuration. Inject a mock in tests.
        settings: Configuration override; defaults to the module singleton.
        max_tokens: Max output tokens for the model call.

    Returns:
        A RecommendationResult. If `retrieved_songs` is empty, returns an empty
        result WITHOUT making an API call.

    Raises:
        MissingAPIKeyError: No API key configured and no client injected.
        APICallError: The Anthropic API call failed.
        EmptyResponseError: The model returned no usable text.
        MalformedResponseError: The response text was not valid JSON.
        InvalidResponseStructureError: The JSON did not match the schema.
    """
    settings = settings or _default_settings

    # Never call the API with no candidates to ground on.
    if not retrieved_songs:
        return RecommendationResult(recommendations=[], model=settings.model)

    resolved = _resolve_client(client, settings)
    prompt = build_user_prompt(user_request, retrieved_songs)

    try:
        response = resolved.messages.create(
            model=settings.model,
            max_tokens=max_tokens,
            system=SYSTEM_PROMPT,
            messages=[{"role": "user", "content": prompt}],
        )
    except AIClientError:
        raise
    except Exception as exc:  # wrap any SDK/transport error
        raise APICallError(f"Anthropic API call failed: {exc}") from exc

    text = _extract_text(response)
    recommendations = _parse_recommendations(text)
    return RecommendationResult(recommendations=recommendations, model=settings.model)
