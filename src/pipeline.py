"""
Central application pipeline for the Explainable AI Music Discovery Assistant.

This is where Retrieval-Augmented Generation becomes part of the real
processing flow: deterministic retrieval → Claude generation (grounded on the
retrieved candidates only) → reliability validation → safe deterministic
fallback.

Sequence
--------
a. Validate basic inputs.
b. Retrieve top candidates via the functional core `recommend_songs()`.
c. Pass ONLY those retrieved candidates (never the full catalog) to the AI
   generation function.
d. Validate the AI response with `validate_recommendations()`.
e. Return grounded AI recommendations only if the complete response passed.
f. Otherwise return safe deterministic fallback recommendations.

Guarantees / non-responsibilities (Phase 4)
-------------------------------------------
The pipeline never: exposes an API key, reads configuration secrets, builds its
own Anthropic client, loads data/songs.csv, computes a second retrieval score
(it defers entirely to `recommend_songs()`), includes rejected AI
recommendations in the final output, depends on Streamlit, or logs anything.

The AI generation function is injected (`generation_function`) so tests stay
mocked and offline. In production it defaults to `ai_client.generate_recommendations`,
which owns all configuration/secret handling and lazy client construction.
"""

from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Tuple

from .ai_client import (
    AIClientError,
    APICallError,
    EmptyResponseError,
    InvalidResponseStructureError,
    MalformedResponseError,
    MissingAPIKeyError,
    RecommendationResult,
    generate_recommendations,
)
from .preference_parser import ParsedPreferences, parse_preferences
from .recommender import recommend_songs
from .reliability import ValidationReport, validate_recommendations


# --- Fallback reason codes (stable, secret-free strings) -------------------

FALLBACK_MISSING_API_KEY = "missing_api_key"
FALLBACK_API_ERROR = "api_error"
FALLBACK_EMPTY_RESPONSE = "empty_response"
FALLBACK_MALFORMED_JSON = "malformed_json"
FALLBACK_INVALID_STRUCTURE = "invalid_structure"
FALLBACK_AI_ERROR = "ai_error"
FALLBACK_VALIDATION_FAILED = "validation_failed"
FALLBACK_NO_CANDIDATES = "no_candidates"

# Required keys in the preferences dict for deterministic scoring
# (matches the canonical schema consumed by recommender.score_song).
REQUIRED_PREF_KEYS = ("genre", "mood", "energy")


# --- Typed results ---------------------------------------------------------

@dataclass(frozen=True)
class FinalRecommendation:
    """
    One recommendation in the final output, carrying enough for a future UI.

    For AI output, `explanation` is Claude's explanation; for fallback it is the
    deterministic scoring explanation. `score` is the deterministic score of the
    song when known (available for every retrieved candidate).
    """
    song_id: Any
    title: str
    artist: str
    explanation: str
    score: Optional[float] = None
    source: str = "fallback"  # "ai" or "fallback"


@dataclass(frozen=True)
class DiscoveryResult:
    """The outcome of a discovery run."""
    final_recommendations: List[FinalRecommendation] = field(default_factory=list)
    retrieved_candidates: List[Dict[str, Any]] = field(default_factory=list)
    source: str = "fallback"                       # "ai" or "fallback"
    used_fallback: bool = True
    fallback_reason: Optional[str] = None          # stable code; never a secret
    validation_report: Optional[ValidationReport] = None
    model: Optional[str] = None


# Type alias for the injected AI generation function.
GenerationFn = Callable[[str, List[Dict[str, Any]]], RecommendationResult]

# Type alias for the injected preference parser.
ParserFn = Callable[[str], ParsedPreferences]


@dataclass(frozen=True)
class RequestDiscoveryResult:
    """Result of the natural-language entry point: the parse plus the discovery."""
    parsed_preferences: ParsedPreferences
    discovery: DiscoveryResult


# --- Input validation ------------------------------------------------------

def _validate_inputs(
    user_request: str,
    preferences: Dict[str, Any],
    songs: List[Dict[str, Any]],
    retrieval_k: int,
    output_k: int,
) -> None:
    """Validate basic inputs; raise ValueError with a clear message on failure."""
    if not isinstance(user_request, str):
        raise ValueError("user_request must be a string.")
    if not isinstance(preferences, dict):
        raise ValueError("preferences must be a dict.")
    missing = [k for k in REQUIRED_PREF_KEYS if k not in preferences]
    if missing:
        raise ValueError(f"preferences missing required keys: {missing}")
    if not isinstance(songs, list):
        raise ValueError("songs must be a list.")
    if not isinstance(retrieval_k, int) or retrieval_k < 1:
        raise ValueError("retrieval_k must be a positive integer.")
    if not isinstance(output_k, int) or output_k < 1:
        raise ValueError("output_k must be a positive integer.")


# --- Result builders -------------------------------------------------------

def _build_fallback_result(
    ranked: List[Tuple[Dict[str, Any], float, str]],
    retrieved: List[Dict[str, Any]],
    output_k: int,
    reason: str,
    model: Optional[str] = None,
    report: Optional[ValidationReport] = None,
) -> DiscoveryResult:
    """Build a DiscoveryResult from the deterministic retrieval results."""
    recs = [
        FinalRecommendation(
            song_id=song["id"],
            title=song["title"],
            artist=song["artist"],
            explanation=explanation,
            score=score,
            source="fallback",
        )
        for song, score, explanation in ranked[:output_k]
    ]
    return DiscoveryResult(
        final_recommendations=recs,
        retrieved_candidates=retrieved,
        source="fallback",
        used_fallback=True,
        fallback_reason=reason,
        validation_report=report,
        model=model,
    )


def _build_ai_result(
    report: ValidationReport,
    ranked: List[Tuple[Dict[str, Any], float, str]],
    retrieved: List[Dict[str, Any]],
    output_k: int,
    model: Optional[str],
) -> DiscoveryResult:
    """Build a DiscoveryResult from validated AI recommendations."""
    score_by_id = {str(song["id"]): score for song, score, _ in ranked}
    recs = [
        FinalRecommendation(
            song_id=rec.song_id,
            title=rec.title,
            artist=rec.artist,
            explanation=rec.explanation,
            score=score_by_id.get(str(rec.song_id)),
            source="ai",
        )
        for rec in report.valid[:output_k]
    ]
    return DiscoveryResult(
        final_recommendations=recs,
        retrieved_candidates=retrieved,
        source="ai",
        used_fallback=False,
        fallback_reason=None,
        validation_report=report,
        model=model,
    )


# --- Public interface ------------------------------------------------------

def discover_music(
    user_request: str,
    preferences: Dict[str, Any],
    songs: List[Dict[str, Any]],
    *,
    retrieval_k: int = 5,
    output_k: int = 3,
    generation_function: GenerationFn = generate_recommendations,
) -> DiscoveryResult:
    """
    Run the end-to-end discovery pipeline.

    Args:
        user_request: The user's natural-language music request.
        preferences: Canonical preferences dict (keys: genre, mood, energy),
            used by the deterministic scorer.
        songs: The full song catalog (already loaded by the caller — the
            pipeline does not read data/songs.csv).
        retrieval_k: How many candidates to retrieve and hand to the AI.
        output_k: Maximum number of recommendations in the final output.
        generation_function: Injected AI generation function. Defaults to
            ai_client.generate_recommendations. Injected as a mock in tests.

    Returns:
        A DiscoveryResult. AI output is used only when the complete response
        passes validation; otherwise a safe deterministic fallback is returned.
    """
    _validate_inputs(user_request, preferences, songs, retrieval_k, output_k)

    # b. Deterministic retrieval — single source of truth for scoring.
    ranked = recommend_songs(preferences, songs, k=retrieval_k)
    retrieved = [song for song, _score, _explanation in ranked]

    # Safe handling when there is nothing to ground on.
    if not retrieved:
        return _build_fallback_result(
            ranked, retrieved, output_k, reason=FALLBACK_NO_CANDIDATES
        )

    # c. Generation on retrieved candidates only. Any AI failure -> fallback.
    try:
        result = generation_function(user_request, retrieved)
    except MissingAPIKeyError:
        return _build_fallback_result(ranked, retrieved, output_k, FALLBACK_MISSING_API_KEY)
    except APICallError:
        return _build_fallback_result(ranked, retrieved, output_k, FALLBACK_API_ERROR)
    except EmptyResponseError:
        return _build_fallback_result(ranked, retrieved, output_k, FALLBACK_EMPTY_RESPONSE)
    except MalformedResponseError:
        return _build_fallback_result(ranked, retrieved, output_k, FALLBACK_MALFORMED_JSON)
    except InvalidResponseStructureError:
        return _build_fallback_result(ranked, retrieved, output_k, FALLBACK_INVALID_STRUCTURE)
    except AIClientError:
        # Any other AI client error — stay safe.
        return _build_fallback_result(ranked, retrieved, output_k, FALLBACK_AI_ERROR)

    # d. Validate against the retrieved set.
    report = validate_recommendations(result, retrieved)

    # e. Use AI output only if the complete response passed.
    if report.passed:
        return _build_ai_result(report, ranked, retrieved, output_k, result.model)

    # f. Otherwise, safe deterministic fallback (no second AI call).
    return _build_fallback_result(
        ranked, retrieved, output_k, FALLBACK_VALIDATION_FAILED,
        model=result.model, report=report,
    )


def discover_from_request(
    user_request: str,
    songs: List[Dict[str, Any]],
    *,
    retrieval_k: int = 5,
    output_k: int = 3,
    parser_function: ParserFn = parse_preferences,
    generation_function: GenerationFn = generate_recommendations,
) -> RequestDiscoveryResult:
    """
    Natural-language entry point.

    Parses `user_request` into canonical preferences, then delegates to the
    existing `discover_music()` pipeline (no retrieval/generation/validation/
    fallback logic is duplicated here). Both parser and generation functions are
    injectable for offline testing.

    Returns a RequestDiscoveryResult exposing the parsed preferences and the
    DiscoveryResult produced by discover_music().
    """
    parsed = parser_function(user_request)
    preferences = parsed.to_prefs()
    discovery = discover_music(
        user_request,
        preferences,
        songs,
        retrieval_k=retrieval_k,
        output_k=output_k,
        generation_function=generation_function,
    )
    return RequestDiscoveryResult(parsed_preferences=parsed, discovery=discovery)
