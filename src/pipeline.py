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

Guarantees / non-responsibilities
---------------------------------
The pipeline never: exposes an API key, reads configuration secrets, builds its
own Anthropic client, loads data/songs.csv, computes a second retrieval score
(it defers entirely to `recommend_songs()`), or includes rejected AI
recommendations in the final output.

Logging is structured and injected (see `logging_config`), non-sensitive, and
fully contained — a logging failure never changes recommendation behavior. The
AI generation function is injected so tests stay mocked and offline.
"""

import uuid
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
from .logging_config import (
    EVENT_AI_GENERATION_COMPLETED,
    EVENT_AI_GENERATION_STARTED,
    EVENT_FALLBACK_USED,
    EVENT_PIPELINE_COMPLETED,
    EVENT_PIPELINE_INPUT_ERROR,
    EVENT_PIPELINE_STARTED,
    EVENT_PREFERENCES_PARSED,
    EVENT_RETRIEVAL_COMPLETED,
    EVENT_VALIDATION_COMPLETED,
    get_logger,
    log_event,
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

# Required keys in the preferences dict for deterministic scoring.
REQUIRED_PREF_KEYS = ("genre", "mood", "energy")


# --- Typed results ---------------------------------------------------------

@dataclass(frozen=True)
class FinalRecommendation:
    """One recommendation in the final output, carrying enough for a future UI."""
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
    request_id: Optional[str] = None               # correlates log events


@dataclass(frozen=True)
class RequestDiscoveryResult:
    """Result of the natural-language entry point: the parse plus the discovery."""
    parsed_preferences: ParsedPreferences
    discovery: DiscoveryResult
    request_id: Optional[str] = None


GenerationFn = Callable[[str, List[Dict[str, Any]]], RecommendationResult]
ParserFn = Callable[[str], ParsedPreferences]


# --- Helpers ---------------------------------------------------------------

def _new_request_id() -> str:
    """Generate a short unique id to correlate one run's events."""
    return uuid.uuid4().hex


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


def _build_fallback_result(
    ranked: List[Tuple[Dict[str, Any], float, str]],
    retrieved: List[Dict[str, Any]],
    output_k: int,
    reason: str,
    model: Optional[str] = None,
    report: Optional[ValidationReport] = None,
    request_id: Optional[str] = None,
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
        request_id=request_id,
    )


def _build_ai_result(
    report: ValidationReport,
    ranked: List[Tuple[Dict[str, Any], float, str]],
    retrieved: List[Dict[str, Any]],
    output_k: int,
    model: Optional[str],
    request_id: Optional[str] = None,
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
        request_id=request_id,
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
    logger: Any = None,
    request_id: Optional[str] = None,
) -> DiscoveryResult:
    """
    Run the end-to-end discovery pipeline.

    Args:
        user_request: The user's natural-language music request.
        preferences: Canonical preferences dict (keys: genre, mood, energy).
        songs: The full song catalog (already loaded by the caller).
        retrieval_k: How many candidates to retrieve and hand to the AI.
        output_k: Maximum number of recommendations in the final output.
        generation_function: Injected AI generation function.
        logger: Optional logger for structured events (injected; defaults to the
            shared silent logger so existing callers are unaffected).
        request_id: Optional correlation id; generated when omitted.

    Returns:
        A DiscoveryResult. AI output is used only when the complete response
        passes validation; otherwise a safe deterministic fallback is returned.
    """
    logger = logger or get_logger()
    request_id = request_id or _new_request_id()

    req_is_str = isinstance(user_request, str)
    log_event(
        logger, EVENT_PIPELINE_STARTED,
        request_id=request_id,
        request_chars=len(user_request) if req_is_str else 0,
        request_empty=(not (req_is_str and user_request.strip())),
        retrieval_k=retrieval_k,
        output_k=output_k,
        genre=preferences.get("genre") if isinstance(preferences, dict) else None,
        mood=preferences.get("mood") if isinstance(preferences, dict) else None,
        energy=preferences.get("energy") if isinstance(preferences, dict) else None,
    )

    try:
        _validate_inputs(user_request, preferences, songs, retrieval_k, output_k)
    except ValueError as exc:
        # Our own, non-sensitive validation messages only.
        log_event(logger, EVENT_PIPELINE_INPUT_ERROR, request_id=request_id, detail=str(exc))
        raise

    # b. Deterministic retrieval — single source of truth for scoring.
    ranked = recommend_songs(preferences, songs, k=retrieval_k)
    retrieved = [song for song, _score, _explanation in ranked]
    retrieved_ids = [song["id"] for song in retrieved]

    log_event(
        logger, EVENT_RETRIEVAL_COMPLETED,
        request_id=request_id,
        retrieved_count=len(retrieved),
        retrieved_ids=retrieved_ids,
        retrieval_k=retrieval_k,
    )

    def _finish_fallback(reason: str, model=None, report=None) -> DiscoveryResult:
        log_event(logger, EVENT_FALLBACK_USED, request_id=request_id, reason=reason)
        obj = _build_fallback_result(
            ranked, retrieved, output_k, reason,
            model=model, report=report, request_id=request_id,
        )
        log_event(
            logger, EVENT_PIPELINE_COMPLETED,
            request_id=request_id, source=obj.source, used_fallback=True,
            output_count=len(obj.final_recommendations),
            fallback_reason=reason, model=model,
        )
        return obj

    def _finish_ai(report: ValidationReport, model) -> DiscoveryResult:
        obj = _build_ai_result(report, ranked, retrieved, output_k, model, request_id)
        log_event(
            logger, EVENT_PIPELINE_COMPLETED,
            request_id=request_id, source=obj.source, used_fallback=False,
            output_count=len(obj.final_recommendations),
            reliability_score=report.reliability_score, model=model,
        )
        return obj

    # Safe handling when there is nothing to ground on.
    if not retrieved:
        return _finish_fallback(FALLBACK_NO_CANDIDATES)

    # c. Generation on retrieved candidates only. Any AI failure -> fallback.
    log_event(
        logger, EVENT_AI_GENERATION_STARTED,
        request_id=request_id, candidate_count=len(retrieved),
    )
    try:
        result = generation_function(user_request, retrieved)
    except MissingAPIKeyError:
        return _finish_fallback(FALLBACK_MISSING_API_KEY)
    except APICallError:
        return _finish_fallback(FALLBACK_API_ERROR)
    except EmptyResponseError:
        return _finish_fallback(FALLBACK_EMPTY_RESPONSE)
    except MalformedResponseError:
        return _finish_fallback(FALLBACK_MALFORMED_JSON)
    except InvalidResponseStructureError:
        return _finish_fallback(FALLBACK_INVALID_STRUCTURE)
    except AIClientError:
        return _finish_fallback(FALLBACK_AI_ERROR)

    log_event(
        logger, EVENT_AI_GENERATION_COMPLETED,
        request_id=request_id,
        recommendation_count=len(result.recommendations),
        model=result.model,
    )

    # d. Validate against the retrieved set.
    report = validate_recommendations(result, retrieved)
    log_event(
        logger, EVENT_VALIDATION_COMPLETED,
        request_id=request_id,
        total_requested=report.total_requested,
        valid_count=report.valid_count,
        rejected_count=len(report.rejected),
        reliability_score=report.reliability_score,
        passed=report.passed,
    )

    # e. Use AI output only if the complete response passed.
    if report.passed:
        return _finish_ai(report, result.model)

    # f. Otherwise, safe deterministic fallback (no second AI call).
    return _finish_fallback(FALLBACK_VALIDATION_FAILED, model=result.model, report=report)


def discover_from_request(
    user_request: str,
    songs: List[Dict[str, Any]],
    *,
    retrieval_k: int = 5,
    output_k: int = 3,
    parser_function: ParserFn = parse_preferences,
    generation_function: GenerationFn = generate_recommendations,
    logger: Any = None,
    request_id: Optional[str] = None,
) -> RequestDiscoveryResult:
    """
    Natural-language entry point.

    Parses `user_request` into canonical preferences, then delegates to the
    existing `discover_music()` pipeline (no retrieval/generation/validation/
    fallback logic is duplicated here). A single request_id correlates the
    parse event with the discovery events.
    """
    logger = logger or get_logger()
    request_id = request_id or _new_request_id()

    parsed = parser_function(user_request)
    log_event(
        logger, EVENT_PREFERENCES_PARSED,
        request_id=request_id,
        genre=parsed.genre, mood=parsed.mood, energy=parsed.energy,
        used_defaults=parsed.used_defaults,
        matched_terms_count=len(parsed.matched_terms),
    )

    discovery = discover_music(
        user_request,
        parsed.to_prefs(),
        songs,
        retrieval_k=retrieval_k,
        output_k=output_k,
        generation_function=generation_function,
        logger=logger,
        request_id=request_id,
    )
    return RequestDiscoveryResult(
        parsed_preferences=parsed, discovery=discovery, request_id=request_id
    )
