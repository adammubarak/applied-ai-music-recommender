"""
Reliability and guardrail layer for the Music Discovery Assistant.

Responsibility (Phase 3 only)
-----------------------------
Validate Claude's recommendations against the exact set of songs that was
retrieved and supplied to Claude, and compute a transparent, deterministic
reliability score. This is the trust boundary: a song the model names that
does not faithfully match a retrieved candidate is rejected.

This module is pure and offline. It does NOT:
- call Anthropic or construct any AI client,
- perform retrieval or read data/songs.csv (or any file),
- create fallback recommendations,
- log anything,
- depend on Streamlit.

Validation rules
----------------
A recommendation is VALID only when all hold:
  1. its song_id exists in the retrieved set,
  2. its title exactly matches the retrieved song with that ID,
  3. its artist exactly matches the retrieved song with that ID.

IDs are compared by string-normalizing both sides (so an id of 1 and "1" match),
because the model may echo the id as a number or a string. Title and artist are
compared with exact string equality — no trimming, no case-folding.

Duplicate policy
----------------
The FIRST valid occurrence of a song ID is accepted; any LATER recommendation
with an already-accepted ID is rejected with reason "duplicate". Duplicate
detection is measured against IDs that were actually accepted — repeated
recommendations for an ID that never validated get their own specific rejection
reason (e.g. "unknown_id"), not "duplicate".

Per-recommendation precedence: unknown_id → duplicate → title_mismatch →
artist_mismatch → valid.

Reliability score
-----------------
    reliability_score = valid_recommendation_count / total_recommendation_count

- If there are no AI recommendations, the score is 0.0.
- The result is clamped to [0.0, 1.0].
- A confidence value supplied by Claude is never used.
"""

from dataclasses import dataclass, field
from typing import Any, Dict, List

from .ai_client import RecommendationResult, SongRecommendation


# --- Rejection reasons (stable string constants) ---------------------------

REASON_UNKNOWN_ID = "unknown_id"
REASON_TITLE_MISMATCH = "title_mismatch"
REASON_ARTIST_MISMATCH = "artist_mismatch"
REASON_DUPLICATE = "duplicate"


# --- Typed results ---------------------------------------------------------

@dataclass(frozen=True)
class RejectedRecommendation:
    """A recommendation that failed validation, with the reason it failed."""
    recommendation: SongRecommendation
    reason: str


@dataclass(frozen=True)
class ValidationReport:
    """
    The outcome of validating a RecommendationResult against retrieved songs.

    Attributes:
        valid: Recommendations that passed all checks (never contains rejects).
        rejected: Recommendations that failed, each with a reason.
        total_requested: Number of recommendations the model produced.
        valid_count: Number of valid recommendations.
        passed: True only if there was at least one recommendation and every
            recommendation was valid (valid_count == total_requested > 0).
        reliability_score: valid_count / total_requested, clamped to [0, 1];
            0.0 when there are no recommendations.
    """
    valid: List[SongRecommendation] = field(default_factory=list)
    rejected: List[RejectedRecommendation] = field(default_factory=list)
    total_requested: int = 0
    valid_count: int = 0
    passed: bool = False
    reliability_score: float = 0.0


# --- Helpers ---------------------------------------------------------------

def _index_by_id(retrieved_songs: List[Dict[str, Any]]) -> Dict[str, Dict[str, Any]]:
    """Index retrieved songs by string-normalized id for lookup."""
    index: Dict[str, Dict[str, Any]] = {}
    for song in retrieved_songs:
        if "id" in song:
            index[str(song["id"])] = song
    return index


def _clamp(value: float, low: float = 0.0, high: float = 1.0) -> float:
    """Clamp a float into [low, high]."""
    return max(low, min(high, value))


# --- Public interface ------------------------------------------------------

def validate_recommendations(
    result: RecommendationResult,
    retrieved_songs: List[Dict[str, Any]],
) -> ValidationReport:
    """
    Validate every AI recommendation against the retrieved song set.

    Args:
        result: The RecommendationResult produced by src.ai_client.
        retrieved_songs: The exact list of retrieved song dicts supplied to
            Claude. Not modified, not reloaded.

    Returns:
        A ValidationReport. Rejected recommendations never appear in `valid`.
    """
    recommendations = list(result.recommendations)
    by_id = _index_by_id(retrieved_songs)

    valid: List[SongRecommendation] = []
    rejected: List[RejectedRecommendation] = []
    accepted_ids: set = set()

    for rec in recommendations:
        key = str(rec.song_id)
        song = by_id.get(key)

        if song is None:
            rejected.append(RejectedRecommendation(rec, REASON_UNKNOWN_ID))
            continue
        if key in accepted_ids:
            rejected.append(RejectedRecommendation(rec, REASON_DUPLICATE))
            continue
        if rec.title != song["title"]:
            rejected.append(RejectedRecommendation(rec, REASON_TITLE_MISMATCH))
            continue
        if rec.artist != song["artist"]:
            rejected.append(RejectedRecommendation(rec, REASON_ARTIST_MISMATCH))
            continue

        valid.append(rec)
        accepted_ids.add(key)

    total_requested = len(recommendations)
    valid_count = len(valid)

    # Transparent, deterministic formula. 0.0 when there are no recommendations.
    if total_requested == 0:
        reliability_score = 0.0
    else:
        reliability_score = _clamp(valid_count / total_requested)

    passed = total_requested > 0 and valid_count == total_requested

    return ValidationReport(
        valid=valid,
        rejected=rejected,
        total_requested=total_requested,
        valid_count=valid_count,
        passed=passed,
        reliability_score=reliability_score,
    )
