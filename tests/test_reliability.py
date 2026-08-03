"""
Tests for src.reliability — pure, offline validation of AI recommendations.

No file, network, or Anthropic access is required or performed.
"""

from src.ai_client import RecommendationResult, SongRecommendation
from src.reliability import (
    REASON_ARTIST_MISMATCH,
    REASON_DUPLICATE,
    REASON_TITLE_MISMATCH,
    REASON_UNKNOWN_ID,
    ValidationReport,
    validate_recommendations,
)

RETRIEVED = [
    {"id": 1, "title": "Sunrise City", "artist": "Neon Echo", "genre": "pop"},
    {"id": 3, "title": "Storm Runner", "artist": "Voltline", "genre": "rock"},
]


def rec(song_id, title, artist, explanation="because"):
    return SongRecommendation(song_id=song_id, title=title, artist=artist, explanation=explanation)


def result(*recs):
    return RecommendationResult(recommendations=list(recs), model="claude-test-model")


# --- Core validity ---------------------------------------------------------

def test_all_valid():
    report = validate_recommendations(
        result(rec(1, "Sunrise City", "Neon Echo"), rec(3, "Storm Runner", "Voltline")),
        RETRIEVED,
    )
    assert isinstance(report, ValidationReport)
    assert report.total_requested == 2
    assert report.valid_count == 2
    assert report.rejected == []
    assert report.passed is True
    assert report.reliability_score == 1.0
    assert [r.song_id for r in report.valid] == [1, 3]


def test_unknown_song_rejected():
    report = validate_recommendations(
        result(rec(99, "Invented Track", "Ghost Band")), RETRIEVED
    )
    assert report.valid_count == 0
    assert report.rejected[0].reason == REASON_UNKNOWN_ID
    assert report.passed is False
    assert report.reliability_score == 0.0


def test_title_mismatch_rejected():
    report = validate_recommendations(
        result(rec(1, "Wrong Title", "Neon Echo")), RETRIEVED
    )
    assert report.valid_count == 0
    assert report.rejected[0].reason == REASON_TITLE_MISMATCH


def test_artist_mismatch_rejected():
    report = validate_recommendations(
        result(rec(1, "Sunrise City", "Wrong Artist")), RETRIEVED
    )
    assert report.valid_count == 0
    assert report.rejected[0].reason == REASON_ARTIST_MISMATCH


def test_mixed_valid_and_invalid():
    report = validate_recommendations(
        result(
            rec(1, "Sunrise City", "Neon Echo"),   # valid
            rec(99, "Nope", "Nobody"),              # unknown id
            rec(3, "Wrong", "Voltline"),            # title mismatch
        ),
        RETRIEVED,
    )
    assert report.total_requested == 3
    assert report.valid_count == 1
    assert [r.song_id for r in report.valid] == [1]
    reasons = sorted(r.reason for r in report.rejected)
    assert reasons == sorted([REASON_UNKNOWN_ID, REASON_TITLE_MISMATCH])
    assert report.passed is False
    assert report.reliability_score == 1 / 3
    # Rejected recommendations never leak into valid.
    valid_ids = {r.song_id for r in report.valid}
    assert 99 not in valid_ids and all(rr.recommendation.song_id not in valid_ids for rr in report.rejected)


# --- Edge cases ------------------------------------------------------------

def test_empty_ai_recommendations():
    report = validate_recommendations(result(), RETRIEVED)
    assert report.total_requested == 0
    assert report.valid_count == 0
    assert report.valid == []
    assert report.passed is False
    assert report.reliability_score == 0.0


def test_empty_retrieved_songs():
    report = validate_recommendations(result(rec(1, "Sunrise City", "Neon Echo")), [])
    assert report.valid_count == 0
    assert report.rejected[0].reason == REASON_UNKNOWN_ID
    assert report.reliability_score == 0.0


def test_duplicate_recommendation_rejected():
    report = validate_recommendations(
        result(
            rec(1, "Sunrise City", "Neon Echo"),  # first -> valid
            rec(1, "Sunrise City", "Neon Echo"),  # duplicate -> rejected
        ),
        RETRIEVED,
    )
    assert report.total_requested == 2
    assert report.valid_count == 1
    assert len(report.rejected) == 1
    assert report.rejected[0].reason == REASON_DUPLICATE
    assert report.reliability_score == 0.5


def test_duplicate_of_never_valid_id_is_not_duplicate():
    """Two bad recs for the same unknown id are both unknown_id, not duplicate."""
    report = validate_recommendations(
        result(rec(99, "A", "B"), rec(99, "A", "B")), RETRIEVED
    )
    assert report.valid_count == 0
    assert [r.reason for r in report.rejected] == [REASON_UNKNOWN_ID, REASON_UNKNOWN_ID]


def test_string_id_matches_numeric_retrieved_id():
    """Model echoing the id as a string still matches a numeric retrieved id."""
    report = validate_recommendations(
        result(rec("1", "Sunrise City", "Neon Echo")), RETRIEVED
    )
    assert report.valid_count == 1
    assert report.passed is True


# --- Score properties ------------------------------------------------------

def test_score_equals_documented_formula():
    report = validate_recommendations(
        result(
            rec(1, "Sunrise City", "Neon Echo"),  # valid
            rec(3, "Storm Runner", "Voltline"),   # valid
            rec(99, "X", "Y"),                     # invalid
            rec(2, "Z", "W"),                      # invalid (unknown)
        ),
        RETRIEVED,
    )
    assert report.reliability_score == report.valid_count / report.total_requested
    assert report.reliability_score == 2 / 4


def test_score_within_unit_interval():
    cases = [
        result(),
        result(rec(1, "Sunrise City", "Neon Echo")),
        result(rec(99, "X", "Y")),
        result(rec(1, "Sunrise City", "Neon Echo"), rec(1, "Sunrise City", "Neon Echo")),
    ]
    for r in cases:
        report = validate_recommendations(r, RETRIEVED)
        assert 0.0 <= report.reliability_score <= 1.0


# --- No side effects -------------------------------------------------------

def test_no_file_network_or_api_access(monkeypatch):
    """Validation must not open files, hit the network, or build an AI client."""
    import builtins
    import socket

    real_open = builtins.open

    def open_guard(path, *args, **kwargs):
        assert "songs.csv" not in str(path), "reliability must not read the catalog"
        return real_open(path, *args, **kwargs)

    def net_guard(*args, **kwargs):
        raise AssertionError("reliability must not touch the network")

    monkeypatch.setattr(builtins, "open", open_guard)
    monkeypatch.setattr(socket, "create_connection", net_guard)
    monkeypatch.setattr(socket.socket, "connect", net_guard)

    report = validate_recommendations(
        result(rec(1, "Sunrise City", "Neon Echo"), rec(99, "X", "Y")),
        RETRIEVED,
    )
    assert report.valid_count == 1
    assert report.reliability_score == 0.5
