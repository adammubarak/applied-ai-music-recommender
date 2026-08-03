"""
Tests for src.pipeline — end-to-end RAG flow with an injected, mocked AI
generation function. No real API or network access.
"""

from unittest.mock import MagicMock

import pytest

from src.ai_client import (
    APICallError,
    EmptyResponseError,
    InvalidResponseStructureError,
    MalformedResponseError,
    MissingAPIKeyError,
    RecommendationResult,
    SongRecommendation,
)
from src.pipeline import (
    FALLBACK_API_ERROR,
    FALLBACK_EMPTY_RESPONSE,
    FALLBACK_INVALID_STRUCTURE,
    FALLBACK_MALFORMED_JSON,
    FALLBACK_MISSING_API_KEY,
    FALLBACK_NO_CANDIDATES,
    FALLBACK_VALIDATION_FAILED,
    DiscoveryResult,
    RequestDiscoveryResult,
    discover_from_request,
    discover_music,
)
from src.preference_parser import ParsedPreferences

# A catalog larger than retrieval_k so we can prove the full catalog isn't sent.
SONGS = [
    {"id": 1, "title": "Sunrise City", "artist": "Neon Echo", "genre": "pop", "mood": "happy", "energy": 0.82, "tempo_bpm": 118, "valence": 0.84, "danceability": 0.79, "acousticness": 0.18},
    {"id": 2, "title": "Midnight Coding", "artist": "LoRoom", "genre": "lofi", "mood": "chill", "energy": 0.42, "tempo_bpm": 78, "valence": 0.56, "danceability": 0.62, "acousticness": 0.71},
    {"id": 3, "title": "Storm Runner", "artist": "Voltline", "genre": "rock", "mood": "intense", "energy": 0.91, "tempo_bpm": 152, "valence": 0.48, "danceability": 0.66, "acousticness": 0.10},
    {"id": 4, "title": "Gym Hero", "artist": "PulseWave", "genre": "pop", "mood": "energetic", "energy": 0.88, "tempo_bpm": 128, "valence": 0.7, "danceability": 0.8, "acousticness": 0.12},
    {"id": 5, "title": "Rooftop Lights", "artist": "Skyline", "genre": "indie", "mood": "happy", "energy": 0.6, "tempo_bpm": 110, "valence": 0.75, "danceability": 0.7, "acousticness": 0.3},
    {"id": 6, "title": "Library Rain", "artist": "Paper Lanterns", "genre": "lofi", "mood": "chill", "energy": 0.35, "tempo_bpm": 72, "valence": 0.6, "danceability": 0.58, "acousticness": 0.86},
]

PREFS = {"genre": "pop", "mood": "happy", "energy": 0.8}


def valid_gen_from_retrieved(n=1):
    """Return a generation fn that echoes the first n retrieved songs as valid recs."""
    def gen(request, retrieved):
        recs = [
            SongRecommendation(
                song_id=retrieved[i]["id"],
                title=retrieved[i]["title"],
                artist=retrieved[i]["artist"],
                explanation=f"AI says: matches your request '{request}'.",
            )
            for i in range(min(n, len(retrieved)))
        ]
        return RecommendationResult(recommendations=recs, model="claude-test-model")
    return MagicMock(side_effect=gen)


# --- Happy path ------------------------------------------------------------

def test_success_retrieval_generation_validation():
    gen = valid_gen_from_retrieved(1)
    result = discover_music("upbeat pop", PREFS, SONGS, retrieval_k=5, output_k=3, generation_function=gen)

    assert isinstance(result, DiscoveryResult)
    assert result.source == "ai"
    assert result.used_fallback is False
    assert result.fallback_reason is None
    assert result.model == "claude-test-model"
    assert len(result.final_recommendations) == 1
    assert result.final_recommendations[0].source == "ai"
    assert result.validation_report is not None and result.validation_report.passed
    gen.assert_called_once()


def test_ai_receives_only_retrieved_topk_not_full_catalog():
    gen = valid_gen_from_retrieved(1)
    discover_music("x", PREFS, SONGS, retrieval_k=3, output_k=3, generation_function=gen)

    passed_request, passed_retrieved = gen.call_args.args
    assert passed_request == "x"
    assert len(passed_retrieved) == 3                      # exactly retrieval_k
    assert len(passed_retrieved) < len(SONGS)              # not the full catalog
    passed_ids = {s["id"] for s in passed_retrieved}
    all_ids = {s["id"] for s in SONGS}
    assert passed_ids.issubset(all_ids)
    assert len(all_ids - passed_ids) == len(SONGS) - 3     # some catalog songs withheld


def test_valid_ai_output_becomes_final_output():
    gen = valid_gen_from_retrieved(2)
    result = discover_music("x", PREFS, SONGS, retrieval_k=5, output_k=5, generation_function=gen)
    assert result.source == "ai"
    assert all(r.source == "ai" for r in result.final_recommendations)
    assert len(result.final_recommendations) == 2


# --- Fallback on AI exceptions ---------------------------------------------

@pytest.mark.parametrize("exc, reason", [
    (MissingAPIKeyError("no key"), FALLBACK_MISSING_API_KEY),
    (APICallError("boom"), FALLBACK_API_ERROR),
    (EmptyResponseError("empty"), FALLBACK_EMPTY_RESPONSE),
    (MalformedResponseError("bad json"), FALLBACK_MALFORMED_JSON),
    (InvalidResponseStructureError("bad shape"), FALLBACK_INVALID_STRUCTURE),
])
def test_ai_exceptions_trigger_fallback(exc, reason):
    gen = MagicMock(side_effect=exc)
    result = discover_music("x", PREFS, SONGS, retrieval_k=4, output_k=3, generation_function=gen)

    assert result.source == "fallback"
    assert result.used_fallback is True
    assert result.fallback_reason == reason
    assert len(result.final_recommendations) == 3
    assert all(r.source == "fallback" for r in result.final_recommendations)
    gen.assert_called_once()  # no second AI call during fallback


# --- Fallback on validation failures ---------------------------------------

def test_invented_song_triggers_fallback():
    def gen(request, retrieved):
        return RecommendationResult(
            recommendations=[SongRecommendation(999, "Ghost Track", "Nobody", "made up")],
            model="claude-test-model",
        )
    result = discover_music("x", PREFS, SONGS, generation_function=MagicMock(side_effect=gen))
    assert result.source == "fallback"
    assert result.fallback_reason == FALLBACK_VALIDATION_FAILED
    # The invented song must never appear in the final output.
    assert all(r.song_id != 999 for r in result.final_recommendations)
    assert all(r.source == "fallback" for r in result.final_recommendations)


def test_mismatched_song_triggers_fallback():
    def gen(request, retrieved):
        s = retrieved[0]
        return RecommendationResult(
            recommendations=[SongRecommendation(s["id"], "WRONG TITLE", s["artist"], "x")],
            model="claude-test-model",
        )
    result = discover_music("x", PREFS, SONGS, generation_function=MagicMock(side_effect=gen))
    assert result.source == "fallback"
    assert result.fallback_reason == FALLBACK_VALIDATION_FAILED


def test_duplicate_recommendation_triggers_fallback():
    def gen(request, retrieved):
        s = retrieved[0]
        rec = SongRecommendation(s["id"], s["title"], s["artist"], "x")
        return RecommendationResult(recommendations=[rec, rec], model="claude-test-model")
    result = discover_music("x", PREFS, SONGS, generation_function=MagicMock(side_effect=gen))
    # One valid + one duplicate => not all valid => fallback.
    assert result.source == "fallback"
    assert result.fallback_reason == FALLBACK_VALIDATION_FAILED


def test_empty_ai_recommendations_trigger_fallback():
    gen = MagicMock(side_effect=lambda r, s: RecommendationResult(recommendations=[], model="claude-test-model"))
    result = discover_music("x", PREFS, SONGS, generation_function=gen)
    assert result.source == "fallback"
    assert result.fallback_reason == FALLBACK_VALIDATION_FAILED


# --- Edge cases ------------------------------------------------------------

def test_no_retrieved_songs_handled_safely():
    gen = valid_gen_from_retrieved(1)
    result = discover_music("x", PREFS, [], retrieval_k=5, output_k=3, generation_function=gen)
    assert result.source == "fallback"
    assert result.fallback_reason == FALLBACK_NO_CANDIDATES
    assert result.final_recommendations == []
    gen.assert_not_called()  # nothing to ground on -> no AI call


def test_output_k_respected_on_ai_path():
    gen = valid_gen_from_retrieved(5)
    result = discover_music("x", PREFS, SONGS, retrieval_k=5, output_k=2, generation_function=gen)
    assert result.source == "ai"
    assert len(result.final_recommendations) == 2


def test_output_k_respected_on_fallback_path():
    gen = MagicMock(side_effect=APICallError("boom"))
    result = discover_music("x", PREFS, SONGS, retrieval_k=5, output_k=2, generation_function=gen)
    assert result.source == "fallback"
    assert len(result.final_recommendations) == 2


def test_fallback_uses_deterministic_retrieval_results():
    from src.recommender import recommend_songs
    ranked = recommend_songs(PREFS, SONGS, k=5)
    expected_top = [(s["id"], score) for s, score, _ in ranked][:3]

    gen = MagicMock(side_effect=APICallError("boom"))
    result = discover_music("x", PREFS, SONGS, retrieval_k=5, output_k=3, generation_function=gen)

    got = [(r.song_id, r.score) for r in result.final_recommendations]
    assert got == expected_top  # exact deterministic scores & order preserved
    assert all(r.explanation and r.source == "fallback" for r in result.final_recommendations)


def test_rejected_ai_recs_never_reach_final_output():
    def gen(request, retrieved):
        good = retrieved[0]
        return RecommendationResult(
            recommendations=[
                SongRecommendation(good["id"], good["title"], good["artist"], "ok"),  # valid
                SongRecommendation(999, "Ghost", "Nobody", "invented"),               # rejected
            ],
            model="claude-test-model",
        )
    result = discover_music("x", PREFS, SONGS, generation_function=MagicMock(side_effect=gen))
    # Mixed => not all valid => fallback; the invented rec must not appear.
    assert result.source == "fallback"
    assert all(r.song_id != 999 for r in result.final_recommendations)


def test_fallback_reason_contains_no_secret():
    known_codes = {
        FALLBACK_MISSING_API_KEY, FALLBACK_API_ERROR, FALLBACK_EMPTY_RESPONSE,
        FALLBACK_MALFORMED_JSON, FALLBACK_INVALID_STRUCTURE, FALLBACK_VALIDATION_FAILED,
        FALLBACK_NO_CANDIDATES,
    }
    gen = MagicMock(side_effect=APICallError("sk-ant-should-not-leak-into-reason"))
    result = discover_music("x", PREFS, SONGS, generation_function=gen)
    assert result.fallback_reason in known_codes
    assert "sk-" not in (result.fallback_reason or "")


# --- No side effects -------------------------------------------------------

def test_no_network_during_pipeline(monkeypatch):
    import socket

    def net_guard(*a, **k):
        raise AssertionError("pipeline must not touch the network")

    monkeypatch.setattr(socket, "create_connection", net_guard)
    monkeypatch.setattr(socket.socket, "connect", net_guard)

    gen = valid_gen_from_retrieved(1)
    result = discover_music("x", PREFS, SONGS, generation_function=gen)
    assert result.source == "ai"


def test_invalid_inputs_raise():
    with pytest.raises(ValueError):
        discover_music("x", {"mood": "happy"}, SONGS, generation_function=valid_gen_from_retrieved(1))  # missing keys
    with pytest.raises(ValueError):
        discover_music("x", PREFS, SONGS, retrieval_k=0, generation_function=valid_gen_from_retrieved(1))


# --- discover_from_request (natural-language entry point) -------------------

def fixed_parser(genre="pop", mood="happy", energy=0.8):
    parsed = ParsedPreferences(genre=genre, mood=mood, energy=energy,
                               matched_terms=("pop", "happy"), used_defaults=False)
    return MagicMock(side_effect=lambda req: parsed), parsed


def test_request_entry_point_returns_parse_and_discovery():
    parser, parsed = fixed_parser()
    gen = valid_gen_from_retrieved(1)
    result = discover_from_request(
        "upbeat pop for the gym", SONGS, retrieval_k=5, output_k=3,
        parser_function=parser, generation_function=gen,
    )
    assert isinstance(result, RequestDiscoveryResult)
    assert result.parsed_preferences is parsed          # parse included in result
    assert isinstance(result.discovery, DiscoveryResult)
    assert result.discovery.source == "ai"
    parser.assert_called_once_with("upbeat pop for the gym")


def test_request_entry_point_passes_parsed_prefs_into_pipeline(monkeypatch):
    """Delegation: discover_music is called with the parsed canonical prefs."""
    import src.pipeline as pipeline

    parser, parsed = fixed_parser(genre="rock", mood="intense", energy=0.9)
    spy = MagicMock(return_value=DiscoveryResult(source="ai", used_fallback=False))
    monkeypatch.setattr(pipeline, "discover_music", spy)

    gen = valid_gen_from_retrieved(1)
    result = discover_from_request("x", SONGS, retrieval_k=4, output_k=2,
                                   parser_function=parser, generation_function=gen)

    # Called exactly once — retrieval/gen/validation/fallback not re-implemented.
    spy.assert_called_once()
    call = spy.call_args
    assert call.args[1] == {"genre": "rock", "mood": "intense", "energy": 0.9}
    assert call.kwargs["retrieval_k"] == 4
    assert call.kwargs["output_k"] == 2
    assert call.kwargs["generation_function"] is gen
    assert result.discovery is spy.return_value


def test_request_entry_point_only_retrieved_songs_reach_generator():
    parser, _ = fixed_parser()
    gen = valid_gen_from_retrieved(1)
    discover_from_request("x", SONGS, retrieval_k=3, parser_function=parser, generation_function=gen)

    _req, passed_retrieved = gen.call_args.args
    assert len(passed_retrieved) == 3
    assert len(passed_retrieved) < len(SONGS)


def test_request_entry_point_default_parser_is_offline(monkeypatch):
    """With the real parser and a mocked generator: fully offline, AI path used."""
    import socket
    monkeypatch.setattr(socket, "create_connection",
                        lambda *a, **k: (_ for _ in ()).throw(AssertionError("no network")))
    gen = valid_gen_from_retrieved(1)
    result = discover_from_request("upbeat pop for the gym", SONGS, generation_function=gen)
    assert isinstance(result.parsed_preferences, ParsedPreferences)
    assert result.discovery.source == "ai"
