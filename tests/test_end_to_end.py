"""
End-to-end tests: natural-language request through the complete system, using
the real parser / recommender / reliability / pipeline and the real catalog.
Only the AI generator is mocked, so no network or Anthropic call ever occurs.
"""

import logging

from unittest.mock import MagicMock

from src.ai_client import (
    APICallError,
    MissingAPIKeyError,
    RecommendationResult,
    SongRecommendation,
)
from src.app import get_catalog_path
from src.pipeline import (
    FALLBACK_API_ERROR,
    FALLBACK_MISSING_API_KEY,
    FALLBACK_VALIDATION_FAILED,
    discover_from_request,
)
from src.recommender import load_songs

# Real catalog, loaded once through the real load_songs().
CATALOG = load_songs(str(get_catalog_path()))


class _ListHandler(logging.Handler):
    def __init__(self):
        super().__init__()
        self.messages = []

    def emit(self, record):
        self.messages.append(record.getMessage())


def capturing_logger():
    logger = logging.getLogger("e2e-" + str(id(object())))
    logger.handlers = []
    h = _ListHandler()
    logger.addHandler(h)
    logger.setLevel(logging.DEBUG)
    logger.propagate = False
    return logger, h


def echo_first_valid(request, retrieved):
    """A grounded generator: recommend the first retrieved song, faithfully."""
    s = retrieved[0]
    return RecommendationResult(
        recommendations=[SongRecommendation(s["id"], s["title"], s["artist"], "AI explanation.")],
        model="claude-test-model",
    )


# --- Full traversal --------------------------------------------------------

def test_natural_language_request_travels_through_system():
    gen = MagicMock(side_effect=echo_first_valid)
    result = discover_from_request("upbeat pop music for the gym", CATALOG,
                                   retrieval_k=5, output_k=3, generation_function=gen)
    assert result.parsed_preferences.genre == "pop"
    assert result.discovery.source == "ai"
    assert result.discovery.final_recommendations
    gen.assert_called_once()


def test_parsed_preferences_affect_retrieval():
    gen = MagicMock(side_effect=echo_first_valid)
    pop = discover_from_request("happy pop songs", CATALOG, generation_function=gen)
    gen2 = MagicMock(side_effect=echo_first_valid)
    lofi = discover_from_request("calm lofi for studying", CATALOG, generation_function=gen2)

    pop_ids = [s["id"] for s in pop.discovery.retrieved_candidates]
    lofi_ids = [s["id"] for s in lofi.discovery.retrieved_candidates]
    # Different preferences retrieve different candidate sets/orderings.
    assert pop_ids != lofi_ids
    # Top candidate genre matches the parsed genre for each clear request.
    assert pop.discovery.retrieved_candidates[0]["genre"] == "pop"
    assert lofi.discovery.retrieved_candidates[0]["genre"] == "lofi"


def test_only_retrieved_candidates_reach_generator():
    gen = MagicMock(side_effect=echo_first_valid)
    discover_from_request("intense rock for a workout", CATALOG, retrieval_k=4, generation_function=gen)
    _req, passed = gen.call_args.args
    assert len(passed) == 4
    assert len(passed) < len(CATALOG)
    passed_ids = {s["id"] for s in passed}
    assert passed_ids.issubset({s["id"] for s in CATALOG})


def test_valid_grounded_ai_output_reaches_final():
    gen = MagicMock(side_effect=echo_first_valid)
    result = discover_from_request("upbeat pop", CATALOG, generation_function=gen)
    recs = result.discovery.final_recommendations
    assert recs and all(r.source == "ai" for r in recs)
    assert result.discovery.validation_report.passed is True


# --- Fallbacks -------------------------------------------------------------

def test_invented_ai_output_causes_fallback():
    def invented(request, retrieved):
        return RecommendationResult(
            recommendations=[SongRecommendation(9999, "Ghost", "Nobody", "made up")],
            model="claude-test-model",
        )
    result = discover_from_request("pop", CATALOG, generation_function=MagicMock(side_effect=invented))
    assert result.discovery.source == "fallback"
    assert result.discovery.fallback_reason == FALLBACK_VALIDATION_FAILED
    assert all(r.song_id != 9999 for r in result.discovery.final_recommendations)


def test_missing_api_key_causes_fallback():
    gen = MagicMock(side_effect=MissingAPIKeyError("no key"))
    result = discover_from_request("pop", CATALOG, generation_function=gen)
    assert result.discovery.source == "fallback"
    assert result.discovery.fallback_reason == FALLBACK_MISSING_API_KEY


def test_api_failure_causes_fallback():
    gen = MagicMock(side_effect=APICallError("boom"))
    result = discover_from_request("pop", CATALOG, generation_function=gen)
    assert result.discovery.source == "fallback"
    assert result.discovery.fallback_reason == FALLBACK_API_ERROR


def test_fallback_recs_have_scores_and_explanations():
    gen = MagicMock(side_effect=APICallError("boom"))
    result = discover_from_request("pop", CATALOG, output_k=3, generation_function=gen)
    recs = result.discovery.final_recommendations
    assert len(recs) == 3
    for r in recs:
        assert r.source == "fallback"
        assert r.score is not None
        assert isinstance(r.explanation, str) and r.explanation.strip() != ""


def test_no_rejected_ai_rec_reaches_final():
    def mixed(request, retrieved):
        good = retrieved[0]
        return RecommendationResult(
            recommendations=[
                SongRecommendation(good["id"], good["title"], good["artist"], "ok"),
                SongRecommendation(9999, "Ghost", "Nobody", "invented"),
            ],
            model="claude-test-model",
        )
    result = discover_from_request("pop", CATALOG, generation_function=MagicMock(side_effect=mixed))
    assert result.discovery.source == "fallback"
    assert all(r.song_id != 9999 for r in result.discovery.final_recommendations)


# --- Logging & correlation -------------------------------------------------

def test_logging_records_major_events():
    logger, handler = capturing_logger()
    gen = MagicMock(side_effect=echo_first_valid)
    discover_from_request("upbeat pop", CATALOG, generation_function=gen, logger=logger)
    events = {m.split(" ", 1)[0].split("=", 1)[1] for m in handler.messages}
    for expected in {"preferences_parsed", "pipeline_started", "retrieval_completed",
                     "ai_generation_started", "ai_generation_completed",
                     "validation_completed", "pipeline_completed"}:
        assert expected in events


def test_request_ids_remain_consistent():
    logger, handler = capturing_logger()
    gen = MagicMock(side_effect=echo_first_valid)
    result = discover_from_request("upbeat pop", CATALOG, generation_function=gen,
                                   logger=logger, request_id="e2e-rid")
    assert result.request_id == "e2e-rid"
    assert result.discovery.request_id == "e2e-rid"
    assert handler.messages
    for m in handler.messages:
        assert "request_id=e2e-rid" in m


# --- No side effects -------------------------------------------------------

def test_no_network_end_to_end(monkeypatch):
    import socket
    monkeypatch.setattr(socket, "create_connection",
                        lambda *a, **k: (_ for _ in ()).throw(AssertionError("no network")))
    monkeypatch.setattr(socket.socket, "connect",
                        lambda *a, **k: (_ for _ in ()).throw(AssertionError("no network")))
    gen = MagicMock(side_effect=echo_first_valid)
    result = discover_from_request("upbeat pop", CATALOG, generation_function=gen)
    assert result.discovery.source == "ai"
