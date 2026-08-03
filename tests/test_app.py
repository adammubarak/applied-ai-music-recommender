"""
Tests for src.app — the Streamlit UI helpers and a mocked smoke test.

All offline: no real .env, no API key, no network, no Anthropic call.
"""

from pathlib import Path
from unittest.mock import MagicMock

import pytest

import src.app as app
from src.ai_client import RecommendationResult, SongRecommendation
from src.pipeline import (
    FALLBACK_API_ERROR,
    FALLBACK_MISSING_API_KEY,
    FALLBACK_NO_CANDIDATES,
    FALLBACK_VALIDATION_FAILED,
    DiscoveryResult,
    RequestDiscoveryResult,
)
from src.preference_parser import ParsedPreferences


# --- Import safety & path resolution ---------------------------------------

def test_app_imports_safely():
    assert hasattr(app, "main")
    assert callable(app.main)


def test_catalog_path_resolves_to_data_songs_csv():
    path = app.get_catalog_path()
    assert isinstance(path, Path)
    assert path.name == "songs.csv"
    assert path.parent.name == "data"
    assert path.exists()  # the real catalog is present in the repo


def test_load_catalog_uses_load_songs(monkeypatch):
    called = {}

    def fake_load_songs(p):
        called["path"] = p
        return [{"id": 1, "title": "T", "artist": "A"}]

    monkeypatch.setattr(app, "load_songs", fake_load_songs)
    songs = app.load_catalog()
    assert songs == [{"id": 1, "title": "T", "artist": "A"}]
    assert called["path"].endswith("data/songs.csv") or called["path"].endswith("data\\songs.csv")


def test_load_catalog_really_parses_repo_catalog():
    songs = app.load_catalog()
    assert len(songs) == 18
    assert all({"id", "title", "artist"}.issubset(s) for s in songs)


# --- Delegation to the pipeline --------------------------------------------

SONGS = [
    {"id": 1, "title": "Sunrise City", "artist": "Neon Echo", "genre": "pop", "mood": "happy", "energy": 0.82, "tempo_bpm": 118, "valence": 0.84, "danceability": 0.79, "acousticness": 0.18},
    {"id": 2, "title": "Storm Runner", "artist": "Voltline", "genre": "rock", "mood": "intense", "energy": 0.91, "tempo_bpm": 152, "valence": 0.48, "danceability": 0.66, "acousticness": 0.10},
    {"id": 3, "title": "Library Rain", "artist": "Paper Lanterns", "genre": "lofi", "mood": "chill", "energy": 0.35, "tempo_bpm": 72, "valence": 0.6, "danceability": 0.58, "acousticness": 0.86},
]


def valid_gen(request, retrieved):
    s = retrieved[0]
    return RecommendationResult(
        recommendations=[SongRecommendation(s["id"], s["title"], s["artist"], "AI explanation.")],
        model="claude-test-model",
    )


def test_run_discovery_delegates_to_pipeline(monkeypatch):
    spy = MagicMock(return_value="SENTINEL")
    monkeypatch.setattr(app, "discover_from_request", spy)

    gen = MagicMock(side_effect=valid_gen)
    out = app.run_discovery("upbeat pop", SONGS, retrieval_k=4, output_k=2,
                            logger=None, generation_function=gen)

    assert out == "SENTINEL"
    spy.assert_called_once()
    assert spy.call_args.args[0] == "upbeat pop"
    assert spy.call_args.kwargs["retrieval_k"] == 4
    assert spy.call_args.kwargs["output_k"] == 2
    assert spy.call_args.kwargs["generation_function"] is gen


def test_run_discovery_end_to_end_with_injected_generator():
    gen = MagicMock(side_effect=valid_gen)
    result = app.run_discovery("upbeat pop", SONGS, generation_function=gen)
    assert isinstance(result, RequestDiscoveryResult)
    assert result.discovery.source == "ai"
    gen.assert_called_once()


# --- Fallback message mapping ----------------------------------------------

def test_stable_fallback_codes_map_to_safe_messages():
    for code in (FALLBACK_MISSING_API_KEY, FALLBACK_API_ERROR,
                 FALLBACK_VALIDATION_FAILED, FALLBACK_NO_CANDIDATES):
        msg = app.humanize_fallback_reason(code)
        assert isinstance(msg, str) and msg
        assert code not in msg           # never surface the raw code
        assert "sk-" not in msg


def test_unknown_fallback_code_does_not_expose_raw_data():
    weird = "sk-ant-secret-or-raw-exception-text"
    msg = app.humanize_fallback_reason(weird)
    assert msg == app.DEFAULT_FALLBACK_MESSAGE
    assert weird not in msg


def test_none_fallback_reason_is_safe():
    assert app.humanize_fallback_reason(None) == app.DEFAULT_FALLBACK_MESSAGE


# --- Safe handling ---------------------------------------------------------

def test_missing_catalog_handled(monkeypatch):
    def boom(p):
        raise FileNotFoundError(p)
    monkeypatch.setattr(app, "load_songs", boom)
    with pytest.raises(FileNotFoundError):
        app.load_catalog()  # helper raises; main() catches and shows a safe message


def test_empty_results_render_safely():
    """A discovery with no recommendations renders without error via a fake st."""
    parsed = ParsedPreferences(genre="pop", mood="happy", energy=0.5, matched_terms=(), used_defaults=True)
    discovery = DiscoveryResult(final_recommendations=[], retrieved_candidates=[],
                                source="fallback", used_fallback=True,
                                fallback_reason=FALLBACK_NO_CANDIDATES, request_id="rid")
    result = RequestDiscoveryResult(parsed_preferences=parsed, discovery=discovery, request_id="rid")
    app._render_result(_FakeSt(), result)  # must not raise


def test_helper_output_contains_no_secret_or_exception_text():
    """humanize + a rendered result must not leak secrets/raw exceptions."""
    fake = _FakeSt()
    parsed = ParsedPreferences(genre="rock", mood="intense", energy=0.9, matched_terms=("rock",), used_defaults=False)
    discovery = DiscoveryResult(
        final_recommendations=[], retrieved_candidates=SONGS,
        source="fallback", used_fallback=True,
        fallback_reason=FALLBACK_API_ERROR, model="claude-test-model", request_id="rid-xyz",
    )
    app._render_result(fake, RequestDiscoveryResult(parsed, discovery, "rid-xyz"))
    blob = fake.text_blob()
    assert "sk-" not in blob
    assert "Traceback" not in blob
    assert "ANTHROPIC_API_KEY" not in blob


# --- Fake Streamlit for _render_result -------------------------------------

class _FakeCol:
    def metric(self, *a, **k): pass


class _FakeExpander:
    def __enter__(self): return self
    def __exit__(self, *a): return False
    def __getattr__(self, _):  # any st.* call inside expander is a no-op recorder
        return lambda *a, **k: None


class _FakeSt:
    def __init__(self):
        self._texts = []

    def _rec(self, *a, **k):
        for x in a:
            if isinstance(x, str):
                self._texts.append(x)

    subheader = write = markdown = caption = info = warning = error = success = _rec

    def columns(self, n): return [_FakeCol() for _ in range(n)]
    def table(self, *a, **k): pass
    def expander(self, *a, **k): return _FakeExpander()

    def text_blob(self): return "\n".join(self._texts)


# --- Streamlit AppTest smoke test ------------------------------------------

def test_streamlit_smoke_apptest():
    """Render the app with streamlit.testing.v1.AppTest; no API call is made."""
    from streamlit.testing.v1 import AppTest

    at = AppTest.from_file(str(Path(app.__file__).resolve()))
    at.run(timeout=30)

    assert not at.exception
    # Title visible.
    titles = [t.value for t in at.title]
    assert any("Music Discovery" in t for t in titles)
    # Request input exists.
    assert len(at.text_area) >= 1
    # Generate button exists.
    labels = [b.label for b in at.button]
    assert any("Generate" in lbl for lbl in labels)
    # Initial render performs no discovery (button not clicked) -> no AI path.
