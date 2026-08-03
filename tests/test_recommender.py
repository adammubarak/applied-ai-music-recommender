"""
Tests for the functional recommendation core and the OOP compatibility wrapper.

These exercise the real scoring/ranking logic (score_song, recommend_songs,
load_songs) and verify that the Recommender class delegates to that core rather
than reimplementing it.
"""

import os

from src.recommender import (
    Song,
    UserProfile,
    Recommender,
    load_songs,
    score_song,
    recommend_songs,
)

DATA_PATH = os.path.join(os.path.dirname(__file__), "..", "data", "songs.csv")


def make_song(**overrides) -> dict:
    """Build a canonical song dict with sensible defaults, overridable per test."""
    base = {
        "id": 1,
        "title": "Test Track",
        "artist": "Test Artist",
        "genre": "pop",
        "mood": "happy",
        "energy": 0.8,
        "tempo_bpm": 120.0,
        "valence": 0.9,
        "danceability": 0.8,
        "acousticness": 0.2,
    }
    base.update(overrides)
    return base


PREFS = {"genre": "pop", "mood": "happy", "energy": 0.8}


# --- load_songs ------------------------------------------------------------

def test_load_songs_reads_full_catalog():
    songs = load_songs(DATA_PATH)
    assert len(songs) == 18
    first = songs[0]
    # Correct types and keys after parsing
    assert isinstance(first["id"], int)
    assert isinstance(first["energy"], float)
    assert set(first.keys()) == {
        "id", "title", "artist", "genre", "mood", "energy",
        "tempo_bpm", "valence", "danceability", "acousticness",
    }


# --- score_song ------------------------------------------------------------

def test_genre_match_adds_two_points():
    match = score_song(PREFS, make_song(genre="pop"))[0]
    no_match = score_song(PREFS, make_song(genre="rock"))[0]
    assert round(match - no_match, 5) == 2.0


def test_mood_match_adds_one_point():
    match = score_song(PREFS, make_song(mood="happy"))[0]
    no_match = score_song(PREFS, make_song(mood="sad"))[0]
    assert round(match - no_match, 5) == 1.0


def test_energy_similarity_scoring():
    # Identical energy -> similarity 1.0 -> +2.0
    exact = score_song(PREFS, make_song(energy=0.8))[0]
    # 0.3 energy gap -> similarity 0.7 -> +1.4  (a 0.6 score drop)
    far = score_song(PREFS, make_song(energy=0.5))[0]
    assert round(exact - far, 5) == 0.6


def test_score_song_returns_reasons():
    score, reasons = score_song(PREFS, make_song(genre="pop", mood="happy", energy=0.8))
    assert score > 0
    assert any("genre match" in r for r in reasons)
    assert any("mood match" in r for r in reasons)
    assert any("energy similarity" in r for r in reasons)


# --- recommend_songs -------------------------------------------------------

def test_recommendations_sorted_descending():
    songs = [
        make_song(id=1, genre="rock", mood="sad", energy=0.1),   # weak match
        make_song(id=2, genre="pop", mood="happy", energy=0.8),  # strong match
        make_song(id=3, genre="pop", mood="sad", energy=0.6),    # medium match
    ]
    ranked = recommend_songs(PREFS, songs, k=3)
    scores = [score for _song, score, _why in ranked]
    assert scores == sorted(scores, reverse=True)
    assert ranked[0][0]["id"] == 2  # strongest match first


def test_recommend_songs_respects_k_limit():
    songs = load_songs(DATA_PATH)
    ranked = recommend_songs(PREFS, songs, k=3)
    assert len(ranked) == 3


# --- Recommender wrapper delegation ---------------------------------------

def make_small_recommender() -> Recommender:
    songs = [
        Song(1, "Test Pop Track", "Test Artist", "pop", "happy",
             0.8, 120, 0.9, 0.8, 0.2),
        Song(2, "Chill Lofi Loop", "Test Artist", "lofi", "chill",
             0.4, 80, 0.6, 0.5, 0.9),
    ]
    return Recommender(songs)


def test_recommender_delegates_and_returns_song_objects():
    user = UserProfile(
        favorite_genre="pop", favorite_mood="happy",
        target_energy=0.8, likes_acoustic=False,
    )
    rec = make_small_recommender()
    results = rec.recommend(user, k=2)

    assert len(results) == 2
    assert all(isinstance(s, Song) for s in results)
    # The pop/happy/high-energy song must rank first via the functional core.
    assert results[0].genre == "pop"
    assert results[0].mood == "happy"


def test_recommender_matches_functional_core():
    """Wrapper output order must equal the functional core's order (by id)."""
    user = UserProfile(
        favorite_genre="pop", favorite_mood="happy",
        target_energy=0.8, likes_acoustic=False,
    )
    rec = make_small_recommender()

    wrapper_ids = [s.id for s in rec.recommend(user, k=2)]
    song_dicts = [song.__dict__ for song in rec.songs]
    core_ids = [song["id"] for song, _s, _w in recommend_songs(user.to_prefs(), song_dicts, k=2)]
    assert wrapper_ids == core_ids


def test_explain_recommendation_returns_real_explanation():
    user = UserProfile(
        favorite_genre="pop", favorite_mood="happy",
        target_energy=0.8, likes_acoustic=False,
    )
    rec = make_small_recommender()
    explanation = rec.explain_recommendation(user, rec.songs[0])

    assert isinstance(explanation, str)
    assert explanation.strip() != ""
    assert explanation != "Explanation placeholder"
    # Real explanation derived from the scoring reasons.
    assert "genre match" in explanation
