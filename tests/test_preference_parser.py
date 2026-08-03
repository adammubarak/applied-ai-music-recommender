"""
Tests for src.preference_parser — deterministic, offline NL parsing.
"""

from src.preference_parser import (
    DEFAULT_ENERGY,
    DEFAULT_GENRE,
    DEFAULT_MOOD,
    ParsedPreferences,
    parse_preferences,
)


# --- Explicit extraction ---------------------------------------------------

def test_explicit_genre_extraction():
    p = parse_preferences("I love rock music")
    assert p.genre == "rock"
    assert "rock" in p.matched_terms


def test_explicit_mood_extraction():
    p = parse_preferences("give me something happy")
    assert p.mood == "happy"
    assert "happy" in p.matched_terms


def test_high_energy_extraction():
    p = parse_preferences("intense energetic music")
    assert p.energy >= 0.8


def test_low_energy_extraction():
    p = parse_preferences("calm relaxing music")
    assert p.energy <= 0.3


# --- Activity requests -----------------------------------------------------

def test_workout_request():
    p = parse_preferences("Play intense rock music for a workout.")
    assert p.genre == "rock"
    assert p.mood == "intense"
    assert p.energy >= 0.8


def test_study_request():
    p = parse_preferences("I need calm lofi songs for studying.")
    assert p.genre == "lofi"
    assert p.mood == "chill"
    assert p.energy <= 0.3


def test_party_request():
    p = parse_preferences("party music for dancing")
    assert p.mood == "happy"
    assert p.energy >= 0.8
    # No genre stated -> default genre, used_defaults True.
    assert p.genre == DEFAULT_GENRE
    assert p.used_defaults is True


def test_gym_upbeat_pop():
    p = parse_preferences("Give me upbeat pop music for the gym.")
    assert p.genre == "pop"
    assert p.energy >= 0.8
    assert p.mood == "happy"  # explicit "upbeat" beats gym's intense


# --- Defaults --------------------------------------------------------------

def test_default_behavior_on_empty():
    p = parse_preferences("")
    assert p.genre == DEFAULT_GENRE
    assert p.mood == DEFAULT_MOOD
    assert p.energy == DEFAULT_ENERGY
    assert p.matched_terms == ()
    assert p.used_defaults is True


def test_whitespace_only_input():
    p = parse_preferences("     \t   ")
    assert (p.genre, p.mood, p.energy) == (DEFAULT_GENRE, DEFAULT_MOOD, DEFAULT_ENERGY)
    assert p.used_defaults is True


def test_all_fields_present_no_defaults():
    p = parse_preferences("energetic rock")
    assert p.genre == "rock"
    assert p.mood == "energetic"
    assert p.energy >= 0.8
    assert p.used_defaults is False


# --- Case, punctuation, unknown words --------------------------------------

def test_case_and_punctuation_handling():
    p = parse_preferences("ROCK!!!  music, please.")
    assert p.genre == "rock"


def test_unknown_words_fall_back_to_defaults():
    p = parse_preferences("xyzzy florb quux")
    assert p.genre == DEFAULT_GENRE
    assert p.mood == DEFAULT_MOOD
    assert p.energy == DEFAULT_ENERGY
    assert p.used_defaults is True


# --- Conflicts & multiples -------------------------------------------------

def test_multiple_genres_first_wins():
    p = parse_preferences("rock and pop and disco")
    assert p.genre == "rock"


def test_conflicting_energy_earliest_equal_strength_wins():
    # "calm" (low, pos 0) vs "workout" (high, later) — equal strength -> earliest.
    p = parse_preferences("calm workout")
    assert p.energy <= 0.3
    assert p.mood == "chill"  # "calm" is an explicit mood word


def test_strong_energy_overrides_weaker():
    # Intensified "very energetic" (strength 2) beats "calm" (strength 1).
    p = parse_preferences("calm but very energetic")
    assert p.energy >= 0.9


def test_high_energy_phrase_is_strong():
    p = parse_preferences("slow but high energy please")
    assert p.energy >= 0.85


# --- Energy bounds ---------------------------------------------------------

def test_energy_within_unit_interval():
    for req in ["very energetic party workout", "very calm sleepy quiet",
                "", "pop", "intense rock for the gym", "sad emotional low energy"]:
        p = parse_preferences(req)
        assert 0.0 <= p.energy <= 1.0


def test_result_is_frozen_typed():
    p = parse_preferences("pop")
    assert isinstance(p, ParsedPreferences)
    assert p.to_prefs() == {"genre": p.genre, "mood": p.mood, "energy": p.energy}


# --- No side effects -------------------------------------------------------

def test_no_file_network_or_api_access(monkeypatch):
    import builtins
    import socket

    def open_guard(*a, **k):
        raise AssertionError("parser must not open files")

    def net_guard(*a, **k):
        raise AssertionError("parser must not touch the network")

    monkeypatch.setattr(builtins, "open", open_guard)
    monkeypatch.setattr(socket, "create_connection", net_guard)
    monkeypatch.setattr(socket.socket, "connect", net_guard)

    p = parse_preferences("upbeat pop for the gym")
    assert p.genre == "pop"
