"""
Deterministic natural-language preference parser.

Converts a free-text music request (e.g. "upbeat pop music for the gym") into
the canonical internal preference schema consumed by the recommender:

    {"genre": str, "mood": str, "energy": float}

This parser is fully deterministic and transparent — it uses fixed keyword and
activity maps and simple, documented rules. It never calls Claude, any model,
the network, or the filesystem.

Keyword / activity maps
-----------------------
- GENRE_TERMS: explicit genre words -> canonical genre.
- MOOD_TERMS: explicit mood words -> canonical mood.
- ACTIVITY_TERMS: activity phrases -> (mood, energy) assumptions.
- ENERGY_TERMS: energy words/phrases -> energy value in [0, 1].

Defaults (used when the request lacks the information)
------------------------------------------------------
    genre  -> "pop"
    mood   -> "happy"
    energy -> 0.5   (moderate)
`used_defaults` is True whenever ANY of the three fields fell back to a default.

Conflict-resolution policies
----------------------------
1. Explicit genre terms take priority over activity assumptions; genre is not
   inferred from activities.
2. When multiple genres appear, the FIRST (leftmost) one wins.
3. Explicit mood words take priority over activity-based mood assumptions.
4. Energy: each energy signal has a strength (1 = normal, 2 = strong, e.g.
   "high energy"/"low energy" or an intensified term like "very energetic").
   The highest-strength signal wins; ties are broken by earliest position.
5. Energy is always clamped to [0.0, 1.0].
"""

import re
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

# --- Defaults --------------------------------------------------------------

DEFAULT_GENRE = "pop"
DEFAULT_MOOD = "happy"
DEFAULT_ENERGY = 0.5


# --- Keyword / activity maps ----------------------------------------------

GENRE_TERMS: Dict[str, str] = {
    "pop": "pop",
    "rock": "rock",
    "lofi": "lofi",
    "lo fi": "lofi",
    "hip hop": "hip hop",
    "hiphop": "hip hop",
    "rap": "hip hop",
    "electronic": "electronic",
    "edm": "electronic",
    "techno": "electronic",
    "classical": "classical",
    "country": "country",
    "soul": "soul",
    "folk": "folk",
    "disco": "disco",
}

MOOD_TERMS: Dict[str, str] = {
    "happy": "happy",
    "upbeat": "happy",
    "cheerful": "happy",
    "joyful": "happy",
    "chill": "chill",
    "calm": "chill",
    "relaxing": "chill",
    "relaxed": "chill",
    "mellow": "chill",
    "intense": "intense",
    "aggressive": "intense",
    "powerful": "intense",
    "sad": "sad",
    "emotional": "sad",
    "melancholy": "sad",
    "moody": "sad",
    "energetic": "energetic",
    "hype": "energetic",
    "pumped": "energetic",
    "dreamy": "dreamy",
    "romantic": "romantic",
    "nostalgic": "nostalgic",
}

# Activity phrase -> (assumed mood, assumed energy value).
ACTIVITY_TERMS: Dict[str, Tuple[str, float]] = {
    "workout": ("intense", 0.9),
    "gym": ("intense", 0.9),
    "exercise": ("intense", 0.9),
    "running": ("energetic", 0.9),
    "run": ("energetic", 0.9),
    "jog": ("energetic", 0.9),
    "party": ("happy", 0.9),
    "dancing": ("happy", 0.9),
    "dance": ("happy", 0.9),
    "study": ("chill", 0.25),
    "studying": ("chill", 0.25),
    "focus": ("chill", 0.25),
    "sleep": ("chill", 0.15),
    "sleeping": ("chill", 0.15),
    "relax": ("chill", 0.25),
}

# Energy word/phrase -> energy value. Multi-word "high/low energy" are strong.
ENERGY_TERMS: Dict[str, float] = {
    "energetic": 0.9,
    "intense": 0.9,
    "upbeat": 0.9,
    "hype": 0.9,
    "pumped": 0.9,
    "fast": 0.85,
    "powerful": 0.85,
    "high energy": 0.9,
    "calm": 0.25,
    "relaxing": 0.25,
    "relaxed": 0.25,
    "mellow": 0.25,
    "chill": 0.3,
    "slow": 0.25,
    "quiet": 0.25,
    "soft": 0.3,
    "sleepy": 0.15,
    "sad": 0.3,
    "emotional": 0.35,
    "low energy": 0.2,
}

STRONG_ENERGY_PHRASES = {"high energy", "low energy"}
INTENSIFIERS = ("very", "super", "really", "extremely", "so")


# --- Typed result ----------------------------------------------------------

@dataclass(frozen=True)
class ParsedPreferences:
    """Deterministic parse of a natural-language request."""
    genre: str
    mood: str
    energy: float
    matched_terms: Tuple[str, ...] = ()
    used_defaults: bool = False

    def to_prefs(self) -> Dict[str, Any]:
        """Convert to the canonical preferences dict for the recommender."""
        return {"genre": self.genre, "mood": self.mood, "energy": self.energy}


# --- Helpers ---------------------------------------------------------------

def _normalize(text: str) -> str:
    """Lowercase, replace punctuation/hyphens with spaces, collapse whitespace."""
    lowered = text.lower()
    cleaned = re.sub(r"[^a-z0-9\s]", " ", lowered)  # drops punctuation & hyphens
    return re.sub(r"\s+", " ", cleaned).strip()


def _clamp(value: float, low: float = 0.0, high: float = 1.0) -> float:
    return max(low, min(high, value))


def _first_match(text: str, phrase: str) -> Optional[int]:
    """Return the start index of the first whole-word occurrence, or None."""
    m = re.search(r"\b" + re.escape(phrase) + r"\b", text)
    return m.start() if m else None


def _matches(text: str, terms: Dict[str, Any]) -> List[Tuple[int, str, Any]]:
    """Find the first occurrence of each phrase in `terms`, sorted by position."""
    found: List[Tuple[int, str, Any]] = []
    for phrase, payload in terms.items():
        start = _first_match(text, phrase)
        if start is not None:
            found.append((start, phrase, payload))
    found.sort(key=lambda t: t[0])
    return found


def _select_energy(text: str) -> Tuple[Optional[float], List[Tuple[int, str]]]:
    """
    Choose an energy value from all signals.

    Returns (energy_or_None, [(start, phrase), ...] for matched energy terms).
    Selection: highest strength wins; ties broken by earliest position.
    """
    candidates: List[Tuple[int, int, float]] = []  # (start, strength, value)
    matched: List[Tuple[int, str]] = []

    for start, phrase, value in _matches(text, ENERGY_TERMS):
        strength = 2 if phrase in STRONG_ENERGY_PHRASES else 1
        candidates.append((start, strength, value))
        matched.append((start, phrase))

    for start, phrase, (_mood, energy) in _matches(text, ACTIVITY_TERMS):
        candidates.append((start, 1, energy))
        matched.append((start, phrase))

    # Intensified single words: "very energetic", "super calm", ...
    single_energy = {p: v for p, v in ENERGY_TERMS.items() if " " not in p}
    for m in re.finditer(r"\b(?:%s)\s+([a-z]+)" % "|".join(INTENSIFIERS), text):
        word = m.group(1)
        if word in single_energy:
            base = single_energy[word]
            pushed = 0.95 if base >= 0.5 else 0.1
            candidates.append((m.start(), 2, pushed))

    if not candidates:
        return None, matched

    # Highest strength, then earliest position.
    best = min(candidates, key=lambda c: (-c[1], c[0]))
    return _clamp(best[2]), matched


# --- Public interface ------------------------------------------------------

def parse_preferences(user_request: str) -> ParsedPreferences:
    """
    Parse a natural-language request into ParsedPreferences.

    Deterministic and offline. Empty/whitespace-only input yields all defaults
    with `used_defaults=True` and no matched terms.
    """
    text = _normalize(user_request or "")

    # Genre — first explicit genre wins.
    genre_matches = _matches(text, GENRE_TERMS)
    genre = genre_matches[0][2] if genre_matches else DEFAULT_GENRE
    genre_defaulted = not genre_matches

    # Mood — explicit mood beats activity-derived mood.
    mood_matches = _matches(text, MOOD_TERMS)
    activity_matches = _matches(text, ACTIVITY_TERMS)
    if mood_matches:
        mood = mood_matches[0][2]
        mood_defaulted = False
    elif activity_matches:
        mood = activity_matches[0][2][0]  # (mood, energy)[0]
        mood_defaulted = False
    else:
        mood = DEFAULT_MOOD
        mood_defaulted = True

    # Energy — strongest signal wins.
    energy_value, _energy_matched = _select_energy(text)
    energy_defaulted = energy_value is None
    energy = DEFAULT_ENERGY if energy_defaulted else energy_value

    # Matched terms — all distinct surface phrases, ordered by position.
    all_hits: List[Tuple[int, str]] = []
    for start, phrase, _ in genre_matches:
        all_hits.append((start, phrase))
    for start, phrase, _ in mood_matches:
        all_hits.append((start, phrase))
    for start, phrase, _ in activity_matches:
        all_hits.append((start, phrase))
    for start, phrase, _ in _matches(text, ENERGY_TERMS):
        all_hits.append((start, phrase))

    all_hits.sort(key=lambda t: t[0])
    seen: set = set()
    matched_terms: List[str] = []
    for _start, phrase in all_hits:
        if phrase not in seen:
            seen.add(phrase)
            matched_terms.append(phrase)

    return ParsedPreferences(
        genre=genre,
        mood=mood,
        energy=_clamp(energy),
        matched_terms=tuple(matched_terms),
        used_defaults=genre_defaulted or mood_defaulted or energy_defaulted,
    )
