"""
Core recommendation logic for the Music Discovery project.

Single source of truth
----------------------
The functional API below is the ONLY implementation of scoring and ranking:

- load_songs()       -> read the CSV catalog into a list of song dicts
- score_song()       -> score one song dict against a preferences dict
- recommend_songs()  -> rank the whole catalog and return the top-k

The OOP classes (Song, UserProfile, Recommender) are thin compatibility
wrappers. Recommender delegates to the functions above; it does NOT contain a
second scoring or ranking implementation.

Canonical preference schema
----------------------------
Internally, user preferences are always a plain dict with these keys:

    {
        "genre":  str,     # favorite genre, e.g. "pop"
        "mood":   str,     # favorite mood,  e.g. "happy"
        "energy": float,   # target energy in [0.0, 1.0]
    }

UserProfile (favorite_genre / favorite_mood / target_energy / likes_acoustic)
is a user-facing convenience type. It is converted to the canonical dict via
UserProfile.to_prefs() before it reaches the scoring core. `likes_acoustic` is
kept for backward compatibility but is not used by the current scoring recipe.

Song schema
-----------
Songs are plain dicts with the columns of data/songs.csv:
id, title, artist, genre, mood, energy, tempo_bpm, valence, danceability,
acousticness. The Song dataclass mirrors these fields for callers that prefer
objects; it is converted to a dict via song_to_dict() before scoring.
"""

import csv
from typing import List, Dict, Tuple
from dataclasses import dataclass, asdict


@dataclass
class Song:
    """A song and its attributes. Compatibility type; see song_to_dict()."""
    id: int
    title: str
    artist: str
    genre: str
    mood: str
    energy: float
    tempo_bpm: float
    valence: float
    danceability: float
    acousticness: float


@dataclass
class UserProfile:
    """
    A user's taste preferences (user-facing convenience type).

    Use to_prefs() to convert into the canonical preferences dict that the
    scoring core understands.
    """
    favorite_genre: str
    favorite_mood: str
    target_energy: float
    likes_acoustic: bool

    def to_prefs(self) -> Dict:
        """Convert to the canonical preferences dict used by score_song()."""
        return {
            "genre": self.favorite_genre,
            "mood": self.favorite_mood,
            "energy": self.target_energy,
        }


def song_to_dict(song: Song) -> Dict:
    """Convert a Song dataclass into the canonical song dict."""
    return asdict(song)


class Recommender:
    """
    Thin object-oriented wrapper over the functional core.

    Stores a list of Song objects and delegates all scoring and ranking to
    score_song() / recommend_songs(). It holds no independent logic.
    """

    def __init__(self, songs: List[Song]):
        self.songs = songs

    def recommend(self, user: UserProfile, k: int = 5) -> List[Song]:
        """
        Rank the stored songs for `user` and return the top-k Song objects.

        Delegates to recommend_songs(). The functional core works on dicts, so
        songs are converted to dicts for scoring and mapped back to the
        original Song objects (by id) to preserve the OOP return type.
        """
        prefs = user.to_prefs()
        song_dicts = [song_to_dict(song) for song in self.songs]
        ranked = recommend_songs(prefs, song_dicts, k=k)

        by_id = {song.id: song for song in self.songs}
        return [by_id[song["id"]] for song, _score, _explanation in ranked]

    def explain_recommendation(self, user: UserProfile, song: Song) -> str:
        """
        Return a human-readable explanation for why `song` matches `user`,
        built from the same scoring reasons used for ranking.
        """
        prefs = user.to_prefs()
        _score, reasons = score_song(prefs, song_to_dict(song))
        return ", ".join(reasons) if reasons else "no matching criteria"


def load_songs(csv_path: str) -> List[Dict]:
    """Load songs from a CSV file into a list of canonical song dicts."""
    with open(csv_path, newline="", encoding="utf-8") as csv_file:
        reader = csv.DictReader(csv_file)
        songs = []
        for row in reader:
            songs.append(
                {
                    "id": int(row["id"]),
                    "title": row["title"],
                    "artist": row["artist"],
                    "genre": row["genre"],
                    "mood": row["mood"],
                    "energy": float(row["energy"]),
                    "tempo_bpm": float(row["tempo_bpm"]),
                    "valence": float(row["valence"]),
                    "danceability": float(row["danceability"]),
                    "acousticness": float(row["acousticness"]),
                }
            )

    print(f"Loaded songs: {len(songs)}")
    return songs


def score_song(user_prefs: Dict, song: Dict) -> Tuple[float, List[str]]:
    """
    Score a song against the canonical preferences dict.

    Recipe:
      +2.0  when song genre matches user's favorite genre
      +1.0  when song mood matches user's favorite mood
      +energy_similarity * 2.0, where energy_similarity = 1 - |Δenergy|

    Returns (score, reasons) where reasons is a list of human-readable strings.
    """
    score = 0.0
    reasons: List[str] = []

    if song["genre"] == user_prefs["genre"]:
        score += 2.0
        reasons.append("genre match (+2.0)")

    if song["mood"] == user_prefs["mood"]:
        score += 1.0
        reasons.append("mood match (+1.0)")

    energy_similarity = 1 - abs(song["energy"] - user_prefs["energy"])
    energy_score = energy_similarity * 2.0
    if energy_score > 0:
        score += energy_score
        reasons.append(f"energy similarity (+{energy_score:.2f})")

    return float(score), reasons


def recommend_songs(
    user_prefs: Dict, songs: List[Dict], k: int = 5
) -> List[Tuple[Dict, float, str]]:
    """
    Rank songs by relevance for a preferences dict and return the top-k as
    (song, score, explanation) tuples, sorted by descending score.
    """
    scored_songs = []
    for song in songs:
        score, reasons = score_song(user_prefs, song)
        explanation = ", ".join(reasons) if reasons else "no matching criteria"
        scored_songs.append((song, float(score), explanation))

    ranked_songs = sorted(scored_songs, key=lambda item: item[1], reverse=True)
    return ranked_songs[:k]
