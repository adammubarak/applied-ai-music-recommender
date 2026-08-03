# 🎵 Music Recommender Simulation

## Project Summary

In this project you will build and explain a small music recommender system.

Your goal is to:

- Represent songs and a user "taste profile" as data
- Design a scoring rule that turns that data into recommendations
- Evaluate what your system gets right and wrong
- Reflect on how this mirrors real world AI recommenders

This project simulates a simple music recommendation system. It compares song features such as genre, mood, energy, tempo, valence, danceability, and acousticness with a user's music preferences. The system gives each song a score and ranks the songs so the best matches appear first.
---

## How The System Works
### Data Flow

Input: The user's favorite genre, favorite mood, target energy, and acoustic preference.

Process: The recommender loops through every song in `songs.csv`, calculates a score for each song using the scoring recipe, and stores the result.

Output: The songs are sorted from highest score to lowest score, and the top `k` songs are returned as recommendations.

This project uses a simple content-based music recommendation system. Real-world platforms may use collaborative filtering, which recommends content based on the behavior of similar users, and content-based filtering, which recommends content based on attributes of songs. These systems may use data such as likes, skips, replays, playlists, listening time, genre, mood, tempo, and energy.

In this simulation, each `Song` stores its title, artist, genre, mood, energy, tempo, valence, danceability, and acousticness. The current `UserProfile` stores the user's favorite genre, favorite mood, target energy, and whether the user likes acoustic music.

### User Taste Profile

The recommender will initially use this taste profile:

    user_profile = {
        "favorite_genre": "pop",
        "favorite_mood": "happy",
        "target_energy": 0.80,
        "likes_acoustic": False
    }

This profile represents a user who prefers happy pop songs with high energy and lower acousticness.
### Algorithm Recipe

The recommender calculates a score for each song using these rules:

- Add `2.0` points when the song's genre matches the user's favorite genre.
- Add `1.0` point when the song's mood matches the user's favorite mood.
- Calculate energy similarity using:

  `1 - abs(song.energy - user.target_energy)`

- Multiply the energy similarity by `2.0` and add it to the score.
- Add `0.5` points when the song matches the user's acoustic preference.

The scoring rule measures how well one song matches the user. The ranking rule then sorts all songs from highest score to lowest score. The songs with the highest scores are returned as recommendations.

### Potential Bias

This system may over-prioritize genre because genre matches are worth more points than mood matches. It may ignore songs from other genres that still match the user's energy or mood well. It also uses only a few user preferences, so it cannot fully represent a person's musical taste.
---
## Sample Recommendation Output

```text
Loaded songs: 18

Top recommendations:

Sunrise City - Score: 4.96
Because: genre match (+2.0), mood match (+1.0), energy similarity (+1.96)

Gym Hero - Score: 3.74
Because: genre match (+2.0), energy similarity (+1.74)

Rooftop Lights - Score: 2.92
Because: mood match (+1.0), energy similarity (+1.92)

Backbeat Bloom - Score: 1.98
Because: energy similarity (+1.98)

Neon Velvet - Score: 1.96
Because: energy similarity (+1.96)

## Getting Started

### Requirements

- **Python 3.10 or newer** (developed and tested on Python 3.13).

### 1. Clone the repository

```bash
git clone <your-repo-url>
cd applied-ai-system-final
```

### 2. Create and activate a virtual environment

**macOS / Linux:**

```bash
python3 -m venv .venv
source .venv/bin/activate
```

**Windows (PowerShell):**

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Configure your API key

Copy the example environment file and add your own Anthropic API key:

```bash
cp .env.example .env        # Windows: copy .env.example .env
```

Then edit `.env`:

```
ANTHROPIC_API_KEY=your_real_key_here
ANTHROPIC_MODEL=claude-sonnet-5
```

> **Never commit `.env`.** It holds your secret key and is already listed in
> `.gitignore`. Only `.env.example` (with placeholders) is tracked.

**No key? The app still works.** If `ANTHROPIC_API_KEY` is missing or the AI
call fails, the application safely falls back to the deterministic
recommendations instead of crashing.

### 5. Run the Streamlit app

```bash
python3 -m streamlit run src/app.py
```

Then open the local URL Streamlit prints (default http://localhost:8501).

### 6. (Optional) Run the CLI demo

A minimal deterministic demo over a few sample profiles:

```bash
python3 -m src.main
```

### 7. Run the tests

The full suite is offline and mocked — it makes no network or Anthropic API
calls and needs no API key:

```bash
python3 -m pytest -q
```

---

## Sample Recommendation Output

Paste a sample of your recommender's output here as a text block so a reader can see what it produces:

```
# e.g.:
# User profile: genre=indie, mood=chill, energy=low
# Recommendations:
#   1. ...
#   2. ...
#   3. ...
```

**Screenshot or video** *(optional)*: <!-- Insert a screenshot or demo video link here -->

---

## Experiments You Tried

Use this section to document the experiments you ran. For example:

- What happened when you changed the weight on genre from 2.0 to 0.5
- What happened when you added tempo or valence to the score
- How did your system behave for different types of users

---

## Limitations and Risks

Summarize some limitations of your recommender.

Examples:

- It only works on a tiny catalog
- It does not understand lyrics or language
- It might over favor one genre or mood

You will go deeper on this in your model card.

---

## Reflection

Read and complete `model_card.md`:

[**Model Card**](model_card.md)

Write 1 to 2 paragraphs here about what you learned:

- about how recommenders turn data into predictions
- about where bias or unfairness could show up in systems like this



