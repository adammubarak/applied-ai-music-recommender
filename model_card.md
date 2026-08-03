# 🎧 Model Card: Explainable AI Music Discovery Assistant

## 1. System Name

**Explainable AI Music Discovery Assistant** (deterministic core: **VibeMatch**).

This model card documents the completed Applied AI System. It builds on the
original Module 3 project, the **Music Recommender Simulation**.

Related documentation:

- [README.md](README.md) — project overview, setup, and sample interactions
- [evaluation_results.md](evaluation_results.md) — reliability and evaluation report
- [diagrams/architecture.mmd](diagrams/architecture.mmd) — system architecture diagram

## 2. Original Project

The original project was the **Music Recommender Simulation** (Module 3). It
represented songs and a user taste profile as structured data, scored each song
by genre match, mood match, and energy similarity, ranked the whole catalog
deterministically, and explained why each song received its score. That
deterministic scorer is preserved in this project as the single source of truth
for retrieval and fallback.

## 3. Intended Use

The Explainable AI Music Discovery Assistant accepts a natural-language music
request, retrieves relevant songs from a small local catalog, asks Claude to
select and explain recommendations **using only the retrieved songs**, validates
that output, and safely falls back to the deterministic ranking when needed. It
is an educational portfolio project for learning how to integrate a language
model into a real application with grounding, validation, and safe fallback.

It is **not** intended to replace a real music platform or to make important
decisions about real users. It uses a small fictional dataset and only a few
preferences, so it should not be treated as proof of a person's full musical
taste.

## 4. How the System Works

1. A natural-language request is parsed into a canonical `{genre, mood, energy}`
   preference by a deterministic keyword parser.
2. The deterministic recommender scores every catalog song (genre match `+2.0`,
   mood match `+1.0`, energy similarity scaled by `2.0`) and retrieves the top
   candidates.
3. Only those retrieved candidates are sent to Claude — never the full catalog.
4. Claude returns structured recommendations and short explanations chosen from
   the retrieved candidates.
5. The reliability layer validates every recommendation's ID, title, and artist
   against the retrieved set, rejects duplicates, and computes a reliability
   score.
6. Fully valid AI output is displayed as AI recommendations; otherwise the system
   returns deterministic fallback recommendations with their scores and reasons.
7. Structured logging records safe pipeline events, and the Streamlit interface
   shows whether the result came from AI or fallback.

## 5. Data

The catalog is a CSV of 18 fictional songs. The starter dataset originally
contained 10 songs, and 8 more were added to increase the variety of genres and
moods.

The catalog includes genres such as pop, lofi, rock, soul, country, classical,
hip hop, folk, disco, funk, and reggae, and moods such as happy, chill, intense,
dreamy, nostalgic, reflective, playful, hopeful, romantic, triumphant, and
serene. Each song includes an ID, title, artist, genre, mood, energy, tempo,
valence, danceability, and acousticness. The dataset is small and does not
represent every genre, culture, artist style, language, or musical preference.

## 6. Deterministic Core Evaluation

The tables below show the deterministic retrieval/fallback core — the same logic
that grounds the AI path and serves as the safe fallback. Scores are produced by
the current scoring recipe.

### High-Energy Pop

```
Preferences: {'genre': 'pop', 'mood': 'happy', 'energy': 0.9}

Sunrise City - Score: 4.84
Because: genre match (+2.0), mood match (+1.0), energy similarity (+1.84)

Gym Hero - Score: 3.94
Because: genre match (+2.0), energy similarity (+1.94)

Rooftop Lights - Score: 2.72
Because: mood match (+1.0), energy similarity (+1.72)
```

### Chill Lofi

```
Preferences: {'genre': 'lofi', 'mood': 'chill', 'energy': 0.25}

Library Rain - Score: 4.80
Because: genre match (+2.0), mood match (+1.0), energy similarity (+1.80)

Midnight Coding - Score: 4.66
Because: genre match (+2.0), mood match (+1.0), energy similarity (+1.66)

Focus Flow - Score: 3.70
Because: genre match (+2.0), energy similarity (+1.70)
```

### Deep Intense Rock

```
Preferences: {'genre': 'rock', 'mood': 'intense', 'energy': 0.9}

Storm Runner - Score: 4.98
Because: genre match (+2.0), mood match (+1.0), energy similarity (+1.98)

Gym Hero - Score: 2.94
Because: mood match (+1.0), energy similarity (+1.94)

Sunset Circuit - Score: 1.96
Because: energy similarity (+1.96)
```

The rankings are transparent because every recommendation includes a numeric
score and specific reasons, and they are sensitive to the chosen weights:
because genre matches are worth more than mood matches, genre can be
over-prioritized. Full reliability and validation results are in
[evaluation_results.md](evaluation_results.md).

---

# Responsible AI Reflection

## Limitations and Biases

**What are the limitations or biases in the system?**

- The catalog contains only 18 synthetic songs, so recommendation variety and
  representativeness are limited.
- The deterministic retrieval formula gives genre matches more weight (`+2.0`)
  than mood matches (`+1.0`), which may over-prioritize genre.
- The keyword parser relies on predefined mappings and defaults, so ambiguous or
  culturally specific language may be interpreted incorrectly.
- The parser defaults to pop, happy, and medium energy (0.5) when information is
  missing, which can introduce a mainstream/default bias.
- The system does not use listening history, audio signals, lyrics, language,
  accessibility needs, cultural background, or collaborative behavior.
- Exact title and artist validation improves safety but may reject harmless
  formatting differences.
- The reliability score measures grounding validity, not whether a
  recommendation is subjectively good.
- A reliability score of 1.0 does not prove that the music matches every user
  preference.
- Claude explanations may vary even when the candidate songs remain grounded.
- The local catalog may not represent all genres, artists, cultures, or musical
  traditions equally.

The system is not unbiased, objective, hallucination-proof, or production-ready.

## Potential Misuse and Safeguards

**Could the AI be misused, and how would misuse be prevented?**

Realistic risks:

- A user could treat the recommendations or explanations as more authoritative
  than they are.
- The system could be repurposed to repeatedly promote a limited group of songs
  or genres.
- Biased catalog data or weighting could systematically exclude some music.
- Logs could create privacy concerns if full requests or secrets were recorded.
- API keys could be exposed if environment files were committed.
- AI-generated explanations could sound confident despite limited evidence.

Safeguards currently implemented:

- Claude receives only the retrieved candidates rather than the full catalog or
  unrestricted freedom to invent songs.
- IDs, titles, and artists are validated against the retrieved candidates.
- Invalid, duplicated, malformed, empty, or unsupported AI output is rejected.
- Missing keys and API failures trigger deterministic fallback.
- Rejected AI recommendations never reach the final output.
- The reliability score is calculated by application logic rather than
  self-reported by Claude.
- Logging excludes full user requests, API keys, raw SDK responses, and raw
  exception messages.
- `.env` is ignored by Git, and `.env.example` contains placeholders only.
- Automated tests run offline with mocked Anthropic behavior.
- The Streamlit interface displays whether output came from AI or fallback.
- Human review remains the final checkpoint.

Safeguards that could be added later:

- broader and audited catalog data
- user feedback controls
- rate limiting
- content moderation if user-generated catalog data were added
- periodic bias evaluation
- live-model monitoring

## Reliability Testing Surprises

**What was surprising while testing reliability?**

- The original project had two recommender implementations that did not agree.
- The early tests passed even though the class-based `Recommender` returned an
  unsorted slice and a placeholder explanation.
- This showed that passing tests do not prove correctness when the tests
  exercise the wrong implementation.
- Strict grounding validation made it possible to reject invented IDs,
  mismatched titles, mismatched artists, and duplicates.
- The system remained usable without an API key because deterministic fallback
  was integrated into the main pipeline.
- Logging itself had to be treated as a possible failure source, so logging
  errors were contained.
- Network-blocked tests demonstrated that the suite was truly offline and did
  not silently contact Anthropic.
- The first architecture diagram incorrectly grouped generation failures with
  validation failures; separating those control-flow stages improved the design.
- A reliability score of 1.0 only means all recommendations were grounded, not
  that they were musically ideal.

Verified results at the time of final evaluation:

- 134 automated tests passed.
- 0 known failing tests.
- No live or paid Anthropic calls occurred during the automated suite.

No formal human evaluation or large-scale live-model evaluation was performed.
Reliability was measured through automated, mocked, and structured offline
testing.

## Collaboration with AI

AI assistance (Claude Code) was used to:

- inspect and explain the existing repository
- propose a phased implementation plan
- generate and revise Python modules
- draft mocked tests
- identify edge cases
- verify terminal output
- help document the architecture and README

The human role was to:

- choose the project direction
- approve each phase
- check requirements against the rubric
- review AI-generated code and reports
- request corrections
- decide which trade-offs to accept
- prevent changes from being committed before verification

AI did not complete the project independently; every phase was human-directed
and human-reviewed.

### Helpful AI Suggestion

The AI recommended unifying the project around the working functional
functions — `load_songs()`, `score_song()`, and `recommend_songs()` — and making
the `Recommender` class delegate to that functional core.

This was helpful because it:

- removed the conflicting implementations
- made the tests and the application use the same logic
- preserved the working behavior
- avoided a risky full object-oriented rewrite
- created a stable foundation for RAG, validation, fallback, and Streamlit

### Flawed or Incorrect AI Suggestion

The AI's first architecture diagram routed missing API keys, API failures,
malformed responses, and empty responses through the "Validation passed?"
decision.

This was incorrect because:

- those failures occur before a structured AI response exists
- the reliability validator should only receive successfully parsed output
- generation failures should bypass validation and go directly to fallback

It was corrected by:

- introducing a separate "Generation succeeded?" decision
- routing generation failures directly to deterministic fallback
- allowing only structured responses to proceed to reliability validation
- confirming that rejected recommendations have no route to the user

**Lesson:** AI was most useful as an implementation and review partner, but its
output still required human verification against the actual code and rubric.
