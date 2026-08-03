# Explainable AI Music Discovery Assistant

## Project Summary

The Explainable AI Music Discovery Assistant turns a plain-language music
request into a small set of recommended songs, each with a clear explanation of
why it was chosen. A user types something like *"upbeat pop music for the
gym,"* and the system parses that request into structured preferences,
deterministically retrieves the most relevant songs from a local catalog, and
asks Claude to select and explain the best picks **using only those retrieved
songs**. Every AI recommendation is then validated against the retrieved set; if
anything is wrong or the AI is unavailable, the system safely returns the
deterministic ranking instead.

This design matters because it makes AI recommendations **grounded**,
**explainable**, and **reliable**. The AI cannot invent songs that are not in
the catalog, its output is checked before a user ever sees it, and a missing API
key or model failure degrades gracefully to a trustworthy deterministic result
rather than a crash or a fabricated answer.

## Original Project

This project extends the **Music Recommender Simulation** from Module 3. That
project represented songs and a user taste profile as structured data, scored
each song by genre match, mood match, and energy similarity, ranked the whole
catalog deterministically, and reported the specific reasons each song received
its score.

The final project keeps that deterministic scorer as its trustworthy core and
builds a complete Applied AI System around it: a natural-language front end, a
Retrieval-Augmented Generation step with Claude that explains grounded picks, a
reliability layer that validates every AI recommendation, structured logging,
safe fallback, and a Streamlit user interface.

## Key Features

- **Natural-language preference parsing** — converts a free-text request into a
  canonical `{genre, mood, energy}` preference.
- **Deterministic retrieval** from `data/songs.csv` using the original scoring
  recipe (genre, mood, energy).
- **Retrieval-Augmented Generation with Claude** — the AI explains and selects
  from retrieved evidence rather than from open-ended memory.
- **Claude receives only the retrieved candidate songs**, never the full
  catalog.
- **Structured AI recommendation output** parsed into typed objects.
- **Reliability validation** of every recommendation against the retrieved set.
- **Transparent reliability score** computed by a simple, documented formula.
- **Safe deterministic fallback** whenever the AI path is unavailable or invalid.
- **Structured, privacy-safe logging** of pipeline events (no secrets, no full
  request text, no raw exception messages).
- **Streamlit user interface** for entering requests and viewing results.
- **Offline, mocked test suite** that makes no live or paid API calls.

This project does **not** use collaborative filtering, embeddings, vector
databases, semantic search, or model fine-tuning.

## How the System Works

1. The user enters a natural-language request in the Streamlit interface.
2. The parser converts the request into `genre`, `mood`, and `energy`
   preferences (with documented defaults when information is missing).
3. The deterministic recommender scores every catalog song and retrieves the
   top candidates.
4. Only those retrieved candidates are sent to Claude.
5. Claude returns structured song recommendations and short explanations.
6. The validator checks each recommendation's ID, title, and artist against the
   retrieved set, rejects duplicates, and confirms grounding.
7. If every recommendation is valid, the results are shown as AI recommendations
   with a reliability score.
8. If parsing, retrieval yields nothing, the API key is missing, the API call
   fails, the response is empty or malformed, or validation rejects any
   recommendation, the system returns safe deterministic fallback results.
9. Structured logs record safe pipeline events (parse, retrieval, generation,
   validation, fallback, completion) without exposing secrets or full requests.

## Architecture Overview

The full system diagram is maintained as Mermaid source at
[diagrams/architecture.mmd](diagrams/architecture.mmd).

At a high level, the application is organized into four areas:

- **Streamlit interface** — collects the request and displays results and
  transparency information.
- **Preference parser** — turns natural language into structured preferences.
- **Pipeline orchestrator** — coordinates the whole flow and owns the guardrails.
- **Deterministic retriever** — scores and ranks the local catalog (single
  source of truth for scoring).
- **Claude client** — generates grounded explanations from the retrieved
  candidates only.
- **Reliability validator** — verifies AI output against the retrieved songs and
  computes the reliability score.
- **Fallback path** — deterministic recommendations used whenever the AI path is
  unavailable or invalid.
- **Logging** — an observing component that records safe events.
- **Automated tests** — exercise every component offline with the AI client
  mocked.
- **Human review** — the user reads the displayed, explained recommendations.

**RAG boundary:** the catalog feeds the retriever, the retriever selects only
the top candidates, and only those candidates are sent to Claude. Claude never
directly accesses the complete `data/songs.csv` catalog.

## Project Structure

```
applied-ai-system-final/
├── data/
│   └── songs.csv              # Local song catalog (18 songs)
├── diagrams/
│   └── architecture.mmd       # Mermaid system architecture diagram
├── src/
│   ├── app.py                 # Streamlit UI (delegates to the pipeline)
│   ├── pipeline.py            # Orchestrator: retrieve -> generate -> validate -> fallback
│   ├── preference_parser.py   # Deterministic natural-language -> preferences
│   ├── recommender.py         # Deterministic scoring/ranking (single source of truth)
│   ├── ai_client.py           # Claude client for grounded explanations
│   ├── reliability.py         # Validation guardrail + reliability score
│   ├── logging_config.py      # Structured, safe logging
│   ├── config.py              # Env/config + safe API-key handling
│   └── main.py                # Optional deterministic CLI demo
├── tests/                     # Offline, mocked test suite
├── model_card.md              # Responsible-AI reflection, evaluation, risks
├── ai_interactions.md         # AI collaboration notes (project write-up)
├── .env.example               # Placeholder environment variables (safe to commit)
└── requirements.txt           # Dependencies: anthropic, python-dotenv, pytest, streamlit
```

## Setup Instructions

**Requirements:** Python 3.10 or newer (developed and tested on Python 3.13).

### 1. Clone the repository

```bash
git clone https://github.com/adammubarak/applied-ai-music-recommender.git
cd applied-ai-music-recommender
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
python3 -m pip install -r requirements.txt
```

### 4. Configure your API key

```bash
cp .env.example .env        # Windows: copy .env.example .env
```

Then edit `.env` and add your personal key:

```
ANTHROPIC_API_KEY=your_real_key_here
ANTHROPIC_MODEL=claude-sonnet-5
```

- `.env` is listed in `.gitignore` and **must never be committed**. Only
  `.env.example` (with placeholders) is tracked.
- `ANTHROPIC_MODEL` is optional. If it is unset, the application uses the default
  model defined in [src/config.py](src/config.py) (`claude-sonnet-5`). Set it to
  choose a different Claude model.

### 5. Run the Streamlit app

```bash
python3 -m streamlit run src/app.py
```

Open the local URL Streamlit prints (default http://localhost:8501).

### 6. (Optional) Run the deterministic CLI demo

```bash
python3 -m src.main
```

### 7. Run the tests

The suite is offline and mocked — it makes no network or Anthropic API calls and
needs no API key:

```bash
python3 -m pytest -q
```

**No API key?** The application still works. When `ANTHROPIC_API_KEY` is missing
(or the AI path fails), it returns deterministic fallback recommendations
instead of crashing.

## Sample Interactions

The outputs below are **representative tested interactions**. The parsed
preferences, retrieved candidates, and deterministic scores were produced by
running the project's real parser and retriever against `data/songs.csv`. The
AI-path examples were produced with a **mocked** generation function (as the
tests do), so no live Anthropic API call was made; a real run would return the
same grounded songs with model-authored explanations.

### Example 1 — Grounded AI path

**Request:** `Give me upbeat pop music for the gym.`

**Parsed preferences:** genre `pop`, mood `happy`, energy `0.9`
(matched terms: `upbeat`, `pop`, `gym`; defaults used: no)

**Retrieved candidates (top 5):**

| ID | Title | Artist | Genre | Mood | Energy |
|----|-------|--------|-------|------|--------|
| 1 | Sunrise City | Neon Echo | pop | happy | 0.82 |
| 5 | Gym Hero | Max Pulse | pop | intense | 0.93 |
| 10 | Rooftop Lights | Indigo Parade | indie pop | happy | 0.76 |
| 3 | Storm Runner | Voltline | rock | intense | 0.91 |
| 17 | Sunset Circuit | Dex Hollow | funk | triumphant | 0.88 |

**Recommendations (source: AI, reliability score: 1.0):**

1. **Sunrise City — Neon Echo** *(deterministic score 4.84)* — a happy, high-energy pop track that matches an upbeat gym session.
2. **Gym Hero — Max Pulse** *(deterministic score 3.94)* — pop with strong, workout-ready energy.
3. **Rooftop Lights — Indigo Parade** *(deterministic score 2.72)* — bright, upbeat feel for keeping momentum.

All three recommendations exist in the retrieved set, so validation passes with a reliability score of 1.0.

### Example 2 — Grounded study request

**Request:** `I need calm lofi songs for studying.`

**Parsed preferences:** genre `lofi`, mood `chill`, energy `0.25`
(matched terms: `calm`, `lofi`, `studying`; defaults used: no)

**Retrieved candidates (top 5):**

| ID | Title | Artist | Genre | Mood | Energy |
|----|-------|--------|-------|------|--------|
| 4 | Library Rain | Paper Lanterns | lofi | chill | 0.35 |
| 2 | Midnight Coding | LoRoom | lofi | chill | 0.42 |
| 9 | Focus Flow | LoRoom | lofi | focused | 0.4 |
| 6 | Spacewalk Thoughts | Orbit Bloom | ambient | chill | 0.28 |
| 13 | Glass Skyline | The Lumen Choir | classical | reflective | 0.29 |

**Recommendations (source: AI, reliability score: 1.0):**

1. **Library Rain — Paper Lanterns** *(deterministic score 4.80)* — calm lofi with low energy, ideal for focused studying.
2. **Midnight Coding — LoRoom** *(deterministic score 4.66)* — a chill lofi track well suited to quiet work.
3. **Focus Flow — LoRoom** *(deterministic score 3.70)* — a low-energy lofi choice for concentration.

Accepted: 3, rejected: 0, requested: 3 — validation passes, reliability score 1.0.

### Example 3 — Safe fallback

**Request:** `Play intense rock music for a workout.`
**Condition:** no `ANTHROPIC_API_KEY` configured (the same fallback occurs if the
AI response fails validation).

**Parsed preferences:** genre `rock`, mood `intense`, energy `0.9`
(matched terms: `intense`, `rock`, `workout`; defaults used: no)

**Recommendations (source: safe fallback, reason: `missing_api_key`):**

1. **Storm Runner — Voltline** *(deterministic score 4.98)* — genre match (+2.0), mood match (+1.0), energy similarity (+1.98).
2. **Gym Hero — Max Pulse** *(deterministic score 2.94)* — mood match (+1.0), energy similarity (+1.94).
3. **Sunset Circuit — Dex Hollow** *(deterministic score 1.96)* — energy similarity (+1.96).

The interface shows a clear, user-friendly message that these are the reliable
deterministic matches. There is no crash and no raw error message.

## Design Decisions and Trade-offs

**Decisions**

- **Deterministic retrieval before generation.** Retrieving first grounds the AI
  in real catalog songs and keeps recommendations explainable.
- **The functional recommender is the single source of truth.** All scoring runs
  through `recommend_songs()`; the pipeline never re-implements scoring.
- **A deterministic keyword parser instead of a second model call.** Parsing is
  transparent, instant, free, and easy to test.
- **Strict all-or-nothing validation before accepting AI output.** If any
  recommendation is invalid, the whole AI response is rejected in favor of the
  trustworthy deterministic result.
- **Stable fallback instead of crashing or showing partial invalid output.**
  Users always receive a coherent, grounded result.
- **Mocked AI tests.** The suite is offline, reproducible, and cost-free.
- **Local CSV catalog.** Simple and transparent for a portfolio-scale project.
- **Streamlit** for a fast, usable interface without heavy front-end work.

**Trade-offs**

- The small local catalog limits recommendation variety.
- Keyword parsing is transparent but less flexible than model-based intent
  extraction.
- Exact title/artist validation is safe but strict.
- All-or-nothing validation can discard an otherwise useful partly-valid
  response.
- The full AI path requires an Anthropic API key.
- The project uses no embeddings, semantic search, or live music-platform data.

## Testing Summary

The final Section 1 verification ran **134 tests, all passing**, including in a
clean virtual environment created outside the repository. The suite covers
natural-language parsing, deterministic retrieval, AI-client response handling,
reliability validation, fallback behavior, structured logging, pipeline
integration, Streamlit helper functions, a Streamlit interaction test, and
end-to-end behavior.

A detailed reliability and evaluation report — including the reliability-score
formula, a per-scenario results table, and quantitative metrics — is available
in [evaluation_results.md](evaluation_results.md).

All Anthropic interactions are **mocked**. The suite makes **no live network or
paid API calls**, and the Streamlit server started successfully in a headless
smoke test.

**What worked:**

- Grounded AI output limited to retrieved songs.
- Hallucination rejection (invented, mismatched, or duplicate picks are
  rejected).
- Safe deterministic fallback on missing key, API failure, or invalid output.
- Consistent request IDs correlating all events of a single run.
- Privacy-safe logging (no secrets, no full request text, no raw exceptions).
- Clean installation and startup.

**What was challenging or did not initially work:**

- The original Module 3 project contained two inconsistent recommender
  implementations (a working functional core and a stubbed OOP class).
- The starter tests exercised placeholder class behavior rather than the real
  functional core, so they passed without testing real logic.
- An early version of the architecture diagram incorrectly represented
  generation failures as validation failures.

These were resolved by unifying on the functional core (with the class delegating
to it), expanding the tests to cover real behavior, and separating the
generation guardrail from the validation guardrail in both the pipeline and the
diagram.

## Limitations

- Small, synthetic catalog (18 songs).
- Simple weighted retrieval using only genre, mood, and energy.
- Deterministic keyword parser with fixed vocabulary.
- No lyric, audio, or listening-history analysis.
- Exact-match validation of IDs, titles, and artists.
- The full AI explanation path depends on the Anthropic API.
- AI explanations may vary between runs even though the recommended songs are
  grounded and consistent.
- Fallback output is safe but less conversational than AI explanations.
- No deployment or persistent user profiles yet.

The detailed responsible-AI limitations and graded reflection are in
[model_card.md](model_card.md).

## Reflection

Building this project taught me to treat deterministic logic and generative AI
as distinct layers with a clear boundary between them. Grounding the AI with
retrieved evidence — and refusing anything it cannot justify against that
evidence — made the difference between a demo and something I would trust in
front of a user. I came to see reliability checks and fallback not as extras but
as core application features, and I learned that designing the system as small,
independently testable modules is what made grounding, validation, and safe
degradation practical to build and verify. Most of all, the project pushed me to
build for a user's experience rather than to simply demonstrate an API call.

## Responsible AI and Model Card

The [model card](model_card.md) contains the responsible-AI reflection,
evaluation details, known risks, and system limitations for this project.

## Running Without an API Key

The application remains fully usable without an Anthropic API key. When
`ANTHROPIC_API_KEY` is missing (or the AI path fails for any reason), the system
returns deterministic fallback recommendations — grounded, scored, and
explained — instead of failing.

## Portfolio Value

This project demonstrates modular Python design, a practical
Retrieval-Augmented Generation integration, working with an external AI API,
a comprehensive offline test suite, reliability guardrails, privacy-safe
logging, responsible-AI practices, and end-to-end user-facing application
development. It shows an AI feature integrated into a real application with the
engineering discipline — grounding, validation, safe fallback, and testing —
that turns a model call into a dependable product.
