"""
Streamlit UI for the Explainable AI Music Discovery Assistant.

The UI is a thin presentation layer. It does NOT parse, retrieve, call Claude,
validate, score, fall back, or log on its own — every one of those steps lives
in the existing pipeline. This module only:

  1. loads the catalog via the existing `load_songs()`,
  2. delegates to `pipeline.discover_from_request()`,
  3. renders the parsed preferences, recommendations, and reliability info.

Nothing sensitive is ever displayed: no API keys, env contents, raw SDK
responses, exception text, stack traces, or full logs. Fallback reasons are
mapped from stable codes to friendly messages via `humanize_fallback_reason()`.
"""

import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

# Make the app runnable both as a package module (`from src.app import ...` in
# tests) and as a top-level script (`streamlit run src/app.py`). Streamlit runs
# the file as a script with no package parent, so relative imports would fail;
# adding the repo root to sys.path lets the absolute `src.*` imports resolve.
_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from src.logging_config import configure_logging
from src.pipeline import (
    FALLBACK_AI_ERROR,
    FALLBACK_API_ERROR,
    FALLBACK_EMPTY_RESPONSE,
    FALLBACK_INVALID_STRUCTURE,
    FALLBACK_MALFORMED_JSON,
    FALLBACK_MISSING_API_KEY,
    FALLBACK_NO_CANDIDATES,
    FALLBACK_VALIDATION_FAILED,
    RequestDiscoveryResult,
    discover_from_request,
    generate_recommendations,
)
from src.recommender import load_songs

APP_TITLE = "🎵 Explainable AI Music Discovery Assistant"
APP_DESCRIPTION = (
    "Describe the music you want in plain language. The system retrieves "
    "matching songs deterministically, asks an AI to explain the best picks "
    "grounded only in those songs, validates every recommendation, and safely "
    "falls back to the deterministic ranking if anything goes wrong."
)

EXAMPLE_REQUESTS = [
    "Give me upbeat pop music for the gym.",
    "I need calm lofi songs for studying.",
    "Play intense rock music for a workout.",
]

# Stable fallback code -> safe, human-readable explanation.
FALLBACK_MESSAGES: Dict[str, str] = {
    FALLBACK_MISSING_API_KEY: (
        "The AI assistant isn't configured (no API key), so these are the "
        "reliable deterministic matches."
    ),
    FALLBACK_API_ERROR: (
        "The AI assistant was temporarily unavailable, so these are the "
        "reliable deterministic matches."
    ),
    FALLBACK_EMPTY_RESPONSE: (
        "The AI assistant returned no usable answer, so these are the reliable "
        "deterministic matches."
    ),
    FALLBACK_MALFORMED_JSON: (
        "The AI assistant's answer couldn't be read, so these are the reliable "
        "deterministic matches."
    ),
    FALLBACK_INVALID_STRUCTURE: (
        "The AI assistant's answer had an unexpected shape, so these are the "
        "reliable deterministic matches."
    ),
    FALLBACK_AI_ERROR: (
        "The AI assistant couldn't complete the request, so these are the "
        "reliable deterministic matches."
    ),
    FALLBACK_VALIDATION_FAILED: (
        "Some AI picks didn't match the retrieved songs and were rejected, so "
        "these are the reliable deterministic matches."
    ),
    FALLBACK_NO_CANDIDATES: (
        "No songs matched your request in the catalog."
    ),
}

DEFAULT_FALLBACK_MESSAGE = (
    "Showing the reliable deterministic matches for your request."
)


# --- Testable helpers ------------------------------------------------------

def get_catalog_path() -> Path:
    """
    Resolve the absolute path to data/songs.csv relative to the repo root,
    so the app works regardless of the current working directory.
    """
    # src/app.py -> src/ -> repo root -> data/songs.csv
    return Path(__file__).resolve().parent.parent / "data" / "songs.csv"


def load_catalog(path: Optional[Path] = None) -> List[Dict[str, Any]]:
    """Load the song catalog using the existing load_songs() function."""
    catalog_path = path or get_catalog_path()
    return load_songs(str(catalog_path))


def humanize_fallback_reason(reason: Optional[str]) -> str:
    """
    Map a stable fallback reason code to a safe, user-facing message.

    Unknown or missing codes fall back to a generic message — raw codes/data are
    never surfaced verbatim.
    """
    if not reason:
        return DEFAULT_FALLBACK_MESSAGE
    return FALLBACK_MESSAGES.get(reason, DEFAULT_FALLBACK_MESSAGE)


def run_discovery(
    user_request: str,
    songs: List[Dict[str, Any]],
    *,
    retrieval_k: int = 5,
    output_k: int = 3,
    logger: Any = None,
    generation_function=generate_recommendations,
) -> RequestDiscoveryResult:
    """
    Delegate a natural-language request to the existing pipeline.

    This is the single call point into the application logic — the UI adds no
    parsing/retrieval/generation/validation/fallback of its own.
    """
    return discover_from_request(
        user_request,
        songs,
        retrieval_k=retrieval_k,
        output_k=output_k,
        generation_function=generation_function,
        logger=logger,
    )


# --- Rendering (imports streamlit lazily so helpers stay import-safe) -------

def _render_result(st, result: RequestDiscoveryResult) -> None:
    """Render a RequestDiscoveryResult with Streamlit widgets."""
    parsed = result.parsed_preferences
    discovery = result.discovery

    st.subheader("Interpreted preferences")
    cols = st.columns(3)
    cols[0].metric("Genre", parsed.genre)
    cols[1].metric("Mood", parsed.mood)
    cols[2].metric("Energy", f"{parsed.energy:.2f}")
    st.caption(
        f"Defaults used: {parsed.used_defaults} · "
        f"Matched terms ({len(parsed.matched_terms)}): "
        f"{', '.join(parsed.matched_terms) if parsed.matched_terms else '—'}"
    )

    # Reliability / transparency panel.
    st.subheader("Reliability & transparency")
    source_label = "AI-generated" if discovery.source == "ai" else "Safe fallback"
    st.write(f"**Final source:** {source_label}")
    st.write(f"**Fallback used:** {discovery.used_fallback}")
    if discovery.used_fallback:
        st.info(humanize_fallback_reason(discovery.fallback_reason))
    report = discovery.validation_report
    if report is not None:
        st.write(f"**Reliability score:** {report.reliability_score:.2f}")
        st.write(
            f"**Accepted:** {report.valid_count} · "
            f"**Rejected:** {len(report.rejected)} · "
            f"**Requested:** {report.total_requested}"
        )
    if discovery.model:
        st.write(f"**Model:** {discovery.model}")
    if discovery.request_id:
        st.caption(f"Request ID: {discovery.request_id}")

    # Recommendations.
    st.subheader("Recommendations")
    if not discovery.final_recommendations:
        st.warning("No recommendations to show for this request.")
    for i, rec in enumerate(discovery.final_recommendations, start=1):
        badge = "🤖 AI" if rec.source == "ai" else "🛟 Fallback"
        st.markdown(f"**{i}. {rec.title} — {rec.artist}**  ·  {badge}")
        if rec.score is not None:
            st.caption(f"Deterministic score: {rec.score:.2f}")
        st.write(rec.explanation)

    # Optional retrieved-candidates expander.
    with st.expander("Show retrieved candidates"):
        rows = [
            {"id": s.get("id"), "title": s.get("title"), "artist": s.get("artist"),
             "genre": s.get("genre"), "mood": s.get("mood"), "energy": s.get("energy")}
            for s in discovery.retrieved_candidates
        ]
        if rows:
            st.table(rows)
        else:
            st.write("No candidates were retrieved.")


# --- Main app --------------------------------------------------------------

def main() -> None:
    """Render the Streamlit application."""
    import streamlit as st

    logger = configure_logging()

    st.set_page_config(page_title="AI Music Discovery", page_icon="🎵")
    st.title(APP_TITLE)
    st.write(APP_DESCRIPTION)

    st.markdown("**Try one of these:**")
    st.markdown("\n".join(f"- {ex}" for ex in EXAMPLE_REQUESTS))

    user_request = st.text_area(
        "Your music request",
        placeholder=EXAMPLE_REQUESTS[0],
        key="user_request",
    )
    generate = st.button("Generate recommendations", type="primary")

    if not generate:
        return

    if not user_request or not user_request.strip():
        st.warning("Please enter a music request to get recommendations.")
        return

    # Load the catalog safely.
    try:
        songs = load_catalog()
    except FileNotFoundError:
        st.error("The song catalog could not be found. Please contact the maintainer.")
        return
    except Exception:
        # Never surface raw exception text / stack traces to the user.
        st.error("The song catalog could not be loaded.")
        return

    # Delegate to the pipeline. The pipeline handles missing API keys and AI
    # failures internally via its safe fallback, so those are not crashes here.
    try:
        result = run_discovery(user_request, songs, logger=logger)
    except Exception:
        st.error("Something went wrong while generating recommendations. Please try again.")
        return

    _render_result(st, result)


if __name__ == "__main__":
    main()
