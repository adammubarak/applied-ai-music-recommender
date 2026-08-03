# Reliability and Evaluation Results

## Evaluation Approach

The Explainable AI Music Discovery Assistant is evaluated through automated,
offline testing rather than live-model experimentation. The approach combines:

- **Automated unit and integration tests** across every module.
- **Mocked Claude API behavior** — the AI client is exercised through injected
  mock generation functions and fabricated responses.
- **Deterministic reliability scoring** validated against known inputs.
- **Strict recommendation validation** against the retrieved candidate set.
- **Fallback testing** for every failure condition.
- **Structured logging and error-handling tests**, including logger-failure
  containment.
- **Streamlit smoke and interaction testing** via `streamlit.testing.v1.AppTest`.

**No live or paid Anthropic API calls are made by the automated test suite.**
Every AI interaction is mocked, and the full suite passes with all outbound
network sockets and the real Anthropic client constructor blocked.

## Reliability Score

The reliability score is computed in [src/reliability.py](src/reliability.py)
using this exact formula:

```
reliability_score = valid_recommendation_count / total_recommendation_count
```

Key properties:

- The score is **0.0** when no AI recommendations are returned.
- The result is **clamped between 0.0 and 1.0**.
- The score is **calculated by application logic, not self-reported by Claude** —
  no model-supplied confidence value is used.
- A score of **1.0** means every returned recommendation passed grounding
  validation (its ID, title, and artist match a retrieved candidate, with no
  duplicates).
- **Any rejected recommendation** (invented, mismatched, or duplicate) causes the
  complete AI response to fail validation, and the pipeline then uses the safe
  deterministic fallback.

## Automated Test Summary

- **Total tests passed:** 134
- **Total tests failed:** 0
- **Python version used:** Python 3.13.7
- **Clean virtual-environment test:** previously verified in Section 1 — a fresh
  venv created outside the repository installed `requirements.txt`, imported all
  modules, and ran the full suite (134 passed).
- **Streamlit startup:** verified via a headless smoke run
  (`python3 -m streamlit run src/app.py --server.headless true`), which started
  successfully and was stopped cleanly.
- **Live network / API calls:** none. The suite also passes with all sockets and
  the Anthropic client constructor blocked (134 passed, exit code 0).

## Evaluation Scenarios

Each scenario below is demonstrated by one or more existing tests.

| Test Scenario | Expected Behavior | Observed Result | Reliability Outcome | Status |
|---|---|---|---|---|
| 1. Valid grounded AI recommendations | Accept AI output grounded in retrieved songs | AI output accepted; all recommendations valid | Score 1.0; source = AI; no crash | PASS |
| 2. Invented or unknown song ID | Reject and fall back | Recommendation rejected (`unknown_id`); fallback used | Rejected; deterministic fallback | PASS |
| 3. Correct ID with incorrect title | Reject and fall back | Recommendation rejected (`title_mismatch`); fallback used | Rejected; deterministic fallback | PASS |
| 4. Correct ID and title with incorrect artist | Reject and fall back | Recommendation rejected (`artist_mismatch`) | Rejected; deterministic fallback | PASS |
| 5. Duplicate AI recommendation | Reject the duplicate and fall back | First accepted, duplicate rejected (`duplicate`); response not fully valid | Fallback used | PASS |
| 6. Missing Anthropic API key | Fall back safely, no crash | `MissingAPIKeyError` handled; fallback reason `missing_api_key` | Safe deterministic fallback | PASS |
| 7. Anthropic / API failure | Fall back safely, no crash | `APICallError` handled; fallback reason `api_error` | Safe deterministic fallback | PASS |
| 8. Malformed JSON response | Fall back safely, no crash | `MalformedResponseError`; fallback reason `malformed_json` | Safe deterministic fallback | PASS |
| 9. Empty AI response | Fall back safely, no crash | `EmptyResponseError` / empty recommendations trigger fallback | Safe deterministic fallback | PASS |
| 10. Empty retrieved-song set | Handle without calling AI or crashing | No candidates; fallback reason `no_candidates`; empty output, no AI call | Safe handling; score 0.0 | PASS |
| 11. Empty or whitespace-only user input | Parse deterministically without crashing | Parser returns documented defaults; UI warns before running | Handled without crash | PASS |
| 12. Logger failure | Logging failure must not affect recommendations | Failing logger swallowed; normal AI result still returned | No crash; behavior unchanged | PASS |
| 13. Streamlit initial render | Page renders with no AI call | Title, request input, and Generate button render; no exception | No crash; no AI call | PASS |
| 14. Streamlit interaction without an API key | Show safe fallback, not a crash | Request entered, button clicked, key absent → fallback UI with transparency; no secret/exception shown | Safe deterministic fallback | PASS |
| 15. Full test suite with network access blocked | Suite passes offline | 134 passed with sockets and Anthropic client blocked | No live network/API call | PASS |

## Quantitative Summary

- **Current passing tests:** 134
- **Known failing tests:** 0
- **Reliability score for a fully valid mocked response:** 1.0
- **Reliability score when all recommendations are invalid:** 0.0
- **Do rejected recommendations reach final output?** No — rejected AI
  recommendations never appear in the final output; a non-passing response routes
  to deterministic fallback.
- **Do tests require a real API key?** No.
- **Do tests make a live network call?** No.

These figures are measured directly from the current test run and the
reliability formula; no averages or percentages are estimated.

## What Worked

- **Deterministic retrieval is reproducible** — the same request and catalog
  always produce the same ranking and scores.
- **Claude is grounded in retrieved candidates** — only retrieved songs are sent
  to the model, and the model is instructed to choose only from them.
- **Validation catches invented or mismatched recommendations** — unknown IDs,
  title mismatches, artist mismatches, and duplicates are all rejected.
- **Safe fallback prevents crashes** — every failure condition returns a coherent
  deterministic result instead of an error.
- **Logging records stable events without exposing secrets** — no API key, full
  request text, or raw exception message is logged.
- **Mocked testing is offline and reproducible** — the suite runs with no network
  and no API key.
- **Streamlit works with and without an API key** — it renders and, on the
  no-key path, shows safe fallback recommendations.

## Problems Found and Improvements Made

- **Original tests exercised placeholder class behavior instead of the functional
  recommender.** The starter tests passed against a stubbed `Recommender` class
  that returned placeholders. *Fixed* by making the class delegate to the
  functional core and expanding the tests to check real scoring, ranking, and
  explanations.
- **Two recommender implementations initially disagreed.** A working functional
  API and a stubbed OOP class coexisted with different behavior. *Fixed* by
  unifying on the functional core as the single source of truth.
- **Invalid AI output required strict validation.** *Fixed* by adding the
  reliability layer that checks every recommendation's ID, title, and artist
  against the retrieved set and rejects duplicates.
- **Generation failures and validation failures needed separate control-flow
  paths.** An early architecture conflated them. *Fixed* by separating the
  generation guardrail (missing key, API failure, malformed, empty, invalid
  structure → straight to fallback) from the validation guardrail (grounding
  check on a structured response), in both the pipeline and the diagram.
- **Logging needed containment so logger failures could not break
  recommendations.** *Fixed* by routing all emission through a `log_event`
  helper that swallows any logging exception, verified by a failing-logger test.

## Remaining Reliability Limitations

- Tests use **mocked Claude output** rather than measuring live-model
  consistency.
- The catalog is **small and synthetic** (18 songs).
- **Exact title and artist matching** may reject harmless formatting differences.
- The reliability score measures **grounding validity, not musical quality**.
- A score of 1.0 does **not** prove a recommendation matches every subjective
  preference.
- **Live API latency, rate limits, and changing model behavior** are not measured
  by offline tests.
- **No large-scale human evaluation** has been performed.

## Testing Conclusion

134 out of 134 automated tests passed (Python 3.13.7). Grounded recommendations
received a reliability score of 1.0, while invented, mismatched, or duplicate
recommendations were rejected and replaced with deterministic fallback. The
system handled missing API keys, malformed responses, empty responses, API
failures, empty retrieved sets, and logging failures without crashing, and the
full suite passed with all network access and the real Anthropic client blocked.

No formal human evaluation was included in this phase; reliability was measured
through automated and structured offline testing.
