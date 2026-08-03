"""
Safe, structured logging for the Music Discovery Assistant.

Uses the standard library `logging`. Events are emitted with stable event names
and flat key=value fields so runs are greppable and machine-parseable.

Safety
------
- `log_event` never raises: if the logger or a handler fails, the exception is
  swallowed so the recommendation pipeline is never affected by logging.
- Nothing here reads or emits secrets. Callers must pass only non-sensitive
  fields (see the pipeline for what is logged). We never log the API key, whole
  config objects, raw SDK exception messages, stack traces, or full user text.

Duplicate handlers
------------------
`configure_logging` tags the handlers it adds and removes previously-tagged
handlers before re-adding, so calling it repeatedly does not accumulate
duplicate handlers.
"""

import logging
from typing import Any, Dict, Optional

# --- Logger identity & handler tagging -------------------------------------

LOGGER_NAME = "music_discovery"
_HANDLER_FLAG = "_music_discovery_handler"


# --- Stable event names ----------------------------------------------------

EVENT_PIPELINE_STARTED = "pipeline_started"
EVENT_PREFERENCES_PARSED = "preferences_parsed"
EVENT_RETRIEVAL_COMPLETED = "retrieval_completed"
EVENT_AI_GENERATION_STARTED = "ai_generation_started"
EVENT_AI_GENERATION_COMPLETED = "ai_generation_completed"
EVENT_VALIDATION_COMPLETED = "validation_completed"
EVENT_FALLBACK_USED = "fallback_used"
EVENT_PIPELINE_COMPLETED = "pipeline_completed"
EVENT_PIPELINE_INPUT_ERROR = "pipeline_input_error"


def get_logger() -> logging.Logger:
    """Return the shared logger, ensuring it never warns about missing handlers."""
    logger = logging.getLogger(LOGGER_NAME)
    if not logger.handlers:
        logger.addHandler(logging.NullHandler())
    return logger


def configure_logging(
    *,
    level: int = logging.INFO,
    log_file: Optional[str] = None,
) -> logging.Logger:
    """
    Configure and return the shared application logger.

    Args:
        level: Logging level (default INFO).
        log_file: Optional path; when given, a FileHandler is added.

    Returns:
        The configured `music_discovery` logger. Safe to call repeatedly —
        handlers this function added on a previous call are removed first, so
        handlers never accumulate.
    """
    logger = logging.getLogger(LOGGER_NAME)
    logger.setLevel(level)
    logger.propagate = False

    # Remove handlers we previously added (avoid duplicates); leave NullHandlers
    # and any handler we did not tag untouched.
    for handler in list(logger.handlers):
        if getattr(handler, _HANDLER_FLAG, False):
            logger.removeHandler(handler)

    formatter = logging.Formatter("%(asctime)s %(levelname)s %(name)s %(message)s")

    stream_handler = logging.StreamHandler()
    setattr(stream_handler, _HANDLER_FLAG, True)
    stream_handler.setLevel(level)
    stream_handler.setFormatter(formatter)
    logger.addHandler(stream_handler)

    if log_file:
        file_handler = logging.FileHandler(log_file)
        setattr(file_handler, _HANDLER_FLAG, True)
        file_handler.setLevel(level)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)

    return logger


# --- Structured event emission ---------------------------------------------

def _format_value(value: Any) -> str:
    """Render a field value as a compact, space-free string."""
    if isinstance(value, (list, tuple)):
        return ",".join(str(v) for v in value)
    return str(value)


def format_event(event: str, fields: Dict[str, Any]) -> str:
    """Build a stable `event=<name> key=value ...` message string."""
    parts = [f"event={event}"]
    for key, value in fields.items():
        parts.append(f"{key}={_format_value(value)}")
    return " ".join(parts)


def log_event(
    logger: Optional[logging.Logger],
    event: str,
    *,
    level: int = logging.INFO,
    **fields: Any,
) -> None:
    """
    Emit a structured event. Never raises — logging failures are contained so
    they cannot affect the recommendation pipeline.
    """
    if logger is None:
        return
    try:
        logger.log(level, format_event(event, fields))
    except Exception:
        # Logging must never change or break recommendation behavior.
        pass
