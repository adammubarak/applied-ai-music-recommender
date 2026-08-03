"""
Tests for src.logging_config — structured, safe, non-duplicating logging.
"""

import logging
import os

from src.logging_config import (
    EVENT_RETRIEVAL_COMPLETED,
    configure_logging,
    format_event,
    get_logger,
    log_event,
)


def test_configure_returns_logger():
    logger = configure_logging(level=logging.DEBUG)
    assert isinstance(logger, logging.Logger)
    assert logger.level == logging.DEBUG


def test_repeated_configuration_does_not_duplicate_handlers():
    configure_logging()
    logger = configure_logging()
    tagged = [h for h in logger.handlers if getattr(h, "_music_discovery_handler", False)]
    assert len(tagged) == 1  # only one stream handler, not two


def test_optional_file_logging(tmp_path):
    log_path = tmp_path / "run.log"
    logger = configure_logging(level=logging.INFO, log_file=str(log_path))
    log_event(logger, EVENT_RETRIEVAL_COMPLETED, request_id="abc", retrieved_count=2)
    for h in logger.handlers:
        h.flush()
    content = log_path.read_text()
    assert "event=retrieval_completed" in content
    assert "retrieved_count=2" in content
    # Clean up so this file handler doesn't linger for later configure calls.
    configure_logging()


def test_stable_structured_event_formatting():
    msg = format_event(EVENT_RETRIEVAL_COMPLETED, {
        "request_id": "abc", "retrieved_count": 2, "retrieved_ids": [1, 3],
    })
    assert msg == "event=retrieval_completed request_id=abc retrieved_count=2 retrieved_ids=1,3"


def test_secrets_not_present_in_log_output(tmp_path, monkeypatch):
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-ant-super-secret")
    log_path = tmp_path / "secret.log"
    logger = configure_logging(log_file=str(log_path))
    # Log only the fields the pipeline would log — never the key.
    log_event(logger, EVENT_RETRIEVAL_COMPLETED, request_id="r1", retrieved_count=3)
    for h in logger.handlers:
        h.flush()
    content = log_path.read_text()
    assert "sk-ant-super-secret" not in content
    assert "ANTHROPIC_API_KEY" not in content
    configure_logging()


def test_log_event_never_raises_on_bad_logger():
    class Boom:
        def log(self, *a, **k):
            raise RuntimeError("logger exploded")
    # Should not raise.
    log_event(Boom(), "some_event", request_id="x")


def test_log_event_none_logger_is_noop():
    log_event(None, "some_event", x=1)  # no exception
