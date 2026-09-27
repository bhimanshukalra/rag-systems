import json
import logging

import pytest

from agentic_rag.observability.logging import JSONFormatter, configure_logging


@pytest.fixture(autouse=True)
def _restore_root_logger():
    root = logging.getLogger()
    original_handlers = root.handlers[:]
    original_level = root.level
    yield
    root.handlers = original_handlers
    root.setLevel(original_level)


def test_json_formatter_produces_valid_json_with_expected_fields():
    formatter = JSONFormatter()
    record = logging.LogRecord(
        name="my.logger",
        level=logging.INFO,
        pathname=__file__,
        lineno=1,
        msg="hello %s",
        args=("world",),
        exc_info=None,
    )

    payload = json.loads(formatter.format(record))

    assert payload["level"] == "INFO"
    assert payload["logger"] == "my.logger"
    assert payload["message"] == "hello world"
    assert "timestamp" in payload


def test_configure_logging_sets_json_formatter_on_root_handler():
    configure_logging()

    root = logging.getLogger()
    assert len(root.handlers) == 1
    assert isinstance(root.handlers[0].formatter, JSONFormatter)


def test_configure_logging_quiets_noisy_third_party_loggers():
    configure_logging()

    assert logging.getLogger("httpx").level == logging.WARNING
    assert logging.getLogger("httpcore").level == logging.WARNING
    assert logging.getLogger("urllib3").level == logging.WARNING
