from __future__ import annotations

import logging
import re
from copy import deepcopy
from typing import Any

from uvicorn.config import LOGGING_CONFIG

_EVIDENCE_PATH = re.compile(r"(/v1/searches/)[^/?\s]+(/evidence/)[^?\s]+")


def _redact_evidence_path(value: str) -> str:
    return _EVIDENCE_PATH.sub(r"\1[redacted]\2[redacted]", value)


class EvidenceAccessLogFilter(logging.Filter):
    """Keep opaque search/evidence identifiers out of HTTP access logs."""

    def filter(self, record: logging.LogRecord) -> bool:
        if isinstance(record.msg, str):
            record.msg = _redact_evidence_path(record.msg)
        if isinstance(record.args, tuple):
            record.args = tuple(
                _redact_evidence_path(value) if isinstance(value, str) else value
                for value in record.args
            )
        elif isinstance(record.args, dict):
            record.args = {
                key: _redact_evidence_path(value) if isinstance(value, str) else value
                for key, value in record.args.items()
            }
        return True


def install_evidence_access_log_filter() -> None:
    logger = logging.getLogger("uvicorn.access")
    if not any(isinstance(item, EvidenceAccessLogFilter) for item in logger.filters):
        logger.addFilter(EvidenceAccessLogFilter())


def uvicorn_log_config() -> dict[str, Any]:
    """Return Uvicorn's standard config with evidence-path redaction attached."""
    config = deepcopy(LOGGING_CONFIG)
    config.setdefault("filters", {})["evidence_access"] = {
        "()": "person_search.privacy.EvidenceAccessLogFilter"
    }
    config["handlers"]["access"]["filters"] = ["evidence_access"]
    return config
