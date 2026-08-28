from __future__ import annotations

import logging

from person_search.privacy import EvidenceAccessLogFilter, uvicorn_log_config


def test_evidence_access_log_filter_redacts_opaque_ids() -> None:
    record = logging.LogRecord(
        "uvicorn.access",
        logging.INFO,
        __file__,
        1,
        '%s - "%s %s HTTP/%s" %d',
        (
            "127.0.0.1:12345",
            "GET",
            "/v1/searches/private-search/evidence/private-evidence?variant=face_crop",
            "1.1",
            200,
        ),
        None,
    )

    assert EvidenceAccessLogFilter().filter(record) is True
    rendered = record.getMessage()
    assert "private-search" not in rendered
    assert "private-evidence" not in rendered
    assert "/v1/searches/[redacted]/evidence/[redacted]?variant=face_crop" in rendered


def test_evidence_access_log_filter_leaves_other_routes_unchanged() -> None:
    record = logging.LogRecord(
        "uvicorn.access",
        logging.INFO,
        __file__,
        1,
        "%s",
        ("GET /healthz",),
        None,
    )

    EvidenceAccessLogFilter().filter(record)

    assert record.getMessage() == "GET /healthz"


def test_console_entrypoint_keeps_the_filter_after_uvicorn_configures_logging() -> None:
    config = uvicorn_log_config()

    assert config["filters"]["evidence_access"] == {
        "()": "person_search.privacy.EvidenceAccessLogFilter"
    }
    assert config["handlers"]["access"]["filters"] == ["evidence_access"]
