from __future__ import annotations

import asyncio

import pytest
from conftest import FakeFaceBackend, FakePersonDetector, make_face
from fastapi.exceptions import RequestValidationError
from fastapi.testclient import TestClient
from pydantic import ValidationError
from starlette.requests import Request

from person_search.api import (
    _api_key_matches,
    _contains_source_validation_error,
    _redact_validation_text,
    _safe_validation_fields,
    create_app,
)
from person_search.config import Settings
from person_search.domain import SourceConfig
from person_search.errors import PersonSearchError
from person_search.service import SearchManager, _sanitize_source


def _source_error(uri: str, *, source_type: str = "rtsp") -> ValidationError:
    with pytest.raises(ValidationError) as caught:
        SourceConfig.model_validate({"type": source_type, "uri": uri})
    return caught.value


@pytest.mark.parametrize(
    ("uri", "message"),
    [
        ("file:///tmp/video%00.mp4", "percent-encoded control"),
        ("file:///tmp/video%1f.mp4", "percent-encoded control"),
        ("file:///tmp/video%7F.mp4", "percent-encoded control"),
    ],
)
def test_file_uri_rejects_percent_encoded_controls(uri: str, message: str) -> None:
    # A path is eventually handed to a native decoder.  Reject encoded C0/DEL
    # bytes before any platform-specific URI decoding can reinterpret the path.
    with pytest.raises(ValidationError, match=message):
        SourceConfig.model_validate({"type": "file", "uri": uri})


@pytest.mark.parametrize(
    "uri",
    [
        "file://localhost:bad/video.mp4",
        "file://localhost:8554/video.mp4",
        "file://user:secret@localhost/video.mp4",
        "file://localhost/video.mp4?token=secret",
    ],
)
def test_file_uri_rejects_ambiguous_authority(uri: str) -> None:
    with pytest.raises(ValidationError):
        SourceConfig.model_validate({"type": "file", "uri": uri})


def test_rtsp_rejects_multiple_raw_credential_delimiters() -> None:
    error = _source_error("rtsp://user:secret@another-user@camera/live")
    assert "invalid credentials" in str(error).lower()


def test_validation_projection_omits_raw_input_and_context() -> None:
    error = _source_error("rtsp://admin:super;secret@camera:invalid/live")
    fields = _safe_validation_fields(RequestValidationError([*error.errors()]))

    assert fields
    assert all(set(field) == {"loc", "type", "msg"} for field in fields)
    rendered = repr(fields)
    assert "super;secret" not in rendered
    assert "input" not in rendered
    assert "ctx" not in rendered


def test_validation_projection_redacts_attacker_controlled_location_tokens() -> None:
    error = RequestValidationError(
        [
            {
                "loc": ("body", "rtsp://user:secret@camera/live\nleak"),
                "type": "extra_forbidden",
                "msg": "Extra inputs are not permitted",
                "input": {"secret": "value"},
            }
        ]
    )
    fields = _safe_validation_fields(error)
    assert fields[0]["loc"][0] == "body"
    assert fields[0]["loc"][1] == "<redacted-uri>"
    assert "secret" not in repr(fields)


def test_validation_message_redaction_handles_uri_punctuation() -> None:
    text = "bad rtsp://user:pa;ss,word@camera/live; retry"
    redacted = _redact_validation_text(text)
    assert "pa;ss,word" not in redacted
    assert "<redacted-uri>" in redacted


def test_validation_message_redaction_handles_whitespace_inside_malformed_uri() -> None:
    text = "bad rtsp://user:secret\nword@camera/live"
    redacted = _redact_validation_text(text)
    assert "secret" not in redacted
    assert "word" not in redacted


@pytest.mark.parametrize(
    ("fields", "expected"),
    [
        (
            [{"loc": ["body", "source"], "type": "missing", "msg": "Field required"}],
            False,
        ),
        (
            [
                {
                    "loc": ["body", "source"],
                    "type": "missing",
                    "msg": "source URI is required",
                }
            ],
            False,
        ),
        (
            [{"loc": ["body", "source"], "type": "model_type", "msg": "Input should be a dictionary"}],
            True,
        ),
        (
            [{"loc": ["body", "source", "unknown"], "type": "extra_forbidden", "msg": "Extra inputs are not permitted"}],
            True,
        ),
        (
            [{"loc": ["body", "target_id"], "type": "missing", "msg": "Field required"}],
            False,
        ),
    ],
)
def test_source_validation_error_classification(fields: list[dict[str, object]], expected: bool) -> None:
    assert _contains_source_validation_error(fields) is expected


def test_api_validation_fields_are_safe_for_malformed_rtsp() -> None:
    manager = SearchManager(Settings(), FakeFaceBackend([make_face()]), FakePersonDetector())
    settings = Settings()
    with TestClient(create_app(settings, manager)) as client:
        response = client.post(
            "/v1/searches",
            json={
                "target_id": "missing",
                "source": {"type": "rtsp", "uri": "rtsp://admin:secret;pw@camera:bad/live"},
            },
        )

    assert response.status_code == 422
    body = response.json()["detail"]
    assert body["code"] == "invalid_source"
    assert all(set(field) == {"loc", "type", "msg"} for field in body["fields"])
    assert "secret" not in response.text
    assert "input" not in response.text
    assert "ctx" not in response.text


@pytest.mark.parametrize(
    ("payload", "expected_code"),
    [
        ({"target_id": "missing"}, "validation_error"),
        ({"target_id": "missing", "source": None}, "invalid_source"),
        (
            {
                "target_id": "missing",
                "source": {"type": "camera", "device_index": 0, "unexpected": True},
            },
            "invalid_source",
        ),
    ],
)
def test_api_source_shape_errors_keep_stable_classification(
    payload: dict[str, object], expected_code: str
) -> None:
    manager = SearchManager(Settings(), FakeFaceBackend([make_face()]), FakePersonDetector())
    with TestClient(create_app(Settings(), manager)) as client:
        response = client.post("/v1/searches", json=payload)

    assert response.status_code == 422
    assert response.json()["detail"]["code"] == expected_code


def test_evidence_api_key_comparison_supports_unicode_without_500() -> None:
    assert _api_key_matches("密钥-🔐", "密钥-🔐") is True
    assert _api_key_matches("密钥-🔐", "密钥-不同") is False
    assert _api_key_matches(None, "密钥-🔐") is False
    assert _api_key_matches(123, "123") is False  # type: ignore[arg-type]


def test_person_search_error_messages_are_uri_redacted() -> None:
    app = create_app(Settings(), SearchManager(Settings(), FakeFaceBackend(), FakePersonDetector()))
    handler = app.exception_handlers[PersonSearchError]
    response = asyncio.run(
        handler(  # type: ignore[arg-type]
            Request({"type": "http"}),
            PersonSearchError(
                "source rtsp://admin:secret@camera/live failed",
                code="source_failed",
                status_code=503,
            ),
        )
    )
    assert response.status_code == 503
    assert "secret" not in response.body.decode()
    assert "<redacted-uri>" in response.body.decode()


def test_ipv6_rtsp_validation_and_sanitization_remain_supported() -> None:
    source = SourceConfig.model_validate(
        {"type": "rtsp", "uri": "rtsp://admin:secret@[::1]:8554/camera"}
    )
    assert source.uri == "rtsp://admin:secret@[::1]:8554/camera"
    assert "admin" not in repr(source)
    assert "secret" not in repr(source)

    sanitized = _sanitize_source(source)
    assert sanitized.uri == "rtsp://[::1]:8554/***"
