from __future__ import annotations

import pytest
from pydantic import ValidationError

from person_search.config import Settings


def test_face_detection_uses_insightface_auto_mode_by_default() -> None:
    assert Settings().face_detection_size == 0


def test_rknn_backend_settings_are_explicit() -> None:
    settings = Settings(
        inference_backend="rknn",
        rknn_person_model="/models/person.rknn",
        rknn_face_detection_model="/models/scrfd.rknn",
        rknn_face_recognition_model="/models/arcface.rknn",
        rknn_core_mask=7,
    )
    assert settings.inference_backend == "rknn"
    assert settings.rknn_core_mask == 7
    assert str(settings.rknn_person_model).endswith("person.rknn")


def test_rknn_blank_checksums_are_optional_and_valid_checksums_are_normalized() -> None:
    settings = Settings(
        rknn_person_sha256="",
        rknn_face_detection_sha256=" " + "A" * 64 + " ",
    )

    assert settings.rknn_person_sha256 is None
    assert settings.rknn_face_detection_sha256 == "a" * 64


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("rknn_core_mask", 8),
        ("person_input_width", 418),
        ("person_input_height", 0),
        ("rknn_person_sha256", "not-a-checksum"),
    ],
)
def test_rknn_settings_reject_invalid_hardware_boundaries(field: str, value: object) -> None:
    with pytest.raises(ValidationError):
        Settings(**{field: value})
