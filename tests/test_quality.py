from __future__ import annotations

import numpy as np
import pytest

from person_search.config import Settings
from person_search.quality import assess_face, normalize_embedding


def test_normalize_embedding() -> None:
    result = normalize_embedding(np.asarray([3.0, 4.0]))
    np.testing.assert_allclose(result, [0.6, 0.8])
    assert np.linalg.norm(result) == pytest.approx(1.0)


@pytest.mark.parametrize("value", [[0.0, 0.0], [np.nan, 1.0], []])
def test_rejects_invalid_embedding(value: list[float]) -> None:
    with pytest.raises(ValueError):
        normalize_embedding(np.asarray(value))


def test_face_quality_accepts_clear_centered_face() -> None:
    checker = np.indices((200, 200)).sum(axis=0) % 2
    frame = np.repeat((checker * 255).astype(np.uint8)[:, :, None], 3, axis=2)
    result = assess_face(
        frame,
        np.asarray([20, 20, 180, 180]),
        np.asarray([[60, 80], [140, 80], [100, 110], [70, 140], [130, 140]]),
        0.99,
        Settings(
            min_enrollment_blur_variance=1,
            min_brightness=0,
            max_brightness=255,
        ),
        enrollment=True,
    )
    assert result.accepted
    assert result.score > 0.7


def test_face_quality_reports_small_dark_blurry_face() -> None:
    result = assess_face(
        np.zeros((100, 100, 3), dtype=np.uint8),
        np.asarray([10, 10, 50, 50]),
        None,
        0.4,
        Settings(),
        enrollment=False,
    )
    assert not result.accepted
    assert {"face_too_small", "detection_score_low", "face_blurry", "face_exposure_bad"} <= set(
        result.reasons
    )


@pytest.mark.parametrize(
    ("short_side", "accepted"),
    [(63, False), (64, True)],
)
def test_search_face_size_boundary(short_side: int, accepted: bool) -> None:
    checker = np.indices((160, 160)).sum(axis=0) % 2
    frame = np.repeat((checker * 255).astype(np.uint8)[:, :, None], 3, axis=2)
    result = assess_face(
        frame,
        np.asarray([20, 20, 20 + short_side, 20 + short_side]),
        None,
        0.99,
        Settings(
            face_detection_threshold=0.0,
            min_search_face_px=64,
            min_search_blur_variance=0.0,
            min_brightness=0.0,
            max_brightness=255.0,
        ),
        enrollment=False,
    )

    assert result.accepted is accepted
    assert result.face_width == short_side
    assert result.face_height == short_side
    assert ("face_too_small" in result.reasons) is not accepted


def test_search_accepts_pose_and_score_that_enrollment_rejects() -> None:
    checker = np.indices((200, 200)).sum(axis=0) % 2
    frame = np.repeat((checker * 255).astype(np.uint8)[:, :, None], 3, axis=2)
    bbox = np.asarray([20, 20, 180, 180])
    side_landmarks = np.asarray(
        [[60, 80], [140, 80], [130, 110], [70, 140], [130, 140]]
    )
    settings = Settings(
        min_enrollment_blur_variance=1,
        min_search_blur_variance=1,
        min_brightness=0,
        max_brightness=255,
    )

    enrollment = assess_face(
        frame, bbox, side_landmarks, 0.5, settings, enrollment=True
    )
    search = assess_face(frame, bbox, side_landmarks, 0.5, settings, enrollment=False)

    assert not enrollment.accepted
    assert {"detection_score_low", "face_yaw_too_large"} <= set(enrollment.reasons)
    assert search.accepted
    assert "face_yaw_too_large" not in search.reasons
