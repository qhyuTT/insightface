from __future__ import annotations

from person_search.config import Settings


def test_face_detection_uses_insightface_auto_mode_by_default() -> None:
    assert Settings().face_detection_size == 0
