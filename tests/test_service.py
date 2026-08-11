from __future__ import annotations

import numpy as np
import pytest
from conftest import FakeFaceBackend, FakePersonDetector, make_face

from person_search.config import Settings
from person_search.errors import EnrollmentError, PersonSearchError
from person_search.service import SearchManager


def test_enroll_requires_exactly_one_face() -> None:
    frame = np.zeros((200, 200, 3), dtype=np.uint8)
    no_face = SearchManager(Settings(), FakeFaceBackend([]), FakePersonDetector())
    with pytest.raises(EnrollmentError, match="no face"):
        no_face.enroll(frame)

    multiple = SearchManager(
        Settings(), FakeFaceBackend([make_face(), make_face(bbox=(100, 20, 160, 80))]), FakePersonDetector()
    )
    with pytest.raises(EnrollmentError, match="exactly one"):
        multiple.enroll(frame)


def test_enroll_rejects_quality_failure() -> None:
    manager = SearchManager(Settings(), FakeFaceBackend([make_face(accepted=False)]), FakePersonDetector())
    with pytest.raises(EnrollmentError, match="face quality"):
        manager.enroll(np.zeros((200, 200, 3), dtype=np.uint8))


def test_target_embedding_is_normalized_and_can_be_deleted() -> None:
    manager = SearchManager(Settings(), FakeFaceBackend([make_face()]), FakePersonDetector())
    target_view = manager.enroll(np.zeros((200, 200, 3), dtype=np.uint8), " Alice ")
    assert target_view.name == "Alice"
    assert manager.get_target(target_view.target_id).name == "Alice"
    assert np.linalg.norm(manager.get_target(target_view.target_id).embedding) == pytest.approx(1.0)
    assert manager.delete_target(target_view.target_id)
    assert not manager.delete_target(target_view.target_id)
    with pytest.raises(PersonSearchError) as caught:
        manager.get_target(target_view.target_id)
    assert caught.value.code == "target_not_found"


def test_enroll_rejects_blank_target_name() -> None:
    manager = SearchManager(Settings(), FakeFaceBackend([make_face()]), FakePersonDetector())
    with pytest.raises(EnrollmentError, match="target name"):
        manager.enroll(np.zeros((200, 200, 3), dtype=np.uint8), "   ")
