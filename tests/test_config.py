from __future__ import annotations

import pytest
from pydantic import ValidationError

from person_search.config import HARD_MIN_SEARCH_FACE_PX, Settings


def test_face_size_tiers_must_be_ordered() -> None:
    with pytest.raises(ValidationError, match="face size tiers"):
        Settings(tiny_face_min_px=64, min_search_face_px=64)


def test_tiny_consistent_votes_cannot_exceed_evidence_count() -> None:
    with pytest.raises(ValidationError, match="consistent_votes"):
        Settings(tiny_face_evidence_required=4, tiny_face_consistent_votes_required=5)


def test_tiny_face_minimum_has_a_non_configurable_48px_floor() -> None:
    assert HARD_MIN_SEARCH_FACE_PX == 48
    with pytest.raises(ValidationError, match="greater than or equal to 48"):
        Settings(tiny_face_min_px=47)

    settings = Settings(tiny_face_enabled=True, tiny_face_min_px=48)
    assert settings.effective_search_min_face_px == 48


def test_effective_search_minimum_defends_against_unvalidated_runtime_values() -> None:
    settings = Settings.model_construct(tiny_face_enabled=True, tiny_face_min_px=1)

    assert settings.effective_search_min_face_px == HARD_MIN_SEARCH_FACE_PX
