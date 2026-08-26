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


def test_responsive_profile_fills_in_only_the_fields_left_alone() -> None:
    """The profile is a default. Anything the operator set has to survive it."""
    conservative = Settings()
    responsive = Settings(match_profile="responsive")

    assert conservative.tiny_face_evidence_required == 6
    assert conservative.evidence_statistic == "median"
    assert responsive.tiny_face_evidence_required == 4
    assert responsive.tiny_face_evidence_window_seconds == pytest.approx(2.0)
    assert responsive.tiny_face_consistent_votes_required == 3
    assert responsive.tiny_face_detection_threshold == pytest.approx(0.55)
    assert responsive.evidence_statistic == "top_k_mean"

    explicit = Settings(match_profile="responsive", tiny_face_evidence_required=8)
    assert explicit.tiny_face_evidence_required == 8
    assert explicit.evidence_statistic == "top_k_mean"


def test_responsive_profile_is_still_checked_for_tier_consistency() -> None:
    """The profile runs before validation, so its values cannot smuggle in a conflict."""
    with pytest.raises(ValidationError, match="consistent_votes"):
        Settings(match_profile="responsive", tiny_face_evidence_required=2)


def test_full_frame_detection_scales_add_the_large_pass_only_on_cuda() -> None:
    settings = Settings()

    assert settings.full_frame_detection_scales(is_cuda=False) == (128, 640)
    assert settings.full_frame_detection_scales(is_cuda=True) == (128, 640, 1280)
    # A shallow pass is how face_deep_scan_every_n lowers the average cost without
    # giving the stage a second throttle of its own.
    assert settings.full_frame_detection_scales(is_cuda=True, deep=False) == (128, 640)
    # A pinned det size replaces Auto's pair rather than joining it.
    pinned = Settings(face_detection_size=640)
    assert pinned.full_frame_detection_scales(is_cuda=True) == (640, 1280)
    assert Settings(face_detection_extra_scale_cuda=0).full_frame_detection_scales(
        is_cuda=True
    ) == (128, 640)


def test_enrollment_scales_stay_on_auto_whatever_search_pins() -> None:
    assert Settings().enrollment_detection_scales() == (128, 640)
    # Production pins 1280 for search frames. An enrollment photo is a close-up, and
    # a lone large scale upscales it past SCRFD's stride-32 anchors into a miss, so
    # the search setting must not reach this pass.
    assert Settings(face_detection_size=1280).enrollment_detection_scales() == (128, 640)
    assert Settings(enrollment_detection_size=640).enrollment_detection_scales() == (640,)


def test_roi_detection_scale_upsamples_small_crops_and_never_shrinks_large_ones() -> None:
    settings = Settings(roi_face_detection_size=320, roi_face_detection_max_size=640)

    # A far person's head crop is tiny: it still gets upsampled to the floor.
    assert settings.roi_detection_scale(90, 130) == 320
    assert settings.roi_detection_scale(320, 200) == 320
    # A close person's crop is larger than the floor; shrinking it back to 320 would
    # throw away the very pixels the crop existed to keep.
    assert settings.roi_detection_scale(400, 520) == 640
    assert settings.roi_detection_scale(700, 900) == 640
    # Two rungs, not one per crop: every distinct ONNX input shape costs ~30ms of
    # re-planning on CUDA, which dwarfs a 320px crop's own inference.
    scales = {
        settings.roi_detection_scale(size, size) for size in range(64, 900, 7)
    }
    assert scales == {320, 640}
