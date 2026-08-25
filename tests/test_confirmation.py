from __future__ import annotations

import numpy as np
import pytest
from conftest import make_face

from person_search.config import Settings
from person_search.confirmation import (
    TrackConfirmation,
    associate_faces_to_tracks,
    associate_faces_to_tracks_detailed,
    default_face_match_policy,
    tiny_face_match_policy,
)
from person_search.domain import MatchState, Target, TargetView, Track


def _target() -> Target:
    view = TargetView(
        target_id="target-1",
        face_width=100,
        face_height=100,
        detection_score=0.99,
        quality_score=0.9,
        model="fake",
    )
    return Target("target-1", np.asarray([1.0, 0.0], dtype=np.float32), view)


def _track(track_id: int = 7) -> Track:
    return Track(track_id, np.asarray([0, 0, 100, 200], dtype=np.float32), 0.9)


def _face_with_similarity(short_side: int, similarity: float):
    return make_face(
        embedding=(similarity, float(np.sqrt(1.0 - similarity**2))),
        bbox=(10, 10, 10 + short_side, 10 + short_side),
    )


def _face_with_embedding(
    short_side: int,
    embedding: tuple[float, float],
    *,
    detection_score: float = 0.99,
    quality: float = 0.9,
):
    face = make_face(
        embedding=embedding,
        bbox=(10, 10, 10 + short_side, 10 + short_side),
        quality=quality,
    )
    face.detection_score = detection_score
    return face


def test_three_separated_frames_confirm_then_face_timeout_loses() -> None:
    settings = Settings(
        similarity_threshold=0.8,
        evidence_required=3,
        evidence_window_seconds=1.5,
        confirmed_track_grace_seconds=2.0,
        candidate_emit_interval_seconds=0.5,
    )
    matcher = TrackConfirmation(settings)
    face = make_face()
    track = _track()

    states: list[MatchState] = []
    for frame_id, timestamp in enumerate((0.0, 0.2, 0.4)):
        decisions = matcher.process(
            frame_id=frame_id,
            timestamp=timestamp,
            frame_shape=(200, 100, 3),
            tracks=[track],
            faces=[face],
            target=_target(),
        )
        states.extend(item.state for item in decisions)
    assert states.count(MatchState.CONFIRMED) == 1
    assert states[-1] == MatchState.CONFIRMED

    before_timeout = matcher.process(
        frame_id=3,
        timestamp=2.39,
        frame_shape=(200, 100, 3),
        tracks=[track],
        faces=[],
        target=_target(),
    )
    assert not any(item.state == MatchState.LOST for item in before_timeout)
    at_timeout = matcher.process(
        frame_id=4,
        timestamp=2.4,
        frame_shape=(200, 100, 3),
        tracks=[track],
        faces=[],
        target=_target(),
    )
    assert [item.state for item in at_timeout] == [MatchState.LOST]


def test_duplicate_or_too_close_frames_do_not_accumulate() -> None:
    settings = Settings(similarity_threshold=0.8, evidence_required=2)
    matcher = TrackConfirmation(settings)
    track = _track()
    face = make_face()
    first = matcher.process(
        frame_id=1,
        timestamp=0.0,
        frame_shape=(200, 100, 3),
        tracks=[track],
        faces=[face],
        target=_target(),
    )
    second = matcher.process(
        frame_id=1,
        timestamp=0.3,
        frame_shape=(200, 100, 3),
        tracks=[track],
        faces=[face],
        target=_target(),
    )
    assert first[0].evidence_count == 1
    assert not any(item.state == MatchState.CONFIRMED for item in second)


def test_normal_low_similarity_is_not_collected_as_evidence() -> None:
    settings = Settings(similarity_threshold=0.8, evidence_required=2)
    matcher = TrackConfirmation(settings)
    policy = default_face_match_policy(_face_with_similarity(80, 0.79), settings)

    result = matcher.process_with_stats(
        frame_id=1,
        timestamp=0.0,
        frame_shape=(200, 100, 3),
        tracks=[_track()],
        faces=[_face_with_similarity(80, 0.79)],
        target=_target(),
        face_policies={0: policy},
    )

    assert not policy.accepts_observation(0.99, 0.79)
    assert result.decisions == []
    assert result.evidence_collected == 0
    assert matcher.track_progress() == {}


def test_tiny_low_similarity_is_collected_as_negative_evidence() -> None:
    settings = Settings(tiny_face_enabled=True)
    matcher = TrackConfirmation(settings)
    policy = tiny_face_match_policy(settings)

    result = matcher.process_with_stats(
        frame_id=1,
        timestamp=0.0,
        frame_shape=(200, 100, 3),
        tracks=[_track()],
        faces=[_face_with_similarity(48, 0.50)],
        target=_target(),
        face_policies={0: policy},
    )

    assert policy.accepts_observation(0.99, 0.50)
    assert result.decisions == []
    assert result.evidence_collected == 1
    progress = matcher.track_progress()[7]
    assert (progress.observed, progress.required, progress.qualifying) == (1, 6, 0)


def test_eligible_observation_inside_minimum_interval_is_not_collected() -> None:
    settings = Settings(similarity_threshold=0.8, evidence_required=3)
    matcher = TrackConfirmation(settings)
    face = _face_with_similarity(80, 0.90)
    policy = default_face_match_policy(face, settings)

    first = matcher.process_with_stats(
        frame_id=1,
        timestamp=0.0,
        frame_shape=(200, 100, 3),
        tracks=[_track()],
        faces=[face],
        target=_target(),
        face_policies={0: policy},
    )
    too_close = matcher.process_with_stats(
        frame_id=2,
        timestamp=0.1,
        frame_shape=(200, 100, 3),
        tracks=[_track()],
        faces=[face],
        target=_target(),
        face_policies={0: policy},
    )

    assert policy.accepts_observation(0.99, 0.90)
    assert first.evidence_collected == 1
    assert too_close.evidence_collected == 0
    progress = matcher.track_progress()[7]
    assert (progress.observed, progress.required, progress.qualifying) == (1, 3, 1)


def test_face_associates_to_smallest_upper_body_box() -> None:
    face = make_face(bbox=(40, 20, 60, 50))
    large = Track(1, np.asarray([0, 0, 100, 200]), 0.9)
    small = Track(2, np.asarray([30, 10, 70, 100]), 0.9)
    assert associate_faces_to_tracks([face], [large, small]) == {0: 2}


def test_face_below_upper_body_is_not_associated() -> None:
    face = make_face(bbox=(40, 150, 60, 180))
    assert associate_faces_to_tracks([face], [_track()]) == {}


def test_strict_association_includes_the_upper_sixty_percent_boundary() -> None:
    at_boundary = make_face(bbox=(40, 110, 60, 130))
    below_boundary = make_face(bbox=(40, 111, 60, 131))

    assert associate_faces_to_tracks([at_boundary], [_track()]) == {0: 7}
    assert associate_faces_to_tracks([below_boundary], [_track()]) == {}


def test_seated_face_uses_relaxed_association_for_one_containing_track() -> None:
    face = make_face(bbox=(40, 150, 60, 170))

    assert associate_faces_to_tracks_detailed([face], [_track()]) == {0: (7, "person_relaxed")}


def test_relaxed_association_rejects_multiple_containing_tracks() -> None:
    face = make_face(bbox=(40, 150, 60, 170))
    tracks = [
        _track(track_id=7),
        Track(8, np.asarray([20, 0, 80, 220], dtype=np.float32), 0.8),
    ]

    assert associate_faces_to_tracks_detailed([face], tracks) == {}


@pytest.mark.parametrize("short_side", [64, 79])
def test_small_face_requires_four_frames_at_point_six_without_candidate(
    short_side: int,
) -> None:
    settings = Settings(
        similarity_threshold=0.55,
        evidence_required=3,
        small_face_similarity_threshold=0.60,
        small_face_evidence_required=4,
        small_face_evidence_window_seconds=2.0,
    )
    matcher = TrackConfirmation(settings)
    face = _face_with_similarity(short_side, 0.60)
    policy = default_face_match_policy(face, settings)

    assert policy.threshold == 0.60
    assert policy.evidence_required == 4
    assert policy.suppress_candidate

    decisions = []
    for frame_id, timestamp in enumerate((0.0, 0.25, 0.5, 0.75)):
        frame_decisions = matcher.process(
            frame_id=frame_id,
            timestamp=timestamp,
            frame_shape=(200, 100, 3),
            tracks=[_track()],
            faces=[face],
            target=_target(),
            face_policies={0: policy},
        )
        if frame_id < 3:
            assert not any(item.state == MatchState.CONFIRMED for item in frame_decisions)
            assert matcher.active_track_states() == {}
        decisions.extend(frame_decisions)

    assert not any(item.state == MatchState.CANDIDATE for item in decisions)
    confirmed = [item for item in decisions if item.state == MatchState.CONFIRMED]
    assert len(confirmed) == 1
    assert confirmed[0].evidence_count == 4


def test_small_face_rejects_similarity_below_point_six() -> None:
    settings = Settings(
        similarity_threshold=0.55,
        small_face_similarity_threshold=0.60,
        small_face_evidence_required=4,
    )
    matcher = TrackConfirmation(settings)
    face = _face_with_similarity(64, 0.59)
    policy = default_face_match_policy(face, settings)

    decisions = []
    for frame_id, timestamp in enumerate((0.0, 0.25, 0.5, 0.75)):
        decisions.extend(
            matcher.process(
                frame_id=frame_id,
                timestamp=timestamp,
                frame_shape=(200, 100, 3),
                tracks=[_track()],
                faces=[face],
                target=_target(),
                face_policies={0: policy},
            )
        )

    assert decisions == []
    assert matcher.active_track_states() == {}


def test_preferred_size_face_uses_normal_three_frame_policy() -> None:
    settings = Settings(
        similarity_threshold=0.55,
        evidence_required=3,
        small_face_similarity_threshold=0.60,
        small_face_evidence_required=4,
    )
    matcher = TrackConfirmation(settings)
    face = _face_with_similarity(80, 0.56)
    policy = default_face_match_policy(face, settings)

    assert policy.threshold == 0.55
    assert policy.evidence_required == 3
    assert not policy.suppress_candidate

    decisions = []
    for frame_id, timestamp in enumerate((0.0, 0.25, 0.5)):
        frame_decisions = matcher.process(
            frame_id=frame_id,
            timestamp=timestamp,
            frame_shape=(200, 100, 3),
            tracks=[_track()],
            faces=[face],
            target=_target(),
            face_policies={0: policy},
        )
        if frame_id < 2:
            assert not any(item.state == MatchState.CONFIRMED for item in frame_decisions)
        decisions.extend(frame_decisions)

    assert any(item.state == MatchState.CANDIDATE for item in decisions)
    confirmed = [item for item in decisions if item.state == MatchState.CONFIRMED]
    assert len(confirmed) == 1
    assert confirmed[0].evidence_count == 3


@pytest.mark.parametrize("short_side", [48, 55, 56, 63])
def test_tiny_face_policy_is_opt_in(short_side: int) -> None:
    disabled = Settings(tiny_face_enabled=False)
    enabled = Settings(tiny_face_enabled=True)
    face = _face_with_similarity(short_side, 0.70)

    disabled_policy = default_face_match_policy(face, disabled)
    enabled_policy = default_face_match_policy(face, enabled)

    assert disabled_policy.threshold == disabled.similarity_threshold
    assert enabled_policy == tiny_face_match_policy(enabled)
    assert enabled_policy.evidence_required == 6
    assert enabled_policy.consistent_votes_required == 5
    assert enabled_policy.evidence_window_seconds == 3.0
    assert enabled_policy.min_observation_interval_seconds == 0.2
    assert enabled_policy.min_detection_score == 0.65
    assert enabled_policy.min_top1_margin == 0.08
    assert enabled_policy.suppress_candidate
    assert enabled_policy.collect_all_observations
    assert enabled_policy.requires_strict_association
    assert enabled_policy.shadow_eligible


def test_face_below_tiny_minimum_has_no_match_policy() -> None:
    settings = Settings(tiny_face_enabled=True)

    with pytest.raises(ValueError, match="below the effective search size floor"):
        default_face_match_policy(_face_with_similarity(47, 0.90), settings)


def test_tiny_face_collects_six_observations_and_requires_five_consistent_votes() -> None:
    settings = Settings(tiny_face_enabled=True)
    matcher = TrackConfirmation(settings)
    policy = tiny_face_match_policy(settings)
    decisions = []

    # The first low-scoring observation is retained as negative evidence instead of
    # being discarded before the six-frame vote.
    similarities = (0.50, 0.75, 0.75, 0.75, 0.75, 0.75)
    for frame_id, (timestamp, similarity) in enumerate(
        zip((0.0, 0.2, 0.4, 0.6, 0.8, 1.0), similarities, strict=True)
    ):
        face = _face_with_similarity(48, similarity)
        decisions.extend(
            matcher.process(
                frame_id=frame_id,
                timestamp=timestamp,
                frame_shape=(200, 100, 3),
                tracks=[_track()],
                faces=[face],
                target=_target(),
                face_policies={0: policy},
            )
        )

    assert not any(item.state == MatchState.CANDIDATE for item in decisions)
    confirmed = [item for item in decisions if item.state == MatchState.CONFIRMED]
    assert len(confirmed) == 1
    assert confirmed[0].evidence_count == 6
    progress = matcher.track_progress()[7]
    assert (progress.observed, progress.required, progress.qualifying) == (6, 6, 5)


def test_tiny_face_rejects_four_of_six_consistent_votes() -> None:
    settings = Settings(tiny_face_enabled=True)
    matcher = TrackConfirmation(settings)
    policy = tiny_face_match_policy(settings)

    decisions = []
    for frame_id, similarity in enumerate((0.50, 0.50, 0.85, 0.85, 0.85, 0.85)):
        decisions.extend(
            matcher.process(
                frame_id=frame_id,
                timestamp=frame_id * 0.2,
                frame_shape=(200, 100, 3),
                tracks=[_track()],
                faces=[_face_with_similarity(48, similarity)],
                target=_target(),
                face_policies={0: policy},
            )
        )

    assert not any(item.state == MatchState.CONFIRMED for item in decisions)


def test_tiny_face_requires_quality_weighted_aggregate_similarity() -> None:
    settings = Settings(tiny_face_enabled=True)
    matcher = TrackConfirmation(settings)
    policy = tiny_face_match_policy(settings)

    # Every individual target similarity clears 0.64, while the normalized
    # quality-weighted aggregate stays below the stricter 0.68 threshold.
    faces = [_face_with_embedding(48, (0.65, 0.76), quality=1.0) for _ in range(6)]
    decisions = []
    for frame_id, face in enumerate(faces):
        decisions.extend(
            matcher.process(
                frame_id=frame_id,
                timestamp=frame_id * 0.2,
                frame_shape=(200, 100, 3),
                tracks=[_track()],
                faces=[face],
                target=_target(),
                face_policies={0: policy},
            )
        )

    assert not any(item.state == MatchState.CONFIRMED for item in decisions)


def test_tiny_face_rejects_detection_score_below_point_six_five() -> None:
    settings = Settings(tiny_face_enabled=True)
    matcher = TrackConfirmation(settings)
    policy = tiny_face_match_policy(settings)
    assert not policy.accepts_observation(0.64, 1.0)

    decisions = []
    for frame_id in range(6):
        decisions.extend(
            matcher.process(
                frame_id=frame_id,
                timestamp=frame_id * 0.2,
                frame_shape=(200, 100, 3),
                tracks=[_track()],
                faces=[_face_with_embedding(48, (1.0, 0.0), detection_score=0.64)],
                target=_target(),
                face_policies={0: policy},
            )
        )

    assert decisions == []
    assert matcher.active_track_states() == {}


def test_shadow_confirmed_tiny_track_can_transition_to_normal_confirmation() -> None:
    settings = Settings(tiny_face_enabled=True, evidence_required=1)
    matcher = TrackConfirmation(settings)
    tiny_policy = tiny_face_match_policy(settings)

    shadow_decisions = []
    for frame_id in range(6):
        shadow_decisions.extend(
            matcher.process(
                frame_id=frame_id,
                timestamp=frame_id * 0.2,
                frame_shape=(200, 100, 3),
                tracks=[_track()],
                faces=[_face_with_similarity(48, 0.75)],
                target=_target(),
                face_policies={0: tiny_policy},
            )
        )

    shadow_confirmed = [item for item in shadow_decisions if item.state == MatchState.CONFIRMED]
    assert len(shadow_confirmed) == 1
    assert shadow_confirmed[0].shadow
    assert matcher.active_track_states()[7][0] == MatchState.CONFIRMED

    close_face = _face_with_similarity(80, 0.75)
    close_decisions = matcher.process(
        frame_id=6,
        timestamp=1.2,
        frame_shape=(200, 100, 3),
        tracks=[_track()],
        faces=[close_face],
        target=_target(),
        face_policies={0: default_face_match_policy(close_face, settings)},
    )

    assert any(item.state == MatchState.LOST and item.shadow for item in close_decisions)
    assert any(item.state == MatchState.CONFIRMED and not item.shadow for item in close_decisions)


def test_confirmed_track_keeps_last_matching_similarity_when_face_drops_below_threshold() -> None:
    settings = Settings(similarity_threshold=0.55, evidence_required=3)
    matcher = TrackConfirmation(settings)
    face = _face_with_similarity(80, 0.70)
    policy = default_face_match_policy(face, settings)

    for frame_id, timestamp in enumerate((0.0, 0.25, 0.5)):
        matcher.process(
            frame_id=frame_id,
            timestamp=timestamp,
            frame_shape=(200, 100, 3),
            tracks=[_track()],
            faces=[face],
            target=_target(),
            face_policies={0: policy},
        )

    assert matcher.active_track_states()[7][0] == MatchState.CONFIRMED
    assert matcher.active_track_states()[7][1] == pytest.approx(0.70, abs=1e-6)

    # A glancing frame far below the threshold is not a sighting: it must not be
    # reported as the track's similarity, or the preview shows "FOUND 0.30".
    matcher.process(
        frame_id=3,
        timestamp=0.75,
        frame_shape=(200, 100, 3),
        tracks=[_track()],
        faces=[_face_with_similarity(80, 0.30)],
        target=_target(),
        face_policies={0: policy},
    )

    assert matcher.active_track_states()[7][1] == pytest.approx(0.70, abs=1e-6)


def test_shadow_confirmed_tiny_track_is_lost_after_grace_when_similarity_stays_low() -> None:
    settings = Settings(tiny_face_enabled=True, confirmed_track_grace_seconds=2.0)
    matcher = TrackConfirmation(settings)
    policy = tiny_face_match_policy(settings)

    for frame_id in range(6):
        matcher.process(
            frame_id=frame_id,
            timestamp=frame_id * 0.2,
            frame_shape=(200, 100, 3),
            tracks=[_track()],
            faces=[_face_with_similarity(48, 0.75)],
            target=_target(),
            face_policies={0: policy},
        )

    assert matcher.active_track_states()[7][0] == MatchState.CONFIRMED

    # Sub-threshold tiny samples still enter the evidence window, but they must not
    # keep refreshing the grace timer, or a shadow track never reports LOST.
    decisions = []
    for offset, frame_id in enumerate(range(6, 24)):
        decisions.extend(
            matcher.process(
                frame_id=frame_id,
                timestamp=1.2 + offset * 0.2,
                frame_shape=(200, 100, 3),
                tracks=[_track()],
                faces=[_face_with_similarity(48, 0.30)],
                target=_target(),
                face_policies={0: policy},
            )
        )

    lost = [item for item in decisions if item.state == MatchState.LOST]
    assert len(lost) == 1
    assert lost[0].shadow
    assert matcher.active_track_states() == {}


def test_candidate_state_disappears_when_evidence_expires() -> None:
    matcher = TrackConfirmation(
        Settings(similarity_threshold=0.8, evidence_required=3, evidence_window_seconds=1.0)
    )
    track = _track()
    matcher.process(
        frame_id=1,
        timestamp=0.0,
        frame_shape=(200, 100, 3),
        tracks=[track],
        faces=[make_face()],
        target=_target(),
    )
    assert matcher.active_track_states()[track.track_id][0] == MatchState.CANDIDATE

    matcher.process(
        frame_id=2,
        timestamp=1.01,
        frame_shape=(200, 100, 3),
        tracks=[track],
        faces=[],
        target=_target(),
    )
    assert matcher.active_track_states() == {}


def test_saturated_tiny_evidence_reports_qualifying_shortfall() -> None:
    """The field failure: 6/6 banked, one sample qualifying, never confirmed.

    collect_all_observations banks sub-threshold samples too, so `observed`
    pins at `required` forever. Only `qualifying` and `median_similarity`
    reveal that the track is nowhere near the gate.
    """
    settings = Settings(tiny_face_enabled=True)
    matcher = TrackConfirmation(settings)
    policy = tiny_face_match_policy(settings)
    decisions = []
    # One lucky frame above 0.64, five at the ~0.41 the 50px field face produced.
    similarities = (0.70, 0.41, 0.40, 0.42, 0.39, 0.41)
    for frame_id, (timestamp, similarity) in enumerate(
        zip((0.0, 0.2, 0.4, 0.6, 0.8, 1.0), similarities, strict=True)
    ):
        decisions.extend(
            matcher.process(
                frame_id=frame_id,
                timestamp=timestamp,
                frame_shape=(200, 100, 3),
                tracks=[_track()],
                faces=[_face_with_similarity(50, similarity)],
                target=_target(),
                face_policies={0: policy},
            )
        )

    assert not any(item.state == MatchState.CONFIRMED for item in decisions)
    progress = matcher.track_progress()[7]
    assert progress.observed == progress.required == 6
    assert progress.qualifying == 1
    assert progress.threshold == pytest.approx(settings.tiny_face_similarity_threshold)
    assert progress.median_similarity == pytest.approx(0.41, abs=1e-2)
    assert progress.best_similarity == pytest.approx(0.70, abs=1e-2)


def test_track_progress_is_empty_before_any_evidence() -> None:
    matcher = TrackConfirmation(Settings())
    matcher.process(
        frame_id=1,
        timestamp=0.0,
        frame_shape=(200, 100, 3),
        tracks=[_track()],
        faces=[],
        target=_target(),
    )
    assert matcher.track_progress() == {}


def _feed(
    matcher: TrackConfirmation,
    settings: Settings,
    samples: tuple[tuple[float, int, float], ...],
    *,
    start_frame: int = 0,
) -> list:
    """Drive the matcher with (timestamp, short_side, similarity) samples.

    The policy is resolved from each face's own size, exactly as service._run
    does, so a track that grows across a tier boundary is exercised end to end.
    """
    decisions = []
    for offset, (timestamp, short_side, similarity) in enumerate(samples):
        face = _face_with_similarity(short_side, similarity)
        decisions.extend(
            matcher.process(
                frame_id=start_frame + offset,
                timestamp=timestamp,
                frame_shape=(200, 100, 3),
                tracks=[_track()],
                faces=[face],
                target=_target(),
                face_policies={0: default_face_match_policy(face, settings)},
            )
        )
    return decisions


def test_track_confirms_on_the_normal_tier_after_its_face_grows_past_the_far_bar() -> None:
    """A passenger walking closer must be re-judged, not held to the far-face bar.

    The policy used to latch the strictest tier a track ever saw, so someone first
    seen at 55px kept 0.64/6-frames/aggregate-0.68 even at 120px, where the normal
    tier only asks for 0.55 over 3 frames.
    """
    settings = Settings(tiny_face_enabled=True)
    matcher = TrackConfirmation(settings)

    decisions = _feed(
        matcher,
        settings,
        (
            (0.0, 55, 0.60),
            (0.2, 55, 0.60),
            (0.4, 120, 0.70),
            (0.6, 120, 0.70),
            (0.8, 120, 0.70),
        ),
    )

    confirmed = [item for item in decisions if item.state == MatchState.CONFIRMED]
    assert len(confirmed) == 1
    assert confirmed[0].evidence_count == settings.evidence_required
    progress = matcher.track_progress()[7]
    assert progress.threshold == pytest.approx(settings.similarity_threshold)
    assert progress.required == settings.evidence_required
    # The far-face aggregate gate does not apply to the normal tier.
    assert progress.aggregate_threshold is None


def test_relaxing_the_tier_clears_the_evidence_window() -> None:
    """Samples taken under different thresholds must never share one window."""
    settings = Settings(tiny_face_enabled=True)
    matcher = TrackConfirmation(settings)

    _feed(matcher, settings, ((0.0, 55, 0.60), (0.2, 55, 0.60)))
    assert matcher.track_progress()[7].observed == 2

    _feed(matcher, settings, ((0.4, 120, 0.70),), start_frame=2)
    progress = matcher.track_progress()[7]
    assert progress.observed == 1
    assert progress.median_similarity == pytest.approx(0.70, abs=1e-2)


def test_track_progress_reports_the_far_face_aggregate_gate() -> None:
    """The aggregate gate is invisible unless it is reported next to the median."""
    settings = Settings(tiny_face_enabled=True)
    matcher = TrackConfirmation(settings)
    policy = tiny_face_match_policy(settings)

    for frame_id, timestamp in enumerate((0.0, 0.2, 0.4)):
        matcher.process(
            frame_id=frame_id,
            timestamp=timestamp,
            frame_shape=(200, 100, 3),
            tracks=[_track()],
            faces=[_face_with_similarity(55, 0.66)],
            target=_target(),
            face_policies={0: policy},
        )

    reported = matcher.track_progress(_target())[7]
    assert reported.aggregate_threshold == pytest.approx(
        settings.tiny_face_aggregate_similarity_threshold
    )
    assert reported.aggregate_similarity == pytest.approx(0.66, abs=1e-2)
    # Without a target the rest of the report still works; only the gate is unknown.
    assert matcher.track_progress()[7].aggregate_similarity is None

