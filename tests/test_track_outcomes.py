from __future__ import annotations

import numpy as np
import pytest
from conftest import make_face

from person_search.config import Settings
from person_search.confirmation import (
    GATE_AGGREGATE_LOW,
    GATE_INSUFFICIENT_SAMPLES,
    GATE_VOTES_LOW,
    GATE_WINDOW_STATISTIC_LOW,
    TrackConfirmation,
    TrackOutcome,
    _Evidence,
    _TrackState,
    _window_statistic,
    default_face_match_policy,
)
from person_search.domain import MatchState, Target, TargetView, Track

# The far-face tier is the only one that banks sub-threshold samples, so it is the
# only tier on which the statistic, votes and aggregate gates can fail at all --
# every other tier refuses those observations before they reach the window.
TINY_PX = 55
NORMAL_PX = 90


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


def _face(short_side: int, similarity: float):
    return make_face(
        embedding=(similarity, float(np.sqrt(1.0 - similarity**2))),
        bbox=(10, 10, 10 + short_side, 10 + short_side),
    )


def _tiny_settings(**overrides) -> Settings:
    base: dict[str, object] = {
        "tiny_face_enabled": True,
        "tiny_face_shadow_mode": False,
        "tiny_face_evidence_required": 6,
        "tiny_face_evidence_window_seconds": 3.0,
        "tiny_face_consistent_votes_required": 5,
        "tiny_face_similarity_threshold": 0.64,
        "tiny_face_aggregate_similarity_threshold": 0.68,
        "confirmed_track_grace_seconds": 2.0,
    }
    base.update(overrides)
    return Settings(**base)


def _drive(
    matcher: TrackConfirmation,
    similarities: list[float],
    *,
    short_side: int,
    step: float = 0.25,
) -> None:
    """Feed one sample per frame onto a single track.

    Resolves the size tier per frame against the track's current tier exactly as
    ``SearchSession._run`` does. Calling ``process`` without policies would silently
    judge a 55px face by the normal tier, and the far-face gates -- the only ones
    that can fail on anything but sample count -- would never be exercised.
    """
    track = _track()
    for frame_id, similarity in enumerate(similarities):
        face = _face(short_side, similarity)
        policy = default_face_match_policy(
            face, matcher.settings, matcher.tier_of(track.track_id)
        )
        matcher.process(
            frame_id=frame_id,
            timestamp=frame_id * step,
            frame_shape=(200, 100, 3),
            tracks=[track],
            faces=[face],
            target=_target(),
            face_policies={0: policy},
        )


def _abandon(matcher: TrackConfirmation, at: float) -> list[TrackOutcome]:
    """Drop the track and run far enough past the window and grace to delete it."""
    return matcher.process_with_stats(
        frame_id=999,
        timestamp=at,
        frame_shape=(200, 100, 3),
        tracks=[],
        faces=[],
        target=_target(),
    ).outcomes


def _post_mortem(settings: Settings, similarities: list[float], short_side: int) -> TrackOutcome:
    matcher = TrackConfirmation(settings)
    _drive(matcher, similarities, short_side=short_side)
    outcomes = _abandon(matcher, at=len(similarities) * 0.25 + 10.0)
    assert len(outcomes) == 1
    return outcomes[0]


def test_post_mortem_reports_the_window_the_track_actually_had() -> None:
    """The deque is empty at deletion; the report must not be.

    This is the whole point of keeping a best-attempt snapshot: an unconfirmed
    track is only deleted *after* its window has expired, so reading the deque at
    that moment would blame every single failure on "not enough samples" and
    report zero banked for all of them.
    """
    outcome = _post_mortem(_tiny_settings(), [0.70] * 3, TINY_PX)
    assert outcome.confirmed is False
    assert outcome.banked == 3
    assert outcome.sampled == 3
    assert outcome.required == 6
    assert outcome.blocking_gate == GATE_INSUFFICIENT_SAMPLES
    assert outcome.best_similarity == pytest.approx(0.70, abs=1e-6)
    assert outcome.dwell_seconds == pytest.approx(0.5, abs=1e-6)


def test_gate_insufficient_samples_on_the_normal_tier() -> None:
    settings = Settings(
        similarity_threshold=0.60,
        evidence_required=3,
        evidence_window_seconds=1.5,
        confirmed_track_grace_seconds=2.0,
    )
    outcome = _post_mortem(settings, [0.90, 0.90], NORMAL_PX)
    assert outcome.blocking_gate == GATE_INSUFFICIENT_SAMPLES
    assert (outcome.banked, outcome.required) == (2, 3)


def test_gate_window_statistic_low() -> None:
    """Enough samples, but the window's own statistic never clears the bar."""
    outcome = _post_mortem(_tiny_settings(), [0.50] * 6, TINY_PX)
    assert outcome.banked == 6
    assert outcome.blocking_gate == GATE_WINDOW_STATISTIC_LOW
    assert outcome.window_similarity == pytest.approx(0.50, abs=1e-6)
    assert outcome.threshold == pytest.approx(0.64)


def test_gate_votes_low() -> None:
    """The statistic passes on a few good looks; too few samples individually do."""
    settings = _tiny_settings(evidence_statistic="top_k_mean", evidence_top_k=3)
    outcome = _post_mortem(settings, [0.70, 0.70, 0.70, 0.50, 0.50, 0.50], TINY_PX)
    assert outcome.banked == 6
    assert outcome.window_similarity == pytest.approx(0.70, abs=1e-6)
    assert outcome.qualifying == 3
    assert outcome.blocking_gate == GATE_VOTES_LOW


def test_gate_aggregate_low() -> None:
    """Every sample clears 0.64; the pooled embedding still misses 0.68.

    Reported as its own gate because it was previously a local variable inside the
    verdict: the panel could show a passing window statistic and the track would
    never confirm, with nothing anywhere explaining why.
    """
    outcome = _post_mortem(_tiny_settings(), [0.65] * 6, TINY_PX)
    assert outcome.banked == 6
    assert outcome.qualifying == 6
    assert outcome.window_similarity == pytest.approx(0.65, abs=1e-6)
    assert outcome.blocking_gate == GATE_AGGREGATE_LOW
    assert outcome.aggregate_similarity == pytest.approx(0.65, abs=1e-4)
    assert outcome.aggregate_threshold == pytest.approx(0.68)


def test_each_gate_is_reachable_and_distinct() -> None:
    """Guards the ordering: a gate that can never be reported is a dead branch."""
    settings = _tiny_settings(evidence_statistic="top_k_mean", evidence_top_k=3)
    reported = {
        _post_mortem(_tiny_settings(), [0.70] * 3, TINY_PX).blocking_gate,
        _post_mortem(_tiny_settings(), [0.50] * 6, TINY_PX).blocking_gate,
        _post_mortem(settings, [0.70, 0.70, 0.70, 0.50, 0.50, 0.50], TINY_PX).blocking_gate,
        _post_mortem(_tiny_settings(), [0.65] * 6, TINY_PX).blocking_gate,
    }
    assert reported == {
        GATE_INSUFFICIENT_SAMPLES,
        GATE_WINDOW_STATISTIC_LOW,
        GATE_VOTES_LOW,
        GATE_AGGREGATE_LOW,
    }


def test_no_gate_means_confirmed_and_vice_versa() -> None:
    """The verdict is exactly `gate is None`, on every frame of a real sequence."""
    settings = Settings(
        similarity_threshold=0.60,
        evidence_required=3,
        evidence_window_seconds=1.5,
    )
    matcher = TrackConfirmation(settings)
    track = _track()
    for frame_id in range(4):
        decisions = matcher.process(
            frame_id=frame_id,
            timestamp=frame_id * 0.25,
            frame_shape=(200, 100, 3),
            tracks=[track],
            faces=[_face(NORMAL_PX, 0.90)],
            target=_target(),
        )
        state = matcher._states[track.track_id]
        gate = matcher._evaluate(state, _target()).gate
        assert (gate is None) is matcher._is_confirmed(state, _target())
        if any(item.state == MatchState.CONFIRMED for item in decisions):
            assert gate is None


def test_confirmed_track_reports_time_to_confirm() -> None:
    settings = Settings(
        similarity_threshold=0.60,
        evidence_required=3,
        evidence_window_seconds=1.5,
    )
    matcher = TrackConfirmation(settings)
    track = _track()
    outcomes: list[TrackOutcome] = []
    for frame_id in range(3):
        outcomes.extend(
            matcher.process_with_stats(
                frame_id=frame_id,
                timestamp=frame_id * 0.25,
                frame_shape=(200, 100, 3),
                tracks=[track],
                faces=[_face(NORMAL_PX, 0.90)],
                target=_target(),
            ).outcomes
        )
    assert len(outcomes) == 1
    assert outcomes[0].confirmed is True
    # First sample at t=0, third at t=0.5 -- the wall-clock cost of the quorum.
    assert outcomes[0].time_to_confirm_seconds == pytest.approx(0.5, abs=1e-6)
    assert outcomes[0].sampled == 3


def test_tracks_that_never_banked_evidence_report_nothing() -> None:
    """Otherwise every passer-by drowns the one track that got close."""
    matcher = TrackConfirmation(Settings(similarity_threshold=0.60))
    track = _track()
    matcher.process(
        frame_id=0,
        timestamp=0.0,
        frame_shape=(200, 100, 3),
        tracks=[track],
        faces=[],
        target=_target(),
    )
    assert _abandon(matcher, at=10.0) == []


def _abandon_result(matcher: TrackConfirmation, at: float):
    return matcher.process_with_stats(
        frame_id=999,
        timestamp=at,
        frame_shape=(200, 100, 3),
        tracks=[],
        faces=[],
        target=_target(),
    )


def _departure_settings(**overrides) -> Settings:
    base: dict[str, object] = {
        "similarity_threshold": 0.60,
        "evidence_required": 3,
        "evidence_window_seconds": 1.5,
        "confirmed_track_grace_seconds": 2.0,
        "departure_adjudication_enabled": True,
        "departure_min_samples": 2,
        "departure_similarity_margin": 0.05,
    }
    base.update(overrides)
    return Settings(**base)


def test_departure_adjudication_is_off_by_default() -> None:
    """The shipped default must not turn a short dwell into a hit on its own."""
    assert Settings().departure_adjudication_enabled is False
    settings = _departure_settings(departure_adjudication_enabled=False)
    matcher = TrackConfirmation(settings)
    _drive(matcher, [0.90, 0.90], short_side=NORMAL_PX)
    result = _abandon_result(matcher, at=10.0)
    assert result.decisions == []
    assert [outcome.confirmed for outcome in result.outcomes] == [False]


def test_departure_adjudication_emits_a_paired_shadow_hit() -> None:
    matcher = TrackConfirmation(_departure_settings())
    _drive(matcher, [0.90, 0.90], short_side=NORMAL_PX)
    result = _abandon_result(matcher, at=10.0)
    # The track is already gone, so the pair has to close in the same frame or the
    # console keeps showing a live lead for someone who has left.
    assert [(item.state, item.shadow) for item in result.decisions] == [
        (MatchState.CONFIRMED, True),
        (MatchState.LOST, True),
    ]
    # The window had expired by deletion time; the count must come from the
    # attempt that was actually judged, not from the empty deque.
    assert [item.evidence_count for item in result.decisions] == [2, 2]
    assert result.outcomes[0].confirmed is True
    assert result.outcomes[0].shadow is True


def test_departure_adjudication_demands_a_margin_over_the_threshold() -> None:
    """Fewer frames are paid for with a higher bar on the frames there were."""
    matcher = TrackConfirmation(_departure_settings())
    _drive(matcher, [0.62, 0.62], short_side=NORMAL_PX)
    result = _abandon_result(matcher, at=10.0)
    assert result.decisions == []
    assert result.outcomes[0].confirmed is False


def test_departure_adjudication_respects_the_minimum_sample_count() -> None:
    matcher = TrackConfirmation(_departure_settings(departure_min_samples=3))
    _drive(matcher, [0.90, 0.90], short_side=NORMAL_PX)
    assert _abandon_result(matcher, at=10.0).decisions == []


def test_departure_adjudication_never_rescues_a_track_that_scored_too_low() -> None:
    """Only a missing-frames failure is eligible.

    A track that *was* sampled enough and still failed on the statistic, the votes
    or the aggregate has already been judged. Re-judging it on the way out would be
    an uncalibrated threshold cut wearing a different name -- so the far-face track
    that banks a full window at 0.65 against a 0.68 aggregate stays unconfirmed.
    """
    settings = _tiny_settings(
        departure_adjudication_enabled=True,
        departure_min_samples=2,
        departure_similarity_margin=0.0,
    )
    matcher = TrackConfirmation(settings)
    _drive(matcher, [0.65] * 6, short_side=TINY_PX)
    result = _abandon_result(matcher, at=10.0)
    assert result.outcomes[0].blocking_gate == GATE_AGGREGATE_LOW
    assert result.decisions == []
    assert result.outcomes[0].confirmed is False


def _legacy_is_confirmed(matcher: TrackConfirmation, state, target: Target) -> bool:
    """The verdict exactly as it read before ``_evaluate`` existed.

    Kept verbatim so the refactor that split the gates apart for reporting can be
    shown not to have moved any of them. Instrumentation that changes a verdict is
    not instrumentation.
    """
    required = matcher._policy_required(state)
    if len(state.evidence) < required:
        return False
    similarities = [item.similarity for item in state.evidence]
    threshold = matcher._policy_threshold(state)
    statistic, top_k = matcher._policy_statistic(state)
    if _window_statistic(similarities, statistic, top_k) < threshold:
        return False
    policy = state.policy
    votes_required = policy.consistent_votes_required if policy is not None else 0
    if votes_required and sum(value >= threshold for value in similarities) < votes_required:
        return False
    if policy is not None and policy.aggregate_threshold is not None:
        aggregate_similarity = matcher._aggregate_similarity(state, target)
        if aggregate_similarity is None:
            return False
        if aggregate_similarity < policy.aggregate_threshold:
            return False
    return True


def test_evaluate_matches_the_pre_refactor_verdict_over_a_random_sweep() -> None:
    rng = np.random.default_rng(20260828)
    target = _target()
    settings = _tiny_settings(evidence_statistic="top_k_mean", evidence_top_k=3)
    policies = [
        default_face_match_policy(_face(TINY_PX, 0.7), settings, None),
        default_face_match_policy(_face(70, 0.7), settings, None),
        default_face_match_policy(_face(NORMAL_PX, 0.7), settings, None),
    ]
    agreed = 0
    confirmed_seen = 0
    for _ in range(2000):
        matcher = TrackConfirmation(settings)
        state = _TrackState()
        state.policy = policies[int(rng.integers(len(policies)))]
        for index in range(int(rng.integers(0, 9))):
            similarity = float(rng.uniform(0.3, 0.95))
            state.evidence.append(
                _Evidence(
                    frame_id=index,
                    timestamp=index * 0.25,
                    similarity=similarity,
                    quality=float(rng.uniform(0.1, 1.0)),
                    embedding=np.asarray(
                        [similarity, float(np.sqrt(max(0.0, 1.0 - similarity**2)))],
                        dtype=np.float32,
                    ),
                )
            )
        matcher._states[1] = state
        expected = _legacy_is_confirmed(matcher, state, target)
        assert (matcher._evaluate(state, target).gate is None) == expected
        agreed += 1
        confirmed_seen += expected
    assert agreed == 2000
    # A sweep that never confirms would agree trivially with anything.
    assert 0 < confirmed_seen < 2000
