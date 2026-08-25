from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field

import numpy as np

from .config import HARD_MIN_SEARCH_FACE_PX, Settings
from .domain import FaceObservation, MatchState, Target, Track


@dataclass(frozen=True, slots=True)
class MatchDecision:
    state: MatchState
    track_id: int
    bbox: np.ndarray
    similarity: float
    quality: float
    evidence_count: int
    association: str = "person_strict"
    shadow: bool = False


@dataclass(frozen=True, slots=True)
class ConfirmationResult:
    decisions: list[MatchDecision]
    evidence_collected: int = 0


@dataclass(frozen=True, slots=True)
class TrackProgress:
    """How close one track is to confirmation, and why it is not there yet.

    ``observed`` counts every banked sample; under ``collect_all_observations``
    that includes sub-threshold ones, so it saturates at ``required`` and says
    nothing about progress. ``qualifying`` and ``median_similarity`` are what
    the confirmation gate actually reads.
    """

    observed: int
    required: int
    qualifying: int
    threshold: float
    median_similarity: float | None
    best_similarity: float | None


@dataclass(frozen=True, slots=True)
class FaceMatchPolicy:
    threshold: float
    evidence_required: int
    evidence_window_seconds: float
    suppress_candidate: bool = False
    aggregate_threshold: float | None = None
    consistent_votes_required: int = 0
    min_observation_interval_seconds: float = 0.2
    min_detection_score: float = 0.0
    # Enforced per frame by the caller (see service._run) before an observation
    # reaches process(); it deliberately takes no part in the window verdict.
    min_top1_margin: float = 0.0
    collect_all_observations: bool = False
    requires_strict_association: bool = False
    shadow_eligible: bool = False

    def accepts_observation(self, detection_score: float, similarity: float) -> bool:
        return detection_score >= self.min_detection_score and (
            similarity >= self.threshold or self.collect_all_observations
        )


@dataclass(slots=True)
class _Evidence:
    frame_id: int
    timestamp: float
    similarity: float
    quality: float
    embedding: np.ndarray


@dataclass(slots=True)
class _TrackState:
    evidence: deque[_Evidence] = field(default_factory=deque)
    confirmed: bool = False
    shadow_confirmed: bool = False
    last_track_seen: float = 0.0
    last_face_seen: float = 0.0
    last_candidate_emit: float = -1e9
    last_bbox: np.ndarray = field(default_factory=lambda: np.zeros(4, dtype=np.float32))
    last_similarity: float = -1.0
    last_quality: float = 0.0
    last_association: str = "person_strict"
    policy: FaceMatchPolicy | None = None


class TrackConfirmation:
    def __init__(self, settings: Settings):
        self.settings = settings
        self._states: dict[int, _TrackState] = {}

    def reset(self) -> None:
        self._states.clear()

    def active_track_states(self) -> dict[int, tuple[MatchState, float]]:
        return {
            track_id: (
                MatchState.CONFIRMED
                if state.confirmed or state.shadow_confirmed
                else MatchState.CANDIDATE,
                state.last_similarity,
            )
            for track_id, state in self._states.items()
            if state.confirmed
            or state.shadow_confirmed
            or (
                state.evidence
                and not (state.policy is not None and state.policy.suppress_candidate)
            )
        }

    def track_progress(self) -> dict[int, TrackProgress]:
        """Return per-track confirmation progress for active tracks."""
        progress: dict[int, TrackProgress] = {}
        for track_id, state in self._states.items():
            if not (state.confirmed or state.shadow_confirmed or state.evidence):
                continue
            threshold = self._policy_threshold(state)
            similarities = [item.similarity for item in state.evidence]
            progress[track_id] = TrackProgress(
                observed=len(state.evidence),
                required=self._policy_required(state),
                qualifying=sum(value >= threshold for value in similarities),
                threshold=threshold,
                median_similarity=float(np.median(similarities)) if similarities else None,
                best_similarity=max(similarities) if similarities else None,
            )
        return progress

    def _policy_threshold(self, state: _TrackState) -> float:
        if state.policy is not None:
            return state.policy.threshold
        return self.settings.similarity_threshold

    def _policy_required(self, state: _TrackState) -> int:
        if state.policy is not None:
            return state.policy.evidence_required
        return self.settings.evidence_required

    def _policy_window(self, state: _TrackState) -> float:
        if state.policy is not None:
            return state.policy.evidence_window_seconds
        return self.settings.evidence_window_seconds

    def process(
        self,
        *,
        frame_id: int,
        timestamp: float,
        frame_shape: tuple[int, ...],
        tracks: list[Track],
        faces: list[FaceObservation],
        target: Target,
        associations: dict[int, int] | None = None,
        association_modes: dict[int, str] | None = None,
        face_policies: dict[int, FaceMatchPolicy] | None = None,
    ) -> list[MatchDecision]:
        return self.process_with_stats(
            frame_id=frame_id,
            timestamp=timestamp,
            frame_shape=frame_shape,
            tracks=tracks,
            faces=faces,
            target=target,
            associations=associations,
            association_modes=association_modes,
            face_policies=face_policies,
        ).decisions

    def process_with_stats(
        self,
        *,
        frame_id: int,
        timestamp: float,
        frame_shape: tuple[int, ...],
        tracks: list[Track],
        faces: list[FaceObservation],
        target: Target,
        associations: dict[int, int] | None = None,
        association_modes: dict[int, str] | None = None,
        face_policies: dict[int, FaceMatchPolicy] | None = None,
    ) -> ConfirmationResult:
        decisions: list[MatchDecision] = []
        evidence_collected = 0
        tracks_by_id = {track.track_id: track for track in tracks}
        for track in tracks:
            state = self._states.setdefault(track.track_id, _TrackState())
            state.last_track_seen = timestamp
            state.last_bbox = track.bbox.copy()

        associations = (
            associate_faces_to_tracks(faces, tracks) if associations is None else associations
        )
        association_modes = association_modes or {}
        face_policies = face_policies or {}
        # At most one face contributes to a track in a frame: keep the highest-quality face.
        best_by_track: dict[int, tuple[int, FaceObservation]] = {}
        for face_index, track_id in associations.items():
            face = faces[face_index]
            if not face.accepted:
                continue
            previous = best_by_track.get(track_id)
            if previous is None or face.quality > previous[1].quality:
                best_by_track[track_id] = (face_index, face)

        for track_id, (face_index, face) in best_by_track.items():
            state = self._states[track_id]
            policy = face_policies.get(face_index) or FaceMatchPolicy(
                threshold=self.settings.similarity_threshold,
                evidence_required=self.settings.evidence_required,
                evidence_window_seconds=self.settings.evidence_window_seconds,
            )
            if state.policy is None:
                state.policy = policy
            elif state.shadow_confirmed and not self._is_shadow_policy(policy):
                decisions.append(self._decision(MatchState.LOST, track_id, state))
                state.evidence.clear()
                state.shadow_confirmed = False
                state.last_candidate_emit = -1e9
                state.policy = policy
            elif _is_stricter_policy(policy, state.policy):
                if not state.confirmed:
                    state.evidence.clear()
                    state.last_candidate_emit = -1e9
                state.policy = policy
            else:
                policy = state.policy
            similarity = float(np.dot(target.embedding, face.embedding))
            if not policy.accepts_observation(face.detection_score, similarity):
                continue
            if similarity >= policy.threshold:
                # A sub-threshold tiny-face sample counts as evidence, not as a sighting:
                # it must not refresh the reported state or the confirmed-track grace timer.
                state.last_similarity = similarity
                state.last_quality = face.quality
                state.last_association = association_modes.get(face_index, "person_strict")
                state.last_face_seen = timestamp
            self._expire_evidence(state, timestamp)
            duplicate_frame = any(item.frame_id == frame_id for item in state.evidence)
            separated = (
                not state.evidence
                or timestamp - state.evidence[-1].timestamp + 1e-9
                >= policy.min_observation_interval_seconds
            )
            if not duplicate_frame and separated:
                state.evidence.append(
                    _Evidence(
                        frame_id=frame_id,
                        timestamp=timestamp,
                        similarity=similarity,
                        quality=face.quality,
                        embedding=face.embedding.copy(),
                    )
                )
                evidence_collected += 1
                if policy.collect_all_observations:
                    while len(state.evidence) > policy.evidence_required:
                        state.evidence.popleft()

            if not state.confirmed and not state.shadow_confirmed:
                if (
                    not policy.suppress_candidate
                    and timestamp - state.last_candidate_emit
                    >= self.settings.candidate_emit_interval_seconds
                ):
                    decisions.append(self._decision(MatchState.CANDIDATE, track_id, state))
                    state.last_candidate_emit = timestamp
                if self._is_confirmed(state, target):
                    if self._is_shadow_policy(state.policy):
                        state.shadow_confirmed = True
                    else:
                        state.confirmed = True
                    decisions.append(self._decision(MatchState.CONFIRMED, track_id, state))

        grace = self.settings.confirmed_track_grace_seconds
        for track_id, state in list(self._states.items()):
            self._expire_evidence(state, timestamp)
            if state.confirmed or state.shadow_confirmed:
                track_missing = track_id not in tracks_by_id
                track_expired = track_missing and timestamp - state.last_track_seen >= grace
                face_expired = timestamp - state.last_face_seen >= grace
                if track_expired or face_expired:
                    decisions.append(self._decision(MatchState.LOST, track_id, state))
                    del self._states[track_id]
            elif (
                not state.evidence
                and track_id not in tracks_by_id
                and timestamp - state.last_track_seen >= grace
            ):
                del self._states[track_id]

        return ConfirmationResult(
            decisions=decisions,
            evidence_collected=evidence_collected,
        )

    def _expire_evidence(self, state: _TrackState, timestamp: float) -> None:
        window = self._policy_window(state)
        while state.evidence and timestamp - state.evidence[0].timestamp > window:
            state.evidence.popleft()

    def _is_confirmed(self, state: _TrackState, target: Target) -> bool:
        required = self._policy_required(state)
        if len(state.evidence) < required:
            return False
        similarities = [item.similarity for item in state.evidence]
        threshold = self._policy_threshold(state)
        if float(np.median(similarities)) < threshold:
            return False
        policy = state.policy
        votes_required = policy.consistent_votes_required if policy is not None else 0
        if votes_required and sum(value >= threshold for value in similarities) < votes_required:
            return False
        if policy is not None and policy.aggregate_threshold is not None:
            embeddings = np.stack([item.embedding for item in state.evidence])
            weights = np.asarray(
                [max(item.quality, 0.0) for item in state.evidence], dtype=np.float32
            )
            if not np.any(weights):
                weights = np.ones(len(state.evidence), dtype=np.float32)
            aggregate = np.average(embeddings, axis=0, weights=weights)
            magnitude = float(np.linalg.norm(aggregate))
            if magnitude <= 1e-12:
                return False
            aggregate_similarity = float(np.dot(target.embedding, aggregate / magnitude))
            if aggregate_similarity < policy.aggregate_threshold:
                return False
        return True

    def _decision(self, state_name: MatchState, track_id: int, state: _TrackState) -> MatchDecision:
        shadow = self._is_shadow_policy(state.policy)
        return MatchDecision(
            state=state_name,
            track_id=track_id,
            bbox=state.last_bbox.copy(),
            similarity=state.last_similarity,
            quality=state.last_quality,
            evidence_count=len(state.evidence),
            association=state.last_association,
            shadow=shadow,
        )

    def _is_shadow_policy(self, policy: FaceMatchPolicy | None) -> bool:
        return bool(
            self.settings.tiny_face_shadow_mode and policy is not None and policy.shadow_eligible
        )


def associate_faces_to_tracks(faces: list[FaceObservation], tracks: list[Track]) -> dict[int, int]:
    return {
        face_index: track_id
        for face_index, (track_id, _) in associate_faces_to_tracks_detailed(
            faces, tracks, allow_relaxed=False
        ).items()
    }


def associate_faces_to_tracks_detailed(
    faces: list[FaceObservation], tracks: list[Track], *, allow_relaxed: bool = True
) -> dict[int, tuple[int, str]]:
    """Associate faces with person tracks and label the association path.

    The strict path uses the upper 60% head-region rule.  The relaxed path is
    intentionally accepted only when exactly one full-body box contains the
    face center, which makes seated/occluded scenes more tolerant without
    silently choosing between overlapping people.
    """
    associations: dict[int, tuple[int, str]] = {}
    for face_index, face in enumerate(faces):
        center_x = float((face.bbox[0] + face.bbox[2]) / 2.0)
        center_y = float((face.bbox[1] + face.bbox[3]) / 2.0)
        candidates: list[tuple[float, int]] = []
        for track in tracks:
            x1, y1, x2, y2 = track.bbox
            upper_limit = y1 + 0.6 * (y2 - y1)
            if x1 <= center_x <= x2 and y1 <= center_y <= upper_limit:
                area = float(max(0.0, x2 - x1) * max(0.0, y2 - y1))
                candidates.append((area, track.track_id))
        if candidates:
            # The smallest containing person box is the least ambiguous assignment.
            associations[face_index] = (min(candidates)[1], "person_strict")
            continue
        if allow_relaxed:
            relaxed = []
            for track in tracks:
                x1, y1, x2, y2 = track.bbox
                if x1 <= center_x <= x2 and y1 <= center_y <= y2:
                    relaxed.append(track.track_id)
            if len(relaxed) == 1:
                associations[face_index] = (relaxed[0], "person_relaxed")
    return associations


def default_face_match_policy(face: FaceObservation, settings: Settings) -> FaceMatchPolicy:
    """Return the normal or small-face evidence policy for one observation."""
    if face.short_side < HARD_MIN_SEARCH_FACE_PX:
        raise ValueError("face is below the effective search size floor")
    is_tiny = settings.tiny_face_min_px <= face.short_side < settings.min_search_face_px
    if settings.tiny_face_enabled and is_tiny:
        return tiny_face_match_policy(settings)
    is_small = settings.min_search_face_px <= face.short_side < settings.preferred_search_face_px
    if is_small:
        return fallback_face_match_policy(settings)
    return FaceMatchPolicy(
        threshold=settings.similarity_threshold,
        evidence_required=settings.evidence_required,
        evidence_window_seconds=settings.evidence_window_seconds,
    )


def fallback_face_match_policy(settings: Settings) -> FaceMatchPolicy:
    return FaceMatchPolicy(
        threshold=settings.small_face_similarity_threshold,
        evidence_required=settings.small_face_evidence_required,
        evidence_window_seconds=settings.small_face_evidence_window_seconds,
        suppress_candidate=True,
    )


def tiny_face_match_policy(settings: Settings) -> FaceMatchPolicy:
    """Return the strict multi-frame policy used for opt-in 48-63px faces."""
    return FaceMatchPolicy(
        threshold=settings.tiny_face_similarity_threshold,
        aggregate_threshold=settings.tiny_face_aggregate_similarity_threshold,
        evidence_required=settings.tiny_face_evidence_required,
        consistent_votes_required=settings.tiny_face_consistent_votes_required,
        evidence_window_seconds=settings.tiny_face_evidence_window_seconds,
        min_observation_interval_seconds=settings.tiny_face_evidence_min_interval_seconds,
        min_detection_score=settings.tiny_face_detection_threshold,
        min_top1_margin=settings.tiny_face_min_top1_margin,
        suppress_candidate=True,
        collect_all_observations=True,
        requires_strict_association=True,
        shadow_eligible=True,
    )


def _is_stricter_policy(candidate: FaceMatchPolicy, current: FaceMatchPolicy) -> bool:
    candidate_aggregate = (
        candidate.aggregate_threshold if candidate.aggregate_threshold is not None else -1.0
    )
    current_aggregate = (
        current.aggregate_threshold if current.aggregate_threshold is not None else -1.0
    )
    return (
        candidate.threshold > current.threshold
        or candidate_aggregate > current_aggregate
        or candidate.evidence_required > current.evidence_required
        or candidate.consistent_votes_required > current.consistent_votes_required
        or candidate.min_detection_score > current.min_detection_score
        or candidate.min_top1_margin > current.min_top1_margin
        or (candidate.requires_strict_association and not current.requires_strict_association)
        or (candidate.suppress_candidate and not current.suppress_candidate)
    )


def normalize_bbox(
    bbox: np.ndarray, frame_shape: tuple[int, ...]
) -> tuple[float, float, float, float]:
    height, width = frame_shape[:2]
    x1, y1, x2, y2 = bbox.astype(float)
    return (
        float(np.clip(x1 / width, 0.0, 1.0)),
        float(np.clip(y1 / height, 0.0, 1.0)),
        float(np.clip(x2 / width, 0.0, 1.0)),
        float(np.clip(y2 / height, 0.0, 1.0)),
    )
