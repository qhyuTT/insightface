from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field

import numpy as np

from .config import Settings
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


@dataclass(frozen=True, slots=True)
class FaceMatchPolicy:
    threshold: float
    evidence_required: int
    evidence_window_seconds: float
    suppress_candidate: bool = False


@dataclass(slots=True)
class _Evidence:
    frame_id: int
    timestamp: float
    similarity: float
    quality: float


@dataclass(slots=True)
class _TrackState:
    evidence: deque[_Evidence] = field(default_factory=deque)
    confirmed: bool = False
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
                MatchState.CONFIRMED if state.confirmed else MatchState.CANDIDATE,
                state.last_similarity,
            )
            for track_id, state in self._states.items()
            if state.confirmed
            or (
                state.evidence
                and not (state.policy is not None and state.policy.suppress_candidate)
            )
        }

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
        decisions: list[MatchDecision] = []
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
            elif _is_stricter_policy(policy, state.policy):
                if not state.confirmed:
                    state.evidence.clear()
                    state.last_candidate_emit = -1e9
                state.policy = policy
            else:
                policy = state.policy
            similarity = float(np.dot(target.embedding, face.embedding))
            state.last_similarity = similarity
            state.last_quality = face.quality
            state.last_association = association_modes.get(face_index, "person_strict")
            if similarity < policy.threshold:
                continue
            state.last_face_seen = timestamp
            self._expire_evidence(state, timestamp)
            duplicate_frame = any(item.frame_id == frame_id for item in state.evidence)
            separated = not state.evidence or timestamp - state.evidence[-1].timestamp >= 0.2
            if not duplicate_frame and separated:
                state.evidence.append(
                    _Evidence(
                        frame_id=frame_id,
                        timestamp=timestamp,
                        similarity=similarity,
                        quality=face.quality,
                    )
                )

            if not state.confirmed:
                if (
                    not policy.suppress_candidate
                    and timestamp - state.last_candidate_emit
                    >= self.settings.candidate_emit_interval_seconds
                ):
                    decisions.append(
                        self._decision(MatchState.CANDIDATE, track_id, state)
                    )
                    state.last_candidate_emit = timestamp
                if self._is_confirmed(state):
                    state.confirmed = True
                    decisions.append(
                        self._decision(MatchState.CONFIRMED, track_id, state)
                    )

        grace = self.settings.confirmed_track_grace_seconds
        for track_id, state in list(self._states.items()):
            self._expire_evidence(state, timestamp)
            if state.confirmed:
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

        return decisions

    def _expire_evidence(self, state: _TrackState, timestamp: float) -> None:
        window = (
            state.policy.evidence_window_seconds
            if state.policy is not None
            else self.settings.evidence_window_seconds
        )
        while state.evidence and timestamp - state.evidence[0].timestamp > window:
            state.evidence.popleft()

    def _is_confirmed(self, state: _TrackState) -> bool:
        required = (
            state.policy.evidence_required
            if state.policy is not None
            else self.settings.evidence_required
        )
        if len(state.evidence) < required:
            return False
        similarities = [item.similarity for item in state.evidence]
        threshold = (
            state.policy.threshold
            if state.policy is not None
            else self.settings.similarity_threshold
        )
        return float(np.median(similarities)) >= threshold

    @staticmethod
    def _decision(state_name: MatchState, track_id: int, state: _TrackState) -> MatchDecision:
        return MatchDecision(
            state=state_name,
            track_id=track_id,
            bbox=state.last_bbox.copy(),
            similarity=state.last_similarity,
            quality=state.last_quality,
            evidence_count=len(state.evidence),
            association=state.last_association,
        )


def associate_faces_to_tracks(
    faces: list[FaceObservation], tracks: list[Track]
) -> dict[int, int]:
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


def _is_stricter_policy(candidate: FaceMatchPolicy, current: FaceMatchPolicy) -> bool:
    return (
        candidate.threshold > current.threshold
        or candidate.evidence_required > current.evidence_required
        or (candidate.suppress_candidate and not current.suppress_candidate)
    )


def normalize_bbox(bbox: np.ndarray, frame_shape: tuple[int, ...]) -> tuple[float, float, float, float]:
    height, width = frame_shape[:2]
    x1, y1, x2, y2 = bbox.astype(float)
    return (
        float(np.clip(x1 / width, 0.0, 1.0)),
        float(np.clip(y1 / height, 0.0, 1.0)),
        float(np.clip(x2 / width, 0.0, 1.0)),
        float(np.clip(y2 / height, 0.0, 1.0)),
    )
