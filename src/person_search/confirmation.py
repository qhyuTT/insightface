from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field

import numpy as np

from .config import HARD_MIN_SEARCH_FACE_PX, Settings
from .domain import FaceObservation, MatchState, Target, Track

# Size tiers, loosest first. The name travels with the policy so progress reporting
# and the panel can say which bar a track is being held to.
TIER_NORMAL = "normal"
TIER_SMALL = "small"
TIER_TINY = "tiny"

# The confirmation gates, named so a track that never confirmed can say which one
# stopped it. Ordered furthest-from-confirmation first: a post-mortem reports the
# closest the track ever came, so "closer" has to be an explicit ranking rather
# than something implied by the order of the checks in _evaluate.
GATE_INSUFFICIENT_SAMPLES = "insufficient_samples"
GATE_WINDOW_STATISTIC_LOW = "window_statistic_low"
GATE_VOTES_LOW = "votes_low"
GATE_AGGREGATE_LOW = "aggregate_low"
GATE_ORDER: tuple[str, ...] = (
    GATE_INSUFFICIENT_SAMPLES,
    GATE_WINDOW_STATISTIC_LOW,
    GATE_VOTES_LOW,
    GATE_AGGREGATE_LOW,
)


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
    # The tracked person box stays in ``bbox`` for event compatibility.  This is
    # the actual face detector box and must be used when making a face crop.
    face_bbox: np.ndarray | None = None


@dataclass(frozen=True, slots=True)
class TrackOutcome:
    """How one track ended, recorded once per track rather than per frame.

    Emitted at confirmation, and at deletion for a track that banked evidence but
    never confirmed. It exists because the two ways a search fails --- "the person
    was never sampled enough times" and "the samples never scored high enough" ---
    are indistinguishable on the panel today: the state is dropped and everything
    it knew goes with it.

    The window is necessarily *empty* by the time an unconfirmed track is deleted
    (it had to expire for the track to be dropped at all), so the gate fields come
    from the snapshot taken when the track was **closest** to confirming, not from
    the deque at deletion time. Reading the deque would make every post-mortem say
    "0 samples" regardless of what actually happened.

    ``banked`` is the in-window count the gate reads; ``sampled`` counts every
    observation ever banked on this track and, over ``dwell_seconds``, is the
    achieved sampling rate to compare against the rate the window requires.
    """

    track_id: int
    confirmed: bool
    tier: str | None
    association: str
    banked: int
    sampled: int
    required: int
    qualifying: int
    threshold: float
    window_similarity: float | None
    window_statistic: str
    best_similarity: float | None
    aggregate_similarity: float | None
    aggregate_threshold: float | None
    dwell_seconds: float
    blocking_gate: str | None = None
    time_to_confirm_seconds: float | None = None
    shadow: bool = False


@dataclass(frozen=True, slots=True)
class ConfirmationResult:
    decisions: list[MatchDecision]
    evidence_collected: int = 0
    outcomes: list[TrackOutcome] = field(default_factory=list)


@dataclass(frozen=True, slots=True)
class TrackProgress:
    """How close one track is to confirmation, and why it is not there yet.

    ``observed`` counts every banked sample; under ``collect_all_observations``
    that includes sub-threshold ones, so it saturates at ``required`` and says
    nothing about progress. ``qualifying`` and ``window_similarity`` are what
    the confirmation gate actually reads.

    ``window_similarity`` is reduced by ``window_statistic``; the name travels
    with the value because a top-K mean read as a median is a wrong number.

    ``aggregate_similarity`` is the far-face tier's second gate: a
    quality-weighted mean embedding compared against the target. Without it a
    track can show a passing window value and still never confirm, with nothing
    on the panel explaining why. It is ``None`` for tiers that do not use the gate.
    """

    observed: int
    required: int
    qualifying: int
    threshold: float
    window_similarity: float | None
    best_similarity: float | None
    window_statistic: str = "median"
    aggregate_similarity: float | None = None
    aggregate_threshold: float | None = None
    tier: str | None = None


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
    # Must sit on a person track: a face-only fallback track is never enough.
    requires_strict_association: bool = False
    # Whether the relaxed person path (exactly one body box contains the face
    # center) counts as that association. The two are separate questions: the
    # seated and truncated cases are relaxed but perfectly unambiguous, while a
    # face-only track carries no body evidence at all.
    allows_relaxed_association: bool = True
    shadow_eligible: bool = False
    # Which size tier produced this policy, and how its window is reduced to the
    # one number _is_confirmed compares against the threshold.
    tier: str = TIER_NORMAL
    statistic: str = "median"
    top_k: int = 3

    def accepts_observation(self, detection_score: float, similarity: float) -> bool:
        return detection_score >= self.min_detection_score and (
            similarity >= self.threshold or self.collect_all_observations
        )


@dataclass(frozen=True, slots=True)
class _Attempt:
    """One evaluation of the confirmation gates for a track.

    ``gate`` is the first gate that failed, or ``None`` when all of them passed.
    ``TrackProgress``, ``_is_confirmed`` and the post-mortem are all built from
    this one object so the panel, the verdict and the diagnosis cannot disagree.
    """

    gate: str | None
    banked: int
    required: int
    qualifying: int
    threshold: float
    window_similarity: float | None
    window_statistic: str
    best_similarity: float | None
    aggregate_similarity: float | None
    aggregate_threshold: float | None

    def rank(self) -> tuple[int, int, float]:
        """How close this attempt came to confirming, for picking a track's best."""
        ordinal = len(GATE_ORDER) if self.gate is None else GATE_ORDER.index(self.gate)
        return (ordinal, self.qualifying, self.window_similarity or -1.0)


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
    last_face_bbox: np.ndarray | None = None
    last_similarity: float = -1.0
    last_quality: float = 0.0
    last_association: str = "person_strict"
    policy: FaceMatchPolicy | None = None
    tier: str | None = None
    # Post-mortem bookkeeping. ``first_evidence_at`` is never reset once set --- a
    # policy change clears the evidence window, but the track was still being
    # observed the whole time, and dwell has to span that. ``sampled`` likewise
    # counts every observation ever banked, so ``sampled / dwell`` is the sampling
    # rate the track actually achieved rather than what survived in the window.
    first_evidence_at: float | None = None
    last_evidence_at: float = 0.0
    sampled: int = 0
    best_attempt: _Attempt | None = None
    outcome_emitted: bool = False


class TrackConfirmation:
    def __init__(self, settings: Settings):
        self.settings = settings
        self._states: dict[int, _TrackState] = {}

    def reset(self) -> None:
        for state in self._states.values():
            self._clear_state_evidence(state)
        self._states.clear()

    def clear_sensitive(self) -> None:
        """Wipe per-track biometric evidence before a session is retained.

        ``reset`` is used while a search is still running and deliberately only
        drops the mapping.  A terminal session can remain available for HTTP
        reconciliation, however, so simply dropping the mapping would leave
        NumPy arrays reachable from the old state until the whole session is
        collected.  Best-effort zeroing makes the lifecycle boundary explicit
        while keeping the operation idempotent.
        """
        for state in self._states.values():
            self._clear_state_evidence(state)
            _wipe_array(state.last_bbox)
            _wipe_array(state.last_face_bbox)
        self._states.clear()

    @staticmethod
    def _clear_state_evidence(state: _TrackState) -> None:
        for item in state.evidence:
            _wipe_array(item.embedding)
        state.evidence.clear()

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

    def track_progress(self, target: Target | None = None) -> dict[int, TrackProgress]:
        """Return per-track confirmation progress for active tracks.

        ``target`` is only needed to report the far-face aggregate gate; without
        it ``aggregate_similarity`` stays ``None`` and the rest is unchanged.
        """
        progress: dict[int, TrackProgress] = {}
        for track_id, state in self._states.items():
            if not (state.confirmed or state.shadow_confirmed or state.evidence):
                continue
            attempt = self._evaluate(state, target)
            progress[track_id] = TrackProgress(
                observed=attempt.banked,
                required=attempt.required,
                qualifying=attempt.qualifying,
                threshold=attempt.threshold,
                window_similarity=attempt.window_similarity,
                best_similarity=attempt.best_similarity,
                window_statistic=attempt.window_statistic,
                aggregate_similarity=attempt.aggregate_similarity,
                aggregate_threshold=attempt.aggregate_threshold,
                tier=state.tier,
            )
        return progress

    def tier_of(self, track_id: int) -> str | None:
        """Return the size tier a track is currently judged by, if it has one.

        The caller resolves the next observation's tier against this so a face
        drifting across a boundary does not flip tiers, and therefore does not
        clear the evidence window, on every frame.
        """
        state = self._states.get(track_id)
        return None if state is None else state.tier

    def _policy_threshold(self, state: _TrackState) -> float:
        if state.policy is not None:
            return state.policy.threshold
        return self.settings.similarity_threshold

    def _policy_statistic(self, state: _TrackState) -> tuple[str, int]:
        if state.policy is not None:
            return state.policy.statistic, state.policy.top_k
        return self.settings.evidence_statistic, self.settings.evidence_top_k

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
        outcomes: list[TrackOutcome] = []
        evidence_collected = 0
        # A terminal session keeps an embedding-free Target metadata snapshot.
        # It must remain safe to call this method defensively after cleanup, even
        # though the normal worker never processes another frame at that point.
        if target.embedding is None:
            return ConfirmationResult(decisions=[], evidence_collected=0, outcomes=[])
        # Validate the target once before walking the frame.  Provider shims and
        # callers of this low-level class can hand us a scalar, ragged, or
        # non-finite array; letting NumPy broadcast it in ``dot`` would either
        # raise in the worker or produce a vector that is accidentally treated as
        # a score.  A malformed target is a fail-closed frame, just like a
        # malformed face embedding.
        try:
            target_array = np.asarray(target.embedding, dtype=np.float32)
        except (TypeError, ValueError, OverflowError, FloatingPointError):
            return ConfirmationResult(decisions=[], evidence_collected=0, outcomes=[])
        if (
            target_array.ndim != 1
            or target_array.size == 0
            or not np.isfinite(target_array).all()
        ):
            return ConfirmationResult(decisions=[], evidence_collected=0, outcomes=[])
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
            if not 0 <= face_index < len(faces):
                # Associations are normally produced by the same frame's
                # matcher, but a stale/third-party mapping must not turn a bad
                # index into a worker-wide failure.
                continue
            if track_id not in tracks_by_id:
                # Likewise, ignore an association to a track that was already
                # evicted between matching and confirmation.
                continue
            face = faces[face_index]
            if not face.accepted or face.embedding is None:
                continue
            try:
                face_array = np.asarray(face.embedding, dtype=np.float32).reshape(-1)
            except (TypeError, ValueError, OverflowError, FloatingPointError):
                continue
            if (
                face_array.size == 0
                or face_array.size != target_array.size
                or not np.isfinite(face_array).all()
            ):
                continue
            previous = best_by_track.get(track_id)
            if previous is None or face.quality > previous[1].quality:
                best_by_track[track_id] = (face_index, face)

        for track_id, (face_index, face) in best_by_track.items():
            try:
                face_array = np.asarray(face.embedding, dtype=np.float32).reshape(-1)
            except (TypeError, ValueError, OverflowError, FloatingPointError):
                continue
            if (
                face_array.size == 0
                or face_array.size != target_array.size
                or not np.isfinite(face_array).all()
            ):
                continue
            state = self._states[track_id]
            policy = face_policies.get(face_index) or FaceMatchPolicy(
                threshold=self.settings.similarity_threshold,
                evidence_required=self.settings.evidence_required,
                evidence_window_seconds=self.settings.evidence_window_seconds,
                min_observation_interval_seconds=self.settings.evidence_min_interval_seconds,
            )
            if state.policy is None:
                state.policy = policy
            elif state.shadow_confirmed and not self._is_shadow_policy(policy):
                decisions.append(self._decision(MatchState.LOST, track_id, state))
                self._clear_state_evidence(state)
                state.shadow_confirmed = False
                state.last_candidate_emit = -1e9
                state.policy = policy
            elif is_stricter_policy(policy, state.policy):
                if not state.confirmed:
                    self._clear_state_evidence(state)
                    state.last_candidate_emit = -1e9
                state.policy = policy
            elif policy != state.policy:
                # A face that grew back into a looser tier must be judged by that
                # tier. Latching the strictest policy a track ever saw meant a
                # passenger first seen at 55px kept the 0.64/6-frame far-face bar
                # for the whole track, even at 120px where 0.55/3 frames applies.
                # Evidence is cleared in both directions so one window never mixes
                # samples taken under different thresholds.
                if not state.confirmed:
                    self._clear_state_evidence(state)
                    state.last_candidate_emit = -1e9
                state.policy = policy
            else:
                policy = state.policy
            state.tier = policy.tier
            try:
                similarity = float(np.dot(target_array, face_array))
            except (TypeError, ValueError, OverflowError, FloatingPointError):
                continue
            if not np.isfinite(similarity):
                continue
            if not policy.accepts_observation(face.detection_score, similarity):
                continue
            # The crop belongs to the observation that is currently driving the
            # verdict. Keep it independent of ``last_face_seen``: a collect-all
            # policy may legally confirm on a sub-threshold final sample when the
            # complete evidence window still passes its aggregate gates.
            state.last_face_bbox = face.bbox.copy()
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
                        embedding=face_array.copy(),
                    )
                )
                evidence_collected += 1
                if state.first_evidence_at is None:
                    state.first_evidence_at = timestamp
                state.last_evidence_at = timestamp
                state.sampled += 1
                if policy.collect_all_observations:
                    while len(state.evidence) > policy.evidence_required:
                        old = state.evidence.popleft()
                        _wipe_array(old.embedding)

            if not state.confirmed and not state.shadow_confirmed:
                if (
                    not policy.suppress_candidate
                    and timestamp - state.last_candidate_emit
                    >= self.settings.candidate_emit_interval_seconds
                ):
                    decisions.append(self._decision(MatchState.CANDIDATE, track_id, state))
                    state.last_candidate_emit = timestamp
                attempt = self._evaluate(state, target)
                # Keep the closest this track ever came, not the last thing it did:
                # by the time an unconfirmed track is deleted its window has expired
                # to nothing, and a post-mortem read off the empty deque would blame
                # every failure on "not enough samples".
                if state.best_attempt is None or attempt.rank() > state.best_attempt.rank():
                    state.best_attempt = attempt
                if attempt.gate is None:
                    shadow = self._is_shadow_policy(state.policy)
                    if shadow:
                        state.shadow_confirmed = True
                    else:
                        state.confirmed = True
                    decisions.append(self._decision(MatchState.CONFIRMED, track_id, state))
                    outcomes.append(
                        self._outcome(
                            track_id,
                            state,
                            attempt,
                            confirmed=True,
                            shadow=shadow,
                            # `or` would read a first sample at t=0.0 as "unset"
                            # and report every such confirmation as instantaneous.
                            time_to_confirm_seconds=timestamp
                            - (
                                timestamp
                                if state.first_evidence_at is None
                                else state.first_evidence_at
                            ),
                        )
                    )

        grace = self.settings.confirmed_track_grace_seconds
        for track_id, state in list(self._states.items()):
            self._expire_evidence(state, timestamp)
            if state.confirmed or state.shadow_confirmed:
                track_missing = track_id not in tracks_by_id
                track_expired = track_missing and timestamp - state.last_track_seen >= grace
                face_expired = timestamp - state.last_face_seen >= grace
                if track_expired or face_expired:
                    decisions.append(self._decision(MatchState.LOST, track_id, state))
                    self._clear_state_evidence(state)
                    del self._states[track_id]
            elif (
                not state.evidence
                and track_id not in tracks_by_id
                and timestamp - state.last_track_seen >= grace
            ):
                self._record_unconfirmed_outcome(track_id, state, outcomes, decisions)
                self._clear_state_evidence(state)
                del self._states[track_id]

        return ConfirmationResult(
            decisions=decisions,
            evidence_collected=evidence_collected,
            outcomes=outcomes,
        )

    def _record_unconfirmed_outcome(
        self,
        track_id: int,
        state: _TrackState,
        outcomes: list[TrackOutcome],
        decisions: list[MatchDecision],
    ) -> None:
        """Report why a track that banked evidence went away without confirming.

        Tracks that never banked anything are skipped: every passer-by whose face
        was never matchable would otherwise drown the one track that got close.
        """
        if state.outcome_emitted or state.first_evidence_at is None:
            return
        attempt = state.best_attempt
        if attempt is None:
            return
        adjudicated = self._adjudicate_departure(attempt)
        dwell = max(0.0, state.last_evidence_at - state.first_evidence_at)
        if adjudicated:
            decisions.append(
                self._decision(
                    MatchState.CONFIRMED,
                    track_id,
                    state,
                    shadow=True,
                    evidence_count=attempt.banked,
                )
            )
            # The track is already gone, so the pair has to close in the same frame.
            # A shadow hit with no matching lost leaves the console showing a live
            # lead for someone who has left, and the pairing is what every consumer
            # of the shadow channel is written against.
            decisions.append(
                self._decision(
                    MatchState.LOST,
                    track_id,
                    state,
                    shadow=True,
                    evidence_count=attempt.banked,
                )
            )
        outcomes.append(
            self._outcome(
                track_id,
                state,
                attempt,
                confirmed=adjudicated,
                shadow=adjudicated,
                time_to_confirm_seconds=dwell if adjudicated else None,
            )
        )

    def _adjudicate_departure(self, attempt: _Attempt) -> bool:
        """Whether a track that ran out of frames still earns a lead-grade hit.

        Only ``insufficient_samples`` is eligible. A track that *was* sampled
        enough and still scored too low has already been judged; re-judging it on
        the way out at a lower bar is how an uncalibrated threshold cut gets
        smuggled in under another name. What is traded here is frames --- the ones
        the subject never stayed still long enough to give --- against a higher bar
        on the frames there were.
        """
        settings = self.settings
        if not settings.departure_adjudication_enabled:
            return False
        if attempt.gate != GATE_INSUFFICIENT_SAMPLES:
            return False
        if attempt.banked < settings.departure_min_samples:
            return False
        if attempt.window_similarity is None:
            return False
        return (
            attempt.window_similarity >= attempt.threshold + settings.departure_similarity_margin
        )

    def _outcome(
        self,
        track_id: int,
        state: _TrackState,
        attempt: _Attempt,
        *,
        confirmed: bool,
        shadow: bool = False,
        time_to_confirm_seconds: float | None = None,
    ) -> TrackOutcome:
        state.outcome_emitted = True
        first_at = state.first_evidence_at
        return TrackOutcome(
            track_id=track_id,
            confirmed=confirmed,
            tier=state.tier,
            association=state.last_association,
            banked=attempt.banked,
            sampled=state.sampled,
            required=attempt.required,
            qualifying=attempt.qualifying,
            threshold=attempt.threshold,
            window_similarity=attempt.window_similarity,
            window_statistic=attempt.window_statistic,
            best_similarity=attempt.best_similarity,
            aggregate_similarity=attempt.aggregate_similarity,
            aggregate_threshold=attempt.aggregate_threshold,
            dwell_seconds=(
                0.0 if first_at is None else max(0.0, state.last_evidence_at - first_at)
            ),
            blocking_gate=attempt.gate,
            time_to_confirm_seconds=time_to_confirm_seconds,
            shadow=shadow,
        )

    def _expire_evidence(self, state: _TrackState, timestamp: float) -> None:
        window = self._policy_window(state)
        while state.evidence and timestamp - state.evidence[0].timestamp > window:
            old = state.evidence.popleft()
            _wipe_array(old.embedding)

    def _evaluate(self, state: _TrackState, target: Target | None) -> _Attempt:
        """Run the confirmation gates in order and report the first one that fails.

        The single place the gates live. ``_is_confirmed`` is ``gate is None``,
        ``track_progress`` renders the same object, and a track's post-mortem
        names ``gate`` --- so what the panel shows, what the verdict used and why
        a track failed are by construction the same numbers.

        ``target`` may be ``None`` when only the progress fields are wanted; the
        aggregate gate then fails for want of a comparison, exactly as it already
        did when the weighted mean was unusable.
        """
        policy = state.policy
        similarities = [item.similarity for item in state.evidence]
        required = self._policy_required(state)
        threshold = self._policy_threshold(state)
        statistic, top_k = self._policy_statistic(state)
        aggregate_threshold = policy.aggregate_threshold if policy is not None else None
        aggregate_similarity = (
            self._aggregate_similarity(state, target)
            if target is not None and aggregate_threshold is not None
            else None
        )
        window_similarity = (
            _window_statistic(similarities, statistic, top_k) if similarities else None
        )
        qualifying = sum(value >= threshold for value in similarities)
        votes_required = policy.consistent_votes_required if policy is not None else 0

        gate: str | None = None
        if len(state.evidence) < required:
            gate = GATE_INSUFFICIENT_SAMPLES
        elif window_similarity is None or window_similarity < threshold:
            gate = GATE_WINDOW_STATISTIC_LOW
        elif votes_required and qualifying < votes_required:
            gate = GATE_VOTES_LOW
        elif aggregate_threshold is not None and (
            aggregate_similarity is None or aggregate_similarity < aggregate_threshold
        ):
            gate = GATE_AGGREGATE_LOW

        return _Attempt(
            gate=gate,
            banked=len(state.evidence),
            required=required,
            qualifying=qualifying,
            threshold=threshold,
            window_similarity=window_similarity,
            window_statistic=statistic,
            best_similarity=max(similarities) if similarities else None,
            aggregate_similarity=aggregate_similarity,
            aggregate_threshold=aggregate_threshold,
        )

    def _is_confirmed(self, state: _TrackState, target: Target) -> bool:
        return self._evaluate(state, target).gate is None

    def _aggregate_similarity(self, state: _TrackState, target: Target) -> float | None:
        """Cosine between the target and the window's quality-weighted mean embedding.

        ``track_progress`` and ``_is_confirmed`` both read this so the panel can
        never disagree with the verdict. ``None`` means the aggregate is unusable
        (no evidence, or the weighted mean cancelled out), which the gate treats
        as a failure.
        """
        if not state.evidence or target.embedding is None:
            return None
        try:
            target_embedding = np.asarray(target.embedding, dtype=np.float32).reshape(-1)
            embeddings = np.stack(
                [np.asarray(item.embedding, dtype=np.float32).reshape(-1) for item in state.evidence]
            )
        except (TypeError, ValueError, OverflowError, FloatingPointError):
            return None
        if (
            target_embedding.size == 0
            or embeddings.ndim != 2
            or embeddings.shape[1] != target_embedding.size
            or not np.isfinite(target_embedding).all()
            or not np.isfinite(embeddings).all()
        ):
            return None
        weights = np.asarray([max(item.quality, 0.0) for item in state.evidence], dtype=np.float32)
        if not np.any(weights):
            weights = np.ones(len(state.evidence), dtype=np.float32)
        try:
            aggregate = np.average(embeddings, axis=0, weights=weights)
            magnitude = float(np.linalg.norm(aggregate))
        except (TypeError, ValueError, OverflowError, FloatingPointError):
            return None
        if magnitude <= 1e-12:
            return None
        try:
            similarity = float(np.dot(target_embedding, aggregate / magnitude))
        except (TypeError, ValueError, OverflowError, FloatingPointError):
            return None
        return similarity if np.isfinite(similarity) else None

    def _decision(
        self,
        state_name: MatchState,
        track_id: int,
        state: _TrackState,
        *,
        shadow: bool | None = None,
        evidence_count: int | None = None,
    ) -> MatchDecision:
        """Build one decision from a track's state.

        ``shadow`` and ``evidence_count`` are normally derived from the state, but
        a departure adjudication happens after the window has expired: the deque is
        empty, so the count has to come from the attempt that was actually judged.
        """
        shadow = self._is_shadow_policy(state.policy) if shadow is None else shadow
        return MatchDecision(
            state=state_name,
            track_id=track_id,
            bbox=state.last_bbox.copy(),
            similarity=state.last_similarity,
            quality=state.last_quality,
            evidence_count=len(state.evidence) if evidence_count is None else evidence_count,
            face_bbox=None if state.last_face_bbox is None else state.last_face_bbox.copy(),
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


def resolve_face_tier(short_side: int, settings: Settings, current_tier: str | None = None) -> str:
    """Return the size tier for one observation, sticky by ``face_tier_hysteresis_px``.

    Leaving a tier costs a margin. Without one, a robot in motion sweeps a single
    face back and forth across 64 or 80 px, and because changing tier clears the
    evidence window, the window never fills — which reads on the panel as
    "evidence is stuck at zero" rather than as tier thrash. A track with no tier
    yet gets no margin: the first observation must land on its honest tier.
    """
    margin = settings.face_tier_hysteresis_px if current_tier else 0
    small_floor = settings.min_search_face_px
    normal_floor = settings.preferred_search_face_px
    if current_tier == TIER_TINY:
        small_floor += margin
    elif current_tier == TIER_SMALL:
        small_floor -= margin
        normal_floor += margin
    elif current_tier == TIER_NORMAL:
        small_floor -= margin
        normal_floor -= margin
    if short_side >= normal_floor:
        return TIER_NORMAL
    if short_side >= small_floor:
        return TIER_SMALL
    return TIER_TINY


def default_face_match_policy(
    face: FaceObservation, settings: Settings, current_tier: str | None = None
) -> FaceMatchPolicy:
    """Return the evidence policy for one observation, given the track's current tier."""
    if face.short_side < HARD_MIN_SEARCH_FACE_PX:
        raise ValueError("face is below the effective search size floor")
    tier = resolve_face_tier(face.short_side, settings, current_tier)
    if tier == TIER_TINY and not settings.tiny_face_enabled:
        # The far tier is opt-in. With it off, a sub-tier face cannot legally reach
        # here at all -- _is_face_matchable rejects anything below
        # effective_search_min_face_px -- so nothing tier-specific applies.
        tier = TIER_NORMAL
    if tier == TIER_TINY:
        return tiny_face_match_policy(settings)
    if tier == TIER_SMALL:
        return fallback_face_match_policy(settings)
    return FaceMatchPolicy(
        threshold=settings.similarity_threshold,
        evidence_required=settings.evidence_required,
        evidence_window_seconds=settings.evidence_window_seconds,
        min_observation_interval_seconds=settings.evidence_min_interval_seconds,
        tier=TIER_NORMAL,
        statistic=settings.evidence_statistic,
        top_k=settings.evidence_top_k,
    )


def fallback_face_match_policy(settings: Settings) -> FaceMatchPolicy:
    return FaceMatchPolicy(
        threshold=settings.small_face_similarity_threshold,
        evidence_required=settings.small_face_evidence_required,
        evidence_window_seconds=settings.small_face_evidence_window_seconds,
        min_observation_interval_seconds=settings.evidence_min_interval_seconds,
        suppress_candidate=True,
        tier=TIER_SMALL,
        statistic=settings.evidence_statistic,
        top_k=settings.evidence_top_k,
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
        # A far face that reached only the relaxed path is the seated or truncated
        # case, not an ambiguous one -- relaxed already refuses to choose between
        # overlapping people. Dropping it made "sitting at a distance"
        # unconfirmable by construction.
        allows_relaxed_association=settings.tiny_face_allow_relaxed_association,
        shadow_eligible=True,
        tier=TIER_TINY,
        statistic=settings.evidence_statistic,
        top_k=settings.evidence_top_k,
    )


def _window_statistic(similarities: list[float], statistic: str, top_k: int) -> float:
    """Reduce an evidence window to the one number the verdict compares.

    ``median`` demands that most of the window be good, which is right for a fixed
    camera and wrong for a robot that samples plenty of unusable poses on the way
    past. ``top_k_mean`` asks instead for K genuinely good looks. Both
    ``_is_confirmed`` and ``track_progress`` read this, so the panel can never
    disagree with the verdict.
    """
    if not similarities:
        return float("-inf")
    if statistic == "top_k_mean":
        count = max(1, min(int(top_k), len(similarities)))
        return float(np.mean(sorted(similarities, reverse=True)[:count]))
    return float(np.median(similarities))


def is_stricter_policy(candidate: FaceMatchPolicy, current: FaceMatchPolicy) -> bool:
    """Whether ``candidate`` holds a track to a higher bar than ``current``.

    Also used by the caller to decide whether a relaxed association should pull a
    face down to the small-face policy: it should when that policy is stricter than
    the one the face already has, and must not when it is looser.
    """
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
        or (current.allows_relaxed_association and not candidate.allows_relaxed_association)
        or (candidate.suppress_candidate and not current.suppress_candidate)
    )


def _wipe_array(value: np.ndarray | None) -> None:
    """Best-effort zeroing for sensitive NumPy buffers."""
    if value is None:
        return
    try:
        value.fill(0)
    except (AttributeError, TypeError, ValueError):
        # Read-only views can still be released safely; zeroing is only a
        # defense-in-depth measure and must not turn cleanup into a failure.
        return


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
