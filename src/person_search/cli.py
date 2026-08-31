from __future__ import annotations

import argparse
import json
import threading
import time
from collections import Counter
from pathlib import Path
from types import SimpleNamespace

import cv2
import numpy as np

from .backends import InsightFaceBackend
from .config import Settings
from .confirmation import (
    ConfirmationResult,
    FaceMatchPolicy,
    TrackConfirmation,
    TrackOutcome,
    associate_faces_to_tracks_detailed,
    default_face_match_policy,
    fallback_face_match_policy,
    is_stricter_policy,
    normalize_bbox,
)
from .detector import YoloXOnnxDetector
from .domain import FaceObservation, MatchState, SearchMetrics, Target, Track
from .evaluation import (
    DEFAULT_EVAL_THRESHOLDS,
    MAX_FALSE_CONFIRMATIONS_PER_HOUR,
    MIN_INTERVAL_RECALL,
    MIN_NEGATIVE_EXPOSURE_HOURS,
    aggregate_threshold_results,
    face_px_bucket,
    load_manifest,
    recommend_threshold,
    summarize_events,
    summarize_similarity_samples,
    threshold_key,
    validate_thresholds,
)
from .face_tracking import FaceTracker
from .service import SearchManager, SearchSession, _merge_faces
from .tracker import ByteTracker


def main() -> None:
    parser = argparse.ArgumentParser(description="Run person search on prerecorded videos")
    parser.add_argument("--photo", type=Path, help="single-face target photo")
    parser.add_argument("--name", default=None, help="display name for the target")
    parser.add_argument("--video", type=Path, help="input video")
    parser.add_argument("--manifest", type=Path, help="version 1 or 2 batch evaluation manifest")
    parser.add_argument("--output-dir", type=Path, default=Path("artifacts/eval"))
    parser.add_argument("--threshold", type=float, default=None)
    parser.add_argument("--thresholds", type=float, nargs="+", default=None)
    parser.add_argument(
        "--dump-similarities",
        action="store_true",
        help=(
            "write per-observation similarities and a genuine/impostor distribution "
            "summary, which is what a threshold should be set from"
        ),
    )
    args = parser.parse_args()

    if args.threshold is not None and args.thresholds is not None:
        parser.error("--threshold and --thresholds cannot be used together")
    if args.manifest is not None:
        if args.photo is not None or args.video is not None or args.name is not None:
            parser.error("--manifest cannot be combined with --photo, --video, or --name")
        if args.threshold is not None:
            parser.error("batch evaluation uses --thresholds")
        try:
            thresholds = validate_thresholds(args.thresholds or DEFAULT_EVAL_THRESHOLDS)
            run_manifest(
                args.manifest,
                args.output_dir,
                thresholds,
                dump_similarities=args.dump_similarities,
            )
        except (TypeError, ValueError) as exc:
            parser.error(str(exc))
        return

    if args.photo is None or args.video is None:
        parser.error("--photo and --video are required unless --manifest is used")
    raw_thresholds = args.thresholds
    if raw_thresholds is None:
        raw_thresholds = [
            args.threshold if args.threshold is not None else Settings().similarity_threshold
        ]
    try:
        thresholds = validate_thresholds(raw_thresholds)
    except ValueError as exc:
        parser.error(str(exc))
    run_offline(
        args.photo,
        args.video,
        args.output_dir,
        thresholds=thresholds,
        name=args.name,
        dump_similarities=args.dump_similarities,
    )


def run_manifest(
    manifest_path: Path,
    output_dir: Path,
    thresholds: tuple[float, ...],
    *,
    dump_similarities: bool = False,
) -> None:
    cases = load_manifest(manifest_path)
    output_dir.mkdir(parents=True, exist_ok=True)
    case_results: list[dict[str, object]] = []
    case_index: list[dict[str, str]] = []
    for case in cases:
        case_output = output_dir / case.case_id
        result = run_offline(
            case.photo,
            case.video,
            case_output,
            thresholds=thresholds,
            name=case.target_name,
            expected_intervals=case.expected_intervals_seconds,
            expected_face_px_buckets=case.expected_face_px_buckets,
            print_summary=False,
            dump_similarities=dump_similarities,
        )
        result["case_id"] = case.case_id
        case_results.append(result)
        case_index.append({"case_id": case.case_id, "report": str(case_output / "report.json")})

    aggregate = aggregate_threshold_results(case_results, thresholds)
    shadow_aggregate = aggregate_threshold_results(
        case_results, thresholds, metrics_key="shadow_metrics"
    )
    recommendation = recommend_threshold(aggregate)
    report = {
        "schema_version": 2,
        "manifest": str(manifest_path),
        "thresholds": list(thresholds),
        "acceptance_criteria": {
            "minimum_interval_recall": MIN_INTERVAL_RECALL,
            "maximum_false_confirmations_per_hour": MAX_FALSE_CONFIRMATIONS_PER_HOUR,
            "minimum_negative_exposure_hours": MIN_NEGATIVE_EXPOSURE_HOURS,
        },
        "recommended_similarity_threshold": recommendation,
        "status": "passed" if recommendation is not None else "failed_or_insufficient_data",
        "aggregate": aggregate,
        "shadow_aggregate": shadow_aggregate,
        "cases": case_index,
    }
    _write_report(output_dir / "report.json", report)
    print(json.dumps(report, ensure_ascii=False, indent=2))


def run_offline(
    photo_path: Path,
    video_path: Path,
    output_dir: Path,
    threshold: float | None = None,
    name: str | None = None,
    *,
    thresholds: tuple[float, ...] | None = None,
    expected_intervals: tuple[tuple[float, float], ...] | None = None,
    expected_face_px_buckets: tuple[str | None, ...] | None = None,
    print_summary: bool = True,
    dump_similarities: bool = False,
) -> dict[str, object]:
    if thresholds is not None and threshold is not None:
        raise ValueError("threshold and thresholds cannot both be set")
    settings = Settings()
    selected_thresholds = validate_thresholds(
        thresholds
        if thresholds is not None
        else (threshold if threshold is not None else settings.similarity_threshold,)
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    photo = cv2.imread(str(photo_path))
    if photo is None:
        raise SystemExit(f"cannot read target photo: {photo_path}")
    capture = cv2.VideoCapture(str(video_path))
    if not capture.isOpened():
        raise SystemExit(f"cannot open video: {video_path}")

    face_backend = InsightFaceBackend(settings)
    detector = YoloXOnnxDetector(settings)
    manager = SearchManager(settings, face_backend=face_backend, person_detector=detector)
    target_view = manager.enroll(photo, name or photo_path.stem)
    target = manager.get_target(target_view.target_id)
    detector.ensure_ready()

    fps = capture.get(cv2.CAP_PROP_FPS) or settings.input_fps
    width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
    output_video = output_dir / "annotated.mp4"
    writer = cv2.VideoWriter(
        str(output_video), cv2.VideoWriter_fourcc(*"mp4v"), fps, (width, height)
    )
    if not writer.isOpened():
        capture.release()
        manager.shutdown()
        raise SystemExit(f"cannot create output video: {output_video}")

    threshold_settings = {
        threshold_key(value): settings.model_copy(update={"similarity_threshold": value})
        for value in selected_thresholds
    }
    confirmations = {key: TrackConfirmation(value) for key, value in threshold_settings.items()}
    events_by_threshold: dict[str, list[dict[str, object]]] = {key: [] for key in confirmations}
    stage_counts_by_threshold: dict[str, Counter[str]] = {
        key: Counter(
            {
                "above_threshold": 0,
                "evidence_eligible": 0,
                "evidence_collected": 0,
                "confirmed": 0,
                "shadow_confirmed": 0,
            }
        )
        for key in confirmations
    }
    annotation_key = min(
        confirmations,
        key=lambda key: abs(float(key) - settings.similarity_threshold),
    )
    outcomes_by_threshold: dict[str, list[TrackOutcome]] = {key: [] for key in confirmations}
    rejection_counts: Counter[str] = Counter()
    association_counts: Counter[str] = Counter()
    unassociated_faces = 0
    face_observations = 0
    accepted_faces = 0
    embedding_candidates = 0
    embedding_output_failures = 0
    tracker = ByteTracker()
    face_tracker = FaceTracker(
        iou_threshold=settings.face_track_iou_threshold,
        buffer_seconds=settings.face_track_buffer_seconds,
    )
    # Mirrors the ROI bookkeeping SearchSession owns, so the offline harness
    # exercises the same per-track backoff and ROI selection the service does.
    # The one deliberate difference is the credit gate: offline calibration has no
    # realtime constraint, so it runs ROI at every roi_interval unconditionally.
    # The service now reaches roughly the same cadence, so this no longer
    # calibrates a pipeline production does not run.
    roi_context = SimpleNamespace(
        settings=settings,
        face_backend=face_backend,
        _roi_misses={},
        _roi_skips={},
        _track_states={},
        _motion_hanning=None,
        _lock=threading.Lock(),
        metrics=SearchMetrics(),
    )
    roi_context._note_roi_outcome = lambda track_id, *, hit: SearchSession._note_roi_outcome(
        roi_context, track_id, hit=hit
    )
    roi_context._record_stage = lambda stage, started: SearchSession._record_stage(
        roi_context, stage, started
    )
    roi_context._embed_face_chunk = (
        lambda frame, faces, *, split_depth=0: SearchSession._embed_face_chunk(
            roi_context, frame, faces, split_depth=split_depth
        )
    )
    roi_context._motion_window = lambda shape: SearchSession._motion_window(roi_context, shape)
    tracks = []
    # Tier bookkeeping, owned here for the same reason the session owns it: the tier
    # follows the observation, so every threshold under evaluation has to resolve
    # the hysteresis margin against one answer.
    track_tiers: dict[int, str] = {}
    previous_motion_gray = None
    # Raw per-observation rows for threshold calibration. Collected only on request:
    # a long clip produces one row per accepted face per face pass.
    similarity_samples: list[dict[str, object]] = []
    frame_id = 0
    started = time.monotonic()
    person_hz = (
        settings.person_detection_hz_cuda
        if "CUDA" in detector.provider_name
        else settings.person_detection_hz_cpu
    )
    face_hz = (
        settings.face_detection_hz_cuda
        if "CUDA" in getattr(face_backend, "detection_provider_name", face_backend.provider_name)
        else settings.face_detection_hz_cpu
    )
    face_is_cuda = "CUDA" in getattr(
        face_backend, "detection_provider_name", face_backend.provider_name
    )
    # Same scales, same cadence as the live loop. Calibrating against a detector
    # configuration production does not run is worse than not calibrating.
    shallow_scales = settings.full_frame_detection_scales(is_cuda=face_is_cuda, deep=False)
    deep_scales = settings.full_frame_detection_scales(is_cuda=face_is_cuda)
    face_pass_index = 0
    roi_face_hz = (
        settings.roi_face_detection_hz_cuda
        if "CUDA" in getattr(face_backend, "detection_provider_name", face_backend.provider_name)
        else settings.roi_face_detection_hz_cpu
    )
    person_interval = max(1, round(fps / max(person_hz, 0.1)))
    face_interval = max(1, round(fps / max(face_hz, 0.1)))
    roi_interval = max(1, round(fps / roi_face_hz)) if roi_face_hz > 0 else None
    last_roi_frame_id = -1_000_000_000
    last_frame_shape: tuple[int, ...] = (
        max(height, 1),
        max(width, 1),
        3,
    )
    flush_results: dict[str, ConfirmationResult] = {}
    try:
        while True:
            ok, frame = capture.read()
            if not ok:
                break
            last_frame_shape = frame.shape
            timestamp = frame_id / max(fps, 1.0)
            if frame_id % person_interval == 0:
                motion, previous_motion_gray = SearchSession._estimate_camera_motion(
                    roi_context, frame, previous_motion_gray
                )
                tracks = tracker.update(detector.detect(frame), motion=motion)
            faces = []
            embedded_faces: list[FaceObservation] = []
            if frame_id % face_interval == 0:
                deep = face_pass_index % settings.face_deep_scan_every_n == 0
                face_pass_index += 1
                faces = face_backend.detect_faces(
                    frame,
                    enrollment=False,
                    detection_size=deep_scales if deep else shallow_scales,
                )
                roi_tracks = SearchSession._tracks_needing_roi_face_pass(roi_context, faces, tracks)
                if (
                    roi_interval is not None
                    and frame_id - last_roi_frame_id >= roi_interval
                    and roi_tracks
                ):
                    roi_faces = SearchSession._analyze_person_rois(roi_context, frame, roi_tracks)
                    faces = _merge_faces(faces, roi_faces)
                    last_roi_frame_id = frame_id
                face_observations += len(faces)
                accepted_faces += sum(_is_matchable_face(face, settings) for face in faces)
                rejection_counts.update(
                    reason for face in faces for reason in face.rejection_reasons
                )
                # Embed once, after dedup and the quality gate, exactly as the
                # service does — otherwise calibration would measure a pipeline
                # that does not exist in production.
                embedding_inputs = [
                    face for face in faces if _is_matchable_face(face, settings)
                ]
                embedding_candidates += len(embedding_inputs)
                # Keep the offline harness on the same frame-level face budget
                # and OOM-aware micro-batch path as SearchSession.  The helper
                # also records split/retry counts in the shared SearchMetrics
                # object below, so a calibration report exposes degraded frames
                # instead of silently measuring an unbounded ArcFace call.
                association_cache: dict[int, tuple[int, str]] = {}
                embedding_inputs = SearchSession._limit_matchable_faces(
                    roi_context,
                    embedding_inputs,
                    tracks,
                    association_cache=association_cache,
                )
                embedded = (
                    SearchSession._embed_faces_microbatched(
                        roi_context, frame, embedding_inputs
                    )
                    if embedding_inputs
                    else []
                )
                # Treat a missing/malformed provider item as an embedding miss
                # for this crop.  The offline harness should remain useful for a
                # long recording even when one backend response is short or
                # partially corrupt; the live path applies the same drop policy.
                embedded_by_key: dict[tuple[float, ...], FaceObservation] = {}
                try:
                    embedded_items = [] if embedded is None else list(embedded)
                except (TypeError, ValueError):
                    embedded_items = []
                for embedded_face in embedded_items:
                    key = _bbox_key(getattr(embedded_face, "bbox", None))
                    if key is None:
                        continue
                    if getattr(embedded_face, "embedding", None) is not None:
                        embedded_by_key[key] = embedded_face
                embedding_output_failures += sum(
                    (_bbox_key(face.bbox) not in embedded_by_key)
                    for face in embedding_inputs
                )
                embedded_faces = [
                    embedded_face
                    for embedded_face in embedded_items
                    if isinstance(embedded_face, FaceObservation)
                    and embedded_face.embedding is not None
                    and _is_matchable_face(embedded_face, settings)
                ]
                reconciled_faces: list[FaceObservation] = []
                for face in faces:
                    key = _bbox_key(face.bbox)
                    reconciled_faces.append(
                        embedded_by_key.get(key, face) if key is not None else face
                    )
                faces = reconciled_faces

            # A quality-accepted detection is not necessarily matchable yet:
            # ArcFace can reject a crop (or a provider can return a malformed
            # result) and leave its embedding empty.  Keep that observation in
            # the detection/quality counters, but never pass it to association
            # or confirmation where a dot product would otherwise fail.
            # ``embedded_faces`` is the exact output of the bounded ArcFace stage;
            # use it rather than rediscovering every quality-accepted detection in
            # ``faces`` (which would accidentally admit an unselected/bad crop).
            accepted = embedded_faces
            all_tracks, associations, association_modes = _associate_search_faces(
                accepted,
                tracks,
                settings=settings,
                face_tracker=face_tracker,
                timestamp=timestamp,
                track_tiers=track_tiers,
            )
            association_counts.update(association_modes.values())
            unassociated_faces += sum(
                face_index not in associations for face_index in range(len(accepted))
            )
            # The tier is threshold-independent, so it is resolved once against the
            # base settings and then shared by every threshold being evaluated.
            canonical_policies = _face_policies(
                accepted,
                association_modes,
                settings,
                track_tiers,
                associations,
            )
            for face_index, track_id in associations.items():
                track_tiers[track_id] = canonical_policies[face_index].tier
            live_track_ids = {track.track_id for track in all_tracks}
            track_tiers = {
                key: value for key, value in track_tiers.items() if key in live_track_ids
            }
            if dump_similarities:
                for face_index, face in enumerate(accepted):
                    if face.embedding is None:
                        continue
                    sample = _similarity_sample(
                        face,
                        frame_id=frame_id,
                        timestamp=timestamp,
                        target_embedding=target.embedding,
                        tier=canonical_policies[face_index].tier,
                        association=association_modes.get(face_index, "unassociated"),
                    )
                    if sample is not None:
                        similarity_samples.append(sample)

            decisions_by_threshold = {}
            for key, confirmation in confirmations.items():
                policies = _face_policies(
                    accepted,
                    association_modes,
                    threshold_settings[key],
                    track_tiers,
                    associations,
                )
                counts = stage_counts_by_threshold[key]
                counts.update(
                    _confirmation_input_counts(
                        accepted,
                        associations,
                        policies,
                        target.embedding,
                    )
                )
                confirmation_result = confirmation.process_with_stats(
                    frame_id=frame_id,
                    timestamp=timestamp,
                    frame_shape=frame.shape,
                    tracks=all_tracks,
                    faces=accepted,
                    target=target,
                    associations=associations,
                    association_modes=association_modes,
                    face_policies=policies,
                )
                decisions_by_threshold[key] = confirmation_result.decisions
                counts["evidence_collected"] += confirmation_result.evidence_collected
                outcomes_by_threshold[key].extend(confirmation_result.outcomes)
            for key, decisions in decisions_by_threshold.items():
                for decision in decisions:
                    state = _offline_decision_state(decision.state.value, decision.shadow)
                    if state in {"confirmed", "shadow_confirmed"}:
                        stage_counts_by_threshold[key][state] += 1
                    events_by_threshold[key].append(
                        {
                            "frame_id": frame_id,
                            "timestamp_seconds": timestamp,
                            "state": state,
                            "shadow": decision.shadow,
                            "track_id": decision.track_id,
                            "bbox": normalize_bbox(decision.bbox, frame.shape),
                            "face_bbox": (
                                None
                                if decision.face_bbox is None
                                else normalize_bbox(decision.face_bbox, frame.shape)
                            ),
                            "similarity": decision.similarity,
                            "quality": decision.quality,
                            "evidence_count": decision.evidence_count,
                            "association": decision.association,
                        }
                    )

            for track in tracks:
                x1, y1, x2, y2 = track.bbox.astype(int)
                cv2.rectangle(frame, (x1, y1), (x2, y2), (180, 180, 0), 2)
                cv2.putText(
                    frame,
                    f"track {track.track_id}",
                    (x1, max(20, y1 - 5)),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5,
                    (180, 180, 0),
                    1,
                )
            for decision in decisions_by_threshold[annotation_key]:
                if decision.shadow:
                    color = (210, 110, 245)
                elif decision.state == MatchState.CONFIRMED:
                    color = (0, 255, 0)
                else:
                    color = (0, 165, 255)
                x1, y1, x2, y2 = decision.bbox.astype(int)
                cv2.rectangle(frame, (x1, y1), (x2, y2), color, 3)
                cv2.putText(
                    frame,
                    f"{_offline_decision_state(decision.state.value, decision.shadow)} "
                    f"{decision.similarity:.3f}",
                    (x1, min(height - 10, y2 + 20)),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6,
                    color,
                    2,
                )
            writer.write(frame)
            frame_id += 1
        # A prerecorded clip has no reader tick after its last frame.  Without
        # one explicit expiry pass, tracks that disappeared at EOF remain in the
        # confirmation windows forever and are omitted from ``track_outcomes``.
        # The helper advances a private clock past every configured evidence
        # window and track grace period; event timestamps remain anchored to the
        # final real video frame below.
        flush_results = _flush_offline_confirmations(
            confirmations,
            target=target,
            frame_id=frame_id,
            fps=fps,
            frame_shape=last_frame_shape,
            settings=settings,
        )
    finally:
        capture.release()
        writer.release()
        manager.shutdown()

    if flush_results:
        # Keep the same report shape as in-loop decisions.  A synthetic expiry
        # call is used only to settle state; consumers should see terminal events
        # at the last frame's timestamp, never several grace periods after the
        # video supposedly ended.
        terminal_timestamp = max(0.0, (frame_id - 1) / max(fps, 1.0))
        for key, confirmation_result in flush_results.items():
            counts = stage_counts_by_threshold[key]
            counts["evidence_collected"] += confirmation_result.evidence_collected
            outcomes_by_threshold[key].extend(confirmation_result.outcomes)
            for decision in confirmation_result.decisions:
                state = _offline_decision_state(decision.state.value, decision.shadow)
                if state in {"confirmed", "shadow_confirmed"}:
                    counts[state] += 1
                events_by_threshold[key].append(
                    {
                        "frame_id": max(0, frame_id - 1),
                        "timestamp_seconds": terminal_timestamp,
                        "state": state,
                        "shadow": decision.shadow,
                        "track_id": decision.track_id,
                        "bbox": normalize_bbox(decision.bbox, last_frame_shape),
                        "face_bbox": (
                            None
                            if decision.face_bbox is None
                            else normalize_bbox(decision.face_bbox, last_frame_shape)
                        ),
                        "similarity": decision.similarity,
                        "quality": decision.quality,
                        "evidence_count": decision.evidence_count,
                        "association": decision.association,
                    }
                )

    duration_seconds = frame_id / max(fps, 1.0)
    if expected_intervals and expected_intervals[-1][1] > duration_seconds + 1.0 / max(fps, 1.0):
        raise ValueError(
            f"expected interval ends after video duration ({duration_seconds:.3f} seconds)"
        )
    threshold_results: dict[str, dict[str, object]] = {}
    for value in selected_thresholds:
        key = threshold_key(value)
        events = events_by_threshold[key]
        result: dict[str, object] = {
            "threshold": value,
            "confirmed_events": sum(item["state"] == "confirmed" for item in events),
            "shadow_confirmed_events": sum(item["state"] == "shadow_confirmed" for item in events),
            "match_stage_counts": dict(stage_counts_by_threshold[key]),
            "track_outcomes": _summarize_track_outcomes(outcomes_by_threshold[key]),
            "events": events,
        }
        if expected_intervals is not None:
            result["metrics"] = summarize_events(
                events,
                expected_intervals,
                duration_seconds,
                expected_face_px_buckets,
            )
            result["shadow_metrics"] = summarize_events(
                events,
                expected_intervals,
                duration_seconds,
                expected_face_px_buckets,
                confirmation_state="shadow_confirmed",
            )
        threshold_results[key] = result

    runtime_diagnostics = roi_context.metrics.snapshot()

    summary: dict[str, object] = {
        "schema_version": 2,
        "photo": str(photo_path),
        "video": str(video_path),
        "model": face_backend.model_name,
        "target_name": target.name,
        "provider": face_backend.provider_name,
        "frames": frame_id,
        "video_duration_seconds": duration_seconds,
        "elapsed_seconds": time.monotonic() - started,
        "annotation_threshold": float(annotation_key),
        "quality_diagnostics": {
            "face_observations": face_observations,
            "accepted_faces": accepted_faces,
            "embedding_candidates": embedding_candidates,
            # ``embedding_failures`` counts input rows for which no valid
            # embedding came back.  The provider-attempt counter is retained
            # separately because one failed batch may be retried/split and is
            # not one-to-one with faces.
            "embedding_failures": embedding_output_failures,
            "embedding_provider_failures": runtime_diagnostics.get(
                "embedding_failures", 0
            ),
            "embedding_batch_count": runtime_diagnostics.get(
                "embedding_batch_count", 0
            ),
            "faces_dropped_by_budget": runtime_diagnostics.get(
                "faces_dropped_by_budget", 0
            ),
            "rejection_counts": dict(sorted(rejection_counts.items())),
            "association_counts": dict(sorted(association_counts.items())),
            "unassociated_faces": unassociated_faces,
        },
        "threshold_results": threshold_results,
    }
    if dump_similarities:
        summary["similarity_distribution"] = summarize_similarity_samples(
            similarity_samples, expected_intervals
        )
        _write_report(output_dir / "similarities.json", {"samples": similarity_samples})
    if len(selected_thresholds) == 1:
        only_result = threshold_results[threshold_key(selected_thresholds[0])]
        summary.update(
            {
                "similarity_threshold": selected_thresholds[0],
                "confirmed_events": only_result["confirmed_events"],
                "shadow_confirmed_events": only_result["shadow_confirmed_events"],
                "events": only_result["events"],
            }
        )
    _write_report(output_dir / "report.json", summary)
    if print_summary:
        printable = summary.copy()
        printable.pop("events", None)
        printable["threshold_results"] = {
            key: {field: value for field, value in result.items() if field != "events"}
            for key, result in threshold_results.items()
        }
        print(json.dumps(printable, ensure_ascii=False, indent=2))
    return summary


def _associate_search_faces(
    faces: list[FaceObservation],
    tracks: list[Track],
    *,
    settings: Settings,
    face_tracker: FaceTracker,
    timestamp: float,
    track_tiers: dict[int, str] | None = None,
) -> tuple[list[Track], dict[int, int], dict[int, str]]:
    """Build the same detailed person/fallback associations as the live search."""
    detailed = associate_faces_to_tracks_detailed(faces, tracks)
    associations = {face_index: track_id for face_index, (track_id, _) in detailed.items()}
    modes = {face_index: mode for face_index, (_, mode) in detailed.items()}
    all_tracks = list(tracks)
    policies = _policies_for_faces(faces, settings, associations, track_tiers)
    for face_index, mode in list(modes.items()):
        if mode == "person_relaxed" and not policies[face_index].allows_relaxed_association:
            associations.pop(face_index, None)
            modes.pop(face_index, None)
    unassociated_indices = [
        face_index
        for face_index, face in enumerate(faces)
        if face_index not in associations
        and not policies[face_index].requires_strict_association
        and face.detection_score >= settings.fallback_face_detection_threshold
    ]
    if settings.face_fallback_enabled and unassociated_indices:
        fallback_faces = [faces[index] for index in unassociated_indices]
        fallback_tracks = face_tracker.update(fallback_faces, timestamp)
        all_tracks.extend(track for track in fallback_tracks if track is not None)
        for fallback_index, fallback_track in enumerate(fallback_tracks):
            if fallback_track is None:
                continue
            original_index = unassociated_indices[fallback_index]
            associations[original_index] = fallback_track.track_id
            modes[original_index] = "face_fallback"
    return all_tracks, associations, modes


def _policies_for_faces(
    faces: list[FaceObservation],
    settings: Settings,
    associations: dict[int, int] | None = None,
    track_tiers: dict[int, str] | None = None,
) -> dict[int, FaceMatchPolicy]:
    """Resolve one policy per face, honouring the tier a track is already held to."""
    associations = associations or {}
    track_tiers = track_tiers or {}
    policies: dict[int, FaceMatchPolicy] = {}
    for face_index, face in enumerate(faces):
        track_id = associations.get(face_index)
        current_tier = None if track_id is None else track_tiers.get(track_id)
        policies[face_index] = default_face_match_policy(face, settings, current_tier)
    return policies


def _face_policies(
    faces: list[FaceObservation],
    association_modes: dict[int, str],
    settings: Settings,
    track_tiers: dict[int, str] | None = None,
    associations: dict[int, int] | None = None,
) -> dict[int, FaceMatchPolicy]:
    fallback_policy = fallback_face_match_policy(settings)
    policies = _policies_for_faces(faces, settings, associations, track_tiers)
    for face_index, mode in association_modes.items():
        # Weaker body evidence pulls a face up to the small-face bar, but only when
        # that bar is actually higher: applying it unconditionally relaxed the far
        # tier instead of tightening it.
        if mode in {"person_relaxed", "face_fallback"} and is_stricter_policy(
            fallback_policy, policies[face_index]
        ):
            policies[face_index] = fallback_policy
    return policies


def _safe_similarity(
    target_embedding: np.ndarray | object,
    face_embedding: np.ndarray | object | None,
) -> float | None:
    """Return one finite cosine-like score, or ``None`` for malformed vectors.

    Offline evaluation is often run over long recordings and should not lose the
    whole report because one provider response is scalar, ragged, or has a shape
    different from the enrolled vector.  The live confirmation class applies the
    same fail-closed rule; keeping this helper local gives calibration the same
    semantics without mutating either input array.
    """

    if face_embedding is None:
        return None
    try:
        target = np.asarray(target_embedding, dtype=np.float32)
        face = np.asarray(face_embedding, dtype=np.float32).reshape(-1)
    except (TypeError, ValueError, OverflowError):
        return None
    if (
        target.ndim != 1
        or target.size == 0
        or face.size == 0
        or target.size != face.size
        or not np.isfinite(target).all()
        or not np.isfinite(face).all()
    ):
        return None
    try:
        similarity = float(np.dot(target, face))
    except (TypeError, ValueError, OverflowError):
        return None
    return similarity if np.isfinite(similarity) else None


def _bbox_key(value: object) -> tuple[float, float, float, float] | None:
    """Canonicalize one face box for provider-output reconciliation."""

    try:
        bbox = np.asarray(value, dtype=np.float32).reshape(-1)
    except (TypeError, ValueError, OverflowError):
        return None
    if bbox.size != 4 or not np.isfinite(bbox).all():
        return None
    return (float(bbox[0]), float(bbox[1]), float(bbox[2]), float(bbox[3]))


def _confirmation_input_counts(
    faces: list[FaceObservation],
    associations: dict[int, int],
    policies: dict[int, FaceMatchPolicy],
    target_embedding: np.ndarray,
) -> Counter[str]:
    # Detection/quality acceptance and successful ArcFace embedding are separate
    # stages.  A provider may return no embedding for one crop; diagnostics should
    # count that face as not computable rather than crashing the whole replay.
    similarities: dict[int, float] = {}
    for face_index, face in enumerate(faces):
        if face.embedding is None:
            continue
        similarity = _safe_similarity(target_embedding, face.embedding)
        if similarity is not None:
            similarities[face_index] = similarity
    return Counter(
        {
            "above_threshold": sum(
                similarities[face_index] >= policies[face_index].threshold
                for face_index in similarities
                if face_index in policies
            ),
            "evidence_eligible": sum(
                (
                    face_index in similarities
                    and face_index in policies
                    and policies[face_index].accepts_observation(
                        faces[face_index].detection_score,
                        similarities[face_index],
                    )
                )
                for face_index in associations
            ),
        }
    )


def _is_matchable_face(face: FaceObservation, settings: Settings) -> bool:
    return bool(face.accepted and face.short_side >= settings.effective_search_min_face_px)


def _summarize_track_outcomes(outcomes: list[TrackOutcome]) -> dict[str, object]:
    """Aggregate per-track post-mortems for one threshold.

    Confirmation counts alone cannot say whether a threshold failed because the
    footage never sampled a track often enough or because the samples never scored
    high enough --- the two are indistinguishable in the event stream. Splitting
    the unconfirmed tracks by the gate that stopped them is what makes a sweep
    actionable rather than just a list of zeroes.
    """
    confirm_times = [
        outcome.time_to_confirm_seconds
        for outcome in outcomes
        if outcome.time_to_confirm_seconds is not None
    ]
    dwells = [outcome.dwell_seconds for outcome in outcomes]
    rates = [
        outcome.sampled / outcome.dwell_seconds
        for outcome in outcomes
        if outcome.dwell_seconds > 0 and outcome.sampled > 1
    ]
    gates: Counter[str] = Counter(
        outcome.blocking_gate
        for outcome in outcomes
        if not outcome.confirmed and outcome.blocking_gate is not None
    )
    return {
        "tracks": len(outcomes),
        "confirmed_tracks": len(confirm_times),
        "time_to_confirm_p50_seconds": _percentile(confirm_times, 50),
        "time_to_confirm_p95_seconds": _percentile(confirm_times, 95),
        "track_dwell_p50_seconds": _percentile(dwells, 50),
        "achieved_sampling_hz": _percentile(rates, 50),
        "unconfirmed_gate_counts": dict(sorted(gates.items())),
    }


def _percentile(values: list[float], percentile: float) -> float:
    return float(np.percentile(values, percentile)) if values else 0.0


def _offline_decision_state(state: str, shadow: bool) -> str:
    return f"shadow_{state}" if shadow else state


def _flush_offline_confirmations(
    confirmations: dict[str, TrackConfirmation],
    *,
    target: Target,
    frame_id: int,
    fps: float,
    frame_shape: tuple[int, ...],
    settings: Settings,
) -> dict[str, ConfirmationResult]:
    """Expire confirmation state that has no reader tick after the final frame.

    ``TrackConfirmation`` intentionally settles departures on a later timestamp
    so its normal grace/window semantics are shared by live and offline paths.
    A file reaches EOF without that later timestamp, however, which would leave
    an unconfirmed track in memory and omit its post-mortem from the report.  We
    therefore run one empty, synthetic tick after the largest configured window;
    callers should use the real final frame timestamp when serializing any
    resulting decisions.
    """

    if frame_id <= 0 or not confirmations:
        return {}
    safe_fps = max(float(fps), 1.0)
    last_timestamp = (frame_id - 1) / safe_fps
    durations = [
        float(getattr(settings, "confirmed_track_grace_seconds", 0.0)),
        float(getattr(settings, "evidence_window_seconds", 0.0)),
        float(getattr(settings, "small_face_evidence_window_seconds", 0.0)),
        float(getattr(settings, "tiny_face_evidence_window_seconds", 0.0)),
    ]
    flush_timestamp = last_timestamp + max(0.0, *durations) + 1e-6
    return {
        key: confirmation.process_with_stats(
            frame_id=frame_id,
            timestamp=flush_timestamp,
            frame_shape=frame_shape,
            tracks=[],
            faces=[],
            target=target,
        )
        for key, confirmation in confirmations.items()
    }


def _write_report(path: Path, payload: dict[str, object]) -> None:
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def _similarity_sample(
    face: FaceObservation,
    *,
    frame_id: int,
    timestamp: float,
    target_embedding: np.ndarray,
    tier: str,
    association: str,
) -> dict[str, object] | None:
    """One row of the calibration dump: everything a gate reads, plus the score."""
    similarity = _safe_similarity(target_embedding, face.embedding)
    if similarity is None:
        return None
    return {
        "frame_id": frame_id,
        "timestamp_seconds": timestamp,
        "similarity": similarity,
        "face_px": face.short_side,
        "face_px_bucket": face_px_bucket(face.short_side),
        "detection_score": face.detection_score,
        "blur_variance": face.blur_variance,
        "quality": face.quality,
        "tier": tier,
        "association": association,
    }


if __name__ == "__main__":
    main()
