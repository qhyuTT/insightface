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
    FaceMatchPolicy,
    TrackConfirmation,
    associate_faces_to_tracks_detailed,
    default_face_match_policy,
    fallback_face_match_policy,
    is_stricter_policy,
    normalize_bbox,
)
from .detector import YoloXOnnxDetector
from .domain import FaceObservation, MatchState, SearchMetrics, Track
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
    rejection_counts: Counter[str] = Counter()
    association_counts: Counter[str] = Counter()
    unassociated_faces = 0
    face_observations = 0
    accepted_faces = 0
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
    try:
        while True:
            ok, frame = capture.read()
            if not ok:
                break
            timestamp = frame_id / max(fps, 1.0)
            if frame_id % person_interval == 0:
                motion, previous_motion_gray = SearchSession._estimate_camera_motion(
                    roi_context, frame, previous_motion_gray
                )
                tracks = tracker.update(detector.detect(frame), motion=motion)
            faces = []
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
                embedded = face_backend.embed_faces(
                    frame, [face for face in faces if _is_matchable_face(face, settings)]
                )
                embedded_by_key = {tuple(face.bbox.tolist()): face for face in embedded}
                faces = [embedded_by_key.get(tuple(face.bbox.tolist()), face) for face in faces]

            accepted = [face for face in faces if _is_matchable_face(face, settings)]
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
                accepted, association_modes, settings, track_tiers
            )
            for face_index, track_id in associations.items():
                track_tiers[track_id] = canonical_policies[face_index].tier
            live_track_ids = {track.track_id for track in all_tracks}
            track_tiers = {
                key: value for key, value in track_tiers.items() if key in live_track_ids
            }
            if dump_similarities:
                similarity_samples.extend(
                    _similarity_sample(
                        face,
                        frame_id=frame_id,
                        timestamp=timestamp,
                        target_embedding=target.embedding,
                        tier=canonical_policies[face_index].tier,
                        association=association_modes.get(face_index, "unassociated"),
                    )
                    for face_index, face in enumerate(accepted)
                    if face.embedding is not None
                )

            decisions_by_threshold = {}
            for key, confirmation in confirmations.items():
                policies = _face_policies(
                    accepted, association_modes, threshold_settings[key], track_tiers
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
    finally:
        capture.release()
        writer.release()
        manager.shutdown()

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


def _confirmation_input_counts(
    faces: list[FaceObservation],
    associations: dict[int, int],
    policies: dict[int, FaceMatchPolicy],
    target_embedding: np.ndarray,
) -> Counter[str]:
    similarities = {
        face_index: float(target_embedding @ face.embedding)
        for face_index, face in enumerate(faces)
    }
    return Counter(
        {
            "above_threshold": sum(
                similarities[face_index] >= policies[face_index].threshold
                for face_index in range(len(faces))
            ),
            "evidence_eligible": sum(
                policies[face_index].accepts_observation(
                    faces[face_index].detection_score,
                    similarities[face_index],
                )
                for face_index in associations
            ),
        }
    )


def _is_matchable_face(face: FaceObservation, settings: Settings) -> bool:
    return bool(face.accepted and face.short_side >= settings.effective_search_min_face_px)


def _offline_decision_state(state: str, shadow: bool) -> str:
    return f"shadow_{state}" if shadow else state


def _write_report(path: Path, payload: dict[str, object]) -> None:
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()


def _similarity_sample(
    face: FaceObservation,
    *,
    frame_id: int,
    timestamp: float,
    target_embedding: np.ndarray,
    tier: str,
    association: str,
) -> dict[str, object]:
    """One row of the calibration dump: everything a gate reads, plus the score."""
    return {
        "frame_id": frame_id,
        "timestamp_seconds": timestamp,
        "similarity": float(target_embedding @ face.embedding),
        "face_px": face.short_side,
        "face_px_bucket": face_px_bucket(face.short_side),
        "detection_score": face.detection_score,
        "blur_variance": face.blur_variance,
        "quality": face.quality,
        "tier": tier,
        "association": association,
    }
