from __future__ import annotations

import argparse
import json
import time
from collections import Counter
from pathlib import Path

import cv2

from .backends import InsightFaceBackend
from .config import Settings
from .confirmation import TrackConfirmation, normalize_bbox
from .detector import YoloXOnnxDetector
from .domain import MatchState
from .evaluation import (
    DEFAULT_EVAL_THRESHOLDS,
    MAX_FALSE_CONFIRMATIONS_PER_HOUR,
    MIN_INTERVAL_RECALL,
    MIN_NEGATIVE_EXPOSURE_HOURS,
    aggregate_threshold_results,
    load_manifest,
    recommend_threshold,
    summarize_events,
    threshold_key,
    validate_thresholds,
)
from .service import SearchManager
from .tracker import ByteTracker


def main() -> None:
    parser = argparse.ArgumentParser(description="Run person search on prerecorded videos")
    parser.add_argument("--photo", type=Path, help="single-face target photo")
    parser.add_argument("--name", default=None, help="display name for the target")
    parser.add_argument("--video", type=Path, help="input video")
    parser.add_argument("--manifest", type=Path, help="version 1 batch evaluation manifest")
    parser.add_argument("--output-dir", type=Path, default=Path("artifacts/eval"))
    parser.add_argument("--threshold", type=float, default=None)
    parser.add_argument("--thresholds", type=float, nargs="+", default=None)
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
            run_manifest(args.manifest, args.output_dir, thresholds)
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
    )


def run_manifest(manifest_path: Path, output_dir: Path, thresholds: tuple[float, ...]) -> None:
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
            print_summary=False,
        )
        result["case_id"] = case.case_id
        case_results.append(result)
        case_index.append(
            {"case_id": case.case_id, "report": str(case_output / "report.json")}
        )

    aggregate = aggregate_threshold_results(case_results, thresholds)
    recommendation = recommend_threshold(aggregate)
    report = {
        "schema_version": 1,
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
    print_summary: bool = True,
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

    confirmations = {
        threshold_key(value): TrackConfirmation(
            settings.model_copy(update={"similarity_threshold": value})
        )
        for value in selected_thresholds
    }
    events_by_threshold: dict[str, list[dict[str, object]]] = {
        key: [] for key in confirmations
    }
    annotation_key = min(
        confirmations,
        key=lambda key: abs(float(key) - settings.similarity_threshold),
    )
    rejection_counts: Counter[str] = Counter()
    face_observations = 0
    accepted_faces = 0
    tracker = ByteTracker()
    tracks = []
    frame_id = 0
    started = time.monotonic()
    try:
        while True:
            ok, frame = capture.read()
            if not ok:
                break
            timestamp = frame_id / max(fps, 1.0)
            if frame_id % max(1, round(fps / settings.person_detection_hz_cpu)) == 0:
                tracks = tracker.update(detector.detect(frame))
            faces = []
            if frame_id % max(1, round(fps / settings.face_detection_hz_cpu)) == 0:
                faces = face_backend.analyze(frame, enrollment=False)
                face_observations += len(faces)
                accepted_faces += sum(face.accepted for face in faces)
                rejection_counts.update(
                    reason for face in faces for reason in face.rejection_reasons
                )

            decisions_by_threshold = {
                key: confirmation.process(
                    frame_id=frame_id,
                    timestamp=timestamp,
                    frame_shape=frame.shape,
                    tracks=tracks,
                    faces=faces,
                    target=target,
                )
                for key, confirmation in confirmations.items()
            }
            for key, decisions in decisions_by_threshold.items():
                for decision in decisions:
                    events_by_threshold[key].append(
                        {
                            "frame_id": frame_id,
                            "timestamp_seconds": timestamp,
                            "state": decision.state.value,
                            "track_id": decision.track_id,
                            "bbox": normalize_bbox(decision.bbox, frame.shape),
                            "similarity": decision.similarity,
                            "quality": decision.quality,
                            "evidence_count": decision.evidence_count,
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
                color = (
                    (0, 255, 0)
                    if decision.state == MatchState.CONFIRMED
                    else (0, 165, 255)
                )
                x1, y1, x2, y2 = decision.bbox.astype(int)
                cv2.rectangle(frame, (x1, y1), (x2, y2), color, 3)
                cv2.putText(
                    frame,
                    f"{decision.state.value} {decision.similarity:.3f}",
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
            "events": events,
        }
        if expected_intervals is not None:
            result["metrics"] = summarize_events(events, expected_intervals, duration_seconds)
        threshold_results[key] = result

    summary: dict[str, object] = {
        "schema_version": 1,
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
        },
        "threshold_results": threshold_results,
    }
    if len(selected_thresholds) == 1:
        only_result = threshold_results[threshold_key(selected_thresholds[0])]
        summary.update(
            {
                "similarity_threshold": selected_thresholds[0],
                "confirmed_events": only_result["confirmed_events"],
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


def _write_report(path: Path, payload: dict[str, object]) -> None:
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
