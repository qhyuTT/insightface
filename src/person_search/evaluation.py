from __future__ import annotations

import json
import math
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

DEFAULT_EVAL_THRESHOLDS = (0.5, 0.55, 0.6, 0.65)
MIN_INTERVAL_RECALL = 0.9
MAX_FALSE_CONFIRMATIONS_PER_HOUR = 0.01
MIN_NEGATIVE_EXPOSURE_HOURS = 100.0
FACE_PX_BUCKETS = {"<48", "48-55", "56-63", "64-79", ">=80"}
_CASE_ID_PATTERN = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]*$")


@dataclass(frozen=True, slots=True)
class EvaluationCase:
    case_id: str
    photo: Path
    video: Path
    target_name: str
    expected_intervals_seconds: tuple[tuple[float, float], ...]
    expected_face_px_buckets: tuple[str | None, ...] = ()


def load_manifest(path: Path) -> list[EvaluationCase]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"cannot read evaluation manifest: {exc}") from exc
    if not isinstance(payload, dict) or payload.get("version") not in (1, 2):
        raise ValueError("evaluation manifest version must be 1 or 2")
    manifest_version = int(payload["version"])
    raw_cases = payload.get("cases")
    if not isinstance(raw_cases, list) or not raw_cases:
        raise ValueError("evaluation manifest must contain at least one case")

    base_dir = path.resolve().parent
    cases: list[EvaluationCase] = []
    seen_ids: set[str] = set()
    for index, raw_case in enumerate(raw_cases):
        if not isinstance(raw_case, dict):
            raise TypeError(f"case {index} must be an object")
        case_id = raw_case.get("id")
        if not isinstance(case_id, str) or not _CASE_ID_PATTERN.fullmatch(case_id):
            raise ValueError(f"case {index} has an invalid id")
        if case_id in seen_ids:
            raise ValueError(f"duplicate case id: {case_id}")
        seen_ids.add(case_id)
        target_name = raw_case.get("target_name")
        if not isinstance(target_name, str) or not target_name.strip():
            raise ValueError(f"case {case_id} must have a target_name")
        photo = _resolve_manifest_path(base_dir, raw_case.get("photo"), case_id, "photo")
        video = _resolve_manifest_path(base_dir, raw_case.get("video"), case_id, "video")
        intervals, face_px_buckets = validate_manifest_intervals(
            raw_case.get("expected_intervals_seconds"), case_id, manifest_version
        )
        cases.append(
            EvaluationCase(
                case_id=case_id,
                photo=photo,
                video=video,
                target_name=target_name.strip(),
                expected_intervals_seconds=intervals,
                expected_face_px_buckets=face_px_buckets,
            )
        )
    return cases


def validate_thresholds(values: list[float] | tuple[float, ...]) -> tuple[float, ...]:
    thresholds = tuple(float(value) for value in values)
    if not thresholds:
        raise ValueError("at least one similarity threshold is required")
    if any(not math.isfinite(value) or value < -1.0 or value > 1.0 for value in thresholds):
        raise ValueError("similarity thresholds must be finite values between -1 and 1")
    if len(set(thresholds)) != len(thresholds):
        raise ValueError("similarity thresholds must be unique")
    return thresholds


def validate_intervals(value: Any, case_id: str = "case") -> tuple[tuple[float, float], ...]:
    if not isinstance(value, list):
        raise TypeError(f"case {case_id} expected_intervals_seconds must be a list")
    intervals: list[tuple[float, float]] = []
    for index, raw_interval in enumerate(value):
        if not isinstance(raw_interval, list | tuple) or len(raw_interval) != 2:
            raise ValueError(f"case {case_id} interval {index} must contain start and end")
        start, end = (float(item) for item in raw_interval)
        if not math.isfinite(start) or not math.isfinite(end) or start < 0 or end <= start:
            raise ValueError(f"case {case_id} interval {index} is invalid")
        if intervals and start < intervals[-1][1]:
            raise ValueError(f"case {case_id} intervals must be sorted and non-overlapping")
        intervals.append((start, end))
    return tuple(intervals)


def validate_manifest_intervals(
    value: Any, case_id: str, manifest_version: int
) -> tuple[tuple[tuple[float, float], ...], tuple[str | None, ...]]:
    if not isinstance(value, list):
        raise TypeError(f"case {case_id} expected_intervals_seconds must be a list")
    if manifest_version == 1:
        intervals = validate_intervals(value, case_id)
        return intervals, (None,) * len(intervals)

    raw_intervals: list[tuple[float, float]] = []
    buckets: list[str | None] = []
    for index, raw_interval in enumerate(value):
        if not isinstance(raw_interval, dict):
            raise TypeError(f"case {case_id} interval {index} must be an object")
        unknown_fields = set(raw_interval) - {"start", "end", "face_px_bucket"}
        if unknown_fields:
            unknown = ", ".join(sorted(unknown_fields))
            raise ValueError(f"case {case_id} interval {index} has unknown fields: {unknown}")
        bucket = raw_interval.get("face_px_bucket")
        if not isinstance(bucket, str) or not bucket.strip():
            raise ValueError(f"case {case_id} interval {index} must have a face_px_bucket")
        bucket = bucket.strip()
        if bucket not in FACE_PX_BUCKETS:
            allowed = ", ".join(sorted(FACE_PX_BUCKETS))
            raise ValueError(
                f"case {case_id} interval {index} face_px_bucket must be one of: {allowed}"
            )
        try:
            start = float(raw_interval["start"])
            end = float(raw_interval["end"])
        except (KeyError, TypeError, ValueError) as exc:
            raise ValueError(
                f"case {case_id} interval {index} must contain numeric start and end"
            ) from exc
        raw_intervals.append((start, end))
        buckets.append(bucket)
    intervals = validate_intervals(raw_intervals, case_id)
    return intervals, tuple(buckets)


def summarize_events(
    events: list[dict[str, object]],
    intervals: tuple[tuple[float, float], ...],
    duration_seconds: float,
    face_px_buckets: tuple[str | None, ...] | None = None,
    *,
    confirmation_state: str = "confirmed",
) -> dict[str, object]:
    if face_px_buckets is None:
        face_px_buckets = (None,) * len(intervals)
    if len(face_px_buckets) != len(intervals):
        raise ValueError("face_px_buckets must align with expected intervals")
    confirmed_at = [
        float(event["timestamp_seconds"])
        for event in events
        if event.get("state") == confirmation_state
    ]
    detected_intervals = 0
    confirmation_latencies: list[float] = []
    bucket_counts: dict[str, dict[str, object]] = {}
    for (start, end), bucket in zip(intervals, face_px_buckets, strict=True):
        matches = [timestamp for timestamp in confirmed_at if start <= timestamp < end]
        bucket_metrics = None
        if bucket is not None:
            bucket_metrics = bucket_counts.setdefault(
                bucket,
                {
                    "expected_intervals": 0,
                    "detected_intervals": 0,
                    "confirmation_latencies_seconds": [],
                },
            )
            bucket_metrics["expected_intervals"] = int(bucket_metrics["expected_intervals"]) + 1
        if bucket == "<48":
            if bucket_metrics is not None:
                bucket_metrics["unexpected_confirmations"] = int(
                    bucket_metrics.get("unexpected_confirmations", 0)
                ) + len(matches)
            continue
        if matches:
            detected_intervals += 1
            latency = min(matches) - start
            confirmation_latencies.append(latency)
            if bucket_metrics is not None:
                bucket_metrics["detected_intervals"] = int(bucket_metrics["detected_intervals"]) + 1
                bucket_metrics["confirmation_latencies_seconds"].append(latency)  # type: ignore[union-attr]

    bucket_metrics_result: dict[str, dict[str, object]] = {}
    for bucket, item in bucket_counts.items():
        expected = int(item["expected_intervals"])
        if bucket == "<48":
            unexpected = int(item.get("unexpected_confirmations", 0))
            bucket_metrics_result[bucket] = {
                "expected_intervals": expected,
                "unexpected_confirmations": unexpected,
                "passed": unexpected == 0,
            }
            continue
        detected = int(item["detected_intervals"])
        latencies = [float(value) for value in item["confirmation_latencies_seconds"]]
        bucket_metrics_result[bucket] = {
            **item,
            "interval_recall": detected / expected if expected else None,
            "mean_confirmation_latency_seconds": (
                sum(latencies) / len(latencies) if latencies else None
            ),
            "p95_confirmation_latency_seconds": _percentile_95(latencies),
        }

    confirmable_intervals = [
        interval
        for interval, bucket in zip(intervals, face_px_buckets, strict=True)
        if bucket != "<48"
    ]
    false_confirmations = sum(
        not any(start <= timestamp < end for start, end in confirmable_intervals)
        for timestamp in confirmed_at
    )
    target_seconds = sum(end - start for start, end in confirmable_intervals)
    negative_seconds = max(0.0, duration_seconds - target_seconds)
    recall = detected_intervals / len(confirmable_intervals) if confirmable_intervals else None
    false_rate = false_confirmations / (negative_seconds / 3600.0) if negative_seconds > 0 else None
    return {
        "expected_intervals": len(confirmable_intervals),
        "detected_intervals": detected_intervals,
        "interval_recall": recall,
        "false_confirmations": false_confirmations,
        "negative_exposure_seconds": negative_seconds,
        "false_confirmations_per_hour": false_rate,
        "confirmation_latencies_seconds": confirmation_latencies,
        "face_px_buckets": bucket_metrics_result,
    }


def aggregate_threshold_results(
    case_results: list[dict[str, object]],
    thresholds: tuple[float, ...],
    *,
    metrics_key: str = "metrics",
) -> dict[str, dict[str, object]]:
    aggregate: dict[str, dict[str, object]] = {}
    for threshold in thresholds:
        key = threshold_key(threshold)
        summaries = [
            case["threshold_results"][key][metrics_key]  # type: ignore[index]
            for case in case_results
        ]
        expected = sum(int(item["expected_intervals"]) for item in summaries)
        detected = sum(int(item["detected_intervals"]) for item in summaries)
        false_confirmations = sum(int(item["false_confirmations"]) for item in summaries)
        negative_seconds = sum(float(item["negative_exposure_seconds"]) for item in summaries)
        latencies = [
            float(latency)
            for item in summaries
            for latency in item["confirmation_latencies_seconds"]
        ]
        bucket_metrics: dict[str, dict[str, object]] = {}
        bucket_names = sorted(
            {bucket for item in summaries for bucket in item.get("face_px_buckets", {})}
        )
        for bucket in bucket_names:
            items = [
                item["face_px_buckets"][bucket]
                for item in summaries
                if bucket in item.get("face_px_buckets", {})
            ]
            bucket_expected = sum(int(item["expected_intervals"]) for item in items)
            if bucket == "<48":
                unexpected = sum(int(item.get("unexpected_confirmations", 0)) for item in items)
                bucket_metrics[bucket] = {
                    "expected_intervals": bucket_expected,
                    "unexpected_confirmations": unexpected,
                    "passed": unexpected == 0,
                }
                continue
            bucket_detected = sum(int(item["detected_intervals"]) for item in items)
            bucket_latencies = [
                float(latency)
                for item in items
                for latency in item["confirmation_latencies_seconds"]
            ]
            bucket_recall = bucket_detected / bucket_expected if bucket_expected else None
            bucket_metrics[bucket] = {
                "expected_intervals": bucket_expected,
                "detected_intervals": bucket_detected,
                "interval_recall": bucket_recall,
                "mean_confirmation_latency_seconds": (
                    sum(bucket_latencies) / len(bucket_latencies) if bucket_latencies else None
                ),
                "p95_confirmation_latency_seconds": _percentile_95(bucket_latencies),
                "passed": bucket_recall is not None and bucket_recall >= MIN_INTERVAL_RECALL,
            }
        recall = detected / expected if expected else None
        false_rate = (
            false_confirmations / (negative_seconds / 3600.0) if negative_seconds > 0 else None
        )
        sufficient_exposure = negative_seconds >= MIN_NEGATIVE_EXPOSURE_HOURS * 3600.0
        bucket_acceptance_passed = all(
            bool(item.get("passed", False)) for item in bucket_metrics.values()
        )
        passed = bool(
            expected
            and recall is not None
            and recall >= MIN_INTERVAL_RECALL
            and false_rate is not None
            and false_rate <= MAX_FALSE_CONFIRMATIONS_PER_HOUR
            and sufficient_exposure
            and bucket_acceptance_passed
        )
        aggregate[key] = {
            "threshold": threshold,
            "expected_intervals": expected,
            "detected_intervals": detected,
            "interval_recall": recall,
            "false_confirmations": false_confirmations,
            "negative_exposure_hours": negative_seconds / 3600.0,
            "false_confirmations_per_hour": false_rate,
            "mean_confirmation_latency_seconds": (
                sum(latencies) / len(latencies) if latencies else None
            ),
            "p95_confirmation_latency_seconds": _percentile_95(latencies),
            "face_px_buckets": bucket_metrics,
            "face_px_buckets_passed": bucket_acceptance_passed,
            "negative_exposure_sufficient": sufficient_exposure,
            "passed": passed,
        }
    return aggregate


def recommend_threshold(aggregate: dict[str, dict[str, object]]) -> float | None:
    passing = [item for item in aggregate.values() if item["passed"]]
    if not passing:
        return None
    passing.sort(
        key=lambda item: (
            -float(item["interval_recall"]),
            float(item["false_confirmations_per_hour"]),
            float(item["mean_confirmation_latency_seconds"]),
            -float(item["threshold"]),
        )
    )
    return float(passing[0]["threshold"])


def threshold_key(threshold: float) -> str:
    return repr(float(threshold))


def _percentile_95(values: list[float]) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    index = max(0, math.ceil(0.95 * len(ordered)) - 1)
    return ordered[index]


def _resolve_manifest_path(base_dir: Path, value: Any, case_id: str, field: str) -> Path:
    if not isinstance(value, str) or not value:
        raise ValueError(f"case {case_id} must have a {field} path")
    candidate = Path(value)
    return candidate if candidate.is_absolute() else (base_dir / candidate).resolve()
