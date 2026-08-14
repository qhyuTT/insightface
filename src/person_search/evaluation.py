from __future__ import annotations

import json
import math
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

DEFAULT_EVAL_THRESHOLDS = (0.5, 0.55, 0.6, 0.65)
MIN_INTERVAL_RECALL = 0.9
MAX_FALSE_CONFIRMATIONS_PER_HOUR = 0.1
MIN_NEGATIVE_EXPOSURE_HOURS = 10.0
_CASE_ID_PATTERN = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]*$")


@dataclass(frozen=True, slots=True)
class EvaluationCase:
    case_id: str
    photo: Path
    video: Path
    target_name: str
    expected_intervals_seconds: tuple[tuple[float, float], ...]


def load_manifest(path: Path) -> list[EvaluationCase]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"cannot read evaluation manifest: {exc}") from exc
    if not isinstance(payload, dict) or payload.get("version") != 1:
        raise ValueError("evaluation manifest version must be 1")
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
        intervals = validate_intervals(raw_case.get("expected_intervals_seconds"), case_id)
        cases.append(
            EvaluationCase(
                case_id=case_id,
                photo=photo,
                video=video,
                target_name=target_name.strip(),
                expected_intervals_seconds=intervals,
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


def summarize_events(
    events: list[dict[str, object]],
    intervals: tuple[tuple[float, float], ...],
    duration_seconds: float,
) -> dict[str, object]:
    confirmed_at = [
        float(event["timestamp_seconds"])
        for event in events
        if event.get("state") == "confirmed"
    ]
    detected_intervals = 0
    confirmation_latencies: list[float] = []
    for start, end in intervals:
        matches = [timestamp for timestamp in confirmed_at if start <= timestamp < end]
        if matches:
            detected_intervals += 1
            confirmation_latencies.append(min(matches) - start)

    false_confirmations = sum(
        not any(start <= timestamp < end for start, end in intervals)
        for timestamp in confirmed_at
    )
    target_seconds = sum(end - start for start, end in intervals)
    negative_seconds = max(0.0, duration_seconds - target_seconds)
    recall = detected_intervals / len(intervals) if intervals else None
    false_rate = (
        false_confirmations / (negative_seconds / 3600.0) if negative_seconds > 0 else None
    )
    return {
        "expected_intervals": len(intervals),
        "detected_intervals": detected_intervals,
        "interval_recall": recall,
        "false_confirmations": false_confirmations,
        "negative_exposure_seconds": negative_seconds,
        "false_confirmations_per_hour": false_rate,
        "confirmation_latencies_seconds": confirmation_latencies,
    }


def aggregate_threshold_results(
    case_results: list[dict[str, object]], thresholds: tuple[float, ...]
) -> dict[str, dict[str, object]]:
    aggregate: dict[str, dict[str, object]] = {}
    for threshold in thresholds:
        key = threshold_key(threshold)
        summaries = [case["threshold_results"][key]["metrics"] for case in case_results]  # type: ignore[index]
        expected = sum(int(item["expected_intervals"]) for item in summaries)
        detected = sum(int(item["detected_intervals"]) for item in summaries)
        false_confirmations = sum(int(item["false_confirmations"]) for item in summaries)
        negative_seconds = sum(float(item["negative_exposure_seconds"]) for item in summaries)
        latencies = [
            float(latency)
            for item in summaries
            for latency in item["confirmation_latencies_seconds"]
        ]
        recall = detected / expected if expected else None
        false_rate = (
            false_confirmations / (negative_seconds / 3600.0) if negative_seconds > 0 else None
        )
        sufficient_exposure = negative_seconds >= MIN_NEGATIVE_EXPOSURE_HOURS * 3600.0
        passed = bool(
            expected
            and recall is not None
            and recall >= MIN_INTERVAL_RECALL
            and false_rate is not None
            and false_rate <= MAX_FALSE_CONFIRMATIONS_PER_HOUR
            and sufficient_exposure
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


def _resolve_manifest_path(base_dir: Path, value: Any, case_id: str, field: str) -> Path:
    if not isinstance(value, str) or not value:
        raise ValueError(f"case {case_id} must have a {field} path")
    candidate = Path(value)
    return candidate if candidate.is_absolute() else (base_dir / candidate).resolve()
