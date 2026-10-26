from __future__ import annotations

from person_search.inference_timing import inference_timing, record_inference_timing


def test_inference_timing_keeps_shared_backend_samples_with_the_calling_source() -> None:
    first: list[tuple[str, float]] = []
    second: list[tuple[str, float]] = []
    with inference_timing("person", lambda key, value: first.append((key, value))):
        record_inference_timing("lock_wait", 0.002)
        with inference_timing("face_roi", lambda key, value: second.append((key, value))):
            record_inference_timing("device_run", 0.05)
        record_inference_timing("device_run", 0.01)
    record_inference_timing("lock_wait", 1.0)
    assert first == [("person.lock_wait", 2.0), ("person.device_run", 10.0)]
    assert second == [("face_roi.device_run", 50.0)]
