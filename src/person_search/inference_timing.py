"""Attribute backend timings to the search worker that made the model call."""

from __future__ import annotations

from collections.abc import Callable, Iterator
from contextlib import contextmanager
from contextvars import ContextVar

TimingSink = Callable[[str, float], None]
_active_timing: ContextVar[tuple[str, TimingSink] | None] = ContextVar(
    "person_search_inference_timing", default=None
)


@contextmanager
def inference_timing(stage: str, sink: TimingSink) -> Iterator[None]:
    token = _active_timing.set((stage, sink))
    try:
        yield
    finally:
        _active_timing.reset(token)


def record_inference_timing(part: str, elapsed_seconds: float) -> None:
    active = _active_timing.get()
    if active is not None:
        stage, sink = active
        sink(f"{stage}.{part}", elapsed_seconds * 1000.0)
