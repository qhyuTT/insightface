from __future__ import annotations

from person_search.service import EventHub


def test_event_hub_assigns_ordered_sequence_and_replays() -> None:
    hub = EventHub(capacity=2)
    hub.publish("candidate", {"track_id": 1})
    second = hub.publish("confirmed", {"track_id": 1})
    third = hub.publish("lost", {"track_id": 1})
    assert second["seq"] == 2
    assert third["seq"] == 3
    assert [event["seq"] for event in hub.after(1, timeout=0)] == [2, 3]
