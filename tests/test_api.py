from __future__ import annotations

from types import MethodType, SimpleNamespace

import cv2
import numpy as np
from conftest import FakeFaceBackend, FakePersonDetector, make_face
from fastapi.testclient import TestClient

from person_search.api import create_app
from person_search.config import Settings
from person_search.domain import SearchStatus, SourceConfig
from person_search.service import PreviewHub, SearchManager, SearchSession


def client_with_face() -> TestClient:
    manager = SearchManager(Settings(), FakeFaceBackend([make_face()]), FakePersonDetector())
    return TestClient(create_app(Settings(), manager))


def test_health_does_not_load_models() -> None:
    with client_with_face() as client:
        assert client.get("/healthz").json() == {
            "status": "ok",
            "api_version": "0.2.0",
            "capabilities": [
                "replace_active",
                "active_search",
                "search_timeout",
                "request_lookup",
                "event_replay",
            ],
            "active_search": False,
        }


def test_monitor_page_is_available() -> None:
    with client_with_face() as client:
        response = client.get("/monitor")
        assert response.status_code == 200
        assert "实时寻人控制台" in response.text
        assert 'id="targetName"' in response.text
        assert "target_name" in response.text
        assert "rtsp://192.168.31.241:8554/camera" in response.text
        assert 'id="sourceType"' not in response.text
        assert 'id="cameraIndex"' not in response.text
        assert 'id="debugPreview" type="checkbox"' in response.text
        assert "debug_preview" in response.text
        assert "MATCH COUNTS" in response.text
        assert "tiny_shadow_lost" in response.text
        assert "SHADOW 命中已失效" in response.text


def test_preview_stream_returns_latest_annotated_jpeg() -> None:
    preview = PreviewHub()
    preview.publish(b"jpeg-frame")
    session = SimpleNamespace(preview=preview, status=SearchStatus.STOPPED)

    class PreviewManager:
        def get_session(self, search_id: str):
            assert search_id == "search-1"
            return session

        def shutdown(self) -> None:
            pass

    with TestClient(create_app(Settings(), PreviewManager())) as client:  # type: ignore[arg-type]
        response = client.get("/v1/searches/search-1/preview.mjpg")
    assert response.status_code == 200
    assert response.headers["content-type"].startswith("multipart/x-mixed-replace")
    assert b"jpeg-frame" in response.content


def test_create_target_and_unknown_search() -> None:
    image = np.full((200, 200, 3), 128, dtype=np.uint8)
    ok, encoded = cv2.imencode(".jpg", image)
    assert ok
    with client_with_face() as client:
        response = client.post(
            "/v1/targets",
            data={"name": "张三"},
            files={"image": ("target.jpg", encoded.tobytes(), "image/jpeg")},
        )
        assert response.status_code == 201
        assert "target_id" in response.json()
        assert response.json()["name"] == "张三"
        unknown = client.post(
            "/v1/searches",
            json={
                "target_id": "missing",
                "source": {"type": "camera", "device_index": 0},
            },
        )
        assert unknown.status_code == 404
        assert unknown.json()["detail"]["code"] == "target_not_found"


def test_delete_active_search_returns_503_until_worker_really_finishes(monkeypatch) -> None:
    settings = Settings()
    manager = SearchManager(settings, FakeFaceBackend([make_face()]), FakePersonDetector())
    frame = np.zeros((200, 200, 3), dtype=np.uint8)
    active_target = manager.enroll(frame, "张三")
    replacement_target = manager.enroll(frame, "李四")

    def start_stuck_worker(session: SearchSession) -> None:
        session._worker = object()  # type: ignore[assignment]

    monkeypatch.setattr(SearchSession, "start", start_stuck_worker)
    search = manager.start_batch_search(
        [active_target.target_id],
        source=SourceConfig(type="camera", device_index=0),
        request_id="request-stuck",
    )
    session = manager.get_session(search.search_id)
    original_stop = SearchSession.stop
    session.stop = MethodType(  # type: ignore[method-assign]
        lambda current: original_stop(current, timeout=0),
        session,
    )

    with TestClient(create_app(settings, manager)) as client:
        response = client.delete("/v1/searches/active")

        assert response.status_code == 503
        assert response.json()["detail"]["code"] == "search_stop_timeout"
        active = client.get("/v1/searches/active").json()
        assert active["search_id"] == search.search_id
        assert active["status"] == "stopping"

        replacement = client.post(
            "/v1/searches",
            json={
                "target_id": replacement_target.target_id,
                "source": {"type": "camera", "device_index": 1},
                "replace_active": True,
                "request_id": "request-replacement",
            },
        )
        assert replacement.status_code == 503
        assert replacement.json()["detail"]["code"] == "search_stop_timeout"
        assert client.get("/v1/searches/active").json()["search_id"] == search.search_id
        assert len(manager._sessions) == 1


def test_target_name_is_required() -> None:
    image = np.full((200, 200, 3), 128, dtype=np.uint8)
    ok, encoded = cv2.imencode(".jpg", image)
    assert ok
    with client_with_face() as client:
        response = client.post(
            "/v1/targets", files={"image": ("target.jpg", encoded.tobytes(), "image/jpeg")}
        )
        assert response.status_code == 422


def test_batch_search_enrolls_multiple_targets(monkeypatch) -> None:
    monkeypatch.setattr(SearchSession, "start", lambda self: None)
    image = np.full((200, 200, 3), 128, dtype=np.uint8)
    ok, encoded = cv2.imencode(".jpg", image)
    assert ok
    payload = encoded.tobytes()

    with client_with_face() as client:
        response = client.post(
            "/v1/batch-searches",
            data={
                "targets": '[{"name":"张三","image_filename":"a.jpg"},'
                '{"name":"李四","image_filename":"b.jpg"}]',
                "source": '{"type":"rtsp","uri":"rtsp://camera.test/live"}',
            },
            files=[
                ("images", ("a.jpg", payload, "image/jpeg")),
                ("images", ("b.jpg", payload, "image/jpeg")),
            ],
        )
        assert response.status_code == 201
        body = response.json()
        assert body["total_count"] == 2
        assert body["found_count"] == 0
        assert [target["name"] for target in body["targets"]] == ["张三", "李四"]
        assert len(body["unfound_target_ids"]) == 2


def test_rejects_bad_media_type_and_invalid_rtsp_scheme() -> None:
    with client_with_face() as client:
        media = client.post(
            "/v1/targets",
            data={"name": "张三"},
            files={"image": ("target.txt", b"not image", "text/plain")},
        )
        assert media.status_code == 415

        invalid = client.post(
            "/v1/searches",
            json={
                "target_id": "missing",
                "source": {"type": "rtsp", "uri": "http://camera/stream"},
            },
        )
        assert invalid.status_code == 422
        assert invalid.json()["detail"]["code"] == "invalid_source"

        file_source = client.post(
            "/v1/searches",
            json={
                "target_id": "missing",
                "source": {"type": "file", "uri": "/tmp/video.mp4"},
            },
        )
        assert file_source.status_code == 422


def test_search_can_be_reconciled_by_request_id_after_terminal_state(monkeypatch) -> None:
    monkeypatch.setattr(SearchSession, "start", lambda self: None)
    settings = Settings()
    manager = SearchManager(settings, FakeFaceBackend([make_face()]), FakePersonDetector())
    target = manager.enroll(np.zeros((200, 200, 3), dtype=np.uint8), "张三")
    search = manager.start_batch_search(
        [target.target_id],
        SourceConfig(type="camera", device_index=0),
        request_id="request-terminal-lookup",
    )
    session = manager.get_session(search.search_id)
    session._transition(SearchStatus.COMPLETED, None, publish=False)
    manager._on_finished(search.search_id, [target.target_id])

    with TestClient(create_app(settings, manager)) as client:
        response = client.get("/v1/searches/by-request/request-terminal-lookup")
        assert response.status_code == 200
        assert response.json()["search_id"] == search.search_id
        assert response.json()["request_id"] == "request-terminal-lookup"
        assert response.json()["status"] == "completed"

        missing = client.get("/v1/searches/by-request/does-not-exist")
        assert missing.status_code == 200
        assert missing.json() is None


def test_evidence_endpoint_requires_key_and_never_sets_cache_headers() -> None:
    class EvidenceSession:
        def __init__(self) -> None:
            self.releases: list[str] = []

        def get_evidence(self, evidence_id: str, variant: str) -> tuple[bytes, str]:
            assert evidence_id == "opaque-evidence-id"
            assert variant == "face_crop"
            return b"jpeg-evidence", "image/jpeg"

        def release_evidence(self, evidence_id: str) -> None:
            self.releases.append(evidence_id)

    evidence_session = EvidenceSession()

    class EvidenceManager:
        def get_session(self, search_id: str) -> EvidenceSession:
            assert search_id == "search-evidence"
            return evidence_session

        def shutdown(self) -> None:
            pass

    settings = Settings(evidence_api_key="test-evidence-key")
    with TestClient(create_app(settings, EvidenceManager())) as client:  # type: ignore[arg-type]
        rejected = client.get("/v1/searches/search-evidence/evidence/opaque-evidence-id")
        assert rejected.status_code == 403
        rejected_delete = client.delete(
            "/v1/searches/search-evidence/evidence/opaque-evidence-id"
        )
        assert rejected_delete.status_code == 403

        response = client.get(
            "/v1/searches/search-evidence/evidence/opaque-evidence-id",
            headers={"X-API-Key": "test-evidence-key"},
        )
        released = client.delete(
            "/v1/searches/search-evidence/evidence/opaque-evidence-id",
            headers={"X-API-Key": "test-evidence-key"},
        )
    assert response.status_code == 200
    assert response.content == b"jpeg-evidence"
    assert response.headers["content-type"] == "image/jpeg"
    assert response.headers["cache-control"] == "no-store"
    assert released.status_code == 204
    assert evidence_session.releases == ["opaque-evidence-id"]


def test_evidence_endpoint_is_unavailable_when_server_key_is_not_configured() -> None:
    with client_with_face() as client:
        response = client.get(
            "/v1/searches/unknown/evidence/unknown",
            headers={"X-API-Key": "caller-key"},
        )
        released = client.delete(
            "/v1/searches/unknown/evidence/unknown",
            headers={"X-API-Key": "caller-key"},
        )
    assert response.status_code == released.status_code == 503
    assert response.json()["detail"]["code"] == "evidence_access_not_configured"
    assert released.json()["detail"]["code"] == "evidence_access_not_configured"


def test_health_advertises_confirmed_evidence_only_when_configured() -> None:
    settings = Settings(evidence_api_key="test-evidence-key")
    manager = SearchManager(settings, FakeFaceBackend([make_face()]), FakePersonDetector())
    with TestClient(create_app(settings, manager)) as client:
        capabilities = client.get("/healthz").json()["capabilities"]
    assert "event_replay" in capabilities
    assert "confirmed_evidence_v1" in capabilities
