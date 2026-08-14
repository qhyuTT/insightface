from __future__ import annotations

from types import SimpleNamespace

import cv2
import numpy as np
from conftest import FakeFaceBackend, FakePersonDetector, make_face
from fastapi.testclient import TestClient

from person_search.api import create_app
from person_search.config import Settings
from person_search.domain import SearchStatus
from person_search.service import PreviewHub, SearchManager, SearchSession


def client_with_face() -> TestClient:
    manager = SearchManager(Settings(), FakeFaceBackend([make_face()]), FakePersonDetector())
    return TestClient(create_app(Settings(), manager))


def test_health_does_not_load_models() -> None:
    with client_with_face() as client:
        assert client.get("/healthz").json() == {"status": "ok"}


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
