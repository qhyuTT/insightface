from __future__ import annotations

import asyncio
import time
from types import SimpleNamespace

import cv2
import httpx
import jwt
import numpy as np
import pytest
from cryptography.hazmat.primitives.asymmetric import rsa
from fastapi.testclient import TestClient
from starlette.websockets import WebSocketDisconnect

from person_search.api import create_app
from person_search.auth import ExecutorAuthorization
from person_search.config import Settings
from person_search.domain import SearchStatus, SearchView, SourceConfig, TargetView
from person_search.service import PreviewHub


class Manager:
    def __init__(self):
        self.deleted = []
        self.active = None
        self.search = SearchView(search_id="search-1", target_id="target-1", status=SearchStatus.STOPPED,
                                 source=SourceConfig(type="rtsp", uri="rtsp://localhost/live"))
        self.preview = PreviewHub()
        self.preview.publish(b"synthetic-jpeg")

    def shutdown(self):
        pass

    def active_search(self):
        return self.active

    def enroll(self, frame, name):
        return TargetView(target_id="target-1", name=name, face_width=60, face_height=60,
                          detection_score=.99, quality_score=.9, model="fake")

    def start_batch_search(self, *args):
        self.active = self.search
        return self.search

    def get_search(self, search_id):
        return self.search

    def delete_target(self, target_id):
        self.deleted.append(target_id)

    def stop_search(self, search_id):
        self.deleted.append(search_id)

    def get_session(self, search_id):
        return SimpleNamespace(
            preview=self.preview, status=SearchStatus.STOPPED,
            events=SimpleNamespace(after=lambda *args: []),
            get_evidence=lambda *args: (b"synthetic-jpeg", "image/jpeg"),
            release_evidence=lambda *args: self.deleted.append("evidence"),
        )


@pytest.fixture
def authenticated():
    private = rsa.generate_private_key(public_exponent=65537, key_size=2048)
    key = jwt.algorithms.RSAAlgorithm.to_jwk(private.public_key(), as_dict=True)
    key.update(kid="key-1", use="sig", alg="RS256")
    state = {"active": True, "status_calls": 0}

    def handle(request):
        if request.url.path == "/jwks":
            return httpx.Response(200, json={"keys": [key]})
        assert request.headers.get("authorization", "").startswith("Bearer ")
        state["status_calls"] += 1
        return httpx.Response(200, json={"active": state["active"]})

    settings = Settings(auth_enabled=True, auth_issuer="https://dispatch.example",
                        auth_jwks_url="https://dispatch.example/jwks",
                        auth_status_url="https://dispatch.example/api/v1/auth/token-status")
    remote = httpx.AsyncClient(transport=httpx.MockTransport(handle))
    authorization = ExecutorAuthorization(settings, remote)
    manager = Manager()

    def token(**overrides):
        now = int(time.time())
        claims = {"iss": settings.auth_issuer, "aud": "insightface:executor-01", "sub": "user-1", "sid": "session-1",
                  "type": "user", "ver": 1, "executor_id": "executor-01", "iat": now, "exp": now + 600,
                  "jti": "jti-1", "scopes": ["executor:read", "executor:write", "evidence:read", "evidence:release"]}
        claims.update(overrides)
        return jwt.encode(claims, private, algorithm="RS256", headers={"kid": "key-1"})

    with TestClient(create_app(settings, manager, authorization)) as client:
        yield client, authorization, manager, token, state
    asyncio.run(remote.aclose())


def bearer(token):
    return {"Authorization": f"Bearer {token}"}


def test_production_default_and_missing_configuration_are_fail_closed(monkeypatch):
    monkeypatch.delenv("PERSON_SEARCH_AUTH_ENABLED")
    settings = Settings()
    assert settings.auth_enabled is True
    with TestClient(create_app(settings, Manager())) as client:
        assert client.get("/livez").json() == {"status": "ok"}
        assert client.get("/monitor").status_code == 200
        for path in ("/healthz", "/readyz", "/v1/searches/active", "/docs", "/openapi.json"):
            assert client.get(path).status_code == 401
        assert client.get("/healthz", headers=bearer("token")).status_code == 503


@pytest.mark.parametrize("url", ["http://dispatch/jwks", "https://user:secret@dispatch/jwks"])
def test_jwks_and_status_urls_require_trusted_https(url):
    with pytest.raises(ValueError):
        Settings(auth_jwks_url=url)
    with pytest.raises(ValueError):
        Settings(auth_status_url=url)


def test_signed_identity_protects_health_and_requires_scopes(authenticated):
    client, _, _, token, _ = authenticated
    response = client.get("/healthz", headers=bearer(token()))
    assert response.status_code == 200
    assert response.json()["executor_id"] == "executor-01"
    assert response.json()["capacity"] == 1
    assert "confirmed_evidence_v1" in response.json()["capabilities"]
    assert client.get("/healthz", headers=bearer(token(scopes=[]))).status_code == 403


@pytest.mark.parametrize("overrides", [
    {"aud": "dispatch"}, {"iss": "https://attacker.example"}, {"exp": 1},
    {"executor_id": "executor-02"}, {"type": "robot"}, {"sid": None}, {"scopes": "executor:read"},
])
def test_rejects_wrong_identity_claims(authenticated, overrides):
    client, _, _, token, _ = authenticated
    assert client.get("/healthz", headers=bearer(token(**overrides))).status_code == 401


def test_rejects_modified_signature_and_cross_origin(authenticated):
    client, _, _, token, _ = authenticated
    forged = token().split(".")
    forged[1] = forged[1][:-5] + "AAAAA"
    assert client.get("/healthz", headers=bearer(".".join(forged))).status_code == 401
    assert client.get("/healthz", headers={**bearer(token()), "Origin": "https://attacker.example"}).status_code == 403


def test_resource_writes_require_owner_lease_and_cannot_steal_active_search(authenticated):
    client, auth, manager, token, _ = authenticated
    assert client.delete("/v1/targets/target-1", headers=bearer(token())).status_code == 403
    auth.target_owners["target-1"] = "lease-other"
    assert client.delete("/v1/targets/target-1", headers=bearer(token(lease_id="lease-1"))).status_code == 403
    assert manager.deleted == []
    assert client.delete("/v1/targets/target-1", headers=bearer(token(lease_id="lease-other"))).status_code == 204
    auth.target_owners["target-1"] = "lease-1"
    auth.search_owners["search-1"] = "lease-other"
    manager.active = manager.search
    response = client.post("/v1/searches", headers=bearer(token(lease_id="lease-1")), json={
        "target_id": "target-1", "source": {"type": "rtsp", "uri": "rtsp://localhost/live"}, "replace_active": True,
    })
    assert response.status_code == 403
    assert auth.search_owners["search-1"] == "lease-other"


def test_enrollment_and_search_bind_to_lease(authenticated):
    client, auth, _manager, token, _ = authenticated
    headers = bearer(token(lease_id="lease-1"))
    _, image = cv2.imencode(".png", np.zeros((80, 80, 3), dtype=np.uint8))
    response = client.post("/v1/targets", headers=headers, data={"name": "synthetic"},
                           files={"image": ("synthetic.png", image.tobytes(), "image/png")})
    assert response.status_code == 201
    assert auth.target_owners["target-1"] == "lease-1"
    response = client.post("/v1/searches", headers=headers, json={
        "target_id": "target-1", "source": {"type": "rtsp", "uri": "rtsp://localhost/live"},
    })
    assert response.status_code == 201
    assert auth.search_owners["search-1"] == "lease-1"
    assert client.get("/v1/searches/search-1", headers=bearer(token(lease_id="lease-2"))).status_code == 403


def test_evidence_uses_scopes_instead_of_shared_key(authenticated):
    client, auth, manager, token, _ = authenticated
    auth.search_owners["search-1"] = "lease-1"
    path = "/v1/searches/search-1/evidence/evidence-1"
    assert client.get(path, headers=bearer(token(scopes=["executor:read"]))).status_code == 403
    response = client.get(path, headers=bearer(token(lease_id="lease-1")))
    assert response.status_code == 200
    assert response.headers["cache-control"] == "no-store"
    assert client.delete(path, headers=bearer(token(lease_id="lease-2"))).status_code == 403
    assert client.delete(path, headers=bearer(token(lease_id="lease-1"))).status_code == 204
    assert manager.deleted == ["evidence"]


def test_ticket_is_single_use_and_bound_to_resource_and_query(authenticated):
    client, auth, _, token, _ = authenticated
    auth.search_owners["search-1"] = "lease-1"
    path = "/v1/searches/search-1/preview.mjpg?source_id=source-1"
    issued = client.post("/v1/auth/tickets", headers=bearer(token(lease_id="lease-1")), json={"path": path, "purpose": "preview"})
    assert issued.status_code == 200
    ticket = issued.json()["ticket"]
    response = client.get(path + "&ticket=" + ticket)
    assert response.status_code == 200
    assert b"synthetic-jpeg" in response.content
    assert client.get(path + "&ticket=" + ticket).status_code == 401
    ticket = client.post("/v1/auth/tickets", headers=bearer(token(lease_id="lease-1")), json={"path": path, "purpose": "preview"}).json()["ticket"]
    assert client.get(path.replace("source-1", "source-2") + "&ticket=" + ticket).status_code == 401
    assert client.post("/v1/auth/tickets", headers=bearer(token()), json={"path": "/healthz", "purpose": "preview"}).status_code == 422


def test_websocket_ticket_and_revocation(authenticated):
    client, auth, _, token, state = authenticated
    path = "/v1/searches/search-1/events?after_seq=0"
    ticket = client.post("/v1/auth/tickets", headers=bearer(token()), json={"path": path, "purpose": "websocket"}).json()["ticket"]
    with client.websocket_connect(path, subprotocols=["ticket." + ticket]) as socket:
        with pytest.raises(WebSocketDisconnect) as closed:
            socket.receive_json()
        assert closed.value.code == 1000
    with pytest.raises(WebSocketDisconnect), client.websocket_connect(path, subprotocols=["ticket." + ticket]):
        pass
    state["active"] = False
    auth._statuses.clear()
    assert client.get("/healthz", headers=bearer(token())).status_code == 401
    with pytest.raises(WebSocketDisconnect), client.websocket_connect(path, headers=bearer(token())):
        pass


def test_status_unavailable_fails_closed_and_does_not_stop_search(authenticated):
    client, auth, manager, token, _ = authenticated

    async def unavailable(*args, **kwargs):
        raise httpx.ConnectError("unavailable")

    auth.client.post = unavailable
    manager.active = manager.search
    assert client.get("/healthz", headers=bearer(token())).status_code == 401
    assert manager.active is manager.search
    assert manager.deleted == []


def test_ticket_expiration_and_parent_expiration_are_enforced(authenticated):
    client, auth, _, token, _ = authenticated
    path = "/v1/searches/search-1/preview.mjpg"
    issued = client.post("/v1/auth/tickets", headers=bearer(token()), json={"path": path, "purpose": "preview"}).json()
    digest = next(iter(auth._tickets))
    item = auth._tickets[digest]
    auth._tickets[digest] = (time.monotonic() - 1, *item[1:])
    assert client.get(path + "?ticket=" + issued["ticket"]).status_code == 401
    issued = client.post("/v1/auth/tickets", headers=bearer(token()), json={"path": path, "purpose": "preview"}).json()
    digest = next(iter(auth._tickets))
    auth._tickets[digest][3].claims["exp"] = time.time() - 1
    assert client.get(path + "?ticket=" + issued["ticket"]).status_code == 401


def test_lease_release_cleans_only_owner_resources_and_blocks_future_writes(authenticated):
    client, auth, manager, token, _ = authenticated
    auth.search_owners.update({"search-1": "lease-1", "search-2": "lease-2"})
    auth.target_owners.update({"target-1": "lease-1", "target-2": "lease-2"})
    headers = bearer(token(type="service", lease_id="lease-1", scopes=["executor:cleanup", "executor:read"]))
    assert client.delete("/v1/leases/lease-2/resources", headers=headers).status_code == 403
    assert client.delete("/v1/leases/lease-1/resources", headers=headers).status_code == 204
    assert manager.deleted == ["search-1", "target-1"]
    assert auth.search_owners == {"search-2": "lease-2"}
    assert auth.target_owners == {"target-2": "lease-2"}
    assert client.delete("/v1/targets/target-1", headers=bearer(token(lease_id="lease-1"))).status_code == 409
    assert client.delete("/v1/leases/lease-1/resources", headers=headers).status_code == 204


def test_cleanup_failure_keeps_lease_closed_and_ownership_for_retry(authenticated):
    from person_search.errors import PersonSearchError

    client, auth, manager, token, _ = authenticated
    auth.search_owners["search-1"] = "lease-1"
    headers = bearer(token(type="service", lease_id="lease-1", scopes=["executor:cleanup"]))
    original = manager.stop_search

    def failure(search_id):
        raise PersonSearchError("still stopping", code="search_stop_timeout", status_code=503)

    manager.stop_search = failure
    assert client.delete("/v1/leases/lease-1/resources", headers=headers).status_code == 503
    assert auth.search_owners == {"search-1": "lease-1"}
    assert "lease-1" in auth.closing_leases
    assert client.delete("/v1/searches/search-1", headers=bearer(token(lease_id="lease-1"))).status_code == 409
    manager.stop_search = original
    assert client.delete("/v1/leases/lease-1/resources", headers=headers).status_code == 204
    assert auth.search_owners == {}


def test_cleanup_waits_for_inflight_enrollment_before_freeing_resources(authenticated):
    from concurrent.futures import ThreadPoolExecutor, TimeoutError
    from threading import Event

    client, auth, manager, token, _ = authenticated
    started, finish = Event(), Event()
    original = manager.enroll

    def delayed(frame, name):
        started.set()
        assert finish.wait(5)
        return original(frame, name)

    manager.enroll = delayed
    _, image = cv2.imencode(".png", np.zeros((80, 80, 3), dtype=np.uint8))
    writer = bearer(token(lease_id="lease-1"))
    cleaner = bearer(token(type="service", lease_id="lease-1", scopes=["executor:cleanup"]))
    with ThreadPoolExecutor(max_workers=2) as workers:
        enrollment = workers.submit(client.post, "/v1/targets", headers=writer, data={"name": "synthetic"},
                                    files={"image": ("synthetic.png", image.tobytes(), "image/png")})
        assert started.wait(5)
        cleanup = workers.submit(client.delete, "/v1/leases/lease-1/resources", headers=cleaner)
        try:
            with pytest.raises(TimeoutError):
                cleanup.result(timeout=.1)
        finally:
            finish.set()
        assert enrollment.result(timeout=5).status_code == 201
        assert cleanup.result(timeout=5).status_code == 204
    assert manager.deleted == ["target-1"]
    assert auth.target_owners == {}
