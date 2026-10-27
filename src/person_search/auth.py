"""Fixed-issuer JWT validation and process-local, one-use stream capabilities.

No user passwords, signing private keys, or media are persisted by the executor.
The executor remains single-worker: resource ownership and tickets live in memory.
"""
from __future__ import annotations

import asyncio
import hashlib
import secrets
import time
from dataclasses import dataclass
from typing import Any
from urllib.parse import parse_qsl, urlencode, urlsplit

import httpx
import jwt
from starlette.requests import HTTPConnection
from starlette.responses import JSONResponse
from starlette.types import ASGIApp, Receive, Scope, Send

from .config import Settings
from .errors import PersonSearchError


@dataclass(frozen=True)
class Principal:
    token: str
    claims: dict[str, Any]

    @property
    def lease_id(self) -> str | None:
        value = self.claims.get("lease_id")
        return value if isinstance(value, str) and value else None

    @property
    def expires_at(self) -> float:
        return float(self.claims["exp"])


def _error(code: str = "invalid_token", status: int = 401) -> PersonSearchError:
    return PersonSearchError("access denied", code=code, status_code=status)


def _resource(path: str, query: str = "") -> str:
    pairs = sorted((k, v) for k, v in parse_qsl(query, keep_blank_values=True) if k != "ticket")
    return path + ("?" + urlencode(pairs) if pairs else "")


class ExecutorAuthorization:
    def __init__(self, settings: Settings, client: httpx.AsyncClient | None = None):
        self.settings = settings
        self.client = client or httpx.AsyncClient(
            timeout=5.0, trust_env=False, verify=str(settings.auth_ca_file) if settings.auth_ca_file else True,
        )
        self._own_client = client is None
        self._jwks: dict[str, Any] = {}
        self._jwks_until = 0.0
        self._last_refresh = 0.0
        self._refresh_lock = asyncio.Lock()
        self._statuses: dict[str, tuple[float, bool]] = {}
        self._tickets: dict[str, tuple[float, str, str, Principal]] = {}
        self.target_owners: dict[str, str] = {}
        self.search_owners: dict[str, str] = {}
        self.closing_leases: set[str] = set()
        self.lease_locks: dict[str, asyncio.Lock] = {}

    async def close(self) -> None:
        if self._own_client:
            await self.client.aclose()
        self._tickets.clear()
        self._statuses.clear()

    async def _load_keys(self, kid: str) -> Any:
        now = time.monotonic()
        if now >= self._jwks_until or kid not in self._jwks:
            async with self._refresh_lock:
                now = time.monotonic()
                if now >= self._jwks_until or kid not in self._jwks:
                    if self._last_refresh and now - self._last_refresh < 1:
                        raise _error()
                    self._last_refresh = now
                    try:
                        response = await self.client.get(self.settings.auth_jwks_url)
                        response.raise_for_status()
                        keys = response.json()["keys"]
                        if not isinstance(keys, list) or len(keys) > 20:
                            raise ValueError("invalid keyset")
                        parsed: dict[str, Any] = {}
                        for item in keys:
                            if item.get("kty") != "RSA" or item.get("use", "sig") != "sig":
                                continue
                            if item.get("alg", "RS256") != "RS256":
                                continue
                            key_id = item.get("kid")
                            if not isinstance(key_id, str) or key_id in parsed:
                                raise ValueError("invalid key id")
                            parsed[key_id] = jwt.PyJWK.from_dict(item).key
                        if not parsed:
                            raise ValueError("empty keyset")
                    except (httpx.HTTPError, ValueError, KeyError, TypeError, AttributeError, jwt.PyJWTError):
                        raise _error("auth_unavailable", 503) from None
                    self._jwks = parsed
                    self._jwks_until = now + 300
        key = self._jwks.get(kid)
        if key is None:
            raise _error()
        return key

    async def authenticate(self, token: str) -> Principal:
        if not token or len(token) > 16384:
            raise _error()
        if not (self.settings.auth_issuer and self.settings.auth_jwks_url and self.settings.auth_status_url):
            raise _error("auth_not_configured", 503)
        try:
            header = jwt.get_unverified_header(token)
            kid = header.get("kid")
            if header.get("alg") != "RS256" or not isinstance(kid, str) or not kid:
                raise _error()
            key = await self._load_keys(kid)
            claims = jwt.decode(
                token, key, algorithms=["RS256"], issuer=self.settings.auth_issuer,
                audience=self.settings.auth_audience or f"insightface:{self.settings.executor_id}",
                options={"require": ["exp", "iat", "iss", "aud", "sub", "jti", "type"]},
            )
        except jwt.PyJWTError:
            raise _error() from None
        if (
            claims.get("type") not in {"user", "service"}
            or claims.get("executor_id") != self.settings.executor_id
            or not isinstance(claims.get("scopes"), list)
            or not all(isinstance(value, str) for value in claims["scopes"])
            or (claims["type"] == "user" and not isinstance(claims.get("sid"), str))
        ):
            raise _error()
        principal = Principal(token, claims)
        if not await self.is_active(principal):
            raise _error()
        return principal

    async def is_active(self, principal: Principal) -> bool:
        if principal.expires_at <= time.time():
            return False
        digest = hashlib.sha256(principal.token.encode()).hexdigest()
        now = time.monotonic()
        cached = self._statuses.get(digest)
        if cached and cached[0] > now:
            return cached[1]
        try:
            response = await self.client.post(
                self.settings.auth_status_url,
                headers={"Authorization": f"Bearer {principal.token}"},
            )
            response.raise_for_status()
            active = response.json().get("active") is True
        except (httpx.HTTPError, ValueError, TypeError, AttributeError):
            return False
        if len(self._statuses) > 4096:
            self._statuses = {key: value for key, value in self._statuses.items() if value[0] > now}
        self._statuses[digest] = (now + min(30, max(0, principal.expires_at - time.time())), active)
        return active

    def require(self, principal: Principal, permission: str, *, write: bool = False) -> None:
        if permission not in principal.claims["scopes"]:
            raise _error("insufficient_scope", 403)
        if write and not principal.lease_id:
            raise _error("lease_required", 403)

    def own(self, principal: Principal | None, kind: str, resource_id: str) -> None:
        if not self.settings.auth_enabled:
            return
        if principal is None:
            raise _error()
        owners = self.target_owners if kind == "target" else self.search_owners
        owner = owners.get(resource_id)
        # A lease-free read-only administrative token may inspect status; every
        # resource mutation must already carry a lease and match that owner.
        if principal.lease_id and owner != principal.lease_id:
            raise _error("lease_mismatch", 403)

    def bind(self, principal: Principal | None, kind: str, resource_id: str) -> None:
        if principal and principal.lease_id:
            owners = self.target_owners if kind == "target" else self.search_owners
            if resource_id in owners and owners[resource_id] != principal.lease_id:
                raise _error("lease_mismatch", 403)
            owners[resource_id] = principal.lease_id

    def issue_ticket(self, principal: Principal, path: str, purpose: str) -> dict[str, Any]:
        selected = urlsplit(path)
        if selected.scheme or selected.netloc or selected.fragment or not selected.path.startswith("/v1/searches/"):
            raise _error("invalid_ticket_resource", 422)
        suffix = "/events" if purpose == "websocket" else "/preview.mjpg" if purpose == "preview" else ""
        if not suffix or not selected.path.endswith(suffix):
            raise _error("invalid_ticket_resource", 422)
        if any(key == "ticket" for key, _ in parse_qsl(selected.query)):
            raise _error("invalid_ticket_resource", 422)
        self.require(principal, "executor:read")
        search_id = selected.path.removeprefix("/v1/searches/").split("/")[0]
        self.own(principal, "search", search_id)
        now = time.monotonic()
        self._tickets = {key: value for key, value in self._tickets.items() if value[0] > now}
        if len(self._tickets) >= 4096:
            raise _error("ticket_capacity_exceeded", 429)
        ticket = secrets.token_urlsafe(32)
        expires_in = min(30, max(0, int(principal.expires_at - time.time())))
        if not expires_in:
            raise _error()
        digest = hashlib.sha256(ticket.encode()).hexdigest()
        self._tickets[digest] = (now + expires_in, _resource(selected.path, selected.query), purpose, principal)
        return {"ticket": ticket, "expires_in": expires_in}

    async def connection_principal(self, connection: HTTPConnection) -> Principal:
        token_header = connection.headers.get("authorization", "")
        if token_header.startswith("Bearer "):
            return await self.authenticate(token_header[7:])
        ticket = connection.query_params.get("ticket")
        if connection.scope["type"] == "websocket":
            for protocol in connection.scope.get("subprotocols", []):
                if protocol.startswith("ticket."):
                    ticket = protocol[7:]
                    break
        if not ticket or len(ticket) > 128:
            raise _error()
        digest = hashlib.sha256(ticket.encode()).hexdigest()
        item = self._tickets.pop(digest, None)
        purpose = "websocket" if connection.scope["type"] == "websocket" else "preview"
        resource = _resource(connection.url.path, connection.url.query)
        if item is None or item[0] <= time.monotonic() or item[1] != resource or item[2] != purpose:
            raise _error()
        if not await self.is_active(item[3]):
            raise _error()
        return item[3]


class ExecutorAuthMiddleware:
    def __init__(self, app: ASGIApp, authorization: ExecutorAuthorization):
        self.app = app
        self.authorization = authorization

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        if scope["type"] not in {"http", "websocket"} or not self.authorization.settings.auth_enabled:
            await self.app(scope, receive, send)
            return
        # The static shell contains no data and performs central login itself.
        if scope["type"] == "http" and scope["path"] in {"/", "/monitor", "/livez"}:
            await self.app(scope, receive, send)
            return
        try:
            connection = HTTPConnection(scope)
            origin = connection.headers.get("origin")
            if origin:
                trusted = {f"https://{connection.headers.get('host', '')}"}
                for value in (self.authorization.settings.auth_issuer, self.authorization.settings.auth_jwks_url, self.authorization.settings.auth_status_url):
                    parsed = urlsplit(value)
                    if parsed.scheme == "https" and parsed.netloc:
                        trusted.add(f"https://{parsed.netloc}")
                if origin not in trusted:
                    raise _error("invalid_origin", 403)
            principal = await self.authorization.connection_principal(connection)
            path, method = scope["path"], scope.get("method", "GET")
            write = method not in {"GET", "HEAD", "OPTIONS"} and path != "/v1/auth/tickets"
            if method == "DELETE" and path.startswith("/v1/leases/") and path.endswith("/resources"):
                self.authorization.require(principal, "executor:cleanup", write=True)
                if principal.claims["type"] != "service":
                    raise _error("insufficient_scope", 403)
            elif "/evidence/" in path:
                permission = "evidence:release" if method == "DELETE" else "evidence:read"
                self.authorization.require(principal, permission, write=method == "DELETE")
            else:
                self.authorization.require(principal, "executor:write" if write else "executor:read", write=write)
            scope.setdefault("state", {})["principal"] = principal
        except PersonSearchError as exc:
            if scope["type"] == "websocket":
                await send({"type": "websocket.close", "code": 1008, "reason": "access denied"})
            else:
                await JSONResponse(
                    {"detail": {"code": exc.code, "message": "access denied"}},
                    status_code=exc.status_code,
                    headers={"WWW-Authenticate": "Bearer", "Cache-Control": "no-store"},
                )(scope, receive, send)
            return
        if write and principal.lease_id:
            lock = self.authorization.lease_locks.setdefault(principal.lease_id, asyncio.Lock())
            async with lock:
                if principal.lease_id in self.authorization.closing_leases and path != f"/v1/leases/{principal.lease_id}/resources":
                    await JSONResponse({"detail": {"code": "lease_closing", "message": "lease is closing"}}, status_code=409)(scope, receive, send)
                    return
                await self.app(scope, receive, send)
            return
        await self.app(scope, receive, send)
