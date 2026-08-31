from __future__ import annotations

import asyncio
import hmac
import json
import re
import time
import uuid
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Annotated, Any
from urllib.parse import urlsplit

import cv2
import numpy as np
from fastapi import (
    FastAPI,
    File,
    Form,
    Header,
    Request,
    Response,
    UploadFile,
    WebSocket,
    WebSocketDisconnect,
)
from fastapi.exceptions import RequestValidationError
from fastapi.responses import HTMLResponse, JSONResponse, StreamingResponse

from . import API_VERSION, __version__
from .config import Settings, get_settings
from .domain import SearchCreate, SearchView, SourceConfig, SourceType, TargetView
from .errors import ModelUnavailableError, PersonSearchError
from .privacy import install_evidence_access_log_filter
from .service import SearchManager

MAX_UPLOAD_BYTES = 10 * 1024 * 1024
MAX_IMAGE_PIXELS = 20_000_000
MAX_BATCH_TARGETS = 20
ALLOWED_IMAGE_TYPES = {"image/jpeg", "image/png", "image/webp"}


def create_app(
    settings: Settings | None = None,
    manager: SearchManager | None = None,
) -> FastAPI:
    install_evidence_access_log_filter()
    settings = settings or get_settings()
    manager = manager or SearchManager(settings)

    @asynccontextmanager
    async def lifespan(_: FastAPI):
        yield
        await asyncio.to_thread(manager.shutdown)

    app = FastAPI(
        title="Robot Person Search PoC",
        version=API_VERSION,
        description="Session-scoped photo-to-live-video person search API.",
        lifespan=lifespan,
    )
    app.state.manager = manager

    @app.exception_handler(PersonSearchError)
    async def handle_person_search_error(_: Request, exc: PersonSearchError) -> JSONResponse:
        # Provider/model exceptions can contain local filesystem paths, model
        # URLs, or credentials echoed by a downloader.  Keep those details in
        # server-side logs only; the public error contract needs a stable,
        # non-sensitive message.
        message = (
            "model unavailable"
            if isinstance(exc, ModelUnavailableError)
            else _redact_validation_text(exc.message)
        )
        return _problem(exc.status_code, exc.code, message)

    @app.exception_handler(RequestValidationError)
    async def handle_validation_error(_: Request, exc: RequestValidationError) -> JSONResponse:
        fields = _safe_validation_fields(exc)
        # SourceConfig performs strict URI validation before the route handler is
        # entered.  Preserve the route's longstanding ``invalid_source`` error
        # code for those failures while keeping unrelated malformed requests under
        # the generic validation code.
        code = "invalid_source" if _contains_source_validation_error(fields) else "validation_error"
        return _problem(422, code, "request validation failed", fields)

    @app.get("/healthz")
    async def health() -> dict[str, Any]:
        active = await asyncio.to_thread(manager.active_search)
        capabilities = [
            "replace_active",
            "active_search",
            "search_timeout",
            "request_lookup",
            "event_replay",
        ]
        if settings.evidence_api_key:
            capabilities.append("confirmed_evidence_v1")
        return {
            "status": "ok",
            "package_version": __version__,
            "api_version": API_VERSION,
            "build_revision": settings.build_revision,
            "capabilities": capabilities,
            "active_search": active is not None,
        }

    @app.get("/readyz")
    async def readiness() -> dict[str, Any]:
        readiness_result = await asyncio.to_thread(manager.ensure_ready)
        return {
            "status": "ready",
            "package_version": __version__,
            "api_version": API_VERSION,
            "build_revision": settings.build_revision,
            **readiness_result,
        }

    @app.get("/", include_in_schema=False)
    @app.get("/monitor", response_class=HTMLResponse, include_in_schema=False)
    async def monitor() -> HTMLResponse:
        page = Path(__file__).with_name("static").joinpath("monitor.html").read_text()
        return HTMLResponse(page)

    @app.post("/v1/targets", response_model=TargetView, status_code=201)
    async def create_target(
        name: Annotated[str, Form(min_length=1, max_length=80)],
        image: Annotated[UploadFile, File()],
    ) -> TargetView:
        if image.content_type not in ALLOWED_IMAGE_TYPES:
            raise PersonSearchError(
                "supported image types are JPEG, PNG, and WebP",
                code="unsupported_media_type",
                status_code=415,
            )
        payload = await image.read(MAX_UPLOAD_BYTES + 1)
        if len(payload) > MAX_UPLOAD_BYTES:
            raise PersonSearchError(
                "image exceeds the 10 MiB limit", code="image_too_large", status_code=413
            )
        array = np.frombuffer(payload, dtype=np.uint8)
        frame = cv2.imdecode(array, cv2.IMREAD_COLOR)
        if frame is None:
            raise PersonSearchError(
                "image cannot be decoded", code="image_decode_failed", status_code=422
            )
        if frame.shape[0] * frame.shape[1] > MAX_IMAGE_PIXELS:
            raise PersonSearchError(
                "decoded image exceeds the pixel limit", code="image_too_large", status_code=413
            )
        return await asyncio.to_thread(manager.enroll, frame, name)

    @app.delete("/v1/targets/{target_id}", status_code=204)
    async def delete_target(target_id: str) -> Response:
        await asyncio.to_thread(manager.delete_target, target_id)
        return Response(status_code=204)

    @app.post("/v1/searches", response_model=SearchView, status_code=201)
    async def create_search(request: SearchCreate) -> SearchView:
        if request.source.type == SourceType.FILE:
            raise PersonSearchError(
                "file sources are supported by person-search-eval, not the live API",
                code="invalid_source",
                status_code=422,
            )
        if request.source.type == SourceType.RTSP:
            scheme = urlsplit(request.source.uri or "").scheme.lower()
            if scheme not in {"rtsp", "rtsps"}:
                raise PersonSearchError(
                    "RTSP source URI must use rtsp:// or rtsps://",
                    code="invalid_source",
                    status_code=422,
                )
        return await asyncio.to_thread(
            manager.start_batch_search,
            [request.target_id],
            request.source,
            request.timeout_seconds,
            request.replace_active,
            request.request_id,
        )

    @app.post("/v1/batch-searches", response_model=SearchView, status_code=201)
    async def create_batch_search(
        targets: Annotated[str, Form()],
        source: Annotated[str, Form()],
        images: Annotated[list[UploadFile], File()],
        timeout_seconds: Annotated[float | None, Form()] = None,
    ) -> SearchView:
        try:
            target_specs = json.loads(targets)
            source_data = json.loads(source)
        except (json.JSONDecodeError, TypeError) as exc:
            raise PersonSearchError(
                "targets and source must be valid JSON",
                code="invalid_batch_request",
                status_code=422,
            ) from exc
        if not isinstance(target_specs, list) or not target_specs:
            raise PersonSearchError(
                "targets must be a non-empty JSON array", code="invalid_targets", status_code=422
            )
        if len(target_specs) > MAX_BATCH_TARGETS:
            raise PersonSearchError(
                f"at most {MAX_BATCH_TARGETS} targets are allowed",
                code="too_many_targets",
                status_code=422,
            )
        if len(images) != len(target_specs):
            raise PersonSearchError(
                "each target must have exactly one image",
                code="image_target_mismatch",
                status_code=422,
            )
        try:
            request_source = SourceConfig.model_validate(source_data)
        except (
            Exception
        ) as exc:  # pydantic's detailed errors are not part of the public API contract
            raise PersonSearchError(
                "invalid source", code="invalid_source", status_code=422
            ) from exc
        if request_source.type == SourceType.FILE:
            raise PersonSearchError(
                "file sources are supported by person-search-eval, not the live API",
                code="invalid_source",
                status_code=422,
            )
        if request_source.type == SourceType.RTSP:
            scheme = urlsplit(request_source.uri or "").scheme.lower()
            if scheme not in {"rtsp", "rtsps"}:
                raise PersonSearchError(
                    "RTSP source URI must use rtsp:// or rtsps://",
                    code="invalid_source",
                    status_code=422,
                )
        if timeout_seconds is not None and timeout_seconds <= 0:
            raise PersonSearchError(
                "timeout_seconds must be positive", code="invalid_timeout", status_code=422
            )

        specs_by_filename: dict[str, dict[str, Any]] = {}
        for spec in target_specs:
            if not isinstance(spec, dict) or not isinstance(spec.get("name"), str):
                raise PersonSearchError(
                    "each target must contain a name and image_filename",
                    code="invalid_targets",
                    status_code=422,
                )
            filename = _batch_filename(spec.get("image_filename"))
            if filename is None or filename in specs_by_filename:
                raise PersonSearchError(
                    "image_filename values must be plain, non-empty, and unique",
                    code="invalid_targets",
                    status_code=422,
                )
            specs_by_filename[filename] = spec

        # Validate the complete filename set before decoding or enrolling any
        # image.  A count-only check lets a duplicated upload stand in for a
        # missing target (and can leave a partially enrolled batch behind).
        uploaded_filenames: list[str] = []
        for image in images:
            filename = _batch_filename(image.filename)
            if filename is None:
                raise PersonSearchError(
                    "uploaded image filenames must be plain and non-empty",
                    code="image_target_mismatch",
                    status_code=422,
                )
            uploaded_filenames.append(filename)
        if len(set(uploaded_filenames)) != len(uploaded_filenames) or set(uploaded_filenames) != set(
            specs_by_filename
        ):
            raise PersonSearchError(
                "uploaded image filenames must exactly match targets",
                code="image_target_mismatch",
                status_code=422,
            )

        enrolled_ids: list[str] = []
        try:
            for image in images:
                # The set was validated above, so this lookup cannot silently
                # reuse one target for two uploads.
                filename = _batch_filename(image.filename)
                assert filename is not None
                spec = specs_by_filename.get(filename)
                if spec is None:
                    raise PersonSearchError(
                        "uploaded image filenames must exactly match targets",
                        code="image_target_mismatch",
                        status_code=422,
                    )
                frame = await _decode_upload(image)
                target = await asyncio.to_thread(manager.enroll, frame, spec["name"])
                enrolled_ids.append(target.target_id)
            return await asyncio.to_thread(
                manager.start_batch_search,
                enrolled_ids,
                request_source,
                timeout_seconds,
                False,
                None,
            )
        except Exception:
            for target_id in enrolled_ids:
                try:
                    manager.delete_target(target_id)
                except PersonSearchError:
                    pass
            raise

    @app.get("/v1/searches/active", response_model=SearchView | None)
    async def active_search() -> SearchView | None:
        return await asyncio.to_thread(manager.active_search)

    @app.get("/v1/searches/by-request/{request_id}", response_model=SearchView | None)
    async def search_by_request(request_id: str) -> SearchView | None:
        return await asyncio.to_thread(manager.search_by_request_id, request_id)

    @app.get("/v1/searches/{search_id}", response_model=SearchView)
    async def get_search(search_id: str) -> SearchView:
        return await asyncio.to_thread(manager.get_search, search_id)

    @app.get("/v1/searches/{search_id}/evidence/{evidence_id}")
    async def get_evidence(
        search_id: str,
        evidence_id: str,
        variant: str = "face_crop",
        x_api_key: Annotated[str | None, Header()] = None,
    ) -> Response:
        """Fetch a short-lived, in-memory confirmation image.

        The opaque id is delivered only in a confirmed event.  No image is
        embedded in events or written to a file, log, or persistence layer.
        """
        if not settings.evidence_api_key:
            raise PersonSearchError(
                "evidence retrieval is not configured",
                code="evidence_access_not_configured",
                status_code=503,
            )
        if not _api_key_matches(x_api_key, settings.evidence_api_key):
            raise PersonSearchError(
                "invalid evidence API key", code="invalid_evidence_api_key", status_code=403
            )
        payload, media_type = await asyncio.to_thread(
            manager.get_session(search_id).get_evidence, evidence_id, variant
        )
        return Response(
            content=payload,
            media_type=media_type,
            headers={"Cache-Control": "no-store", "Pragma": "no-cache"},
        )

    @app.delete("/v1/searches/{search_id}/evidence/{evidence_id}", status_code=204)
    async def release_evidence(
        search_id: str,
        evidence_id: str,
        x_api_key: Annotated[str | None, Header()] = None,
    ) -> Response:
        """Acknowledge durable downstream storage and release in-memory JPEGs."""
        if not settings.evidence_api_key:
            raise PersonSearchError(
                "evidence retrieval is not configured",
                code="evidence_access_not_configured",
                status_code=503,
            )
        if not _api_key_matches(x_api_key, settings.evidence_api_key):
            raise PersonSearchError(
                "invalid evidence API key", code="invalid_evidence_api_key", status_code=403
            )
        await asyncio.to_thread(
            manager.get_session(search_id).release_evidence,
            evidence_id,
        )
        return Response(status_code=204)

    @app.get(
        "/v1/searches/{search_id}/preview.mjpg",
        response_class=StreamingResponse,
        responses={200: {"content": {"multipart/x-mixed-replace": {}}}},
    )
    async def preview_search(search_id: str) -> StreamingResponse:
        session = manager.get_session(search_id)

        async def frames():
            seq = 0
            # The worker only pays the annotate + encode cost while a viewer is
            # registered, so the stream must bracket its own lifetime.
            session.preview.subscribe()
            try:
                while True:
                    seq, jpeg = await asyncio.to_thread(session.preview.after, seq, 1.0)
                    if jpeg is not None:
                        yield b"--frame\r\nContent-Type: image/jpeg\r\n\r\n" + jpeg + b"\r\n"
                    if session.status.value in {"completed", "timed_out", "stopped", "failed"} and jpeg is None:
                        return
            finally:
                session.preview.unsubscribe()

        return StreamingResponse(
            frames(),
            media_type="multipart/x-mixed-replace; boundary=frame",
            headers={"Cache-Control": "no-store, no-cache, must-revalidate"},
        )

    @app.delete("/v1/searches/active", status_code=204)
    async def delete_active_search() -> Response:
        active = await asyncio.to_thread(manager.active_search)
        if active is not None:
            await asyncio.to_thread(manager.stop_search, active.search_id)
        return Response(status_code=204)

    @app.delete("/v1/searches/{search_id}", status_code=204)
    async def delete_search(search_id: str) -> Response:
        await asyncio.to_thread(manager.stop_search, search_id)
        return Response(status_code=204)

    @app.websocket("/v1/searches/{search_id}/events")
    async def search_events(websocket: WebSocket, search_id: str, after_seq: int = 0) -> None:
        try:
            session = manager.get_session(search_id)
        except PersonSearchError:
            await websocket.close(code=1008, reason="search not found")
            return
        await websocket.accept()
        seq = max(0, after_seq)
        try:
            while True:
                replay = getattr(session.events, "after_with_meta", None)
                if replay is not None:
                    events, gap, oldest_seq = await asyncio.to_thread(replay, seq, 1.0)
                    if gap:
                        gap_seq = max(0, int(oldest_seq or 1) - 1)
                        await websocket.send_json(
                            {
                                "schema_version": "1",
                                "seq": gap_seq,
                                "event_id": str(uuid.uuid4()),
                                "type": "replay_gap",
                                "occurred_at": int(time.time() * 1000),
                                "data": {
                                    "oldest_seq": int(oldest_seq or 0),
                                    "requested_after_seq": seq,
                                },
                            }
                        )
                        seq = gap_seq
                        continue
                else:
                    events = await asyncio.to_thread(session.events.after, seq, 1.0)
                for event in events:
                    event["search_id"] = search_id
                    await websocket.send_json(event)
                    seq = int(event["seq"])
                if session.status.value in {"completed", "timed_out", "stopped", "failed"} and not events:
                    await websocket.close(code=1000)
                    return
        except WebSocketDisconnect:
            return

    return app


def _problem(status: int, code: str, message: str, fields: Any = None) -> JSONResponse:
    detail: dict[str, Any] = {"code": code, "message": message}
    if fields is not None:
        detail["fields"] = fields
    return JSONResponse(status_code=status, content={"detail": detail})


def _api_key_matches(provided: str | None, expected: str | None) -> bool:
    """Compare header credentials without leaking timing or raising on Unicode.

    ``hmac.compare_digest`` only accepts ASCII ``str`` values.  Configuration is
    intentionally allowed to contain Unicode (headers are decoded as text by
    Starlette), so compare canonical UTF-8 bytes instead and fail closed for
    absent credentials.
    """
    if not isinstance(provided, str) or not isinstance(expected, str) or not provided or not expected:
        return False
    try:
        return hmac.compare_digest(provided.encode("utf-8"), expected.encode("utf-8"))
    except (UnicodeError, TypeError):
        return False


# Validation messages are generated from third-party/Pydantic errors.  Never
# reflect a submitted URI (which may contain RTSP credentials) back to a client.
# Match from a URI scheme through the end of the message.  A malformed URI can
# contain raw spaces or control characters in credentials; stopping at one of
# those characters would redact only the prefix and leave the remainder of a
# secret visible.  Validation messages are short diagnostics, so consuming the
# suffix is preferable to attempting to parse untrusted URI syntax in an error
# handler.
_VALIDATION_URI_RE = re.compile(r"(?is)(?:rtsps?|https?|file)://.*")
_FILENAME_CONTROL_RE = re.compile(r"[\x00-\x1f\x7f]")
_VALIDATION_LOC_MAX_LENGTH = 128


def _redact_validation_text(value: Any) -> str:
    """Return a JSON-safe validation message with URI credentials removed."""
    text = str(value)
    # Pydantic's normal messages do not echo input values, but custom validators
    # and future dependencies may.  Never let an RTSP credential reach a client.
    text = _VALIDATION_URI_RE.sub("<redacted-uri>", text)
    return _FILENAME_CONTROL_RE.sub(" ", text)


def _safe_validation_fields(exc: RequestValidationError) -> list[dict[str, Any]]:
    """Project FastAPI/Pydantic errors onto the small public error contract.

    ``exc.errors()`` includes ``input`` and, for ``ValueError`` validators, a
    ``ctx.error`` object that the JSON encoder cannot serialize.  Both are
    intentionally omitted; locations, stable error types and human-readable
    messages are sufficient for clients to correct a request and do not expose
    submitted images, credentials or arbitrary Python objects.
    """
    safe: list[dict[str, Any]] = []
    for error in exc.errors():
        raw_loc = error.get("loc", ())
        if isinstance(raw_loc, (tuple, list)):
            loc: list[Any] = []
            for part in raw_loc:
                if isinstance(part, int):
                    loc.append(part)
                    continue
                # Field names normally come from our schema, but an ``extra``
                # key is attacker-controlled.  Apply the same URI/control
                # redaction and length cap as messages before reflecting it.
                rendered = _redact_validation_text(part if isinstance(part, str) else str(part))
                loc.append(rendered[:_VALIDATION_LOC_MAX_LENGTH])
        else:
            if isinstance(raw_loc, int):
                loc = [raw_loc]
            else:
                rendered = _redact_validation_text(
                    raw_loc if isinstance(raw_loc, str) else str(raw_loc)
                )
                loc = [rendered[:_VALIDATION_LOC_MAX_LENGTH]]
        safe.append(
            {
                "loc": loc,
                "type": str(error.get("type", "validation_error")),
                "msg": _redact_validation_text(error.get("msg", "request validation failed")),
            }
        )
    return safe


def _contains_source_validation_error(fields: list[dict[str, Any]]) -> bool:
    """Identify SourceConfig errors without looking at unsafe raw inputs."""
    source_tokens = {"uri", "device_index", "hostname", "port"}
    for field in fields:
        raw_loc = [str(part) for part in field.get("loc", [])]
        loc = set(raw_loc)
        message = str(field.get("msg", "")).lower()
        # A missing top-level ``source`` is an ordinary request-shape error, not
        # a malformed source URI. Keep the longstanding generic validation code
        # for that one case.
        if loc.intersection(source_tokens):
            return True
        if "source" in loc and len(raw_loc) == 2 and field.get("type") == "missing":
            continue
        if "source" in loc and (len(raw_loc) > 2 or field.get("type") != "missing"):
            # ``body.source`` + ``missing`` is the one shape error that remains
            # generic. Nested fields, enum/type errors, and model-level URI
            # validators all describe an actual source value and use the stable
            # invalid_source code.
            return True
        if re.search(r"\b(?:rtsp|rtsps|uri|device_index|hostname|port)\b", message):
            return True
    return False


def _batch_filename(value: Any) -> str | None:
    """Return a safe multipart filename token, or ``None`` when ambiguous.

    Matching by basename alone permits ``dir/a.jpg`` and ``a.jpg`` to alias one
    target and lets duplicate uploads pass a count-only check.  Batch requests use
    plain filename tokens; path components, control characters and surrounding
    whitespace are rejected before any expensive enrollment starts.
    """
    if not isinstance(value, str):
        return None
    if not value or value != value.strip() or len(value) > 255:
        return None
    if "/" in value or "\\" in value or _FILENAME_CONTROL_RE.search(value):
        return None
    return value


async def _decode_upload(image: UploadFile) -> np.ndarray:
    if image.content_type not in ALLOWED_IMAGE_TYPES:
        raise PersonSearchError(
            "supported image types are JPEG, PNG, and WebP",
            code="unsupported_media_type",
            status_code=415,
        )
    payload = await image.read(MAX_UPLOAD_BYTES + 1)
    if len(payload) > MAX_UPLOAD_BYTES:
        raise PersonSearchError(
            "image exceeds the 10 MiB limit", code="image_too_large", status_code=413
        )
    array = np.frombuffer(payload, dtype=np.uint8)
    frame = cv2.imdecode(array, cv2.IMREAD_COLOR)
    if frame is None:
        raise PersonSearchError(
            "image cannot be decoded", code="image_decode_failed", status_code=422
        )
    if frame.shape[0] * frame.shape[1] > MAX_IMAGE_PIXELS:
        raise PersonSearchError(
            "decoded image exceeds the pixel limit", code="image_too_large", status_code=413
        )
    return frame
