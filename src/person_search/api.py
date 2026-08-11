from __future__ import annotations

import asyncio
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
    Request,
    Response,
    UploadFile,
    WebSocket,
    WebSocketDisconnect,
)
from fastapi.exceptions import RequestValidationError
from fastapi.responses import HTMLResponse, JSONResponse, StreamingResponse

from .config import Settings, get_settings
from .domain import SearchCreate, SearchView, SourceType, TargetView
from .errors import PersonSearchError
from .service import SearchManager

MAX_UPLOAD_BYTES = 10 * 1024 * 1024
MAX_IMAGE_PIXELS = 20_000_000
ALLOWED_IMAGE_TYPES = {"image/jpeg", "image/png", "image/webp"}


def create_app(
    settings: Settings | None = None,
    manager: SearchManager | None = None,
) -> FastAPI:
    settings = settings or get_settings()
    manager = manager or SearchManager(settings)

    @asynccontextmanager
    async def lifespan(_: FastAPI):
        yield
        await asyncio.to_thread(manager.shutdown)

    app = FastAPI(
        title="Robot Person Search PoC",
        version="0.1.0",
        description="Session-scoped photo-to-live-video person search API.",
        lifespan=lifespan,
    )
    app.state.manager = manager

    @app.exception_handler(PersonSearchError)
    async def handle_person_search_error(_: Request, exc: PersonSearchError) -> JSONResponse:
        return _problem(exc.status_code, exc.code, exc.message)

    @app.exception_handler(RequestValidationError)
    async def handle_validation_error(_: Request, exc: RequestValidationError) -> JSONResponse:
        return _problem(422, "validation_error", "request validation failed", exc.errors())

    @app.get("/healthz")
    async def health() -> dict[str, str]:
        return {"status": "ok"}

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
        return await asyncio.to_thread(manager.start_search, request.target_id, request.source)

    @app.get("/v1/searches/{search_id}", response_model=SearchView)
    async def get_search(search_id: str) -> SearchView:
        return manager.get_search(search_id)

    @app.get(
        "/v1/searches/{search_id}/preview.mjpg",
        response_class=StreamingResponse,
        responses={200: {"content": {"multipart/x-mixed-replace": {}}}},
    )
    async def preview_search(search_id: str) -> StreamingResponse:
        session = manager.get_session(search_id)

        async def frames():
            seq = 0
            while True:
                seq, jpeg = await asyncio.to_thread(session.preview.after, seq, 1.0)
                if jpeg is not None:
                    yield b"--frame\r\nContent-Type: image/jpeg\r\n\r\n" + jpeg + b"\r\n"
                if session.status.value in {"stopped", "failed"} and jpeg is None:
                    return

        return StreamingResponse(
            frames(),
            media_type="multipart/x-mixed-replace; boundary=frame",
            headers={"Cache-Control": "no-store, no-cache, must-revalidate"},
        )

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
                events = await asyncio.to_thread(session.events.after, seq, 1.0)
                for event in events:
                    event["search_id"] = search_id
                    await websocket.send_json(event)
                    seq = int(event["seq"])
                if session.status.value in {"stopped", "failed"} and not events:
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
