import sqlite3
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Annotated

from fastapi import FastAPI, HTTPException, Query, Request, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse

from futbol_video_analyst.config import settings
from futbol_video_analyst.database import Database
from futbol_video_analyst.domain import Event, EventCreate, EventType, Match, MatchImport
from futbol_video_analyst.video import FFprobeVideoInspector, VideoInspectionError


def create_app(
    database_path: Path | None = None,
    video_inspector: FFprobeVideoInspector | None = None,
) -> FastAPI:
    database = Database(database_path or settings.database_path)

    @asynccontextmanager
    async def lifespan(application: FastAPI) -> AsyncIterator[None]:
        database.initialize()
        application.state.database = database
        application.state.video_inspector = video_inspector or FFprobeVideoInspector()
        yield

    application = FastAPI(
        title="Futbol Video Analyst API",
        description="Motor local para analizar partidos y administrar etiquetas.",
        version="0.2.0",
        lifespan=lifespan,
    )
    application.add_middleware(
        CORSMiddleware,
        allow_origins=[
            "http://127.0.0.1:1420",
            "http://localhost:1420",
            "http://127.0.0.1:5173",
            "http://localhost:5173",
            "tauri://localhost",
            "https://tauri.localhost",
        ],
        allow_methods=["GET", "POST", "DELETE", "PATCH"],
        allow_headers=["Content-Type"],
    )

    @application.get("/health", tags=["system"])
    def health() -> dict[str, str]:
        return {"status": "ok"}

    @application.post(
        "/matches/import",
        response_model=Match,
        status_code=status.HTTP_201_CREATED,
        tags=["matches"],
    )
    def import_match(payload: MatchImport, request: Request) -> Match:
        path = Path(payload.video_path).expanduser().resolve()
        try:
            metadata = request.app.state.video_inspector.inspect(path)
            return request.app.state.database.create_match(payload.title, str(path), metadata)
        except VideoInspectionError as error:
            raise HTTPException(status_code=422, detail=str(error)) from error
        except sqlite3.IntegrityError as error:
            raise HTTPException(status_code=409, detail="This video is already imported") from error

    @application.get("/matches", response_model=list[Match], tags=["matches"])
    def list_matches(request: Request) -> list[Match]:
        return request.app.state.database.list_matches()

    @application.get("/matches/{match_id}", response_model=Match, tags=["matches"])
    def get_match(match_id: str, request: Request) -> Match:
        match = request.app.state.database.get_match(match_id)
        if match is None:
            raise HTTPException(status_code=404, detail="Match not found")
        return match

    @application.get("/matches/{match_id}/video", response_class=FileResponse, tags=["matches"])
    def get_match_video(match_id: str, request: Request) -> FileResponse:
        match = request.app.state.database.get_match(match_id)
        if match is None:
            raise HTTPException(status_code=404, detail="Match not found")
        path = Path(match.video_path)
        if not path.is_file():
            raise HTTPException(status_code=410, detail="Video file is no longer available")
        return FileResponse(path, filename=path.name, content_disposition_type="inline")

    @application.post(
        "/matches/{match_id}/events",
        response_model=Event,
        status_code=status.HTTP_201_CREATED,
        tags=["events"],
    )
    def create_event(match_id: str, payload: EventCreate, request: Request) -> Event:
        match = request.app.state.database.get_match(match_id)
        if match is None:
            raise HTTPException(status_code=404, detail="Match not found")
        if payload.end_seconds > match.duration_seconds:
            raise HTTPException(status_code=422, detail="Event exceeds video duration")
        return request.app.state.database.create_event(match_id, payload)

    @application.get("/matches/{match_id}/events", response_model=list[Event], tags=["events"])
    def list_events(
        match_id: str,
        request: Request,
        event_type: Annotated[EventType | None, Query(alias="type")] = None,
    ) -> list[Event]:
        if request.app.state.database.get_match(match_id) is None:
            raise HTTPException(status_code=404, detail="Match not found")
        return request.app.state.database.list_events(match_id, event_type)

    return application


app = create_app()
