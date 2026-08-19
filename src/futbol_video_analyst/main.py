import sqlite3
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Annotated

from fastapi import FastAPI, HTTPException, Query, Request, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse

from futbol_video_analyst.analysis import AnalysisCoordinator, VisualSignalAnalyzer
from futbol_video_analyst.clips import ClipExportError, FFmpegClipExporter
from futbol_video_analyst.config import settings
from futbol_video_analyst.database import Database
from futbol_video_analyst.domain import (
    AnalysisJob,
    Event,
    EventCreate,
    EventReview,
    EventType,
    EventUpdate,
    Match,
    MatchImport,
    VisualSignal,
)
from futbol_video_analyst.video import FFprobeVideoInspector, VideoInspectionError


def create_app(
    database_path: Path | None = None,
    video_inspector: FFprobeVideoInspector | None = None,
    clips_dir: Path | None = None,
    clip_exporter: FFmpegClipExporter | None = None,
    visual_analyzer: VisualSignalAnalyzer | None = None,
) -> FastAPI:
    database = Database(database_path or settings.database_path)
    local_clips_dir = clips_dir or settings.clips_dir
    analysis_coordinator = AnalysisCoordinator(database, visual_analyzer)

    @asynccontextmanager
    async def lifespan(application: FastAPI) -> AsyncIterator[None]:
        database.initialize()
        application.state.database = database
        application.state.video_inspector = video_inspector or FFprobeVideoInspector()
        application.state.clip_exporter = clip_exporter or FFmpegClipExporter()
        application.state.analysis_coordinator = analysis_coordinator
        yield
        analysis_coordinator.shutdown()

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
        expose_headers=["Content-Disposition", "X-Exported-Path"],
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

    @application.patch("/events/{event_id}/review", response_model=Event, tags=["events"])
    def review_event(event_id: str, payload: EventReview, request: Request) -> Event:
        event = request.app.state.database.review_event(event_id, payload.review_status)
        if event is None:
            raise HTTPException(status_code=404, detail="Event not found")
        return event

    @application.patch("/events/{event_id}", response_model=Event, tags=["events"])
    def update_event(event_id: str, payload: EventUpdate, request: Request) -> Event:
        existing = request.app.state.database.get_event(event_id)
        if existing is None:
            raise HTTPException(status_code=404, detail="Event not found")
        match = request.app.state.database.get_match(existing.match_id)
        if match is None:
            raise HTTPException(status_code=404, detail="Match not found")
        if payload.end_seconds > match.duration_seconds:
            raise HTTPException(status_code=422, detail="Event exceeds video duration")
        event = request.app.state.database.update_event(event_id, payload)
        assert event is not None
        return event

    @application.patch("/events/{event_id}/reclassify", response_model=Event, tags=["events"])
    def reclassify_event(event_id: str, payload: EventUpdate, request: Request) -> Event:
        existing = request.app.state.database.get_event(event_id)
        if existing is None:
            raise HTTPException(status_code=404, detail="Event not found")
        match = request.app.state.database.get_match(existing.match_id)
        if match is None:
            raise HTTPException(status_code=404, detail="Match not found")
        if existing.review_status != "rejected":
            raise HTTPException(status_code=409, detail="Only rejected events can be reclassified")
        if payload.type == (existing.detected_type or existing.type):
            raise HTTPException(status_code=422, detail="Choose a different event type")
        if payload.end_seconds > match.duration_seconds:
            raise HTTPException(status_code=422, detail="Event exceeds video duration")
        event = request.app.state.database.reclassify_event(event_id, payload)
        assert event is not None
        return event

    @application.delete("/events/{event_id}", status_code=status.HTTP_204_NO_CONTENT)
    def delete_event(event_id: str, request: Request) -> None:
        if not request.app.state.database.delete_event(event_id):
            raise HTTPException(status_code=404, detail="Event not found")

    @application.post("/events/{event_id}/clip", response_class=FileResponse, tags=["clips"])
    def export_event_clip(event_id: str, request: Request) -> FileResponse:
        event = request.app.state.database.get_event(event_id)
        if event is None:
            raise HTTPException(status_code=404, detail="Event not found")
        match = request.app.state.database.get_match(event.match_id)
        if match is None:
            raise HTTPException(status_code=404, detail="Match not found")

        filename = f"{event.type}-{int(event.peak_seconds):06d}-{event.id[:8]}.mp4"
        destination = local_clips_dir / filename
        try:
            request.app.state.clip_exporter.export(
                Path(match.video_path), destination, event.start_seconds, event.end_seconds
            )
        except ClipExportError as error:
            raise HTTPException(status_code=500, detail=str(error)) from error
        return FileResponse(
            destination,
            filename=filename,
            media_type="video/mp4",
            headers={"X-Exported-Path": str(destination.resolve())},
        )

    @application.post(
        "/matches/{match_id}/analysis",
        response_model=AnalysisJob,
        status_code=status.HTTP_202_ACCEPTED,
        tags=["analysis"],
    )
    def start_analysis(match_id: str, request: Request) -> AnalysisJob:
        match = request.app.state.database.get_match(match_id)
        if match is None:
            raise HTTPException(status_code=404, detail="Match not found")
        return request.app.state.analysis_coordinator.start(match)

    @application.get("/analysis/{job_id}", response_model=AnalysisJob, tags=["analysis"])
    def get_analysis(job_id: str, request: Request) -> AnalysisJob:
        job = request.app.state.database.get_analysis_job(job_id)
        if job is None:
            raise HTTPException(status_code=404, detail="Analysis job not found")
        return job

    @application.get(
        "/matches/{match_id}/analysis/latest", response_model=AnalysisJob, tags=["analysis"]
    )
    def get_latest_analysis(match_id: str, request: Request) -> AnalysisJob:
        if request.app.state.database.get_match(match_id) is None:
            raise HTTPException(status_code=404, detail="Match not found")
        job = request.app.state.database.get_latest_analysis_job(match_id)
        if job is None:
            raise HTTPException(status_code=404, detail="Match has not been analyzed")
        return job

    @application.get(
        "/matches/{match_id}/signals", response_model=list[VisualSignal], tags=["analysis"]
    )
    def list_visual_signals(match_id: str, request: Request) -> list[VisualSignal]:
        if request.app.state.database.get_match(match_id) is None:
            raise HTTPException(status_code=404, detail="Match not found")
        return request.app.state.database.list_visual_signals(match_id)

    return application


app = create_app()
