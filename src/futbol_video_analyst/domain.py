from enum import StrEnum

from pydantic import BaseModel, Field, model_validator


class MatchStatus(StrEnum):
    READY = "ready"
    ANALYZING = "analyzing"
    COMPLETED = "completed"
    FAILED = "failed"


class EventType(StrEnum):
    CORNER = "corner"
    THROW_IN = "throw_in"
    PENALTY = "penalty"
    GOAL = "goal"
    SHOT = "shot"
    FOUL = "foul"
    CUSTOM = "custom"


class EventSource(StrEnum):
    MANUAL = "manual"
    DETECTOR = "detector"


class ReviewStatus(StrEnum):
    UNREVIEWED = "unreviewed"
    CONFIRMED = "confirmed"
    REJECTED = "rejected"


class AnalysisStatus(StrEnum):
    QUEUED = "queued"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"


class AnalysisStage(StrEnum):
    QUEUED = "queued"
    SAMPLING = "sampling"
    REFINING = "refining"
    COMPLETED = "completed"
    FAILED = "failed"


class VideoMetadata(BaseModel):
    duration_seconds: float = Field(gt=0)
    width: int = Field(gt=0)
    height: int = Field(gt=0)
    fps: float = Field(gt=0)
    codec: str


class MatchImport(BaseModel):
    title: str = Field(min_length=1, max_length=200)
    video_path: str = Field(min_length=1)


class Match(BaseModel):
    id: str
    title: str
    video_path: str
    duration_seconds: float
    width: int
    height: int
    fps: float
    codec: str
    status: MatchStatus
    created_at: str


class EventCreate(BaseModel):
    type: EventType
    start_seconds: float = Field(ge=0)
    peak_seconds: float = Field(ge=0)
    end_seconds: float = Field(ge=0)
    confidence: float = Field(default=1.0, ge=0, le=1)
    source: EventSource = EventSource.MANUAL
    notes: str | None = Field(default=None, max_length=1000)

    @model_validator(mode="after")
    def validate_timeline(self) -> "EventCreate":
        if not self.start_seconds <= self.peak_seconds <= self.end_seconds:
            raise ValueError("start_seconds <= peak_seconds <= end_seconds is required")
        return self


class EventUpdate(BaseModel):
    type: EventType
    start_seconds: float = Field(ge=0)
    peak_seconds: float = Field(ge=0)
    end_seconds: float = Field(ge=0)
    notes: str | None = Field(default=None, max_length=1000)

    @model_validator(mode="after")
    def validate_timeline(self) -> "EventUpdate":
        if not self.start_seconds <= self.peak_seconds <= self.end_seconds:
            raise ValueError("start_seconds <= peak_seconds <= end_seconds is required")
        return self


class Event(EventCreate):
    id: str
    match_id: str
    detected_type: EventType | None = None
    review_status: ReviewStatus
    created_at: str


class EventReview(BaseModel):
    review_status: ReviewStatus

    @model_validator(mode="after")
    def require_decision(self) -> "EventReview":
        if self.review_status is ReviewStatus.UNREVIEWED:
            raise ValueError("Review must confirm or reject the event")
        return self


class AnalysisJob(BaseModel):
    id: str
    match_id: str
    status: AnalysisStatus
    stage: AnalysisStage
    progress: float = Field(ge=0, le=1)
    samples_processed: int = Field(ge=0)
    error: str | None
    created_at: str
    updated_at: str


class VisualSignal(BaseModel):
    id: str
    match_id: str
    timestamp_seconds: float = Field(ge=0)
    green_ratio: float = Field(ge=0, le=1)
    brightness: float = Field(ge=0, le=1)
    change_score: float = Field(ge=0, le=1)
    likely_field: bool
    player_candidates: int = Field(default=0, ge=0)
    ball_candidates: int = Field(default=0, ge=0)
    line_ratio: float = Field(default=0, ge=0, le=1)
