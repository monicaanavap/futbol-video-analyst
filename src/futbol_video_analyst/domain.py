from enum import StrEnum

from pydantic import BaseModel, Field, model_validator


class MatchStatus(StrEnum):
    READY = "ready"
    ANALYZING = "analyzing"
    COMPLETED = "completed"
    FAILED = "failed"


class EventType(StrEnum):
    CORNER = "corner"
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


class Event(EventCreate):
    id: str
    match_id: str
    review_status: ReviewStatus
    created_at: str
