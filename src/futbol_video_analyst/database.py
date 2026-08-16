import sqlite3
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path
from uuid import uuid4

from futbol_video_analyst.domain import (
    Event,
    EventCreate,
    Match,
    MatchStatus,
    ReviewStatus,
    VideoMetadata,
)

SCHEMA = """
CREATE TABLE IF NOT EXISTS matches (
    id TEXT PRIMARY KEY,
    title TEXT NOT NULL,
    video_path TEXT NOT NULL UNIQUE,
    duration_seconds REAL NOT NULL,
    width INTEGER NOT NULL,
    height INTEGER NOT NULL,
    fps REAL NOT NULL,
    codec TEXT NOT NULL,
    status TEXT NOT NULL,
    created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE IF NOT EXISTS events (
    id TEXT PRIMARY KEY,
    match_id TEXT NOT NULL REFERENCES matches(id) ON DELETE CASCADE,
    type TEXT NOT NULL,
    start_seconds REAL NOT NULL,
    peak_seconds REAL NOT NULL,
    end_seconds REAL NOT NULL,
    confidence REAL NOT NULL,
    source TEXT NOT NULL,
    review_status TEXT NOT NULL,
    notes TEXT,
    created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
);

CREATE INDEX IF NOT EXISTS events_match_time_idx ON events(match_id, peak_seconds);
"""


class Database:
    def __init__(self, path: Path) -> None:
        self.path = path

    def initialize(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self.connect() as connection:
            connection.executescript(SCHEMA)

    @contextmanager
    def connect(self) -> Iterator[sqlite3.Connection]:
        connection = sqlite3.connect(self.path)
        connection.row_factory = sqlite3.Row
        connection.execute("PRAGMA foreign_keys = ON")
        try:
            yield connection
            connection.commit()
        finally:
            connection.close()

    def create_match(self, title: str, video_path: str, metadata: VideoMetadata) -> Match:
        match_id = str(uuid4())
        with self.connect() as connection:
            connection.execute(
                """
                INSERT INTO matches (
                    id, title, video_path, duration_seconds, width, height, fps, codec, status
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    match_id,
                    title,
                    video_path,
                    metadata.duration_seconds,
                    metadata.width,
                    metadata.height,
                    metadata.fps,
                    metadata.codec,
                    MatchStatus.READY,
                ),
            )
        match = self.get_match(match_id)
        assert match is not None
        return match

    def list_matches(self) -> list[Match]:
        with self.connect() as connection:
            rows = connection.execute("SELECT * FROM matches ORDER BY created_at DESC").fetchall()
        return [Match.model_validate(dict(row)) for row in rows]

    def get_match(self, match_id: str) -> Match | None:
        with self.connect() as connection:
            row = connection.execute("SELECT * FROM matches WHERE id = ?", (match_id,)).fetchone()
        return Match.model_validate(dict(row)) if row else None

    def create_event(self, match_id: str, payload: EventCreate) -> Event:
        event_id = str(uuid4())
        with self.connect() as connection:
            connection.execute(
                """
                INSERT INTO events (
                    id, match_id, type, start_seconds, peak_seconds, end_seconds,
                    confidence, source, review_status, notes
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    event_id,
                    match_id,
                    payload.type,
                    payload.start_seconds,
                    payload.peak_seconds,
                    payload.end_seconds,
                    payload.confidence,
                    payload.source,
                    ReviewStatus.UNREVIEWED,
                    payload.notes,
                ),
            )
        event = self.get_event(event_id)
        assert event is not None
        return event

    def get_event(self, event_id: str) -> Event | None:
        with self.connect() as connection:
            row = connection.execute("SELECT * FROM events WHERE id = ?", (event_id,)).fetchone()
        return Event.model_validate(dict(row)) if row else None

    def list_events(self, match_id: str, event_type: str | None = None) -> list[Event]:
        query = "SELECT * FROM events WHERE match_id = ?"
        parameters: list[str] = [match_id]
        if event_type:
            query += " AND type = ?"
            parameters.append(event_type)
        query += " ORDER BY peak_seconds"
        with self.connect() as connection:
            rows = connection.execute(query, parameters).fetchall()
        return [Event.model_validate(dict(row)) for row in rows]
