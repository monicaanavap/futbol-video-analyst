import sqlite3
from pathlib import Path

from futbol_video_analyst.database import Database
from futbol_video_analyst.domain import EventCreate, EventSource, EventType, VideoMetadata


def test_initialize_migrates_existing_visual_signals_table(tmp_path: Path) -> None:
    path = tmp_path / "existing.sqlite3"
    with sqlite3.connect(path) as connection:
        connection.execute(
            """
            CREATE TABLE visual_signals (
                id TEXT PRIMARY KEY,
                match_id TEXT NOT NULL,
                timestamp_seconds REAL NOT NULL,
                green_ratio REAL NOT NULL,
                brightness REAL NOT NULL,
                change_score REAL NOT NULL,
                likely_field INTEGER NOT NULL
            )
            """
        )

    Database(path).initialize()

    with sqlite3.connect(path) as connection:
        columns = {row[1] for row in connection.execute("PRAGMA table_info(visual_signals)")}
    assert {"player_candidates", "ball_candidates", "line_ratio"} <= columns


def test_initialize_adds_soft_delete_column_to_existing_matches(tmp_path: Path) -> None:
    path = tmp_path / "existing.sqlite3"
    with sqlite3.connect(path) as connection:
        connection.execute(
            """
            CREATE TABLE matches (
                id TEXT PRIMARY KEY, title TEXT NOT NULL, video_path TEXT NOT NULL UNIQUE,
                duration_seconds REAL NOT NULL, width INTEGER NOT NULL, height INTEGER NOT NULL,
                fps REAL NOT NULL, codec TEXT NOT NULL, status TEXT NOT NULL,
                created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
            )
            """
        )

    Database(path).initialize()

    with sqlite3.connect(path) as connection:
        columns = {row[1] for row in connection.execute("PRAGMA table_info(matches)")}
    assert "deleted_at" in columns


def test_initialize_migrates_legacy_shots_to_shot_attempts(tmp_path: Path) -> None:
    path = tmp_path / "legacy-shots.sqlite3"
    database = Database(path)
    database.initialize()
    match = database.create_match(
        "Partido",
        str(tmp_path / "match.mp4"),
        VideoMetadata(duration_seconds=90, width=1280, height=720, fps=30, codec="h264"),
    )
    with database.connect() as connection:
        connection.execute(
            """
            INSERT INTO events (
                id, match_id, type, start_seconds, peak_seconds, end_seconds,
                confidence, source, review_status, detected_type
            ) VALUES ('legacy-shot', ?, 'shot', 10, 15, 20, 1, 'manual', 'unreviewed', 'shot')
            """,
            (match.id,),
        )

    database.initialize()

    migrated = database.get_event("legacy-shot")
    assert migrated is not None
    assert migrated.type is EventType.SHOT_ATTEMPT
    assert migrated.detected_type is EventType.SHOT_ATTEMPT


def test_initialize_adds_event_context_and_migrates_simple_outcome_notes(tmp_path: Path) -> None:
    path = tmp_path / "legacy-event-context.sqlite3"
    database = Database(path)
    database.initialize()
    match = database.create_match(
        "Tanda",
        str(tmp_path / "match.mp4"),
        VideoMetadata(duration_seconds=90, width=1280, height=720, fps=30, codec="h264"),
    )
    with database.connect() as connection:
        connection.execute(
            """
            INSERT INTO events (
                id, match_id, type, start_seconds, peak_seconds, end_seconds,
                confidence, source, review_status, notes
            ) VALUES ('legacy-penalty', ?, 'penalty', 10, 15, 20, 1, 'manual',
                'unreviewed', 'gol')
            """,
            (match.id,),
        )
        connection.execute("UPDATE events SET outcome = NULL, phase = NULL")

    database.initialize()

    migrated = database.get_event("legacy-penalty")
    assert migrated is not None
    assert migrated.outcome == "goal"
    assert migrated.phase is None


def test_soft_deletes_and_restores_match_with_its_data(tmp_path: Path) -> None:
    database = Database(tmp_path / "matches.sqlite3")
    database.initialize()
    match = database.create_match(
        "Recuperable",
        str(tmp_path / "match.mp4"),
        VideoMetadata(duration_seconds=90, width=1280, height=720, fps=30, codec="h264"),
    )
    event = database.create_event(
        match.id,
        EventCreate(type=EventType.CORNER, start_seconds=10, peak_seconds=15, end_seconds=22),
    )

    assert database.delete_match(match.id)
    assert database.list_matches() == []
    assert [item.id for item in database.list_deleted_matches()] == [match.id]
    assert database.get_match(match.id) is None
    assert database.get_event(event.id) is not None

    assert database.restore_match(match.id)
    assert [item.id for item in database.list_matches()] == [match.id]
    assert database.list_deleted_matches() == []
    assert database.get_event(event.id) is not None


def test_new_candidates_do_not_duplicate_a_manual_corner(tmp_path: Path) -> None:
    database = Database(tmp_path / "events.sqlite3")
    database.initialize()
    match = database.create_match(
        "Test",
        str(tmp_path / "match.mp4"),
        VideoMetadata(duration_seconds=90, width=1280, height=720, fps=30, codec="h264"),
    )
    manual = database.create_event(
        match.id,
        EventCreate(type=EventType.CORNER, start_seconds=22, peak_seconds=30, end_seconds=42),
    )

    inserted = database.replace_corner_candidates(
        match.id,
        [
            EventCreate(
                type=EventType.CORNER,
                start_seconds=24,
                peak_seconds=32,
                end_seconds=44,
                source=EventSource.DETECTOR,
            ),
            EventCreate(
                type=EventType.CORNER,
                start_seconds=52,
                peak_seconds=60,
                end_seconds=72,
                source=EventSource.DETECTOR,
            ),
        ],
    )

    assert [event.peak_seconds for event in inserted] == [60]
    assert [event.id for event in database.list_events(match.id)] == [manual.id, inserted[0].id]
