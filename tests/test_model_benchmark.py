from pathlib import Path

from futbol_video_analyst.database import Database
from futbol_video_analyst.domain import (
    EventCreate,
    EventSource,
    EventType,
    ReviewStatus,
    VideoMetadata,
)
from futbol_video_analyst.model_benchmark import reviewed_corner_truth, select_matches


def _event(event_type: EventType, source: EventSource, peak: float) -> EventCreate:
    return EventCreate(
        type=event_type,
        start_seconds=max(0, peak - 1),
        peak_seconds=peak,
        end_seconds=peak + 1,
        source=source,
    )


def test_truth_uses_manual_and_confirmed_corners_only(tmp_path: Path) -> None:
    database = Database(tmp_path / "benchmark.sqlite3")
    database.initialize()
    match = database.create_match(
        "Benchmark",
        str(tmp_path / "match.mp4"),
        VideoMetadata(duration_seconds=100, width=640, height=360, fps=25, codec="h264"),
    )
    database.create_event(match.id, _event(EventType.CORNER, EventSource.MANUAL, 10))
    confirmed = database.create_event(
        match.id, _event(EventType.CORNER, EventSource.DETECTOR, 20)
    )
    database.review_event(confirmed.id, ReviewStatus.CONFIRMED)
    pending = database.create_event(
        match.id, _event(EventType.CORNER, EventSource.DETECTOR, 30)
    )
    rejected = database.create_event(
        match.id, _event(EventType.CORNER, EventSource.DETECTOR, 40)
    )
    database.review_event(rejected.id, ReviewStatus.REJECTED)
    database.create_event(match.id, _event(EventType.GOAL, EventSource.MANUAL, 50))

    truth = reviewed_corner_truth(database, match)

    assert [event.timestamp_seconds for event in truth] == [10, 20]
    assert 30 not in {event.timestamp_seconds for event in truth}
    assert pending.review_status is ReviewStatus.UNREVIEWED


def test_select_matches_ignores_case_and_accidental_whitespace(tmp_path: Path) -> None:
    database = Database(tmp_path / "benchmark.sqlite3")
    database.initialize()
    database.create_match(
        "Alemania vs brasil ",
        str(tmp_path / "match.mp4"),
        VideoMetadata(duration_seconds=100, width=640, height=360, fps=25, codec="h264"),
    )
    (tmp_path / "match.mp4").touch()

    selected = select_matches(database, ("ALEMANIA  vs BRASIL",))

    assert selected[0].title == "Alemania vs brasil "
