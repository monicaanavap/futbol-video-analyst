from futbol_video_analyst.analysis import merge_spotter_candidates
from futbol_video_analyst.domain import EventCreate, EventSource, EventType


def candidate(timestamp: float, confidence: float) -> EventCreate:
    return EventCreate(
        type=EventType.CORNER,
        start_seconds=max(0, timestamp - 8),
        peak_seconds=timestamp,
        end_seconds=timestamp + 12,
        confidence=confidence,
        source=EventSource.DETECTOR,
    )


def test_merges_nearby_candidates_and_keeps_stronger_prediction() -> None:
    merged = merge_spotter_candidates(
        [candidate(20, 0.6), candidate(60, 0.7)],
        [candidate(24, 0.9), candidate(90, 0.8)],
    )

    assert [item.peak_seconds for item in merged] == [24, 60, 90]
