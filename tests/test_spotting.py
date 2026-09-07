from pathlib import Path

from futbol_video_analyst.action_training import CONTEXT_SECONDS, expand_context
from futbol_video_analyst.domain import Match, MatchStatus
from futbol_video_analyst.spotting import NeuralCornerSpotter
from futbol_video_analyst.training import TrainingExample


def make_match() -> Match:
    return Match(
        id="match-id",
        title="Test",
        video_path="/tmp/test.mp4",
        duration_seconds=120,
        width=1920,
        height=1080,
        fps=30,
        codec="h264",
        status=MatchStatus.READY,
        created_at="2026-08-28 00:00:00",
    )


def test_merges_nearby_neural_windows_and_keeps_highest_confidence() -> None:
    spotter = NeuralCornerSpotter(Path("corner-spotter-test.pt"))

    events = spotter._merge_predictions(
        make_match(),
        [(10, 0.71), (14, 0.92), (18, 0.81), (45, 0.69), (70, 0.88)],
        threshold=0.7,
    )

    assert [event.peak_seconds for event in events] == [14, 70]
    assert [event.confidence for event in events] == [0.92, 0.88]
    assert all(event.source == "detector" for event in events)
    assert all("corner-spotter-test" in (event.notes or "") for event in events)


def test_returns_none_when_the_local_model_is_missing() -> None:
    spotter = NeuralCornerSpotter(Path("missing-model.pt"))

    assert spotter.spot(make_match(), lambda _progress, _windows: None) is None


def test_neural_spotter_limits_full_match_candidates() -> None:
    spotter = NeuralCornerSpotter(Path("corner-spotter-test.pt"))
    scored = [(index * 20, 0.7 + index / 1000) for index in range(40)]

    events = spotter._merge_predictions(
        make_match().model_copy(update={"duration_seconds": 1000}), scored, threshold=0.5
    )

    assert len(events) == 30
    assert [event.peak_seconds for event in events] == sorted(
        event.peak_seconds for event in events
    )


def test_expands_each_training_event_into_temporal_context() -> None:
    example = TrainingExample(
        clip_path=Path("clip.mp4"),
        event_id="event-id",
        label=1,
        match_id="match-id",
        match_title="Test",
        peak_in_clip=8,
    )

    expanded = expand_context([example])

    assert len(expanded) == len(CONTEXT_SECONDS)
    assert [item.peak_in_clip for item in expanded] == [0, 4, 8, 12, 16]
    assert len({item.event_id for item in expanded}) == len(CONTEXT_SECONDS)
