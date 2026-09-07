import json
from pathlib import Path

import numpy as np

from futbol_video_analyst.domain import Match, MatchStatus, VisualSignal
from futbol_video_analyst.temporal import TemporalCornerSpotter, select_peaks, temporal_features


def signal(timestamp: float, change: float = 0.05) -> VisualSignal:
    return VisualSignal(
        id=f"signal-{timestamp}",
        match_id="match-id",
        timestamp_seconds=timestamp,
        green_ratio=0.7,
        brightness=0.5,
        change_score=change,
        likely_field=True,
        player_candidates=5,
        ball_candidates=1,
        line_ratio=0.02,
    )


def match() -> Match:
    return Match(
        id="match-id",
        title="Temporal test",
        video_path="/tmp/test.mp4",
        duration_seconds=120,
        width=1280,
        height=720,
        fps=30,
        codec="h264",
        status=MatchStatus.READY,
        created_at="2026-09-01 00:00:00",
    )


def test_temporal_features_include_multiple_context_windows() -> None:
    values = temporal_features([signal(index * 2, index / 20) for index in range(10)])

    assert values.shape == (10, 70)
    assert not np.array_equal(values[0], values[5])


def test_peak_selection_groups_neighbors_and_limits_results() -> None:
    timestamps = np.arange(0, 200, 2, dtype=np.float32)
    scores = np.zeros_like(timestamps)
    scores[[5, 7, 20, 40, 60, 80]] = [0.8, 0.9, 0.95, 0.91, 0.92, 0.93]

    peaks = select_peaks(timestamps, scores, 0.7, cooldown_seconds=20, max_candidates=3)

    assert len(peaks) == 3
    assert peaks == sorted(peaks)


def test_temporal_spotter_returns_none_without_model(tmp_path: Path) -> None:
    assert TemporalCornerSpotter(tmp_path / "missing.json").detect(match(), [signal(0)]) is None


def test_temporal_spotter_loads_portable_json_model(tmp_path: Path) -> None:
    signals = [signal(index * 2, 0.9 if index == 10 else 0.01) for index in range(30)]
    dimensions = temporal_features(signals).shape[1]
    model_path = tmp_path / "temporal.json"
    model_path.write_text(
        json.dumps(
            {
                "version": 1,
                "feature_mean": [0] * dimensions,
                "feature_scale": [1] * dimensions,
                "weights": [0] * dimensions,
                "bias": 10,
                "threshold": 0.5,
                "max_candidates": 2,
                "cooldown_seconds": 20,
            }
        )
    )

    events = TemporalCornerSpotter(model_path).detect(match(), signals)

    assert events is not None
    assert len(events) <= 2
    assert all("temporal" in (event.notes or "").lower() for event in events)
