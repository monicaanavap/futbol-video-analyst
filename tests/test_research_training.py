from collections import defaultdict

import numpy as np

from futbol_video_analyst.research_training import (
    _next_sample_time,
    build_targets,
    build_temporal_head,
    window_starts,
)


def test_build_targets_marks_nearby_samples_by_class() -> None:
    timestamps = np.arange(0, 5, 0.5)
    events = defaultdict(list, {"corner": [2.0], "shot_attempt": [4.0]})

    targets = build_targets(
        timestamps,
        events,
        ("corner", "shot_attempt"),
        positive_radius_seconds=0.5,
    )

    assert np.flatnonzero(targets[:, 0]).tolist() == [3, 4, 5]
    assert np.flatnonzero(targets[:, 1]).tolist() == [7, 8, 9]


def test_window_starts_covers_final_timestep() -> None:
    assert window_starts(300, 128, 64) == [0, 64, 128, 172]
    assert window_starts(80, 128, 64) == [0]


def test_sampling_grid_tolerates_frame_jitter_and_skips_true_gaps() -> None:
    assert _next_sample_time(0.5, timestamp=0.52, interval=0.5) == 1.0
    assert _next_sample_time(10.0, timestamp=20.0, interval=0.5) == 20.5


def test_temporal_heads_preserve_timeline_shape() -> None:
    import torch

    features = torch.zeros((2, 32, 24), dtype=torch.float32)

    for architecture in ("bigru", "tcn"):
        head = build_temporal_head(24, 16, 3, architecture=architecture)
        assert head(features).shape == (2, 32, 3)
