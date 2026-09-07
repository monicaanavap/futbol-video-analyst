import numpy as np
import pytest

from futbol_video_analyst.action_spotting import ActionSpottingConfig, extract_spots


def test_action_spotting_contract_uses_evidence_backed_window() -> None:
    config = ActionSpottingConfig()

    assert config.window_seconds == 64
    assert config.stride_seconds == 32
    assert config.labels == ("corner", "goal_kick", "shot_attempt")


def test_extracts_local_maxima_with_class_specific_suppression() -> None:
    config = ActionSpottingConfig()
    scores = np.zeros((8, 3), dtype=np.float32)
    scores[1, 0] = 0.8
    scores[3, 0] = 0.9
    scores[5, 2] = 0.75
    scores[7, 2] = 0.85

    spots = extract_spots(
        scores,
        np.arange(8) * 2.0,
        config,
        {"corner": 0.7, "goal_kick": 0.7, "shot_attempt": 0.7},
    )

    assert [(spot.event_type, spot.timestamp_seconds) for spot in spots] == [
        ("corner", 6.0),
        ("shot_attempt", 10.0),
        ("shot_attempt", 14.0),
    ]


def test_rejects_missing_per_class_thresholds() -> None:
    with pytest.raises(ValueError, match="Missing thresholds"):
        extract_spots(
            np.zeros((1, 3)),
            np.zeros(1),
            ActionSpottingConfig(),
            {"corner": 0.5},
        )
