import subprocess
from pathlib import Path

import cv2
import numpy as np

from futbol_video_analyst.analysis import VisualSignalAnalyzer, detect_object_candidates
from futbol_video_analyst.domain import Match, MatchStatus


def test_marks_green_video_samples_as_likely_field(tmp_path: Path) -> None:
    video = tmp_path / "green-field.mp4"
    subprocess.run(
        [
            "ffmpeg",
            "-y",
            "-f",
            "lavfi",
            "-i",
            "color=c=0x238b45:s=320x180:d=2",
            "-c:v",
            "mpeg4",
            str(video),
        ],
        capture_output=True,
        check=True,
    )
    match = Match(
        id="match-id",
        title="Synthetic match",
        video_path=str(video),
        duration_seconds=2,
        width=320,
        height=180,
        fps=25,
        codec="mpeg4",
        status=MatchStatus.READY,
        created_at="2026-08-17 00:00:00",
    )
    progress: list[float] = []

    signals = VisualSignalAnalyzer(sample_interval_seconds=0.5).analyze(
        match, lambda value, _: progress.append(value)
    )

    assert len(signals) == 4
    assert all(signal.likely_field for signal in signals)
    assert all(signal.green_ratio > 0.8 for signal in signals)
    assert progress[-1] == 1


def test_detects_experimental_player_ball_and_line_candidates() -> None:
    frame = np.full((216, 384, 3), (45, 135, 45), dtype=np.uint8)
    cv2.line(frame, (20, 180), (360, 180), (255, 255, 255), 3)
    cv2.rectangle(frame, (100, 90), (110, 125), (20, 20, 220), -1)
    cv2.rectangle(frame, (220, 80), (231, 120), (220, 20, 20), -1)
    cv2.circle(frame, (180, 145), 4, (255, 255, 255), -1)
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    green_mask = cv2.inRange(hsv, (30, 35, 30), (95, 255, 255))

    players, balls, line_ratio = detect_object_candidates(frame, hsv, green_mask)

    assert players >= 2
    assert balls >= 1
    assert line_ratio > 0
