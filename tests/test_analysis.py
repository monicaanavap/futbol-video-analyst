import subprocess
from pathlib import Path

from futbol_video_analyst.analysis import VisualSignalAnalyzer
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
