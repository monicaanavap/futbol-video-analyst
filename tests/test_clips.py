import subprocess
from pathlib import Path

from futbol_video_analyst.clips import FFmpegClipExporter


def test_exports_a_playable_mp4_interval(tmp_path: Path) -> None:
    source = tmp_path / "source.mp4"
    destination = tmp_path / "clips" / "event.mp4"
    subprocess.run(
        [
            "ffmpeg",
            "-y",
            "-f",
            "lavfi",
            "-i",
            "color=c=green:s=320x180:d=2",
            "-c:v",
            "mpeg4",
            str(source),
        ],
        capture_output=True,
        check=True,
    )

    result = FFmpegClipExporter().export(source, destination, 0.25, 1.25)

    assert result == destination
    assert destination.stat().st_size > 0
    probe = subprocess.run(
        [
            "ffprobe",
            "-v",
            "error",
            "-show_entries",
            "format=duration",
            "-of",
            "default=noprint_wrappers=1:nokey=1",
            str(destination),
        ],
        capture_output=True,
        text=True,
        check=True,
    )
    assert 0.9 <= float(probe.stdout) <= 1.1
