import json
import subprocess
from pathlib import Path

from futbol_video_analyst.domain import VideoMetadata


class VideoInspectionError(ValueError):
    pass


def _parse_frame_rate(value: str) -> float:
    numerator, separator, denominator = value.partition("/")
    if not separator:
        return float(value)
    if float(denominator) == 0:
        raise VideoInspectionError("The video reports an invalid frame rate")
    return float(numerator) / float(denominator)


class FFprobeVideoInspector:
    def inspect(self, path: Path) -> VideoMetadata:
        resolved_path = path.expanduser().resolve()
        if not resolved_path.is_file():
            raise VideoInspectionError("Video file does not exist")
        command = [
            "ffprobe",
            "-v",
            "error",
            "-select_streams",
            "v:0",
            "-show_entries",
            "stream=width,height,codec_name,avg_frame_rate:format=duration",
            "-of",
            "json",
            str(resolved_path),
        ]
        try:
            result = subprocess.run(command, capture_output=True, text=True, check=True)
            payload = json.loads(result.stdout)
            stream = payload["streams"][0]
            return VideoMetadata(
                duration_seconds=float(payload["format"]["duration"]),
                width=int(stream["width"]),
                height=int(stream["height"]),
                fps=_parse_frame_rate(stream["avg_frame_rate"]),
                codec=stream["codec_name"],
            )
        except (
            subprocess.CalledProcessError,
            KeyError,
            IndexError,
            ValueError,
            json.JSONDecodeError,
        ) as error:
            raise VideoInspectionError("Unable to read video metadata") from error
