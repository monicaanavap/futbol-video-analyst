import subprocess
from pathlib import Path


class ClipExportError(RuntimeError):
    pass


class FFmpegClipExporter:
    def export(self, source: Path, destination: Path, start: float, end: float) -> Path:
        if end <= start:
            raise ClipExportError("Clip end must be after its start")
        if not source.is_file():
            raise ClipExportError("The original video is no longer available")

        destination.parent.mkdir(parents=True, exist_ok=True)
        command = [
            "ffmpeg",
            "-y",
            "-ss",
            str(start),
            "-i",
            str(source),
            "-t",
            str(end - start),
            "-map",
            "0:v:0",
            "-map",
            "0:a?",
            "-c:v",
            "libx264",
            "-preset",
            "veryfast",
            "-crf",
            "21",
            "-c:a",
            "aac",
            "-movflags",
            "+faststart",
            str(destination),
        ]
        try:
            subprocess.run(command, capture_output=True, text=True, check=True)
        except (OSError, subprocess.CalledProcessError) as error:
            destination.unlink(missing_ok=True)
            raise ClipExportError("FFmpeg could not export this clip") from error
        return destination
