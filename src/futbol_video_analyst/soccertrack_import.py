import argparse
import json
from collections import Counter
from collections.abc import Callable
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import cv2

LABEL_MAP = {
    "SHOT": "shot_attempt",
    "GOAL": "shot_attempt",
    "FREE KICK": "free_kick",
}
LICENSE = "CC BY 4.0"
SOURCE_URL = "https://huggingface.co/datasets/atomscott/soccertrack-v2"


def _half(action: dict[str, Any]) -> int:
    game_time = str(action.get("gameTime", ""))
    try:
        half = int(game_time.split("-", 1)[0].strip())
    except (TypeError, ValueError) as error:
        raise ValueError(f"gameTime inválido: {game_time!r}") from error
    if half not in {1, 2}:
        raise ValueError(f"Mitad no soportada en gameTime: {game_time!r}")
    return half


def _video_path(source: Path, match_id: str, half: int) -> Path:
    ordinal = "1st" if half == 1 else "2nd"
    return source / "videos" / match_id / f"{match_id}_panorama_{ordinal}_half.mp4"


def _video_duration(path: Path) -> float:
    capture = cv2.VideoCapture(str(path))
    try:
        if not capture.isOpened():
            raise ValueError(f"No se pudo abrir el video {path}")
        fps = capture.get(cv2.CAP_PROP_FPS)
        frames = capture.get(cv2.CAP_PROP_FRAME_COUNT)
        if fps <= 0 or frames <= 0:
            raise ValueError(f"El video no contiene duración válida: {path}")
        return frames / fps
    finally:
        capture.release()


def import_matches(
    source: Path,
    output: Path,
    match_ids: list[str],
    duration_reader: Callable[[Path], float] = _video_duration,
) -> Path:
    source = source.resolve()
    output = output.resolve()
    records: list[dict[str, Any]] = []
    source_counts: Counter[str] = Counter()
    mapped_counts: Counter[str] = Counter()
    skipped_counts: Counter[str] = Counter()

    for match_id in match_ids:
        annotations_path = source / "bas" / match_id / f"{match_id}_12_class_events.json"
        if not annotations_path.is_file():
            raise ValueError(f"No se encontraron anotaciones BAS para {match_id}")
        payload = json.loads(annotations_path.read_text(encoding="utf-8"))
        actions = payload.get("actions")
        if not isinstance(actions, list):
            raise TypeError(f"Formato BAS inválido en {annotations_path}")

        videos = {half: _video_path(source, match_id, half) for half in (1, 2)}
        missing = [path for path in videos.values() if not path.is_file()]
        if missing:
            raise ValueError(f"Faltan videos para {match_id}: {', '.join(map(str, missing))}")
        durations = {half: duration_reader(path) for half, path in videos.items()}

        for index, action in enumerate(actions):
            try:
                half = _half(action)
            except ValueError:
                skipped_counts["invalid_game_time"] += 1
                continue
            source_label = str(action["label"]).strip().upper()
            label = LABEL_MAP.get(source_label, f"soccertrack_{source_label.casefold().replace(' ', '_')}")
            source_position_seconds = float(action["position"]) / 1000
            peak_seconds = source_position_seconds - (half - 1) * 45 * 60
            if not 0 <= peak_seconds <= durations[half]:
                skipped_counts["outside_video"] += 1
                continue
            source_counts[source_label] += 1
            mapped_counts[label] += 1
            records.append(
                {
                    "clip_path": str(videos[half]),
                    "label": label,
                    "match_id": f"soccertrack-v2-{match_id}",
                    "match_title": f"SoccerTrack v2 {match_id}",
                    "event_id": f"soccertrack-v2-{match_id}-{index}",
                    "start_seconds": 0,
                    "peak_seconds": peak_seconds,
                    "end_seconds": peak_seconds,
                    "source": "soccertrack-v2",
                    "source_label": source_label,
                    "source_half": half,
                    "source_position_seconds": source_position_seconds,
                    "team": action.get("team"),
                    "player_id": action.get("player_id"),
                    "license": LICENSE,
                }
            )

    output.mkdir(parents=True, exist_ok=False)
    manifest_path = output / "manifest.jsonl"
    manifest_path.write_text(
        "\n".join(json.dumps(record, ensure_ascii=False) for record in records) + "\n",
        encoding="utf-8",
    )
    summary = {
        "created_at": datetime.now(UTC).isoformat(),
        "source": "SoccerTrack v2",
        "source_url": SOURCE_URL,
        "license": LICENSE,
        "license_url": "https://creativecommons.org/licenses/by/4.0/",
        "changes": "Converted BAS annotations to futbol-video-analyst manifest format.",
        "matches": match_ids,
        "events": len(records),
        "source_labels": dict(sorted(source_counts.items())),
        "mapped_labels": dict(sorted(mapped_counts.items())),
        "skipped": dict(sorted(skipped_counts.items())),
    }
    (output / "summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    return manifest_path


def main() -> None:
    parser = argparse.ArgumentParser(description="Convierte SoccerTrack v2 al manifiesto local")
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--matches", nargs="+", required=True)
    arguments = parser.parse_args()
    manifest = import_matches(arguments.source, arguments.output, arguments.matches)
    print(f"Dataset convertido: {manifest}")


if __name__ == "__main__":
    main()
