import argparse
import json
import re
from collections import Counter
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

SOURCE_URL = "https://www.soccer-net.org/data"
USAGE_RESTRICTION = "Research-only; not intended for commercial purposes"


def _normalize_label(label: str) -> str:
    return re.sub(r"[^a-z0-9]+", " ", label.casefold()).strip()


LABEL_MAP = {
    "corner": "corner",
    "shot on target": "shot_attempt",
    "shots on target": "shot_attempt",
    "shot off target": "shot_attempt",
    "shots off target": "shot_attempt",
    "goal": "shot_attempt",
    "direct free kick": "free_kick",
    "indirect free kick": "free_kick",
    "free kick": "free_kick",
}


def _annotation_time(annotation: dict[str, Any]) -> tuple[int, float]:
    game_time = str(annotation.get("gameTime", ""))
    match = re.match(r"\s*([12])\s*-", game_time)
    if not match:
        raise ValueError(f"Invalid SoccerNet gameTime: {game_time!r}")
    half = int(match.group(1))
    return half, float(annotation["position"]) / 1000


def _find_video(game: Path, half: int) -> Path | None:
    for filename in (f"{half}_224p.mkv", f"{half}_720p.mkv", f"{half}.mkv", f"{half}.mp4"):
        candidate = game / filename
        if candidate.is_file():
            return candidate.resolve()
    return None


def import_soccernet(
    source: Path,
    output: Path,
    require_videos: bool = True,
    selected_splits: dict[str, str] | None = None,
    expected_video_paths: bool = False,
) -> Path:
    source = source.resolve()
    label_files = sorted(source.rglob("Labels-v2.json"))
    if selected_splits is not None:
        label_files = [
            path
            for path in label_files
            if str(path.parent.relative_to(source)) in selected_splits
        ]
        found = {str(path.parent.relative_to(source)) for path in label_files}
        missing = set(selected_splits) - found
        if missing:
            raise ValueError(f"Selected matches without labels: {', '.join(sorted(missing))}")
    if not label_files:
        raise ValueError(f"No Labels-v2.json files were found below {source}")
    if output.exists():
        raise ValueError(f"Output already exists: {output}")

    records: list[dict[str, Any]] = []
    mapped_counts: Counter[str] = Counter()
    source_counts: Counter[str] = Counter()
    skipped: Counter[str] = Counter()
    matches: set[str] = set()
    for labels_path in label_files:
        game = labels_path.parent
        match_id = str(game.relative_to(source))
        payload = json.loads(labels_path.read_text(encoding="utf-8"))
        annotations = payload.get("annotations")
        if not isinstance(annotations, list):
            raise TypeError(f"Invalid annotations in {labels_path}")
        matches.add(match_id)
        for index, annotation in enumerate(annotations):
            try:
                half, timestamp = _annotation_time(annotation)
            except (KeyError, TypeError, ValueError):
                skipped["invalid_timestamp"] += 1
                continue
            video = _find_video(game, half)
            if video is None and expected_video_paths:
                video = (game / f"{half}_224p.mkv").resolve()
            if video is None and require_videos:
                raise ValueError(f"Missing video for half {half}: {game}")
            source_label = str(annotation.get("label", "")).strip()
            normalized = _normalize_label(source_label)
            mapped_label = LABEL_MAP.get(normalized, f"soccernet_{normalized.replace(' ', '_')}")
            source_counts[source_label] += 1
            mapped_counts[mapped_label] += 1
            records.append(
                {
                    "clip_path": str(video) if video else None,
                    "label": mapped_label,
                    "match_id": f"soccernet-v2/{match_id}/half-{half}",
                    "match_title": game.name,
                    "event_id": f"soccernet-v2/{match_id}/{half}/{index}",
                    "start_seconds": 0,
                    "peak_seconds": timestamp,
                    "end_seconds": timestamp,
                    "source": "SoccerNet-v2",
                    "split": selected_splits.get(match_id) if selected_splits else None,
                    "source_label": source_label,
                    "source_half": half,
                    "visibility": annotation.get("visibility"),
                    "team": annotation.get("team"),
                    "usage_restriction": USAGE_RESTRICTION,
                }
            )

    output.mkdir(parents=True)
    manifest = output / "manifest.jsonl"
    manifest.write_text(
        "\n".join(json.dumps(record, ensure_ascii=False) for record in records) + "\n",
        encoding="utf-8",
    )
    (output / "summary.json").write_text(
        json.dumps(
            {
                "created_at": datetime.now(UTC).isoformat(),
                "source": "SoccerNet-v2",
                "source_url": SOURCE_URL,
                "usage_restriction": USAGE_RESTRICTION,
                "commercial_model_eligible": False,
                "changes": "Mapped SoccerNet annotations to the local research manifest; videos were not copied.",
                "matches": len(matches),
                "splits": dict(sorted(Counter(selected_splits.values()).items()))
                if selected_splits
                else None,
                "events": len(records),
                "mapped_labels": dict(sorted(mapped_counts.items())),
                "source_labels": dict(sorted(source_counts.items())),
                "skipped": dict(sorted(skipped.items())),
            },
            indent=2,
            ensure_ascii=False,
        )
        + "\n",
        encoding="utf-8",
    )
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description="Convert SoccerNet-v2 for research experiments")
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--labels-only", action="store_true")
    parser.add_argument("--plan", type=Path)
    parser.add_argument("--expected-video-paths", action="store_true")
    arguments = parser.parse_args()
    selected_splits = None
    if arguments.plan:
        plan = json.loads(arguments.plan.read_text(encoding="utf-8"))
        selected_splits = {item["game"]: item["split"] for item in plan["matches"]}
    manifest = import_soccernet(
        arguments.source,
        arguments.output,
        require_videos=not arguments.labels_only,
        selected_splits=selected_splits,
        expected_video_paths=arguments.expected_video_paths,
    )
    print(f"SoccerNet research manifest: {manifest}")


if __name__ == "__main__":
    main()
