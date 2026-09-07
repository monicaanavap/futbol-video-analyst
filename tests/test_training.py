import json
from pathlib import Path

import pytest

from futbol_video_analyst.database import Database
from futbol_video_analyst.domain import VideoMetadata
from futbol_video_analyst.training import (
    balance_examples,
    choose_validation_match,
    load_background_examples,
    load_examples,
    split_examples,
)


def write_manifest(dataset: Path) -> None:
    records = []
    for match_id, title, labels in [
        ("first", "Primero", ["corner", "corner", "negative", "negative", "negative"]),
        (
            "second",
            "Segundo",
            ["corner", "corner", "corner", "negative", "negative", "negative", "negative", "foul"],
        ),
    ]:
        for index, label in enumerate(labels):
            clip = Path(match_id) / label / f"{index}.mp4"
            (dataset / clip).parent.mkdir(parents=True, exist_ok=True)
            (dataset / clip).touch()
            records.append(
                {
                    "clip_path": clip.as_posix(),
                    "label": label,
                    "match_id": match_id,
                    "match_title": title,
                    "event_id": f"{match_id}-{index}",
                    "start_seconds": 10,
                    "peak_seconds": 15,
                }
            )
    (dataset / "manifest.jsonl").write_text(
        "\n".join(json.dumps(record) for record in records), encoding="utf-8"
    )


def test_loads_and_splits_examples_by_complete_match(tmp_path: Path) -> None:
    write_manifest(tmp_path)
    examples = load_examples(tmp_path)
    validation_id = choose_validation_match(examples, "Primero")
    training, validation = split_examples(examples, validation_id)

    assert {example.match_id for example in training} == {"second"}
    assert {example.match_id for example in validation} == {"first"}
    assert next(example for example in examples if example.clip_path.parent.name == "foul").label == 0


def test_requires_more_than_one_match(tmp_path: Path) -> None:
    write_manifest(tmp_path)
    examples = [example for example in load_examples(tmp_path) if example.match_id == "first"]

    with pytest.raises(ValueError, match="al menos dos partidos"):
        choose_validation_match(examples, None)


def test_loads_task_specific_positive_labels(tmp_path: Path) -> None:
    write_manifest(tmp_path)
    records = [json.loads(line) for line in (tmp_path / "manifest.jsonl").read_text().splitlines()]
    records[0]["label"] = "shot_attempt"
    (tmp_path / "manifest.jsonl").write_text(
        "\n".join(json.dumps(record) for record in records), encoding="utf-8"
    )

    examples = load_examples(tmp_path, task="shot_attempt")

    assert sum(example.label for example in examples) == 1


def test_balances_negatives_per_match(tmp_path: Path) -> None:
    write_manifest(tmp_path)
    examples = load_examples(tmp_path)

    balanced = balance_examples(examples, negative_ratio=1)

    for match_id in {example.match_id for example in balanced}:
        current = [example for example in balanced if example.match_id == match_id]
        assert sum(example.label == 0 for example in current) == sum(
            example.label == 1 for example in current
        )

    with pytest.raises(ValueError, match="al menos uno"):
        balance_examples(examples, negative_ratio=0)


def test_samples_background_away_from_known_corners(tmp_path: Path) -> None:
    database_path = tmp_path / "training.sqlite3"
    database = Database(database_path)
    database.initialize()
    video = tmp_path / "match.mp4"
    video.touch()
    match = database.create_match(
        "Partido",
        str(video),
        VideoMetadata(duration_seconds=600, width=1280, height=720, fps=30, codec="h264"),
    )
    record = {
        "clip_path": "partido/corner/one.mp4",
        "label": "corner",
        "match_id": match.id,
        "match_title": match.title,
        "event_id": "corner-one",
        "start_seconds": 52,
        "peak_seconds": 60,
    }
    (tmp_path / "manifest.jsonl").write_text(json.dumps(record), encoding="utf-8")

    examples = load_background_examples(tmp_path, database_path, interval_seconds=120)

    assert [example.peak_in_clip for example in examples] == [180, 300, 420, 540]
    assert all(example.label == 0 for example in examples)

    with pytest.raises(ValueError, match="mayor que cero"):
        load_background_examples(tmp_path, database_path, interval_seconds=0)


def test_keeps_exported_dataset_usable_after_original_match_is_deleted(tmp_path: Path) -> None:
    database_path = tmp_path / "training.sqlite3"
    database = Database(database_path)
    database.initialize()
    record = {
        "clip_path": "partido/corner/one.mp4",
        "label": "corner",
        "match_id": "deleted-match",
        "match_title": "Partido eliminado",
        "event_id": "corner-one",
        "start_seconds": 52,
        "peak_seconds": 60,
    }
    (tmp_path / "manifest.jsonl").write_text(json.dumps(record), encoding="utf-8")

    assert load_background_examples(tmp_path, database_path, interval_seconds=120) == []
