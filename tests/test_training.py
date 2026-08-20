import json
from pathlib import Path

import pytest

from futbol_video_analyst.training import choose_validation_match, load_examples, split_examples


def write_manifest(dataset: Path) -> None:
    records = []
    for match_id, title, labels in [
        ("first", "Primero", ["corner", "corner", "negative", "negative", "negative"]),
        (
            "second",
            "Segundo",
            ["corner", "corner", "corner", "negative", "negative", "negative", "negative", "negative"],
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


def test_requires_more_than_one_match(tmp_path: Path) -> None:
    write_manifest(tmp_path)
    examples = [example for example in load_examples(tmp_path) if example.match_id == "first"]

    with pytest.raises(ValueError, match="al menos dos partidos"):
        choose_validation_match(examples, None)
