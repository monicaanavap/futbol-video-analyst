import json
from pathlib import Path

import pytest

from futbol_video_analyst.soccertrack_import import import_matches


def prepare_source(root: Path) -> Path:
    match_id = "118575"
    annotations = root / "bas" / match_id
    videos = root / "videos" / match_id
    annotations.mkdir(parents=True)
    videos.mkdir(parents=True)
    (videos / f"{match_id}_panorama_1st_half.mp4").touch()
    (videos / f"{match_id}_panorama_2nd_half.mp4").touch()
    payload = {
        "match_id": match_id,
        "fps": 25.0,
        "actions": [
            {"gameTime": "1 - 12:30", "position": "750000", "label": "SHOT"},
            {"gameTime": "2 - 49:02", "position": "2942000", "label": "GOAL"},
            {"gameTime": "1 - 8:00", "position": "480000", "label": "FREE KICK"},
            {"gameTime": "1 - 1:00", "position": "60000", "label": "PASS"},
        ],
    }
    (annotations / f"{match_id}_12_class_events.json").write_text(
        json.dumps(payload), encoding="utf-8"
    )
    return root


def test_imports_actions_with_commercial_provenance(tmp_path: Path) -> None:
    source = prepare_source(tmp_path / "source")
    output = tmp_path / "converted"

    manifest = import_matches(source, output, ["118575"], duration_reader=lambda _: 2800)
    records = [json.loads(line) for line in manifest.read_text().splitlines()]
    summary = json.loads((output / "summary.json").read_text())

    assert [record["label"] for record in records] == [
        "shot_attempt",
        "shot_attempt",
        "free_kick",
        "soccertrack_pass",
    ]
    assert records[1]["source_half"] == 2
    assert records[1]["peak_seconds"] == 242
    assert Path(records[0]["clip_path"]).is_absolute()
    assert all(record["license"] == "CC BY 4.0" for record in records)
    assert summary["mapped_labels"]["shot_attempt"] == 2


def test_requires_both_half_videos(tmp_path: Path) -> None:
    source = prepare_source(tmp_path / "source")
    (source / "videos" / "118575" / "118575_panorama_2nd_half.mp4").unlink()

    with pytest.raises(ValueError, match="Faltan videos"):
        import_matches(
            source,
            tmp_path / "converted",
            ["118575"],
            duration_reader=lambda _: 2800,
        )


def test_skips_malformed_and_out_of_video_actions(tmp_path: Path) -> None:
    source = prepare_source(tmp_path / "source")
    path = source / "bas" / "118575" / "118575_12_class_events.json"
    payload = json.loads(path.read_text())
    payload["actions"].extend(
        [
            {"gameTime": "90:00", "position": "5400000", "label": "SHOT"},
            {"gameTime": "2 - 99:00", "position": "5940000", "label": "SHOT"},
        ]
    )
    path.write_text(json.dumps(payload), encoding="utf-8")

    output = tmp_path / "converted"
    import_matches(source, output, ["118575"], duration_reader=lambda _: 2800)
    summary = json.loads((output / "summary.json").read_text())

    assert summary["skipped"] == {"invalid_game_time": 1, "outside_video": 1}
