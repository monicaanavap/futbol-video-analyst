import json
from pathlib import Path

import pytest

from futbol_video_analyst.soccernet_import import import_soccernet


def prepare_match(root: Path) -> Path:
    game = root / "england_epl" / "2016-2017" / "Example Match"
    game.mkdir(parents=True)
    (game / "1_224p.mkv").touch()
    (game / "2_224p.mkv").touch()
    (game / "Labels-v2.json").write_text(
        json.dumps(
            {
                "annotations": [
                    {"gameTime": "1 - 10:30", "position": "630000", "label": "Corner"},
                    {"gameTime": "2 - 01:02", "position": "62000", "label": "Shots on target"},
                    {"gameTime": "2 - 02:00", "position": "120000", "label": "Kick-off"},
                ]
            }
        ),
        encoding="utf-8",
    )
    return game


def test_imports_soccernet_without_copying_videos(tmp_path: Path) -> None:
    source = tmp_path / "SoccerNet"
    prepare_match(source)

    manifest = import_soccernet(source, tmp_path / "converted")
    records = [json.loads(line) for line in manifest.read_text().splitlines()]
    summary = json.loads((manifest.parent / "summary.json").read_text())

    assert [record["label"] for record in records] == [
        "corner",
        "shot_attempt",
        "soccernet_kick_off",
    ]
    assert records[1]["peak_seconds"] == 62
    assert records[0]["clip_path"].endswith("1_224p.mkv")
    assert summary["commercial_model_eligible"] is False
    assert summary["matches"] == 1


def test_requires_half_video_by_default(tmp_path: Path) -> None:
    source = tmp_path / "SoccerNet"
    game = prepare_match(source)
    (game / "2_224p.mkv").unlink()

    with pytest.raises(ValueError, match="Missing video for half 2"):
        import_soccernet(source, tmp_path / "converted")


def test_can_write_expected_paths_before_batch_download(tmp_path: Path) -> None:
    source = tmp_path / "SoccerNet"
    game = prepare_match(source)
    (game / "2_224p.mkv").unlink()

    manifest = import_soccernet(
        source,
        tmp_path / "converted",
        require_videos=False,
        expected_video_paths=True,
    )
    records = [json.loads(line) for line in manifest.read_text().splitlines()]

    assert records[1]["clip_path"].endswith("2_224p.mkv")


def test_imports_only_matches_selected_by_the_pilot_plan(tmp_path: Path) -> None:
    source = tmp_path / "SoccerNet"
    prepare_match(source)
    other = source / "england_epl" / "2016-2017" / "Other Match"
    other.mkdir(parents=True)
    (other / "Labels-v2.json").write_text(json.dumps({"annotations": []}))
    selected = {"england_epl/2016-2017/Example Match": "valid"}

    manifest = import_soccernet(
        source,
        tmp_path / "converted",
        selected_splits=selected,
    )
    records = [json.loads(line) for line in manifest.read_text().splitlines()]
    summary = json.loads((manifest.parent / "summary.json").read_text())

    assert {record["match_title"] for record in records} == {"Example Match"}
    assert {record["split"] for record in records} == {"valid"}
    assert summary["splits"] == {"valid": 1}
