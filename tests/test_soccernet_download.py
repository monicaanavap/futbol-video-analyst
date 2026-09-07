import json
from pathlib import Path

import numpy as np

from futbol_video_analyst.soccernet_download import (
    create_pending_batches,
    create_pilot_plan,
    purge_prepared_videos,
)


def write_labels(root: Path, game: str, corners: int, shots: int) -> None:
    folder = root / game
    folder.mkdir(parents=True)
    annotations = [
        {"label": "Corner", "gameTime": "1 - 00:01", "position": "1000"}
        for _ in range(corners)
    ] + [
        {"label": "Shots on target", "gameTime": "1 - 00:02", "position": "2000"}
        for _ in range(shots)
    ]
    (folder / "Labels-v2.json").write_text(json.dumps({"annotations": annotations}))


def test_pilot_plan_preserves_split_and_prefers_target_rich_matches(tmp_path: Path) -> None:
    source = tmp_path / "source"
    write_labels(source, "train/a", corners=2, shots=2)
    write_labels(source, "train/b", corners=5, shots=5)
    write_labels(source, "valid/c", corners=3, shots=4)
    output = tmp_path / "pilot.json"

    plan = create_pilot_plan(
        source,
        output,
        {"train": 1, "valid": 1},
        {"train": ["train/a", "train/b"], "valid": ["valid/c"]},
    )

    assert [(item["split"], item["game"]) for item in plan["matches"]] == [
        ("train", "train/b"),
        ("valid", "valid/c"),
    ]
    assert plan["target_counts"] == {"corner": 8, "shot_attempt": 9}
    assert json.loads(output.read_text())["commercial_model_eligible"] is False


def test_pending_batches_skip_matches_with_both_cached_halves(tmp_path: Path) -> None:
    plan = {
        "purpose": "test",
        "video_files": ["1_224p.mkv", "2_224p.mkv"],
        "matches": [
            {"game": "league/season/cached", "split": "train"},
            {"game": "league/season/pending", "split": "train"},
        ],
    }
    plan_path = tmp_path / "plan.json"
    plan_path.write_text(json.dumps(plan), encoding="utf-8")
    cache = tmp_path / "cache"
    cache.mkdir()
    for half in (1, 2):
        (cache / f"soccernet-v2__league__season__cached__half-{half}.npz").touch()

    paths = create_pending_batches(plan_path, tmp_path / "batches", cache, 25)

    assert len(paths) == 1
    result = json.loads(paths[0].read_text(encoding="utf-8"))
    assert [item["game"] for item in result["matches"]] == ["league/season/pending"]


def test_purge_requires_valid_cache_for_every_half(tmp_path: Path) -> None:
    source = tmp_path / "source"
    game = source / "league" / "season" / "game"
    game.mkdir(parents=True)
    for half in (1, 2):
        (game / f"{half}_224p.mkv").touch()
    plan = {
        "video_files": ["1_224p.mkv", "2_224p.mkv"],
        "matches": [{"game": "league/season/game", "split": "train"}],
    }
    plan_path = tmp_path / "plan.json"
    plan_path.write_text(json.dumps(plan), encoding="utf-8")
    cache = tmp_path / "cache"
    cache.mkdir()
    for half in (1, 2):
        np.savez_compressed(
            cache / f"soccernet-v2__league__season__game__half-{half}.npz",
            embeddings=np.ones((2, 2)),
            timebase=np.asarray("presentation_timestamp"),
        )

    removed = purge_prepared_videos(source, plan_path, cache)

    assert len(removed) == 2
    assert not (game / "1_224p.mkv").exists()
