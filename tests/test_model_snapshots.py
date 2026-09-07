import json
from pathlib import Path

import pytest

from futbol_video_analyst.model_snapshots import (
    create_snapshot,
    list_snapshots,
    restore_snapshot,
)


def prepare_project(project: Path) -> None:
    models = project / "models"
    models.mkdir()
    (models / "corner-action-v006.pt").write_bytes(b"action model")
    (models / "corner-action-v006.json").write_text('{"metric": 0.9}', encoding="utf-8")
    (models / "corner-temporal-v001.json").write_text('{"threshold": 0.5}', encoding="utf-8")
    (project / ".env").write_text(
        "APP_ENV=development\n"
        "CORNER_MODEL_ENABLED=true\n"
        "CORNER_MODEL_PATH=./models/corner-action-v006.pt\n"
        "CORNER_MODEL_DEVICE=auto\n"
        "CORNER_TEMPORAL_MODEL_ENABLED=true\n"
        "CORNER_TEMPORAL_MODEL_PATH=./models/corner-temporal-v001.json\n",
        encoding="utf-8",
    )


def test_snapshot_copies_active_models_and_restores_configuration(tmp_path: Path) -> None:
    prepare_project(tmp_path)

    manifest_path = create_snapshot(tmp_path, "Antes de bulk training")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))

    assert manifest["name"] == "Antes de bulk training"
    assert len(manifest["artifacts"]) == 3
    assert all(len(artifact["sha256"]) == 64 for artifact in manifest["artifacts"])
    assert list_snapshots(tmp_path)[0]["id"] == manifest["id"]

    (tmp_path / ".env").write_text("CORNER_MODEL_ENABLED=false\n", encoding="utf-8")
    restore_snapshot(tmp_path, manifest["id"])
    restored = (tmp_path / ".env").read_text(encoding="utf-8")

    assert "CORNER_MODEL_ENABLED=true" in restored
    assert f"models/snapshots/{manifest['id']}/artifacts" in restored
    assert "APP_ENV" not in restored


def test_restore_rejects_changed_snapshot_artifact(tmp_path: Path) -> None:
    prepare_project(tmp_path)
    manifest_path = create_snapshot(tmp_path, "Seguro")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    artifact = manifest_path.parent / manifest["artifacts"][0]["snapshot_path"]
    artifact.write_bytes(b"changed")

    with pytest.raises(ValueError, match="incompleto o alterado"):
        restore_snapshot(tmp_path, manifest["id"])


def test_snapshot_records_training_lineage(tmp_path: Path) -> None:
    prepare_project(tmp_path)
    lineage = {
        "commercial_status": "candidate_pending_rights_review",
        "training_sources": ["user-reviewed local matches"],
        "excluded_sources": ["SoccerNet-v2"],
    }

    manifest_path = create_snapshot(tmp_path, "Pre SoccerNet", lineage)
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))

    assert manifest["lineage"] == lineage
