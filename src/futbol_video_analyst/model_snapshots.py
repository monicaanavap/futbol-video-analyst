import argparse
import hashlib
import json
import re
import shutil
from datetime import UTC, datetime
from pathlib import Path
from typing import Any


def _read_env(path: Path) -> dict[str, str]:
    values: dict[str, str] = {}
    if not path.is_file():
        return values
    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        values[key.strip()] = value.strip()
    return values


def _write_env_values(path: Path, updates: dict[str, str]) -> None:
    lines = path.read_text(encoding="utf-8").splitlines() if path.is_file() else []
    remaining = dict(updates)
    result: list[str] = []
    for line in lines:
        stripped = line.strip()
        if stripped and not stripped.startswith("#") and "=" in stripped:
            key = stripped.split("=", 1)[0].strip()
            if key in remaining:
                result.append(f"{key}={remaining.pop(key)}")
                continue
        result.append(line)
    if remaining and result and result[-1]:
        result.append("")
    result.extend(f"{key}={value}" for key, value in sorted(remaining.items()))
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text("\n".join(result) + "\n", encoding="utf-8")
    temporary.replace(path)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _slug(value: str) -> str:
    normalized = re.sub(r"[^a-z0-9]+", "-", value.casefold()).strip("-")
    return normalized or "snapshot"


def _model_settings(values: dict[str, str]) -> dict[str, str]:
    return {
        key: value
        for key, value in values.items()
        if "MODEL" in key and key == key.upper()
    }


def create_snapshot(
    project_dir: Path, name: str, lineage: dict[str, Any] | None = None
) -> Path:
    project_dir = project_dir.resolve()
    env_path = project_dir / ".env"
    settings = _model_settings(_read_env(env_path))
    if not settings:
        raise ValueError(f"No se encontraron configuraciones de modelos en {env_path}")

    timestamp = datetime.now(UTC)
    snapshot_id = f"{timestamp:%Y%m%d-%H%M%S}-{_slug(name)}"
    snapshot_dir = project_dir / "models" / "snapshots" / snapshot_id
    artifacts_dir = snapshot_dir / "artifacts"
    artifacts_dir.mkdir(parents=True, exist_ok=False)

    artifacts: list[dict[str, Any]] = []
    restored_settings = dict(settings)
    copied_sources: dict[Path, Path] = {}
    for key, raw_path in settings.items():
        if not key.endswith("MODEL_PATH"):
            continue
        source = Path(raw_path)
        if not source.is_absolute():
            source = project_dir / source
        source = source.resolve()
        if not source.is_file():
            raise ValueError(f"El modelo activo de {key} no existe: {source}")
        destination = artifacts_dir / f"{key.casefold()}-{source.name}"
        shutil.copy2(source, destination)
        copied_sources[source] = destination
        restored_settings[key] = f"./{destination.relative_to(project_dir).as_posix()}"
        artifacts.append(
            {
                "config_key": key,
                "source": str(source),
                "snapshot_path": str(destination.relative_to(snapshot_dir)),
                "sha256": _sha256(destination),
                "bytes": destination.stat().st_size,
            }
        )

        companion = source.with_suffix(".json")
        if source.suffix != ".json" and companion.is_file() and companion not in copied_sources:
            companion_destination = artifacts_dir / f"metadata-{companion.name}"
            shutil.copy2(companion, companion_destination)
            copied_sources[companion] = companion_destination
            artifacts.append(
                {
                    "config_key": None,
                    "source": str(companion),
                    "snapshot_path": str(companion_destination.relative_to(snapshot_dir)),
                    "sha256": _sha256(companion_destination),
                    "bytes": companion_destination.stat().st_size,
                }
            )

    manifest = {
        "snapshot_version": 1,
        "id": snapshot_id,
        "name": name,
        "created_at": timestamp.isoformat(),
        "project_dir": str(project_dir),
        "original_settings": settings,
        "restore_settings": restored_settings,
        "lineage": lineage or {
            "commercial_status": "unknown",
            "training_sources": [],
            "excluded_sources": [],
        },
        "artifacts": artifacts,
    }
    manifest_path = snapshot_dir / "snapshot.json"
    manifest_path.write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    return manifest_path


def list_snapshots(project_dir: Path) -> list[dict[str, Any]]:
    manifests = sorted(
        (project_dir.resolve() / "models" / "snapshots").glob("*/snapshot.json"),
        reverse=True,
    )
    return [json.loads(path.read_text(encoding="utf-8")) for path in manifests]


def restore_snapshot(project_dir: Path, snapshot_id: str) -> Path:
    project_dir = project_dir.resolve()
    manifest_path = project_dir / "models" / "snapshots" / snapshot_id / "snapshot.json"
    if not manifest_path.is_file():
        raise ValueError(f"No existe el snapshot {snapshot_id}")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    snapshot_dir = manifest_path.parent
    for artifact in manifest["artifacts"]:
        path = snapshot_dir / artifact["snapshot_path"]
        if not path.is_file() or _sha256(path) != artifact["sha256"]:
            raise ValueError(f"El snapshot está incompleto o alterado: {path}")
    _write_env_values(project_dir / ".env", manifest["restore_settings"])
    return manifest_path


def main() -> None:
    parser = argparse.ArgumentParser(description="Crea y restaura snapshots de modelos activos")
    parser.add_argument("--project-dir", type=Path, default=Path.cwd())
    commands = parser.add_subparsers(dest="command", required=True)
    create = commands.add_parser("create", help="Guardar los modelos activos")
    create.add_argument("--name", required=True)
    create.add_argument(
        "--commercial-status",
        default="unknown",
        choices=["unknown", "candidate_pending_rights_review", "research_only"],
    )
    create.add_argument("--data-source", action="append", default=[])
    create.add_argument("--excluded-source", action="append", default=[])
    commands.add_parser("list", help="Listar snapshots disponibles")
    restore = commands.add_parser("restore", help="Activar un snapshot anterior")
    restore.add_argument("snapshot_id")
    arguments = parser.parse_args()

    if arguments.command == "create":
        path = create_snapshot(
            arguments.project_dir,
            arguments.name,
            {
                "commercial_status": arguments.commercial_status,
                "training_sources": arguments.data_source,
                "excluded_sources": arguments.excluded_source,
            },
        )
        print(f"Snapshot creado: {path}")
    elif arguments.command == "list":
        snapshots = list_snapshots(arguments.project_dir)
        if not snapshots:
            print("No hay snapshots")
        for snapshot in snapshots:
            print(f"{snapshot['id']}  {snapshot['name']}  {snapshot['created_at']}")
    else:
        path = restore_snapshot(arguments.project_dir, arguments.snapshot_id)
        print(f"Snapshot restaurado: {path}")
        print("Reinicia la aplicación para cargar los modelos restaurados.")


if __name__ == "__main__":
    main()
