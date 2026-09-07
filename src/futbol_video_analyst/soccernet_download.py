import argparse
import getpass
import json
import subprocess
from collections import Counter
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np

from futbol_video_analyst.soccernet_import import _normalize_label

TARGET_LABELS = {
    "corner": "corner",
    "shot on target": "shot_attempt",
    "shots on target": "shot_attempt",
    "shot off target": "shot_attempt",
    "shots off target": "shot_attempt",
    "goal": "shot_attempt",
}


def _official_games(split: str) -> list[str]:
    try:
        from SoccerNet.Downloader import getListGames
    except ImportError as error:
        raise RuntimeError("Instala el extra: pip install -e '.[external-data]'") from error
    return list(getListGames(split))


def _target_counts(labels_path: Path) -> Counter[str]:
    payload = json.loads(labels_path.read_text(encoding="utf-8"))
    counts: Counter[str] = Counter()
    for annotation in payload.get("annotations", []):
        mapped = TARGET_LABELS.get(_normalize_label(str(annotation.get("label", ""))))
        if mapped:
            counts[mapped] += 1
    return counts


def create_pilot_plan(
    source: Path,
    output: Path,
    split_sizes: dict[str, int] | None = None,
    games_by_split: dict[str, list[str]] | None = None,
) -> dict[str, Any]:
    sizes = split_sizes or {"train": 20, "valid": 5}
    official = games_by_split or {split: _official_games(split) for split in sizes}
    selections: list[dict[str, Any]] = []
    for split, wanted in sizes.items():
        candidates: list[tuple[int, int, str, Counter[str]]] = []
        for game in official[split]:
            labels_path = source / game / "Labels-v2.json"
            if not labels_path.is_file():
                continue
            counts = _target_counts(labels_path)
            # Prefer matches that contain both targets, then maximize the rarer one.
            candidates.append(
                (
                    int(bool(counts["corner"]) and bool(counts["shot_attempt"])),
                    min(counts["corner"], counts["shot_attempt"]),
                    game,
                    counts,
                )
            )
        if len(candidates) < wanted:
            raise ValueError(f"Only {len(candidates)} labeled {split} matches are available")
        selected = sorted(candidates, key=lambda item: (-item[0], -item[1], item[2]))[:wanted]
        selections.extend(
            {
                "split": split,
                "game": game,
                "target_counts": dict(sorted(counts.items())),
            }
            for _, _, game, counts in selected
        )

    total_counts: Counter[str] = Counter()
    for selection in selections:
        total_counts.update(selection["target_counts"])
    plan = {
        "created_at": datetime.now(UTC).isoformat(),
        "purpose": "SoccerNet-v2 research-only architecture pilot",
        "commercial_model_eligible": False,
        "video_files": ["1_224p.mkv", "2_224p.mkv"],
        "matches": selections,
        "target_counts": dict(sorted(total_counts.items())),
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(plan, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    return plan


def download_pilot(source: Path, plan_path: Path) -> None:
    try:
        import SoccerNet.Downloader as downloader_module
    except ImportError as error:
        raise RuntimeError("Instala el extra: pip install -e '.[external-data]'") from error

    plan = json.loads(plan_path.read_text(encoding="utf-8"))
    password = getpass.getpass("Contraseña de videos SoccerNet (no se guardará): ")
    if not password:
        raise ValueError("La contraseña no puede estar vacía")
    # A telemetry timeout must not interrupt authenticated dataset downloads.
    downloader_module.report = lambda *_args, **_kwargs: None
    downloader = downloader_module.SoccerNetDownloader(LocalDirectory=str(source))
    downloader.password = password
    try:
        for index, selection in enumerate(plan["matches"], 1):
            game = selection["game"]
            split = selection["split"]
            print(f"[{index}/{len(plan['matches'])}] {split}: {game}", flush=True)
            downloader.downloadGame(
                game,
                files=plan["video_files"],
                spl=selection.get("source_split", split),
                verbose=True,
            )
    finally:
        password = ""  # Drop the reference as soon as downloads finish.


def keychain_password(service: str) -> str:
    result = subprocess.run(
        ["security", "find-generic-password", "-w", "-s", service],
        check=True,
        capture_output=True,
        text=True,
    )
    password = result.stdout.rstrip("\n")
    if not password:
        raise ValueError(f"El llavero no contiene una contraseña para {service}")
    return password


def download_with_password(
    source: Path, plan_path: Path, password: str, cache_dir: Path | None = None
) -> None:
    try:
        import SoccerNet.Downloader as downloader_module
    except ImportError as error:
        raise RuntimeError("Instala el extra: pip install -e '.[external-data]'") from error
    plan = json.loads(plan_path.read_text(encoding="utf-8"))
    downloader_module.report = lambda *_args, **_kwargs: None
    downloader = downloader_module.SoccerNetDownloader(LocalDirectory=str(source))
    downloader.password = password
    for index, selection in enumerate(plan["matches"], 1):
        game = selection["game"]
        if cache_dir is not None and all(
            (
                cache_dir
                / f"{f'soccernet-v2/{game}/half-{half}'.replace('/', '__')}.npz"
            ).is_file()
            for half in (1, 2)
        ):
            print(f"[{index}/{len(plan['matches'])}] caché completa: {game}")
            continue
        print(
            f"[{index}/{len(plan['matches'])}] {selection['split']}: {selection['game']}",
            flush=True,
        )
        downloader.downloadGame(
            selection["game"],
            files=plan["video_files"],
            spl=selection.get("source_split", selection["split"]),
            verbose=False,
        )


def create_pending_batches(
    plan_path: Path, output_dir: Path, cache_dir: Path, batch_size: int
) -> list[Path]:
    if batch_size < 1:
        raise ValueError("batch_size must be positive")
    plan = json.loads(plan_path.read_text(encoding="utf-8"))
    pending = []
    for selection in plan["matches"]:
        game = selection["game"]
        cached = all(
            (
                cache_dir
                / f"{f'soccernet-v2/{game}/half-{half}'.replace('/', '__')}.npz"
            ).is_file()
            for half in (1, 2)
        )
        if not cached:
            pending.append(selection)
    output_dir.mkdir(parents=True, exist_ok=True)
    paths: list[Path] = []
    for index, offset in enumerate(range(0, len(pending), batch_size), 1):
        batch = dict(plan)
        batch["purpose"] = f"{plan['purpose']} - pending batch {index}"
        batch["matches"] = pending[offset : offset + batch_size]
        path = output_dir / f"batch-{index:03d}.json"
        path.write_text(json.dumps(batch, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
        paths.append(path)
    return paths


def purge_prepared_videos(source: Path, plan_path: Path, cache_dir: Path) -> list[Path]:
    source = source.resolve()
    plan = json.loads(plan_path.read_text(encoding="utf-8"))
    targets: list[Path] = []
    for selection in plan["matches"]:
        game = selection["game"]
        for half, filename in enumerate(plan["video_files"], 1):
            cache_path = (
                cache_dir
                / f"{f'soccernet-v2/{game}/half-{half}'.replace('/', '__')}.npz"
            )
            if not cache_path.is_file():
                raise ValueError(f"No se borrará {game}: falta {cache_path.name}")
            with np.load(cache_path) as cached:
                timebase = str(cached.get("timebase", ""))
                if timebase != "presentation_timestamp" or not len(cached["embeddings"]):
                    raise ValueError(f"Caché inválido: {cache_path}")
            video = (source / game / filename).resolve()
            if not video.is_relative_to(source):
                raise ValueError(f"Ruta fuera de SoccerNet: {video}")
            if video.is_file():
                targets.append(video)
    for video in targets:
        video.unlink()
        print(f"Eliminado después de verificar caché: {video}", flush=True)
    return targets


def main() -> None:
    parser = argparse.ArgumentParser(description="Plan or download a SoccerNet research pilot")
    subparsers = parser.add_subparsers(dest="command", required=True)
    plan_parser = subparsers.add_parser("plan")
    plan_parser.add_argument("--source", type=Path, required=True)
    plan_parser.add_argument("--output", type=Path, required=True)
    plan_parser.add_argument("--train-matches", type=int, default=20)
    plan_parser.add_argument("--valid-matches", type=int, default=5)
    download_parser = subparsers.add_parser("download")
    download_parser.add_argument("--source", type=Path, required=True)
    download_parser.add_argument("--plan", type=Path, required=True)
    download_parser.add_argument("--keychain-service")
    download_parser.add_argument("--cache-dir", type=Path)
    batches_parser = subparsers.add_parser("batches")
    batches_parser.add_argument("--plan", type=Path, required=True)
    batches_parser.add_argument("--output-dir", type=Path, required=True)
    batches_parser.add_argument("--cache-dir", type=Path, required=True)
    batches_parser.add_argument("--batch-size", type=int, default=25)
    purge_parser = subparsers.add_parser("purge")
    purge_parser.add_argument("--source", type=Path, required=True)
    purge_parser.add_argument("--plan", type=Path, required=True)
    purge_parser.add_argument("--cache-dir", type=Path, required=True)
    purge_parser.add_argument("--confirm", action="store_true")
    arguments = parser.parse_args()
    if arguments.command == "plan":
        plan = create_pilot_plan(
            arguments.source,
            arguments.output,
            {"train": arguments.train_matches, "valid": arguments.valid_matches},
        )
        print(json.dumps(plan["target_counts"], indent=2, ensure_ascii=False))
    elif arguments.command == "download":
        if arguments.keychain_service:
            password = keychain_password(arguments.keychain_service)
            try:
                download_with_password(
                    arguments.source, arguments.plan, password, arguments.cache_dir
                )
            finally:
                password = ""
        else:
            download_pilot(arguments.source, arguments.plan)
    elif arguments.command == "batches":
        paths = create_pending_batches(
            arguments.plan, arguments.output_dir, arguments.cache_dir, arguments.batch_size
        )
        print(f"{len(paths)} batches pendientes")
        for path in paths:
            print(path)
    elif arguments.confirm:
        removed = purge_prepared_videos(
            arguments.source, arguments.plan, arguments.cache_dir
        )
        print(f"Videos eliminados: {len(removed)}")
    else:
        raise ValueError("La limpieza requiere --confirm")


if __name__ == "__main__":
    main()
