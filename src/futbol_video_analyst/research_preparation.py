import argparse
from pathlib import Path

from futbol_video_analyst.action_spotting import ActionSpottingConfig
from futbol_video_analyst.research_training import (
    _ml,
    extract_video_embeddings,
    load_manifest,
)
from futbol_video_analyst.training import _device


def prepare_available(
    dataset: Path,
    cache_dir: Path,
    requested_device: str,
    encoder_batch_size: int,
) -> tuple[int, int]:
    config = ActionSpottingConfig(labels=("corner", "shot_attempt"))
    timelines = load_manifest(dataset / "manifest.jsonl", config.labels)
    torch, timm = _ml()
    device = _device(torch, requested_device)
    encoder = timm.create_model(
        "regnety_002", pretrained=True, num_classes=0, global_pool="avg"
    ).eval().to(device)
    prepared = skipped = 0
    for index, timeline in enumerate(timelines.values(), 1):
        cache_path = cache_dir / f"{timeline['match_id'].replace('/', '__')}.npz"
        if not cache_path.is_file() and not timeline["video_path"].is_file():
            skipped += 1
            continue
        extract_video_embeddings(
            timeline["video_path"],
            cache_path,
            encoder,
            torch,
            device,
            config,
            encoder_batch_size,
        )
        prepared += 1
        print(f"Preparadas {prepared}; revisadas {index}/{len(timelines)}", flush=True)
    print(f"Mitades preparadas: {prepared}; pendientes: {skipped}", flush=True)
    return prepared, skipped


def main() -> None:
    parser = argparse.ArgumentParser(description="Extract available SoccerNet embeddings")
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--cache-dir", type=Path, required=True)
    parser.add_argument("--device", default="auto", choices=["auto", "cpu", "mps", "cuda"])
    parser.add_argument("--encoder-batch-size", type=int, default=64)
    arguments = parser.parse_args()
    prepare_available(
        arguments.dataset.resolve(),
        arguments.cache_dir.resolve(),
        arguments.device,
        arguments.encoder_batch_size,
    )


if __name__ == "__main__":
    main()
