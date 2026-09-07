import argparse
import json
import re
from datetime import UTC, datetime
from pathlib import Path

from futbol_video_analyst.training import (
    TrainingExample,
    _ml,
    choose_validation_match,
    extract_embeddings,
    load_background_examples,
    load_examples,
    split_examples,
    train_head,
)

CONTEXT_SECONDS = (-8.0, -4.0, 0.0, 4.0, 8.0)
CONTEXT_STEPS = (-2, -1, 0, 1, 2)


def expand_context(examples: list[TrainingExample]) -> list[TrainingExample]:
    return [
        TrainingExample(
            clip_path=example.clip_path,
            event_id=f"{example.event_id}@{offset:+g}",
            label=example.label,
            match_id=example.match_id,
            match_title=example.match_title,
            peak_in_clip=max(0, example.peak_in_clip + offset),
        )
        for example in examples
        for offset in CONTEXT_SECONDS
    ]


def _next_path(models_dir: Path) -> Path:
    versions = []
    for path in models_dir.glob("corner-action-v*.pt"):
        match = re.search(r"v(\d+)\.pt$", path.name)
        if match:
            versions.append(int(match.group(1)))
    return models_dir / f"corner-action-v{max(versions, default=0) + 1:03d}.pt"


def run_training(
    dataset: Path,
    database: Path,
    models_dir: Path,
    validation_match: str | None,
    background_interval: float,
    device: str,
) -> Path:
    examples = load_examples(dataset)
    backgrounds = load_background_examples(dataset, database, background_interval)
    examples.extend(backgrounds)
    validation_id = choose_validation_match(examples, validation_match)
    training, validation = split_examples(examples, validation_id)
    expanded = expand_context(examples)
    cache = Path("data/training_cache") / dataset.name / "r3d18-action-context.npz"
    window_embeddings = extract_embeddings(expanded, cache, device)
    sequence_embeddings = window_embeddings.reshape(len(examples), len(CONTEXT_SECONDS), -1)
    sequence_embeddings = sequence_embeddings.reshape(len(examples), -1)
    checkpoint, report = train_head(
        sequence_embeddings, examples, training, validation, device
    )
    checkpoint.update(
        {
            "dataset": str(dataset.resolve()),
            "validation_match_id": validation_id,
            "metrics": report,
            "temporal_context_seconds": CONTEXT_SECONDS,
            "temporal_context_steps": CONTEXT_STEPS,
        }
    )
    models_dir.mkdir(parents=True, exist_ok=True)
    model_path = _next_path(models_dir)
    torch, _, _ = _ml()
    torch.save(checkpoint, model_path)
    metadata = {
        "model": model_path.name,
        "created_at": datetime.now(UTC).isoformat(),
        "dataset": str(dataset.resolve()),
        "training_matches": sorted({item.match_title for item in training}),
        "validation_match": validation[0].match_title,
        "training_sequences": len(training),
        "validation_sequences": len(validation),
        "background_sequences": len(backgrounds),
        "context_seconds": CONTEXT_SECONDS,
        "metrics": report,
        "experimental": True,
        "activated_in_app": False,
    }
    model_path.with_suffix(".json").write_text(
        json.dumps(metadata, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    print(json.dumps(metadata, indent=2, ensure_ascii=False))
    return model_path


def main() -> None:
    parser = argparse.ArgumentParser(description="Entrena action spotting temporal con R3D-18")
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--database", type=Path, default=Path("data/futbol-video-analyst.sqlite3"))
    parser.add_argument("--models-dir", type=Path, default=Path("models"))
    parser.add_argument("--validation-match")
    parser.add_argument("--background-interval", type=float, default=60)
    parser.add_argument("--device", default="auto", choices=["auto", "cpu", "mps", "cuda"])
    arguments = parser.parse_args()
    run_training(
        arguments.dataset.resolve(),
        arguments.database,
        arguments.models_dir,
        arguments.validation_match,
        arguments.background_interval,
        arguments.device,
    )


if __name__ == "__main__":
    main()
