import argparse
import json
import math
import re
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import cv2
import numpy as np

from futbol_video_analyst.database import Database


@dataclass(frozen=True)
class TrainingExample:
    clip_path: Path
    event_id: str
    label: int
    match_id: str
    match_title: str
    peak_in_clip: float


def load_examples(dataset: Path, task: str = "corner") -> list[TrainingExample]:
    manifest_path = dataset / "manifest.jsonl"
    if not manifest_path.is_file():
        raise ValueError(f"No se encontró {manifest_path}")

    examples: list[TrainingExample] = []
    for line_number, line in enumerate(manifest_path.read_text(encoding="utf-8").splitlines(), 1):
        record = json.loads(line)
        labels = record.get("labels", [record["label"]])
        clip_path = dataset / record["clip_path"]
        if not clip_path.is_file():
            raise ValueError(f"Falta el clip de la línea {line_number}: {clip_path}")
        examples.append(
            TrainingExample(
                clip_path=clip_path,
                event_id=record["event_id"],
                label=1 if task in labels else 0,
                match_id=record["match_id"],
                match_title=record["match_title"],
                peak_in_clip=float(record["peak_seconds"]) - float(record["start_seconds"]),
            )
        )
    if not examples:
        raise ValueError("El dataset no contiene etiquetas")
    return examples


def load_background_examples(
    dataset: Path,
    database_path: Path,
    interval_seconds: float,
    exclusion_seconds: float = 16.0,
) -> list[TrainingExample]:
    """Sample ordinary full-match windows so a clip classifier can learn background play."""
    if interval_seconds <= 0:
        raise ValueError("El intervalo de fondo debe ser mayor que cero")
    records = [
        json.loads(line)
        for line in (dataset / "manifest.jsonl").read_text(encoding="utf-8").splitlines()
    ]
    match_ids = {record["match_id"] for record in records}
    corner_peaks: dict[str, list[float]] = {match_id: [] for match_id in match_ids}
    for record in records:
        if record["label"] == "corner":
            corner_peaks[record["match_id"]].append(float(record["peak_seconds"]))

    database = Database(database_path)
    examples: list[TrainingExample] = []
    for match_id in sorted(match_ids):
        match = database.get_match(match_id)
        if match is None:
            # Exported event clips remain valid training data even if the user
            # deletes the original match after preparing the dataset. We only
            # lose the ability to sample extra background windows for it.
            continue
        video_path = Path(match.video_path)
        if not video_path.is_file():
            continue
        timestamp = interval_seconds / 2
        while timestamp <= match.duration_seconds - 2:
            if all(
                abs(timestamp - corner_peak) > exclusion_seconds
                for corner_peak in corner_peaks[match_id]
            ):
                examples.append(
                    TrainingExample(
                        clip_path=video_path,
                        event_id=f"background-{match_id}-{int(timestamp * 1000)}",
                        label=0,
                        match_id=match_id,
                        match_title=match.title,
                        peak_in_clip=timestamp,
                    )
                )
            timestamp += interval_seconds
    return examples


def choose_validation_match(examples: list[TrainingExample], requested: str | None) -> str:
    by_match: dict[str, list[TrainingExample]] = {}
    for example in examples:
        by_match.setdefault(example.match_id, []).append(example)
    if len(by_match) < 2:
        raise ValueError("Se necesitan al menos dos partidos para separar validación")

    if requested:
        matches = {
            match_id
            for match_id, items in by_match.items()
            if match_id == requested or items[0].match_title.casefold() == requested.casefold()
        }
        if len(matches) != 1:
            raise ValueError(f"No se encontró un único partido de validación: {requested}")
        return matches.pop()

    candidates: list[tuple[int, int, str]] = []
    for match_id, items in by_match.items():
        positives = sum(item.label for item in items)
        negatives = len(items) - positives
        if positives >= 2 and negatives >= 5:
            candidates.append((positives, -negatives, match_id))
    if not candidates:
        raise ValueError(
            "Ningún partido tiene al menos dos corners y cinco negativos para validación"
        )
    return min(candidates)[2]


def split_examples(
    examples: list[TrainingExample], validation_match_id: str
) -> tuple[list[TrainingExample], list[TrainingExample]]:
    training = [example for example in examples if example.match_id != validation_match_id]
    validation = [example for example in examples if example.match_id == validation_match_id]
    if sum(example.label for example in training) < 2:
        raise ValueError("Quedaron menos de dos corners para entrenamiento")
    if not {example.label for example in validation} == {0, 1}:
        raise ValueError("Validación necesita ejemplos corner y negative")
    return training, validation


def balance_examples(
    examples: list[TrainingExample], negative_ratio: int | None
) -> list[TrainingExample]:
    if negative_ratio is None:
        return examples
    if negative_ratio < 1:
        raise ValueError("La proporción de negativos debe ser al menos uno")
    by_match: dict[str, list[TrainingExample]] = {}
    for example in examples:
        by_match.setdefault(example.match_id, []).append(example)
    balanced: list[TrainingExample] = []
    for items in by_match.values():
        positives = [item for item in items if item.label == 1]
        negatives = [item for item in items if item.label == 0]
        wanted = min(len(negatives), len(positives) * negative_ratio)
        if wanted and wanted < len(negatives):
            indices = np.linspace(0, len(negatives) - 1, wanted, dtype=int)
            negatives = [negatives[index] for index in indices]
        balanced.extend(positives + negatives)
    return balanced


def _sample_clip(example: TrainingExample, frames: int = 16, window_seconds: float = 4) -> np.ndarray:
    capture = cv2.VideoCapture(str(example.clip_path))
    if not capture.isOpened():
        raise ValueError(f"No se pudo abrir {example.clip_path}")
    fps = capture.get(cv2.CAP_PROP_FPS) or 25
    frame_count = capture.get(cv2.CAP_PROP_FRAME_COUNT)
    duration = frame_count / fps if frame_count > 0 else example.peak_in_clip + window_seconds
    half_window = window_seconds / 2
    start = max(0, min(example.peak_in_clip - half_window, max(0, duration - window_seconds)))
    # Container metadata can point just beyond the final decodable frame.
    end_margin = max(1 / fps, 0.1)
    end = min(max(0, duration - end_margin), start + window_seconds)
    timestamps = np.linspace(start, end, frames)
    sampled: list[np.ndarray] = []
    try:
        capture.set(cv2.CAP_PROP_POS_FRAMES, max(0, int(start * fps)))
        current_frame = max(0, int(capture.get(cv2.CAP_PROP_POS_FRAMES)))
        for timestamp in timestamps:
            target_frame = max(current_frame, int(float(timestamp) * fps))
            success = True
            while current_frame <= target_frame:
                success = capture.grab()
                if not success:
                    break
                current_frame += 1
            success, frame = capture.retrieve() if success else (False, None)
            if not success:
                raise ValueError(f"No se pudo leer {example.clip_path} en {timestamp:.2f}s")
            sampled.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    finally:
        capture.release()
    return np.stack(sampled)


def _ml() -> tuple[Any, Any, Any]:
    try:
        import torch
        from torchvision.models.video import R3D_18_Weights, r3d_18
    except ImportError as error:
        raise RuntimeError("Instala el extra de entrenamiento: pip install -e '.[training]'") from error
    return torch, R3D_18_Weights, r3d_18


def _device(torch: Any, requested: str) -> Any:
    if requested != "auto":
        return torch.device(requested)
    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def extract_embeddings(
    examples: list[TrainingExample], cache_path: Path, requested_device: str
) -> np.ndarray:
    torch, weights_type, model_factory = _ml()
    ids = np.array([example.event_id for example in examples])
    cached_by_id: dict[str, np.ndarray] = {}
    if cache_path.is_file():
        cached = np.load(cache_path)
        cached_by_id = {
            str(event_id): embedding
            for event_id, embedding in zip(cached["event_ids"], cached["embeddings"])
        }
        if all(event_id in cached_by_id for event_id in ids):
            print(f"Usando características guardadas en {cache_path}", flush=True)
            return np.stack([cached_by_id[event_id] for event_id in ids])

    missing_examples = [example for example in examples if example.event_id not in cached_by_id]

    device = _device(torch, requested_device)
    print(
        f"Reutilizando {len(examples) - len(missing_examples)} características; "
        f"extrayendo {len(missing_examples)} con R3D-18 en {device}",
        flush=True,
    )
    weights = weights_type.DEFAULT
    model = model_factory(weights=weights)
    model.fc = torch.nn.Identity()
    model.eval().to(device)
    transform = weights.transforms()
    embeddings: list[np.ndarray] = []
    batch: list[Any] = []
    batch_size = 4 if device.type != "cpu" else 2

    def process_batch() -> None:
        if not batch:
            return
        inputs = torch.stack(batch).to(device)
        with torch.inference_mode():
            result = model(inputs).detach().cpu().numpy()
        embeddings.extend(result)
        batch.clear()

    for index, example in enumerate(missing_examples, 1):
        frames = _sample_clip(example)
        tensor = torch.from_numpy(frames.copy()).permute(0, 3, 1, 2)
        batch.append(transform(tensor))
        if len(batch) == batch_size:
            process_batch()
        print(f"Características {index}/{len(missing_examples)}", end="\r", flush=True)
    process_batch()
    print()
    cached_by_id.update(
        {
            example.event_id: embedding
            for example, embedding in zip(missing_examples, embeddings)
        }
    )
    matrix = np.stack([cached_by_id[event_id] for event_id in ids]).astype(np.float32)
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(cache_path, event_ids=ids, embeddings=matrix)
    return matrix


def _metrics(probabilities: np.ndarray, labels: np.ndarray, threshold: float = 0.5) -> dict[str, Any]:
    predictions = probabilities >= threshold
    truth = labels == 1
    true_positive = int(np.logical_and(predictions, truth).sum())
    false_positive = int(np.logical_and(predictions, ~truth).sum())
    true_negative = int(np.logical_and(~predictions, ~truth).sum())
    false_negative = int(np.logical_and(~predictions, truth).sum())
    precision = true_positive / (true_positive + false_positive) if true_positive + false_positive else 0
    recall = true_positive / (true_positive + false_negative) if true_positive + false_negative else 0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0
    accuracy = (true_positive + true_negative) / len(labels)
    return {
        "threshold": threshold,
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "true_positive": true_positive,
        "false_positive": false_positive,
        "true_negative": true_negative,
        "false_negative": false_negative,
    }


def train_head(
    embeddings: np.ndarray,
    examples: list[TrainingExample],
    training_examples: list[TrainingExample],
    validation_examples: list[TrainingExample],
    requested_device: str,
    task: str = "corner",
) -> tuple[dict[str, Any], dict[str, Any]]:
    torch, _, _ = _ml()
    device = _device(torch, requested_device)
    index_by_id = {example.event_id: index for index, example in enumerate(examples)}
    train_indices = [index_by_id[example.event_id] for example in training_examples]
    validation_indices = [index_by_id[example.event_id] for example in validation_examples]
    x_train = embeddings[train_indices]
    y_train = np.array([example.label for example in training_examples], dtype=np.float32)
    x_validation = embeddings[validation_indices]
    y_validation = np.array([example.label for example in validation_examples], dtype=np.float32)

    mean = x_train.mean(axis=0)
    standard_deviation = x_train.std(axis=0)
    standard_deviation[standard_deviation < 1e-6] = 1
    x_train = (x_train - mean) / standard_deviation
    x_validation = (x_validation - mean) / standard_deviation

    torch.manual_seed(7)
    head = torch.nn.Linear(x_train.shape[1], 1).to(device)
    positives = float(y_train.sum())
    negatives = float(len(y_train) - positives)
    loss_function = torch.nn.BCEWithLogitsLoss(
        pos_weight=torch.tensor([negatives / positives], device=device)
    )
    optimizer = torch.optim.AdamW(head.parameters(), lr=0.01, weight_decay=0.05)
    train_x = torch.from_numpy(x_train).to(device)
    train_y = torch.from_numpy(y_train).to(device).unsqueeze(1)
    validation_x = torch.from_numpy(x_validation).to(device)
    validation_y = torch.from_numpy(y_validation).to(device).unsqueeze(1)
    best_loss = math.inf
    best_state: dict[str, Any] | None = None
    patience = 60
    remaining_patience = patience

    for _ in range(500):
        head.train()
        optimizer.zero_grad()
        loss = loss_function(head(train_x), train_y)
        loss.backward()
        optimizer.step()
        head.eval()
        with torch.inference_mode():
            validation_loss = float(loss_function(head(validation_x), validation_y).item())
        if validation_loss < best_loss - 1e-5:
            best_loss = validation_loss
            best_state = {key: value.detach().cpu().clone() for key, value in head.state_dict().items()}
            remaining_patience = patience
        else:
            remaining_patience -= 1
            if remaining_patience == 0:
                break

    assert best_state is not None
    head.load_state_dict(best_state)
    head.eval()
    with torch.inference_mode():
        validation_probabilities = torch.sigmoid(head(validation_x)).cpu().numpy().reshape(-1)
        training_probabilities = torch.sigmoid(head(train_x)).cpu().numpy().reshape(-1)

    threshold = 0.5
    validation_metrics = _metrics(validation_probabilities, y_validation, threshold)
    best_rank = (
        validation_metrics["f1"],
        validation_metrics["recall"],
        validation_metrics["precision"],
    )
    for candidate in np.arange(0.525, 0.951, 0.025):
        candidate_metrics = _metrics(
            validation_probabilities, y_validation, float(candidate)
        )
        candidate_rank = (
            candidate_metrics["f1"],
            candidate_metrics["recall"],
            candidate_metrics["precision"],
        )
        if candidate_rank > best_rank:
            threshold = float(candidate)
            validation_metrics = candidate_metrics
            best_rank = candidate_rank

    report = {
        "training": _metrics(training_probabilities, y_train, threshold),
        "validation": validation_metrics,
        "validation_loss": best_loss,
    }
    checkpoint = {
        "backbone": "torchvision/r3d_18/KINETICS400_V1",
        "head_state_dict": best_state,
        "feature_mean": torch.from_numpy(mean),
        "feature_standard_deviation": torch.from_numpy(standard_deviation),
        "label_map": {"negative": 0, task: 1},
        "threshold": threshold,
        "sampling": {"frames": 16, "window_seconds": 4},
    }
    return checkpoint, report


def _next_model_path(models_dir: Path, task: str = "corner") -> Path:
    prefix = task.replace("_", "-")
    versions = []
    for path in models_dir.glob(f"{prefix}-spotter-v*.pt"):
        match = re.search(r"v(\d+)\.pt$", path.name)
        if match:
            versions.append(int(match.group(1)))
    return models_dir / f"{prefix}-spotter-v{max(versions, default=0) + 1:03d}.pt"


def run_training(
    dataset: Path,
    models_dir: Path,
    validation_match: str | None,
    device: str,
    background_interval: float | None = None,
    database_path: Path = Path("data/futbol-video-analyst.sqlite3"),
    task: str = "corner",
    negative_ratio: int | None = None,
    cache_path: Path | None = None,
) -> Path:
    examples = load_examples(dataset, task)
    background_examples: list[TrainingExample] = []
    if background_interval is not None:
        if task != "corner":
            raise ValueError("El muestreo de fondo automático todavía solo admite corner")
        background_examples = load_background_examples(
            dataset, database_path, background_interval
        )
        examples.extend(background_examples)
    examples = balance_examples(examples, negative_ratio)
    validation_match_id = choose_validation_match(examples, validation_match)
    training_examples, validation_examples = split_examples(examples, validation_match_id)
    validation_title = validation_examples[0].match_title
    print(
        f"Entrenamiento: {len(training_examples)} clips de "
        f"{len({item.match_id for item in training_examples})} partidos; "
        f"validación: {len(validation_examples)} clips de {validation_title}",
        flush=True,
    )
    embedding_cache = cache_path or (
        Path("data/training_cache") / dataset.name / "r3d18-kinetics400.npz"
    )
    embeddings = extract_embeddings(examples, embedding_cache, device)
    checkpoint, report = train_head(
        embeddings, examples, training_examples, validation_examples, device, task
    )
    models_dir.mkdir(parents=True, exist_ok=True)
    model_path = _next_model_path(models_dir, task)
    torch, _, _ = _ml()
    checkpoint["dataset"] = str(dataset.resolve())
    checkpoint["validation_match_id"] = validation_match_id
    checkpoint["metrics"] = report
    torch.save(checkpoint, model_path)
    metadata = {
        "model": model_path.name,
        "task": task,
        "created_at": datetime.now(UTC).isoformat(),
        "dataset": str(dataset.resolve()),
        "training_matches": sorted({item.match_title for item in training_examples}),
        "validation_match": validation_title,
        "training_clips": len(training_examples),
        "validation_clips": len(validation_examples),
        "background_clips": len(background_examples),
        "negative_ratio": negative_ratio,
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
    parser = argparse.ArgumentParser(description="Entrena un clasificador local de corners")
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--task", default="corner", choices=["corner", "free_kick", "shot_attempt"])
    parser.add_argument("--models-dir", type=Path, default=Path("models"))
    parser.add_argument("--validation-match")
    parser.add_argument("--device", default="auto", choices=["auto", "cpu", "mps", "cuda"])
    parser.add_argument("--background-interval", type=float)
    parser.add_argument("--negative-ratio", type=int)
    parser.add_argument("--cache", type=Path)
    parser.add_argument("--database", type=Path, default=Path("data/futbol-video-analyst.sqlite3"))
    arguments = parser.parse_args()
    run_training(
        arguments.dataset.resolve(),
        arguments.models_dir,
        arguments.validation_match,
        arguments.device,
        arguments.background_interval,
        arguments.database,
        arguments.task,
        arguments.negative_ratio,
        arguments.cache,
    )


if __name__ == "__main__":
    main()
