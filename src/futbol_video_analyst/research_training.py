import argparse
import json
import math
import re
from collections import defaultdict
from dataclasses import asdict
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import cv2
import numpy as np

from futbol_video_analyst.action_spotting import (
    ActionSpottingConfig,
    extract_spots,
)
from futbol_video_analyst.spotting_evaluation import TimelineEvent, evaluate
from futbol_video_analyst.training import _device


def _ml() -> tuple[Any, Any]:
    try:
        import timm
        import torch
    except ImportError as error:
        raise RuntimeError("Instala el extra de entrenamiento: pip install -e '.[training]'") from error
    return torch, timm


def _next_model_path(models_dir: Path) -> Path:
    versions: list[int] = []
    for path in models_dir.glob("soccernet-research-temporal-v*.pt"):
        match = re.search(r"v(\d+)\.pt$", path.name)
        if match:
            versions.append(int(match.group(1)))
    return models_dir / f"soccernet-research-temporal-v{max(versions, default=0) + 1:03d}.pt"


def load_manifest(
    manifest_path: Path, labels: tuple[str, ...]
) -> dict[str, dict[str, Any]]:
    timelines: dict[str, dict[str, Any]] = {}
    for line in manifest_path.read_text(encoding="utf-8").splitlines():
        record = json.loads(line)
        match_id = str(record["match_id"])
        clip_path = record.get("clip_path")
        if not clip_path:
            raise ValueError(f"Missing video for {match_id}")
        timeline = timelines.setdefault(
            match_id,
            {
                "match_id": match_id,
                "match_title": record["match_title"],
                "video_path": Path(clip_path),
                "split": record.get("split"),
                "events": defaultdict(list),
            },
        )
        if timeline["video_path"] != Path(clip_path):
            raise ValueError(f"Multiple videos found for {match_id}")
        if record["label"] in labels:
            timeline["events"][record["label"]].append(float(record["peak_seconds"]))
    if not timelines:
        raise ValueError("The manifest contains no timelines")
    unsupported = {timeline["split"] for timeline in timelines.values()} - {"train", "valid"}
    if unsupported:
        raise ValueError(f"Unsupported or missing splits: {sorted(unsupported, key=str)}")
    return timelines


def _letterbox(frame: np.ndarray, size: int) -> np.ndarray:
    height, width = frame.shape[:2]
    scale = min(size / width, size / height)
    resized = cv2.resize(frame, (round(width * scale), round(height * scale)))
    canvas = np.zeros((size, size, 3), dtype=np.uint8)
    y = (size - resized.shape[0]) // 2
    x = (size - resized.shape[1]) // 2
    canvas[y : y + resized.shape[0], x : x + resized.shape[1]] = resized
    return canvas


def _next_sample_time(current: float, timestamp: float, interval: float) -> float:
    """Advance a fixed sampling grid, snapping only across a true PTS gap."""
    advanced = current + interval
    if advanced < timestamp - interval:
        return timestamp + interval
    return advanced


def extract_video_embeddings(
    video_path: Path,
    cache_path: Path,
    encoder: Any,
    torch: Any,
    device: Any,
    config: ActionSpottingConfig,
    batch_size: int,
) -> tuple[np.ndarray, np.ndarray]:
    if cache_path.is_file():
        cached = np.load(cache_path)
        cache_timebase = str(cached["timebase"]) if "timebase" in cached else "frame_index"
        if cache_timebase != "presentation_timestamp":
            raise ValueError(f"Obsolete frame-index cache: {cache_path}")
        return cached["embeddings"].astype(np.float32), cached["timestamps"]

    capture = cv2.VideoCapture(str(video_path))
    if not capture.isOpened():
        raise ValueError(f"Could not open {video_path}")
    source_fps = capture.get(cv2.CAP_PROP_FPS) or 25.0
    frame_count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
    duration = frame_count / source_fps
    expected_samples = max(1, math.floor(duration * config.sample_fps))
    mean = torch.tensor([0.485, 0.456, 0.406], device=device).view(1, 3, 1, 1)
    standard_deviation = torch.tensor([0.229, 0.224, 0.225], device=device).view(1, 3, 1, 1)
    frames: list[np.ndarray] = []
    timestamps: list[float] = []
    embeddings: list[np.ndarray] = []

    def process_batch() -> None:
        if not frames:
            return
        values = np.stack(frames)
        tensor = torch.from_numpy(values).to(device=device, dtype=torch.float32)
        tensor = tensor.permute(0, 3, 1, 2) / 255.0
        tensor = (tensor - mean) / standard_deviation
        with torch.inference_mode():
            output = encoder(tensor).detach().cpu().numpy().astype(np.float16)
        embeddings.extend(output)
        frames.clear()

    try:
        next_sample = 0.0
        frame_index = 0
        while True:
            success = capture.grab()
            if not success:
                break
            timestamp = capture.get(cv2.CAP_PROP_POS_MSEC) / 1000
            if timestamp <= 0 and frame_index:
                timestamp = frame_index / source_fps
            frame_index += 1
            if timestamp + 1e-6 < next_sample:
                continue
            success, frame = capture.retrieve()
            if not success:
                continue
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frames.append(_letterbox(frame, config.input_size))
            timestamps.append(timestamp)
            next_sample = _next_sample_time(
                next_sample, timestamp, 1 / config.sample_fps
            )
            if len(frames) >= batch_size:
                process_batch()
        process_batch()
    finally:
        capture.release()
    if not embeddings:
        raise ValueError(f"No frames decoded from {video_path}")
    matrix = np.asarray(embeddings, dtype=np.float16)
    time_values = np.asarray(timestamps[: len(matrix)], dtype=np.float32)
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        cache_path,
        embeddings=matrix,
        timestamps=time_values,
        timebase=np.asarray("presentation_timestamp"),
        source_duration_seconds=np.asarray(duration),
    )
    print(
        f"{video_path.name}: {len(matrix)}/{expected_samples} frames -> {cache_path.name}",
        flush=True,
    )
    return matrix.astype(np.float32), time_values


def build_targets(
    timestamps: np.ndarray,
    events: dict[str, list[float]],
    labels: tuple[str, ...],
    positive_radius_seconds: float = 1.0,
) -> np.ndarray:
    targets = np.zeros((len(timestamps), len(labels)), dtype=np.float32)
    for class_index, label in enumerate(labels):
        for event_time in events.get(label, []):
            targets[np.abs(timestamps - event_time) <= positive_radius_seconds, class_index] = 1
    return targets


def window_starts(length: int, window: int, stride: int) -> list[int]:
    if length <= window:
        return [0]
    starts = list(range(0, length - window + 1, stride))
    final = length - window
    if starts[-1] != final:
        starts.append(final)
    return starts


def build_temporal_head(
    feature_size: int,
    hidden_size: int,
    classes: int,
    architecture: str = "bigru",
) -> Any:
    torch, _ = _ml()

    class TemporalHead(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.normalize = torch.nn.LayerNorm(feature_size)
            self.temporal = torch.nn.GRU(
                feature_size,
                hidden_size,
                batch_first=True,
                bidirectional=True,
            )
            self.classifier = torch.nn.Linear(hidden_size * 2, classes)

        def forward(self, features: Any) -> Any:
            values, _ = self.temporal(self.normalize(features))
            return self.classifier(values)

    class ResidualTemporalBlock(torch.nn.Module):
        def __init__(self, dilation: int) -> None:
            super().__init__()
            self.depthwise = torch.nn.Conv1d(
                hidden_size,
                hidden_size,
                kernel_size=3,
                padding=dilation,
                dilation=dilation,
                groups=hidden_size,
            )
            self.pointwise = torch.nn.Conv1d(hidden_size, hidden_size, kernel_size=1)
            self.normalize = torch.nn.GroupNorm(8, hidden_size)
            self.activation = torch.nn.GELU()
            self.dropout = torch.nn.Dropout(0.15)

        def forward(self, values: Any) -> Any:
            residual = values
            values = self.depthwise(values)
            values = self.pointwise(values)
            values = self.normalize(values)
            values = self.activation(values)
            return residual + self.dropout(values)

    class TemporalConvolutionHead(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.normalize = torch.nn.LayerNorm(feature_size)
            self.projection = torch.nn.Linear(feature_size, hidden_size)
            self.temporal = torch.nn.Sequential(
                *(ResidualTemporalBlock(dilation) for dilation in (1, 2, 4, 8, 16))
            )
            self.output_normalize = torch.nn.LayerNorm(hidden_size)
            self.classifier = torch.nn.Linear(hidden_size, classes)

        def forward(self, features: Any) -> Any:
            values = self.projection(self.normalize(features)).transpose(1, 2)
            values = self.temporal(values).transpose(1, 2)
            return self.classifier(self.output_normalize(values))

    if architecture == "bigru":
        return TemporalHead()
    if architecture == "tcn":
        return TemporalConvolutionHead()
    raise ValueError(f"Unknown temporal architecture: {architecture}")


def _windows(
    timelines: list[dict[str, Any]], config: ActionSpottingConfig
) -> list[tuple[str, int]]:
    return [
        (timeline["match_id"], start)
        for timeline in timelines
        for start in window_starts(
            len(timeline["embeddings"]), config.window_frames, config.stride_frames
        )
    ]


def _batch(
    items: list[tuple[str, int]],
    timelines: dict[str, dict[str, Any]],
    config: ActionSpottingConfig,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    feature_size = next(iter(timelines.values()))["embeddings"].shape[1]
    x = np.zeros((len(items), config.window_frames, feature_size), dtype=np.float32)
    y = np.zeros((len(items), config.window_frames, len(config.labels)), dtype=np.float32)
    mask = np.zeros((len(items), config.window_frames, 1), dtype=np.float32)
    for index, (match_id, start) in enumerate(items):
        timeline = timelines[match_id]
        end = min(start + config.window_frames, len(timeline["embeddings"]))
        length = end - start
        x[index, :length] = timeline["embeddings"][start:end]
        y[index, :length] = timeline["targets"][start:end]
        mask[index, :length] = 1
    return x, y, mask


def predict_timelines(
    model: Any,
    timelines: list[dict[str, Any]],
    config: ActionSpottingConfig,
    torch: Any,
    device: Any,
    batch_size: int,
) -> dict[str, np.ndarray]:
    by_id = {timeline["match_id"]: timeline for timeline in timelines}
    totals = {
        timeline["match_id"]: np.zeros(
            (len(timeline["embeddings"]), len(config.labels)), dtype=np.float32
        )
        for timeline in timelines
    }
    counts = {
        timeline["match_id"]: np.zeros((len(timeline["embeddings"]), 1), dtype=np.float32)
        for timeline in timelines
    }
    items = _windows(timelines, config)
    model.eval()
    for offset in range(0, len(items), batch_size):
        current = items[offset : offset + batch_size]
        features, _, masks = _batch(current, by_id, config)
        values = torch.from_numpy(features).to(device)
        with torch.inference_mode():
            probabilities = torch.sigmoid(model(values)).cpu().numpy()
        for index, (match_id, start) in enumerate(current):
            length = int(masks[index].sum())
            totals[match_id][start : start + length] += probabilities[index, :length]
            counts[match_id][start : start + length] += 1
    return {match_id: totals[match_id] / np.maximum(counts[match_id], 1) for match_id in totals}


def calibrate_thresholds(
    probabilities: dict[str, np.ndarray],
    timelines: list[dict[str, Any]],
    config: ActionSpottingConfig,
) -> tuple[dict[str, float], dict[str, Any]]:
    truth = [
        TimelineEvent(timeline["match_id"], label, timestamp)
        for timeline in timelines
        for label in config.labels
        for timestamp in timeline["events"].get(label, [])
    ]
    best_thresholds: dict[str, float] = {}
    class_reports: dict[str, Any] = {}
    by_id = {timeline["match_id"]: timeline for timeline in timelines}
    for label in config.labels:
        best_rank = (-1.0, -1.0, -1.0)
        best_report: dict[str, Any] | None = None
        best_threshold = 0.5
        for threshold in np.arange(0.1, 0.951, 0.05):
            thresholds = {item: 1.1 for item in config.labels}
            thresholds[label] = float(threshold)
            predicted = [
                TimelineEvent(
                    match_id,
                    spot.event_type,
                    spot.timestamp_seconds,
                    spot.confidence,
                )
                for match_id, scores in probabilities.items()
                for spot in extract_spots(
                    scores,
                    by_id[match_id]["timestamps"],
                    config,
                    thresholds,
                )
                if spot.event_type == label
            ]
            report = evaluate(
                [event for event in truth if event.event_type == label],
                predicted,
                tolerance_seconds=2.0,
            )["classes"][label]
            rank = (report["f1"], report["recall"], report["precision"])
            if rank > best_rank:
                best_rank = rank
                best_report = report
                best_threshold = float(threshold)
        best_thresholds[label] = best_threshold
        class_reports[label] = best_report
    return best_thresholds, class_reports


def run_training(
    dataset: Path,
    models_dir: Path,
    cache_dir: Path,
    requested_device: str,
    epochs: int,
    batch_size: int,
    encoder_batch_size: int,
    architecture: str = "bigru",
) -> Path:
    config = ActionSpottingConfig(labels=("corner", "shot_attempt"))
    timelines = load_manifest(dataset / "manifest.jsonl", config.labels)
    torch, timm = _ml()
    device = _device(torch, requested_device)
    print(f"Research training device: {device}", flush=True)
    encoder = timm.create_model(
        "regnety_002",
        pretrained=True,
        num_classes=0,
        global_pool="avg",
    ).eval().to(device)
    for parameter in encoder.parameters():
        parameter.requires_grad = False

    for index, timeline in enumerate(timelines.values(), 1):
        cache_path = cache_dir / f"{timeline['match_id'].replace('/', '__')}.npz"
        embeddings, timestamps = extract_video_embeddings(
            timeline["video_path"],
            cache_path,
            encoder,
            torch,
            device,
            config,
            encoder_batch_size,
        )
        timeline["embeddings"] = embeddings
        timeline["timestamps"] = timestamps
        timeline["targets"] = build_targets(
            timestamps, timeline["events"], config.labels
        )
        print(f"Timelines {index}/{len(timelines)}", flush=True)

    training = [timeline for timeline in timelines.values() if timeline["split"] == "train"]
    validation = [timeline for timeline in timelines.values() if timeline["split"] == "valid"]
    train_by_id = {timeline["match_id"]: timeline for timeline in training}
    train_windows = _windows(training, config)
    validation_by_id = {timeline["match_id"]: timeline for timeline in validation}
    validation_windows = _windows(validation, config)
    feature_size = training[0]["embeddings"].shape[1]
    model = build_temporal_head(
        feature_size, 128, len(config.labels), architecture=architecture
    ).to(device)

    target_values = np.concatenate([timeline["targets"] for timeline in training], axis=0)
    positives = target_values.sum(axis=0)
    negatives = len(target_values) - positives
    positive_weight = np.minimum(negatives / np.maximum(positives, 1), 100)
    loss_function = torch.nn.BCEWithLogitsLoss(
        pos_weight=torch.from_numpy(positive_weight.astype(np.float32)).to(device),
        reduction="none",
    )
    learning_rate = 5e-4 if architecture == "tcn" else 1e-3
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=learning_rate, weight_decay=0.01
    )
    random = np.random.default_rng(7)
    best_loss = math.inf
    best_state: dict[str, Any] | None = None
    patience = 4
    remaining_patience = patience
    history: list[dict[str, float]] = []

    for epoch in range(1, epochs + 1):
        random.shuffle(train_windows)
        model.train()
        training_loss = 0.0
        training_batches = 0
        for offset in range(0, len(train_windows), batch_size):
            features, targets, masks = _batch(
                train_windows[offset : offset + batch_size], train_by_id, config
            )
            x = torch.from_numpy(features).to(device)
            y = torch.from_numpy(targets).to(device)
            mask = torch.from_numpy(masks).to(device)
            optimizer.zero_grad()
            loss_values = loss_function(model(x), y) * mask
            loss = loss_values.sum() / (mask.sum() * len(config.labels))
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            optimizer.step()
            training_loss += float(loss.item())
            training_batches += 1

        model.eval()
        validation_loss = 0.0
        validation_batches = 0
        with torch.inference_mode():
            for offset in range(0, len(validation_windows), batch_size):
                features, targets, masks = _batch(
                    validation_windows[offset : offset + batch_size], validation_by_id, config
                )
                x = torch.from_numpy(features).to(device)
                y = torch.from_numpy(targets).to(device)
                mask = torch.from_numpy(masks).to(device)
                values = loss_function(model(x), y) * mask
                loss = values.sum() / (mask.sum() * len(config.labels))
                validation_loss += float(loss.item())
                validation_batches += 1
        training_loss /= max(training_batches, 1)
        validation_loss /= max(validation_batches, 1)
        history.append({"epoch": epoch, "training_loss": training_loss, "validation_loss": validation_loss})
        print(
            f"Epoch {epoch}/{epochs}: train={training_loss:.5f} valid={validation_loss:.5f}",
            flush=True,
        )
        if validation_loss < best_loss - 1e-4:
            best_loss = validation_loss
            best_state = {
                key: value.detach().cpu().clone() for key, value in model.state_dict().items()
            }
            remaining_patience = patience
        else:
            remaining_patience -= 1
            if remaining_patience == 0:
                break

    if best_state is None:
        raise RuntimeError("Training did not produce a checkpoint")
    model.load_state_dict(best_state)
    validation_probabilities = predict_timelines(
        model, validation, config, torch, device, batch_size
    )
    thresholds, metrics = calibrate_thresholds(
        validation_probabilities, validation, config
    )

    models_dir.mkdir(parents=True, exist_ok=True)
    model_path = _next_model_path(models_dir)
    model_type = f"regnety_002_frozen_{architecture}_action_spotter"
    checkpoint = {
        "model_type": model_type,
        "temporal_architecture": architecture,
        "config": config.to_dict(),
        "feature_size": feature_size,
        "hidden_size": 128,
        "learning_rate": learning_rate,
        "temporal_head_state_dict": best_state,
        "thresholds": thresholds,
        "metrics": metrics,
        "lineage": {
            "commercial_status": "research_only",
            "training_sources": ["SoccerNet-v2", "ImageNet pretrained timm weights"],
            "commercial_model_eligible": False,
        },
    }
    torch.save(checkpoint, model_path)
    metadata = {
        "model": model_path.name,
        "created_at": datetime.now(UTC).isoformat(),
        "dataset": str(dataset.resolve()),
        "architecture": checkpoint["model_type"],
        "config": asdict(config),
        "training_halves": len(training),
        "validation_halves": len(validation),
        "training_windows": len(train_windows),
        "validation_windows": len(validation_windows),
        "best_validation_loss": best_loss,
        "thresholds": thresholds,
        "metrics_at_2_seconds": metrics,
        "history": history,
        "lineage": checkpoint["lineage"],
        "activated_in_app": False,
    }
    model_path.with_suffix(".json").write_text(
        json.dumps(metadata, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    print(json.dumps(metadata, indent=2, ensure_ascii=False), flush=True)
    return model_path


def main() -> None:
    parser = argparse.ArgumentParser(description="Train the SoccerNet research temporal model")
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--models-dir", type=Path, default=Path("models"))
    parser.add_argument(
        "--cache-dir",
        type=Path,
        default=Path("data/training_cache/soccernet-regnety002-2fps-pts-v3"),
    )
    parser.add_argument("--device", default="auto", choices=["auto", "cpu", "mps", "cuda"])
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--encoder-batch-size", type=int, default=64)
    parser.add_argument(
        "--architecture", choices=["bigru", "tcn"], default="bigru"
    )
    arguments = parser.parse_args()
    run_training(
        arguments.dataset.resolve(),
        arguments.models_dir,
        arguments.cache_dir,
        arguments.device,
        arguments.epochs,
        arguments.batch_size,
        arguments.encoder_batch_size,
        arguments.architecture,
    )


if __name__ == "__main__":
    main()
