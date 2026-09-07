from dataclasses import asdict, dataclass
from typing import Any

import numpy as np


@dataclass(frozen=True)
class ActionSpottingConfig:
    """Versioned input/output contract for the temporal spotting model."""

    sample_fps: float = 2.0
    window_frames: int = 128
    stride_frames: int = 64
    input_size: int = 224
    labels: tuple[str, ...] = ("corner", "goal_kick", "shot_attempt")

    @property
    def window_seconds(self) -> float:
        return self.window_frames / self.sample_fps

    @property
    def stride_seconds(self) -> float:
        return self.stride_frames / self.sample_fps

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class SpotPrediction:
    event_type: str
    timestamp_seconds: float
    confidence: float
    offset_seconds: float = 0.0


def extract_spots(
    probabilities: np.ndarray,
    timestamps: np.ndarray,
    config: ActionSpottingConfig,
    thresholds: dict[str, float],
    offsets: np.ndarray | None = None,
    suppression_seconds: dict[str, float] | None = None,
) -> list[SpotPrediction]:
    """Convert per-timestep probabilities into class-specific local maxima.

    Each class uses its own threshold and temporal suppression window. This function
    is independent of PyTorch so inference and metric tests can share identical
    postprocessing.
    """

    scores = np.asarray(probabilities, dtype=np.float32)
    times = np.asarray(timestamps, dtype=np.float64)
    if scores.ndim != 2 or scores.shape[1] != len(config.labels):
        raise ValueError("probabilities must have shape [time, number of labels]")
    if times.ndim != 1 or len(times) != len(scores):
        raise ValueError("timestamps must contain one value per probability row")
    if offsets is not None and np.asarray(offsets).shape != scores.shape:
        raise ValueError("offsets must have the same shape as probabilities")
    missing = set(config.labels) - set(thresholds)
    if missing:
        raise ValueError(f"Missing thresholds for: {', '.join(sorted(missing))}")

    offset_values = np.zeros_like(scores) if offsets is None else np.asarray(offsets)
    nms = suppression_seconds or {
        "corner": 10.0,
        "goal_kick": 10.0,
        "shot_attempt": 3.0,
    }
    predictions: list[SpotPrediction] = []
    for class_index, event_type in enumerate(config.labels):
        candidates: list[SpotPrediction] = []
        for index, confidence in enumerate(scores[:, class_index]):
            if confidence < thresholds[event_type]:
                continue
            previous = scores[index - 1, class_index] if index else -np.inf
            following = scores[index + 1, class_index] if index + 1 < len(scores) else -np.inf
            if confidence < previous or confidence < following:
                continue
            offset_seconds = float(offset_values[index, class_index])
            candidates.append(
                SpotPrediction(
                    event_type=event_type,
                    timestamp_seconds=max(0.0, float(times[index]) + offset_seconds),
                    confidence=float(confidence),
                    offset_seconds=offset_seconds,
                )
            )

        kept: list[SpotPrediction] = []
        radius = nms.get(event_type, 5.0)
        for candidate in sorted(candidates, key=lambda item: item.confidence, reverse=True):
            if all(
                abs(candidate.timestamp_seconds - accepted.timestamp_seconds) > radius
                for accepted in kept
            ):
                kept.append(candidate)
        predictions.extend(kept)
    return sorted(predictions, key=lambda item: item.timestamp_seconds)


def build_regnet_bigru(
    number_of_classes: int,
    hidden_size: int = 128,
    pretrained: bool = False,
) -> Any:
    """Build the experimental RegNetY-200MF + BiGRU spotting network.

    PyTorch remains an optional training dependency; importing the production API
    does not load it. The returned module emits class logits and sub-sample offsets
    for every input timestep.
    """

    try:
        import timm
        import torch
    except ImportError as error:
        raise RuntimeError("Instala el extra de entrenamiento: pip install -e '.[training]'") from error

    frame_encoder = timm.create_model(
        "regnety_002",
        pretrained=pretrained,
        num_classes=0,
        global_pool="avg",
    )
    feature_size = int(frame_encoder.num_features)

    class RegNetBiGRUSpotter(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.frame_encoder = frame_encoder
            self.temporal = torch.nn.GRU(
                input_size=feature_size,
                hidden_size=hidden_size,
                batch_first=True,
                bidirectional=True,
            )
            self.classifier = torch.nn.Linear(hidden_size * 2, number_of_classes)
            self.offset = torch.nn.Linear(hidden_size * 2, number_of_classes)

        def forward(self, frames: Any) -> tuple[Any, Any]:
            if frames.ndim != 5:
                raise ValueError("frames must have shape [batch, time, channels, height, width]")
            batch, time, channels, height, width = frames.shape
            encoded = self.frame_encoder(
                frames.reshape(batch * time, channels, height, width)
            ).reshape(batch, time, feature_size)
            temporal, _ = self.temporal(encoded)
            return self.classifier(temporal), torch.tanh(self.offset(temporal))

    return RegNetBiGRUSpotter()
