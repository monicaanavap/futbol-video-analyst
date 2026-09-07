import json
import math
from pathlib import Path
from typing import Any

import numpy as np

from futbol_video_analyst.domain import (
    EventCreate,
    EventSource,
    EventType,
    Match,
    VisualSignal,
)

TEMPORAL_MODEL_VERSION = 1
CONTEXT_RADII = (1, 3, 6)
BASE_FEATURES = (
    "green_ratio",
    "brightness",
    "change_score",
    "likely_field",
    "players_log1p",
    "balls_log1p",
    "line_ratio",
)


def signal_matrix(signals: list[VisualSignal]) -> np.ndarray:
    return np.asarray(
        [
            (
                signal.green_ratio,
                signal.brightness,
                signal.change_score,
                float(signal.likely_field),
                math.log1p(signal.player_candidates),
                math.log1p(signal.ball_candidates),
                signal.line_ratio,
            )
            for signal in signals
        ],
        dtype=np.float32,
    )


def temporal_features(signals: list[VisualSignal]) -> np.ndarray:
    """Build causal-and-future context features over an already sampled timeline."""
    base = signal_matrix(signals)
    if not len(base):
        return np.empty((0, len(BASE_FEATURES) * (1 + 3 * len(CONTEXT_RADII))))
    rows: list[np.ndarray] = []
    for index, current in enumerate(base):
        parts = [current]
        for radius in CONTEXT_RADII:
            start = max(0, index - radius)
            end = min(len(base), index + radius + 1)
            window = base[start:end]
            before = base[start:index] if index > start else current[None, :]
            after = base[index + 1 : end] if index + 1 < end else current[None, :]
            parts.extend((window.mean(axis=0), window.std(axis=0), after.mean(axis=0) - before.mean(axis=0)))
        rows.append(np.concatenate(parts))
    return np.asarray(rows, dtype=np.float32)


def probabilities(features: np.ndarray, model: dict[str, Any]) -> np.ndarray:
    mean = np.asarray(model["feature_mean"], dtype=np.float32)
    scale = np.asarray(model["feature_scale"], dtype=np.float32)
    weights = np.asarray(model["weights"], dtype=np.float32)
    normalized = (features - mean) / scale
    logits = np.clip(normalized @ weights + float(model["bias"]), -30, 30)
    raw = 1 / (1 + np.exp(-logits))
    if len(raw) < 3:
        return raw
    smoothed = raw.copy()
    smoothed[1:-1] = 0.25 * raw[:-2] + 0.5 * raw[1:-1] + 0.25 * raw[2:]
    return smoothed


def select_peaks(
    timestamps: np.ndarray,
    scores: np.ndarray,
    threshold: float,
    cooldown_seconds: float = 20.0,
    max_candidates: int = 30,
) -> list[tuple[float, float]]:
    local: list[tuple[float, float]] = []
    for index, score in enumerate(scores):
        if score < threshold:
            continue
        start = max(0, index - 3)
        end = min(len(scores), index + 4)
        if score < float(scores[start:end].max()):
            continue
        local.append((float(timestamps[index]), float(score)))

    selected: list[tuple[float, float]] = []
    for timestamp, score in local:
        if selected and timestamp - selected[-1][0] < cooldown_seconds:
            if score > selected[-1][1]:
                selected[-1] = (timestamp, score)
        else:
            selected.append((timestamp, score))
    if len(selected) > max_candidates:
        selected = sorted(
            sorted(selected, key=lambda item: item[1], reverse=True)[:max_candidates]
        )
    return selected


class TemporalCornerSpotter:
    """Lightweight action spotter over full-match visual-signal sequences."""

    def __init__(self, model_path: Path) -> None:
        self.model_path = model_path
        self._model: dict[str, Any] | None = None

    def detect(self, match: Match, signals: list[VisualSignal]) -> list[EventCreate] | None:
        model = self._load()
        if model is None:
            return None
        features = temporal_features(signals)
        scores = probabilities(features, model)
        timestamps = np.asarray([signal.timestamp_seconds for signal in signals])
        peaks = select_peaks(
            timestamps,
            scores,
            float(model["threshold"]),
            float(model.get("cooldown_seconds", 20)),
            int(model.get("max_candidates", 30)),
        )
        return [
            EventCreate(
                type=EventType.CORNER,
                start_seconds=max(0, timestamp - 8),
                peak_seconds=timestamp,
                end_seconds=min(match.duration_seconds, timestamp + 12),
                confidence=score,
                source=EventSource.DETECTOR,
                notes=f"Candidato temporal {self.model_path.stem}: contexto de 24 segundos",
            )
            for timestamp, score in peaks
        ]

    def _load(self) -> dict[str, Any] | None:
        if self._model is not None:
            return self._model
        if not self.model_path.is_file():
            return None
        model = json.loads(self.model_path.read_text(encoding="utf-8"))
        if model.get("version") != TEMPORAL_MODEL_VERSION:
            raise ValueError("Versión de modelo temporal no compatible")
        self._model = model
        return model
