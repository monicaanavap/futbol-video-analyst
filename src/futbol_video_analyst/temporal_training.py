import argparse
import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np

from futbol_video_analyst.database import Database
from futbol_video_analyst.domain import Event, EventSource, EventType, Match, ReviewStatus
from futbol_video_analyst.temporal import (
    BASE_FEATURES,
    CONTEXT_RADII,
    TEMPORAL_MODEL_VERSION,
    probabilities,
    select_peaks,
    temporal_features,
)


def true_corner_peaks(events: list[Event]) -> list[float]:
    return [
        event.peak_seconds
        for event in events
        if event.type is EventType.CORNER
        and (event.source is EventSource.MANUAL or event.review_status is ReviewStatus.CONFIRMED)
    ]


def hard_negative_peaks(events: list[Event]) -> list[float]:
    return [
        event.peak_seconds
        for event in events
        if event.review_status is ReviewStatus.REJECTED
        or (event.detected_type is EventType.CORNER and event.type is not EventType.CORNER)
    ]


def nearest_indices(timestamps: np.ndarray, peaks: list[float]) -> list[int]:
    return sorted({int(np.abs(timestamps - peak).argmin()) for peak in peaks})


def training_rows(
    features: np.ndarray,
    timestamps: np.ndarray,
    corners: list[float],
    hard_negatives: list[float],
    negative_ratio: int = 12,
) -> tuple[np.ndarray, np.ndarray]:
    positives = nearest_indices(timestamps, corners)
    hard = nearest_indices(timestamps, hard_negatives)
    allowed = [
        index
        for index, timestamp in enumerate(timestamps)
        if all(abs(float(timestamp) - peak) > 20 for peak in corners)
    ]
    wanted = min(len(allowed), max(len(positives) * negative_ratio, len(hard)))
    regular = []
    if wanted:
        regular = [allowed[index] for index in np.linspace(0, len(allowed) - 1, wanted, dtype=int)]
    negatives = sorted(set(hard + regular) - set(positives))
    indices = positives + negatives
    labels = np.asarray([1] * len(positives) + [0] * len(negatives), dtype=np.float32)
    return features[indices], labels


def fit_logistic(features: np.ndarray, labels: np.ndarray) -> dict[str, Any]:
    mean = features.mean(axis=0)
    scale = features.std(axis=0)
    scale[scale < 1e-6] = 1
    values = (features - mean) / scale
    weights = np.zeros(values.shape[1], dtype=np.float32)
    bias = 0.0
    first_w = np.zeros_like(weights)
    second_w = np.zeros_like(weights)
    first_b = 0.0
    second_b = 0.0
    positives = max(1.0, float(labels.sum()))
    positive_weight = float(len(labels) - labels.sum()) / positives
    sample_weights = np.where(labels > 0, positive_weight, 1.0).astype(np.float32)
    for step in range(1, 2001):
        logits = np.clip(values @ weights + bias, -30, 30)
        predicted = 1 / (1 + np.exp(-logits))
        errors = (predicted - labels) * sample_weights
        gradient_w = values.T @ errors / sample_weights.sum() + 0.002 * weights
        gradient_b = float(errors.sum() / sample_weights.sum())
        first_w = 0.9 * first_w + 0.1 * gradient_w
        second_w = 0.999 * second_w + 0.001 * gradient_w * gradient_w
        first_b = 0.9 * first_b + 0.1 * gradient_b
        second_b = 0.999 * second_b + 0.001 * gradient_b * gradient_b
        correction_w = first_w / (1 - 0.9**step)
        correction_v = second_w / (1 - 0.999**step)
        correction_b = first_b / (1 - 0.9**step)
        correction_bv = second_b / (1 - 0.999**step)
        weights -= 0.02 * correction_w / (np.sqrt(correction_v) + 1e-8)
        bias -= 0.02 * correction_b / (correction_bv**0.5 + 1e-8)
    return {
        "feature_mean": mean.tolist(),
        "feature_scale": scale.tolist(),
        "weights": weights.tolist(),
        "bias": bias,
    }


def spotting_metrics(predictions: list[tuple[float, float]], truth: list[float]) -> dict[str, Any]:
    matched_predictions: set[int] = set()
    true_positive = 0
    for peak in truth:
        candidates = [
            (abs(timestamp - peak), index)
            for index, (timestamp, _) in enumerate(predictions)
            if index not in matched_predictions and abs(timestamp - peak) <= 16
        ]
        if candidates:
            _, index = min(candidates)
            matched_predictions.add(index)
            true_positive += 1
    false_positive = len(predictions) - true_positive
    false_negative = len(truth) - true_positive
    precision = true_positive / len(predictions) if predictions else 0.0
    recall = true_positive / len(truth) if truth else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return {
        "true_positive": true_positive,
        "false_positive": false_positive,
        "false_negative": false_negative,
        "precision": precision,
        "recall": recall,
        "f1": f1,
    }


def run_training(database_path: Path, output: Path, validation_match: str | None) -> Path:
    database = Database(database_path)
    prepared: list[tuple[Match, np.ndarray, np.ndarray, list[float], list[float]]] = []
    for match in database.list_matches():
        signals = database.list_visual_signals(match.id)
        events = database.list_events(match.id)
        corners = true_corner_peaks(events)
        if signals and len(corners) >= 2:
            prepared.append(
                (
                    match,
                    temporal_features(signals),
                    np.asarray([signal.timestamp_seconds for signal in signals]),
                    corners,
                    hard_negative_peaks(events),
                )
            )
    if len(prepared) < 2:
        raise ValueError("Se necesitan al menos dos partidos con dos corners etiquetados")
    if validation_match:
        matches = [
            item for item in prepared
            if item[0].id == validation_match or item[0].title.casefold() == validation_match.casefold()
        ]
        if len(matches) != 1:
            raise ValueError(f"No se encontró un partido de validación único: {validation_match}")
        validation = matches[0]
    else:
        validation = min(prepared, key=lambda item: len(item[3]))
    rows: list[np.ndarray] = []
    labels: list[np.ndarray] = []
    training_titles: list[str] = []
    for match, features, timestamps, corners, hard in prepared:
        if match.id == validation[0].id:
            continue
        current_rows, current_labels = training_rows(features, timestamps, corners, hard)
        rows.append(current_rows)
        labels.append(current_labels)
        training_titles.append(match.title)
    model = fit_logistic(np.concatenate(rows), np.concatenate(labels))
    validation_match_model, validation_features, validation_timestamps, truth, _ = validation
    model.update(
        {
            "version": TEMPORAL_MODEL_VERSION,
            "model_type": "temporal_logistic_action_spotter",
            "base_features": list(BASE_FEATURES),
            "context_radii": list(CONTEXT_RADII),
            "cooldown_seconds": 20,
            "max_candidates": 30,
        }
    )
    scores = probabilities(validation_features, model)
    best: tuple[float, float, dict[str, Any]] | None = None
    for threshold in np.arange(0.35, 0.951, 0.025):
        predictions = select_peaks(validation_timestamps, scores, float(threshold))
        metrics = spotting_metrics(predictions, truth)
        rank = (metrics["f1"], metrics["precision"])
        if best is None or rank > best[:2]:
            best = (*rank, {"threshold": float(threshold), "candidates": len(predictions), **metrics})
    assert best is not None
    model["threshold"] = best[2]["threshold"]
    model["metadata"] = {
        "created_at": datetime.now(UTC).isoformat(),
        "database": str(database_path.resolve()),
        "training_matches": training_titles,
        "validation_match": validation_match_model.title,
        "training_rows": int(sum(len(item) for item in labels)),
        "validation_corners": len(truth),
        "validation": best[2],
        "experimental": True,
        "activated_in_app": False,
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(model, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps(model["metadata"], indent=2, ensure_ascii=False))
    return output


def main() -> None:
    parser = argparse.ArgumentParser(description="Entrena el detector temporal de corners")
    parser.add_argument("--database", type=Path, default=Path("data/futbol-video-analyst.sqlite3"))
    parser.add_argument("--output", type=Path, default=Path("models/corner-temporal-v001.json"))
    parser.add_argument("--validation-match")
    arguments = parser.parse_args()
    run_training(arguments.database, arguments.output, arguments.validation_match)


if __name__ == "__main__":
    main()
