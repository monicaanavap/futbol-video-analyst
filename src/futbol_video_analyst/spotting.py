from collections.abc import Callable
from pathlib import Path
from typing import Any

import cv2
import numpy as np

from futbol_video_analyst.domain import EventCreate, EventSource, EventType, Match

SpottingProgressCallback = Callable[[float, int], None]


class NeuralCornerSpotter:
    """Scan a complete match with the locally trained R3D-18 corner classifier."""

    frames_per_window = 16
    window_seconds = 4.0
    cooldown_seconds = 12.0
    max_candidates = 30

    def __init__(self, model_path: Path, device: str = "auto") -> None:
        self.model_path = model_path
        self.requested_device = device
        self._runtime: tuple[Any, Any, Any, Any, Any] | None = None

    def spot(
        self, match: Match, on_progress: SpottingProgressCallback
    ) -> list[EventCreate] | None:
        runtime = self._load_runtime()
        if runtime is None:
            return None
        torch, backbone, head, transform, device = runtime

        capture = cv2.VideoCapture(match.video_path)
        if not capture.isOpened():
            raise RuntimeError("No se pudo abrir el video para el barrido neuronal")

        fps = capture.get(cv2.CAP_PROP_FPS) or match.fps or 25
        sample_interval = self.window_seconds / self.frames_per_window
        batch_size = 8 if device.type != "cpu" else 2
        frame_index = 0
        next_sample = 0.0
        window_frames: list[np.ndarray] = []
        window_timestamps: list[float] = []
        batch: list[Any] = []
        batch_timestamps: list[float] = []
        embeddings: list[Any] = []
        embedding_timestamps: list[float] = []
        windows_processed = 0

        def process_batch() -> None:
            nonlocal windows_processed
            if not batch:
                return
            inputs = torch.stack(batch).to(device)
            with torch.inference_mode():
                result = backbone(inputs).detach().cpu()
            embeddings.extend(result)
            embedding_timestamps.extend(batch_timestamps)
            windows_processed += len(batch)
            progress = min(1.0, batch_timestamps[-1] / match.duration_seconds)
            on_progress(progress, windows_processed)
            batch.clear()
            batch_timestamps.clear()

        try:
            while True:
                success = capture.grab()
                if not success:
                    break
                timestamp = capture.get(cv2.CAP_PROP_POS_MSEC) / 1000
                if timestamp <= 0 and frame_index:
                    timestamp = frame_index / fps
                frame_index += 1
                if timestamp + 1e-3 < next_sample:
                    continue
                success, frame = capture.retrieve()
                if not success:
                    continue
                window_frames.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
                window_timestamps.append(timestamp)
                next_sample += sample_interval
                while next_sample <= timestamp:
                    next_sample += sample_interval
                if len(window_frames) < self.frames_per_window:
                    continue

                frames = np.stack(window_frames)
                tensor = torch.from_numpy(frames.copy()).permute(0, 3, 1, 2)
                batch.append(transform(tensor))
                batch_timestamps.append(
                    (window_timestamps[0] + window_timestamps[-1]) / 2
                )
                window_frames.clear()
                window_timestamps.clear()
                if len(batch) == batch_size:
                    process_batch()
            process_batch()
        finally:
            capture.release()
        on_progress(1.0, windows_processed)

        checkpoint = self._checkpoint()
        context_steps = checkpoint.get("temporal_context_steps")
        if context_steps:
            vectors = [
                torch.cat(
                    [
                        embeddings[min(max(index + int(step), 0), len(embeddings) - 1)]
                        for step in context_steps
                    ]
                )
                for index in range(len(embeddings))
            ]
        else:
            vectors = embeddings
        scored: list[tuple[float, float]] = []
        scoring_batch_size = 64
        for start in range(0, len(vectors), scoring_batch_size):
            values = torch.stack(vectors[start : start + scoring_batch_size]).to(device)
            with torch.inference_mode():
                batch_probabilities = torch.sigmoid(head(values)).cpu().numpy().reshape(-1)
            scored.extend(
                (timestamp, float(probability))
                for timestamp, probability in zip(
                    embedding_timestamps[start : start + scoring_batch_size],
                    batch_probabilities,
                )
            )
        threshold = float(checkpoint["threshold"])
        return self._merge_predictions(match, scored, threshold)

    def _load_runtime(self) -> tuple[Any, Any, Any, Any, Any] | None:
        if self._runtime is not None:
            return self._runtime
        if not self.model_path.is_file():
            return None
        try:
            import torch
            from torchvision.models.video import R3D_18_Weights, r3d_18
        except ImportError:
            return None

        checkpoint = self._checkpoint(torch)
        if self.requested_device == "auto":
            if torch.backends.mps.is_available():
                device = torch.device("mps")
            elif torch.cuda.is_available():
                device = torch.device("cuda")
            else:
                device = torch.device("cpu")
        else:
            device = torch.device(self.requested_device)

        weights = R3D_18_Weights.DEFAULT
        backbone = r3d_18(weights=weights)
        backbone.fc = torch.nn.Identity()
        backbone.eval().to(device)

        mean = checkpoint["feature_mean"].to(device)
        standard_deviation = checkpoint["feature_standard_deviation"].to(device)
        classifier = torch.nn.Linear(int(mean.numel()), 1).to(device)
        classifier.load_state_dict(checkpoint["head_state_dict"])
        classifier.eval()
        head = torch.nn.Sequential(
            _FeatureStandardizer(mean, standard_deviation), classifier
        ).to(device)
        head.eval()
        self._runtime = (torch, backbone, head, weights.transforms(), device)
        return self._runtime

    def _checkpoint(self, torch: Any | None = None) -> dict[str, Any]:
        if torch is None:
            import torch as torch_module

            torch = torch_module
        return torch.load(self.model_path, map_location="cpu", weights_only=False)

    def _merge_predictions(
        self, match: Match, scored: list[tuple[float, float]], threshold: float
    ) -> list[EventCreate]:
        clusters: list[list[tuple[float, float]]] = []
        for timestamp, probability in scored:
            if probability < threshold:
                continue
            if clusters and timestamp - clusters[-1][-1][0] <= self.cooldown_seconds:
                clusters[-1].append((timestamp, probability))
            else:
                clusters.append([(timestamp, probability)])

        candidates = [
            EventCreate(
                type=EventType.CORNER,
                start_seconds=max(0, timestamp - 8),
                peak_seconds=timestamp,
                end_seconds=min(match.duration_seconds, timestamp + 12),
                confidence=probability,
                source=EventSource.DETECTOR,
                notes=(
                    f"Candidato neuronal {self.model_path.stem}: ventana temporal R3D-18"
                ),
            )
            for cluster in clusters
            for timestamp, probability in [max(cluster, key=lambda item: item[1])]
        ]
        if len(candidates) > self.max_candidates:
            candidates = sorted(
                sorted(candidates, key=lambda event: event.confidence, reverse=True)[
                    : self.max_candidates
                ],
                key=lambda event: event.peak_seconds,
            )
        return candidates


class _FeatureStandardizer:
    """Small callable torch module assembled lazily to keep torch optional at import time."""

    def __new__(cls, mean: Any, standard_deviation: Any) -> Any:
        import torch

        class Standardizer(torch.nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.register_buffer("mean", mean)
                self.register_buffer("standard_deviation", standard_deviation)

            def forward(self, values: Any) -> Any:
                return (values - self.mean) / self.standard_deviation

        return Standardizer()
