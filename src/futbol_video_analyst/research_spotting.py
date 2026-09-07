from collections.abc import Callable
from pathlib import Path
from threading import Lock

from futbol_video_analyst.domain import EventCreate, EventSource, EventType, Match
from futbol_video_analyst.model_benchmark import ResearchCornerSpotter


class ResearchAssistedCornerSpotter:
    """Lazy research-only teacher used to propose human review candidates."""

    def __init__(
        self,
        model_path: Path,
        cache_dir: Path,
        requested_device: str = "auto",
        corner_threshold: float = 0.93,
    ) -> None:
        self.model_path = model_path
        self.cache_dir = cache_dir
        self.requested_device = requested_device
        self.corner_threshold = corner_threshold
        self._spotter: ResearchCornerSpotter | None = None
        self._lock = Lock()

    def _load(self) -> ResearchCornerSpotter:
        with self._lock:
            if self._spotter is None:
                self._spotter = ResearchCornerSpotter(
                    self.model_path,
                    self.cache_dir,
                    self.requested_device,
                    self.corner_threshold,
                )
            return self._spotter

    def spot(
        self, match: Match, on_progress: Callable[[float, int], None]
    ) -> list[EventCreate]:
        on_progress(0.05, 0)
        predictions = self._load().spot(match, encoder_batch_size=64, batch_size=16)
        on_progress(1.0, len(predictions))
        return [
            EventCreate(
                type=EventType.CORNER,
                start_seconds=max(0, prediction.timestamp_seconds - 12),
                peak_seconds=prediction.timestamp_seconds,
                end_seconds=min(match.duration_seconds, prediction.timestamp_seconds + 18),
                confidence=prediction.confidence,
                source=EventSource.DETECTOR,
                notes=(
                    "Revisión asistida por soccernet-research-temporal-v005; "
                    "research_only; requiere validación humana"
                ),
            )
            for prediction in predictions
        ]
