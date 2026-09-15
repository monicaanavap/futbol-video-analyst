from collections.abc import Callable
from pathlib import Path
from threading import Lock

from futbol_video_analyst.domain import EventCreate, EventSource, EventType, Match
from futbol_video_analyst.model_benchmark import ResearchCornerSpotter
from futbol_video_analyst.spotting_evaluation import TimelineEvent


def prefer_goals(
    predictions: list[TimelineEvent], tolerance_seconds: float = 3.0
) -> list[TimelineEvent]:
    goal_times = [
        prediction.timestamp_seconds
        for prediction in predictions
        if prediction.event_type == EventType.GOAL.value
    ]
    return [
        prediction
        for prediction in predictions
        if not (
            prediction.event_type == EventType.SHOT_ATTEMPT.value
            and any(
                abs(prediction.timestamp_seconds - goal_time) <= tolerance_seconds
                for goal_time in goal_times
            )
        )
    ]


class ResearchAssistedCornerSpotter:
    """Lazy research-only teacher used to propose human review candidates."""

    def __init__(
        self,
        model_path: Path,
        cache_dir: Path,
        requested_device: str = "auto",
        corner_threshold: float = 0.93,
        enabled_types: tuple[EventType, ...] = (EventType.CORNER,),
    ) -> None:
        self.model_path = model_path
        self.cache_dir = cache_dir
        self.requested_device = requested_device
        self.corner_threshold = corner_threshold
        self.enabled_types = enabled_types
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
        predictions = prefer_goals(
            self._load().spot(
                match,
                encoder_batch_size=64,
                batch_size=16,
                enabled_labels={event_type.value for event_type in self.enabled_types},
            )
        )
        on_progress(1.0, len(predictions))
        return [
            EventCreate(
                type=EventType(prediction.event_type),
                start_seconds=max(0, prediction.timestamp_seconds - 12),
                peak_seconds=prediction.timestamp_seconds,
                end_seconds=min(match.duration_seconds, prediction.timestamp_seconds + 18),
                confidence=prediction.confidence,
                source=EventSource.DETECTOR,
                notes=(
                    f"Revisión asistida por {self.model_path.stem}; "
                    "research_only; requiere validación humana"
                ),
            )
            for prediction in predictions
        ]
