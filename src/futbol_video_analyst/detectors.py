import cv2

from futbol_video_analyst.domain import (
    EventCreate,
    EventSource,
    EventType,
    Match,
    ReviewStatus,
    VisualSignal,
)


def visual_feature_distance(first: VisualSignal, second: VisualSignal) -> float:
    differences = (
        (min(first.player_candidates, 30) - min(second.player_candidates, 30)) / 30,
        (min(first.ball_candidates, 150) - min(second.ball_candidates, 150)) / 150,
        (min(first.line_ratio, 0.18) - min(second.line_ratio, 0.18)) / 0.18,
        (min(first.change_score, 0.2) - min(second.change_score, 0.2)) / 0.2,
        first.green_ratio - second.green_ratio,
    )
    return sum(value * value for value in differences) ** 0.5


class CornerReviewCalibration:
    minimum_confirmed = 2
    minimum_rejected = 5

    def __init__(self, examples: list[tuple[VisualSignal, ReviewStatus]]) -> None:
        self.confirmed = [signal for signal, status in examples if status is ReviewStatus.CONFIRMED]
        self.rejected = [signal for signal, status in examples if status is ReviewStatus.REJECTED]

    @property
    def ready(self) -> bool:
        return (
            len(self.confirmed) >= self.minimum_confirmed
            and len(self.rejected) >= self.minimum_rejected
        )

    def evaluate(self, signal: VisualSignal) -> tuple[bool, float]:
        if not self.ready:
            return True, 0.5
        positive_distance = min(visual_feature_distance(signal, item) for item in self.confirmed)
        negative_distance = min(visual_feature_distance(signal, item) for item in self.rejected)
        total = positive_distance + negative_distance
        positive_probability = negative_distance / total if total else 0.5
        return positive_distance <= negative_distance * 0.9, positive_probability


def select_motion_timestamp(
    coarse_timestamp: float, observations: list[tuple[float, float, float]]
) -> float:
    """Select a likely restart from (timestamp, motion, green_ratio) observations."""
    usable = [
        observation
        for observation in observations
        if observation[2] >= 0.18 and 0.012 <= observation[1] <= 0.22
    ]
    if not usable:
        return coarse_timestamp
    return max(usable, key=lambda observation: observation[1])[0]


class CornerTimestampRefiner:
    sample_interval_seconds = 0.5
    seconds_before = 8.0
    seconds_after = 4.0
    estimated_kick_delay_seconds = 2.0

    def refine(self, match: Match, candidates: list[EventCreate]) -> list[EventCreate]:
        if not candidates:
            return []
        capture = cv2.VideoCapture(match.video_path)
        if not capture.isOpened():
            return candidates
        refined: list[EventCreate] = []
        try:
            for candidate in candidates:
                observations = self._motion_observations(capture, match, candidate.peak_seconds)
                motion_timestamp = select_motion_timestamp(candidate.peak_seconds, observations)
                timestamp = motion_timestamp
                if motion_timestamp != candidate.peak_seconds:
                    timestamp = min(
                        match.duration_seconds, motion_timestamp + self.estimated_kick_delay_seconds
                    )
                refined.append(
                    candidate.model_copy(
                        update={
                            "start_seconds": max(0, timestamp - 8),
                            "peak_seconds": timestamp,
                            "end_seconds": min(match.duration_seconds, timestamp + 12),
                            "notes": (
                                "Candidato automático con momento afinado: balón, jugadores, "
                                "líneas y cambio de movimiento"
                            ),
                        }
                    )
                )
        finally:
            capture.release()
        return refined

    def _motion_observations(
        self, capture: cv2.VideoCapture, match: Match, coarse_timestamp: float
    ) -> list[tuple[float, float, float]]:
        start = max(0, coarse_timestamp - self.seconds_before)
        end = min(match.duration_seconds - 0.001, coarse_timestamp + self.seconds_after)
        capture.set(cv2.CAP_PROP_POS_MSEC, start * 1000)
        next_sample_timestamp = start
        previous_gray = None
        observations: list[tuple[float, float, float]] = []
        while True:
            success, frame = capture.read()
            if not success:
                break
            timestamp = capture.get(cv2.CAP_PROP_POS_MSEC) / 1000
            if timestamp > end:
                break
            if timestamp + 0.001 < next_sample_timestamp:
                continue
            height, width = frame.shape[:2]
            target_width = min(384, width)
            target_height = max(1, round(height * target_width / width))
            sample = cv2.resize(frame, (target_width, target_height))
            hsv = cv2.cvtColor(sample, cv2.COLOR_BGR2HSV)
            green_mask = cv2.inRange(hsv, (30, 35, 30), (95, 255, 255))
            green_ratio = cv2.countNonZero(green_mask) / green_mask.size
            gray = cv2.cvtColor(sample, cv2.COLOR_BGR2GRAY)
            if previous_gray is not None:
                motion = float(cv2.absdiff(gray, previous_gray).mean() / 255)
                observations.append((timestamp, motion, green_ratio))
            previous_gray = gray
            next_sample_timestamp += self.sample_interval_seconds
        return observations


class CornerCandidateDetector:
    """Find conservative corner-like sequences from experimental visual signals."""

    cooldown_seconds = 20.0

    def detect(
        self,
        match: Match,
        signals: list[VisualSignal],
        review_examples: list[tuple[VisualSignal, ReviewStatus]] | None = None,
    ) -> list[EventCreate]:
        calibration = CornerReviewCalibration(review_examples or [])
        scored: list[tuple[VisualSignal, float]] = []
        for signal in signals:
            if (
                not signal.likely_field
                or signal.ball_candidates < 1
                or signal.player_candidates < 2
                or signal.line_ratio < 0.006
                or signal.change_score >= 0.18
            ):
                continue
            accepted, learned_probability = calibration.evaluate(signal)
            if not accepted:
                continue
            score = (
                0.2
                + min(signal.player_candidates / 6, 1) * 0.25
                + min(signal.ball_candidates, 1) * 0.25
                + min(signal.line_ratio / 0.02, 1) * 0.2
                + 0.1
            )
            if calibration.ready:
                score = 0.45 * score + 0.55 * learned_probability
            scored.append((signal, min(score, 0.92)))

        selected: list[tuple[VisualSignal, float]] = []
        for signal, score in scored:
            if (
                selected
                and signal.timestamp_seconds - selected[-1][0].timestamp_seconds
                < self.cooldown_seconds
            ):
                if score > selected[-1][1]:
                    selected[-1] = (signal, score)
                continue
            selected.append((signal, score))

        return [
            EventCreate(
                type=EventType.CORNER,
                start_seconds=max(0, signal.timestamp_seconds - 8),
                peak_seconds=signal.timestamp_seconds,
                end_seconds=min(match.duration_seconds, signal.timestamp_seconds + 12),
                confidence=score,
                source=EventSource.DETECTOR,
                notes=(
                    "Candidato automático calibrado con revisiones locales"
                    if calibration.ready
                    else "Candidato automático: balón, jugadores y líneas visibles en una toma estable"
                ),
            )
            for signal, score in selected
        ]
