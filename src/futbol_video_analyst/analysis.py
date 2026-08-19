import math
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor
from uuid import uuid4

import cv2

from futbol_video_analyst.database import Database
from futbol_video_analyst.detectors import CornerCandidateDetector, CornerTimestampRefiner
from futbol_video_analyst.domain import (
    AnalysisJob,
    AnalysisStage,
    AnalysisStatus,
    Match,
    VisualSignal,
)

ProgressCallback = Callable[[float, int], None]


class VideoAnalysisError(RuntimeError):
    pass


def detect_object_candidates(sample, hsv, green_mask) -> tuple[int, int, float]:
    """Estimate visible field objects without a trained model.

    These counts are intentionally candidates: camera graphics, uniforms and field
    markings can still cause false positives.
    """
    sample_area = sample.shape[0] * sample.shape[1]
    green_ratio = cv2.countNonZero(green_mask) / green_mask.size
    if green_ratio < 0.18:
        return 0, 0, 0.0

    white_mask = cv2.inRange(hsv, (0, 0, 155), (180, 85, 255))
    line_ratio = cv2.countNonZero(white_mask) / white_mask.size

    saturated_mask = cv2.inRange(hsv, (0, 70, 35), (180, 255, 255))
    player_mask = cv2.bitwise_and(cv2.bitwise_not(green_mask), saturated_mask)
    player_mask = cv2.morphologyEx(
        player_mask, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    )
    player_candidates = 0
    for contour in cv2.findContours(player_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[0]:
        area_ratio = cv2.contourArea(contour) / sample_area
        _, _, width, height = cv2.boundingRect(contour)
        if 0.0002 <= area_ratio <= 0.015 and height >= width * 1.15 and height >= 6:
            player_candidates += 1

    ball_mask = cv2.morphologyEx(
        white_mask, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    )
    ball_candidates = 0
    for contour in cv2.findContours(ball_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[0]:
        area = cv2.contourArea(contour)
        area_ratio = area / sample_area
        _, _, width, height = cv2.boundingRect(contour)
        perimeter = cv2.arcLength(contour, True)
        circularity = 4 * math.pi * area / (perimeter * perimeter) if perimeter else 0
        aspect_ratio = width / height if height else 0
        if 0.00002 <= area_ratio <= 0.0015 and 0.6 <= aspect_ratio <= 1.65 and circularity >= 0.45:
            ball_candidates += 1

    return player_candidates, ball_candidates, line_ratio


class VisualSignalAnalyzer:
    def __init__(self, sample_interval_seconds: float = 2.0) -> None:
        self.sample_interval_seconds = sample_interval_seconds

    def analyze(self, match: Match, on_progress: ProgressCallback) -> list[VisualSignal]:
        capture = cv2.VideoCapture(match.video_path)
        if not capture.isOpened():
            raise VideoAnalysisError("No se pudo abrir el video para analizarlo")

        total_samples = max(1, math.ceil(match.duration_seconds / self.sample_interval_seconds))
        signals: list[VisualSignal] = []
        previous_gray = None
        try:
            for index in range(total_samples):
                timestamp = min(
                    index * self.sample_interval_seconds, match.duration_seconds - 0.001
                )
                capture.set(cv2.CAP_PROP_POS_MSEC, timestamp * 1000)
                success, frame = capture.read()
                if not success:
                    continue

                height, width = frame.shape[:2]
                target_width = min(384, width)
                target_height = max(1, round(height * target_width / width))
                sample = cv2.resize(frame, (target_width, target_height))
                hsv = cv2.cvtColor(sample, cv2.COLOR_BGR2HSV)
                green_mask = cv2.inRange(hsv, (30, 35, 30), (95, 255, 255))
                green_ratio = cv2.countNonZero(green_mask) / green_mask.size
                player_candidates, ball_candidates, line_ratio = detect_object_candidates(
                    sample, hsv, green_mask
                )

                gray = cv2.cvtColor(sample, cv2.COLOR_BGR2GRAY)
                brightness = float(gray.mean() / 255)
                change_score = 0.0
                if previous_gray is not None:
                    change_score = float(cv2.absdiff(gray, previous_gray).mean() / 255)
                previous_gray = gray

                signals.append(
                    VisualSignal(
                        id=str(uuid4()),
                        match_id=match.id,
                        timestamp_seconds=timestamp,
                        green_ratio=green_ratio,
                        brightness=brightness,
                        change_score=min(change_score, 1.0),
                        likely_field=green_ratio >= 0.18,
                        player_candidates=player_candidates,
                        ball_candidates=ball_candidates,
                        line_ratio=line_ratio,
                    )
                )
                on_progress((index + 1) / total_samples, len(signals))
        finally:
            capture.release()

        if not signals:
            raise VideoAnalysisError("No se pudieron extraer muestras del video")
        return signals


class AnalysisCoordinator:
    def __init__(self, database: Database, analyzer: VisualSignalAnalyzer | None = None) -> None:
        self.database = database
        self.analyzer = analyzer or VisualSignalAnalyzer()
        self.corner_detector = CornerCandidateDetector()
        self.corner_refiner = CornerTimestampRefiner()
        self.executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="video-analysis")

    def start(self, match: Match) -> AnalysisJob:
        latest = self.database.get_latest_analysis_job(match.id)
        if latest and latest.status in {AnalysisStatus.QUEUED, AnalysisStatus.RUNNING}:
            return latest
        job = self.database.create_analysis_job(match.id)
        self.executor.submit(self._run, job.id, match)
        return job

    def _run(self, job_id: str, match: Match) -> None:
        samples_processed = 0

        def report(progress: float, samples: int) -> None:
            nonlocal samples_processed
            samples_processed = samples
            self.database.update_analysis_job(
                job_id,
                status=AnalysisStatus.RUNNING,
                stage=AnalysisStage.SAMPLING,
                progress=progress * 0.9,
                samples_processed=samples,
            )

        try:
            self.database.update_analysis_job(
                job_id,
                status=AnalysisStatus.RUNNING,
                stage=AnalysisStage.SAMPLING,
                progress=0,
                samples_processed=0,
            )
            signals = self.analyzer.analyze(match, report)
            self.database.replace_visual_signals(match.id, signals)
            review_examples = self.database.list_corner_review_examples()
            candidates = self.corner_detector.detect(match, signals, review_examples)
            self.database.update_analysis_job(
                job_id,
                status=AnalysisStatus.RUNNING,
                stage=AnalysisStage.REFINING,
                progress=0.92,
                samples_processed=len(signals),
            )
            refined_candidates = self.corner_refiner.refine(match, candidates)
            self.database.replace_corner_candidates(match.id, refined_candidates)
            self.database.update_analysis_job(
                job_id,
                status=AnalysisStatus.COMPLETED,
                stage=AnalysisStage.COMPLETED,
                progress=1,
                samples_processed=len(signals),
            )
        except Exception as error:  # noqa: BLE001 - background failures must be persisted
            self.database.update_analysis_job(
                job_id,
                status=AnalysisStatus.FAILED,
                stage=AnalysisStage.FAILED,
                progress=0,
                samples_processed=samples_processed,
                error=str(error),
            )

    def shutdown(self) -> None:
        self.executor.shutdown(wait=False, cancel_futures=True)
