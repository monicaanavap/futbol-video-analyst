import json
import re
from collections import Counter
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor
from datetime import UTC, datetime
from pathlib import Path
from threading import Lock
from uuid import uuid4

from futbol_video_analyst.clips import ClipExportError, FFmpegClipExporter
from futbol_video_analyst.domain import (
    DatasetExport,
    DatasetExportJob,
    DatasetExportStatus,
    Event,
    EventSource,
    Match,
    ReviewStatus,
)


class DatasetExportError(RuntimeError):
    pass


def _safe_name(value: str) -> str:
    normalized = re.sub(r"[^a-zA-Z0-9_-]+", "-", value.strip()).strip("-").lower()
    return normalized[:60] or "partido"


def _label_for(event: Event) -> str | None:
    if event.source is EventSource.MANUAL and event.review_status is not ReviewStatus.REJECTED:
        return event.type.value
    if event.source is EventSource.DETECTOR and event.review_status is ReviewStatus.CONFIRMED:
        return event.type.value
    if event.source is EventSource.DETECTOR and event.review_status is ReviewStatus.REJECTED:
        return "negative"
    return None


class LocalDatasetExporter:
    def __init__(self, clip_exporter: FFmpegClipExporter) -> None:
        self.clip_exporter = clip_exporter

    def export(
        self,
        matches: list[Match],
        events_by_match: dict[str, list[Event]],
        destination_root: Path,
        on_progress: Callable[[int, int], None] | None = None,
    ) -> DatasetExport:
        selected: list[tuple[Match, Event, str]] = []
        skipped_events = 0
        for match in matches:
            for event in events_by_match.get(match.id, []):
                label = _label_for(event)
                if label is None:
                    skipped_events += 1
                else:
                    selected.append((match, event, label))

        if not selected:
            raise DatasetExportError(
                "No hay etiquetas listas. Confirma, reclasifica o agrega momentos manuales primero."
            )

        if on_progress:
            on_progress(0, len(selected))

        missing = sorted(
            {str(Path(match.video_path)) for match, _, _ in selected if not Path(match.video_path).is_file()}
        )
        if missing:
            raise DatasetExportError(
                "No se encontraron algunos videos originales: " + ", ".join(missing)
            )

        timestamp = datetime.now(UTC).strftime("%Y%m%d-%H%M%S")
        destination = destination_root / f"dataset-{timestamp}-{uuid4().hex[:6]}"
        destination.mkdir(parents=True, exist_ok=False)
        manifest_path = destination / "manifest.jsonl"
        label_counts: Counter[str] = Counter()

        try:
            with manifest_path.open("w", encoding="utf-8") as manifest:
                for match, event, label in selected:
                    match_folder = f"{_safe_name(match.title)}-{match.id[:8]}"
                    filename = f"{int(event.peak_seconds):06d}-{event.id[:8]}.mp4"
                    relative_path = Path(match_folder) / label / filename
                    clip_path = destination / relative_path
                    self.clip_exporter.export(
                        Path(match.video_path),
                        clip_path,
                        event.start_seconds,
                        event.end_seconds,
                    )
                    record = {
                        "clip_path": relative_path.as_posix(),
                        "label": label,
                        "match_id": match.id,
                        "match_title": match.title,
                        "event_id": event.id,
                        "source": event.source.value,
                        "review_status": event.review_status.value,
                        "detected_type": event.detected_type.value if event.detected_type else None,
                        "start_seconds": event.start_seconds,
                        "peak_seconds": event.peak_seconds,
                        "end_seconds": event.end_seconds,
                        "notes": event.notes,
                    }
                    manifest.write(json.dumps(record, ensure_ascii=False) + "\n")
                    label_counts[label] += 1
                    if on_progress:
                        on_progress(sum(label_counts.values()), len(selected))
        except (OSError, ClipExportError) as error:
            raise DatasetExportError(
                f"No se pudo completar el dataset. Los archivos parciales quedaron en {destination}"
            ) from error

        summary = {
            "created_at": datetime.now(UTC).isoformat(),
            "clips": len(selected),
            "matches": len({match.id for match, _, _ in selected}),
            "skipped_events": skipped_events,
            "label_counts": dict(sorted(label_counts.items())),
            "manifest": "manifest.jsonl",
        }
        (destination / "summary.json").write_text(
            json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
        )

        return DatasetExport(
            path=str(destination.resolve()),
            manifest_path=str(manifest_path.resolve()),
            clips=len(selected),
            matches=summary["matches"],
            skipped_events=skipped_events,
            label_counts=dict(label_counts),
        )


class DatasetExportCoordinator:
    def __init__(self, exporter: LocalDatasetExporter, destination_root: Path) -> None:
        self.exporter = exporter
        self.destination_root = destination_root
        self.executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="dataset-export")
        self.jobs: dict[str, DatasetExportJob] = {}
        self.lock = Lock()

    def start(
        self, matches: list[Match], events_by_match: dict[str, list[Event]]
    ) -> DatasetExportJob:
        with self.lock:
            active = next(
                (
                    job
                    for job in self.jobs.values()
                    if job.status in {DatasetExportStatus.QUEUED, DatasetExportStatus.RUNNING}
                ),
                None,
            )
            if active:
                return active.model_copy(deep=True)
            job = DatasetExportJob(
                id=str(uuid4()),
                status=DatasetExportStatus.QUEUED,
                progress=0,
                clips_processed=0,
                total_clips=0,
            )
            self.jobs[job.id] = job
        self.executor.submit(self._run, job.id, matches, events_by_match)
        return job.model_copy(deep=True)

    def get(self, job_id: str) -> DatasetExportJob | None:
        with self.lock:
            job = self.jobs.get(job_id)
            return job.model_copy(deep=True) if job else None

    def _run(
        self, job_id: str, matches: list[Match], events_by_match: dict[str, list[Event]]
    ) -> None:
        def report(processed: int, total: int) -> None:
            with self.lock:
                job = self.jobs[job_id]
                job.status = DatasetExportStatus.RUNNING
                job.clips_processed = processed
                job.total_clips = total
                job.progress = processed / total if total else 0

        try:
            result = self.exporter.export(
                matches, events_by_match, self.destination_root, on_progress=report
            )
            with self.lock:
                job = self.jobs[job_id]
                job.status = DatasetExportStatus.COMPLETED
                job.progress = 1
                job.result = result
        except Exception as error:  # noqa: BLE001 - expose background export failures
            with self.lock:
                job = self.jobs[job_id]
                job.status = DatasetExportStatus.FAILED
                job.error = str(error)

    def shutdown(self) -> None:
        self.executor.shutdown(wait=False, cancel_futures=True)
