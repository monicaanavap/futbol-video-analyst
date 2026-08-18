import time
from collections.abc import Callable
from pathlib import Path
from uuid import uuid4

from fastapi.testclient import TestClient

from futbol_video_analyst.domain import Match, VideoMetadata, VisualSignal
from futbol_video_analyst.main import create_app


class FakeVideoInspector:
    def inspect(self, path: Path) -> VideoMetadata:
        return VideoMetadata(duration_seconds=90, width=1920, height=1080, fps=30, codec="h264")


class FakeClipExporter:
    def export(self, source: Path, destination: Path, start: float, end: float) -> Path:
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_bytes(f"{source.name}:{start}:{end}".encode())
        return destination


class FakeVisualAnalyzer:
    def analyze(
        self, match: Match, on_progress: Callable[[float, int], None]
    ) -> list[VisualSignal]:
        match_id = match.id
        on_progress(1.0, 1)
        return [
            VisualSignal(
                id=str(uuid4()),
                match_id=match_id,
                timestamp_seconds=4,
                green_ratio=0.72,
                brightness=0.5,
                change_score=0.14,
                likely_field=True,
                player_candidates=4,
                ball_candidates=1,
                line_ratio=0.02,
            )
        ]


def make_client(tmp_path: Path) -> TestClient:
    return TestClient(
        create_app(
            tmp_path / "test.sqlite3",
            FakeVideoInspector(),
            clips_dir=tmp_path / "clips",
            clip_exporter=FakeClipExporter(),
            visual_analyzer=FakeVisualAnalyzer(),
        )
    )


def import_match(client: TestClient, tmp_path: Path) -> dict[str, object]:
    video = tmp_path / "match.mp4"
    video.touch()
    response = client.post(
        "/matches/import", json={"title": "Local vs Visitante", "video_path": str(video)}
    )
    assert response.status_code == 201
    return response.json()


def test_imports_and_lists_a_match(tmp_path: Path) -> None:
    with make_client(tmp_path) as client:
        created = import_match(client, tmp_path)
        response = client.get("/matches")
    assert response.status_code == 200
    assert response.json() == [created]


def test_serves_the_original_video(tmp_path: Path) -> None:
    with make_client(tmp_path) as client:
        match = import_match(client, tmp_path)
        response = client.get(f"/matches/{match['id']}/video")
    assert response.status_code == 200
    assert response.headers["content-disposition"].startswith("inline")


def test_creates_and_filters_manual_events(tmp_path: Path) -> None:
    with make_client(tmp_path) as client:
        match_id = import_match(client, tmp_path)["id"]
        corner = client.post(
            f"/matches/{match_id}/events",
            json={"type": "corner", "start_seconds": 10, "peak_seconds": 15, "end_seconds": 22},
        )
        client.post(
            f"/matches/{match_id}/events",
            json={"type": "foul", "start_seconds": 30, "peak_seconds": 31, "end_seconds": 35},
        )
        filtered = client.get(f"/matches/{match_id}/events", params={"type": "corner"})
    assert corner.status_code == 201
    assert corner.json()["source"] == "manual"
    assert [event["type"] for event in filtered.json()] == ["corner"]


def test_rejects_an_event_outside_the_video(tmp_path: Path) -> None:
    with make_client(tmp_path) as client:
        match_id = import_match(client, tmp_path)["id"]
        response = client.post(
            f"/matches/{match_id}/events",
            json={"type": "goal", "start_seconds": 85, "peak_seconds": 89, "end_seconds": 95},
        )
    assert response.status_code == 422
    assert response.json()["detail"] == "Event exceeds video duration"


def test_exports_the_event_interval_as_a_clip(tmp_path: Path) -> None:
    with make_client(tmp_path) as client:
        match_id = import_match(client, tmp_path)["id"]
        event = client.post(
            f"/matches/{match_id}/events",
            json={"type": "corner", "start_seconds": 10, "peak_seconds": 15, "end_seconds": 25},
        ).json()
        response = client.post(f"/events/{event['id']}/clip")

    assert response.status_code == 200
    assert response.headers["content-type"] == "video/mp4"
    assert response.headers["content-disposition"].startswith("attachment")
    assert response.content.endswith(b":10.0:25.0")


def test_runs_visual_analysis_in_the_background(tmp_path: Path) -> None:
    with make_client(tmp_path) as client:
        match_id = import_match(client, tmp_path)["id"]
        started = client.post(f"/matches/{match_id}/analysis")
        assert started.status_code == 202
        job_id = started.json()["id"]

        job = started.json()
        for _ in range(20):
            job = client.get(f"/analysis/{job_id}").json()
            if job["status"] == "completed":
                break
            time.sleep(0.01)

        signals = client.get(f"/matches/{match_id}/signals")
        events = client.get(f"/matches/{match_id}/events")

    assert job["status"] == "completed"
    assert job["progress"] == 1
    assert job["samples_processed"] == 1
    assert signals.json()[0]["likely_field"] is True
    assert events.json()[0]["type"] == "corner"
    assert events.json()[0]["source"] == "detector"


def test_confirms_and_rejects_detector_candidates(tmp_path: Path) -> None:
    with make_client(tmp_path) as client:
        match_id = import_match(client, tmp_path)["id"]
        created = client.post(
            f"/matches/{match_id}/events",
            json={
                "type": "corner",
                "start_seconds": 10,
                "peak_seconds": 15,
                "end_seconds": 22,
                "source": "detector",
                "confidence": 0.7,
            },
        ).json()
        confirmed = client.patch(
            f"/events/{created['id']}/review", json={"review_status": "confirmed"}
        )

    assert confirmed.status_code == 200
    assert confirmed.json()["review_status"] == "confirmed"
