import json
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
            datasets_dir=tmp_path / "datasets",
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
    assert response.headers["x-exported-path"].endswith(".mp4")
    assert response.content.endswith(b":10.0:25.0")


def test_exports_reviewed_events_as_a_grouped_training_dataset(tmp_path: Path) -> None:
    with make_client(tmp_path) as client:
        match_id = import_match(client, tmp_path)["id"]
        client.post(
            f"/matches/{match_id}/events",
            json={"type": "corner", "start_seconds": 10, "peak_seconds": 15, "end_seconds": 22},
        )
        confirmed = client.post(
            f"/matches/{match_id}/events",
            json={
                "type": "penalty",
                "start_seconds": 30,
                "peak_seconds": 35,
                "end_seconds": 42,
                "source": "detector",
            },
        ).json()
        client.patch(
            f"/events/{confirmed['id']}/review", json={"review_status": "confirmed"}
        )
        rejected = client.post(
            f"/matches/{match_id}/events",
            json={
                "type": "corner",
                "start_seconds": 45,
                "peak_seconds": 50,
                "end_seconds": 58,
                "source": "detector",
            },
        ).json()
        client.patch(
            f"/events/{rejected['id']}/review", json={"review_status": "rejected"}
        )
        client.post(
            f"/matches/{match_id}/events",
            json={
                "type": "corner",
                "start_seconds": 60,
                "peak_seconds": 65,
                "end_seconds": 72,
                "source": "detector",
            },
        )

        started = client.post("/dataset/export")
        job = started.json()
        for _ in range(50):
            job = client.get(f"/dataset/export/{job['id']}").json()
            if job["status"] in {"completed", "failed"}:
                break
            time.sleep(0.01)

    assert started.status_code == 202
    assert job["status"] == "completed"
    assert job["progress"] == 1
    result = job["result"]
    assert result is not None
    assert result["clips"] == 3
    assert result["matches"] == 1
    assert result["skipped_events"] == 1
    assert result["label_counts"] == {"corner": 1, "negative": 1, "penalty": 1}
    records = [json.loads(line) for line in Path(result["manifest_path"]).read_text().splitlines()]
    assert {record["label"] for record in records} == {"corner", "penalty", "negative"}
    assert len({record["match_id"] for record in records}) == 1
    assert all((Path(result["path"]) / record["clip_path"]).is_file() for record in records)


def test_edits_and_deletes_an_event(tmp_path: Path) -> None:
    with make_client(tmp_path) as client:
        match_id = import_match(client, tmp_path)["id"]
        event = client.post(
            f"/matches/{match_id}/events",
            json={
                "type": "corner",
                "start_seconds": 10.25,
                "peak_seconds": 15.5,
                "end_seconds": 22.75,
                "source": "detector",
            },
        ).json()
        updated = client.patch(
            f"/events/{event['id']}",
            json={
                "type": "throw_in",
                "start_seconds": 11,
                "peak_seconds": 16,
                "end_seconds": 24,
                "notes": "Era un saque de banda, no un corner",
            },
        )
        deleted = client.delete(f"/events/{event['id']}")
        missing = client.get(f"/matches/{match_id}/events")

    assert updated.status_code == 200
    assert updated.json()["type"] == "throw_in"
    assert updated.json()["detected_type"] == "corner"
    assert updated.json()["notes"] == "Era un saque de banda, no un corner"
    assert deleted.status_code == 204
    assert missing.json() == []


def test_reclassified_confirmed_candidate_calibrates_as_rejected_corner(tmp_path: Path) -> None:
    with make_client(tmp_path) as client:
        match_id = import_match(client, tmp_path)["id"]
        job = client.post(f"/matches/{match_id}/analysis").json()
        for _ in range(20):
            if client.get(f"/analysis/{job['id']}").json()["status"] == "completed":
                break
            time.sleep(0.01)
        event = client.get(f"/matches/{match_id}/events").json()[0]
        client.patch(f"/events/{event['id']}/review", json={"review_status": "rejected"})
        reclassified = client.patch(
            f"/events/{event['id']}/reclassify",
            json={
                "type": "penalty",
                "start_seconds": 0,
                "peak_seconds": 4,
                "end_seconds": 10,
                "notes": "Penal confirmado",
            },
        )
        examples = client.app.state.database.list_corner_review_examples()

    assert reclassified.status_code == 200
    assert reclassified.json()["type"] == "penalty"
    assert reclassified.json()["review_status"] == "confirmed"
    assert reclassified.json()["detected_type"] == "corner"
    assert examples[0][1] == "rejected"


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
