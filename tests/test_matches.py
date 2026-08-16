from pathlib import Path

from fastapi.testclient import TestClient

from futbol_video_analyst.domain import VideoMetadata
from futbol_video_analyst.main import create_app


class FakeVideoInspector:
    def inspect(self, path: Path) -> VideoMetadata:
        return VideoMetadata(duration_seconds=90, width=1920, height=1080, fps=30, codec="h264")


def make_client(tmp_path: Path) -> TestClient:
    return TestClient(create_app(tmp_path / "test.sqlite3", FakeVideoInspector()))


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
