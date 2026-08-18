import sqlite3
from pathlib import Path

from futbol_video_analyst.database import Database


def test_initialize_migrates_existing_visual_signals_table(tmp_path: Path) -> None:
    path = tmp_path / "existing.sqlite3"
    with sqlite3.connect(path) as connection:
        connection.execute(
            """
            CREATE TABLE visual_signals (
                id TEXT PRIMARY KEY,
                match_id TEXT NOT NULL,
                timestamp_seconds REAL NOT NULL,
                green_ratio REAL NOT NULL,
                brightness REAL NOT NULL,
                change_score REAL NOT NULL,
                likely_field INTEGER NOT NULL
            )
            """
        )

    Database(path).initialize()

    with sqlite3.connect(path) as connection:
        columns = {row[1] for row in connection.execute("PRAGMA table_info(visual_signals)")}
    assert {"player_candidates", "ball_candidates", "line_ratio"} <= columns
