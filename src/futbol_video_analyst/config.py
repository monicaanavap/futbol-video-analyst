from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    app_env: str = "development"
    media_dir: Path = Path("data/media")
    clips_dir: Path = Path("data/clips")
    database_path: Path = Path("data/futbol-video-analyst.sqlite3")

    model_config = SettingsConfigDict(env_file=".env", extra="ignore")


settings = Settings()
