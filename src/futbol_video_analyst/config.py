from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    app_env: str = "development"
    media_dir: Path = Path("data/media")
    clips_dir: Path = Path("data/clips")
    datasets_dir: Path = Path("data/datasets")
    database_path: Path = Path("data/futbol-video-analyst.sqlite3")
    corner_model_enabled: bool = False
    corner_model_path: Path = Path("models/corner-spotter-v005.pt")
    corner_model_device: str = "auto"
    corner_temporal_model_enabled: bool = False
    corner_temporal_model_path: Path = Path("models/corner-temporal-v001.json")
    research_assisted_enabled: bool = False
    research_model_path: Path = Path("models/soccernet-research-temporal-v005.pt")
    research_model_device: str = "auto"
    research_model_threshold: float = 0.93
    research_cache_dir: Path = Path("data/training_cache/local-research-v005")

    model_config = SettingsConfigDict(env_file=".env", extra="ignore")


settings = Settings()
