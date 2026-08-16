from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    app_env: str = "development"
    media_dir: Path = Path("data/media")
    clips_dir: Path = Path("data/clips")

    model_config = SettingsConfigDict(env_file=".env", extra="ignore")


settings = Settings()
