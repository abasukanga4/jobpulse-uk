"""Runtime configuration loaded from environment variables."""

from __future__ import annotations

from pathlib import Path

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Process-wide settings. Reads from the environment and a local `.env` file."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    adzuna_app_id: str | None = None
    adzuna_app_key: str | None = None
    anthropic_api_key: str | None = None

    data_dir: Path = Field(default=Path("./data"), alias="JOBPULSE_DATA_DIR")
    db_path: Path = Field(default=Path("./data/jobpulse.duckdb"), alias="JOBPULSE_DB_PATH")

    @property
    def has_adzuna(self) -> bool:
        return bool(self.adzuna_app_id and self.adzuna_app_key)

    @property
    def has_anthropic(self) -> bool:
        return bool(self.anthropic_api_key)


def get_settings() -> Settings:
    """Return a fresh Settings instance. Cheap; call per-request if needed."""
    return Settings()
