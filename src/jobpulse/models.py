"""Canonical data models. These are the contracts every layer agrees on."""

from __future__ import annotations

from datetime import date, datetime
from enum import StrEnum

from pydantic import BaseModel, ConfigDict, Field, HttpUrl


class Source(StrEnum):
    ADZUNA = "adzuna"
    MOCK = "mock"


class WorkplaceType(StrEnum):
    REMOTE = "remote"
    HYBRID = "hybrid"
    ONSITE = "onsite"
    UNKNOWN = "unknown"


class Seniority(StrEnum):
    JUNIOR = "junior"
    MID = "mid"
    SENIOR = "senior"
    LEAD = "lead"
    PRINCIPAL = "principal"
    UNKNOWN = "unknown"


class Job(BaseModel):
    """A normalised job posting. All ingest sources flatten to this shape."""

    model_config = ConfigDict(frozen=False, extra="forbid")

    source: Source
    source_id: str = Field(..., description="Stable identifier from the source.")
    title: str
    company: str | None = None
    location: str | None = None
    region: str | None = Field(None, description="UK region, e.g. London, Manchester.")
    workplace_type: WorkplaceType = WorkplaceType.UNKNOWN
    description: str = ""
    url: HttpUrl
    salary_min: float | None = None
    salary_max: float | None = None
    salary_currency: str | None = "GBP"
    posted_at: datetime | None = None
    ingested_at: datetime = Field(default_factory=datetime.utcnow)

    @property
    def salary_mid(self) -> float | None:
        if self.salary_min is not None and self.salary_max is not None:
            return (self.salary_min + self.salary_max) / 2
        return self.salary_min or self.salary_max


class ExtractedSkills(BaseModel):
    """Output of the skill-extraction stage."""

    model_config = ConfigDict(extra="forbid")

    source: Source
    source_id: str
    technologies: list[str] = Field(default_factory=list)
    frameworks: list[str] = Field(default_factory=list)
    cloud: list[str] = Field(default_factory=list)
    domains: list[str] = Field(default_factory=list)
    seniority: Seniority = Seniority.UNKNOWN
    years_experience: int | None = None
    extracted_at: datetime = Field(default_factory=datetime.utcnow)


class JobWithSkills(BaseModel):
    """Convenience wrapper for the joined view used by ML and the dashboard."""

    job: Job
    skills: ExtractedSkills


class IngestStats(BaseModel):
    """Returned by every ingest run for observability."""

    source: Source
    run_date: date
    fetched: int
    inserted: int
    duplicates: int
    duration_seconds: float
