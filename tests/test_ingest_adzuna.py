"""Tests for the Adzuna normaliser (pure functions, no network)."""

from __future__ import annotations

from jobpulse.ingest.adzuna import _extract_region, _infer_workplace, _to_job
from jobpulse.models import Source, WorkplaceType

_ADZUNA_SAMPLE = {
    "id": "999111",
    "title": "Senior Machine Learning Engineer",
    "company": {"display_name": "Acme AI"},
    "location": {
        "display_name": "London",
        "area": ["UK", "London", "Camden"],
    },
    "description": "Hybrid working. Looking for PyTorch experience.",
    "redirect_url": "https://www.adzuna.co.uk/jobs/999111",
    "salary_min": 90000,
    "salary_max": 120000,
    "created": "2026-05-10T09:00:00Z",
}


def test_to_job_maps_basic_fields() -> None:
    job = _to_job(_ADZUNA_SAMPLE)
    assert job.source is Source.ADZUNA
    assert job.source_id == "999111"
    assert job.title == "Senior Machine Learning Engineer"
    assert job.company == "Acme AI"
    assert job.salary_min == 90_000
    assert job.salary_max == 120_000
    assert str(job.url).startswith("https://www.adzuna.co.uk")


def test_extract_region_prefers_city_over_country() -> None:
    assert _extract_region(_ADZUNA_SAMPLE) == "Camden"


def test_extract_region_handles_missing_area() -> None:
    raw = {"location": {"display_name": "Remote"}}
    assert _extract_region(raw) is None


def test_infer_workplace_remote() -> None:
    raw = {**_ADZUNA_SAMPLE, "description": "Fully remote position."}
    assert _infer_workplace(raw) is WorkplaceType.REMOTE


def test_infer_workplace_hybrid() -> None:
    assert _infer_workplace(_ADZUNA_SAMPLE) is WorkplaceType.HYBRID


def test_infer_workplace_unknown_when_no_signal() -> None:
    raw = {
        "title": "Data Engineer",
        "description": "Just a description.",
        "location": {"display_name": "Manchester"},
    }
    assert _infer_workplace(raw) is WorkplaceType.UNKNOWN
