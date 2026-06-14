"""Tests for the deterministic mock generator."""

from __future__ import annotations

from jobpulse.ingest.mock import generate_jobs
from jobpulse.models import Source


def test_generate_jobs_is_deterministic() -> None:
    a = generate_jobs(50, seed=7)
    b = generate_jobs(50, seed=7)
    assert [j.source_id for j in a] == [j.source_id for j in b]


def test_generate_jobs_count() -> None:
    jobs = generate_jobs(123, seed=1)
    assert len(jobs) == 123
    assert all(j.source is Source.MOCK for j in jobs)


def test_generate_jobs_salary_bands_are_realistic() -> None:
    jobs = generate_jobs(200, seed=1)
    assert all(j.salary_min is not None and j.salary_max is not None for j in jobs)
    assert all(j.salary_min <= j.salary_max for j in jobs)  # type: ignore[operator]
    # 2026 UK floor: salaries shouldn't dip under £20k for tech roles
    assert min(j.salary_min for j in jobs) >= 20_000  # type: ignore[type-var]
