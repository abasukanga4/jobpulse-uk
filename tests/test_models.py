"""Tests for the canonical data models."""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from jobpulse.models import Job, Source, WorkplaceType


def test_job_salary_mid_with_both(sample_job: Job) -> None:
    assert sample_job.salary_mid == 105_000


def test_job_salary_mid_with_only_min() -> None:
    job = Job(
        source=Source.MOCK,
        source_id="x",
        title="Data Analyst",
        url="https://example.com/x",  # type: ignore[arg-type]
        salary_min=50_000,
    )
    assert job.salary_mid == 50_000


def test_job_salary_mid_when_missing() -> None:
    job = Job(
        source=Source.MOCK,
        source_id="y",
        title="Data Analyst",
        url="https://example.com/y",  # type: ignore[arg-type]
    )
    assert job.salary_mid is None


def test_job_rejects_unknown_field() -> None:
    with pytest.raises(ValidationError):
        Job(
            source=Source.MOCK,
            source_id="z",
            title="x",
            url="https://example.com/z",  # type: ignore[arg-type]
            unknown_field="boom",  # type: ignore[call-arg]
        )


def test_workplace_type_default() -> None:
    job = Job(
        source=Source.MOCK,
        source_id="w",
        title="x",
        url="https://example.com/w",  # type: ignore[arg-type]
    )
    assert job.workplace_type is WorkplaceType.UNKNOWN
