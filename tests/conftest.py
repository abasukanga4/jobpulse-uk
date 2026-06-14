"""Shared pytest fixtures."""

from __future__ import annotations

from datetime import UTC, datetime
from typing import TYPE_CHECKING

import pytest

from jobpulse.models import Job, Source, WorkplaceType
from jobpulse.storage import JobStore

if TYPE_CHECKING:
    from pathlib import Path


@pytest.fixture
def sample_job() -> Job:
    return Job(
        source=Source.MOCK,
        source_id="abc123",
        title="Senior Machine Learning Engineer",
        company="Acme AI",
        location="London, UK",
        region="London",
        workplace_type=WorkplaceType.HYBRID,
        description=(
            "We are hiring a Senior ML Engineer. Strong Python, PyTorch, "
            "AWS (SageMaker, S3), MLOps experience required. 5+ years of "
            "production ML. Bonus: experience with LLMs and RAG."
        ),
        url="https://example.com/jobs/abc123",  # type: ignore[arg-type]
        salary_min=90_000,
        salary_max=120_000,
        salary_currency="GBP",
        posted_at=datetime(2026, 5, 1, tzinfo=UTC).replace(tzinfo=None),
    )


@pytest.fixture
def store(tmp_path: Path) -> JobStore:
    return JobStore(tmp_path / "test.duckdb")
