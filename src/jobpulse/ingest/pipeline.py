"""Source-agnostic ingest orchestrator. The CLI is a thin wrapper around this."""

from __future__ import annotations

import asyncio
import time
from datetime import UTC, datetime
from typing import TYPE_CHECKING

import pandas as pd
import structlog

from jobpulse.ingest.adzuna import AdzunaClient
from jobpulse.ingest.mock import generate_jobs
from jobpulse.models import IngestStats, Job, Source, WorkplaceType

if TYPE_CHECKING:
    from pathlib import Path

    from jobpulse.storage import JobStore

logger = structlog.get_logger()


def _from_fixture(path: Path) -> list[Job]:
    """Load jobs from a Parquet fixture (used when running the demo offline)."""
    df = pd.read_parquet(path)
    jobs: list[Job] = []
    for row in df.to_dict(orient="records"):
        row["source"] = Source(row["source"])
        row["workplace_type"] = WorkplaceType(row["workplace_type"])
        jobs.append(Job.model_validate(row))
    return jobs


def run_ingest(
    source: Source,
    store: JobStore,
    *,
    mock_n: int = 200,
    adzuna_pages_per_query: int = 1,
    fixture_path: Path | None = None,
) -> IngestStats:
    """Fetch from `source`, persist, return stats. Synchronous facade."""
    start = time.perf_counter()

    if source is Source.MOCK:
        if fixture_path is not None and fixture_path.exists():
            jobs = _from_fixture(fixture_path)
        else:
            jobs = generate_jobs(mock_n)
    elif source is Source.ADZUNA:
        jobs = asyncio.run(_fetch_adzuna(pages=adzuna_pages_per_query))
    else:  # pragma: no cover — exhaustiveness guard
        raise ValueError(f"unsupported source: {source!r}")

    inserted, duplicates = store.upsert_jobs(jobs)
    duration = time.perf_counter() - start

    stats = IngestStats(
        source=source,
        run_date=datetime.now(UTC).date(),
        fetched=len(jobs),
        inserted=inserted,
        duplicates=duplicates,
        duration_seconds=round(duration, 2),
    )
    logger.info("ingest.done", **stats.model_dump(mode="json"))
    return stats


async def _fetch_adzuna(pages: int) -> list[Job]:
    async with AdzunaClient() as client:
        return await client.search(pages_per_query=pages)
