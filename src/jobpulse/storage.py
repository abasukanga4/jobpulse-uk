"""DuckDB-backed storage for jobs and extracted skills.

DuckDB was chosen over Postgres because it's embedded (no server), file-backed
(committable fixtures), and reads Parquet natively — the same engine powers
ingest, the analytics notebooks, and the dashboard.
"""

from __future__ import annotations

from contextlib import contextmanager
from typing import TYPE_CHECKING

import duckdb

from jobpulse.config import get_settings
from jobpulse.models import ExtractedSkills, Job

if TYPE_CHECKING:
    from collections.abc import Iterable, Iterator
    from pathlib import Path

    import pandas as pd


_SCHEMA = """
CREATE TABLE IF NOT EXISTS jobs (
    source              TEXT NOT NULL,
    source_id           TEXT NOT NULL,
    title               TEXT NOT NULL,
    company             TEXT,
    location            TEXT,
    region              TEXT,
    workplace_type      TEXT NOT NULL DEFAULT 'unknown',
    description         TEXT NOT NULL DEFAULT '',
    url                 TEXT NOT NULL,
    salary_min          DOUBLE,
    salary_max          DOUBLE,
    salary_currency     TEXT DEFAULT 'GBP',
    posted_at           TIMESTAMP,
    ingested_at         TIMESTAMP NOT NULL,
    PRIMARY KEY (source, source_id)
);

CREATE INDEX IF NOT EXISTS idx_jobs_ingested_at ON jobs(ingested_at);
CREATE INDEX IF NOT EXISTS idx_jobs_region      ON jobs(region);

CREATE TABLE IF NOT EXISTS extracted_skills (
    source              TEXT NOT NULL,
    source_id           TEXT NOT NULL,
    technologies        TEXT[],
    frameworks          TEXT[],
    cloud               TEXT[],
    domains             TEXT[],
    seniority           TEXT NOT NULL DEFAULT 'unknown',
    years_experience    INTEGER,
    extracted_at        TIMESTAMP NOT NULL,
    PRIMARY KEY (source, source_id)
);
"""


class JobStore:
    """Thin wrapper around a DuckDB file. Cheap to construct."""

    def __init__(self, db_path: Path | None = None) -> None:
        self.db_path = db_path or get_settings().db_path
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        with self._connect() as con:
            con.execute(_SCHEMA)

    @contextmanager
    def _connect(self) -> Iterator[duckdb.DuckDBPyConnection]:
        con = duckdb.connect(str(self.db_path))
        try:
            yield con
        finally:
            con.close()

    @staticmethod
    def _scalar_count(con: duckdb.DuckDBPyConnection) -> int:
        row = con.execute("SELECT COUNT(*) FROM jobs").fetchone()
        return int(row[0]) if row is not None else 0

    # ---- writers --------------------------------------------------------

    def upsert_jobs(self, jobs: Iterable[Job]) -> tuple[int, int]:
        """Insert jobs, skipping duplicates on (source, source_id).

        Returns (inserted, duplicates).
        """
        rows = [self._job_to_row(j) for j in jobs]
        if not rows:
            return 0, 0
        with self._connect() as con:
            before = self._scalar_count(con)
            con.executemany(
                """
                INSERT INTO jobs (
                    source, source_id, title, company, location, region,
                    workplace_type, description, url,
                    salary_min, salary_max, salary_currency,
                    posted_at, ingested_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT (source, source_id) DO NOTHING
                """,
                rows,
            )
            after = self._scalar_count(con)
        inserted = after - before
        return inserted, len(rows) - inserted

    def upsert_skills(self, skills: Iterable[ExtractedSkills]) -> int:
        rows = [self._skills_to_row(s) for s in skills]
        if not rows:
            return 0
        with self._connect() as con:
            con.executemany(
                """
                INSERT INTO extracted_skills (
                    source, source_id, technologies, frameworks, cloud,
                    domains, seniority, years_experience, extracted_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT (source, source_id) DO UPDATE SET
                    technologies     = EXCLUDED.technologies,
                    frameworks       = EXCLUDED.frameworks,
                    cloud            = EXCLUDED.cloud,
                    domains          = EXCLUDED.domains,
                    seniority        = EXCLUDED.seniority,
                    years_experience = EXCLUDED.years_experience,
                    extracted_at     = EXCLUDED.extracted_at
                """,
                rows,
            )
        return len(rows)

    # ---- readers --------------------------------------------------------

    def count_jobs(self) -> int:
        with self._connect() as con:
            return self._scalar_count(con)

    def jobs_without_skills(self, limit: int | None = None) -> list[Job]:
        sql = """
            SELECT j.* FROM jobs j
            LEFT JOIN extracted_skills s
              ON s.source = j.source AND s.source_id = j.source_id
            WHERE s.source_id IS NULL
        """
        if limit is not None:
            sql += f" LIMIT {int(limit)}"
        with self._connect() as con:
            rows = con.execute(sql).fetchall()
            cols = [d[0] for d in con.description]
        return [Job.model_validate(dict(zip(cols, r, strict=True))) for r in rows]

    def jobs_df(self) -> pd.DataFrame:
        """Return all jobs as a pandas DataFrame. Used by the dashboard."""
        with self._connect() as con:
            return con.execute("SELECT * FROM jobs ORDER BY ingested_at DESC").df()

    def joined_df(self) -> pd.DataFrame:
        """Jobs joined with skills — the main analytics view."""
        with self._connect() as con:
            return con.execute(
                """
                SELECT j.*, s.technologies, s.frameworks, s.cloud,
                       s.domains, s.seniority, s.years_experience
                FROM jobs j
                LEFT JOIN extracted_skills s
                  ON s.source = j.source AND s.source_id = j.source_id
                """
            ).df()

    # ---- row conversion -------------------------------------------------

    @staticmethod
    def _job_to_row(j: Job) -> tuple[object, ...]:
        return (
            j.source.value,
            j.source_id,
            j.title,
            j.company,
            j.location,
            j.region,
            j.workplace_type.value,
            j.description,
            str(j.url),
            j.salary_min,
            j.salary_max,
            j.salary_currency,
            j.posted_at,
            j.ingested_at,
        )

    @staticmethod
    def _skills_to_row(s: ExtractedSkills) -> tuple[object, ...]:
        return (
            s.source.value,
            s.source_id,
            s.technologies,
            s.frameworks,
            s.cloud,
            s.domains,
            s.seniority.value,
            s.years_experience,
            s.extracted_at,
        )
