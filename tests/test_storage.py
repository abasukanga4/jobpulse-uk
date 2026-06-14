"""Tests for the DuckDB-backed JobStore."""

from __future__ import annotations

from typing import TYPE_CHECKING

from jobpulse.extract.skills import KeywordExtractor

if TYPE_CHECKING:
    from jobpulse.models import Job
    from jobpulse.storage import JobStore


def test_upsert_jobs_is_idempotent(store: JobStore, sample_job: Job) -> None:
    ins1, dup1 = store.upsert_jobs([sample_job])
    ins2, dup2 = store.upsert_jobs([sample_job])
    assert (ins1, dup1) == (1, 0)
    assert (ins2, dup2) == (0, 1)
    assert store.count_jobs() == 1


def test_jobs_without_skills(store: JobStore, sample_job: Job) -> None:
    store.upsert_jobs([sample_job])
    pending = store.jobs_without_skills()
    assert len(pending) == 1
    assert pending[0].source_id == sample_job.source_id

    skills = KeywordExtractor().extract(sample_job)
    store.upsert_skills([skills])
    assert store.jobs_without_skills() == []


def test_joined_df_contains_skill_columns(store: JobStore, sample_job: Job) -> None:
    store.upsert_jobs([sample_job])
    store.upsert_skills([KeywordExtractor().extract(sample_job)])
    df = store.joined_df()
    assert {"technologies", "frameworks", "cloud", "domains", "seniority"} <= set(df.columns)
    assert len(df) == 1
