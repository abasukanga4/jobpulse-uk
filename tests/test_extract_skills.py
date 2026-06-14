"""Tests for the keyword-based skill extractor."""

from __future__ import annotations

from jobpulse.extract.skills import KeywordExtractor
from jobpulse.models import Job, Seniority, Source


def _job(title: str, desc: str) -> Job:
    return Job(
        source=Source.MOCK,
        source_id="x",
        title=title,
        url="https://example.com/x",  # type: ignore[arg-type]
        description=desc,
    )


def test_keyword_extractor_finds_basics(sample_job: Job) -> None:
    skills = KeywordExtractor().extract(sample_job)
    assert "python" in skills.technologies
    assert "pytorch" in skills.frameworks
    assert "aws" in skills.cloud
    assert "mlops" in skills.domains
    assert "llm" in skills.domains
    assert "rag" in skills.domains
    assert skills.seniority is Seniority.SENIOR
    assert skills.years_experience == 5


def test_seniority_from_title_principal() -> None:
    skills = KeywordExtractor().extract(_job("Principal Data Scientist", "x"))
    assert skills.seniority is Seniority.PRINCIPAL


def test_seniority_from_title_junior() -> None:
    skills = KeywordExtractor().extract(_job("Junior Data Analyst", "x"))
    assert skills.seniority is Seniority.JUNIOR


def test_seniority_default_mid_when_unspecified() -> None:
    skills = KeywordExtractor().extract(_job("Data Engineer", "x"))
    assert skills.seniority is Seniority.MID


def test_no_false_positive_on_go_inside_google() -> None:
    skills = KeywordExtractor().extract(_job("Data Engineer", "We use Google products."))
    assert "go" not in skills.technologies


def test_years_experience_picks_max() -> None:
    skills = KeywordExtractor().extract(
        _job("ML Engineer", "3+ years required. 5 years preferred.")
    )
    assert skills.years_experience == 5
