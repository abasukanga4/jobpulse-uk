"""Skill extraction.

Two backends share the same interface:

- KeywordExtractor — fast, deterministic, no network. Used for the demo,
  in CI, and as the fallback when the LLM is unavailable.
- ClaudeExtractor  — calls Claude with structured tool-use output for richer
  extraction. Used when `ANTHROPIC_API_KEY` is set.

The pipeline picks one via `build_default_extractor()`.
"""

from __future__ import annotations

import json
import re
from typing import TYPE_CHECKING, Protocol

import structlog

from jobpulse.config import get_settings
from jobpulse.models import ExtractedSkills, Job, Seniority

if TYPE_CHECKING:
    from collections.abc import Iterable

logger = structlog.get_logger()


class SkillExtractor(Protocol):
    """A callable that maps a Job to ExtractedSkills."""

    def extract(self, job: Job) -> ExtractedSkills: ...


# ---------------------------------------------------------------------------
# Keyword backend
# ---------------------------------------------------------------------------

# Pattern -> canonical label. Patterns are matched case-insensitively as
# whole words/phrases, so "go" doesn't match "google".
_TECH = {
    r"\bpython\b": "python",
    r"\bsql\b": "sql",
    r"\br\b": "r",
    r"\bscala\b": "scala",
    r"\bjava\b": "java",
    r"\bgolang\b|\bgo language\b": "go",
    r"\brust\b": "rust",
    r"\btypescript\b": "typescript",
}

_FRAMEWORKS = {
    r"\bpytorch\b": "pytorch",
    r"\btensorflow\b|\bkeras\b": "tensorflow",
    r"\bscikit[- ]learn\b|\bsklearn\b": "scikit-learn",
    r"\bxgboost\b|\blightgbm\b|\bcatboost\b": "xgboost",
    r"\bhugging[- ]?face\b|\btransformers\b": "huggingface",
    r"\blangchain\b|\bllama[- ]?index\b": "langchain",
    r"\bspark\b|\bpyspark\b": "spark",
    r"\bdbt\b": "dbt",
    r"\bairflow\b|\bdagster\b|\bprefect\b": "airflow",
    r"\bfastapi\b": "fastapi",
    r"\bstreamlit\b": "streamlit",
}

_CLOUD = {
    r"\baws\b|amazon web services|sagemaker|\bs3\b": "aws",
    r"\bgcp\b|google cloud|bigquery|vertex ai|dataflow": "gcp",
    r"\bazure\b|synapse|data factory": "azure",
    r"\bsnowflake\b": "snowflake",
    r"\bdatabricks\b": "databricks",
    r"\bkubernetes\b|\bk8s\b": "kubernetes",
    r"\bdocker\b": "docker",
    r"\bterraform\b": "terraform",
}

_DOMAINS = {
    r"\bllm\b|\bllms\b|large language model|generative ai|\bgenai\b": "llm",
    r"\brag\b|retrieval augmented": "rag",
    r"\bagentic\b|\bagents?\b|tool use|function calling": "agents",
    r"\bmlops\b|model monitoring|drift detection": "mlops",
    r"\bcausal\b|\ba/b test\b|experimentation": "experimentation",
    r"\bnlp\b|natural language processing": "nlp",
    r"\bcomputer vision\b|\bcv\b": "computer-vision",
    r"\bforecast(ing)?\b|time[- ]series": "forecasting",
    r"\brecommend(er|ation)?\b": "recommenders",
    r"\bfraud\b|anti[- ]?money[- ]?laundering|\baml\b": "fraud",
}

_SENIORITY_TITLE_PATTERNS = (
    (re.compile(r"\bprincipal\b", re.I), Seniority.PRINCIPAL),
    (re.compile(r"\blead|head of|staff\b", re.I), Seniority.LEAD),
    (re.compile(r"\bsenior|snr|sr\.\b", re.I), Seniority.SENIOR),
    (re.compile(r"\bjunior|jr\.|graduate|entry[- ]level\b", re.I), Seniority.JUNIOR),
)
_YEARS_PATTERN = re.compile(r"(\d+)\+?\s*(?:years|yrs)", re.I)


def _match_dict(text: str, patterns: dict[str, str]) -> list[str]:
    matches: set[str] = set()
    for pat, label in patterns.items():
        if re.search(pat, text, flags=re.IGNORECASE):
            matches.add(label)
    return sorted(matches)


def _infer_seniority(job: Job) -> Seniority:
    for pat, level in _SENIORITY_TITLE_PATTERNS:
        if pat.search(job.title):
            return level
    return Seniority.MID  # safest default for ad text that omits the level


def _infer_years(text: str) -> int | None:
    candidates = [int(m.group(1)) for m in _YEARS_PATTERN.finditer(text)]
    return max(candidates) if candidates else None


class KeywordExtractor:
    """Regex-based extractor. No network, deterministic, fast."""

    def extract(self, job: Job) -> ExtractedSkills:
        text = f"{job.title}\n{job.description}"
        return ExtractedSkills(
            source=job.source,
            source_id=job.source_id,
            technologies=_match_dict(text, _TECH),
            frameworks=_match_dict(text, _FRAMEWORKS),
            cloud=_match_dict(text, _CLOUD),
            domains=_match_dict(text, _DOMAINS),
            seniority=_infer_seniority(job),
            years_experience=_infer_years(text),
        )


# ---------------------------------------------------------------------------
# Claude backend
# ---------------------------------------------------------------------------

_CLAUDE_SYSTEM = (
    "You are a precise information extractor for UK tech job postings. "
    "Return only what is explicitly supported by the text. Never invent skills."
)

_CLAUDE_TOOL = {
    "name": "record_skills",
    "description": "Record the structured skills extracted from a job posting.",
    "input_schema": {
        "type": "object",
        "required": ["technologies", "frameworks", "cloud", "domains", "seniority"],
        "properties": {
            "technologies": {
                "type": "array",
                "items": {"type": "string"},
                "description": "Programming languages explicitly required (lowercase, "
                "e.g. python, sql, rust).",
            },
            "frameworks": {
                "type": "array",
                "items": {"type": "string"},
                "description": "ML/data frameworks explicitly required "
                "(e.g. pytorch, scikit-learn, dbt).",
            },
            "cloud": {
                "type": "array",
                "items": {"type": "string"},
                "description": "Cloud platforms or infrastructure tools "
                "(e.g. aws, gcp, kubernetes).",
            },
            "domains": {
                "type": "array",
                "items": {"type": "string"},
                "description": "Problem domains (e.g. llm, rag, mlops, forecasting, fraud).",
            },
            "seniority": {
                "type": "string",
                "enum": [s.value for s in Seniority],
                "description": "Seniority band inferred from the title and required years.",
            },
            "years_experience": {
                "type": ["integer", "null"],
                "description": "Minimum years of experience if explicitly stated, else null.",
            },
        },
    },
}


class ClaudeExtractor:
    """LLM-backed extractor using Claude tool use for structured output."""

    def __init__(self, model: str = "claude-haiku-4-5-20251001") -> None:
        # Imported here so the dependency stays optional.
        from anthropic import Anthropic  # noqa: PLC0415

        self.client = Anthropic(api_key=get_settings().anthropic_api_key)
        self.model = model

    def extract(self, job: Job) -> ExtractedSkills:
        prompt = (
            f"Extract structured skills from the following job posting. "
            f"Call the `record_skills` tool exactly once.\n\n"
            f"Title: {job.title}\n"
            f"Company: {job.company or 'Unknown'}\n"
            f"Location: {job.location or 'Unknown'}\n\n"
            f"Description:\n{job.description[:6000]}"
        )
        resp = self.client.messages.create(
            model=self.model,
            max_tokens=512,
            system=_CLAUDE_SYSTEM,
            tools=[_CLAUDE_TOOL],
            tool_choice={"type": "tool", "name": "record_skills"},
            messages=[{"role": "user", "content": prompt}],
        )
        payload = _first_tool_input(resp)
        if payload is None:
            logger.warning("claude_extractor.no_tool_call", source_id=job.source_id)
            return KeywordExtractor().extract(job)
        return ExtractedSkills(
            source=job.source,
            source_id=job.source_id,
            technologies=[s.lower() for s in payload.get("technologies", [])],
            frameworks=[s.lower() for s in payload.get("frameworks", [])],
            cloud=[s.lower() for s in payload.get("cloud", [])],
            domains=[s.lower() for s in payload.get("domains", [])],
            seniority=Seniority(payload.get("seniority", "unknown")),
            years_experience=payload.get("years_experience"),
        )


def _first_tool_input(resp: object) -> dict | None:  # type: ignore[type-arg]
    """Pull the first tool_use input out of an Anthropic message response."""
    for block in getattr(resp, "content", []):
        if getattr(block, "type", None) == "tool_use":
            data = getattr(block, "input", None)
            if isinstance(data, dict):
                return data
            if isinstance(data, str):
                try:
                    loaded = json.loads(data)
                except json.JSONDecodeError:
                    return None
                return loaded if isinstance(loaded, dict) else None
    return None


# ---------------------------------------------------------------------------
# Defaults
# ---------------------------------------------------------------------------


def build_default_extractor() -> SkillExtractor:
    """Pick the best extractor we have credentials for."""
    if get_settings().has_anthropic:
        try:
            return ClaudeExtractor()
        except Exception as exc:
            logger.warning("claude_extractor.unavailable", error=str(exc))
    return KeywordExtractor()


def extract_many(
    jobs: Iterable[Job], extractor: SkillExtractor | None = None
) -> list[ExtractedSkills]:
    extractor = extractor or build_default_extractor()
    return [extractor.extract(j) for j in jobs]
