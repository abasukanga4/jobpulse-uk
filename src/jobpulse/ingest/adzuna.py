"""Adzuna UK API client.

Adzuna's free tier gives 250 calls/month which is enough for ~12k jobs/month
at 50 results per page — well above what this project ingests.

API ref: https://developer.adzuna.com/docs/search
"""

from __future__ import annotations

import asyncio
from datetime import datetime
from typing import TYPE_CHECKING, Any, cast

import httpx
import structlog
from tenacity import (
    retry,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential,
)

from jobpulse.config import Settings, get_settings
from jobpulse.models import Job, Source, WorkplaceType

if TYPE_CHECKING:
    from collections.abc import Iterable

logger = structlog.get_logger()

_BASE_URL = "https://api.adzuna.com/v1/api/jobs/gb/search/{page}"
_MAX_RESULTS_PER_PAGE = 50
_DEFAULT_QUERIES = (
    "data scientist",
    "machine learning engineer",
    "data engineer",
    "ai engineer",
    "analytics engineer",
)


class AdzunaError(Exception):
    """Wraps non-retryable Adzuna API failures."""


class AdzunaClient:
    """Async client. Use with `async with` to share a connection pool."""

    def __init__(self, settings: Settings | None = None) -> None:
        self.settings = settings or get_settings()
        if not self.settings.has_adzuna:
            raise AdzunaError("ADZUNA_APP_ID and ADZUNA_APP_KEY must be set. See .env.example.")
        self._client: httpx.AsyncClient | None = None

    async def __aenter__(self) -> AdzunaClient:
        self._client = httpx.AsyncClient(timeout=30.0)
        return self

    async def __aexit__(self, *_exc: object) -> None:
        if self._client is not None:
            await self._client.aclose()
            self._client = None

    @retry(
        retry=retry_if_exception_type((httpx.TimeoutException, httpx.NetworkError)),
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=1, max=10),
        reraise=True,
    )
    async def _fetch_page(self, query: str, page: int) -> dict[str, Any]:
        assert self._client is not None, "use AdzunaClient as an async context manager"
        url = _BASE_URL.format(page=page)
        params = {
            "app_id": self.settings.adzuna_app_id,
            "app_key": self.settings.adzuna_app_key,
            "what": query,
            "where": "uk",
            "results_per_page": _MAX_RESULTS_PER_PAGE,
            "content-type": "application/json",
        }
        logger.info("adzuna.fetch", query=query, page=page)
        resp = await self._client.get(url, params=params)
        if resp.status_code == 401:
            raise AdzunaError("Adzuna rejected credentials (401).")
        if resp.status_code == 429:
            raise AdzunaError("Adzuna rate limit hit (429).")
        resp.raise_for_status()
        return cast("dict[str, Any]", resp.json())

    async def search(
        self,
        queries: Iterable[str] = _DEFAULT_QUERIES,
        *,
        pages_per_query: int = 1,
    ) -> list[Job]:
        """Search Adzuna for each query and return normalised Jobs.

        Note: results across queries can overlap. The storage layer dedupes
        on (source, source_id), so we don't need to dedupe here.
        """
        jobs: list[Job] = []
        for query in queries:
            for page in range(1, pages_per_query + 1):
                payload = await self._fetch_page(query, page)
                for raw in payload.get("results", []):
                    jobs.append(_to_job(raw))
                # Free-tier-friendly: yield to the loop between calls
                await asyncio.sleep(0.2)
        logger.info("adzuna.search.done", jobs=len(jobs))
        return jobs


def _to_job(raw: dict[str, Any]) -> Job:
    """Normalise one Adzuna result to our canonical Job."""
    region = _extract_region(raw)
    posted = raw.get("created")
    posted_at = datetime.fromisoformat(posted.replace("Z", "+00:00")) if posted else None
    return Job(
        source=Source.ADZUNA,
        source_id=str(raw["id"]),
        title=raw.get("title", "").strip(),
        company=(raw.get("company") or {}).get("display_name"),
        location=(raw.get("location") or {}).get("display_name"),
        region=region,
        workplace_type=_infer_workplace(raw),
        description=raw.get("description", "") or "",
        url=raw.get("redirect_url", "https://adzuna.co.uk"),
        salary_min=raw.get("salary_min"),
        salary_max=raw.get("salary_max"),
        salary_currency="GBP",
        posted_at=posted_at.replace(tzinfo=None) if posted_at else None,
    )


def _extract_region(raw: dict[str, Any]) -> str | None:
    """Adzuna nests location as `area: [Country, Region, City, ...]`. Pick the most useful."""
    location = raw.get("location") or {}
    area = location.get("area") or []
    # Drop "UK" / "United Kingdom" prefix; prefer the city when present.
    cleaned = [str(a) for a in area if str(a).lower() not in {"uk", "united kingdom"}]
    if not cleaned:
        return None
    # Last element is the most specific (city); fall back to region.
    return cleaned[-1] if len(cleaned) >= 2 else cleaned[0]


_REMOTE_HINTS = ("remote", "work from home", "fully remote", "wfh")
_HYBRID_HINTS = ("hybrid",)


def _infer_workplace(raw: dict[str, Any]) -> WorkplaceType:
    """Adzuna doesn't expose workplace type explicitly — infer from text."""
    haystack = (
        (raw.get("title") or "")
        + " "
        + (raw.get("description") or "")
        + " "
        + ((raw.get("location") or {}).get("display_name") or "")
    ).lower()
    if any(h in haystack for h in _REMOTE_HINTS):
        return WorkplaceType.REMOTE
    if any(h in haystack for h in _HYBRID_HINTS):
        return WorkplaceType.HYBRID
    return WorkplaceType.UNKNOWN
