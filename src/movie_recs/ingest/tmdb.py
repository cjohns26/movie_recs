"""Rate-limited, cache-aware TMDB API client.

Caches raw TMDB payloads via any object satisfying `TmdbCache` (in
production, the Mongo `tmdb_cache` collection — see
`movie_recs.db.tmdb_cache.MongoTmdbCache`) so re-running ingestion doesn't
re-hit the API for movies already fetched.
"""

import logging
import time
from typing import Any, Protocol

import httpx

logger = logging.getLogger(__name__)

_BASE_URL = "https://api.themoviedb.org/3"


class TmdbCache(Protocol):
    """Cache interface `TmdbClient` needs; the Mongo collection satisfies it."""

    def get(self, tmdb_id: int) -> dict[str, Any] | None: ...
    def set(self, tmdb_id: int, payload: dict[str, Any]) -> None: ...


class TmdbClient:
    """Fetches movie metadata from TMDB, rate-limited and cache-backed."""

    def __init__(
        self,
        api_key: str,
        cache: TmdbCache,
        *,
        requests_per_second: float = 4.0,
        client: httpx.Client | None = None,
    ) -> None:
        self._api_key = api_key
        self._cache = cache
        self._min_interval = 1.0 / requests_per_second
        self._client = client or httpx.Client(base_url=_BASE_URL, timeout=10.0)
        self._last_request_at = 0.0

    def get_movie(self, tmdb_id: int) -> dict[str, Any] | None:
        """Return the TMDB movie payload, from cache if already fetched.

        Returns `None` (rather than raising) on a 404 — a stale/removed
        TMDB id is tolerated, not fatal, so ingestion keeps going.
        """
        cached = self._cache.get(tmdb_id)
        if cached is not None:
            return cached

        self._throttle()
        response = self._client.get(f"/movie/{tmdb_id}", params={"api_key": self._api_key})
        if response.status_code == 404:
            logger.warning("TMDB movie %s not found (404)", tmdb_id)
            return None
        response.raise_for_status()

        payload: dict[str, Any] = response.json()
        self._cache.set(tmdb_id, payload)
        return payload

    def _throttle(self) -> None:
        elapsed = time.monotonic() - self._last_request_at
        if elapsed < self._min_interval:
            time.sleep(self._min_interval - elapsed)
        self._last_request_at = time.monotonic()

    def close(self) -> None:
        self._client.close()

    def __enter__(self) -> "TmdbClient":
        return self

    def __exit__(self, *exc_info: object) -> None:
        self.close()
