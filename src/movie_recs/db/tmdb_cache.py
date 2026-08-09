"""Mongo-backed implementation of `movie_recs.ingest.tmdb.TmdbCache`."""

from datetime import UTC, datetime
from typing import Any

from movie_recs.db.collections import tmdb_cache_collection


class MongoTmdbCache:
    """Reads/writes the `tmdb_cache` collection, keyed by `tmdbId`."""

    def get(self, tmdb_id: int) -> dict[str, Any] | None:
        doc = tmdb_cache_collection().find_one({"tmdbId": tmdb_id})
        return doc["payload"] if doc else None

    def set(self, tmdb_id: int, payload: dict[str, Any]) -> None:
        tmdb_cache_collection().update_one(
            {"tmdbId": tmdb_id},
            {"$set": {"payload": payload, "fetched_at": datetime.now(UTC)}},
            upsert=True,
        )
