"""Typed collection accessors. See plan.md's Data Model for the schema."""

from typing import Any

from pymongo.collection import Collection

from movie_recs.db.client import get_db


def movies_collection() -> Collection[dict[str, Any]]:
    return get_db()["movies"]


def ratings_collection() -> Collection[dict[str, Any]]:
    return get_db()["ratings"]


def users_collection() -> Collection[dict[str, Any]]:
    return get_db()["users"]


def tmdb_cache_collection() -> Collection[dict[str, Any]]:
    return get_db()["tmdb_cache"]
