"""MongoDB client factory."""

from functools import lru_cache
from typing import Any

from pymongo import MongoClient
from pymongo.database import Database

from movie_recs.config import get_settings


@lru_cache
def get_client() -> MongoClient[dict[str, Any]]:
    """Return the process-wide cached `MongoClient`."""
    settings = get_settings()
    return MongoClient(str(settings.mongodb_uri))


def get_db() -> Database[dict[str, Any]]:
    """Return the `movie_recs` database (or whatever `MONGODB_DB` names)."""
    settings = get_settings()
    return get_client()[settings.mongodb_db]
