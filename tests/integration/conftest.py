"""Shared fixtures for Mongo integration tests. Requires `docker compose up mongodb`."""

from collections.abc import Iterator
from typing import Any

import pytest
from pymongo.database import Database

from movie_recs.config import get_settings
from movie_recs.db.client import get_client, get_db

TEST_DB_NAME = "movie_recs_test"


@pytest.fixture
def mongo_db(monkeypatch: pytest.MonkeyPatch) -> Iterator[Database[dict[str, Any]]]:
    """An isolated test database, dropped after the test."""
    monkeypatch.setenv("MONGODB_DB", TEST_DB_NAME)
    get_settings.cache_clear()
    get_client.cache_clear()

    yield get_db()

    get_client().drop_database(TEST_DB_NAME)
    get_settings.cache_clear()
    get_client.cache_clear()
