"""Integration test: requires `docker compose up mongodb` (see conftest.py)."""

from pathlib import Path
from typing import Any

import pytest
from pymongo.database import Database

from movie_recs.ingest.run import run

pytestmark = pytest.mark.integration

FIXTURES = Path(__file__).parent.parent / "fixtures" / "ml-latest-small"


def test_ingest_is_idempotent(mongo_db: Database[dict[str, Any]]) -> None:
    """Running ingest twice against the same data yields identical counts —
    no duplicate documents, verifying the upsert-by-natural-key strategy.
    """
    first = run(FIXTURES, sample=False, fetch_tmdb=False)
    second = run(FIXTURES, sample=False, fetch_tmdb=False)

    assert first == second
    assert mongo_db["movies"].count_documents({}) == first["movies"]
    assert mongo_db["ratings"].count_documents({}) == first["ratings"]
    assert mongo_db["users"].count_documents({}) == first["users"]


def test_ingest_populates_expected_document_shapes(mongo_db: Database[dict[str, Any]]) -> None:
    """A sampled movie has the fields the Data Model documents (sans TMDB
    enrichment, since this test runs with fetch_tmdb=False)."""
    run(FIXTURES, sample=False, fetch_tmdb=False)

    movie = mongo_db["movies"].find_one({"_id": 1})
    assert movie is not None
    assert movie["title"] == "Toy Story (1995)"
    assert movie["year"] == 1995
    assert "pixar" in movie["tags"]

    rating = mongo_db["ratings"].find_one({"userId": 1, "movieId": 1})
    assert rating is not None
    assert rating["rating"] == 4.0

    user = mongo_db["users"].find_one({"_id": 1})
    assert user is not None
    assert 1 in user["liked_movieIds"]  # rating 4.0 >= LIKE_THRESHOLD
