"""Integration: real `$vectorSearch` against mongodb-atlas-local + a real model.

Requires `docker compose up mongodb` and the `ml` dependency group
(`uv sync --group ml`); the first run downloads the sentence-transformers
model. Run with: uv run pytest -m integration tests/integration/test_vector_search.py
"""

import time
from collections.abc import Iterator, Sequence
from typing import Any
from uuid import uuid4

import pytest
from pymongo.collection import Collection
from pymongo.database import Database

from movie_recs.config import get_settings
from movie_recs.db.client import get_client, get_db
from movie_recs.db.vector_indexes import (
    TEXT_EMBEDDING_FIELD,
    ensure_vector_index,
    text_vector_spec,
    vector_index_info,
    wait_for_vector_index,
)
from movie_recs.embeddings.run import embed_catalog
from movie_recs.embeddings.text import TEXT_EMBED_DIM, TextEmbedder
from movie_recs.recsys.retrieve import semantic_neighbors, similar_by_text

pytestmark = pytest.mark.integration

# A tiny catalog with two obvious semantic clusters: toys/animation vs. space.
SEED_MOVIES: list[dict[str, Any]] = [
    {
        "_id": 1,
        "title": "Toy Story (1995)",
        "year": 1995,
        "genres": ["Animation", "Children", "Comedy"],
        "overview": "A cowboy doll is threatened by a new spaceman action figure.",
        "tags": ["pixar", "toys"],
    },
    {
        "_id": 2,
        "title": "Toy Story 2 (1999)",
        "year": 1999,
        "genres": ["Animation", "Children", "Comedy"],
        "overview": "The toys mount a rescue when Woody is stolen by a collector.",
        "tags": ["pixar", "toys"],
    },
    {
        "_id": 3,
        "title": "Apollo 13 (1995)",
        "year": 1995,
        "genres": ["Adventure", "Drama"],
        "overview": "NASA must devise a plan to return astronauts safely from a crippled ship.",
        "tags": ["space", "nasa"],
    },
    {
        # No overview and no tags: the 122-movie fallback case must embed too.
        "_id": 4,
        "title": "Interstellar Voyage (2014)",
        "year": 2014,
        "genres": ["Sci-Fi", "Adventure"],
        "overview": None,
        "tags": [],
    },
]


@pytest.fixture
def mongo_db(monkeypatch: pytest.MonkeyPatch) -> Iterator[Database[dict[str, Any]]]:
    """A uniquely-named database per test, shadowing the shared conftest fixture.

    The shared fixture reuses one database name and drops it after each test.
    `mongot` can keep listing the dropped database's search index for a moment
    afterwards, so the next test's `ensure_vector_index` sees a stale index,
    skips creation, and then waits for an index that is in fact being deleted.
    A fresh name per test avoids that namespace reuse entirely.
    """
    name = f"movie_recs_test_vec_{uuid4().hex[:8]}"
    monkeypatch.setenv("MONGODB_DB", name)
    get_settings.cache_clear()
    get_client.cache_clear()

    yield get_db()

    get_client().drop_database(name)
    get_settings.cache_clear()
    get_client.cache_clear()


def _wait_until_indexed(
    collection: Collection[dict[str, Any]], expected: int, *, timeout_s: float = 60.0
) -> None:
    """Wait until every seeded movie is searchable.

    `queryable` says the index is usable, not that it has caught up with the
    collection — doing this once in the fixture keeps the assertions in each
    test about ranking rather than about timing.
    """
    doc = collection.find_one({"_id": SEED_MOVIES[0]["_id"]}, {TEXT_EMBEDDING_FIELD: 1})
    assert doc is not None
    deadline = time.monotonic() + timeout_s
    while True:
        found = semantic_neighbors(doc[TEXT_EMBEDDING_FIELD], k=expected, collection=collection)
        if len(found) >= expected or time.monotonic() >= deadline:
            return
        time.sleep(2.0)


def _search_until_found(
    collection: Collection[dict[str, Any]],
    query: Sequence[float],
    movie_id: int,
    *,
    k: int,
    timeout_s: float = 60.0,
) -> list[int]:
    """Poll `$vectorSearch` until `movie_id` shows up, or fail.

    `mongot` indexes document writes asynchronously, so a movie embedded after
    the index went queryable isn't searchable the same instant it's written —
    a bare assert here would be flaky rather than wrong.
    """
    deadline = time.monotonic() + timeout_s
    while True:
        ids = [c.movieId for c in semantic_neighbors(query, k=k, collection=collection)]
        if movie_id in ids or time.monotonic() >= deadline:
            return ids
        time.sleep(2.0)


@pytest.fixture(scope="module")
def embedder() -> TextEmbedder:
    model = TextEmbedder()
    model.load()
    return model


@pytest.fixture
def indexed_movies(
    mongo_db: Database[dict[str, Any]], embedder: TextEmbedder
) -> Iterator[Database[dict[str, Any]]]:
    """Seed, embed and index the tiny catalog, waiting until it's queryable."""
    mongo_db["movies"].insert_many([dict(movie) for movie in SEED_MOVIES])
    embed_catalog(embedder, batch_size=4)
    spec = text_vector_spec()
    ensure_vector_index(mongo_db["movies"], spec)
    wait_for_vector_index(mongo_db["movies"], spec.name, timeout_s=180)
    _wait_until_indexed(mongo_db["movies"], len(SEED_MOVIES))
    yield mongo_db


def test_embed_catalog_populates_every_movie_including_overview_less_ones(
    mongo_db: Database[dict[str, Any]], embedder: TextEmbedder
) -> None:
    mongo_db["movies"].insert_many([dict(movie) for movie in SEED_MOVIES])

    counts = embed_catalog(embedder, batch_size=2)

    assert counts["embedded"] == len(SEED_MOVIES)
    assert counts["with_embedding"] == len(SEED_MOVIES)
    no_overview = mongo_db["movies"].find_one({"_id": 4})
    assert no_overview is not None
    assert len(no_overview[TEXT_EMBEDDING_FIELD]) == TEXT_EMBED_DIM


def test_embed_catalog_is_idempotent_and_resumes(
    mongo_db: Database[dict[str, Any]], embedder: TextEmbedder
) -> None:
    """A re-run must skip already-embedded movies (so a partial failure resumes
    instead of re-paying for the whole catalog) and must not duplicate anything."""
    mongo_db["movies"].insert_many([dict(movie) for movie in SEED_MOVIES])
    embed_catalog(embedder, batch_size=4)

    second = embed_catalog(embedder, batch_size=4)

    assert second["embedded"] == 0
    assert second["with_embedding"] == len(SEED_MOVIES)
    assert mongo_db["movies"].count_documents({}) == len(SEED_MOVIES)


def test_ensure_vector_index_is_idempotent(indexed_movies: Database[dict[str, Any]]) -> None:
    spec = text_vector_spec()

    created_again = ensure_vector_index(indexed_movies["movies"], spec)

    assert created_again is False
    info = vector_index_info(indexed_movies["movies"], spec.name)
    assert info is not None
    assert info["queryable"] is True


def test_similar_by_text_ranks_the_sequel_above_unrelated_titles(
    indexed_movies: Database[dict[str, Any]],
) -> None:
    """The demoable claim of this session: "movies like Toy Story" returns
    Toy Story 2 ahead of a space drama."""
    neighbors = similar_by_text(1, k=3, collection=indexed_movies["movies"])

    assert [c.movieId for c in neighbors][0] == 2
    assert 1 not in [c.movieId for c in neighbors]  # the seed itself is excluded


def test_new_movie_with_zero_ratings_is_retrievable_by_content(
    indexed_movies: Database[dict[str, Any]], embedder: TextEmbedder
) -> None:
    """Item cold-start: a freshly inserted movie has no ratings and no ALS
    factor, so vector retrieval is the only path that can surface it."""
    indexed_movies["movies"].insert_one(
        {
            "_id": 99,
            "title": "Toy Story 3 (2010)",
            "year": 2010,
            "genres": ["Animation", "Children", "Comedy"],
            "overview": "Woody and the gang face life in a day-care centre after Andy grows up.",
            "tags": ["pixar", "toys"],
        }
    )
    embed_catalog(embedder, batch_size=4)

    query = embedder.encode(["animated films about toys that come to life"])[0]
    found = _search_until_found(indexed_movies["movies"], query, 99, k=4)

    assert 99 in found
    assert indexed_movies["ratings"].count_documents({"movieId": 99}) == 0
