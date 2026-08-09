"""Tests for movie_recs.ingest.tmdb."""

from typing import Any

import httpx
import respx

from movie_recs.ingest.tmdb import TmdbClient


class FakeCache:
    """In-memory stand-in for the Mongo-backed TmdbCache in tests."""

    def __init__(self) -> None:
        self._store: dict[int, dict[str, Any]] = {}

    def get(self, tmdb_id: int) -> dict[str, Any] | None:
        return self._store.get(tmdb_id)

    def set(self, tmdb_id: int, payload: dict[str, Any]) -> None:
        self._store[tmdb_id] = payload


@respx.mock
def test_get_movie_fetches_once_and_serves_second_call_from_cache() -> None:
    route = respx.get("https://api.themoviedb.org/3/movie/862").mock(
        return_value=httpx.Response(200, json={"id": 862, "title": "Toy Story"})
    )
    client = TmdbClient("fake-key", FakeCache(), requests_per_second=1000)

    first = client.get_movie(862)
    second = client.get_movie(862)

    assert first == {"id": 862, "title": "Toy Story"}
    assert second == first
    assert route.call_count == 1  # second call served from cache, no HTTP request


@respx.mock
def test_get_movie_404_returns_none_without_raising() -> None:
    respx.get("https://api.themoviedb.org/3/movie/999999").mock(return_value=httpx.Response(404))
    client = TmdbClient("fake-key", FakeCache(), requests_per_second=1000)

    result = client.get_movie(999999)

    assert result is None


@respx.mock
def test_get_movie_missing_poster_field_is_tolerated() -> None:
    respx.get("https://api.themoviedb.org/3/movie/1").mock(
        return_value=httpx.Response(200, json={"id": 1, "title": "No Poster"})
    )
    client = TmdbClient("fake-key", FakeCache(), requests_per_second=1000)

    payload = client.get_movie(1)

    assert payload is not None
    assert payload.get("poster_path") is None
