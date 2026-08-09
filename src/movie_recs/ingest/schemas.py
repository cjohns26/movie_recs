"""Pydantic schemas validating documents before they're written to Mongo.

Field names intentionally mirror the source data rather than Python
convention: `movieId`/`tmdbId`/`imdbId`/`userId` are the MovieLens CSV
headers verbatim, and `poster_path`/`overview`/`popularity`/`vote_average`
are TMDB's own API field names. Keeping them unchanged means a document
round-trips to Mongo without a translation layer, and matches the exact
field names plan.md's Data Model documents for later sessions to rely on.
"""

from datetime import datetime

from pydantic import BaseModel, ConfigDict, Field


class MovieDoc(BaseModel):
    """A `movies` collection document."""

    model_config = ConfigDict(populate_by_name=True)

    id: int = Field(alias="_id")
    tmdbId: int | None = None
    imdbId: str | None = None
    title: str
    year: int | None = None
    genres: list[str] = Field(default_factory=list)
    overview: str | None = None
    tags: list[str] = Field(default_factory=list)
    poster_path: str | None = None
    backdrop_path: str | None = None
    popularity: float | None = None
    vote_average: float | None = None


class RatingDoc(BaseModel):
    """A `ratings` collection document."""

    userId: int
    movieId: int
    rating: float
    timestamp: int


class UserDoc(BaseModel):
    """A derived `users` collection document."""

    model_config = ConfigDict(populate_by_name=True)

    id: int = Field(alias="_id")
    liked_movieIds: list[int] = Field(default_factory=list)
    n_ratings: int = 0
    last_ts: int | None = None


class TmdbCacheDoc(BaseModel):
    """A `tmdb_cache` collection document — raw TMDB payload, cached so
    re-running ingestion doesn't re-hit the API for movies already fetched.
    """

    tmdbId: int
    payload: dict[str, object]
    fetched_at: datetime
