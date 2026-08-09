"""Ingestion entrypoint: MovieLens + TMDB -> MongoDB.

Usage: uv run python -m movie_recs.ingest [--sample] [--data-dir PATH] [--no-tmdb]
"""

import argparse
import logging
from pathlib import Path
from typing import Any, cast

import pandas as pd

from movie_recs.config import get_settings
from movie_recs.db.collections import movies_collection, ratings_collection, users_collection
from movie_recs.db.indexes import ensure_indexes
from movie_recs.db.tmdb_cache import MongoTmdbCache
from movie_recs.ingest.join import join_movies_links, merge_tmdb_metadata
from movie_recs.ingest.movielens import (
    load_links,
    load_movies,
    load_ratings,
    load_tags,
    tags_by_movie,
)
from movie_recs.ingest.schemas import MovieDoc, RatingDoc, UserDoc
from movie_recs.ingest.tmdb import TmdbClient

logger = logging.getLogger(__name__)

# Movies kept for `--sample`; small enough for a fast local/CI run.
SAMPLE_SIZE = 200

# rating >= LIKE_THRESHOLD counts as "liked" for the derived `users` collection.
# Matches the implicit-feedback binarization threshold Session 3's ALS training uses.
LIKE_THRESHOLD = 4.0


def _int_or_none(value: Any) -> int | None:
    return None if pd.isna(value) else int(value)


def _build_movie_docs(
    joined: pd.DataFrame,
    tags_map: dict[int, list[str]],
    tmdb_client: TmdbClient | None,
) -> list[MovieDoc]:
    docs = []
    for raw_row in joined.to_dict("records"):
        # to_dict("records")'s keys are typed as the broad Hashable union per
        # pandas-stubs; our DataFrame's columns are always str at runtime.
        row = cast(dict[str, Any], raw_row)
        tmdb_id = _int_or_none(row.get("tmdbId"))
        tmdb_payload = (
            tmdb_client.get_movie(tmdb_id) if tmdb_client and tmdb_id is not None else None
        )
        merged = merge_tmdb_metadata(row, tmdb_payload)
        movie_id = int(merged["movieId"])
        docs.append(
            MovieDoc(
                id=movie_id,
                tmdbId=tmdb_id,
                imdbId=merged.get("imdbId"),
                title=merged["title"],
                year=_int_or_none(merged.get("year")),
                genres=merged["genres"],
                tags=tags_map.get(movie_id, []),
                overview=merged.get("overview"),
                poster_path=merged.get("poster_path"),
                backdrop_path=merged.get("backdrop_path"),
                popularity=merged.get("popularity"),
                vote_average=merged.get("vote_average"),
            )
        )
    return docs


def _build_rating_docs(ratings_df: pd.DataFrame) -> list[RatingDoc]:
    return [
        RatingDoc(
            userId=int(row["userId"]),
            movieId=int(row["movieId"]),
            rating=float(row["rating"]),
            timestamp=int(row["timestamp"]),
        )
        for row in ratings_df.to_dict("records")
    ]


def _build_user_docs(ratings_df: pd.DataFrame) -> list[UserDoc]:
    """Derive per-user liked/rated summary from ratings."""
    docs = []
    for user_id, group in ratings_df.groupby("userId"):
        liked = group.loc[group["rating"] >= LIKE_THRESHOLD, "movieId"].astype(int).tolist()
        docs.append(
            UserDoc(
                # groupby's key type is a broad Hashable union per pandas-stubs;
                # "userId" is always numeric at runtime.
                id=int(cast(int, user_id)),
                liked_movieIds=liked,
                n_ratings=len(group),
                last_ts=int(group["timestamp"].max()),
            )
        )
    return docs


def run(data_dir: Path, *, sample: bool, fetch_tmdb: bool = True) -> dict[str, int]:
    """Run the full ingest: parse -> join -> (optional) TMDB enrich -> upsert.

    Idempotent — safe to re-run; every write is a `replace_one(..., upsert=True)`
    keyed on the document's natural id, so counts don't grow on repeat runs.
    """
    settings = get_settings()

    movies_df = load_movies(data_dir / "movies.csv")
    ratings_df = load_ratings(data_dir / "ratings.csv")
    links_df = load_links(data_dir / "links.csv")
    tags_df = load_tags(data_dir / "tags.csv")

    if sample:
        movies_df = movies_df.head(SAMPLE_SIZE)
        sample_ids = set(movies_df["movieId"])
        ratings_df = ratings_df[ratings_df["movieId"].isin(sample_ids)]
        tags_df = tags_df[tags_df["movieId"].isin(sample_ids)]

    joined = join_movies_links(movies_df, links_df)
    tags_map = tags_by_movie(tags_df)

    tmdb_client: TmdbClient | None = None
    if fetch_tmdb:
        if not settings.tmdb_api_key:
            raise RuntimeError("TMDB_API_KEY is not set; pass --no-tmdb to skip enrichment.")
        tmdb_client = TmdbClient(
            settings.tmdb_api_key,
            MongoTmdbCache(),
            requests_per_second=settings.tmdb_requests_per_second,
        )

    ensure_indexes()

    try:
        movie_docs = _build_movie_docs(joined, tags_map, tmdb_client)
    finally:
        if tmdb_client is not None:
            tmdb_client.close()

    rating_docs = _build_rating_docs(ratings_df)
    user_docs = _build_user_docs(ratings_df)

    for movie in movie_docs:
        movies_collection().replace_one(
            {"_id": movie.id}, movie.model_dump(by_alias=True), upsert=True
        )
    for rating in rating_docs:
        ratings_collection().replace_one(
            {"userId": rating.userId, "movieId": rating.movieId},
            rating.model_dump(),
            upsert=True,
        )
    for user in user_docs:
        users_collection().replace_one(
            {"_id": user.id}, user.model_dump(by_alias=True), upsert=True
        )

    counts = {"movies": len(movie_docs), "ratings": len(rating_docs), "users": len(user_docs)}
    logger.info("Ingest complete: %s", counts)
    return counts


def main() -> None:
    logging.basicConfig(level=logging.INFO)
    # httpx's request logger logs the full URL at INFO, including the TMDB
    # api_key query param — never let a key hit the logs.
    logging.getLogger("httpx").setLevel(logging.WARNING)
    parser = argparse.ArgumentParser(description="Ingest MovieLens + TMDB into MongoDB")
    parser.add_argument("--data-dir", type=Path, default=Path("data/ml-latest-small"))
    parser.add_argument(
        "--sample", action="store_true", help="Ingest a small fixed subset for fast local/CI runs"
    )
    parser.add_argument(
        "--no-tmdb", action="store_true", help="Skip TMDB enrichment (MovieLens-only)"
    )
    args = parser.parse_args()
    run(args.data_dir, sample=args.sample, fetch_tmdb=not args.no_tmdb)


if __name__ == "__main__":
    main()
