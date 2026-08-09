"""Tests for movie_recs.ingest.movielens."""

from pathlib import Path

import pandas as pd

from movie_recs.ingest.movielens import (
    load_links,
    load_movies,
    load_ratings,
    load_tags,
    tags_by_movie,
)

FIXTURES = Path(__file__).parent.parent / "fixtures" / "ml-latest-small"


def test_load_movies_parses_year_and_splits_genres() -> None:
    df = load_movies(FIXTURES / "movies.csv")

    assert len(df) == 6
    toy_story = df.loc[df["movieId"] == 1].iloc[0]
    assert toy_story["year"] == 1995
    assert toy_story["genres"] == ["Adventure", "Animation", "Children", "Comedy", "Fantasy"]


def test_load_movies_handles_missing_year_and_no_genres_listed() -> None:
    df = load_movies(FIXTURES / "movies.csv")

    no_genres = df.loc[df["movieId"] == 6].iloc[0]
    assert pd.isna(no_genres["year"])
    assert no_genres["genres"] == []


def test_load_ratings_returns_expected_row_count() -> None:
    df = load_ratings(FIXTURES / "ratings.csv")
    assert len(df) == 7
    assert list(df.columns) == ["userId", "movieId", "rating", "timestamp"]


def test_load_links_zero_pads_imdb_id_and_keeps_missing_tmdb_id_nullable() -> None:
    df = load_links(FIXTURES / "links.csv")

    row1 = df.loc[df["movieId"] == 1].iloc[0]
    assert row1["imdbId"] == "0114709"
    assert row1["tmdbId"] == 862

    row6 = df.loc[df["movieId"] == 6].iloc[0]
    assert pd.isna(row6["tmdbId"])


def test_tags_by_movie_groups_and_deduplicates() -> None:
    tags_df = load_tags(FIXTURES / "tags.csv")
    grouped = tags_by_movie(tags_df)

    assert grouped[1] == ["pixar", "fun"]  # "pixar" from user 2 deduplicated
    assert grouped[2] == ["fantasy"]
    assert 3 not in grouped  # movie 3 has no tags
